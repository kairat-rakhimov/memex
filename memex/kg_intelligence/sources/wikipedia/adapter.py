"""Wikipedia source: MediaWiki search + extracts API for plain text.

Search uses ``list=search``; fetch uses ``prop=extracts&explaintext=true``
so the API returns pre-cleaned plain text (no client-side wikitext parser).
Canonical doc id is the page title (server resolves redirects on fetch).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from typing import Any

import httpx
from pydantic_ai.models import Model
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from memex.common.config import Settings
from memex.common.http import http_timeout, user_agent
from memex.kg_intelligence.agents.base import AgentCallSinks
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.sources.common.base import (
    AdapterDescription,
    ChunkedDoc,
    SourceCandidate,
)
from memex.kg_intelligence.sources.common.chunker import chunk_text
from memex.kg_intelligence.sources.common.fetch_cache import FetchCache
from memex.kg_intelligence.sources.common.query_planner import (
    BaseQueryPlanner,
    plan_and_search,
)
from memex.kg_intelligence.sources.common.throttle import concise_retry_log

_log = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_BASE = "https://en.wikipedia.org/wiki/"
_TAG_STRIP = re.compile(r"<[^>]+>")

_QUERY_GUIDANCE = """\
Wikipedia full-text search resolves best to article titles. Phrase each query
as the likely page title or canonical name of a concept, material, person, or
event — short noun phrases, not questions. Prefer the established encyclopedic
term over jargon or acronyms."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429 or 500 <= code < 600:
            return True
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError))


class WikipediaAdapter:
    name = "wikipedia"
    source: Source = "wikipedia"

    description = AdapterDescription(
        name="wikipedia",
        summary="Encyclopedic overviews + reusable terminology; relaxed "
        "full-text search, returns pre-cleaned plain text.",
        posture="relaxed",
        preferred_doc_cap=3,
        metadata_fields=["pageid", "size", "wordcount", "timestamp"],
    )

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        fetch_cache: FetchCache | None = None,
    ) -> None:
        self.client = client or httpx.Client(
            timeout=http_timeout(30.0),
            headers={"User-Agent": user_agent()},
        )
        self.fetch_cache = fetch_cache or FetchCache(root=None)
        self._planner = BaseQueryPlanner(source_name="wikipedia", guidance=_QUERY_GUIDANCE)

    def probe(self) -> bool:
        return True

    async def plan_and_search(
        self,
        ranked_topics: list[str],
        *,
        relevance: str,
        limit: int,
        settings: Settings,
        sinks: AgentCallSinks,
        llm_model: Model | None = None,
    ) -> list[SourceCandidate]:
        return await plan_and_search(
            planner=self._planner,
            search_fn=self.search,
            posture=self.description.posture,
            ranked_topics=ranked_topics,
            relevance=relevance,
            limit=limit,
            settings=settings,
            sinks=sinks,
            llm_model=llm_model,
        )

    def search(self, queries: list[str], *, limit: int = 20) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for query in queries:
            try:
                resp = self._search_with_retry(query, limit)
            except (httpx.HTTPError, RetryError) as exc:
                _log.warning("wikipedia search: api error for %r: %s", query, exc)
                continue
            for c in _parse_search(resp.json()):
                c.matched_query = query
                out.append(c)
        return out

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=2, max=60),
        before_sleep=concise_retry_log(_log, "wikipedia"),
        reraise=True,
    )
    def _search_with_retry(self, query: str, limit: int) -> httpx.Response:
        resp = self.client.get(
            API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": limit,
                "utf8": 1,
            },
        )
        resp.raise_for_status()
        return resp

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=2, max=60),
        before_sleep=concise_retry_log(_log, "wikipedia-fetch"),
        reraise=True,
    )
    def _fetch_with_retry(self, url: str) -> bytes:
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.content

    def fetch(
        self,
        candidates: list[SourceCandidate],
        *,
        settings: Settings,
    ) -> list[ChunkedDoc]:
        docs: list[ChunkedDoc] = []
        for candidate in candidates:
            doc = self._fetch_one(candidate)
            if doc is not None:
                docs.append(doc)
        return docs

    def _fetch_one(self, candidate: SourceCandidate) -> ChunkedDoc | None:
        extract_url = str(
            httpx.URL(
                API_URL,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exlimit": 1,
                    "explaintext": "true",
                    "titles": candidate.canonical,
                    "format": "json",
                    "redirects": 1,
                },
            )
        )

        def _download() -> bytes:
            return self._fetch_with_retry(extract_url)

        try:
            payload_bytes = self.fetch_cache.get_or_fetch(extract_url, _download)
            payload = json.loads(payload_bytes)
        except (httpx.HTTPError, RetryError, ValueError) as exc:
            _log.warning("wikipedia fetch: api error for %s: %s", candidate.canonical, exc)
            return None

        pages = payload.get("query", {}).get("pages", {}) or {}
        if not pages:
            return None
        page_id, page = next(iter(pages.items()))
        extract = (page.get("extract") or "").strip()
        if not extract:
            _log.warning("wikipedia fetch: empty extract for %s", candidate.canonical)
            return None

        resolved_title = page.get("title", candidate.canonical)
        return ChunkedDoc(
            candidate=candidate,
            cleaned_text=extract,
            parse_method="mediawiki.extracts",
            chunks=chunk_text(extract),
            extra_metadata={
                "page_id": page_id,
                "page_title": resolved_title,
                "snippet": candidate.abstract,
            },
        )


def _parse_search(payload: dict[str, Any]) -> list[SourceCandidate]:
    results = payload.get("query", {}).get("search", []) or []
    candidates: list[SourceCandidate] = []
    for item in results:
        title = item.get("title", "")
        if not title:
            continue
        snippet = _TAG_STRIP.sub("", item.get("snippet", "")).strip()
        snippet = " ".join(snippet.split())
        url = PAGE_BASE + urllib.parse.quote(title.replace(" ", "_"))
        candidates.append(
            SourceCandidate(
                source="wikipedia",
                canonical=title,
                url=url,
                title=title,
                abstract=snippet,
                raw_signals={
                    "pageid": item.get("pageid"),
                    "size": item.get("size"),
                    "wordcount": item.get("wordcount"),
                    "timestamp": item.get("timestamp", ""),
                },
            )
        )
    return candidates


__all__ = ["WikipediaAdapter"]
