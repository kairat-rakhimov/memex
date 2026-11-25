"""GDELT source: DOC 2.0 article search + trafilatura content extraction.

Search hits ``gdeltproject.org/api/v2/doc/doc?mode=ArtList``; fetch
downloads the publisher URL and runs trafilatura. Canonical doc id is the
normalized URL (scheme/host lowercased, query + fragment dropped) so
syndicated mirrors of the same article don't all hash distinctly.

GDELT's public API rate-limits aggressively (HTTP 429) and intermittently
returns 5xx / connection drops. Search retries with respectful backoff;
other errors short-circuit (the query is skipped).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic_ai.models import Model
from tenacity import (
    RetryError,
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
from memex.kg_intelligence.sources.common.throttle import (
    AdapterThrottle,
    concise_retry_log,
    run_retrying,
    wait_respectful,
)

_log = logging.getLogger(__name__)

SEARCH_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_LOOKBACK_YEARS = 5

_QUERY_GUIDANCE = """\
GDELT indexes global news articles. Phrase queries as news-style topics —
the named technologies, companies, policies, places, or events that journalists
write about. Favour proper nouns and concrete deployment/economic angles over
abstract scientific terminology; this is a news index, not a research corpus."""


def _is_429(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 429
    )


def _is_retryable(exc: BaseException) -> bool:
    """429, 5xx, or transport-level network error."""

    if _is_429(exc):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and 500 <= exc.response.status_code < 600:
        return True
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError))


# GDELT 429s in recurring batches and recovers within ~2 retries when it's going
# to recover at all; in a 24-round build every storm that ran the full 5 attempts
# still failed (~4.4 min of wasted 60s-floored backoff each, ~17 min total, zero
# docs). So cap search retries low: 3 keeps the observed recoveries (which land
# by attempt 2-3) and drops the always-wasted tail. Article fetch keeps its own
# retry (no storms observed there).
_GDELT_SEARCH_MAX_ATTEMPTS = 3


class GdeltAdapter:
    """GDELT DOC 2.0 ArtList adapter with a configurable lookback window."""

    name = "gdelt"
    source: Source = "gdelt"

    description = AdapterDescription(
        name="gdelt",
        summary="Global news index (DOC 2.0); throttled article search, "
        "publisher-page text via trafilatura. Economic/policy/deployment context.",
        posture="throttled",
        preferred_doc_cap=3,
        metadata_fields=["seendate", "domain", "language", "sourcecountry", "tone"],
    )

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
        now: datetime | None = None,
        fetch_cache: FetchCache | None = None,
        search_max_attempts: int | None = None,
        fetch_max_attempts: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.client = client or httpx.Client(
            timeout=http_timeout(request_timeout or 30.0),
            headers={"User-Agent": user_agent()},
        )
        self.lookback_years = lookback_years
        self._now = now
        self.fetch_cache = fetch_cache or FetchCache(root=None)
        self._search_max_attempts = search_max_attempts or _GDELT_SEARCH_MAX_ATTEMPTS
        self._fetch_max_attempts = fetch_max_attempts or 5
        # GDELT's public API allows ~1 request / 5s. Plan a SINGLE query per
        # round (no intra-round burst) and space requests at 5s.
        self._throttle = AdapterThrottle(5.0)
        self._planner = BaseQueryPlanner(
            source_name="gdelt", guidance=_QUERY_GUIDANCE, max_queries=1
        )

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

    def _date_params(self) -> dict[str, str]:
        now = self._now or datetime.now(UTC)
        start = now - timedelta(days=int(round(365.25 * self.lookback_years)))
        return {
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": now.strftime("%Y%m%d%H%M%S"),
        }

    def search(self, queries: list[str], *, limit: int = 20) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for query in queries:
            self._throttle.wait()
            try:
                resp = self._get_with_retry(query, limit)
                payload = resp.json()
            except (httpx.HTTPError, RetryError, ValueError) as exc:
                _log.warning("gdelt search: api error for %r: %s", query, exc)
                continue
            for c in _parse_search(payload):
                c.matched_query = query
                out.append(c)
        return out

    def _get_with_retry(self, query: str, limit: int) -> httpx.Response:
        return run_retrying(
            self._do_search_get, query, limit,
            max_attempts=self._search_max_attempts,
            retry_predicate=_is_retryable,
            wait=wait_respectful(min_429=60.0, min_other=5.0, max_wait=300.0),
            before_sleep=concise_retry_log(_log, "gdelt"),
        )

    def _do_search_get(self, query: str, limit: int) -> httpx.Response:
        resp = self.client.get(
            SEARCH_URL,
            params={
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": limit,
                "sort": "hybridrel",
                **self._date_params(),
            },
        )
        # GDELT signals its rate limit as HTTP 200 + a plain-text notice
        # ("Please limit requests to one every 5 seconds"), not a 429 and not
        # JSON. Surface it as a retryable 429 so the respectful backoff applies,
        # instead of resp.json() failing and the query being silently dropped.
        if resp.status_code == 200 and "limit requests" in resp.text[:200].lower():
            raise httpx.HTTPStatusError(
                "GDELT rate-limit notice (HTTP 200 plain-text body)",
                request=resp.request,
                response=httpx.Response(429, request=resp.request),
            )
        resp.raise_for_status()
        return resp

    def _fetch_with_retry(self, url: str) -> bytes:
        return run_retrying(
            self._do_fetch, url,
            max_attempts=self._fetch_max_attempts,
            retry_predicate=_is_retryable,
            wait=wait_random_exponential(multiplier=2, max=60),
            before_sleep=concise_retry_log(_log, "gdelt-fetch"),
        )

    def _do_fetch(self, url: str) -> bytes:
        resp = self.client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.text.encode("utf-8")

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
        def _download() -> bytes:
            return self._fetch_with_retry(candidate.url)

        try:
            html_bytes = self.fetch_cache.get_or_fetch(candidate.canonical, _download)
        except (httpx.HTTPError, RetryError) as exc:
            _log.warning("gdelt fetch: download failed for %s: %s", candidate.url, exc)
            return None

        try:
            import trafilatura

            text = trafilatura.extract(html_bytes.decode("utf-8"), output_format="txt") or ""
        except Exception as exc:
            _log.warning("gdelt fetch: trafilatura failed for %s: %s", candidate.url, exc)
            return None

        text = text.strip()
        if not text:
            _log.warning("gdelt fetch: empty content for %s", candidate.url)
            return None

        return ChunkedDoc(
            candidate=candidate,
            cleaned_text=text,
            parse_method="trafilatura",
            chunks=chunk_text(text),
            extra_metadata={
                "seendate": candidate.raw_signals.get("seendate", ""),
                "domain": candidate.raw_signals.get("domain", ""),
                "language": candidate.raw_signals.get("language", ""),
                "sourcecountry": candidate.raw_signals.get("sourcecountry", ""),
                "tone": candidate.raw_signals.get("tone", ""),
            },
        )


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", "", "", "")
    )


def _parse_search(payload: dict[str, Any]) -> list[SourceCandidate]:
    articles = payload.get("articles", []) or []
    candidates: list[SourceCandidate] = []
    for art in articles:
        url = art.get("url", "")
        if not url:
            continue
        candidates.append(
            SourceCandidate(
                source="gdelt",
                canonical=_normalize_url(url),
                url=url,
                title=art.get("title", ""),
                abstract="",
                raw_signals={
                    "seendate": art.get("seendate", ""),
                    "socialimage": art.get("socialimage", ""),
                    "domain": art.get("domain", ""),
                    "language": art.get("language", ""),
                    "sourcecountry": art.get("sourcecountry", ""),
                    "tone": art.get("tone", ""),
                },
            )
        )
    return candidates


__all__ = ["GdeltAdapter"]
