"""``ArxivAdapter`` — the arXiv source plugin.

Searches the arXiv Atom API and downloads each selected paper's PDF. The PDF
parser is chosen at construction:

- ``MEMEX_GROBID_URL`` set (or a ``GrobidClient`` passed) → GROBID:
  section-aligned chunks plus bibliography and affiliation metadata. Fail-loud:
  an unreachable GROBID aborts construction rather than silently degrading.
- otherwise → pdfminer flat text + the shared paragraph chunker.
"""

from __future__ import annotations

import logging
from io import BytesIO

import httpx
from pydantic_ai.models import Model
from tenacity import RetryError, wait_random_exponential

from memex.common.config import Settings
from memex.common.http import http_timeout
from memex.kg_intelligence.agents.base import AgentCallSinks
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.sources.arxiv.api import AtomSearchMode
from memex.kg_intelligence.sources.arxiv.grobid import (
    GrobidClient,
    grobid_url,
    section_document,
)
from memex.kg_intelligence.sources.arxiv.search import DEFAULT_LOOKBACK_YEARS, is_retryable
from memex.kg_intelligence.sources.common.base import (
    AdapterDescription,
    ChunkedDoc,
    SourceCandidate,
)
from memex.kg_intelligence.sources.common.chunker import Chunk, chunk_text
from memex.kg_intelligence.sources.common.fetch_cache import FetchCache
from memex.kg_intelligence.sources.common.query_planner import (
    BaseQueryPlanner,
    plan_and_search,
)
from memex.kg_intelligence.sources.common.throttle import (
    AdapterThrottle,
    concise_retry_log,
    run_retrying,
)

_log = logging.getLogger(__name__)

_QUERY_GUIDANCE = """\
arXiv full-text/metadata search matches titles + abstracts. Use compact
keyword queries of scientific terms — material names, mechanisms, methods,
quantities. Boolean operators (AND/OR) and quoted multi-word phrases are
supported; avoid natural-language questions and stop-words."""


class ArxivAdapter:
    """arXiv source plugin: Atom-API search, GROBID or pdfminer parsing."""

    name = "arxiv"
    source: Source = "arxiv"

    description = AdapterDescription(
        name="arxiv",
        summary="arXiv preprints (physics/CS/quant-bio); throttled "
        "keyword search over titles + abstracts.",
        posture="throttled",
        preferred_doc_cap=3,
        metadata_fields=[
            "published", "authors", "categories", "primary_category", "pdf_url",
        ],
    )

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        fetch_cache: FetchCache | None = None,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
        grobid_client: GrobidClient | None = None,
        search_max_attempts: int | None = None,
        fetch_max_attempts: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.client = client or httpx.Client(timeout=http_timeout(request_timeout or 60.0))
        self.fetch_cache = fetch_cache or FetchCache(root=None)
        self._throttle = AdapterThrottle(5.0)
        self._planner = BaseQueryPlanner(source_name="arxiv", guidance=_QUERY_GUIDANCE)
        self._grobid: GrobidClient | None = None
        self._fetch_max_attempts = fetch_max_attempts or 10

        self._search_mode = AtomSearchMode(
            self.client, lookback_years=lookback_years,
            max_attempts=search_max_attempts or 5,
        )
        if grobid_client is None and grobid_url():
            grobid_client = GrobidClient()
        if grobid_client is not None:
            grobid_client.check_alive()
        self._grobid = grobid_client
        self.parser = "grobid" if grobid_client is not None else "pdfminer"
        _log.info("arxiv adapter: %s parser", self.parser)

    def probe(self) -> bool:
        """Always ready — the Atom API needs no credentials."""

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
        return self._search_mode.search(queries, limit=limit)

    def _fetch_with_retry(self, url: str) -> bytes:
        return run_retrying(
            self._do_fetch, url,
            max_attempts=self._fetch_max_attempts,
            retry_predicate=is_retryable,
            wait=wait_random_exponential(multiplier=2, max=120),
            before_sleep=concise_retry_log(_log, "arxiv-fetch"),
        )

    def _do_fetch(self, url: str) -> bytes:
        resp = self.client.get(url, follow_redirects=True)
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
        pdf_url = candidate.raw_signals.get("pdf_url") or ""
        if not pdf_url:
            _log.warning("arxiv fetch: missing pdf_url for %s", candidate.canonical)
            return None

        def _download() -> bytes:
            self._throttle.wait()
            return self._fetch_with_retry(pdf_url)

        try:
            pdf_bytes = self.fetch_cache.get_or_fetch(pdf_url, _download)
        except (httpx.HTTPError, RetryError) as exc:
            _log.warning("arxiv fetch: download failed for %s: %s", candidate.canonical, exc)
            return None

        if self._grobid is not None:
            parsed = self._parse_grobid(candidate, pdf_bytes)
        else:
            parsed = self._parse_pdfminer(candidate, pdf_bytes)
        if parsed is None:
            return None
        cleaned_text, chunks, parse_method, extra = parsed

        return ChunkedDoc(
            candidate=candidate,
            cleaned_text=cleaned_text,
            parse_method=parse_method,
            chunks=chunks,
            extra_metadata={
                "arxiv_id": candidate.canonical,
                "pdf_url": pdf_url,
                "published": candidate.raw_signals.get("published", ""),
                "categories": candidate.raw_signals.get("categories", []),
                "primary_category": candidate.raw_signals.get("primary_category", ""),
                "abstract": candidate.abstract,
                **extra,
            },
        )

    def _parse_grobid(
        self, candidate: SourceCandidate, pdf_bytes: bytes
    ) -> tuple[str, list[Chunk], str, dict[str, object]] | None:
        """GROBID → section-aligned chunks + reference metadata.

        ``GrobidUnavailable`` (service down) propagates to abort the build."""

        gdoc = self._grobid.parse(pdf_bytes)
        if gdoc.is_empty:
            _log.warning("arxiv fetch: grobid produced no sections for %s", candidate.canonical)
            return None
        cleaned_text, chunks = section_document(gdoc.sections)
        extra = {
            "references": [r.as_dict() for r in gdoc.references],
            "affiliations": gdoc.affiliations,
        }
        return cleaned_text, chunks, "grobid", extra

    def _parse_pdfminer(
        self, candidate: SourceCandidate, pdf_bytes: bytes
    ) -> tuple[str, list[Chunk], str, dict[str, object]] | None:
        """pdfminer flat text + the shared chunker."""

        try:
            from pdfminer.high_level import extract_text

            text = extract_text(BytesIO(pdf_bytes)) or ""
        except Exception as exc:
            _log.warning("arxiv fetch: pdf parse failed for %s: %s", candidate.canonical, exc)
            return None

        text = _strip_vertical_watermarks(text).strip()
        if not text:
            _log.warning("arxiv fetch: empty text after parse for %s", candidate.canonical)
            return None
        return text, chunk_text(text), "pdfminer.six", {}


def _strip_vertical_watermarks(text: str, *, min_run: int = 6) -> str:
    """Collapse runs of single-character lines (rotated PDF watermarks).

    pdfminer.six renders rotated text — e.g. the arXiv vertical doc-id
    stamp — as one character per line. Such runs never appear in
    legitimate body text. Drop runs of ``min_run`` or more consecutive
    blank/single-character lines."""

    lines = text.splitlines()
    out: list[str] = []
    run_start = -1
    run_singlechars = 0

    def flush_run(end: int) -> None:
        nonlocal run_start, run_singlechars
        if run_start < 0:
            return
        if run_singlechars < min_run:
            out.extend(lines[run_start:end])
        run_start = -1
        run_singlechars = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) <= 1:
            if run_start < 0:
                run_start = i
            if stripped:
                run_singlechars += 1
        else:
            flush_run(i)
            out.append(line)
    flush_run(len(lines))
    return "\n".join(out)


__all__ = ["ArxivAdapter"]
