"""OpenAlex enrichment client: per-doc topics / concepts / citations / related works.

Enriches **arXiv** docs only — Wikipedia and GDELT yield no useful match
in practice. arXiv → OpenAlex match is two-step: arXiv DOI shim
(``10.48550/arXiv.<id>``) then one-result title search; both miss → None.
``mailto`` is attached as a query param for the polite pool when set.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from memex.common.http import http_timeout
from memex.kg_intelligence.persistence import OpenAlexMetadata
from memex.kg_intelligence.sources.common.base import SourceCandidate

_log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
ARXIV_DOI_PREFIX = "10.48550/arXiv."


class OpenAlexClient:
    """Stateless enricher; safe to share one instance across the pipeline."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        mailto: str | None = None,
    ) -> None:
        self.client = client or httpx.Client(timeout=http_timeout(30.0))
        self.mailto = mailto

    def enrich(self, candidate: SourceCandidate) -> OpenAlexMetadata | None:
        """Return slimmed OpenAlex metadata for `candidate`, or None on miss."""

        if candidate.source != "arxiv":
            return None
        work = self._fetch_arxiv_work(candidate.canonical, candidate.title)
        if work is None:
            return None
        return _slim(work)

    # ── internals ─────────────────────────────────────────────────────────

    def _fetch_arxiv_work(self, arxiv_id: str, title: str) -> dict[str, Any] | None:
        doi_url = f"{BASE_URL}/works/doi:{ARXIV_DOI_PREFIX}{arxiv_id}"
        work = self._get(doi_url)
        if work is not None:
            return work
        if not title:
            return None
        return self._search_first(title)

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = dict(extra or {})
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def _get(self, url: str) -> dict[str, Any] | None:
        try:
            resp = self.client.get(url, params=self._params())
        except httpx.HTTPError as exc:
            _log.warning("openalex GET %s failed: %s", url, exc)
            return None
        if resp.status_code == 404:
            return None
        try:
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning("openalex GET %s parse failed: %s", url, exc)
            return None

    def _search_first(self, query: str) -> dict[str, Any] | None:
        try:
            resp = self.client.get(
                f"{BASE_URL}/works",
                params=self._params({"search": query, "per-page": 1}),
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning("openalex search %r failed: %s", query, exc)
            return None
        return results[0] if results else None


def _slim(work: dict[str, Any]) -> OpenAlexMetadata:
    """Project the full OpenAlex Work payload down to what we persist."""

    return OpenAlexMetadata(
        openalex_id=work.get("id"),
        doi=work.get("doi"),
        topics=[
            {
                "id": t.get("id"),
                "name": t.get("display_name"),
                "score": t.get("score"),
            }
            for t in (work.get("topics") or [])
        ],
        concepts=[
            {
                "id": c.get("id"),
                "name": c.get("display_name"),
                "level": c.get("level"),
                "score": c.get("score"),
            }
            for c in (work.get("concepts") or [])
        ],
        cited_by_count=work.get("cited_by_count"),
        related_works=list(work.get("related_works") or []),
    )
