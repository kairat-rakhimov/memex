"""arXiv Atom-API search."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import httpx
from tenacity import RetryError

from memex.kg_intelligence.sources.arxiv.search import (
    DEFAULT_LOOKBACK_YEARS,
    SEARCH_URL,
    is_retryable,
    parse_atom,
)
from memex.kg_intelligence.sources.common.base import SourceCandidate
from memex.kg_intelligence.sources.common.throttle import (
    AdapterThrottle,
    concise_retry_log,
    run_retrying,
    wait_respectful,
)

_log = logging.getLogger(__name__)


class AtomSearchMode:
    """Keyword search against the arXiv Atom API, date-windowed."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
        now: datetime | None = None,
        max_attempts: int = 5,
    ) -> None:
        self.client = client
        self.lookback_years = lookback_years
        self._now = now
        self._max_attempts = max_attempts
        self._throttle = AdapterThrottle(5.0)

    def search(self, queries: list[str], *, limit: int) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for query in queries:
            self._throttle.wait()
            try:
                resp = self._get_with_retry(query, limit)
            except (httpx.HTTPError, RetryError) as exc:
                _log.warning("arxiv search: api error for %r: %s", query, exc)
                continue
            for c in parse_atom(resp.text):
                c.matched_query = query
                out.append(c)
        return out

    def _search_query_with_date(self, query: str) -> str:
        now = self._now or datetime.now(UTC)
        start = now - timedelta(days=int(round(365.25 * self.lookback_years)))
        date_range = (
            f"[{start.strftime('%Y%m%d%H%M')}+TO+{now.strftime('%Y%m%d%H%M')}]"
        )
        return f"all:{query}+AND+submittedDate:{date_range}"

    def _get_with_retry(self, query: str, limit: int) -> httpx.Response:
        return run_retrying(
            self._do_search_get, query, limit,
            max_attempts=self._max_attempts,
            retry_predicate=is_retryable,
            wait=wait_respectful(min_429=60.0, min_other=5.0, max_wait=300.0),
            before_sleep=concise_retry_log(_log, "arxiv"),
        )

    def _do_search_get(self, query: str, limit: int) -> httpx.Response:
        resp = self.client.get(
            SEARCH_URL,
            params={
                "search_query": self._search_query_with_date(query),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
            },
        )
        resp.raise_for_status()
        return resp


__all__ = ["AtomSearchMode"]
