"""Tests for the OpenAlex enrichment client.

Stubs the API via `httpx.MockTransport`; no live network. Polite-pool
behaviour is asserted by capturing the request URLs the transport sees.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from memex.kg_intelligence.openalex import ARXIV_DOI_PREFIX, BASE_URL, OpenAlexClient
from memex.kg_intelligence.sources.common.base import SourceCandidate

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs: object) -> OpenAlexClient:
    return OpenAlexClient(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def _arxiv_candidate(arxiv_id: str = "2105.12345", title: str = "A paper") -> SourceCandidate:
    return SourceCandidate(
        source="arxiv",
        canonical=arxiv_id,
        url=f"http://arxiv.org/abs/{arxiv_id}",
        title=title,
        abstract="",
    )


def test_enrich_arxiv_doi_hit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert f"/works/doi:{ARXIV_DOI_PREFIX}2105.12345" in str(request.url)
        return httpx.Response(200, content=(FIXTURES / "openalex_work.json").read_bytes())

    enriched = _client(handler).enrich(_arxiv_candidate())
    assert enriched is not None
    assert enriched.openalex_id == "https://openalex.org/W4382345678"
    assert enriched.doi == "https://doi.org/10.48550/arXiv.2105.12345"
    assert enriched.cited_by_count == 42
    assert len(enriched.topics) == 2
    assert enriched.topics[0]["name"] == "Photocatalysis"
    assert enriched.topics[0]["score"] == 0.87
    assert len(enriched.concepts) == 2
    assert enriched.concepts[0]["level"] == 3
    assert enriched.related_works == [
        "https://openalex.org/W2741809807",
        "https://openalex.org/W3015421983",
        "https://openalex.org/W4279672541",
    ]


def test_enrich_arxiv_doi_miss_falls_back_to_title_search() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "/works/doi:" in str(request.url):
            return httpx.Response(404, json={"error": "not found"})
        # title search
        assert "search=" in str(request.url)
        return httpx.Response(
            200,
            json={"results": [json.loads((FIXTURES / "openalex_work.json").read_text())]},
        )

    enriched = _client(handler).enrich(_arxiv_candidate(title="Photocatalytic water splitting"))
    assert enriched is not None
    assert enriched.openalex_id == "https://openalex.org/W4382345678"
    # both lookups were attempted, DOI first
    assert "/works/doi:" in calls[0]
    assert "search=" in calls[1]


def test_enrich_returns_none_when_both_lookups_miss() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/works/doi:" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, json={"results": []})

    assert _client(handler).enrich(_arxiv_candidate()) is None


def test_enrich_skips_wikipedia() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("wikipedia should not trigger an OpenAlex lookup")

    candidate = SourceCandidate(
        source="wikipedia",
        canonical="Hydrogen economy",
        url="https://en.wikipedia.org/wiki/Hydrogen_economy",
        title="Hydrogen economy",
    )
    assert _client(handler).enrich(candidate) is None


def test_enrich_skips_gdelt() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("gdelt should not trigger an OpenAlex lookup")

    candidate = SourceCandidate(
        source="gdelt",
        canonical="https://example/news",
        url="https://example/news",
        title="A news article",
    )
    assert _client(handler).enrich(candidate) is None


def test_enrich_attaches_mailto_to_polite_pool() -> None:
    captured: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        return httpx.Response(
            200, content=(FIXTURES / "openalex_work.json").read_bytes()
        )

    _client(handler, mailto="researcher@example.org").enrich(_arxiv_candidate())
    assert captured and captured[0].get("mailto") == "researcher@example.org"


def test_enrich_returns_none_on_http_error() -> None:
    assert _client(lambda _r: httpx.Response(503)).enrich(_arxiv_candidate()) is None


def test_enrich_returns_none_on_invalid_json() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    assert _client(handler).enrich(_arxiv_candidate()) is None


def test_enrich_skips_title_fallback_when_title_empty() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    enriched = _client(handler).enrich(_arxiv_candidate(title=""))
    assert enriched is None
    # only the DOI lookup ran; no title search to a missing title
    assert len(calls) == 1
    assert "/works/doi:" in calls[0]


def test_base_url_is_openalex() -> None:
    assert BASE_URL == "https://api.openalex.org"
