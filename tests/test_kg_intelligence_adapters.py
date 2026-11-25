"""Tests for the per-source adapters.

HTTP is stubbed via `httpx.MockTransport` reading cached payloads from
`tests/fixtures/adapters/` — no live network in CI. The
PDF parser (`pdfminer.high_level.extract_text`) is monkeypatched so the
arXiv tests don't need a real PDF; trafilatura runs for real against a
small fixture HTML.

`search` is per-source-batch (`search([query], limit=...)`) and `fetch`
is per-source-batch (`fetch([candidate], settings=...) -> list[ChunkedDoc]`);
the `_fetch_one` helper unwraps the single-doc case for assertions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from memex.kg_intelligence.sources import (
    ArxivAdapter,
    GdeltAdapter,
    SourceCandidate,
    WikipediaAdapter,
)
from memex.kg_intelligence.sources.arxiv import adapter as arxiv_adapter_mod
from memex.kg_intelligence.sources.common.base import ChunkedDoc
from memex.kg_intelligence.sources.common.fetch_cache import FetchCache

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


@pytest.fixture()
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable time.sleep so tenacity retries and AdapterThrottle don't block."""
    monkeypatch.setattr("time.sleep", lambda _: None)


def _client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _fetch_one(adapter: object, candidate: SourceCandidate) -> ChunkedDoc | None:
    """Fetch a single candidate via the batch API; return its doc or None."""
    docs = adapter.fetch([candidate], settings=None)  # type: ignore[attr-defined]
    return docs[0] if docs else None


class _StubGrobid:
    """Stand-in for ``GrobidClient`` so GROBID-mode construction needs no live
    GROBID service. ``parse`` returns a fixed one-section doc by default."""

    def __init__(self, doc: object | None = None) -> None:
        self._doc = doc

    def check_alive(self) -> None:
        return None

    def parse(self, pdf_bytes: bytes) -> object:
        from memex.kg_intelligence.sources.arxiv.grobid import GrobidDoc, Section

        return self._doc or GrobidDoc(
            sections=[Section(head="1 Introduction", paragraphs=["body about catalysis"])],
            references=[],
            authors=[],
            affiliations=[],
        )


# ── arxiv ──────────────────────────────────────────────────────────────────


def _arxiv_search_handler(request: httpx.Request) -> httpx.Response:
    assert "export.arxiv.org" in request.url.host
    return httpx.Response(200, content=(FIXTURES / "arxiv_search.xml").read_bytes())


def test_arxiv_search_parses_candidates() -> None:
    adapter = ArxivAdapter(client=_client_for(_arxiv_search_handler))
    results = adapter.search(["iridium catalyst"], limit=10)

    assert len(results) == 2
    first = results[0]
    assert first.source == "arxiv"
    assert first.canonical == "2105.12345"
    assert first.title == "Photocatalytic water splitting with iridium catalysts"
    assert "iridium-based photocatalyst" in first.abstract
    assert first.raw_signals["pdf_url"] == "http://arxiv.org/pdf/2105.12345v2"
    assert first.raw_signals["primary_category"] == "cond-mat.mtrl-sci"
    assert first.raw_signals["authors"] == ["Alice Researcher", "Bob Scientist"]
    assert first.raw_signals["categories"] == ["cond-mat.mtrl-sci", "physics.chem-ph"]
    assert first.matched_query == "iridium catalyst"


def test_arxiv_search_strips_version_suffix() -> None:
    adapter = ArxivAdapter(client=_client_for(_arxiv_search_handler))
    results = adapter.search(["anything"], limit=10)
    assert results[0].canonical == "2105.12345"  # v2 stripped
    assert results[1].canonical == "2206.99999"  # no version present


def test_arxiv_fetch_returns_doc_with_text_and_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    def pdf_handler(request: httpx.Request) -> httpx.Response:
        assert "arxiv.org/pdf/" in str(request.url)
        return httpx.Response(200, content=b"%PDF-1.4 fake bytes")

    adapter = ArxivAdapter(client=_client_for(pdf_handler))
    monkeypatch.setattr(
        "pdfminer.high_level.extract_text",
        lambda _stream: "Full paper body about iridium photocatalysis.",
    )

    candidate = SourceCandidate(
        source="arxiv",
        canonical="2105.12345",
        url="http://arxiv.org/abs/2105.12345v2",
        title="t",
        abstract="abs",
        raw_signals={"pdf_url": "http://arxiv.org/pdf/2105.12345v2", "published": "2021-05-26T13:00:00Z"},
    )
    doc = _fetch_one(adapter, candidate)

    assert doc is not None
    assert doc.parse_method == "pdfminer.six"
    assert "iridium" in doc.cleaned_text
    assert doc.extra_metadata["arxiv_id"] == "2105.12345"
    assert doc.extra_metadata["abstract"] == "abs"
    # chunking happens inside fetch now
    assert len(doc.chunks) == 1
    assert doc.chunks[0].text == doc.cleaned_text


def test_arxiv_fetch_returns_none_on_download_error(_no_sleep: None) -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    adapter = ArxivAdapter(client=_client_for(boom))
    candidate = SourceCandidate(
        source="arxiv",
        canonical="x",
        url="u",
        title="t",
        raw_signals={"pdf_url": "http://arxiv.org/pdf/x"},
    )
    assert _fetch_one(adapter, candidate) is None


def test_arxiv_fetch_returns_none_on_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ArxivAdapter(
        client=_client_for(lambda _r: httpx.Response(200, content=b"%PDF-1.4 broken"))
    )

    def boom(_stream: object) -> str:
        raise RuntimeError("pdf is mangled")

    monkeypatch.setattr("pdfminer.high_level.extract_text", boom)
    candidate = SourceCandidate(
        source="arxiv",
        canonical="x",
        url="u",
        title="t",
        raw_signals={"pdf_url": "http://arxiv.org/pdf/x"},
    )
    assert _fetch_one(adapter, candidate) is None


def test_arxiv_fetch_returns_none_without_pdf_url() -> None:
    adapter = ArxivAdapter(client=_client_for(lambda _r: httpx.Response(200, content=b"")))
    candidate = SourceCandidate(
        source="arxiv", canonical="x", url="u", title="t", raw_signals={}
    )
    assert _fetch_one(adapter, candidate) is None


def test_arxiv_module_has_no_top_level_pdfminer_import() -> None:
    assert not hasattr(arxiv_adapter_mod, "extract_text"), (
        "pdfminer import must remain inside fetch() so import-time failures "
        "don't break the rest of the adapter module"
    )


# ── arxiv PDF parser (GROBID when configured, else pdfminer) ─────────────────


def test_arxiv_parser_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMEX_GROBID_URL", raising=False)
    assert ArxivAdapter().parser == "pdfminer"
    assert ArxivAdapter(grobid_client=_StubGrobid()).parser == "grobid"


def test_arxiv_grobid_from_env_fails_loud_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memex.kg_intelligence.sources.arxiv.grobid import GrobidUnavailable

    def down(self, url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setenv("MEMEX_GROBID_URL", "http://localhost:8070")
    monkeypatch.setattr(httpx.Client, "get", down)
    with pytest.raises(GrobidUnavailable):
        ArxivAdapter()


def test_arxiv_parses_with_grobid(monkeypatch: pytest.MonkeyPatch) -> None:
    # With GROBID configured, PDFs parse into section-aligned chunks with the
    # section header inline + reference metadata.
    from memex.kg_intelligence.sources.arxiv.grobid import GrobidDoc, Reference, Section

    def pdf_handler(request: httpx.Request) -> httpx.Response:
        assert "arxiv.org/pdf/" in str(request.url)
        return httpx.Response(200, content=b"%PDF-1.4 fake")

    gdoc = GrobidDoc(
        sections=[Section(head="1 Introduction", paragraphs=["body about catalysis"])],
        references=[Reference(raw="r", title="Prior work", authors=["X Y"], year="2020", doi="")],
        authors=["A Author"],
        affiliations=["Some Lab"],
    )
    adapter = ArxivAdapter(
        client=_client_for(pdf_handler),
        grobid_client=_StubGrobid(gdoc),
    )
    candidate = SourceCandidate(
        source="arxiv", canonical="2108.13292", url="u", title="t",
        raw_signals={"pdf_url": "https://arxiv.org/pdf/2108.13292"},
    )
    doc = _fetch_one(adapter, candidate)
    assert doc is not None
    assert doc.parse_method == "grobid"
    assert doc.cleaned_text.startswith("## 1 Introduction")
    assert "catalysis" in doc.cleaned_text
    assert doc.extra_metadata["references"][0]["title"] == "Prior work"
    assert doc.extra_metadata["affiliations"] == ["Some Lab"]
    c0 = doc.chunks[0]
    assert c0.text == doc.cleaned_text[c0.char_start:c0.char_end]


# ── wikipedia ──────────────────────────────────────────────────────────────


def _wikipedia_handler(request: httpx.Request) -> httpx.Response:
    params = dict(request.url.params)
    if params.get("list") == "search":
        return httpx.Response(
            200, content=(FIXTURES / "wikipedia_search.json").read_bytes()
        )
    if params.get("prop") == "extracts":
        return httpx.Response(
            200, content=(FIXTURES / "wikipedia_extract.json").read_bytes()
        )
    return httpx.Response(404)


def test_wikipedia_search_parses_candidates() -> None:
    adapter = WikipediaAdapter(client=_client_for(_wikipedia_handler))
    results = adapter.search(["hydrogen economy"], limit=5)

    assert len(results) == 2
    first = results[0]
    assert first.source == "wikipedia"
    assert first.canonical == "Hydrogen economy"
    assert first.title == "Hydrogen economy"
    assert first.url == "https://en.wikipedia.org/wiki/Hydrogen_economy"
    # snippet HTML tags stripped, whitespace collapsed
    assert "<span" not in first.abstract
    assert first.abstract.startswith("The hydrogen economy refers")
    assert first.raw_signals["pageid"] == 12345


def test_wikipedia_fetch_returns_extract() -> None:
    adapter = WikipediaAdapter(client=_client_for(_wikipedia_handler))
    candidate = SourceCandidate(
        source="wikipedia",
        canonical="Hydrogen economy",
        url="https://en.wikipedia.org/wiki/Hydrogen_economy",
        title="Hydrogen economy",
    )
    doc = _fetch_one(adapter, candidate)

    assert doc is not None
    assert doc.parse_method == "mediawiki.extracts"
    assert "hydrogen economy" in doc.cleaned_text.lower()
    assert doc.extra_metadata["page_id"] == "12345"
    assert doc.extra_metadata["page_title"] == "Hydrogen economy"
    assert len(doc.chunks) >= 1


def test_wikipedia_fetch_returns_none_for_empty_extract() -> None:
    def empty(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"query": {"pages": {"99": {"pageid": 99, "title": "X", "extract": ""}}}},
        )

    adapter = WikipediaAdapter(client=_client_for(empty))
    candidate = SourceCandidate(
        source="wikipedia", canonical="X", url="u", title="X"
    )
    assert _fetch_one(adapter, candidate) is None


# ── gdelt ──────────────────────────────────────────────────────────────────


def test_gdelt_search_parses_candidates_with_normalized_canonical() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(FIXTURES / "gdelt_search.json").read_bytes())

    adapter = GdeltAdapter(client=_client_for(handler))
    results = adapter.search(["UAE hydrogen"], limit=10)

    assert len(results) == 2
    first = results[0]
    assert first.source == "gdelt"
    # query string + fragment stripped, host lowercased
    assert first.canonical == "https://example.com/hydrogen-news"
    assert first.url == "https://example.com/hydrogen-news?utm_source=foo#section"
    assert first.title == "UAE announces hydrogen strategy"
    assert first.raw_signals["domain"] == "example.com"
    assert first.raw_signals["sourcecountry"] == "United Arab Emirates"


def test_gdelt_soft_rate_limit_is_retried_then_dropped(_no_sleep: None) -> None:
    """GDELT signals its rate limit as HTTP 200 + a plain-text notice (not 429,
    not JSON). It must be treated as a retryable 429 (so the backoff applies),
    not silently json()-dropped on the first hit — then give up at the capped
    attempt count. GDELT 429s recover within ~2 retries or not at all, so
    search is capped at 3 attempts."""
    from memex.kg_intelligence.sources.gdelt.adapter import _GDELT_SEARCH_MAX_ATTEMPTS

    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text="Please limit requests to one every 5 seconds.")

    adapter = GdeltAdapter(client=_client_for(handler))
    results = adapter.search(["UAE hydrogen"], limit=10)

    assert results == []          # persistent rate-limit → no candidates
    assert calls["n"] == _GDELT_SEARCH_MAX_ATTEMPTS == 3   # capped


def test_gdelt_fetch_extracts_main_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(FIXTURES / "gdelt_article.html").read_bytes(),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    adapter = GdeltAdapter(client=_client_for(handler))
    candidate = SourceCandidate(
        source="gdelt",
        canonical="https://example.com/hydrogen-news",
        url="https://example.com/hydrogen-news",
        title="UAE Hydrogen Strategy Announced",
        raw_signals={"domain": "example.com", "tone": "2.5", "seendate": "20260315T120000Z"},
    )
    doc = _fetch_one(adapter, candidate)

    assert doc is not None
    assert doc.parse_method == "trafilatura"
    assert "hydrogen" in doc.cleaned_text.lower()
    # nav / footer dropped by trafilatura
    assert "Copyright 2026" not in doc.cleaned_text
    assert doc.extra_metadata["tone"] == "2.5"
    assert doc.extra_metadata["seendate"] == "20260315T120000Z"


def test_gdelt_fetch_returns_none_on_http_error(_no_sleep: None) -> None:
    adapter = GdeltAdapter(client=_client_for(lambda _r: httpx.Response(500)))
    candidate = SourceCandidate(
        source="gdelt", canonical="https://x/", url="https://x/", title="t"
    )
    assert _fetch_one(adapter, candidate) is None


def test_gdelt_search_returns_empty_on_api_error(_no_sleep: None) -> None:
    adapter = GdeltAdapter(client=_client_for(lambda _r: httpx.Response(503)))
    assert adapter.search(["anything"]) == []


def test_gdelt_search_applies_lookback_window_to_params() -> None:
    import datetime as _dt

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, content=(FIXTURES / "gdelt_search.json").read_bytes())

    fixed_now = _dt.datetime(2026, 5, 22, 12, 0, 0, tzinfo=_dt.UTC)
    adapter = GdeltAdapter(
        client=_client_for(handler),
        lookback_years=3,
        now=fixed_now,
    )
    adapter.search(["uae hydrogen"], limit=5)
    assert captured["enddatetime"] == "20260522120000"
    assert captured["startdatetime"] == "20230522120000"  # 3 × 365.25d ≈ 1096d earlier


# ── fetch-cache integration ────────────────────────────────────────────────


def test_arxiv_fetch_second_call_hits_cache_not_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = {"calls": 0}

    def pdf_handler(_request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        return httpx.Response(200, content=b"%PDF-1.4 fake bytes")

    monkeypatch.setattr(
        "pdfminer.high_level.extract_text",
        lambda _stream: "Body text.",
    )

    cache = FetchCache(root=tmp_path / "cache")
    adapter = ArxivAdapter(client=_client_for(pdf_handler), fetch_cache=cache)
    candidate = SourceCandidate(
        source="arxiv",
        canonical="2105.12345",
        url="http://arxiv.org/abs/2105.12345",
        title="t",
        raw_signals={"pdf_url": "http://arxiv.org/pdf/2105.12345"},
    )

    first = _fetch_one(adapter, candidate)
    second = _fetch_one(adapter, candidate)

    assert counter["calls"] == 1
    assert first is not None and second is not None
    assert first.cleaned_text == second.cleaned_text
    assert cache.hits == 1
    assert cache.misses == 1


def test_gdelt_fetch_second_call_hits_cache_not_network(tmp_path: Path) -> None:
    counter = {"calls": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        return httpx.Response(
            200,
            content=(FIXTURES / "gdelt_article.html").read_bytes(),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    cache = FetchCache(root=tmp_path / "cache")
    adapter = GdeltAdapter(client=_client_for(handler), fetch_cache=cache)
    candidate = SourceCandidate(
        source="gdelt",
        canonical="https://example.com/hydrogen-news",
        url="https://example.com/hydrogen-news",
        title="t",
        raw_signals={},
    )

    first = _fetch_one(adapter, candidate)
    second = _fetch_one(adapter, candidate)

    assert counter["calls"] == 1
    assert first is not None and second is not None
    assert first.cleaned_text == second.cleaned_text
    assert cache.hits == 1
    assert cache.misses == 1


def test_gdelt_cache_key_uses_canonical_not_raw_url(tmp_path: Path) -> None:
    """Two surfaces of the same article (different tracking params) must
    share a cache entry — gdelt rate-limits make refetches expensive, and
    tracking params don't change the publisher's response body."""

    counter = {"calls": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        return httpx.Response(
            200,
            content=(FIXTURES / "gdelt_article.html").read_bytes(),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    cache = FetchCache(root=tmp_path / "cache")
    adapter = GdeltAdapter(client=_client_for(handler), fetch_cache=cache)

    candidate_a = SourceCandidate(
        source="gdelt",
        canonical="https://example.com/hydrogen-news",
        url="https://example.com/hydrogen-news?utm_source=twitter",
        title="t",
        raw_signals={},
    )
    candidate_b = SourceCandidate(
        source="gdelt",
        canonical="https://example.com/hydrogen-news",
        url="https://example.com/hydrogen-news?utm_source=newsletter#share",
        title="t",
        raw_signals={},
    )

    first = _fetch_one(adapter, candidate_a)
    second = _fetch_one(adapter, candidate_b)

    assert counter["calls"] == 1
    assert first is not None and second is not None
    assert first.cleaned_text == second.cleaned_text
    assert cache.hits == 1
    assert cache.misses == 1


def test_wikipedia_fetch_second_call_hits_cache_not_network(tmp_path: Path) -> None:
    counter = {"calls": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        return httpx.Response(
            200, content=(FIXTURES / "wikipedia_extract.json").read_bytes()
        )

    cache = FetchCache(root=tmp_path / "cache")
    adapter = WikipediaAdapter(client=_client_for(handler), fetch_cache=cache)
    candidate = SourceCandidate(
        source="wikipedia",
        canonical="Hydrogen economy",
        url="https://en.wikipedia.org/wiki/Hydrogen_economy",
        title="Hydrogen economy",
    )

    first = _fetch_one(adapter, candidate)
    second = _fetch_one(adapter, candidate)

    assert counter["calls"] == 1
    assert first is not None and second is not None
    assert first.cleaned_text == second.cleaned_text
    assert cache.hits == 1
    assert cache.misses == 1
