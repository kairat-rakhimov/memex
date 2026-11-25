"""GROBID client + section-aware chunking for arXiv PDFs.

GROBID parses scholarly PDFs into structured TEI XML. One
``processFulltextDocument`` call yields three things we use: section-
segmented body text (chunked along section boundaries), the bibliography,
and header authors/affiliations. Enabled by ``MEMEX_GROBID_URL``; without it
the adapter falls back to pdfminer + the shared ``chunk_text``.

Error model (fail-loud): service-unreachable raises ``GrobidUnavailable``,
which aborts the build; a
single PDF GROBID can't handle returns an empty ``GrobidDoc`` so the
adapter drops just that doc.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx
from lxml import etree
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from memex.common.http import http_timeout
from memex.kg_intelligence.sources.common.chunker import TARGET_CHARS, Chunk
from memex.kg_intelligence.sources.common.throttle import concise_retry_log

_log = logging.getLogger(__name__)

_NS = {"t": "http://www.tei-c.org/ns/1.0"}


def grobid_url() -> str:
    """Configured GROBID base URL, or ``""`` when GROBID is disabled."""

    return (os.environ.get("MEMEX_GROBID_URL") or "").strip().rstrip("/")


class GrobidUnavailable(RuntimeError):
    """GROBID service unreachable. Fatal for the build (fail-loud)."""


@dataclass(frozen=True)
class Section:
    head: str
    paragraphs: list[str]


@dataclass(frozen=True)
class Reference:
    raw: str
    title: str
    authors: list[str]
    year: str
    doi: str

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
        }


@dataclass(frozen=True)
class GrobidDoc:
    sections: list[Section]
    references: list[Reference]
    authors: list[str]
    affiliations: list[str]

    @property
    def is_empty(self) -> bool:
        return not any(s.paragraphs or s.head for s in self.sections)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class GrobidClient:
    def __init__(self, url: str | None = None, *, client: httpx.Client | None = None) -> None:
        self.url = url or grobid_url() or "http://localhost:8070"
        self.client = client or httpx.Client(timeout=http_timeout(120.0))

    def check_alive(self) -> None:
        """Raise ``GrobidUnavailable`` unless ``/api/isalive`` returns true."""

        try:
            resp = self.client.get(f"{self.url}/api/isalive", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise GrobidUnavailable(f"GROBID not reachable at {self.url}: {exc}") from exc
        if resp.text.strip().lower() != "true":
            raise GrobidUnavailable(f"GROBID at {self.url} returned isalive={resp.text!r}")

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=2, max=60),
        before_sleep=concise_retry_log(_log, "grobid"),
        reraise=True,
    )
    def _post(self, pdf_bytes: bytes) -> bytes:
        resp = self.client.post(
            f"{self.url}/api/processFulltextDocument",
            files={"input": ("doc.pdf", pdf_bytes, "application/pdf")},
            data={
                "consolidateCitations": "0",
                "consolidateHeader": "0",
                "segmentSentences": "0",
            },
        )
        resp.raise_for_status()
        return resp.content

    def parse(self, pdf_bytes: bytes) -> GrobidDoc:
        """Parse one PDF. Raises ``GrobidUnavailable`` on a server/connection
        problem (fatal); returns an empty ``GrobidDoc`` for a PDF GROBID
        rejects (4xx) or an empty/unparseable TEI (drop just this doc)."""

        try:
            tei = self._post(pdf_bytes)
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if 400 <= code < 500:
                _log.warning("grobid: HTTP %s on a PDF; dropping it", code)
                return _EMPTY
            raise GrobidUnavailable(f"GROBID server error {code} at {self.url}") from exc
        except httpx.HTTPError as exc:
            raise GrobidUnavailable(f"GROBID unreachable at {self.url}: {exc}") from exc
        return _parse_tei(tei)


_EMPTY = GrobidDoc(sections=[], references=[], authors=[], affiliations=[])


def _norm(text: str) -> str:
    return " ".join(text.split())


def _text(el: etree._Element) -> str:
    return _norm("".join(el.itertext()))


def _parse_tei(tei: bytes) -> GrobidDoc:
    if not tei or not tei.strip():
        return _EMPTY
    try:
        root = etree.fromstring(tei)
    except etree.XMLSyntaxError as exc:
        _log.warning("grobid: TEI parse failed: %s", exc)
        return _EMPTY
    authors, affiliations = _parse_header(root)
    return GrobidDoc(
        sections=_parse_sections(root),
        references=_parse_references(root),
        authors=authors,
        affiliations=affiliations,
    )


def _parse_sections(root: etree._Element) -> list[Section]:
    body = root.find(".//t:text/t:body", _NS)
    if body is None:
        return []
    sections: list[Section] = []
    for div in body.findall("t:div", _NS):
        head_el = div.find("t:head", _NS)
        if head_el is not None:
            n = head_el.get("n")
            head = _norm(f"{n} {_text(head_el)}") if n else _text(head_el)
        else:
            head = ""
        paragraphs = [t for t in (_text(p) for p in div.findall("t:p", _NS)) if t]
        if head or paragraphs:
            sections.append(Section(head=head, paragraphs=paragraphs))
    return sections


def _parse_references(root: etree._Element) -> list[Reference]:
    refs: list[Reference] = []
    for b in root.findall(".//t:text/t:back//t:listBibl/t:biblStruct", _NS):
        title = (
            b.findtext('.//t:title[@level="a"]', namespaces=_NS)
            or b.findtext('.//t:title[@level="m"]', namespaces=_NS)
            or b.findtext(".//t:title", namespaces=_NS)
            or ""
        )
        authors: list[str] = []
        for pn in b.findall(".//t:author/t:persName", _NS):
            name = _norm(
                f"{pn.findtext('t:forename', default='', namespaces=_NS)} "
                f"{pn.findtext('t:surname', default='', namespaces=_NS)}"
            )
            if name:
                authors.append(name)
        date_el = b.find(".//t:date", _NS)
        when = (date_el.get("when") or _text(date_el)) if date_el is not None else ""
        doi = b.findtext('.//t:idno[@type="DOI"]', default="", namespaces=_NS)
        refs.append(
            Reference(
                raw=_text(b),
                title=_norm(title),
                authors=authors,
                year=when[:4] if when else "",
                doi=_norm(doi),
            )
        )
    return refs


def _parse_header(root: etree._Element) -> tuple[list[str], list[str]]:
    hdr = root.find(".//t:teiHeader", _NS)
    if hdr is None:
        return [], []
    authors: list[str] = []
    for a in hdr.findall(".//t:sourceDesc//t:biblStruct//t:analytic/t:author", _NS):
        pn = a.find("t:persName", _NS)
        if pn is None:
            continue
        name = _norm(
            f"{pn.findtext('t:forename', default='', namespaces=_NS)} "
            f"{pn.findtext('t:surname', default='', namespaces=_NS)}"
        )
        if name and name not in authors:
            authors.append(name)
    affiliations: list[str] = []
    for org in hdr.findall(".//t:affiliation//t:orgName", _NS):
        v = _text(org)
        if v and v not in affiliations:
            affiliations.append(v)
    return authors, affiliations


def section_document(sections: list[Section]) -> tuple[str, list[Chunk]]:
    """Reconstruct ``cleaned_text`` with section headers inline, and chunk it
    along section boundaries.

    Each block is a whole section (``## head`` + its paragraphs). Adjacent
    blocks merge greedily toward ``TARGET_CHARS``; a section larger than the
    target stands alone. Sections are never split and chunks never overlap,
    so every chunk is a contiguous slice of ``cleaned_text`` whose head sits
    at its top — and each chunk is a clean section unit for session
    extraction.
    """

    blocks: list[str] = []
    pending_heads: list[str] = []  # header-only sections, carried to the next content section
    for sec in sections:
        if not sec.paragraphs:
            if sec.head:
                pending_heads.append(f"## {sec.head}")
            continue
        head = [f"## {sec.head}"] if sec.head else []
        blocks.append("\n\n".join([*pending_heads, *head, *sec.paragraphs]))
        pending_heads = []
    if pending_heads:  # trailing bare headers with no following content
        tail = "\n\n".join(pending_heads)
        blocks[-1] = f"{blocks[-1]}\n\n{tail}" if blocks else tail
    if not blocks:
        return "", []

    sep = "\n\n"
    cleaned_text = sep.join(blocks)

    spans: list[tuple[int, int]] = []
    pos = 0
    for i, block in enumerate(blocks):
        if i:
            pos += len(sep)
        spans.append((pos, pos + len(block)))
        pos += len(block)

    chunks: list[Chunk] = []
    start = end = None
    for s, e in spans:
        if start is None:
            start, end = s, e
        elif e - start <= TARGET_CHARS:
            end = e
        else:
            chunks.append(Chunk(len(chunks), start, end, cleaned_text[start:end]))
            start, end = s, e
    if start is not None:
        chunks.append(Chunk(len(chunks), start, end, cleaned_text[start:end]))

    return cleaned_text, chunks


__all__ = [
    "GrobidClient",
    "GrobidDoc",
    "GrobidUnavailable",
    "Reference",
    "Section",
    "grobid_url",
    "section_document",
]
