"""Shared arXiv search helpers: Atom feed parsing + retry predicate.

Used by the Atom-API search (``api.py``). Canonical doc id is the bare arXiv id
(no version) so re-fetching different versions hashes to the same id.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from memex.kg_intelligence.sources.common.base import SourceCandidate

SEARCH_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_VERSION_SUFFIX = re.compile(r"v\d+$")
DEFAULT_LOOKBACK_YEARS = 10


def is_retryable(exc: BaseException) -> bool:
    """429, 5xx, or transport-level network error."""

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429 or 500 <= code < 600:
            return True
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError))


def _flatten(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_atom(xml: str) -> list[SourceCandidate]:
    root = ET.fromstring(xml)
    candidates: list[SourceCandidate] = []
    for entry in root.findall(f"{_ATOM}entry"):
        id_url = (entry.findtext(f"{_ATOM}id") or "").strip()
        if "/abs/" not in id_url:
            continue
        with_version = id_url.split("/abs/", 1)[-1]
        canonical = _VERSION_SUFFIX.sub("", with_version)
        if not canonical:
            continue

        title = _flatten(entry.findtext(f"{_ATOM}title"))
        abstract = _flatten(entry.findtext(f"{_ATOM}summary"))
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        authors = [
            _flatten(author.findtext(f"{_ATOM}name"))
            for author in entry.findall(f"{_ATOM}author")
        ]
        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall(f"{_ATOM}category")
        ]
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        primary = entry.find(f"{_ARXIV_NS}primary_category")
        primary_term = primary.attrib.get("term", "") if primary is not None else ""

        candidates.append(
            SourceCandidate(
                source="arxiv",
                canonical=canonical,
                url=id_url,
                title=title,
                abstract=abstract,
                raw_signals={
                    "published": published,
                    "authors": [a for a in authors if a],
                    "categories": categories,
                    "primary_category": primary_term,
                    "pdf_url": pdf_url,
                },
            )
        )
    return candidates


__all__ = [
    "SEARCH_URL",
    "DEFAULT_LOOKBACK_YEARS",
    "is_retryable",
    "parse_atom",
]
