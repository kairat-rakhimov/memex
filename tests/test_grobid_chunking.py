"""Unit tests for GROBID section-aware chunking (`section_document`).

Pure function over `Section` lists — no network, no GROBID service.
"""

from __future__ import annotations

from memex.kg_intelligence.sources.arxiv.grobid import Section, section_document
from memex.kg_intelligence.sources.common.chunker import TARGET_CHARS


def _para(n: int) -> str:
    return "word " * n  # ~5 chars/word


def test_offset_invariant_and_headers_at_top() -> None:
    sections = [
        Section(head="1 Introduction", paragraphs=[_para(500)]),
        Section(head="2 Methods", paragraphs=[_para(500)]),
    ]
    cleaned, chunks = section_document(sections)
    assert chunks
    for c in chunks:
        assert c.text == cleaned[c.char_start:c.char_end]
        assert c.text.startswith("## ")


def test_header_only_section_folds_into_next_no_tiny_chunk() -> None:
    # A bare parent header (no paragraphs) followed by a large section must
    # not become its own tiny chunk — its head prepends to the next block.
    sections = [
        Section(head="3 Model Architecture", paragraphs=[]),
        Section(head="3.1 Encoder", paragraphs=[_para(600)]),
    ]
    cleaned, chunks = section_document(sections)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("## 3 Model Architecture\n\n## 3.1 Encoder")
    # no orphan chunk that is only the bare header
    assert all(len(c.text) > len("## 3 Model Architecture") for c in chunks)


def test_small_adjacent_sections_merge_large_stands_alone() -> None:
    sections = [
        Section(head="A", paragraphs=[_para(50)]),   # small
        Section(head="B", paragraphs=[_para(50)]),   # small -> merges with A
        Section(head="C", paragraphs=[_para(700)]),  # ~3500 chars > TARGET -> alone
    ]
    cleaned, chunks = section_document(sections)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("## A") and "## B" in chunks[0].text
    assert chunks[1].text.startswith("## C")
    assert chunks[1].char_end - chunks[1].char_start > TARGET_CHARS
    for c in chunks:
        assert c.text == cleaned[c.char_start:c.char_end]


def test_empty_sections_yield_no_chunks() -> None:
    assert section_document([]) == ("", [])
