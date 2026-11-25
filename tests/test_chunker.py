"""Tests for memex.kg_intelligence.sources.common.chunker."""

from __future__ import annotations

from memex.kg_intelligence.sources.common.chunker import (
    HARD_MAX_CHARS,
    OVERLAP_CHARS,
    TARGET_CHARS,
    chunk_text,
)


def _assert_offsets_consistent(text: str, chunks: list) -> None:
    """Every chunk's text equals the corresponding source slice."""

    for ch in chunks:
        assert ch.text == text[ch.char_start : ch.char_end], (
            f"chunk {ch.chunk_index} text/offset mismatch"
        )
        assert ch.char_start < ch.char_end, (
            f"chunk {ch.chunk_index} has empty span"
        )


def _assert_indices_monotonic(chunks: list) -> None:
    for i, ch in enumerate(chunks):
        assert ch.chunk_index == i


def test_empty_input_returns_empty_list():
    assert chunk_text("") == []


def test_short_single_paragraph_yields_one_chunk():
    text = "A short paragraph well under the target size."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_multiple_short_paragraphs_pack_into_one_chunk():
    paragraphs = ["Para one.", "Para two.", "Para three."]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_two_paragraphs_split_when_combined_exceeds_target():
    para_a = "a" * (TARGET_CHARS - 100)
    para_b = "b" * (TARGET_CHARS - 100)
    text = f"{para_a}\n\n{para_b}"
    chunks = chunk_text(text)
    assert len(chunks) == 2
    _assert_offsets_consistent(text, chunks)
    _assert_indices_monotonic(chunks)


def test_overlap_carries_tail_of_previous_chunk_into_next():
    para_a = "a" * (TARGET_CHARS - 100)
    para_b = "b" * (TARGET_CHARS - 100)
    text = f"{para_a}\n\n{para_b}"
    chunks = chunk_text(text)
    assert len(chunks) == 2
    prev_end = chunks[0].char_end
    overlap_expected_start = prev_end - OVERLAP_CHARS
    assert chunks[1].char_start == overlap_expected_start
    # Second chunk should literally start with the tail of the first.
    assert chunks[1].text.startswith(text[overlap_expected_start:prev_end])


def test_overlap_clamps_when_previous_chunk_shorter_than_overlap():
    # First "chunk" is a tiny paragraph; the next paragraph forces a split
    # because they together exceed TARGET. The overlap copy must clamp.
    small = "x" * 50
    big = "b" * (TARGET_CHARS + 50)  # > TARGET, ≤ HARD_MAX (4000)
    text = f"{small}\n\n{big}"
    chunks = chunk_text(text)
    # The big paragraph triggers split-before-append; overlap is 50 chars
    # (the full first chunk), not OVERLAP_CHARS=200.
    assert len(chunks) == 2
    overlap_start = chunks[1].char_start
    prev_end = chunks[0].char_end
    assert prev_end - overlap_start == min(OVERLAP_CHARS, prev_end)
    _assert_offsets_consistent(text, chunks)


def test_long_single_paragraph_triggers_sliding_window_fallback():
    long_para = "z" * (HARD_MAX_CHARS + 1000)
    chunks = chunk_text(long_para)
    assert len(chunks) > 1
    _assert_offsets_consistent(long_para, chunks)
    _assert_indices_monotonic(chunks)
    # First window starts at 0; last window ends at len(text).
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(long_para)


def test_sliding_window_overlap_is_correct():
    long_para = "z" * (HARD_MAX_CHARS + 1000)
    chunks = chunk_text(long_para)
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        if cur.char_end == len(long_para):
            # Last window may be clamped; just check it doesn't run past
            # the document end.
            assert cur.char_start < cur.char_end
            continue
        assert prev.char_end - cur.char_start == OVERLAP_CHARS


def test_long_paragraph_flushes_pending_buffer_first():
    # A short paragraph followed by a >HARD_MAX paragraph: the short one
    # gets emitted as its own chunk before the sliding-window fallback runs.
    short = "short paragraph."
    long_para = "z" * (HARD_MAX_CHARS + 500)
    text = f"{short}\n\n{long_para}"
    chunks = chunk_text(text)
    assert chunks[0].text == short
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(short)
    # Remaining chunks all cover the long paragraph (no overlap into `short`).
    for ch in chunks[1:]:
        assert ch.char_start >= len(short) + len("\n\n")
    _assert_offsets_consistent(text, chunks)


def test_empty_paragraphs_are_skipped():
    text = "first\n\n\n\nsecond\n\n\n\n\n\nthird"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    # Combined chunk text reconstructs the original slice byte-for-byte.
    _assert_offsets_consistent(text, chunks)


def test_chunk_indices_are_zero_based_contiguous():
    para_a = "a" * (TARGET_CHARS - 100)
    para_b = "b" * (TARGET_CHARS - 100)
    para_c = "c" * (TARGET_CHARS - 100)
    text = "\n\n".join([para_a, para_b, para_c])
    chunks = chunk_text(text)
    assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))
