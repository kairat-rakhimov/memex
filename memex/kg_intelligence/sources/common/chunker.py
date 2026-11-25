"""Paragraph-aware chunker for per-chunk extraction.

Pure function. Greedy paragraph packing toward ``TARGET_CHARS`` with
an ``OVERLAP_CHARS`` tail so relations spanning the boundary still
see both endpoints. Single paragraphs exceeding ``HARD_MAX_CHARS``
fall back to a character-based sliding window.

``chunk_id`` is assigned by ``TraceStore.record_chunk()`` at write
time — chunks here carry only offsets + text.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_CHARS = 2000
OVERLAP_CHARS = 200
HARD_MAX_CHARS = 4000
PARAGRAPH_SEP = "\n\n"


@dataclass(frozen=True)
class Chunk:
    """One chunk emitted by `chunk_text`.

    `text == source_text[char_start:char_end]` is guaranteed for chunks
    produced from the paragraph path; the sliding-window fallback path
    (long paragraphs) also satisfies this since it slices the same
    source string.
    """

    chunk_index: int
    char_start: int
    char_end: int
    text: str


def chunk_text(text: str) -> list[Chunk]:
    """Split `text` into overlapping paragraph-aware chunks.

    Returns `[]` for empty input. Single-paragraph documents shorter
    than `TARGET_CHARS` yield one chunk covering the whole text.
    """

    if not text:
        return []

    chunks: list[Chunk] = []
    paragraphs = _split_paragraphs(text)

    buf_pieces: list[tuple[int, int, str]] = []  # (start, end, text)
    buf_chars = 0

    def flush() -> None:
        nonlocal buf_pieces, buf_chars
        if not buf_pieces:
            return
        start = buf_pieces[0][0]
        end = buf_pieces[-1][1]
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                char_start=start,
                char_end=end,
                text=text[start:end],
            )
        )
        buf_pieces = []
        buf_chars = 0

    for p_start, p_end in paragraphs:
        p_len = p_end - p_start

        if p_len > HARD_MAX_CHARS:
            flush()
            for s_start, s_end in _sliding_windows(p_start, p_end):
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        char_start=s_start,
                        char_end=s_end,
                        text=text[s_start:s_end],
                    )
                )
            continue

        sep_len = len(PARAGRAPH_SEP) if buf_pieces else 0
        if buf_pieces and buf_chars + sep_len + p_len > TARGET_CHARS:
            prev_end = buf_pieces[-1][1]
            flush()
            overlap_start = max(0, prev_end - OVERLAP_CHARS)
            if overlap_start < prev_end:
                buf_pieces.append(
                    (overlap_start, prev_end, text[overlap_start:prev_end])
                )
                buf_chars += prev_end - overlap_start

        buf_pieces.append((p_start, p_end, text[p_start:p_end]))
        buf_chars += sep_len + p_len

    flush()
    return chunks


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` offsets for each `\\n\\n`-separated paragraph.

    Empty paragraphs (consecutive `\\n\\n`) are skipped. The final
    paragraph may not end with `\\n\\n` — its `end` is `len(text)`.
    """

    spans: list[tuple[int, int]] = []
    sep = PARAGRAPH_SEP
    sep_len = len(sep)
    i = 0
    n = len(text)
    while i < n:
        j = text.find(sep, i)
        end = j if j != -1 else n
        if end > i:
            spans.append((i, end))
        if j == -1:
            break
        i = j + sep_len
    return spans


def _sliding_windows(start: int, end: int) -> list[tuple[int, int]]:
    """Slide a `TARGET_CHARS`-wide window with `OVERLAP_CHARS` overlap.

    Used for single paragraphs that exceed `HARD_MAX_CHARS`. The last
    window is clamped to `end` so no characters are duplicated past the
    paragraph boundary.
    """

    stride = TARGET_CHARS - OVERLAP_CHARS
    if stride <= 0:
        raise ValueError("TARGET_CHARS must exceed OVERLAP_CHARS")

    windows: list[tuple[int, int]] = []
    i = start
    while i < end:
        w_end = min(i + TARGET_CHARS, end)
        windows.append((i, w_end))
        if w_end == end:
            break
        i += stride
    return windows


__all__ = [
    "HARD_MAX_CHARS",
    "OVERLAP_CHARS",
    "TARGET_CHARS",
    "Chunk",
    "chunk_text",
]
