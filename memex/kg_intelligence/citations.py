"""Per-source license-posture renderer for cited chunks.

Pure function — see NOTICE.md for the license posture each source
maps onto. GDELT excerpts are clipped to ``GDELT_EXCERPT_MAX_CHARS``
since original article bodies remain publisher copyright.
"""

from __future__ import annotations

from memex.kg_intelligence.ids import Source

GDELT_EXCERPT_MAX_CHARS = 200


def render_chunk_citation(
    *,
    chunk_text: str,
    source: Source,
    doc_title: str,
    doc_url: str | None,
) -> str:
    """Render one cited chunk under the source's license posture.

    ``chunk_text`` is the raw chunk body; ``doc_url`` may be ``None``
    for sources where the upstream URL isn't recorded (rare; it is
    stored in every document's metadata, but the type allows None to
    keep the call site simple).

    Returns a multi-line string with the cited excerpt + attribution
    suffix. Wikipedia / arXiv get the full chunk; GDELT is trimmed to
    ``GDELT_EXCERPT_MAX_CHARS`` characters (with an ellipsis suffix
    when the chunk was longer).
    """

    excerpt = (chunk_text or "").strip()
    url_suffix = f"\n— {doc_url}" if doc_url else ""

    if source == "wikipedia":
        return (
            f'"{excerpt}"\n'
            f"— Wikipedia, {doc_title!r}{url_suffix}\n"
            f"(CC BY-SA 4.0)"
        )
    if source == "arxiv":
        return (
            f'"{excerpt}"\n'
            f"— arXiv, {doc_title!r}{url_suffix}"
        )
    if source == "gdelt":
        capped = excerpt
        if len(capped) > GDELT_EXCERPT_MAX_CHARS:
            capped = capped[: GDELT_EXCERPT_MAX_CHARS - 1].rstrip() + "…"
        return (
            f'"{capped}"\n'
            f"— {doc_title!r}{url_suffix}\n"
            f"(GDELT-indexed; ≤{GDELT_EXCERPT_MAX_CHARS}-char excerpt)"
        )
    # Defensive: unknown source falls back to a bare attributed quote
    # rather than crashing the render pipeline.
    return f'"{excerpt}"\n— {source}, {doc_title!r}{url_suffix}'
