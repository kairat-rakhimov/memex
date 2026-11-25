"""Tests for the per-source license-posture renderer.

Asserts each of Wikipedia / arXiv / GDELT renders its cited excerpt
under the right posture (per-source excerpt policy).
"""

from __future__ import annotations

from memex.kg_intelligence.citations import GDELT_EXCERPT_MAX_CHARS, render_chunk_citation


def test_wikipedia_full_chunk_with_cc_by_sa_attribution() -> None:
    """Wikipedia (CC BY-SA 4.0): full chunk text + URL attribution + license tag."""

    chunk = (
        "A proton-exchange membrane (PEM) electrolyzer uses a solid polymer "
        "electrolyte to split water into hydrogen and oxygen."
    )
    rendered = render_chunk_citation(
        chunk_text=chunk,
        source="wikipedia",
        doc_title="Proton-exchange membrane electrolysis",
        doc_url="https://en.wikipedia.org/wiki/Proton-exchange_membrane_electrolysis",
    )
    # Full chunk preserved verbatim.
    assert chunk in rendered
    # Attribution back to the originating article.
    assert "Wikipedia" in rendered
    assert "Proton-exchange membrane electrolysis" in rendered
    assert "https://en.wikipedia.org/wiki/" in rendered
    # CC BY-SA license tag inline.
    assert "CC BY-SA 4.0" in rendered


def test_arxiv_full_chunk_with_paper_metadata() -> None:
    """arXiv: full chunk text + arXiv id / title attribution. No CC license
    tag — arXiv submissions vary; we attribute the source, not the license."""

    chunk = (
        "Iridium oxide catalyzes the oxygen-evolution reaction (OER) at "
        "low overpotential on Nafion membranes."
    )
    rendered = render_chunk_citation(
        chunk_text=chunk,
        source="arxiv",
        doc_title="Iridium oxide OER kinetics in PEM electrolyzers",
        doc_url="https://arxiv.org/abs/2401.12345",
    )
    assert chunk in rendered
    assert "arXiv" in rendered
    assert "2401.12345" in rendered  # arXiv id surfaces via URL
    assert "Iridium oxide OER kinetics" in rendered
    # No CC license tag — arXiv submissions aren't blanket-licensed.
    assert "CC BY-SA" not in rendered


def test_gdelt_excerpt_capped_at_200_chars_with_publisher_attribution() -> None:
    """GDELT: chunk trimmed to ≤200 chars + publisher URL + ≤200-char tag."""

    # 320-char chunk — must get trimmed.
    long_chunk = (
        "Officials in the UAE announced a major new green-hydrogen "
        "investment programme today, totalling $1.4B over five years, "
        "aimed at electrolyzer-stack manufacturing capacity, PEM stack "
        "research, and grid-tied production trials at three sites. "
        "Details will be released later."
    )
    assert len(long_chunk) > GDELT_EXCERPT_MAX_CHARS  # confirm the test premise
    rendered = render_chunk_citation(
        chunk_text=long_chunk,
        source="gdelt",
        doc_title="UAE green hydrogen $1.4B programme",
        doc_url="https://reuters.example/uae-hydrogen-2026",
    )
    # The excerpt portion (between the opening and closing quote) is ≤ cap.
    excerpt = rendered.split('"')[1]
    assert len(excerpt) <= GDELT_EXCERPT_MAX_CHARS
    # Ellipsis indicator on the trimmed text.
    assert excerpt.endswith("…")
    # Publisher URL surfaces in the attribution line.
    assert "reuters.example" in rendered
    # ≤200-char tag is documented inline.
    assert "≤200" in rendered
