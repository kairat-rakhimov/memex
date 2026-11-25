"""The grounding ladder: a fact id → a :class:`GroundedFact` carrying all three
provenance levels the substrate already stores (no ingest writes needed).

- **L1 canonical** — the synonym-merged fact sentence (also what's embedded).
- **L2 raw** — the Extractor's surface form (``raw_*`` columns).
- **L3 source span** — the paper prose the fact came from
  (``source_chunk_id`` → ``chunks.char_start/end`` → ``documents.text``).

Plus the document :class:`Citation` (parsed from ``documents.metadata``) and the
read-time ``corroboration`` count (distinct source docs asserting the same
canonical fact). Agents may only cite fact ids the toolbox handed them; these
functions are the gate that turns an id into real, quotable evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from memex.kg_query.deps import QueryDeps


@dataclass(frozen=True)
class Citation:
    doc_id: str
    title: str
    url: str | None
    source: str
    authors: list[str]


@dataclass(frozen=True)
class GroundedFact:
    """One fact with its full provenance ladder. ``kind`` ∈ {relation,
    attribute}; ``predicate`` holds the predicate (relation) or property
    (attribute) label; ``object`` the object entity (relation) or the value
    display (attribute)."""

    kind: str
    fact_id: int
    subject: str
    predicate: str
    object: str
    canonical: str
    raw: str
    condition: str
    confidence: float | None
    corroboration: int
    source_doc_id: str | None
    chunk_span: str | None
    citation: Citation | None
    source_quote: str = ""  # the verbatim source sentence behind the fact id

    def cite_label(self) -> str:
        if self.citation is None:
            return self.source_doc_id or "(uncited)"
        return self.citation.title or self.citation.doc_id


# Wide enough to hold a whole chunk (~2000 chars + overlap) so a fact's
# supporting sentence is never clipped out of its own grounding span — a clipped
# span made VerifyEvidence reject correctly-grounded legs as "unsupported".
_SPAN_MAX_CHARS = 2400


def format_value(value_raw: str, unit: str, condition: str = "") -> str:
    """Compose an attribute value for display: ``value_raw`` + ``unit`` (skipped
    when the raw value already carries the unit, e.g. ``"62,045 MW"`` + ``"MW"``)
    + an optional ``(condition)`` suffix."""

    v = (value_raw or "").strip()
    u = (unit or "").strip()
    if u and u.lower() not in v.lower():
        v = f"{v} {u}".strip()
    if condition:
        v = f"{v} ({condition})"
    return v


_QUOTE_MAX_CHARS = 500


def _clean_quote(quote: str | None) -> str:
    """The stored verbatim source sentence, whitespace-normalized and length-capped
    for display — the human-readable text behind an otherwise opaque fact id."""

    q = " ".join((quote or "").split())
    if len(q) > _QUOTE_MAX_CHARS:
        q = q[:_QUOTE_MAX_CHARS].rstrip() + "…"
    return q


def _chunk_span(deps: QueryDeps, chunk_id: int | None) -> str | None:
    if chunk_id is None:
        return None
    row = deps.conn.execute(
        """
        SELECT substr(d.text, c.char_start + 1, c.char_end - c.char_start) AS span
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None or row["span"] is None:
        return None
    span = " ".join(row["span"].split())
    if len(span) > _SPAN_MAX_CHARS:
        span = span[:_SPAN_MAX_CHARS].rstrip() + "…"
    return span


def _normalize_ws_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space, recording the ORIGINAL index of
    each emitted char so a match in the normalized string maps back to a span in
    the original text. Source prose (especially PDF text) has irregular whitespace
    / newlines a stored quote won't reproduce exactly."""

    chars: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            chars.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(chars), idx_map


def locate_quote(quote: str, chunk_text: str) -> tuple[int, int] | None:
    """Best-effort resolve a stored verbatim ``quote`` to a ``(start, end)``
    char span within ``chunk_text``. Exact substring first; then a whitespace-
    normalized match (PDF prose carries newlines / runs the LLM quote won't
    reproduce), mapped back to original offsets. Returns ``None`` when the quote is
    empty or unlocatable — the caller falls back to the whole chunk. Deterministic;
    never trusts an LLM-emitted char offset (the reason we store the quote, not an
    offset)."""

    q = (quote or "").strip()
    if not q or not chunk_text:
        return None
    idx = chunk_text.find(q)
    if idx != -1:
        return (idx, idx + len(q))
    norm_q = " ".join(q.split())
    norm_chunk, idx_map = _normalize_ws_with_map(chunk_text)
    pos = norm_chunk.find(norm_q)
    if pos == -1:
        return None
    return (idx_map[pos], idx_map[pos + len(norm_q) - 1] + 1)


def citation_for(deps: QueryDeps, doc_id: str | None) -> Citation | None:
    """Parse ``documents.metadata`` JSON into a :class:`Citation`. Returns
    ``None`` when the doc id is missing / absent."""

    if not doc_id:
        return None
    row = deps.conn.execute(
        "SELECT metadata FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return None
    try:
        meta = json.loads(row["metadata"])
    except (TypeError, ValueError):
        meta = {}
    authors = meta.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    return Citation(
        doc_id=doc_id,
        title=(meta.get("title") or "").strip() or doc_id,
        url=meta.get("url"),
        source=meta.get("source") or "unknown",
        authors=[str(a) for a in authors],
    )


def ground_relation(deps: QueryDeps, relation_id: int) -> GroundedFact | None:
    row = deps.conn.execute(
        """
        SELECT r.id, r.subject_entity_id, r.object_entity_id, r.predicate,
               s.canonical_name AS subj, o.canonical_name AS obj,
               r.raw_subject, r.raw_predicate, r.raw_object,
               r.mapper_confidence, r.source_doc_id, r.source_chunk_id, r.source_quote
        FROM kg_relations r
        JOIN kg_entities s ON s.id = r.subject_entity_id
        JOIN kg_entities o ON o.id = r.object_entity_id
        WHERE r.id = ?
        """,
        (relation_id,),
    ).fetchone()
    if row is None:
        return None
    corr = deps.conn.execute(
        """
        SELECT COUNT(DISTINCT source_doc_id) AS n FROM kg_relations
        WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ?
        """,
        (row["subject_entity_id"], row["predicate"], row["object_entity_id"]),
    ).fetchone()
    pred_display = row["predicate"].replace("_", " ")
    canonical = f"{row['subj']} {pred_display} {row['obj']}"
    raw = " ".join(
        p for p in (row["raw_subject"], row["raw_predicate"], row["raw_object"]) if p
    ) or canonical
    return GroundedFact(
        kind="relation",
        fact_id=row["id"],
        subject=row["subj"],
        predicate=row["predicate"],
        object=row["obj"],
        canonical=canonical,
        raw=raw,
        condition="",
        confidence=row["mapper_confidence"],
        corroboration=int(corr["n"]) if corr else 1,
        source_doc_id=row["source_doc_id"],
        chunk_span=_chunk_span(deps, row["source_chunk_id"]),
        citation=citation_for(deps, row["source_doc_id"]),
        source_quote=_clean_quote(row["source_quote"]),
    )


def ground_attribute(deps: QueryDeps, attribute_id: int) -> GroundedFact | None:
    row = deps.conn.execute(
        """
        SELECT p.id, p.entity_id, e.canonical_name AS subj, p.property,
               p.value_num, p.value_raw, p.unit, p.condition,
               p.raw_property, p.raw_value, p.mapper_confidence,
               p.source_doc_id, p.source_chunk_id, p.source_quote
        FROM kg_entity_properties p
        JOIN kg_entities e ON e.id = p.entity_id
        WHERE p.id = ?
        """,
        (attribute_id,),
    ).fetchone()
    if row is None:
        return None
    corr = deps.conn.execute(
        """
        SELECT COUNT(DISTINCT source_doc_id) AS n FROM kg_entity_properties
        WHERE entity_id = ? AND property = ? AND value_raw = ? AND condition = ?
        """,
        (row["entity_id"], row["property"], row["value_raw"], row["condition"]),
    ).fetchone()
    prop_display = row["property"].replace("_", " ")
    value_display = format_value(row["value_raw"], row["unit"])
    cond = row["condition"] or ""
    cond_suffix = f" ({cond})" if cond else ""
    canonical = f"{row['subj']} {prop_display} {value_display}{cond_suffix}"
    raw = " ".join(p for p in (row["raw_property"], row["raw_value"]) if p) or canonical
    return GroundedFact(
        kind="attribute",
        fact_id=row["id"],
        subject=row["subj"],
        predicate=row["property"],
        object=value_display,
        canonical=canonical,
        raw=raw,
        condition=cond,
        confidence=row["mapper_confidence"],
        corroboration=int(corr["n"]) if corr else 1,
        source_doc_id=row["source_doc_id"],
        chunk_span=_chunk_span(deps, row["source_chunk_id"]),
        citation=citation_for(deps, row["source_doc_id"]),
        source_quote=_clean_quote(row["source_quote"]),
    )


def format_bundle(facts: list[GroundedFact], *, with_snippet: bool = True) -> str:
    """Render a grounded bundle as a numbered evidence list for an agent prompt.
    Each fact shows L1 canonical, the L2 "as stated" raw form, the citation, the
    corroboration count, and (optionally) the L3 source snippet — the ``[n]``
    index is what answers cite."""

    lines: list[str] = []
    for i, f in enumerate(facts, start=1):
        cite = f.citation
        src = f"{cite.title} ({cite.source})" if cite else (f.source_doc_id or "uncited")
        url = f" {cite.url}" if cite and cite.url else ""
        conf = f"{f.confidence:.2f}" if f.confidence is not None else "—"
        lines.append(f"[{i}] {f.canonical}")
        if f.raw and f.raw.lower() != f.canonical.lower():
            lines.append(f'    as stated: "{f.raw}"')
        lines.append(f"    source: {src}{url} · corroborated by {f.corroboration} doc(s) · conf {conf}")
        if with_snippet and f.chunk_span:
            snippet = f.chunk_span[:300]
            lines.append(f'    snippet: "{snippet}"')
    return "\n".join(lines) or "(no evidence retrieved)"


def source_view(deps: QueryDeps, ref: str) -> dict | None:
    """The agent-facing 'read the source' view of a fact ref: the verbatim source
    sentence + surrounding context + citation. Returns ``None`` for a bad/unknown
    ref. Shared by the Retriever and Explorer ``read_source`` tools."""

    parsed = parse_ref(ref)
    if parsed is None:
        return None
    facts = ground_facts(deps, [parsed])
    if not facts:
        return None
    f = facts[0]
    return {
        "ref": ref,
        "fact": f.canonical,
        "source_quote": f.source_quote or "(no verbatim sentence stored)",
        "context": (f.chunk_span or "")[:1200],
        "source": f.citation.title if f.citation else (f.source_doc_id or "—"),
        "url": f.citation.url if f.citation else None,
        "corroboration": f.corroboration,
    }


def parse_ref(ref: str) -> tuple[str, int] | None:
    """Parse a ``"kind:id"`` fact ref. Returns ``None`` for malformed refs."""

    parts = (ref or "").split(":")
    if len(parts) != 2:
        return None
    kind, raw_id = parts[0].strip().lower(), parts[1].strip()
    if kind not in ("relation", "attribute") or not raw_id.isdigit():
        return None
    return kind, int(raw_id)


def ground_refs(deps: QueryDeps, refs: list[str]) -> list[GroundedFact]:
    """Ground a list of ``"kind:id"`` refs (the Retriever's ``selected``),
    deduping by fact id and dropping unparseable / unknown refs."""

    seen: set[tuple[str, int]] = set()
    pairs: list[tuple[str, int]] = []
    for ref in refs:
        parsed = parse_ref(ref)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        pairs.append(parsed)
    return ground_facts(deps, pairs)


def ground_facts(
    deps: QueryDeps, fact_ids: list[tuple[str, int]]
) -> list[GroundedFact]:
    """Ground a mixed list of ``(kind, id)`` references. Unknown ids are
    dropped (agents may only cite ids the toolbox handed them — a hallucinated
    id simply doesn't ground)."""

    out: list[GroundedFact] = []
    for kind, fact_id in fact_ids:
        fact = (
            ground_relation(deps, fact_id)
            if kind == "relation"
            else ground_attribute(deps, fact_id)
            if kind == "attribute"
            else None
        )
        if fact is not None:
            out.append(fact)
    return out
