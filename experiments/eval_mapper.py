"""Per-agent harness: A/B the Mapper's canonicalization prompt on ONE fixed doc.

Reuses the real Mapper user-prompt builder + output schema; swaps only the
SYSTEM prompt so prompt variants run against the SAME fixed raw triples (read
from a prior build's stored ``raw_*`` columns) — cheap iteration without
re-running a full build. Quality-refocus metrics: passive-form count + sentence-
predicate count (FORM, lower better), and the canonical←{raw} MERGE GROUPS for
human judgement (synonyms = good, distinct-meaning = over-collapse = bad).
Distinct-predicate count is informational, NOT a defect.

    python experiments/eval_mapper.py <kg_id> \
        --variant original,shipped [--doc-id ID] [--limit 120] \
        [--candidates empty|topN]

All variants see identical input + candidate set, so the diff is the prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import Counter
from pathlib import Path

from pydantic_ai import Agent

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import build_llm_model
from memex.kg_intelligence.agents.mapper import (
    _MAPPER_SYSTEM_PROMPT as SHIPPED_PROMPT,
)
from memex.kg_intelligence.agents.mapper import (
    MAPPER_DOC_BUDGET,
    RawTriple,
    _build_mapper_prompt,
)
from memex.kg_intelligence.agents.schemas import MapperResult
from memex.kg_intelligence.views import KgSchemaView

# ── prompt variants under test ──────────────────────────────────────────────
#
# ``shipped`` is the live Mapper prompt (imported). To tune, add a frozen
# candidate string here and run e.g. ``--variant shipped,candidate``; the diff
# is the prompt (identical input + candidate set). Pre-refactor prompts aren't
# kept here — they instruct per-triple entity types the current schema forbids.

# `candidate` strengthens the FORM bullet to strip baked-in subject/object nouns
# and stranded trailing prepositions (the clause-singleton tail seen in early builds:
# performance_depends_on, contains_guest_molecule, enables_generation_of).
#
# A/B verdict (2026-06-01, one smoke-build doc, 120 raw triples): NOT SHIPPED.
# form_violations 0→0 — neither prompt produced gate-violating forms on the test
# doc, so no measurable benefit, and the deterministic gate already
# rejects the 3-token stranded-prep cases. (The merge-group spread in topN mode is
# a small-seed artifact — the real build sprawls to 326 predicates, it does not
# over-collapse.) Kept here as a reproducible reference variant.
_OLD_FORM_BULLET = (
    "- **A relation phrase, not a sentence — max 3 words.** Strip modality, hedges,\n"
    "  quantifiers, and clauses; the object goes in the object slot, never inside the\n"
    "  predicate: `aims_to_reduce_cost_of` → `reduces` (object=cost);\n"
    "  `expected_to_grow_by` → `grows`; `does_not_mandate_use_of` → `not_required_for`;\n"
    "  `investment_increasing_due_to_policy` → `driven_by`."
)
_NEW_FORM_BULLET = (
    "- **A relation phrase, not a sentence — max 3 words, and NEVER a baked-in\n"
    "  subject or object noun or a stranded trailing preposition.** Strip modality,\n"
    "  hedges, quantifiers, and clauses; the subject AND object live in their slots,\n"
    "  never inside the predicate. Reduce to the core active verb:\n"
    "  `aims_to_reduce_cost_of` → `reduces`; `performance_depends_on` → `depends_on`;\n"
    "  `enables_generation_of` → `generates`; `contains_guest_molecule` → `contains`;\n"
    "  `sets_standards_for` → `regulates`; `investment_increasing_due_to_policy` →\n"
    "  `driven_by`. A 2-word `verb_prep` (`used_in`, `depends_on`) is fine; a 3-word\n"
    "  one ending in a preposition is a clause — drop the embedded noun."
)
_CANDIDATE_PROMPT = SHIPPED_PROMPT.replace(_OLD_FORM_BULLET, _NEW_FORM_BULLET)
assert _CANDIDATE_PROMPT != SHIPPED_PROMPT, "candidate form-bullet anchor did not match"

VARIANTS = {"shipped": SHIPPED_PROMPT, "candidate": _CANDIDATE_PROMPT}

# A small seed of general, active canonical predicates for the `topN` candidate
# mode — lets the harness measure REUSE behaviour (does the variant map onto
# these instead of minting?).
_SEED_PREDICATES = [
    "used_in", "produces", "affects", "requires", "enables", "contains",
    "located_in", "synthesizes", "enhances", "reduces", "catalyzes", "inhibits",
    "compared_with", "depends_on", "supports", "includes", "has_property",
    "driven_by", "improves_with", "tested_in",
]
_SEED_TYPES = ["concept", "process", "material", "chemical_substance", "device",
               "facility", "organization", "catalyst", "method", "region"]


def _load(db: Path, doc_id: str | None, limit: int):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    if doc_id is None:
        doc_id = con.execute(
            "SELECT source_doc_id FROM kg_relations WHERE source_doc_id IS NOT NULL "
            "GROUP BY source_doc_id ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()[0]
    title = con.execute(
        "SELECT json_extract(metadata,'$.title') FROM documents WHERE doc_id=?", (doc_id,)
    ).fetchone()
    title = title[0] if title else "(untitled)"
    text = con.execute("SELECT text FROM documents WHERE doc_id=?", (doc_id,)).fetchone()[0]
    rows = con.execute(
        "SELECT raw_subject, raw_predicate, raw_object FROM kg_relations "
        "WHERE source_doc_id=? AND raw_predicate IS NOT NULL ORDER BY id LIMIT ?",
        (doc_id, limit),
    ).fetchall()
    con.close()
    triples = [RawTriple(s, p, o) for s, p, o in rows]
    return doc_id, title, text, triples


def _is_passive(p: str) -> bool:
    return p.startswith(("is_", "has_", "are_", "was_", "be_")) or p.endswith("_by")


_CLAUSE_MARKERS = ("due_to", "because", "in_order", "_to_", "expected", "planned",
                   "aims", "likely", "necessarily", "able_to")


def _is_sentence(p: str) -> bool:
    """Malformed: a clause, not a relation phrase. Specificity is allowed —
    only >4 tokens or an embedded modality/clause marker counts."""
    return len(p.split("_")) > 4 or any(m in p for m in _CLAUSE_MARKERS)


def _report(label: str, result: MapperResult) -> dict:
    from memex.kg_intelligence.agents.gate import predicate_form_violation

    lc = [t.canonical_predicate.lower() for t in result.triples]
    distinct = sorted(set(lc))
    passive = {p for p in lc if _is_passive(p)}
    sentences = {p for p in lc if _is_sentence(p)}
    # The gate's own form contract — directly measures the clause/stranded-prep
    # target (what the candidate prompt aims to reduce at the source).
    form_violations = sorted({p for p in distinct if predicate_form_violation(p)})
    # Collapse groups: canonical ← {distinct raw predicates}. The KEY quality
    # signal now — a group is GOOD if its raws are synonyms, BAD if they mean
    # different things (over-collapse). Human-judged.
    groups: dict[str, set[str]] = {}
    for t in result.triples:
        groups.setdefault(t.canonical_predicate.lower(), set()).add(t.raw_predicate.lower())
    merged = {c: rs for c, rs in groups.items() if len(rs) > 1}
    merge_rels = sum(len(rs) for rs in merged.values())
    print(f"\n===== {label} =====")
    print(f"  triples mapped:        {len(result.triples)}")
    print(f"  distinct predicates:   {len(distinct)}  (informational, not a defect)")
    print(f"  FORM VIOLATIONS (gate): {len(form_violations)} predicates  [quality: lower better]")
    if form_violations:
        print(f"    {', '.join(form_violations[:20])}")
    print(f"  PASSIVE (is_/has_/_by): {sum(1 for p in lc if _is_passive(p))} rels / {len(passive)} predicates  [quality: lower better]")
    print(f"  SENTENCE-predicates:    {sum(1 for p in lc if _is_sentence(p))} rels / {len(sentences)} predicates  [quality: lower better]")
    print(f"  merges (canonical←≥2 raws): {len(merged)} groups covering {merge_rels} raw forms")
    if passive:
        print(f"  passive forms: {', '.join(sorted(passive)[:12])}")
    if sentences:
        print(f"  sentence forms: {', '.join(sorted(sentences)[:10])}")
    print("  MERGE GROUPS (judge: synonyms=good, distinct-meaning=BAD):")
    for c, rs in sorted(merged.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"    {c}  ←  {sorted(rs)}")

    # ── entity-type quality (per-entity typing) ──
    types = [et.canonical_type.lower() for et in result.entity_types]
    generic = {"concept", "thing", "entity", "object", "item"}
    n_generic = sum(1 for t in types if t in generic)
    malformed = [t for t in types if " " in t or t != t.strip() or len(t.split("_")) > 3]
    tcounts = Counter(types)
    print(f"  entity_types emitted:  {len(types)}  ({len(set(types))} distinct)")
    print(f"  GENERIC (concept/...): {n_generic}  ({n_generic/max(1,len(types)):.0%})  [quality: lower better]")
    print(f"  malformed (space/>3w): {len(malformed)}")
    print(f"  top types: {', '.join(f'{t}×{n}' for t, n in tcounts.most_common(12))}")
    return {
        "distinct": len(distinct),
        "form_violations": len(form_violations),
        "passive": sum(1 for p in lc if _is_passive(p)),
        "sentences": sum(1 for p in lc if _is_sentence(p)),
        "merge_groups": len(merged),
        "generic_types": n_generic,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kg_id")
    ap.add_argument("--variant", default="shipped",
                    help="comma-list of " + "|".join(VARIANTS)
                    + " or new frozen candidates (default shipped)")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--candidates", choices=["empty", "topN"], default="topN")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    db = Path("kg") / args.kg_id / "kg.db"
    doc_id, title, text, triples = _load(db, args.doc_id, args.limit)
    settings = Settings.from_env()
    if settings.sample_mode or not settings.api_key:
        raise SystemExit("Need OPENAI_API_KEY set and SAMPLE_MODE off.")
    model = build_llm_model(settings, "reasoning")

    schema = (
        KgSchemaView(entity_types=list(_SEED_TYPES), relation_types=list(_SEED_PREDICATES))
        if args.candidates == "topN" else KgSchemaView()
    )
    print(f"doc {doc_id} | {len(triples)} raw triples | candidates={args.candidates} "
          f"({len(schema.relation_types)} preds) | {len(text)} chars")
    print(f"title: {title}")

    user_prompt = _build_mapper_prompt(title, text[:MAPPER_DOC_BUDGET], triples, schema, None)
    to_run = [v.strip() for v in args.variant.split(",") if v.strip()]
    for v in to_run:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {list(VARIANTS)}")
    order, summary = [], {}
    for v in to_run:
        agent: Agent[None, MapperResult] = Agent(
            model=None, output_type=MapperResult, system_prompt=VARIANTS[v],
            retries=2, defer_model_check=True,
        )
        run = await agent.run(user_prompt, model=model)
        summary[v] = _report(v, run.output)
        order.append(v)

    if len(order) >= 2:
        a, b = order[0], order[1]
        print(f"\n===== DELTA ({b} vs {a}) =====")
        for k in ["distinct", "form_violations", "passive", "sentences", "merge_groups", "generic_types"]:
            print(f"  {k:14s} {summary[a][k]:4d} → {summary[b][k]:4d}  ({summary[b][k]-summary[a][k]:+d})")


if __name__ == "__main__":
    asyncio.run(main())
