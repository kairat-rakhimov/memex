"""Per-agent harness: A/B the AttributeMapper prompt on ONE fixed doc.

The attribute-lane analog of ``eval_mapper.py``. Reuses the real AttributeMapper
user-prompt builder + output schema; swaps only the SYSTEM prompt so variants run
against the SAME fixed raw attributes (entity, raw_property, raw_value), read from
a prior build's ``kg_entity_properties`` rows — cheap iteration without a rebuild.

Targets two attribute-quality fixes:
  A1 (condition capture)  — condition-fill % on NUMERIC attributes [higher better]
  A2 (unit normalization) — (property → distinct units) mixing     [lower better]
No-regression guard: rich multi-value attributes (a property with ≥2 conditions)
must keep their conditions; the harness prints them for eyeballing.

    python experiments/eval_attr_mapper.py <kg_id> \
        --variant shipped,candidate [--doc-id ID] [--limit 80]

All variants see identical input, so the diff is the prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import defaultdict
from pathlib import Path

from pydantic_ai import Agent

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import build_llm_model
from memex.kg_intelligence.agents.mapper import (
    _ATTRIBUTE_MAPPER_SYSTEM_PROMPT as SHIPPED_PROMPT,
)
from memex.kg_intelligence.agents.mapper import (
    MAPPER_DOC_BUDGET,
    RawAttribute,
    _build_attribute_prompt,
)
from memex.kg_intelligence.agents.schemas import AttributeMapperResult
from memex.kg_intelligence.views import KgSchemaView

_DUMP_CONDITIONS = False  # set by --dump-conditions; prints filled conditions for faithfulness review

# ── prompt variants under test ──────────────────────────────────────────────
#
# A/B verdict (2026-06-04, green-hydrogen + saline-electrolysis docs): SHIPPED the
# conservative A1 (capture conditions from the value's OWN sentence; no
# section-scanning) + A2 unit normalization. green-H2 doc: condition-fill
# 34%→68%, property sprawl 59→44, and scenarios land in `condition` not baked
# into property NAMES (`allocation_share_for_hydrogen`×3 with `conservative/
# medium/extreme scenario` conditions, vs the aggressive split into
# `conservative_allocation_share` / `medium_…` / `extreme_…`). saline doc:
# 95%→95% (no over-fabrication), notation canonicalized (`m2 g-1`→`m2/g`,
# `US$ m-2`→`USD/m2`). The ``aggressive`` variant matched the fill rate but
# stuffed value-restatement / criteria lists into conditions — conservative was
# cleaner, so it shipped. ``baseline`` reverts both bullets to the pre-A1/A2
# prompt to reproduce the headline delta.
#
# ``shipped`` is the live prompt (imported — now contains conservative A1 + A2).

_NEW_CONDITION_BULLET = (
    "- `condition`: the scenario the value holds under — **capture it whenever the\n"
    "  value's own sentence states one**: system boundary (stack vs system, LHV vs\n"
    "  HHV), scenario / case (`RCP2.6`, `2030 policy case`), YEAR, LOCATION / REGION,\n"
    "  and measurement setup (electrolyte, temperature, current density, reaction).\n"
    "  Normalize + comma-separate (`Southern Africa, 2040`; `RCP2.6, 2016-2035`).\n"
    "  Do NOT reach into other sentences or invent a plausible scenario — only what\n"
    "  the value's immediate context states. Prefer dropping a value you cannot\n"
    "  qualify over a misleading blank. `\"\"` when the sentence states no condition."
)
_NEW_UNIT_BULLET = (
    "- `unit`: canonicalize the unit to a STABLE notation and, when the conversion is\n"
    "  a pure scale factor, convert `value_num` to ONE canonical unit per quantity so\n"
    "  `(property, unit)` groups are genuinely comparable:\n"
    "  - normalize notation: `US$/kg`, `USD/kgH2`, `$/kg H2` → `USD/kg_H2`;\n"
    "    `mA cm-2` → `mA/cm2`; `degC` → `°C`. Same quantity ⇒ same unit string.\n"
    "  - convert pure-scale units to the canonical one and SCALE `value_num` to match\n"
    "    (`1.5 V` where the quantity's canonical unit is `mV` → 1500, unit `mV`;\n"
    "    `0.5 MW` where canonical is `kW` → 500). Pick the unit that fits most values.\n"
    "  - **NEVER fabricate a conversion you cannot do exactly** — different currencies\n"
    "    (EUR vs USD) or any unit needing an external rate / assumption pass through\n"
    "    UNCONVERTED with their own unit. Do NOT guess an FX rate.\n"
    "  - `\"\"` when unitless or qualitative."
)
# Pre-A1/A2 bullets — to reconstruct the ``baseline`` variant from the shipped prompt.
_OLD_CONDITION_BULLET = (
    "- `condition`: the conditions the value holds under (electrolyte, temperature,\n"
    "  current density, reaction), normalized + comma-separated; \"\" if none stated."
)
_OLD_UNIT_BULLET = (
    "- `unit`: the unit string (`mV`, `%`, `°C`, `mA cm-2`, `MJ/kg`, `US$/kg`); \"\" when\n"
    "  unitless or qualitative."
)
# The aggressive A1 (rejected): scan the surrounding section/table, not just the
# value's sentence — tempted the model to split property names + stuff conditions.
_AGGRESSIVE_CONDITION_BULLET = (
    "- `condition`: the scenario the value holds under — **always capture it when the\n"
    "  document states one.** Look beyond the value's own clause to the surrounding\n"
    "  sentence / table caption / section for: system boundary, scenario / case,\n"
    "  YEAR, LOCATION / REGION, and measurement setup. Normalize + comma-separate.\n"
    "  Only leave \"\" when the document genuinely states no qualifying condition."
)

_BASELINE_PROMPT = SHIPPED_PROMPT.replace(
    _NEW_CONDITION_BULLET, _OLD_CONDITION_BULLET
).replace(_NEW_UNIT_BULLET, _OLD_UNIT_BULLET)
_AGGRESSIVE_PROMPT = SHIPPED_PROMPT.replace(_NEW_CONDITION_BULLET, _AGGRESSIVE_CONDITION_BULLET)
assert _NEW_CONDITION_BULLET in SHIPPED_PROMPT, "shipped prompt missing A1 bullet — did the live prompt change?"
assert _NEW_UNIT_BULLET in SHIPPED_PROMPT, "shipped prompt missing A2 bullet — did the live prompt change?"
assert _BASELINE_PROMPT not in (SHIPPED_PROMPT, _AGGRESSIVE_PROMPT), "baseline/aggressive anchors did not match"

VARIANTS = {
    "shipped": SHIPPED_PROMPT,
    "baseline": _BASELINE_PROMPT,
    "aggressive": _AGGRESSIVE_PROMPT,
}


def _load(db: Path, doc_id: str | None, limit: int):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    if doc_id is None:
        doc_id = con.execute(
            "SELECT source_doc_id FROM kg_entity_properties "
            "WHERE source_doc_id IS NOT NULL AND value_num IS NOT NULL "
            "GROUP BY source_doc_id ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()[0]
    title = con.execute(
        "SELECT json_extract(metadata,'$.title') FROM documents WHERE doc_id=?", (doc_id,)
    ).fetchone()
    title = title[0] if title else "(untitled)"
    text = con.execute("SELECT text FROM documents WHERE doc_id=?", (doc_id,)).fetchone()[0]
    rows = con.execute(
        "SELECT e.canonical_name, p.raw_property, p.raw_value "
        "FROM kg_entity_properties p JOIN kg_entities e ON e.id=p.entity_id "
        "WHERE p.source_doc_id=? AND p.raw_property IS NOT NULL "
        "ORDER BY p.id LIMIT ?",
        (doc_id, limit),
    ).fetchall()
    con.close()
    attrs = [RawAttribute(entity=e, property=pr, value=v) for e, pr, v in rows]
    return doc_id, title, text, attrs


def _report(label: str, result: AttributeMapperResult) -> dict:
    attrs = result.attributes
    numeric = [a for a in attrs if a.value_num is not None]
    cond_filled = [a for a in numeric if (a.condition or "").strip()]
    cond_rate = len(cond_filled) / max(1, len(numeric))

    # A2: same canonical property mapped to ≥2 distinct (non-empty) units = a
    # comparability hazard. Report the mixed groups for judgement.
    units_by_prop: dict[str, set[str]] = defaultdict(set)
    for a in attrs:
        if (a.unit or "").strip():
            units_by_prop[a.canonical_property.lower()].add(a.unit.strip())
    mixed = {p: u for p, u in units_by_prop.items() if len(u) > 1}

    distinct_props = len({a.canonical_property.lower() for a in attrs})
    blank_numeric = [a for a in numeric if not (a.condition or "").strip()]

    print(f"\n===== {label} =====")
    print(f"  attributes mapped:      {len(attrs)}")
    print(f"  numeric (value_num):    {len(numeric)}")
    print(f"  CONDITION-FILL (numeric): {len(cond_filled)}/{len(numeric)} = {cond_rate:.0%}  [A1: higher better]")
    print(f"  distinct properties:    {distinct_props}  (informational)")
    print(f"  MIXED-UNIT properties:  {len(mixed)}  [A2: lower better]")
    for p, u in sorted(mixed.items())[:10]:
        print(f"    {p}  ←  {sorted(u)}")
    if blank_numeric:
        print(f"  blank-condition numeric ({len(blank_numeric)}) — sample:")
        for a in blank_numeric[:8]:
            print(f"    {a.canonical_entity!r} {a.canonical_property} = {a.value_num} {a.unit!r}  (raw={a.raw_value!r})")
    if _DUMP_CONDITIONS and cond_filled:
        print(f"  FILLED conditions ({len(cond_filled)}) — judge faithfulness vs the doc:")
        for a in cond_filled:
            print(f"    {a.canonical_entity!r} {a.canonical_property}={a.value_num}{a.unit} | cond={a.condition!r} (raw={a.raw_value!r})")
    return {
        "attributes": len(attrs),
        "numeric": len(numeric),
        "cond_fill_pct": round(cond_rate * 100),
        "distinct_props": distinct_props,
        "mixed_units": len(mixed),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kg_id")
    ap.add_argument("--variant", default="baseline,shipped",
                    help="comma-list of " + "|".join(VARIANTS) + " (default reproduces the A1/A2 delta)")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--dump-conditions", action="store_true",
                    help="print each variant's filled conditions for faithfulness review")
    args = ap.parse_args()

    global _DUMP_CONDITIONS
    _DUMP_CONDITIONS = args.dump_conditions

    from dotenv import load_dotenv
    load_dotenv()

    db = Path("kg") / args.kg_id / "kg.db"
    doc_id, title, text, attrs = _load(db, args.doc_id, args.limit)
    settings = Settings.from_env()
    if settings.sample_mode or not settings.api_key:
        raise SystemExit("Need OPENAI_API_KEY set and SAMPLE_MODE off.")
    model = build_llm_model(settings, "reasoning")

    schema = KgSchemaView()
    print(f"doc {doc_id} | {len(attrs)} raw attributes | {len(text)} chars")
    print(f"title: {title}")

    user_prompt = _build_attribute_prompt(title, text[:MAPPER_DOC_BUDGET], attrs, schema)
    to_run = [v.strip() for v in args.variant.split(",") if v.strip()]
    for v in to_run:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {list(VARIANTS)}")
    order, summary = [], {}
    for v in to_run:
        agent: Agent[None, AttributeMapperResult] = Agent(
            model=None, output_type=AttributeMapperResult, system_prompt=VARIANTS[v],
            retries=2, defer_model_check=True,
        )
        run = await agent.run(user_prompt, model=model)
        summary[v] = _report(v, run.output)
        order.append(v)

    if len(order) >= 2:
        a, b = order[0], order[1]
        print(f"\n===== DELTA ({b} vs {a}) =====")
        for k in ["attributes", "numeric", "cond_fill_pct", "distinct_props", "mixed_units"]:
            print(f"  {k:16s} {summary[a][k]:4d} → {summary[b][k]:4d}  ({summary[b][k]-summary[a][k]:+d})")


if __name__ == "__main__":
    asyncio.run(main())
