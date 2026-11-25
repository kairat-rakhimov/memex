"""Per-agent harness: A/B the entity Resolver's merge judge on gold pairs.

The Resolver decides merge / distinct / ambiguous for candidate entity pairs. The
known failure mode is the CONTRASTIVE QUALIFIER: `green hydrogen` vs `gray
hydrogen` share a head noun + type + sources, so the judge can wrongly MERGE them
(erasing a real production-pathway distinction the graph depends on). The candidate prompt adds a
contrastive-qualifier rule + dials the judge conservative.

Reuses the real Resolver agent + `_build_user_prompt` + a real `kg.db` entity
snapshot (so each pair carries its true type / mentions / shared-docs context);
swaps only the SYSTEM prompt. Scores a curated GOLD set: must-merge aliases
(acronym/full-form — the conservative dialing must NOT break these) + must-distinct
contrastive qualifiers (the candidate prompt's target).

    python experiments/eval_resolver.py \\
        [--variant baseline,candidate] [--kg green-hydrogen]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic_ai import Agent

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import build_llm_model
from memex.kg_intelligence.agents.resolver import _SYSTEM_PROMPT, _build_user_prompt
from memex.kg_intelligence.agents.schemas import ResolverResult
from memex.kg_intelligence.embeddings import load_matrix, top_k_similar
from memex.observability import TraceStore

# ── gold pairs (must exist in the kg snapshot) ───────────────────────────────
# expected: "merge" (genuine alias) | "distinct" (contrastive qualifier / different thing)
#
# EASY: must-merge acronyms + obvious contrastive qualifiers (the famous hydrogen
# colours — any LLM gets these; baseline already 8/8). HARD: subset / hypernym
# pairs the Resolver's tier-2 subset rule actually surfaces, where merging would
# collapse a hyponym into its hypernym (the subtler false-merge risk the candidate prompt targets;
# relatedness is left to a possible later non-destructive entity overlay, so the
# Resolver should DECLINE — distinct — not merge).
GOLD_EASY: list[tuple[str, str, str]] = [
    ("PEM electrolysis", "proton exchange membrane electrolysis", "merge"),
    ("PEM fuel cell", "proton exchange membrane fuel cell", "merge"),
    ("green hydrogen", "gray hydrogen", "distinct"),
    ("green hydrogen", "blue hydrogen", "distinct"),
    ("blue hydrogen", "gray hydrogen", "distinct"),
    ("offshore wind", "onshore wind", "distinct"),
    ("PEM fuel cell", "PEM electrolyser", "distinct"),
    ("proton exchange membrane fuel cell", "proton exchange membrane electrolysis", "distinct"),
]
GOLD_HARD: list[tuple[str, str, str]] = [
    ("electrolyser", "PEM electrolyser", "distinct"),                 # general vs PEM-specific
    ("green hydrogen", "green hydrogen production", "distinct"),      # substance vs process
    ("offshore wind", "offshore wind farm", "distinct"),             # energy concept vs facility
    ("hydrogen infrastructure", "hydrogen refueling infrastructure", "distinct"),  # general vs specific
    ("alkaline electrolysis", "PEM electrolysis", "distinct"),       # different electrolysis tech
    ("elemental hydrogen", "green hydrogen", "distinct"),            # element vs production-grade
    ("wind", "onshore wind", "distinct"),                            # general vs location-qualified
    ("onshore wind", "onshore wind turbine", "distinct"),           # energy concept vs device
]
GOLD_SETS = {"easy": GOLD_EASY, "hard": GOLD_HARD, "both": GOLD_EASY + GOLD_HARD}

# ── candidate prompt ───────────────────────────────────────────────────
# A/B verdict (2026-06-04, an 80-doc green-hydrogen build): NOT SHIPPED — no measurable benefit.
# The candidate (contrastive-qualifier rule + conservative dialing) changed ZERO
# verdicts across 40 pairs on three surfaces: easy gold 8/8→8/8, hard subset/
# hypernym gold 8/8→8/8, and 24 REAL high-cosine candidate pairs (the live tier-3
# surface) — identical 9 merge / 15 distinct, candidate == baseline. The current judge prompt
# already carries "be conservative on merge" and, with the tier-3
# co-occurrence bypass, prevents the contrastive-qualifier
# false-merge on its own (green/gray/blue hydrogen, offshore/onshore, value-differing
# `23% vs 28% vs 42% emissions` all correctly distinct under baseline). Kept here as
# a regression artifact + the reusable entity-Resolver A/B harness (gold + `--auto`).
# (Parked observation: both prompts merge `…carbon capture` ↔ `…carbon capture and
# storage` — debatable; a job for a non-destructive entity overlay, not a judge-prompt fix.)
_OLD_DISTINCT = (
    "- distinct: the names look similar but refer to different things. Examples:\n"
    "  `Metal hydrides` ↔ `Metal oxides` (same category, different materials),\n"
    "  `iridium catalyst` ↔ `iridium oxide` (related but separate concepts)."
)
_NEW_DISTINCT = (
    "- distinct: the names look similar but refer to different things. Examples:\n"
    "  `Metal hydrides` ↔ `Metal oxides` (same category, different materials),\n"
    "  `iridium catalyst` ↔ `iridium oxide` (related but separate concepts),\n"
    "  `green hydrogen` ↔ `gray hydrogen` (a CONTRASTIVE QUALIFIER — the modifier\n"
    "  names a different production pathway / grade / kind), `offshore wind` ↔\n"
    "  `onshore wind`, `PEM fuel cell` ↔ `PEM electrolyser` (shared acronym,\n"
    "  different device)."
)
_OLD_CONSERVATIVE = (
    "Be conservative on `merge`: a false positive corrupts the KG. When unsure,\n"
    "return `ambiguous` or `distinct`."
)
_NEW_CONSERVATIVE = (
    "**Contrastive qualifiers make entities DISTINCT, not aliases.** When two names\n"
    "share a head noun but differ by a modifier that changes the KIND, grade,\n"
    "colour-code, production pathway, polarity, or location (green / blue / gray /\n"
    "clean hydrogen; n-type / p-type; anode / cathode; onshore / offshore; PEM fuel\n"
    "cell vs PEM electrolyser), they are DISTINCT real-world things — do NOT merge\n"
    "them even when they share sources and have the same type. A merge here erases a\n"
    "real distinction the graph depends on.\n\n"
    "Be conservative on `merge`: a false positive corrupts the KG and is far worse\n"
    "than a missed alias (a non-destructive cluster overlay recovers relatedness\n"
    "later). Merge ONLY a genuine alias — an acronym ↔ full-form pair, a spelling /\n"
    "morphology variant, or an obvious renaming. When unsure, return `distinct` or\n"
    "`ambiguous`."
)
_CANDIDATE_PROMPT = _SYSTEM_PROMPT.replace(_OLD_DISTINCT, _NEW_DISTINCT).replace(
    _OLD_CONSERVATIVE, _NEW_CONSERVATIVE
)
assert _CANDIDATE_PROMPT != _SYSTEM_PROMPT, "candidate anchors did not match"
assert _NEW_DISTINCT in _CANDIDATE_PROMPT and _NEW_CONSERVATIVE in _CANDIDATE_PROMPT

VARIANTS = {"baseline": _SYSTEM_PROMPT, "candidate": _CANDIDATE_PROMPT}


def _verdict_map(result: ResolverResult) -> dict:
    """Order-insensitive {frozenset(name_a, name_b): verdict}."""
    out = {}
    for v in result.verdicts:
        out[frozenset([v.name_a, v.name_b])] = (v.verdict, v.confidence, v.reason)
    return out


def _report(label: str, vmap: dict, gold: list[tuple[str, str, str]]) -> dict:
    print(f"\n===== {label} =====")
    false_merge = 0  # expected distinct, judged merge — the critical failure
    broke_merge = 0  # expected merge, judged not-merge
    correct = 0
    for a, b, expected in gold:
        verdict, conf, reason = vmap.get(frozenset([a, b]), ("(missing)", 0.0, ""))
        safe = (verdict == "merge") if expected == "merge" else (verdict in ("distinct", "ambiguous"))
        if safe:
            correct += 1
        if expected == "distinct" and verdict == "merge":
            false_merge += 1
        if expected == "merge" and verdict not in ("merge",):
            broke_merge += 1
        flag = "  ✓" if safe else "  ✗ <<<"
        print(f"  [{expected:8s}→{verdict:9s}] {a!r} ↔ {b!r}  (c={conf:.2f}) {reason[:42]}{flag}")
    print(f"  score: {correct}/{len(gold)} safe   | FALSE-MERGES={false_merge} (critical)"
          f"   broke-aliases={broke_merge}")
    return {"correct": correct, "false_merge": false_merge, "broke_merge": broke_merge}


def _auto_pairs(store, snapshot, n: int, min_cosine: float):
    """Surface the real tier-3 candidate surface: the highest-cosine entity pairs
    (sharing ≥1 source doc, mirroring the live cooccurrence filter) — the actual
    near-duplicate distribution the build judges, not hand-picked gold. Returns
    [(name_a, name_b, cosine)] sorted by cosine desc, capped at n."""
    ids, matrix = load_matrix(store, "entity")
    if matrix.size == 0:
        return []
    id_to_name = {str(ent.id): name for name, ent in snapshot.items()}
    pair_score: dict[tuple[str, str], float] = {}
    for i, eid in enumerate(ids):
        name = id_to_name.get(eid)
        if name is None:
            continue
        for idx, score in top_k_similar(matrix[i], matrix, k=4):
            if score < min_cosine:
                continue
            hit = id_to_name.get(ids[idx])
            if hit is None or hit == name:
                continue
            ea, eb = snapshot.get(name), snapshot.get(hit)
            if not ea or not eb:
                continue
            if not (set(ea.source_doc_ids) & set(eb.source_doc_ids)):
                continue  # mirror the live cooccurrence filter
            key = tuple(sorted([name, hit]))
            pair_score[key] = max(pair_score.get(key, 0.0), float(score))
    ranked = sorted(pair_score.items(), key=lambda kv: -kv[1])[:n]
    return [(a, b, s) for (a, b), s in ranked]


def _report_auto(label: str, vmap: dict, pairs) -> dict:
    """No gold labels — report the verdict distribution + flag every MERGE (the
    judge's destructive action) for human eyeballing."""
    counts = {"merge": 0, "distinct": 0, "ambiguous": 0}
    merges = []
    for a, b, cos in pairs:
        verdict, conf, reason = vmap.get(frozenset([a, b]), ("(missing)", 0.0, ""))
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "merge":
            merges.append((a, b, cos, conf, reason))
    print(f"\n===== {label} =====  merge={counts['merge']} distinct={counts['distinct']} ambiguous={counts['ambiguous']}")
    for a, b, cos, conf, reason in merges:
        print(f"  MERGE  {a!r} ↔ {b!r}  (cos={cos:.2f} c={conf:.2f}) {reason[:48]}")
    return {"merge": counts["merge"], "distinct": counts["distinct"],
            "ambiguous": counts["ambiguous"], "merges": {frozenset([a, b]) for a, b, *_ in merges}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline,candidate",
                    help="comma-list of " + "|".join(VARIANTS) + " (default both)")
    ap.add_argument("--set", default="both", choices=list(GOLD_SETS),
                    help="which gold set to judge (default both)")
    ap.add_argument("--auto", type=int, default=0,
                    help="judge N real high-cosine candidate pairs instead of gold (the live surface)")
    ap.add_argument("--min-cosine", type=float, default=0.84)
    ap.add_argument("--kg", default="green-hydrogen")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    settings = Settings.from_env()
    if settings.sample_mode or not settings.api_key:
        raise SystemExit("Need OPENAI_API_KEY set and SAMPLE_MODE off.")
    model = build_llm_model(settings, "reasoning")

    store = TraceStore.open(Path("kg") / args.kg)
    snapshot = store.entity_snapshot_for_resolver()
    to_run = [v.strip() for v in args.variant.split(",") if v.strip()]
    for v in to_run:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {list(VARIANTS)}")

    if args.auto:
        auto = _auto_pairs(store, snapshot, args.auto, args.min_cosine)
        pairs = [(a, b) for a, b, _ in auto]
        print(f"kg={args.kg} | {len(snapshot)} entities | {len(pairs)} real high-cosine "
              f"pairs (cos≥{args.min_cosine}, share-docs) — the live candidate surface")
        user_prompt = _build_user_prompt(pairs, snapshot, do_sweep=True)
        summary = {}
        for v in to_run:
            agent: Agent[None, ResolverResult] = Agent(
                model=None, output_type=ResolverResult, system_prompt=VARIANTS[v],
                retries=2, defer_model_check=True,
            )
            run = await agent.run(user_prompt, model=model)
            summary[v] = _report_auto(v, _verdict_map(run.output), auto)
        store.close()
        if len(to_run) >= 2:
            a, b = to_run[0], to_run[1]
            only_a = summary[a]["merges"] - summary[b]["merges"]
            only_b = summary[b]["merges"] - summary[a]["merges"]
            print(f"\n===== DELTA ({b} vs {a}) =====")
            print(f"  merges: {summary[a]['merge']} → {summary[b]['merge']}")
            print(f"  merges only {a} declined by {b} (candidate wins if real-distinct): {len(only_a)}")
            for fs in only_a:
                print(f"    {tuple(fs)}")
            print(f"  merges only {b} (candidate regressions if real-alias): {len(only_b)}")
            for fs in only_b:
                print(f"    {tuple(fs)}")
        return

    gold_in = GOLD_SETS[args.set]
    pairs = [(a, b) for a, b, _ in gold_in]
    missing = [(a, b) for a, b in pairs if a not in snapshot or b not in snapshot]
    if missing:
        for a, b in missing:
            who = a if a not in snapshot else b
            print(f"  MISSING from {args.kg}: {who!r}  (pair {a!r} ↔ {b!r})")
        pairs = [(a, b) for a, b in pairs if a in snapshot and b in snapshot]
        gold = [g for g in gold_in if (g[0], g[1]) in set(pairs)]
    else:
        gold = gold_in
    print(f"kg={args.kg} | {len(snapshot)} entities | judging {len(pairs)} gold pairs")

    user_prompt = _build_user_prompt(pairs, snapshot, do_sweep=True)
    summary = {}
    for v in to_run:
        agent: Agent[None, ResolverResult] = Agent(
            model=None, output_type=ResolverResult, system_prompt=VARIANTS[v],
            retries=2, defer_model_check=True,
        )
        run = await agent.run(user_prompt, model=model)
        summary[v] = _report(v, _verdict_map(run.output), gold)
    store.close()

    if len(to_run) >= 2:
        a, b = to_run[0], to_run[1]
        print(f"\n===== DELTA ({b} vs {a}) =====")
        for k in ["correct", "false_merge", "broke_merge"]:
            print(f"  {k:12s} {summary[a][k]:3d} → {summary[b][k]:3d}  ({summary[b][k] - summary[a][k]:+d})")


if __name__ == "__main__":
    asyncio.run(main())
