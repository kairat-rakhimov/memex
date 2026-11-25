"""Confidence-calibration harness for the Mapper's mapper_confidence.

``mapper_confidence`` is persisted on every relation + attribute but is
currently inert — an earlier build showed avg 0.992 (min 0.5, max 1.0), no spread, so
it looks informative without being so. This harness has two jobs:

  measure  Read a built ``kg.db``, bucket statements by their PREDICTED
           mapper_confidence, LLM-judge each statement's ACTUAL faithfulness
           against its source excerpt, and print the calibration curve
           (predicted vs actual) + expected-calibration-error + the raw
           confidence spread. Surfaces whether the stored number means anything.

  ab       A/B the confidence ELICITATION prompt: re-elicit a confidence for a
           fixed sample of (source excerpt, statement) pairs under several
           prompt variants and compare the resulting SPREAD (stdev / range /
           histogram). The fix for the degenerate distribution — pick the
           variant that earns a real spread, then port it into the Mapper.

    python experiments/eval_confidence_calibration.py \
        <kg_id> [--mode measure|ab] [--lane relation|attribute|both] \
        [--limit 120] [--variant plain,full_range,rubric]

The bucketing / spread / ECE math is pure + unit-tested
(``tests/test_confidence_calibration.py``); the judge + elicitor are live
LLM (reasoning tier). SAMPLE_MODE short-circuits the LLM for a plumbing
dry-run (degenerate numbers — expected).
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import build_llm_model

# ── bucketing / spread / ECE (pure; unit-tested) ──────────────────────────

_DEFAULT_BOUNDS: tuple[float, ...] = (0.0, 0.5, 0.7, 0.9, 1.0)
_EXCERPT_CAP = 4000


@dataclass(frozen=True)
class Observation:
    predicted: float
    faithful: bool


@dataclass(frozen=True)
class BucketStat:
    lo: float
    hi: float
    n: int
    mean_pred: float | None
    faithful_rate: float | None


def bucketize(
    observations: list[Observation], bounds: tuple[float, ...] = _DEFAULT_BOUNDS
) -> list[BucketStat]:
    """Group observations into ``[lo, hi)`` confidence bands (the final band is
    closed on the right so 1.0 lands), reporting per-band count, mean predicted
    confidence, and actual faithful rate. Empty bands report ``None`` rates."""

    edges = list(bounds)
    stats: list[BucketStat] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        members = [
            o
            for o in observations
            if o.predicted >= lo and (o.predicted <= hi if last else o.predicted < hi)
        ]
        if members:
            mean_pred = sum(o.predicted for o in members) / len(members)
            rate = sum(1 for o in members if o.faithful) / len(members)
        else:
            mean_pred = rate = None
        stats.append(BucketStat(lo, hi, len(members), mean_pred, rate))
    return stats


def expected_calibration_error(buckets: list[BucketStat]) -> float:
    """Σ (n_b / N) · |mean_predicted_b − actual_rate_b| over non-empty bands.
    0 = perfectly calibrated; large = the stored confidence is misleading."""

    total = sum(b.n for b in buckets)
    if total == 0:
        return 0.0
    return sum(
        (b.n / total) * abs(b.mean_pred - b.faithful_rate)
        for b in buckets
        if b.n and b.mean_pred is not None and b.faithful_rate is not None
    )


def spread_stats(values: list[float], *, bins: int = 10) -> dict:
    """Distribution summary for a confidence column: a degenerate spread (tiny
    stdev, everything in the top histogram bin) is the symptom this harness
    exists to catch."""

    if not values:
        return {"n": 0}
    hist = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int(v * bins)))
        hist[idx] += 1
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "histogram": hist,
    }


# ── live agents (LLM, reasoning tier) ─────────────────────────────────


class FaithfulnessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithful: bool
    reason: str = ""


class ConfidenceElicitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(ge=0.0, le=1.0)


_JUDGE_PROMPT = """\
You are a faithfulness judge. Given a SOURCE excerpt and a STATEMENT extracted
from it, decide whether the statement is actually supported by the excerpt —
the entities, the relationship/value, and the direction must all be present or
clearly entailed. Answer `faithful` true/false with a one-clause reason. Judge
ONLY against the excerpt; do not use outside knowledge.
"""

_ELICITATION_VARIANTS: dict[str, str] = {
    # The degenerate baseline: a bare ask. Models pile up near 1.0.
    "plain": (
        "Rate your confidence from 0 to 1 that the STATEMENT is faithfully "
        "supported by the SOURCE excerpt."
    ),
    # Range-forcing: name what each band means and warn against top-piling.
    "full_range": (
        "Rate your confidence the STATEMENT is faithfully supported, using the "
        "FULL 0-1 range. Reserve 0.9-1.0 for near-verbatim support; 0.6-0.85 for "
        "well-supported paraphrases; 0.3-0.55 for plausible inferences that the "
        "text only partly states; below 0.3 for weak/unsupported. Most "
        "extractions are NOT 0.95+ — do not pile up near 1.0."
    ),
    # Rubric-anchored: explicit evidence criteria.
    "rubric": (
        "Score confidence the STATEMENT is faithfully supported by this rubric: "
        "1.0 the excerpt states it explicitly; 0.8 it paraphrases it closely; "
        "0.6 it is a reasonable inference from stated facts; 0.4 it is partly "
        "supported / direction unclear; 0.2 only a keyword overlaps; 0.0 it "
        "contradicts or is absent. Interpolate."
    ),
}


def _judge_agent() -> Agent[None, FaithfulnessVerdict]:
    return Agent(
        model=None, output_type=FaithfulnessVerdict, system_prompt=_JUDGE_PROMPT,
        retries=2, defer_model_check=True,
    )


def _elicitor_agent(variant_instruction: str) -> Agent[None, ConfidenceElicitation]:
    system = (
        "You estimate how well a SOURCE excerpt supports an extracted STATEMENT.\n"
        + variant_instruction
    )
    return Agent(
        model=None, output_type=ConfidenceElicitation, system_prompt=system,
        retries=2, defer_model_check=True,
    )


def _pair_prompt(excerpt: str, statement: str) -> str:
    return f"SOURCE excerpt:\n{excerpt}\n\nSTATEMENT:\n{statement}"


# ── kg.db loading ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Statement:
    text: str
    predicted: float
    excerpt: str


def _excerpt_for(
    con: sqlite3.Connection,
    chunk_id: int | None,
    doc_id: str | None,
    doc_text_cache: dict[str, str | None],
) -> str:
    if doc_id not in doc_text_cache:
        row = con.execute(
            "SELECT text FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        doc_text_cache[doc_id] = row[0] if row else None
    full = doc_text_cache.get(doc_id)
    if full is None:
        return ""
    if chunk_id is not None:
        ch = con.execute(
            "SELECT char_start, char_end FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if ch is not None:
            return full[ch[0] : ch[1]][:_EXCERPT_CAP]
    return full[:_EXCERPT_CAP]


def _load_statements(db: Path, lane: str, limit: int) -> list[Statement]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cache: dict[str, str | None] = {}
    out: list[Statement] = []

    if lane in ("relation", "both"):
        rows = con.execute(
            """
            SELECT s.canonical_name AS subj, r.predicate AS pred,
                   o.canonical_name AS obj, r.mapper_confidence AS conf,
                   r.source_chunk_id AS chunk_id, r.source_doc_id AS doc_id
            FROM kg_relations r
            JOIN kg_entities s ON s.id = r.subject_entity_id
            JOIN kg_entities o ON o.id = r.object_entity_id
            WHERE r.mapper_confidence IS NOT NULL
            ORDER BY r.id
            """
        ).fetchall()
        for row in rows:
            out.append(
                Statement(
                    text=f"{row['subj']} — {row['pred']} — {row['obj']}",
                    predicted=float(row["conf"]),
                    excerpt=_excerpt_for(con, row["chunk_id"], row["doc_id"], cache),
                )
            )

    if lane in ("attribute", "both"):
        rows = con.execute(
            """
            SELECT e.canonical_name AS ent, p.property AS prop, p.value_raw AS val,
                   p.unit AS unit, p.condition AS cond, p.mapper_confidence AS conf,
                   p.source_chunk_id AS chunk_id, p.source_doc_id AS doc_id
            FROM kg_entity_properties p
            JOIN kg_entities e ON e.id = p.entity_id
            WHERE p.mapper_confidence IS NOT NULL
            ORDER BY p.id
            """
        ).fetchall()
        for row in rows:
            cond = f" ({row['cond']})" if row["cond"] else ""
            unit = f" {row['unit']}" if row["unit"] else ""
            out.append(
                Statement(
                    text=f"{row['ent']} — {row['prop']} = {row['val']}{unit}{cond}",
                    predicted=float(row["conf"]),
                    excerpt=_excerpt_for(con, row["chunk_id"], row["doc_id"], cache),
                )
            )
    con.close()
    return _subsample(out, limit)


def _subsample(items: list[Statement], limit: int) -> list[Statement]:
    """Even stride over the corpus (deterministic) rather than a head slice, so
    the sample isn't biased toward the first doc."""

    if limit <= 0 or len(items) <= limit:
        return items
    stride = len(items) / limit
    return [items[int(i * stride)] for i in range(limit)]


# ── modes ──────────────────────────────────────────────────────────────────


async def _gather_capped(coros, cap: int = 8):
    sem = asyncio.Semaphore(cap)

    async def _one(c):
        async with sem:
            return await c

    return await asyncio.gather(*(_one(c) for c in coros))


async def run_measure(
    statements: list[Statement], settings: Settings, model
) -> None:
    judge = _judge_agent()

    async def _judge(stmt: Statement) -> Observation:
        if settings.sample_mode:
            faithful = True
        else:
            run = await judge.run(_pair_prompt(stmt.excerpt, stmt.text), model=model)
            faithful = run.output.faithful
        return Observation(predicted=stmt.predicted, faithful=faithful)

    observations = await _gather_capped([_judge(s) for s in statements])

    spread = spread_stats([o.predicted for o in observations])
    buckets = bucketize(observations)
    ece = expected_calibration_error(buckets)

    print(f"\n===== CONFIDENCE CALIBRATION ({len(observations)} statements) =====")
    print(f"  predicted-confidence spread: {_fmt_spread(spread)}")
    print("  calibration curve (predicted band → actual faithful rate):")
    for b in buckets:
        if not b.n:
            print(f"    [{b.lo:.2f},{b.hi:.2f}]  n=0")
            continue
        print(
            f"    [{b.lo:.2f},{b.hi:.2f}]  n={b.n:4d}  "
            f"mean_pred={b.mean_pred:.3f}  actual_faithful={b.faithful_rate:.3f}"
        )
    print(f"  expected calibration error (ECE): {ece:.3f}  (0 = perfect)")
    if spread.get("stdev", 0.0) < 0.05:
        print(
            "  ⚠ DEGENERATE spread (stdev < 0.05): the stored mapper_confidence "
            "carries ~no information — run `--mode ab` to fix the elicitation."
        )


async def run_ab(
    statements: list[Statement], variants: list[str], settings: Settings, model
) -> None:
    print(f"\n===== ELICITATION A/B ({len(statements)} pairs) =====")
    summary: dict[str, dict] = {}
    for variant in variants:
        agent = _elicitor_agent(_ELICITATION_VARIANTS[variant])

        async def _elicit(stmt: Statement, _agent=agent) -> float:
            if settings.sample_mode:
                return 0.9
            run = await _agent.run(_pair_prompt(stmt.excerpt, stmt.text), model=model)
            return run.output.confidence

        values = await _gather_capped([_elicit(s) for s in statements])
        spread = spread_stats(values)
        summary[variant] = spread
        print(f"\n  --- {variant} ---")
        print(f"    spread: {_fmt_spread(spread)}")

    if len(summary) >= 2:
        print("\n  ===== SPREAD DELTA (stdev — higher = more discriminating) =====")
        base = variants[0]
        for v in variants[1:]:
            d = summary[v].get("stdev", 0) - summary[base].get("stdev", 0)
            print(f"    {v} vs {base}: stdev {summary[base].get('stdev', 0):.3f} → "
                  f"{summary[v].get('stdev', 0):.3f}  ({d:+.3f})")


def _fmt_spread(spread: dict) -> str:
    if not spread.get("n"):
        return "(no data)"
    return (
        f"n={spread['n']} mean={spread['mean']:.3f} stdev={spread['stdev']:.3f} "
        f"min={spread['min']:.3f} max={spread['max']:.3f} hist={spread['histogram']}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kg_id")
    ap.add_argument("--mode", choices=["measure", "ab"], default="measure")
    ap.add_argument("--lane", choices=["relation", "attribute", "both"], default="relation")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument(
        "--variant",
        default="plain,full_range,rubric",
        help="comma-list of " + "|".join(_ELICITATION_VARIANTS) + " (ab mode)",
    )
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    db = Path("kg") / args.kg_id / "kg.db"
    if not db.exists():
        raise SystemExit(f"no kg.db at {db}")

    settings = Settings.from_env()
    model = None
    if not settings.sample_mode:
        if not settings.api_key:
            raise SystemExit("Need OPENAI_API_KEY set (or SAMPLE_MODE=true for a dry-run).")
        model = build_llm_model(settings, "reasoning")

    statements = _load_statements(db, args.lane, args.limit)
    if not statements:
        raise SystemExit(f"no {args.lane} statements with mapper_confidence in {db}")
    print(f"kg={args.kg_id} lane={args.lane} loaded={len(statements)} mode={args.mode}")

    if args.mode == "measure":
        await run_measure(statements, settings, model)
    else:
        variants = [v.strip() for v in args.variant.split(",") if v.strip()]
        for v in variants:
            if v not in _ELICITATION_VARIANTS:
                raise SystemExit(f"unknown variant {v!r}; choose from {list(_ELICITATION_VARIANTS)}")
        await run_ab(statements, variants, settings, model)


if __name__ == "__main__":
    asyncio.run(main())
