"""Deterministic hypothesis seeds — the structural patterns the
HypothesisGenerator articulates. Pure (no LLM): SQL + numpy + NetworkX over
``QueryDeps``. Seeds bound the LLM fan-out; each carries grounded leg fact ids so
the generated claim's premises are real.

Five lanes:

- **bridge** — (A,C) sharing an intermediate B, no direct edge, A–C *distant* in
  embedding space (surprising cross-subfield link, Swanson ABC).
- **gap** — same machinery, but A–C *close* / same community (an expected-but-
  missing link).
- **property** — per comparable numeric (property, unit): distribution outliers +
  cross-entity-TYPE shared-value clusters ("different materials, same value ⟹
  shared mechanism"). The lane prose RAG can't reach one entity at a time.
- **attribute_contested** — same (entity, property), divergent ``value_num`` under
  different ``condition`` (reconcilable — attributes carry conditions).
- **relation_contested** — (subj,obj) with ≥2 predicates; flagged low-confidence
  (an LLM judge decides genuine opposition; embeddings can't separate antonyms).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from memex.kg_query.deps import QueryDeps
from memex.kg_query.retrieval import communities

LaneName = str


@dataclass
class Seed:
    lane: LaneName
    title: str
    rationale: str
    entity_ids: list[int]
    leg_fact_ids: list[tuple[str, int]]
    score: float
    payload: dict = field(default_factory=dict)
    # relation_contested only: set by the ContestationJudge. None = unassessed
    # (treated as genuine); False = synonym/multi-facet sprawl → no contested bonus.
    genuine_conflict: bool | None = None


# ── helpers ──────────────────────────────────────────────────────────────────


def _entity_vectors(deps: QueryDeps) -> dict[int, np.ndarray]:
    ids, matrix = deps.matrix("entity")
    if matrix.size == 0:
        return {}
    return {int(tid): matrix[i] for i, tid in enumerate(ids)}


def _edge_relation_id(deps: QueryDeps, u: int, v: int) -> int | None:
    """A representative relation row id for the edge between u and v (best
    corroboration), for grounding a bridge/gap leg."""

    g = deps.graph()
    for a, b in ((u, v), (v, u)):
        if g.has_edge(a, b):
            data = g.get_edge_data(a, b)
            best = max(data.values(), key=lambda d: d.get("corroboration", 0))
            return best.get("sample_relation_id")
    return None


def _name(deps: QueryDeps, entity_id: int) -> str:
    g = deps.graph()
    return g.nodes[entity_id].get("name", str(entity_id)) if entity_id in g else str(entity_id)


# ── bridge + gap (shared ABC machinery) ──────────────────────────────────────


def _abc_triples(
    deps: QueryDeps,
    *,
    min_degree: int = 2,
    max_degree: int = 30,
    max_neighbors: int = 24,
    max_triples: int = 6000,
) -> list[tuple[int, int, int, float, bool]]:
    """Collect (A, B, C, ac_cosine, same_community) where A–B–C is a 2-path and
    A,C are NOT directly connected. Bounded for tractability; cached on ``deps``
    so the bridge + gap lanes share one sweep."""

    cached = deps._expansion.get("abc_triples")
    if cached is not None:
        return cached  # type: ignore[return-value]

    und = deps.undirected()
    vecs = _entity_vectors(deps)
    comm = {n: i for i, c in enumerate(communities(deps, min_size=3)) for n in c}
    triples: list[tuple[int, int, int, float, bool]] = []
    seen_pairs: set[tuple[int, int]] = set()

    nodes = sorted(und.nodes, key=lambda n: und.degree(n))
    for b in nodes:
        deg = und.degree(b)
        if deg < min_degree or deg > max_degree:
            continue
        nbrs = sorted(
            und.neighbors(b),
            key=lambda n: und.nodes[n].get("mentions", 0),
            reverse=True,
        )[:max_neighbors]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, c = nbrs[i], nbrs[j]
                key = (a, c) if a < c else (c, a)
                if key in seen_pairs or und.has_edge(a, c):
                    continue
                seen_pairs.add(key)
                cos = float(np.dot(vecs[a], vecs[c])) if a in vecs and c in vecs else 0.0
                same = comm.get(a) is not None and comm.get(a) == comm.get(c)
                triples.append((a, b, c, cos, same))
                if len(triples) >= max_triples:
                    deps._expansion["abc_triples"] = triples
                    return triples
    deps._expansion["abc_triples"] = triples
    return triples


def _bridge_gap_seed(deps: QueryDeps, a: int, b: int, c: int, lane: str, score: float) -> Seed | None:
    ab = _edge_relation_id(deps, a, b)
    bc = _edge_relation_id(deps, b, c)
    if ab is None or bc is None:
        return None
    legs = [("relation", ab), ("relation", bc)]
    na, nb, nc = _name(deps, a), _name(deps, b), _name(deps, c)
    if lane == "bridge":
        rationale = (
            f"{na} and {nc} both connect to {nb} but have no direct link and are "
            f"distant in concept space — a possible non-obvious bridge."
        )
    else:
        rationale = (
            f"{na} and {nc} share neighbour {nb} and are conceptually close yet "
            f"unlinked — an expected-but-missing connection."
        )
    return Seed(
        lane=lane,
        title=f"{na} ⟷ {nc} (via {nb})",
        rationale=rationale,
        entity_ids=[a, b, c],
        leg_fact_ids=legs,
        score=score,
        payload={"intermediate": nb},
    )


def bridge_seeds(deps: QueryDeps, *, limit: int = 12) -> list[Seed]:
    triples = _abc_triples(deps)
    triples.sort(key=lambda t: t[3])  # ascending cosine = most distant first
    out: list[Seed] = []
    for a, b, c, cos, _same in triples:
        seed = _bridge_gap_seed(deps, a, b, c, "bridge", score=1.0 - cos)
        if seed:
            out.append(seed)
        if len(out) >= limit:
            break
    return out


def gap_seeds(deps: QueryDeps, *, limit: int = 12) -> list[Seed]:
    triples = _abc_triples(deps)
    # prefer same-community first, then closest in embedding space
    triples.sort(key=lambda t: (not t[4], -t[3]))
    out: list[Seed] = []
    for a, b, c, cos, same in triples:
        seed = _bridge_gap_seed(deps, a, b, c, "gap", score=(cos + (0.5 if same else 0.0)))
        if seed:
            out.append(seed)
        if len(out) >= limit:
            break
    return out


# ── property-driven ──────────────────────────────────────────────────────────


def property_seeds(deps: QueryDeps, *, limit: int = 12, min_group: int = 4) -> list[Seed]:
    """Property-driven seeds, comparing only COMPARABLE values:

    - **outliers** are computed **within `(property, unit, entity_type)`** — so a
      hydro dam's capacity isn't pitted against an electrolyser's, nor a heat-pump
      COP against a % efficiency. A group that is mostly **blank-condition** is
      down-ranked + flagged (`low_condition_coverage`): the missing system boundary
      (stack vs system, regional vs global) means the comparison may mix scopes — a
      spurious-outlier smell the per-leg verifier can't catch, since each value is
      individually faithful.
    - **cross-type shared values** stay type-CROSSING by design (different materials
      with the same value ⟹ a possible shared mechanism)."""

    groups = deps.conn.execute(
        """
        SELECT property, unit, COUNT(*) AS n
        FROM kg_entity_properties
        WHERE value_num IS NOT NULL
        GROUP BY property, unit
        HAVING n >= ?
        ORDER BY n DESC
        LIMIT 40
        """,
        (min_group,),
    ).fetchall()
    out: list[Seed] = []
    for grp in groups:
        prop, unit = grp["property"], grp["unit"]
        rows = deps.conn.execute(
            """
            SELECT p.id, p.entity_id, e.canonical_name AS name, e.entity_type AS etype,
                   p.value_num, p.value_raw, p.unit, p.condition
            FROM kg_entity_properties p JOIN kg_entities e ON e.id = p.entity_id
            WHERE p.property = ? AND p.unit = ? AND p.value_num IS NOT NULL
            """,
            (prop, unit),
        ).fetchall()
        if len(rows) < min_group:
            continue

        # ── outliers WITHIN entity_type (comparable values only) ──
        by_type: dict[str, list] = {}
        for r in rows:
            by_type.setdefault(r["etype"] or "(untyped)", []).append(r)
        for etype, trows in by_type.items():
            if len(trows) < min_group:
                continue
            tvals = [r["value_num"] for r in trows]
            try:
                spread = statistics.pstdev(tvals)
            except statistics.StatisticsError:
                spread = 0.0
            if spread <= 0:
                continue
            med = statistics.median(tvals)
            outlier = max(trows, key=lambda r: abs(r["value_num"] - med))
            z = abs(outlier["value_num"] - med) / spread
            if z < 1.5:
                continue
            # mostly-unconditioned group with a high-z outlier ⇒ likely scope-mix
            blank = sum(1 for r in trows if not (r["condition"] or "").strip())
            low_cond = blank / len(trows) > 0.7
            type_word = "" if etype == "(untyped)" else f" {etype}"
            type_label = "" if etype == "(untyped)" else f" among {etype}s"
            caveat = " — low condition coverage; the comparison may mix scopes" if low_cond else ""
            out.append(
                Seed(
                    lane="property",
                    title=f"{outlier['name']} is an outlier in {prop}{type_label}",
                    rationale=(
                        f"{outlier['name']} has {prop}={outlier['value_num']:g} {unit} vs a median "
                        f"{med:g} {unit} across {len(tvals)}{type_word} entities (z≈{z:.1f}){caveat}."
                    ),
                    entity_ids=[outlier["entity_id"]],
                    leg_fact_ids=[("attribute", outlier["id"])],
                    score=min(1.0, z / 4.0) * (0.5 if low_cond else 1.0),
                    payload={"property": prop, "unit": unit, "entity_type": etype,
                             "median": med, "z": z, "low_condition_coverage": low_cond},
                )
            )

        # ── cross-type shared value: DIFFERENT types, close value (type-crossing) ──
        typed = [r for r in rows if r["etype"]]
        for i in range(len(typed)):
            done = False
            for j in range(i + 1, len(typed)):
                ri, rj = typed[i], typed[j]
                if ri["etype"] == rj["etype"]:
                    continue
                denom = max(abs(ri["value_num"]), abs(rj["value_num"]), 1e-9)
                if abs(ri["value_num"] - rj["value_num"]) / denom <= 0.1:
                    out.append(
                        Seed(
                            lane="property",
                            title=f"{ri['name']} ≈ {rj['name']} in {prop}",
                            rationale=(
                                f"{ri['name']} ({ri['etype']}) and {rj['name']} ({rj['etype']}) — "
                                f"different kinds — share {prop}≈{ri['value_num']:g} {unit}; "
                                f"a shared mechanism may underlie it."
                            ),
                            entity_ids=[ri["entity_id"], rj["entity_id"]],
                            leg_fact_ids=[("attribute", ri["id"]), ("attribute", rj["id"])],
                            score=0.6,
                            payload={"property": prop, "unit": unit},
                        )
                    )
                    done = True
                    break
            if done:
                break
        if len(out) >= limit * 3:
            break

    # diversity: cap to 2 seeds per property, then global score sort/limit
    out.sort(key=lambda s: s.score, reverse=True)
    capped: list[Seed] = []
    seen_prop: dict[str, int] = {}
    for s in out:
        p = s.payload.get("property", "")
        if seen_prop.get(p, 0) >= 2:
            continue
        seen_prop[p] = seen_prop.get(p, 0) + 1
        capped.append(s)
        if len(capped) >= limit:
            break
    return capped


# ── attribute-contested / reconcile ──────────────────────────────────────────


def attribute_contested_seeds(deps: QueryDeps, *, limit: int = 12) -> list[Seed]:
    rows = deps.conn.execute(
        """
        SELECT entity_id, property,
               COUNT(DISTINCT value_num) AS distinct_vals,
               COUNT(DISTINCT condition) AS distinct_conds
        FROM kg_entity_properties
        WHERE value_num IS NOT NULL
        GROUP BY entity_id, property
        HAVING distinct_vals >= 2 AND distinct_conds >= 2
        ORDER BY distinct_vals DESC
        LIMIT ?
        """,
        (limit * 2,),
    ).fetchall()
    out: list[Seed] = []
    for r in rows:
        detail = deps.conn.execute(
            """
            SELECT p.id, e.canonical_name AS name, p.value_raw, p.unit, p.condition, p.value_num
            FROM kg_entity_properties p JOIN kg_entities e ON e.id = p.entity_id
            WHERE p.entity_id = ? AND p.property = ? AND p.value_num IS NOT NULL
            ORDER BY p.value_num
            """,
            (r["entity_id"], r["property"]),
        ).fetchall()
        # keep distinct (value, condition) pairs
        seen: set[tuple] = set()
        legs: list[tuple[str, int]] = []
        variants: list[str] = []
        for d in detail:
            key = (d["value_num"], d["condition"])
            if key in seen:
                continue
            seen.add(key)
            legs.append(("attribute", d["id"]))
            cond = f" [{d['condition']}]" if d["condition"] else ""
            variants.append(f"{d['value_raw']} {d['unit']}{cond}".strip())
        if len(legs) < 2:
            continue
        name = detail[0]["name"]
        out.append(
            Seed(
                lane="attribute_contested",
                title=f"{name}: {r['property']} varies by condition",
                rationale=(
                    f"{name} reports {r['property']} as {', '.join(variants[:4])} — divergent "
                    f"values under different conditions; reconcilable or genuinely contested?"
                ),
                entity_ids=[r["entity_id"]],
                leg_fact_ids=legs[:6],
                score=min(1.0, 0.4 + 0.15 * len(legs)),
                payload={"property": r["property"], "variants": variants},
            )
        )
        if len(out) >= limit:
            break
    return out


# ── relation-contested ───────────────────────────────────────────────────────


def relation_contested_seeds(deps: QueryDeps, *, limit: int = 8) -> list[Seed]:
    rows = deps.conn.execute(
        """
        SELECT subject_entity_id AS s, object_entity_id AS o,
               COUNT(DISTINCT predicate) AS preds
        FROM kg_relations
        GROUP BY subject_entity_id, object_entity_id
        HAVING preds >= 2
        ORDER BY preds DESC
        LIMIT ?
        """,
        (limit * 2,),
    ).fetchall()
    out: list[Seed] = []
    for r in rows:
        legs_rows = deps.conn.execute(
            """
            SELECT predicate, MIN(id) AS fact_id FROM kg_relations
            WHERE subject_entity_id = ? AND object_entity_id = ?
            GROUP BY predicate
            """,
            (r["s"], r["o"]),
        ).fetchall()
        preds = [lr["predicate"] for lr in legs_rows]
        legs = [("relation", lr["fact_id"]) for lr in legs_rows]
        ns, no = _name(deps, r["s"]), _name(deps, r["o"])
        out.append(
            Seed(
                lane="relation_contested",
                title=f"{ns} → {no}: {len(preds)} predicates",
                rationale=(
                    f"{ns} relates to {no} via {', '.join(preds[:5])} — possibly conflicting "
                    f"or complementary; flagged low-confidence."
                ),
                entity_ids=[r["s"], r["o"]],
                leg_fact_ids=legs[:6],
                score=0.3 + 0.05 * len(preds),
                payload={"predicates": preds},
            )
        )
        if len(out) >= limit:
            break
    return out


# ── combine ──────────────────────────────────────────────────────────────────

_LANES = {
    "bridge": bridge_seeds,
    "gap": gap_seeds,
    "property": property_seeds,
    "attribute_contested": attribute_contested_seeds,
    "relation_contested": relation_contested_seeds,
}


def generate_seeds(
    deps: QueryDeps,
    *,
    max_seeds: int = 40,
    lanes: list[str] | None = None,
    per_lane: int = 12,
) -> list[Seed]:
    """All-lane seeds, round-robin interleaved (so a capped run spans classes),
    then truncated to ``max_seeds``. Within a lane, higher-score first."""

    active = lanes or list(_LANES)
    by_lane: dict[str, list[Seed]] = {}
    for lane in active:
        fn = _LANES.get(lane)
        if fn is None:
            continue
        seeds = fn(deps, limit=per_lane)
        seeds.sort(key=lambda s: s.score, reverse=True)
        by_lane[lane] = seeds

    interleaved: list[Seed] = []
    idx = 0
    while len(interleaved) < max_seeds and any(idx < len(v) for v in by_lane.values()):
        for lane in active:
            seeds = by_lane.get(lane, [])
            if idx < len(seeds):
                interleaved.append(seeds[idx])
                if len(interleaved) >= max_seeds:
                    break
        idx += 1
    return interleaved
