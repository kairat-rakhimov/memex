"""The shared retrieval toolbox — three modalities over the read-only KG.

- **symbolic** — SQL lookups (entity by name, an entity's facts, facts by
  expanded predicate / property). Exact + LIKE + alias-aware.
- **semantic** — cosine of a query VECTOR against the per-fact relation /
  attribute embedding matrices. The vector is supplied by the caller (the
  ``@agent.tool`` wrapper embeds the query text), so this stays pure numpy.
- **topological** — NetworkX over the cached graph: neighbours, common
  neighbours, shortest paths, communities.

Every result dataclass carries the ``(kind, fact_id)`` handle the
:mod:`memex.kg_query.grounding` ladder turns into quotable evidence — agents cite
ids the toolbox hands them, never invented ones. Tools cap at small ``limit``s
(compact, id-bearing rows) to keep the ReAct context bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import format_value


@dataclass(frozen=True)
class EntityHit:
    entity_id: int
    name: str
    entity_type: str | None
    mentions: int
    degree: int


@dataclass(frozen=True)
class FactHit:
    """One groundable fact. ``kind`` ∈ {relation, attribute}; ``fact_id`` is a
    representative row id (provenance collapsed); ``corroboration`` = distinct
    source docs; ``score`` is the semantic cosine when applicable, else None."""

    kind: str
    fact_id: int
    subject: str
    predicate: str
    object: str
    confidence: float | None
    corroboration: int
    score: float | None = None

    def text(self) -> str:
        return f"{self.subject} —[{self.predicate}]→ {self.object}"


@dataclass(frozen=True)
class PathHit:
    entity_ids: list[int]
    names: list[str]
    predicates: list[str]


# ── helpers ────────────────────────────────────────────────────────────────


def align_query(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Match a query vector to the stored matrix dimension. A slim-exported db
    stores Matryoshka-truncated embeddings (e.g. 1024-dim) while a freshly embedded
    query vector is full (3072-dim) — truncate + renormalize so the dot product is
    still cosine. No-op when dims already match."""

    if matrix.size == 0 or vec.size == 0:
        return vec
    d = matrix.shape[1]
    if vec.shape[0] == d:
        return vec
    v = vec[:d] if vec.shape[0] > d else vec
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _top_indices(scores: np.ndarray, k: int) -> list[int]:
    if scores.size == 0:
        return []
    if scores.size <= k:
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, k)[:k]
    return list(idx[np.argsort(-scores[idx])])


def _relation_hit(deps: QueryDeps, relation_id: int, score: float | None) -> FactHit | None:
    row = deps.conn.execute(
        """
        SELECT r.id, r.subject_entity_id, r.predicate, r.object_entity_id,
               s.canonical_name AS subj, o.canonical_name AS obj,
               r.mapper_confidence
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
        "SELECT COUNT(DISTINCT source_doc_id) AS n FROM kg_relations "
        "WHERE subject_entity_id=? AND predicate=? AND object_entity_id=?",
        (row["subject_entity_id"], row["predicate"], row["object_entity_id"]),
    ).fetchone()
    return FactHit(
        kind="relation",
        fact_id=row["id"],
        subject=row["subj"],
        predicate=row["predicate"],
        object=row["obj"],
        confidence=row["mapper_confidence"],
        corroboration=int(corr["n"]) if corr else 1,
        score=score,
    )


def _attribute_hit(deps: QueryDeps, attribute_id: int, score: float | None) -> FactHit | None:
    row = deps.conn.execute(
        """
        SELECT p.id, p.entity_id, e.canonical_name AS subj, p.property,
               p.value_raw, p.unit, p.condition, p.mapper_confidence
        FROM kg_entity_properties p
        JOIN kg_entities e ON e.id = p.entity_id
        WHERE p.id = ?
        """,
        (attribute_id,),
    ).fetchone()
    if row is None:
        return None
    corr = deps.conn.execute(
        "SELECT COUNT(DISTINCT source_doc_id) AS n FROM kg_entity_properties "
        "WHERE entity_id=? AND property=? AND value_raw=? AND condition=?",
        (row["entity_id"], row["property"], row["value_raw"], row["condition"]),
    ).fetchone()
    value = format_value(row["value_raw"], row["unit"], row["condition"])
    return FactHit(
        kind="attribute",
        fact_id=row["id"],
        subject=row["subj"],
        predicate=row["property"],
        object=value,
        confidence=row["mapper_confidence"],
        corroboration=int(corr["n"]) if corr else 1,
        score=score,
    )


# ── symbolic ────────────────────────────────────────────────────────────────


def find_entities(deps: QueryDeps, name: str, *, limit: int = 8) -> list[EntityHit]:
    """Entities matching ``name`` — exact (NOCASE), then alias, then substring.
    Ranked by mentions. The Retriever's entry point into the graph."""

    name = (name or "").strip()
    if not name:
        return []
    seen: set[int] = set()
    out: list[EntityHit] = []

    def _add(rows) -> None:
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append(
                EntityHit(
                    entity_id=r["id"],
                    name=r["canonical_name"],
                    entity_type=r["entity_type"],
                    mentions=r["mentions"],
                    degree=r["degree"],
                )
            )

    _add(
        deps.conn.execute(
            "SELECT id, canonical_name, entity_type, mentions, degree FROM kg_entities "
            "WHERE canonical_name = ? COLLATE NOCASE ORDER BY mentions DESC LIMIT ?",
            (name, limit),
        )
    )
    if len(out) < limit:
        _add(
            deps.conn.execute(
                """
                SELECT ke.id, ke.canonical_name, ke.entity_type, ke.mentions, ke.degree
                FROM entity_aliases ea JOIN kg_entities ke ON ke.id = ea.entity_id
                WHERE ea.alias_text = ? COLLATE NOCASE
                ORDER BY ke.mentions DESC LIMIT ?
                """,
                (name, limit),
            )
        )
    if len(out) < limit:
        _add(
            deps.conn.execute(
                "SELECT id, canonical_name, entity_type, mentions, degree FROM kg_entities "
                "WHERE canonical_name LIKE ? COLLATE NOCASE ORDER BY mentions DESC LIMIT ?",
                (f"%{name}%", limit),
            )
        )
    return out[:limit]


def entity_facts(deps: QueryDeps, entity_id: int, *, limit: int = 10) -> list[FactHit]:
    """An entity's facts — outgoing + incoming relations + attributes, provenance
    collapsed, ranked by corroboration. Representative ``fact_id`` per distinct
    fact."""

    rows = deps.conn.execute(
        """
        SELECT 'relation' AS kind, MIN(r.id) AS fact_id,
               s.canonical_name AS subj, r.predicate AS pred, o.canonical_name AS obj,
               AVG(r.mapper_confidence) AS conf,
               COUNT(DISTINCT r.source_doc_id) AS corr
        FROM kg_relations r
        JOIN kg_entities s ON s.id = r.subject_entity_id
        JOIN kg_entities o ON o.id = r.object_entity_id
        WHERE r.subject_entity_id = ? OR r.object_entity_id = ?
        GROUP BY r.subject_entity_id, r.predicate, r.object_entity_id
        ORDER BY corr DESC, conf DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    out = [
        FactHit(
            kind="relation",
            fact_id=r["fact_id"],
            subject=r["subj"],
            predicate=r["pred"],
            object=r["obj"],
            confidence=r["conf"],
            corroboration=int(r["corr"]),
        )
        for r in rows
    ]
    attr_rows = deps.conn.execute(
        """
        SELECT MIN(p.id) AS fact_id, e.canonical_name AS subj, p.property AS prop,
               p.value_raw, p.unit, p.condition,
               AVG(p.mapper_confidence) AS conf,
               COUNT(DISTINCT p.source_doc_id) AS corr
        FROM kg_entity_properties p JOIN kg_entities e ON e.id = p.entity_id
        WHERE p.entity_id = ?
        GROUP BY p.property, p.value_raw, p.condition
        ORDER BY corr DESC, conf DESC
        LIMIT ?
        """,
        (entity_id, limit),
    ).fetchall()
    for r in attr_rows:
        value = format_value(r["value_raw"], r["unit"], r["condition"])
        out.append(
            FactHit(
                kind="attribute",
                fact_id=r["fact_id"],
                subject=r["subj"],
                predicate=r["prop"],
                object=value,
                confidence=r["conf"],
                corroboration=int(r["corr"]),
            )
        )
    out.sort(key=lambda h: h.corroboration, reverse=True)
    return out[:limit]


def relations_with_predicate(
    deps: QueryDeps, predicates: list[str], *, limit: int = 10
) -> list[FactHit]:
    """Relations whose predicate is in the (already-expanded) ``predicates`` set,
    provenance collapsed, ranked by corroboration."""

    if not predicates:
        return []
    ph = ",".join("?" * len(predicates))
    rows = deps.conn.execute(
        f"""
        SELECT MIN(r.id) AS fact_id, s.canonical_name AS subj, r.predicate AS pred,
               o.canonical_name AS obj, AVG(r.mapper_confidence) AS conf,
               COUNT(DISTINCT r.source_doc_id) AS corr
        FROM kg_relations r
        JOIN kg_entities s ON s.id = r.subject_entity_id
        JOIN kg_entities o ON o.id = r.object_entity_id
        WHERE r.predicate IN ({ph}) COLLATE NOCASE
        GROUP BY r.subject_entity_id, r.predicate, r.object_entity_id
        ORDER BY corr DESC, conf DESC
        LIMIT ?
        """,
        (*predicates, limit),
    ).fetchall()
    return [
        FactHit(
            kind="relation",
            fact_id=r["fact_id"],
            subject=r["subj"],
            predicate=r["pred"],
            object=r["obj"],
            confidence=r["conf"],
            corroboration=int(r["corr"]),
        )
        for r in rows
    ]


def attributes_with_property(
    deps: QueryDeps, properties: list[str], *, limit: int = 10
) -> list[FactHit]:
    """Attributes whose property is in the (expanded) ``properties`` set."""

    if not properties:
        return []
    ph = ",".join("?" * len(properties))
    rows = deps.conn.execute(
        f"""
        SELECT MIN(p.id) AS fact_id, e.canonical_name AS subj, p.property AS prop,
               p.value_raw, p.unit, p.condition,
               AVG(p.mapper_confidence) AS conf,
               COUNT(DISTINCT p.source_doc_id) AS corr
        FROM kg_entity_properties p JOIN kg_entities e ON e.id = p.entity_id
        WHERE p.property IN ({ph}) COLLATE NOCASE
        GROUP BY p.entity_id, p.property, p.value_raw, p.condition
        ORDER BY corr DESC, conf DESC
        LIMIT ?
        """,
        (*properties, limit),
    ).fetchall()
    out = []
    for r in rows:
        value = format_value(r["value_raw"], r["unit"], r["condition"])
        out.append(
            FactHit(
                kind="attribute",
                fact_id=r["fact_id"],
                subject=r["subj"],
                predicate=r["prop"],
                object=value,
                confidence=r["conf"],
                corroboration=int(r["corr"]),
            )
        )
    return out


# ── semantic ────────────────────────────────────────────────────────────────


def semantic_facts(
    deps: QueryDeps,
    query_vec: np.ndarray,
    *,
    kinds: tuple[str, ...] = ("relation", "attribute"),
    k: int = 10,
) -> list[FactHit]:
    """Top-``k`` facts by cosine of ``query_vec`` against the per-fact embedding
    matrices, deduped by display text (keeping the best-scoring row). Immune to
    predicate sprawl — synonym-labelled facts cluster by meaning."""

    if query_vec.size == 0:
        return []
    pool: list[FactHit] = []
    for kind in kinds:
        ids, matrix = deps.matrix(kind)
        if matrix.size == 0:
            continue
        scores = matrix @ align_query(query_vec, matrix)
        for idx in _top_indices(scores, k * 3):
            fact_id = int(ids[idx])
            score = float(scores[idx])
            hit = (
                _relation_hit(deps, fact_id, score)
                if kind == "relation"
                else _attribute_hit(deps, fact_id, score)
            )
            if hit is not None:
                pool.append(hit)
    best: dict[str, FactHit] = {}
    for h in pool:
        key = f"{h.kind}|{h.subject}|{h.predicate}|{h.object}".lower()
        if key not in best or (h.score or 0) > (best[key].score or 0):
            best[key] = h
    ranked = sorted(best.values(), key=lambda h: h.score or 0.0, reverse=True)
    return ranked[:k]


def semantic_entities(
    deps: QueryDeps, query_vec: np.ndarray, *, k: int = 10
) -> list[EntityHit]:
    """Top-``k`` entities by cosine of ``query_vec`` against the entity embedding
    matrix — find entities close in MEANING (analogy / "what else is like this"),
    unlike :func:`find_entities` which matches names. Powers cross-entity landscape
    + bridge exploration."""

    if query_vec.size == 0:
        return []
    ids, matrix = deps.matrix("entity")
    if matrix.size == 0:
        return []
    scores = matrix @ align_query(query_vec, matrix)
    out: list[EntityHit] = []
    for idx in _top_indices(scores, k):
        hit = _entity_hit(deps, int(ids[idx]))
        if hit is not None:
            out.append(hit)
    return out


# ── topological ─────────────────────────────────────────────────────────────


def _entity_hit(deps: QueryDeps, entity_id: int) -> EntityHit | None:
    g = deps.graph()
    if entity_id not in g:
        return None
    d = g.nodes[entity_id]
    return EntityHit(
        entity_id=entity_id,
        name=d.get("name", str(entity_id)),
        entity_type=d.get("entity_type"),
        mentions=d.get("mentions", 0),
        degree=d.get("degree", 0),
    )


def neighbors(deps: QueryDeps, entity_id: int, *, limit: int = 12) -> list[EntityHit]:
    und = deps.undirected()
    if entity_id not in und:
        return []
    nbrs = sorted(
        und.neighbors(entity_id),
        key=lambda n: und.nodes[n].get("mentions", 0),
        reverse=True,
    )
    out = [_entity_hit(deps, n) for n in nbrs[:limit]]
    return [h for h in out if h is not None]


def common_neighbors(
    deps: QueryDeps, a_id: int, b_id: int, *, limit: int = 12
) -> list[EntityHit]:
    und = deps.undirected()
    if a_id not in und or b_id not in und:
        return []
    common = sorted(
        __import__("networkx").common_neighbors(und, a_id, b_id),
        key=lambda n: und.nodes[n].get("mentions", 0),
        reverse=True,
    )
    out = [_entity_hit(deps, n) for n in common[:limit]]
    return [h for h in out if h is not None]


def shortest_paths(
    deps: QueryDeps, a_id: int, b_id: int, *, cutoff: int = 3, limit: int = 3
) -> list[PathHit]:
    """Up to ``limit`` simple paths (≤ ``cutoff`` hops) between two entities on
    the undirected view, each annotated with the predicate(s) on every hop."""

    import networkx as nx

    und = deps.undirected()
    if a_id not in und or b_id not in und or a_id == b_id:
        return []
    try:
        gen = nx.all_simple_paths(und, a_id, b_id, cutoff=cutoff)
        paths = []
        for p in gen:
            paths.append(p)
            if len(paths) >= limit:
                break
    except nx.NetworkXNoPath:
        return []
    g = deps.graph()
    out: list[PathHit] = []
    for path in paths:
        names = [und.nodes[n].get("name", str(n)) for n in path]
        preds: list[str] = []
        for u, v in zip(path, path[1:], strict=False):
            label = _edge_predicate(g, u, v)
            preds.append(label)
        out.append(PathHit(entity_ids=path, names=names, predicates=preds))
    return out


def _edge_predicate(g, u: int, v: int) -> str:
    if g.has_edge(u, v):
        data = g.get_edge_data(u, v)
    elif g.has_edge(v, u):
        data = g.get_edge_data(v, u)
    else:
        return "related_to"
    keys = list(data.keys())
    return str(keys[0]) if keys else "related_to"


def communities(deps: QueryDeps, *, min_size: int = 3) -> list[list[int]]:
    """Greedy-modularity communities over the undirected view (cached). Each
    is a list of entity ids; communities smaller than ``min_size`` dropped."""

    cache_key = "communities"
    cached = deps._expansion.get(cache_key)
    if cached is None:
        from networkx.algorithms.community import greedy_modularity_communities

        und = deps.undirected()
        if und.number_of_edges() == 0:
            cached = []
        else:
            raw = greedy_modularity_communities(und)
            cached = [sorted(int(n) for n in c) for c in raw]
        deps._expansion[cache_key] = cached
    return [c for c in cached if len(c) >= min_size]  # type: ignore[union-attr]
