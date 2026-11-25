"""``QueryDeps`` — the read-only handle the query toolbox + agents share.

Opens a built graph's ``kg.db`` READ-ONLY (the build's canonical store is never
mutated by a query run) and lazily caches the expensive derived structures a run
reuses across many tool calls: per-kind embedding matrices, the NetworkX graph,
and the vocabulary-expansion tables. One instance per chat /
hypothesize run, passed as pydantic_ai ``deps`` and to every pure toolbox
function.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from memex.common.config import Settings
from memex.kg_intelligence.embeddings import blob_to_vec

if TYPE_CHECKING:
    import networkx as nx

_EMBED_KINDS = ("relation", "attribute", "entity", "property", "predicate", "chunk")


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only (URI ``mode=ro``) with a Row factory.

    Writes raise ``sqlite3.OperationalError`` — the graph a query run
    reads is build-time output and must stay untouched.
    """

    # check_same_thread=False: pydantic-ai runs sync @agent.tool functions in a
    # worker thread, so a tool-using agent (Retriever / Explorer) accesses this
    # connection off the creating thread. Reads are sequential within a ReAct loop
    # (one awaited tool at a time) and the db is read-only, so this is safe.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class QueryDeps:
    """Read-only KG access + lazily-built caches for one query run.

    Construct via :meth:`open`. ``conn`` is read-only; the matrix / graph /
    expansion caches are built on first use and reused for the run's lifetime.
    """

    db_path: Path
    settings: Settings
    conn: sqlite3.Connection

    _matrices: dict[str, tuple[list[str], np.ndarray]] = field(default_factory=dict)
    _graph: nx.MultiDiGraph | None = None
    _undirected: nx.Graph | None = None
    _expansion: dict[str, object] = field(default_factory=dict)

    @classmethod
    def open(cls, kg_root: Path, settings: Settings) -> QueryDeps:
        db_path = kg_root / "kg.db"
        if not db_path.exists():
            raise FileNotFoundError(f"no kg.db at {db_path}")
        return cls(db_path=db_path, settings=settings, conn=open_readonly(db_path))

    def close(self) -> None:
        self.conn.close()

    # ── embeddings ────────────────────────────────────────────────────

    def list_embeddings_by_kind(self, kind: str) -> list[tuple[str, bytes]]:
        """Mirror of ``TraceStore.list_embeddings_by_kind`` so the shared
        ``embeddings.load_matrix`` helper works against the read-only conn."""

        rows = self.conn.execute(
            "SELECT target_id, vector FROM embeddings WHERE target_kind = ?",
            (kind,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def matrix(self, kind: str) -> tuple[list[str], np.ndarray]:
        """``(target_ids, (N, dim) float32 matrix)`` for one embedding kind,
        cached. Empty ``([], (0,0))`` when the kind has no rows. LLM
        ``text-embedding-3-large`` returns unit vectors, so ``matrix @ q`` is
        cosine similarity."""

        if kind not in self._matrices:
            rows = self.list_embeddings_by_kind(kind)
            if not rows:
                self._matrices[kind] = ([], np.empty((0, 0), dtype=np.float32))
            else:
                ids = [r[0] for r in rows]
                vecs = np.stack([blob_to_vec(r[1]) for r in rows])
                self._matrices[kind] = (ids, vecs)
        return self._matrices[kind]

    # ── graph ─────────────────────────────────────────────────────────

    def graph(self) -> nx.MultiDiGraph:
        """Directed multigraph over ``kg_relations`` (one edge per
        ``(subj, predicate, obj)``; ``corroboration`` = distinct source docs).
        Nodes carry ``name`` / ``entity_type`` / ``mentions`` / ``degree``.
        Built once per run."""

        if self._graph is None:
            self._build_graph()
        assert self._graph is not None
        return self._graph

    def undirected(self) -> nx.Graph:
        """Simple undirected view (parallel predicates + direction collapsed)
        for connectivity queries — common neighbours, communities, bridge
        detection."""

        if self._undirected is None:
            self._build_graph()
        assert self._undirected is not None
        return self._undirected

    def _build_graph(self) -> None:
        import networkx as nx

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        for r in self.conn.execute(
            "SELECT id, canonical_name, entity_type, mentions, degree FROM kg_entities"
        ):
            g.add_node(
                r["id"],
                name=r["canonical_name"],
                entity_type=r["entity_type"],
                mentions=r["mentions"],
                degree=r["degree"],
            )
        for r in self.conn.execute(
            """
            SELECT subject_entity_id AS s, predicate AS p, object_entity_id AS o,
                   COUNT(DISTINCT source_doc_id) AS corroboration,
                   AVG(mapper_confidence) AS avg_confidence,
                   MIN(id) AS sample_relation_id
            FROM kg_relations
            GROUP BY subject_entity_id, predicate, object_entity_id
            """
        ):
            # Defensive: a relation can reference an entity row that lost its
            # node in a partial db; add_edge would silently create a node
            # without attributes, so skip such dangling edges.
            if r["s"] not in g or r["o"] not in g:
                continue
            g.add_edge(
                r["s"],
                r["o"],
                key=r["p"],
                predicate=r["p"],
                corroboration=int(r["corroboration"]),
                avg_confidence=r["avg_confidence"],
                sample_relation_id=r["sample_relation_id"],
            )
        und: nx.Graph = nx.Graph()
        und.add_nodes_from(g.nodes(data=True))
        und.add_edges_from((u, v) for u, v, _k in g.edges(keys=True))
        self._graph = g
        self._undirected = und

    # ── vocabulary expansion tables (loaded once) ─────────────────────

    def alias_pairs(self, table: str) -> list[tuple[str, str]]:
        """All ``(alias, canonical)`` rows (original casing) for
        ``predicate_aliases`` / ``property_aliases``. Empty when the build skipped
        vocabulary finalization (expansion then degrades to embed-only)."""

        key = f"alias:{table}"
        if key not in self._expansion:
            rows = self.conn.execute(
                f"SELECT alias, canonical FROM {table}"
            ).fetchall()
            self._expansion[key] = [(r["alias"], r["canonical"]) for r in rows]
        return self._expansion[key]  # type: ignore[return-value]

    def children_of(self, kind: str) -> dict[str, list[str]]:
        """``{parent_lower: [child, ...]}`` from ``schema_parents`` for ``kind``
        ('relation_type' | 'property_type') — the rollup-expansion substrate."""

        key = f"children:{kind}"
        if key not in self._expansion:
            mapping: dict[str, list[str]] = {}
            for r in self.conn.execute(
                "SELECT child, parent FROM schema_parents WHERE kind = ?", (kind,)
            ):
                mapping.setdefault(r["parent"].lower(), []).append(r["child"])
            self._expansion[key] = mapping
        return self._expansion[key]  # type: ignore[return-value]

    def schema_names(self, kind: str) -> list[str]:
        """Canonical names in ``kg_schema`` for ``kind`` ('relation_type' |
        'property_type' | 'entity_type')."""

        rows = self.conn.execute(
            "SELECT name FROM kg_schema WHERE kind = ? ORDER BY id ASC", (kind,)
        ).fetchall()
        return [r["name"] for r in rows]
