"""Side-by-side KG quality diff for fixed-corpus A/B comparison.

    python experiments/eval_corpus_compare.py <baseline_kg.db> <new_kg.db>

Read-only. Prints baseline vs new on the metrics that distinguish a
typed-statements pipeline (relations + attributes) from an older one: counts, measurement-leaf
rate, relation purity, topology, generic-type rate, near-dupe predicate
families, embeddings coverage. The new db may carry a ``kg_entity_properties``
table the baseline lacks — every such read degrades gracefully.
"""

from __future__ import annotations

import re
import sqlite3
import sys

# entity_types treated as literal-valued leaves (measurements etc.).
_MEASUREMENT_TYPES = ("measurement", "experimental_condition", "economic_quantity")
# Generic catch-all entity types the typing pass aims to eliminate.
_GENERIC_TYPES = ("concept", "thing", "entity", "other")
# A name that is essentially "<number><unit>" — a literal, not an entity.
_NUMBER_UNIT = re.compile(r"^\s*[<>~±]?\s*\d[\d,\.]*\s*[%°a-zA-Zµ/·\-]*\s*$")
# Predicate-family normalization: strip common form-leak prefixes.
_PRED_PREFIX = re.compile(r"^(has_|have_|shows?_|is_|are_|was_|were_|been_)")


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _one(conn: sqlite3.Connection, q: str, *a) -> object:
    row = conn.execute(q, a).fetchone()
    return row[0] if row else None


def _rows(conn: sqlite3.Connection, q: str, *a) -> list[sqlite3.Row]:
    return conn.execute(q, a).fetchall()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return _one(
        conn, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", name
    ) is not None


def _cluster_lane(
    conn: sqlite3.Connection,
    *,
    alias_table: str,
    parent_kind: str,
    edge_table: str,
    edge_col: str,
) -> dict | None:
    """Vocabulary cluster-structure metrics for one lane (predicate / property).

    Returns None when neither the alias table nor schema_parents exists (an
    older baseline). ``covered`` = distinct edge vocab terms that
    participate in any alias or parent edge — the navigability the
    hypothesis-layer expansion walks."""

    has_alias = _has_table(conn, alias_table)
    has_parents = _has_table(conn, "schema_parents")
    if not has_alias and not has_parents:
        return None

    alias_rows = _rows(conn, f"SELECT alias, canonical FROM {alias_table}") if has_alias else []
    parent_rows = (
        _rows(conn, "SELECT child, parent FROM schema_parents WHERE kind = ?", parent_kind)
        if has_parents else []
    )
    clusters: dict[str, set[str]] = {}
    touched: set[str] = set()
    for r in alias_rows:
        clusters.setdefault(r["canonical"], set()).add(r["alias"])
        touched.add(r["alias"].lower())
        touched.add(r["canonical"].lower())
    for r in parent_rows:
        touched.add(r["child"].lower())
        touched.add(r["parent"].lower())
    sizes = [len(v) for v in clusters.values()]

    distinct_edges = (
        {(row[edge_col] or "").lower() for row in _rows(conn, f"SELECT DISTINCT {edge_col} FROM {edge_table}")}
        if _has_table(conn, edge_table) else set()
    )
    return {
        "alias_rows": len(alias_rows),
        "parent_rows": len(parent_rows),
        "alias_clusters": len(clusters),
        "alias_max": max(sizes) if sizes else 0,
        "distinct_edges": len(distinct_edges),
        "covered": len(touched & distinct_edges),
    }


def _pct(num: int | None, den: int | None) -> str:
    num = num or 0
    den = den or 0
    return f"{num}/{den} ({(num / den * 100 if den else 0):.1f}%)"


# ── per-db metric collection ──────────────────────────────────────────────


def collect(conn: sqlite3.Connection) -> dict:
    m: dict[str, object] = {}

    m["kg_entities"] = _one(conn, "SELECT COUNT(*) FROM kg_entities") or 0
    m["kg_relations"] = _one(conn, "SELECT COUNT(*) FROM kg_relations") or 0
    m["distinct_predicates"] = (
        _one(conn, "SELECT COUNT(DISTINCT predicate) FROM kg_relations") or 0
    )
    m["kg_schema"] = _one(conn, "SELECT COUNT(*) FROM kg_schema") or 0
    m["kg_entity_properties"] = (
        _one(conn, "SELECT COUNT(*) FROM kg_entity_properties")
        if _has_table(conn, "kg_entity_properties")
        else None
    )
    if m["kg_entity_properties"]:
        m["attr_distinct"] = _one(conn, "SELECT COUNT(DISTINCT property) FROM kg_entity_properties") or 0
        m["attr_value_num"] = _one(conn, "SELECT COUNT(*) FROM kg_entity_properties WHERE value_num IS NOT NULL") or 0
        m["attr_condition"] = _one(conn, "SELECT COUNT(*) FROM kg_entity_properties WHERE condition <> ''") or 0
        m["attr_mapped"] = _one(conn, "SELECT COUNT(*) FROM kg_entity_properties WHERE mapped_flag = 1") or 0
    else:
        m["attr_distinct"] = m["attr_value_num"] = m["attr_condition"] = m["attr_mapped"] = None

    rel = m["kg_relations"]
    ent = m["kg_entities"]

    # ── measurement-leaf-node rate (object is a literal-valued leaf) ──
    placeholders = ",".join("?" * len(_MEASUREMENT_TYPES))
    m["measure_leaf_by_type"] = _one(
        conn,
        f"""SELECT COUNT(*) FROM kg_relations r
            JOIN kg_entities o ON o.id = r.object_entity_id
            WHERE LOWER(o.entity_type) IN ({placeholders})""",
        *_MEASUREMENT_TYPES,
    ) or 0
    obj_names = _rows(
        conn,
        """SELECT o.canonical_name AS name FROM kg_relations r
           JOIN kg_entities o ON o.id = r.object_entity_id""",
    )
    m["measure_leaf_by_regex"] = sum(
        1 for row in obj_names if _NUMBER_UNIT.match(row["name"] or "")
    )
    m["_rel_for_leaf"] = len(obj_names)

    # ── relation purity: both endpoints non-literal ──
    literal_ids = {
        row["id"]
        for row in _rows(
            conn,
            f"""SELECT id FROM kg_entities
                WHERE LOWER(entity_type) IN ({placeholders})""",
            *_MEASUREMENT_TYPES,
        )
    }
    literal_ids |= {
        row["id"]
        for row in _rows(conn, "SELECT id, canonical_name FROM kg_entities")
        if _NUMBER_UNIT.match(row["canonical_name"] or "")
    }
    rel_endpoints = _rows(
        conn, "SELECT subject_entity_id AS s, object_entity_id AS o FROM kg_relations"
    )
    m["pure_relations"] = sum(
        1 for r in rel_endpoints if r["s"] not in literal_ids and r["o"] not in literal_ids
    )

    # ── topology ──
    m["orphan"] = _one(conn, "SELECT COUNT(*) FROM kg_entities WHERE degree = 0") or 0
    m["leaf"] = _one(conn, "SELECT COUNT(*) FROM kg_entities WHERE degree = 1") or 0

    # ── generic entity-type rate ──
    gen_ph = ",".join("?" * len(_GENERIC_TYPES))
    m["generic"] = _one(
        conn,
        f"SELECT COUNT(*) FROM kg_entities WHERE LOWER(entity_type) IN ({gen_ph})",
        *_GENERIC_TYPES,
    ) or 0

    # ── near-dupe predicate families ──
    fams: dict[str, set[str]] = {}
    for row in _rows(
        conn, "SELECT DISTINCT predicate FROM kg_relations WHERE predicate IS NOT NULL"
    ):
        pred = row["predicate"]
        key = _PRED_PREFIX.sub("", pred.lower())
        fams.setdefault(key, set()).add(pred)
    m["pred_families"] = {k: sorted(v) for k, v in fams.items() if len(v) > 1}

    # ── embeddings present by target_kind ──
    m["embeddings_by_kind"] = {
        row["target_kind"]: row["n"]
        for row in _rows(
            conn,
            "SELECT target_kind, COUNT(*) AS n FROM embeddings GROUP BY target_kind",
        )
    } if _has_table(conn, "embeddings") else {}

    # ── vocabulary cluster structure (VocabResolver) ──
    m["pred_cluster"] = _cluster_lane(
        conn, alias_table="predicate_aliases", parent_kind="relation_type",
        edge_table="kg_relations", edge_col="predicate",
    )
    m["prop_cluster"] = _cluster_lane(
        conn, alias_table="property_aliases", parent_kind="property_type",
        edge_table="kg_entity_properties", edge_col="property",
    )

    m["_ent"] = ent
    m["_rel"] = rel
    return m


# ── rendering ─────────────────────────────────────────────────────────────


def _row(label: str, base: str, new: str) -> None:
    print(f"  {label:<34} {base:>22} {new:>22}")


def render(base: dict, new: dict, base_path: str, new_path: str) -> None:
    print(f"\n===== baseline: {base_path}")
    print(f"===== new:      {new_path}\n")
    print(f"  {'metric':<34} {'baseline':>22} {'new':>22}")
    print("  " + "-" * 80)

    _row("kg_entities", str(base["kg_entities"]), str(new["kg_entities"]))
    _row("kg_relations", str(base["kg_relations"]), str(new["kg_relations"]))
    _row("distinct predicates", str(base["distinct_predicates"]), str(new["distinct_predicates"]))
    _row("kg_schema", str(base["kg_schema"]), str(new["kg_schema"]))
    _row(
        "kg_entity_properties",
        "—" if base["kg_entity_properties"] is None else str(base["kg_entity_properties"]),
        "—" if new["kg_entity_properties"] is None else str(new["kg_entity_properties"]),
    )

    print("\n  -- measurement-leaf-node rate (object is a literal) --")
    _row(
        "by entity_type",
        _pct(base["measure_leaf_by_type"], base["_rel_for_leaf"]),
        _pct(new["measure_leaf_by_type"], new["_rel_for_leaf"]),
    )
    _row(
        "by number+unit regex",
        _pct(base["measure_leaf_by_regex"], base["_rel_for_leaf"]),
        _pct(new["measure_leaf_by_regex"], new["_rel_for_leaf"]),
    )

    print("\n  -- relation purity (both endpoints non-literal) --")
    _row(
        "pure relations",
        _pct(base["pure_relations"], base["_rel"]),
        _pct(new["pure_relations"], new["_rel"]),
    )

    print("\n  -- topology --")
    _row("orphan rate (degree=0)", _pct(base["orphan"], base["_ent"]), _pct(new["orphan"], new["_ent"]))
    _row("leaf rate (degree=1)", _pct(base["leaf"], base["_ent"]), _pct(new["leaf"], new["_ent"]))

    print("\n  -- generic entity-type rate (concept/thing/entity/other) --")
    _row("generic", _pct(base["generic"], base["_ent"]), _pct(new["generic"], new["_ent"]))

    if new["kg_entity_properties"]:
        print("\n  -- attribute lane (entity → value statements) --")
        np_ = new["kg_entity_properties"]
        _row("entity_properties", "—", str(np_))
        _row("distinct properties", "—", str(new["attr_distinct"]))
        _row("value_num parsed", "—", _pct(new["attr_value_num"], np_))
        _row("with condition", "—", _pct(new["attr_condition"], np_))
        _row("mapped (canonical)", "—", _pct(new["attr_mapped"], np_))

    print("\n  -- embeddings present by target_kind --")
    kinds = sorted(set(base["embeddings_by_kind"]) | set(new["embeddings_by_kind"]))
    for kind in kinds:
        _row(
            kind,
            str(base["embeddings_by_kind"].get(kind, 0)),
            str(new["embeddings_by_kind"].get(kind, 0)),
        )
    if not kinds:
        _row("(none)", "0", "0")

    print("\n  -- vocabulary cluster structure (VocabResolver) --")

    def _cl(d: dict | None, key: str) -> str:
        return "—" if d is None else str(d[key])

    def _cl_cov(d: dict | None) -> str:
        return "—" if d is None else _pct(d["covered"], d["distinct_edges"])

    for lane, bkey in (("predicate", "pred_cluster"), ("property", "prop_cluster")):
        b, n = base[bkey], new[bkey]
        _row(f"{lane} alias rows (synonym)", _cl(b, "alias_rows"), _cl(n, "alias_rows"))
        _row(f"{lane} parent rows (child_of)", _cl(b, "parent_rows"), _cl(n, "parent_rows"))
        _row(f"{lane} alias clusters", _cl(b, "alias_clusters"), _cl(n, "alias_clusters"))
        _row(f"{lane} largest alias cluster", _cl(b, "alias_max"), _cl(n, "alias_max"))
        _row(f"{lane} vocab covered by structure", _cl_cov(b), _cl_cov(n))

    print("\n  -- near-dupe predicate families (>1 distinct, normalized key) --")
    for tag, m in (("baseline", base), ("new", new)):
        fams = m["pred_families"]
        print(f"    [{tag}] {len(fams)} families")
        for key, preds in sorted(fams.items(), key=lambda kv: -len(kv[1]))[:15]:
            print(f"      {key}: {', '.join(preds)}")
    print()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python experiments/eval_corpus_compare.py <baseline_kg.db> <new_kg.db>",
            file=sys.stderr,
        )
        return 2
    base_path, new_path = argv
    base_conn = _connect(base_path)
    new_conn = _connect(new_path)
    try:
        render(collect(base_conn), collect(new_conn), base_path, new_path)
    finally:
        base_conn.close()
        new_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
