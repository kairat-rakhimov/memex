"""Query-time vocabulary expansion over the cluster-structured substrate.

Predicate / property sprawl is the query-time tax: the same relationship is
spread across synonym labels (``used_in`` / ``used_for`` / ``uses``) and
specializations (``electrolyzes`` under ``produces``). Symbolic + typed-
topological retrieval miss edges labelled with a synonym unless the query label
is expanded to its whole cluster. Semantic fact-embeddings already cluster
synonyms regardless of label, so only the symbolic lane needs this.

Expansion uses three signals, in order of trust:
1. **alias equivalence class** — ``predicate_aliases`` / ``property_aliases``
   (alias ↔ canonical), the VocabResolver's synonym verdicts.
2. **specialization rollup** — ``schema_parents`` children of the label
   (a broad query subsumes its narrower forms).
3. **embedding neighbours** — cosine over the embedded predicate/property NAMES
   (no LLM: names are already embedded), catching near-synonyms the alias
   table missed.

If a build skipped vocabulary finalization (e.g. it was interrupted), the alias /
parent tables are empty and expansion **degrades to self + embedding
neighbours** — still better than the bare label.
"""

from __future__ import annotations

import numpy as np

from memex.kg_query.deps import QueryDeps

_DEFAULT_EMBED_K = 6
_DEFAULT_EMBED_FLOOR = 0.62


def _embed_neighbours(
    deps: QueryDeps, labels: set[str], *, embed_kind: str, k: int, floor: float
) -> set[str]:
    ids, matrix = deps.matrix(embed_kind)
    if matrix.size == 0 or not ids:
        return set()
    index = {name.lower(): i for i, name in enumerate(ids)}
    found: set[str] = set()
    for label in labels:
        i = index.get(label.lower())
        if i is None:
            continue
        scores = matrix @ matrix[i]
        order = np.argsort(-scores)
        taken = 0
        for j in order:
            if j == i or scores[j] < floor:
                continue
            found.add(ids[int(j)])
            taken += 1
            if taken >= k:
                break
    return found


def _expand(
    deps: QueryDeps,
    label: str,
    *,
    alias_table: str,
    parent_kind: str,
    embed_kind: str,
    embed_k: int,
    embed_floor: float,
    include_children: bool,
) -> list[str]:
    pairs = deps.alias_pairs(alias_table)
    alias_to_canon = {a.lower(): c for a, c in pairs}

    result: set[str] = {label}
    canon = alias_to_canon.get(label.lower(), label)
    result.add(canon)

    # Alias equivalence class: every alias that resolves to the same canonical.
    canon_lower = canon.lower()
    for a, c in pairs:
        if c.lower() == canon_lower:
            result.add(a)

    if include_children:
        children = deps.children_of(parent_kind)
        for parent in {label.lower(), canon_lower}:
            for child in children.get(parent, []):
                result.add(child)
                # roll a child's own aliases in too
                for a, c in pairs:
                    if c.lower() == child.lower():
                        result.add(a)

    result |= _embed_neighbours(
        deps, {label, canon}, embed_kind=embed_kind, k=embed_k, floor=embed_floor
    )
    return sorted(result)


def expand_predicate(
    deps: QueryDeps,
    predicate: str,
    *,
    embed_k: int = _DEFAULT_EMBED_K,
    embed_floor: float = _DEFAULT_EMBED_FLOOR,
    include_children: bool = True,
) -> list[str]:
    """All predicate labels equivalent or narrower than ``predicate`` that may
    actually appear in ``kg_relations`` — alias class ∪ children ∪ embed-nn.
    Always includes ``predicate`` itself."""

    return _expand(
        deps,
        predicate,
        alias_table="predicate_aliases",
        parent_kind="relation_type",
        embed_kind="predicate",
        embed_k=embed_k,
        embed_floor=embed_floor,
        include_children=include_children,
    )


def expand_property(
    deps: QueryDeps,
    property_name: str,
    *,
    embed_k: int = _DEFAULT_EMBED_K,
    embed_floor: float = _DEFAULT_EMBED_FLOOR,
    include_children: bool = True,
) -> list[str]:
    """The attribute-lane analogue of :func:`expand_predicate` over
    ``property_aliases`` + ``schema_parents`` (property_type) + the embedded
    property NAMES."""

    return _expand(
        deps,
        property_name,
        alias_table="property_aliases",
        parent_kind="property_type",
        embed_kind="property",
        embed_k=embed_k,
        embed_floor=embed_floor,
        include_children=include_children,
    )
