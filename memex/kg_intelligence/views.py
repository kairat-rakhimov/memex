"""Read-only views over the canonical KG (kg.db).

Built by querying ``TraceStore``. Writes flow through ``TraceStore``
directly — re-load the view between agent calls if the loop has just
written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memex.observability import EntitySnapshot, TraceStore


@dataclass(frozen=True)
class KgSchemaView:
    """``entity_types`` / ``relation_types`` carry canonicals only;
    aliases are surfaced separately so callers can filter against
    canonicals ∪ alias keys without re-proposing routed forms."""

    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    property_types: list[str] = field(default_factory=list)
    predicate_aliases: list[tuple[str, str]] = field(default_factory=list)
    entity_type_aliases: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class KgView:
    schema: KgSchemaView
    entities: dict[str, EntitySnapshot]
    predicate_counts: list[tuple[str, int]]

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return sum(n for _, n in self.predicate_counts)


def load_schema_view(store: TraceStore) -> KgSchemaView:
    schema = store.list_kg_schema()
    return KgSchemaView(
        entity_types=list(schema["entity_types"]),
        relation_types=list(schema["relation_types"]),
        property_types=list(schema["property_types"]),
        predicate_aliases=store.list_predicate_aliases(),
        entity_type_aliases=store.list_entity_type_aliases(),
    )


def load_kg_view(store: TraceStore) -> KgView:
    schema = load_schema_view(store)
    entities = store.entity_snapshot_for_resolver()
    predicates = store.predicate_counts()
    return KgView(
        schema=schema,
        entities=entities,
        predicate_counts=predicates,
    )
