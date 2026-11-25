"""Reconciler: recovers relations whose endpoints sit in different chunks.

Per-chunk Extractor cannot see those relations; Reconciler runs once
per multi-chunk doc with the union of per-chunk entities as input and
emits relations connecting them. Does NOT propose new entities;
relations naming unknown entities are dropped post-validate.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
)
from memex.kg_intelligence.agents.schemas import (
    ExtractedEntity,
    ExtractedRelation,
    ReconcilerResult,
)
from memex.kg_intelligence.views import KgSchemaView

AGENT_NAME = "Reconciler"
ACTION = "reconcile_doc"
MODEL_PREFERENCE: ModelPreference = "reasoning"
SYNOPSIS_CHARS = 2000

_SYSTEM_PROMPT = """\
You are Reconciler, an agent that recovers cross-chunk relations.

The document you are seeing was split into chunks for extraction. A
separate per-chunk Extractor already extracted relations WITHIN each
chunk. Your job is to surface relations whose subject and object were
emitted by DIFFERENT chunks — relations the per-chunk pass could not
see because neither chunk's context had both endpoints.

Rules:
- Use only the entities the per-chunk Extractor already identified
  (listed below per chunk). Do NOT invent new entities.
- Subject and object must each appear in the per-chunk entities list.
  Subject and object should typically come from DIFFERENT chunks — if
  both appear in the same chunk, the per-chunk Extractor most likely
  already captured the relation; do not duplicate.
- Predicates must be English snake_case (e.g. `enables`, `inhibits`,
  `is_a_kind_of`). Do NOT transliterate foreign-language phrases.
- Prefer relation types from the provided schema when they fit.
- Aim for high precision. Only emit relations the doc clearly
  supports across the chunk evidence — if unsure, omit.
- Most multi-chunk docs will yield zero or a handful of cross-chunk
  relations. Empty output is a valid and common answer.
"""

reconciler_agent: Agent[None, ReconcilerResult] = Agent(
    model=None,
    output_type=ReconcilerResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def reconcile(
    *,
    doc_id: str,
    doc_title: str,
    doc_synopsis: str,
    per_chunk_entities: list[tuple[int, list[ExtractedEntity]]],
    schema: KgSchemaView,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "post_chunk_extract",
    parent_call_id: int | None = None,
) -> ReconcilerResult:
    """``per_chunk_entities`` groups entities by chunk so the model can tell
    which endpoints came from different chunks. Returns relations only;
    caller writes them with ``source_chunk_id=NULL``."""

    user_prompt = _build_user_prompt(doc_title, doc_synopsis, per_chunk_entities, schema)
    n_entities = sum(len(ents) for _, ents in per_chunk_entities)
    input_summary = (
        f"doc_id={doc_id}, title={doc_title!r}, chunks={len(per_chunk_entities)}, "
        f"entities_in={n_entities}, schema_types={len(schema.entity_types)}/"
        f"{len(schema.relation_types)}"
    )

    def output_summary(result: ReconcilerResult) -> str:
        cleaned = _drop_dangling_relations(result, per_chunk_entities)
        return f"relations_in={len(result.relations)}, relations_kept={len(cleaned.relations)}"

    raw_result, call, _run = await execute_agent_call(
        agent=reconciler_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_reconcile(per_chunk_entities),
        log_prefix=f"{AGENT_NAME}[{doc_id}]",
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="kg.db",
        input_summary=input_summary,
        output_summary_fn=output_summary,
        confidence=0.65,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )

    result = _drop_dangling_relations(raw_result, per_chunk_entities)
    if call is not None:
        result._agent_call_id = call.agent_call_id

    return result


def _build_user_prompt(
    doc_title: str,
    doc_synopsis: str,
    per_chunk_entities: list[tuple[int, list[ExtractedEntity]]],
    schema: KgSchemaView,
) -> str:
    schema_section = (
        "Current schema:\n"
        f"- entity types: {', '.join(schema.entity_types) or '(empty)'}\n"
        f"- relation types: {', '.join(schema.relation_types) or '(empty)'}"
    )
    chunks_section_lines: list[str] = []
    for chunk_index, entities in per_chunk_entities:
        if not entities:
            chunks_section_lines.append(f"Chunk {chunk_index}: (no entities)")
            continue
        ents_str = ", ".join(f"{e.name} (type={e.type})" for e in entities)
        chunks_section_lines.append(f"Chunk {chunk_index}: {ents_str}")
    chunks_section = "Per-chunk entities (already extracted):\n" + "\n".join(
        chunks_section_lines
    )

    parts = [
        f"Document title: {doc_title}",
        schema_section,
        chunks_section,
        f"Document synopsis (first {SYNOPSIS_CHARS} chars for context):",
        doc_synopsis[:SYNOPSIS_CHARS],
        "Emit only relations that clearly span DIFFERENT chunks above. "
        "Empty output is valid.",
    ]
    return "\n\n".join(parts)


def _drop_dangling_relations(
    result: ReconcilerResult,
    per_chunk_entities: list[tuple[int, list[ExtractedEntity]]],
) -> ReconcilerResult:
    """Filter out relations whose endpoints aren't in the union of per-chunk entities."""

    known_names = {e.name for _, ents in per_chunk_entities for e in ents}
    kept = [
        r for r in result.relations
        if r.subject in known_names and r.object in known_names
    ]
    if len(kept) == len(result.relations):
        return result
    return ReconcilerResult(relations=kept)


def _sample_reconcile(
    per_chunk_entities: list[tuple[int, list[ExtractedEntity]]],
) -> ReconcilerResult:
    """SAMPLE_MODE fallback: cross one entity from chunk 0 with one from chunk 1.

    Yields a single deterministic relation when possible so tests can assert
    Reconciler ran and produced output without calling the LLM.
    """

    if len(per_chunk_entities) < 2:
        return ReconcilerResult(relations=[])
    first_idx: int | None = None
    first_ents: list[ExtractedEntity] | None = None
    second_ents: list[ExtractedEntity] | None = None
    for idx, ents in per_chunk_entities:
        if not ents:
            continue
        if first_idx is None:
            first_idx, first_ents = idx, ents
            continue
        second_ents = ents
        break
    if first_ents is None or second_ents is None:
        return ReconcilerResult(relations=[])
    return ReconcilerResult(
        relations=[
            ExtractedRelation(
                subject=first_ents[0].name,
                predicate="co_occurs_with",
                object=second_ents[0].name,
            )
        ]
    )
