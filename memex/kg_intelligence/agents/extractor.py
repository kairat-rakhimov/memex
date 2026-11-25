"""Extractor: per-chunk entity + relation extraction.

Free-form: the Extractor sees NO canonical schema —
it is a pure faithfulness pass that captures whatever the text states, in its
own surface vocabulary. The downstream per-doc Mapper canonicalizes that
vocabulary. Relations referencing entities NOT in the same response are dropped
post-validate.
"""

from __future__ import annotations

from typing import Literal

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
    ExtractionResult,
)

AGENT_NAME = "Extractor"
MODEL_PREFERENCE: ModelPreference = "chat"
ACTION = "extract_doc"
ACTION_FOCUSED = "extract_doc_focused"
ACTION_SESSION = "extract_doc_session"
DOC_TEXT_BUDGET = 30_000
# Session mode puts the whole doc in the cached prefix; cap well under the
# 128k model context, leaving room for output.
SESSION_DOC_BUDGET = 360_000  # ~90k tokens

ExtractorMode = Literal["initial", "focused"]

_SYSTEM_PROMPT = """\
You are Extractor, an agent that turns one document into a structured set of
entities, relations, and attributes for a knowledge graph.

Rules:
- Only extract entities + relations explicitly present in the document text.
  Do NOT invent connections from outside knowledge.
- **Entity `name` and `type` MUST be in English.** When the source text
  refers to an entity by a non-English surface form (e.g. `绿氢`,
  `hidrógeno verde`, `España`, `Países Bajos`, `國家能源局`,
  `Alemania`, `Frankreich`, `Dusíková past`), translate the canonical name
  to English (`green hydrogen`, `Spain`, `Netherlands`, `National Energy
  Administration of China`, `Germany`, `France`, `Nitrogen trap`). Do NOT
  store the original non-English surface form as `name`. Common English
  conventions: chemicals → English IUPAC/common name; countries →
  English exonym; organizations → official English name when available,
  else direct translation. This keeps the KG monolingual so the same
  real-world entity from English + Chinese + Spanish sources resolves
  to one node.
- **Relations vs attributes — decide per statement by the OBJECT:**
  - **Relation** — the object is ANOTHER ENTITY. Emit in `relations` with a
    `subject`, an `object` (both naming entities in the entities list), and a
    `predicate` that is a BARE relation verb phrase — **≤3 words, active voice,
    and NEVER with the object, a qualifier, or modality baked in** (those belong
    in the `object` slot, not the predicate). Reduce to the core verb:
    `aims_to_establish_entry_point_for_energy_supplies_in` → `supplies`;
    `deems_all_grid_power_as_renewable_for` → `classifies`;
    `signed_agreement_to_reduce_emissions_with` → `partners_with`. A 2-word
    `verb_prep` (`used_in`, `located_in`, `depends_on`) is fine; a longer phrase
    ending in a preposition is a clause — drop the embedded noun. Use for links
    between real things: `A produces B`, `X more_efficient_than Y`,
    `M located_in N`.
  - **Attribute** — the object is a measured / qualitative VALUE (a number+unit,
    a percentage, a date, a bare quantity, a qualitative grade). Emit in
    `attributes`, NOT `relations`: give `entity` (the thing the property is OF,
    present in the entities list), `property` (the property phrase, e.g.
    `overpotential`, `Tafel slope`, `efficiency`, `cost`), `value` (the value
    phrase verbatim, e.g. `491 mV`, `99.4%`, `$1-3/kg`), `unit` if cleanly
    separable (e.g. `mV`), and `condition` if the value holds only under stated
    conditions (e.g. `0.1 M KOH, pH 7`).
  - **A statement whose object is a number/measurement is an ATTRIBUTE.** Never
    force a measured value into a relation's object slot — that pollutes the
    graph with value-nodes. Only use a relation when the object is genuinely
    another named entity.
- **Predicates must be English snake_case.** Do NOT transliterate or extract
  foreign-language phrases verbatim. If a relation is only expressed in a
  non-English quoted phrase, either translate it to a canonical English
  predicate (e.g. `es promovido por` → `promoted_by`) or skip the relation
  entirely.
- Aim for high precision; the loop will accumulate edges from many docs.
- **Do NOT extract individual person entities** (paper authors, conference
  call participants, transcript speakers, journalists, etc.). Author
  attribution is tracked separately via document metadata, not in the
  graph. Only extract a person when the person is themselves a subject
  of substantive domain content (e.g. a named researcher whose specific
  discovery is being discussed in the text — rare). Default: skip.
- **Do NOT extract self-loop relations** where subject and object refer
  to the same entity.
- **For EVERY relation and attribute, set `source_quote` to the VERBATIM
  sentence from the document text that states it** — copied EXACTLY, character
  for character (do NOT paraphrase, summarize, translate, normalize whitespace,
  or stitch together distant fragments). Quote the single smallest contiguous
  span that contains the fact, so it can be located in the source verbatim.
  Prefer one clean sentence. Entities themselves do NOT get a `source_quote`.
"""

extractor_agent: Agent[None, ExtractionResult] = Agent(
    model=None,
    output_type=ExtractionResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def extract(
    *,
    doc_id: str,
    doc_title: str,
    doc_text: str,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    chunk_id: int | None = None,
    triggered_by: str = "per_doc",
    mode: ExtractorMode = "initial",
    neighbour_entities: list[tuple[str, str]] | None = None,
    parent_call_id: int | None = None,
    whole_doc: str | None = None,
) -> ExtractionResult:
    """``mode="focused"`` re-fires on a low-yield doc with neighbour-entity
    priming; the call links to the initial Extractor call via
    ``parent_call_id`` and ``action`` becomes ``extract_doc_focused``.

    When ``whole_doc`` is given (session mode), ``doc_text`` is the
    **focus chunk** and ``whole_doc`` is the full body. The prompt places
    ``whole_doc`` first (byte-identical across a doc's forks → a cached prefix)
    and the focus chunk last; the model extracts the focus chunk with the whole
    doc visible both directions."""

    session = whole_doc is not None
    if session:
        truncated = (whole_doc or "")[:SESSION_DOC_BUDGET]
        user_prompt = _build_session_prompt(doc_title, truncated, doc_text)
        action = ACTION_SESSION
    else:
        truncated = (doc_text or "")[:DOC_TEXT_BUDGET]
        user_prompt = _build_user_prompt(
            doc_title, truncated,
            mode=mode, neighbour_entities=neighbour_entities or [],
        )
        action = ACTION if mode == "initial" else ACTION_FOCUSED
    input_summary = (
        f"doc_id={doc_id}, title={doc_title!r}, "
        f"text_chars={len(truncated)} (of {len(whole_doc) if session else len(doc_text)}), "
        f"mode={'session' if session else mode}"
    )

    def output_summary(result: ExtractionResult) -> str:
        return (
            f"entities={len(result.entities)}, relations={len(result.relations)}, "
            f"attributes={len(result.attributes)}"
        )

    result, call, _run = await execute_agent_call(
        agent=extractor_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_extraction(doc_title),
        log_prefix=f"{AGENT_NAME}[{doc_id}:{mode}]",
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=action,
        triggered_by=triggered_by,
        target_agent="kg.db",
        input_summary=input_summary,
        output_summary_fn=output_summary,
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )

    # Dangling relations (an endpoint not separately extracted as an entity)
    # are NOT dropped here anymore: the per-doc Mapper canonicalizes every
    # triple endpoint into an entity, and session-mode cross-section relations
    # legitimately reference entities introduced in other chunks. Keeping them
    # recovers recall the old per-chunk filter discarded.
    if call is not None:
        result._agent_call_id = call.agent_call_id

    if call is not None and sinks.store is not None:
        sinks.store.record_extraction(
            agent_call_id=call.agent_call_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            entities_extracted=len(result.entities),
            relations_extracted=len(result.relations),
            relations_dropped_dangling=0,
        )

    return result


def _build_user_prompt(
    doc_title: str,
    doc_text: str,
    *,
    mode: ExtractorMode = "initial",
    neighbour_entities: list[tuple[str, str]] = (),  # type: ignore[assignment]
) -> str:
    parts = [
        f"Document title: {doc_title}",
    ]
    if mode == "focused":
        parts.append(
            "**Focused re-extraction.** The first extraction pass on this doc "
            "yielded few entities. The doc may have unusual structure or sparse "
            "phrasing. Look more carefully for entities of these types and "
            "names, drawn from same-round neighbour docs:"
        )
        if neighbour_entities:
            parts.append(
                "\n".join(
                    f"- {name} (type={etype})" for name, etype in neighbour_entities
                )
            )
        else:
            parts.append("(no neighbour entities available this round)")
    parts.append("Document text:")
    parts.append(doc_text)
    return "\n\n".join(parts)


def _build_session_prompt(
    doc_title: str,
    whole_doc: str,
    focus_chunk: str,
) -> str:
    """Whole-doc-in-context prompt for session mode.

    Order matters for prompt caching: everything identical across a doc's
    forks (`title`, whole doc) goes first so it forms a cached prefix; only
    the trailing focus chunk varies per fork.
    """

    instructions = (
        "You are given the FULL document for context, then ONE section of it "
        "(shown last). Extract entities, relations, and attributes for THAT "
        "section:\n"
        "- Extract every entity, relation, and attribute expressed in the "
        "section (route measured/numeric-valued statements to attributes).\n"
        "- ALSO emit a relation when the section connects one of its entities "
        "to an entity introduced elsewhere in the full document (a cross-"
        "section relation) — use the full document to identify and name that "
        "entity. This is the point of having the whole document.\n"
        "- Use the full document to resolve coreferences (pronouns, "
        "abbreviations, synonyms) and to use ONE consistent canonical name per "
        "entity.\n"
        "- Do NOT re-extract content that belongs only to other sections and "
        "is not connected to this one."
    )
    return "\n\n".join(
        [
            f"Document title: {doc_title}",
            instructions,
            "Full document (context):",
            whole_doc,
            "----",
            "Section to extract from:",
            focus_chunk,
        ]
    )


def _sample_extraction(doc_title: str) -> ExtractionResult:
    """SAMPLE_MODE fallback: minimal extraction grounded in the title."""

    cleaned = (doc_title or "topic").strip()
    head = cleaned.split()[0] if cleaned else "topic"
    return ExtractionResult(
        entities=[ExtractedEntity(name=head, type="concept")],
        relations=[],
    )
