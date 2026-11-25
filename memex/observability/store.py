"""Per-build SQLite store: one ``kg/<kg_id>/kg.db`` (canonical KG + trace).

Writes serialise on a per-instance ``threading.Lock``; connection
opens with ``check_same_thread=False`` for ``asyncio.to_thread``
workers that share the store. Read methods return plain dicts for
direct JSON passthrough.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def apply_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: applies ``schema.sql`` and enables FK enforcement.

    Fresh-open only — no migrations, no ALTER TABLE; reopens of a
    pre-existing kg.db with a stale shape will fail noisily.
    """

    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection wired for this project's usage pattern."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        isolation_level=None,  # autocommit; we use explicit transactions where needed
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    apply_schema(conn)
    return conn


def kg_db_path(kg_root: Path) -> Path:
    """Canonical location of the kg.db file for a kg-loop build root."""

    return kg_root / "kg.db"


# ── handles ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildHandle:
    build_id: int
    build_run_id: str


@dataclass(frozen=True)
class RoundHandle:
    round_id: int
    round_number: int


@dataclass(frozen=True)
class AgentCallHandle:
    agent_call_id: int
    sequence_in_round: int


# ── KG view shapes (read-only) ────────────────────────────────────────────


@dataclass(frozen=True)
class EntitySnapshot:
    """One entity in the form the Resolver / KGReviewer consume.

    ``source_doc_ids`` is computed on demand from kg_relations;
    not denormalised on kg_entities. Empty list when the entity has
    no relations yet.
    """

    id: int
    canonical_name: str
    entity_type: str | None
    mentions: int
    degree: int
    source_doc_ids: list[str]


@dataclass(frozen=True)
class RelationSnapshot:
    """One (subj, pred, obj, source_doc) row from kg_relations."""

    id: int
    subject_name: str
    predicate: str
    object_name: str
    source_doc_id: str | None


@dataclass(frozen=True)
class MappedRelation:
    """One gated, canonicalized triple ready to merge.

    The gate has already decided ``predicate`` (the final form to store:
    Mapper's canonical, a fallback canonical, or the raw surface form when
    ``mapped_flag`` is False) and ``mapped_flag``. ``canonical_subject`` /
    ``canonical_object`` carry the Mapper's surface forms — identity dedup is
    the Resolver's job, downstream. Entity *types* are not carried here: they
    live on ``kg_entities`` and are supplied to the merge as a separate
    name→type list, so one entity can't acquire two conflicting types from two
    triples at the SQL layer. Scores are per-observation; aggregated at read
    time.
    """

    raw_subject: str
    raw_predicate: str
    raw_object: str
    canonical_subject: str
    canonical_object: str
    predicate: str
    mapper_confidence: float
    newness_score: float
    mapped_flag: bool
    source_chunk_id: int | None = None
    source_quote: str = ""  # verbatim source sentence from the Extractor ('' when absent)


@dataclass(frozen=True)
class MappedEntityProperty:
    """One gated, canonicalized ATTRIBUTE ready to merge.

    The attribute-lane analogue of :class:`MappedRelation`: a statement whose
    object is a VALUE rather than an entity. ``entity`` is the canonical subject
    surface form (resolved to an entity_id at merge, like a relation endpoint).
    The gate has decided ``property`` (canonical / raw) and ``mapped_flag``;
    ``value_num`` is the comparable numeric (None when non-numeric), ``value_raw``
    the surface. Scores are per-observation; aggregated at read time.
    """

    entity: str
    raw_property: str
    raw_value: str
    property: str
    value_num: float | None
    value_raw: str
    unit: str
    condition: str
    mapper_confidence: float
    newness_score: float
    mapped_flag: bool
    source_chunk_id: int | None = None
    source_quote: str = ""  # verbatim source sentence from the Extractor ('' when absent)


@dataclass(frozen=True)
class AggregatedEdge:
    """A read-time roll-up of all per-source observations of one canonical
    edge. ``avg_confidence`` is the mean of ``mapper_confidence`` over the
    observation rows; ``corroboration`` is the distinct source-doc count.
    Confidence is NEVER stacked — corroboration is the separate axis.
    """

    subject_name: str
    predicate: str
    object_name: str
    avg_confidence: float | None
    corroboration: int
    observations: int


# ── store ────────────────────────────────────────────────────────────────


class TraceStore:
    """Thread-safe writer + reader over the per-build trace DB.

    Construct via :meth:`open` so the DB file gets created + schema
    applied in one step. Pass the same instance through every agent and
    pipeline helper for one build.
    """

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._lock = threading.RLock()
        self._sequence_by_round: dict[int, int] = {}

    # ── lifecycle ────────────────────────────────────────────────────

    @classmethod
    def open(cls, kg_root: Path) -> TraceStore:
        path = kg_db_path(kg_root)
        conn = connect(path)
        return cls(conn, path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ── writes: builds + rounds ──────────────────────────────────────

    def start_build(
        self,
        *,
        build_run_id: str,
        kg_id: str,
        base_query: str,
        config: dict[str, Any],
        sources: list[str],
        sample_mode: bool,
        resume_from_round: int | None,
    ) -> BuildHandle:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO builds (
                    build_run_id, kg_id, base_query, config_json,
                    sources_json, sample_mode, resume_from_round, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_run_id,
                    kg_id,
                    base_query,
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    json.dumps(sources, ensure_ascii=False),
                    1 if sample_mode else 0,
                    resume_from_round,
                    _utc_now(),
                ),
            )
            return BuildHandle(build_id=cur.lastrowid, build_run_id=build_run_id)

    def finish_build(
        self,
        build: BuildHandle,
        *,
        stop_reason: str,
        rounds_completed: int,
        total_docs: int,
        total_entities: int,
        total_relations: int,
        schema_entity_types: int,
        schema_relation_types: int,
    ) -> None:
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE builds SET
                    finished_at = ?,
                    stop_reason = ?,
                    rounds_completed = ?,
                    total_docs = ?,
                    total_entities = ?,
                    total_relations = ?,
                    schema_entity_types = ?,
                    schema_relation_types = ?,
                    total_prompt_tokens = COALESCE((
                        SELECT SUM(prompt_tokens) FROM rounds WHERE build_id = ?
                    ), 0),
                    total_completion_tokens = COALESCE((
                        SELECT SUM(completion_tokens) FROM rounds WHERE build_id = ?
                    ), 0)
                WHERE id = ?
                """,
                (
                    _utc_now(),
                    stop_reason,
                    rounds_completed,
                    total_docs,
                    total_entities,
                    total_relations,
                    schema_entity_types,
                    schema_relation_types,
                    build.build_id,
                    build.build_id,
                    build.build_id,
                ),
            )

    def start_round(
        self,
        build: BuildHandle,
        *,
        round_number: int,
        entities_before: int,
        relations_before: int,
        schema_entity_types_before: int,
        schema_relation_types_before: int,
    ) -> RoundHandle:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO rounds (
                    build_id, round_number, started_at,
                    entities_before, relations_before,
                    schema_entity_types_before, schema_relation_types_before
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build.build_id,
                    round_number,
                    _utc_now(),
                    entities_before,
                    relations_before,
                    schema_entity_types_before,
                    schema_relation_types_before,
                ),
            )
            round_id = cur.lastrowid
            self._sequence_by_round[round_id] = 0
            return RoundHandle(round_id=round_id, round_number=round_number)

    def finish_round(
        self,
        round_: RoundHandle,
        *,
        entities_after: int,
        relations_after: int,
        schema_entity_types_after: int,
        schema_relation_types_after: int,
        docs_fetched: int,
        docs_skipped: int,
        docs_failed: int,
        stop_criterion_fired: str,
        duration_seconds: float,
    ) -> None:
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE rounds SET
                    finished_at = ?,
                    duration_seconds = ?,
                    entities_after = ?,
                    relations_after = ?,
                    schema_entity_types_after = ?,
                    schema_relation_types_after = ?,
                    docs_fetched = ?,
                    docs_skipped = ?,
                    docs_failed = ?,
                    stop_criterion_fired = ?,
                    prompt_tokens = COALESCE((
                        SELECT SUM(prompt_tokens) FROM agent_calls WHERE round_id = ?
                    ), 0),
                    completion_tokens = COALESCE((
                        SELECT SUM(completion_tokens) FROM agent_calls WHERE round_id = ?
                    ), 0)
                WHERE id = ?
                """,
                (
                    _utc_now(),
                    duration_seconds,
                    entities_after,
                    relations_after,
                    schema_entity_types_after,
                    schema_relation_types_after,
                    docs_fetched,
                    docs_skipped,
                    docs_failed,
                    stop_criterion_fired,
                    round_.round_id,
                    round_.round_id,
                    round_.round_id,
                ),
            )

    # ── writes: agent calls ──────────────────────────────────────────

    def begin_agent_call(
        self,
        round_: RoundHandle,
        *,
        agent_name: str,
        action: str,
        triggered_by: str,
        target_agent: str | None = None,
        parent_call_id: int | None = None,
        input_summary: str = "",
    ) -> AgentCallHandle:
        with self._lock:
            seq = self._sequence_by_round.get(round_.round_id, 0) + 1
            self._sequence_by_round[round_.round_id] = seq
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO agent_calls (
                    round_id, parent_call_id, triggered_by, sequence_in_round,
                    agent_name, action, target_agent, started_at,
                    status, input_summary, output_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, '')
                """,
                (
                    round_.round_id,
                    parent_call_id,
                    triggered_by,
                    seq,
                    agent_name,
                    action,
                    target_agent,
                    _utc_now(),
                    input_summary,
                ),
            )
            return AgentCallHandle(agent_call_id=cur.lastrowid, sequence_in_round=seq)

    def finish_agent_call(
        self,
        call: AgentCallHandle,
        *,
        status: str,
        duration_ms: int,
        output_summary: str,
        confidence: float | None = None,
        retry_count: int = 0,
        model_name: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        raw_response: str | None = None,
        parsed_output: Any = None,
    ) -> None:
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE agent_calls SET
                    finished_at = ?,
                    duration_ms = ?,
                    status = ?,
                    retry_count = ?,
                    confidence = ?,
                    model_name = ?,
                    output_summary = ?,
                    prompt_tokens = ?,
                    completion_tokens = ?,
                    cached_tokens = ?,
                    error_type = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    _utc_now(),
                    duration_ms,
                    status,
                    retry_count,
                    confidence,
                    model_name,
                    output_summary,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    error_type,
                    error_message,
                    call.agent_call_id,
                ),
            )
            parsed_json = (
                json.dumps(parsed_output, ensure_ascii=False, default=str)
                if parsed_output is not None
                else None
            )
            conn.execute(
                """
                INSERT INTO agent_call_details (
                    agent_call_id, system_prompt, user_prompt,
                    raw_response, parsed_output_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (agent_call_id) DO UPDATE SET
                    system_prompt = excluded.system_prompt,
                    user_prompt = excluded.user_prompt,
                    raw_response = excluded.raw_response,
                    parsed_output_json = excluded.parsed_output_json
                """,
                (
                    call.agent_call_id,
                    system_prompt,
                    user_prompt,
                    raw_response,
                    parsed_json,
                ),
            )

    # ── writes: fetch + chunks + extraction ──────────────────────────

    def record_fetched_doc(
        self,
        round_: RoundHandle,
        *,
        source: str,
        query: str | None,
        url: str | None,
        title: str | None,
        doc_id: str | None,
        outcome: str,
        outcome_detail: str | None = None,
        ranker_position: int | None = None,
        ranker_score: float | None = None,
        parse_method: str | None = None,
        cleaned_text_chars: int | None = None,
        openalex_enriched: bool = False,
    ) -> int:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO fetched_docs (
                    round_id, doc_id, source, url, title, query,
                    ranker_position, ranker_score, outcome, outcome_detail,
                    parse_method, cleaned_text_chars, openalex_enriched
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    round_.round_id,
                    doc_id,
                    source,
                    url,
                    title,
                    query,
                    ranker_position,
                    ranker_score,
                    outcome,
                    outcome_detail,
                    parse_method,
                    cleaned_text_chars,
                    1 if openalex_enriched else 0,
                ),
            )
            return cur.lastrowid

    def update_fetched_doc_outcome(
        self,
        row_id: int,
        *,
        outcome: str,
        outcome_detail: str | None = None,
    ) -> None:
        """Overwrite ``fetched_docs.outcome`` (+ optionally outcome_detail) on
        an existing row. Used by the UsefulnessGate to
        downgrade ``selected_and_fetched`` rows to
        ``extraction_skipped_low_relevance`` after the body's been read but
        before the extract fan-out runs."""

        with self._txn() as conn:
            conn.execute(
                "UPDATE fetched_docs SET outcome = ?, outcome_detail = ? WHERE id = ?",
                (outcome, outcome_detail, row_id),
            )

    def record_doc_assessment(
        self,
        round_: RoundHandle,
        *,
        doc_id: str,
        source: str,
        usefulness: float,
        decision: str,
        reason: str | None = None,
        summary: str | None = None,
    ) -> int:
        """Persist the UsefulnessGate's verdict for one fetched doc (drops
        included). ``summary`` is the gate's second-pass artifact — present for
        keepers, empty/None for drops (the gate skips pass 2 on a drop)."""

        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO doc_assessments (
                    round_id, doc_id, source, usefulness, decision,
                    reason, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    round_.round_id,
                    doc_id,
                    source,
                    usefulness,
                    decision,
                    reason,
                    summary,
                    _utc_now(),
                ),
            )
            return cur.lastrowid

    def record_document(self, doc_id: str, metadata: str, text: str) -> None:
        """Store the canonical per-doc record (serialized metadata + full
        cleaned text). One row per doc_id; re-ingesting upserts. Chunks
        index into ``text`` via char offsets."""

        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, metadata, text, chars, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    metadata = excluded.metadata,
                    text = excluded.text,
                    chars = excluded.chars,
                    created_at = excluded.created_at
                """,
                (doc_id, metadata, text, len(text), _utc_now()),
            )

    def get_document_metadata(self, doc_id: str) -> str | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT metadata FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row is not None else None

    def get_document_text(self, doc_id: str) -> str | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT text FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row is not None else None

    def has_document(self, doc_id: str) -> bool:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row is not None

    def document_ids(self) -> list[str]:
        with self._txn() as conn:
            return [
                r[0] for r in conn.execute(
                    "SELECT doc_id FROM documents ORDER BY doc_id"
                ).fetchall()
            ]

    def record_chunk(
        self,
        *,
        fetched_doc_id: int,
        doc_id: str,
        chunk_index: int,
        char_start: int,
        char_end: int,
        text_chars: int,
        token_estimate: int | None = None,
    ) -> int:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO chunks (
                    fetched_doc_id, doc_id, chunk_index,
                    char_start, char_end, text_chars, token_estimate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fetched_doc_id,
                    doc_id,
                    chunk_index,
                    char_start,
                    char_end,
                    text_chars,
                    token_estimate,
                ),
            )
            return cur.lastrowid

    # ── embeddings ────────────────────────────────────────────────────

    def record_embedding(
        self,
        *,
        target_kind: str,
        target_id: str,
        vector: bytes,
        provider: str,
        model: str,
        dimension: int,
    ) -> int | None:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO embeddings (
                    target_kind, target_id, provider, model,
                    dimension, vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (target_kind, target_id, provider, model, dimension, vector, _utc_now()),
            )
            return cur.lastrowid if cur.rowcount > 0 else None

    def record_embeddings_batch(
        self,
        rows: list[tuple[str, str, str, str, int, bytes]],
    ) -> int:
        with self._txn() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO embeddings (
                    target_kind, target_id, provider, model,
                    dimension, vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [(*r, _utc_now()) for r in rows],
            )
            return conn.execute("SELECT changes()").fetchone()[0]

    def list_embeddings_by_kind(
        self, kind: str
    ) -> list[tuple[str, bytes]]:
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT target_id, vector FROM embeddings WHERE target_kind = ?",
                (kind,),
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

    def count_embeddings(self, kind: str | None = None) -> int:
        with self._txn() as conn:
            if kind is None:
                return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE target_kind = ?", (kind,)
            ).fetchone()[0]

    def delete_embedding(self, *, target_kind: str, target_id: str) -> bool:
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE target_kind = ? AND target_id = ?",
                (target_kind, target_id),
            )
            return conn.execute("SELECT changes()").fetchone()[0] > 0

    def embedding_model_check(self, expected_model: str) -> str | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT DISTINCT model FROM embeddings LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return row[0] if row[0] != expected_model else None

    def chunks_without_embeddings(self) -> list[tuple[int, str, int, int, int]]:
        """Return (id, doc_id, chunk_index, char_start, char_end) for chunks missing embeddings."""
        with self._txn() as conn:
            return conn.execute(
                """
                SELECT c.id, c.doc_id, c.chunk_index, c.char_start, c.char_end
                FROM chunks c
                LEFT JOIN embeddings e
                  ON e.target_kind = 'chunk' AND e.target_id = CAST(c.id AS TEXT)
                WHERE e.id IS NULL
                """
            ).fetchall()

    def entities_without_embeddings(self) -> list[tuple[int, str]]:
        """Return (id, canonical_name) for entities missing embeddings."""
        with self._txn() as conn:
            return conn.execute(
                """
                SELECT e.id, e.canonical_name
                FROM kg_entities e
                LEFT JOIN embeddings emb
                  ON emb.target_kind = 'entity' AND emb.target_id = CAST(e.id AS TEXT)
                WHERE emb.id IS NULL
                """
            ).fetchall()

    def predicates_without_embeddings(self) -> list[str]:
        """Return predicate names from kg_schema missing embeddings."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT s.name
                FROM kg_schema s
                LEFT JOIN embeddings e
                  ON e.target_kind = 'predicate' AND e.target_id = s.name
                WHERE s.kind = 'relation_type' AND e.id IS NULL
                """
            ).fetchall()
            return [r[0] for r in rows]

    def properties_without_embeddings(self) -> list[str]:
        """Return property-type names from kg_schema missing a 'property' (name)
        embedding — the attribute-lane analogue of
        ``predicates_without_embeddings``. Distinct from the per-fact 'attribute'
        embeddings: this embeds the property NAME so the VocabResolver can cluster
        the property vocabulary (and the hypothesis layer can retrieve by it)."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT s.name
                FROM kg_schema s
                LEFT JOIN embeddings e
                  ON e.target_kind = 'property' AND e.target_id = s.name
                WHERE s.kind = 'property_type' AND e.id IS NULL
                """
            ).fetchall()
            return [r[0] for r in rows]

    def relations_without_embeddings(self) -> list[tuple[int, str]]:
        """Return (id, fact-sentence) for kg_relations rows missing a 'relation'
        embedding — the canonical fact as a natural sentence ('subject predicate
        object') for semantic FACT retrieval."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT r.id,
                       s.canonical_name || ' ' || REPLACE(r.predicate, '_', ' ')
                         || ' ' || o.canonical_name AS fact
                FROM kg_relations r
                JOIN kg_entities s ON s.id = r.subject_entity_id
                JOIN kg_entities o ON o.id = r.object_entity_id
                LEFT JOIN embeddings e
                  ON e.target_kind = 'relation' AND e.target_id = CAST(r.id AS TEXT)
                WHERE e.id IS NULL
                """
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

    def entity_properties_without_embeddings(self) -> list[tuple[int, str]]:
        """Return (id, fact-sentence) for kg_entity_properties rows missing an
        'attribute' embedding — 'entity property value (condition)'."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT p.id,
                       e.canonical_name || ' ' || REPLACE(p.property, '_', ' ')
                         || ' ' || p.value_raw
                         || CASE WHEN p.condition <> ''
                                 THEN ' (' || p.condition || ')' ELSE '' END AS fact
                FROM kg_entity_properties p
                JOIN kg_entities e ON e.id = p.entity_id
                LEFT JOIN embeddings emb
                  ON emb.target_kind = 'attribute' AND emb.target_id = CAST(p.id AS TEXT)
                WHERE emb.id IS NULL
                """
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

    # ── extractions ──────────────────────────────────────────────────

    def record_extraction(
        self,
        *,
        agent_call_id: int,
        doc_id: str,
        chunk_id: int | None,
        entities_extracted: int,
        relations_extracted: int,
        relations_dropped_dangling: int,
    ) -> int:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO extractions (
                    agent_call_id, chunk_id, doc_id,
                    entities_extracted, relations_extracted,
                    relations_dropped_dangling
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_call_id,
                    chunk_id,
                    doc_id,
                    entities_extracted,
                    relations_extracted,
                    relations_dropped_dangling,
                ),
            )
            return cur.lastrowid

    def record_entity_chunk_ref(
        self,
        *,
        entity_id: int,
        chunk_id: int,
        agent_call_id: int | None = None,
    ) -> int | None:
        """Record that ``entity_id`` was mentioned in ``chunk_id``.

        Idempotent on re-snapshot via ``INSERT OR IGNORE`` against the
        UNIQUE(entity_id, chunk_id, char_start, char_end) constraint.
        Char offsets default to ``0, 0`` (whole-chunk mention) rather
        than ``NULL`` — SQLite treats NULLs in UNIQUE constraints as
        distinct, which would let re-snapshots stack duplicate rows.
        Future per-mention offsets can use real spans.
        """

        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO entity_chunk_refs (
                    entity_id, chunk_id, char_start, char_end,
                    confidence, agent_call_id
                ) VALUES (?, ?, 0, 0, NULL, ?)
                RETURNING id
                """,
                (entity_id, chunk_id, agent_call_id),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def record_schema_change(
        self,
        round_: RoundHandle,
        *,
        agent_call_id: int,
        event: str,
        type_kind: str,
        name: str,
        rationale: str | None = None,
        decision_reason: str | None = None,
    ) -> int:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO schema_changes (
                    round_id, agent_call_id, event, type_kind, name,
                    rationale, decision_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    round_.round_id,
                    agent_call_id,
                    event,
                    type_kind,
                    name,
                    rationale,
                    decision_reason,
                ),
            )
            return cur.lastrowid

    # ── writes: KG snapshot (called at finalise) ──

    def upsert_kg_entity(
        self,
        *,
        canonical_name: str,
        entity_type: str | None,
        mentions: int,
        degree: int,
        first_seen_round: int | None,
        last_seen_round: int | None,
        created_by_call_id: int,
        last_modified_by_call_id: int | None = None,
        last_modified_at_round: int | None = None,
    ) -> int:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO kg_entities (
                    canonical_name, entity_type, mentions, degree,
                    first_seen_round, last_seen_round,
                    created_by_call_id, last_modified_by_call_id,
                    last_modified_at_round, last_modified_at_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (canonical_name) DO UPDATE SET
                    entity_type = COALESCE(excluded.entity_type, kg_entities.entity_type),
                    mentions = excluded.mentions,
                    degree = excluded.degree,
                    -- Preserve the earliest round this entity was seen across
                    -- re-snapshots; only set first_seen_round on the very first
                    -- insert.
                    first_seen_round = COALESCE(kg_entities.first_seen_round, excluded.first_seen_round),
                    last_seen_round = COALESCE(excluded.last_seen_round, kg_entities.last_seen_round),
                    -- creator stamp is set only on first insert; subsequent
                    -- upserts refresh the modifier stamp.
                    created_by_call_id = COALESCE(kg_entities.created_by_call_id, excluded.created_by_call_id),
                    last_modified_by_call_id = COALESCE(excluded.last_modified_by_call_id, kg_entities.last_modified_by_call_id),
                    last_modified_at_round = COALESCE(excluded.last_modified_at_round, kg_entities.last_modified_at_round),
                    last_modified_at_ts = COALESCE(excluded.last_modified_at_ts, kg_entities.last_modified_at_ts)
                RETURNING id
                """,
                (
                    canonical_name,
                    entity_type,
                    mentions,
                    degree,
                    first_seen_round,
                    last_seen_round,
                    created_by_call_id,
                    last_modified_by_call_id,
                    last_modified_at_round,
                    _utc_now() if last_modified_by_call_id is not None else None,
                ),
            )
            row = cur.fetchone()
            return row[0]

    def record_kg_relation(
        self,
        *,
        subject_entity_id: int,
        predicate: str,
        object_entity_id: int,
        source_doc_id: str | None,
        created_by_call_id: int,
        source_chunk_id: int | None = None,
        round_observed: int | None = None,
        agent_call_id: int | None = None,
    ) -> int | None:
        """Idempotent: re-inserting the same (subj, pred, obj, source_doc) is
        a no-op, returning ``None``. Real new rows return their ``id``.

        Each chunk's extraction calls this through
        :meth:`merge_extraction`; the UNIQUE constraint dedupes,
        ``round_observed`` of an existing row is preserved (it stays the
        round it first appeared)."""

        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO kg_relations (
                    subject_entity_id, predicate, object_entity_id,
                    source_doc_id, source_chunk_id, round_observed, agent_call_id,
                    created_by_call_id, last_modified_at_round, last_modified_at_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    source_doc_id,
                    source_chunk_id,
                    round_observed,
                    agent_call_id,
                    created_by_call_id,
                    round_observed,
                    _utc_now(),
                ),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def update_kg_relation_endpoints(
        self,
        *,
        relation_id: int,
        subject_entity_id: int | None = None,
        object_entity_id: int | None = None,
        modified_by_call_id: int | None = None,
        modified_at_round: int | None = None,
    ) -> dict[str, Any] | None:
        """Rewrite a relation's endpoints (used by Resolver merge fan-out).

        Returns the before-state dict (subject_entity_id + predicate +
        object_entity_id + source_doc_id) so the caller can persist a
        ``kg_mutations`` row with the before/after pair, or ``None`` if the
        relation has been deleted between read and write.
        """

        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT subject_entity_id, predicate, object_entity_id, source_doc_id
                FROM kg_relations WHERE id = ?
                """,
                (relation_id,),
            ).fetchone()
            if row is None:
                return None
            before = dict(row)
            new_subj = subject_entity_id if subject_entity_id is not None else before["subject_entity_id"]
            new_obj = object_entity_id if object_entity_id is not None else before["object_entity_id"]
            conn.execute(
                """
                UPDATE kg_relations SET
                    subject_entity_id = ?,
                    object_entity_id = ?,
                    last_modified_by_call_id = COALESCE(?, last_modified_by_call_id),
                    last_modified_at_round = COALESCE(?, last_modified_at_round),
                    last_modified_at_ts = ?
                WHERE id = ?
                """,
                (
                    new_subj,
                    new_obj,
                    modified_by_call_id,
                    modified_at_round,
                    _utc_now(),
                    relation_id,
                ),
            )
            return before

    # ── writes: mutation log ─────────────────────────────────────────

    def record_kg_mutation(
        self,
        *,
        kind: str,
        target_kind: str,
        target_id: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        reason: str | None,
        confidence: float | None,
        agent_call_id: int | None,
        round: int | None,
        parent_mutation_id: int | None = None,
    ) -> int:
        """Append one row to ``kg_mutations``. Returns the row id so the
        caller can chain endpoint-rewrite mutations under a parent merge.

        ``target_id`` is stored as TEXT so entity/relation row ids (ints)
        and predicate names share the column. Convert at the call site.
        """

        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO kg_mutations (
                    kind, target_kind, target_id,
                    before_json, after_json, reason, confidence,
                    agent_call_id, parent_mutation_id, round, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    target_kind,
                    str(target_id),
                    json.dumps(before, ensure_ascii=False, default=str) if before is not None else None,
                    json.dumps(after, ensure_ascii=False, default=str) if after is not None else None,
                    reason,
                    confidence,
                    agent_call_id,
                    parent_mutation_id,
                    round,
                    _utc_now(),
                ),
            )
            return cur.lastrowid

    # ── writes: kg_schema ────────────────────────────────────────────

    def record_kg_schema_type(
        self,
        *,
        name: str,
        kind: str,
        added_at_call_id: int,
        added_at_round: int,
    ) -> int | None:
        """Insert one schema type. Idempotent on (name, kind) via INSERT
        OR IGNORE — re-applying an already-accepted decision is a no-op
        and returns ``None``."""

        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO kg_schema (
                    name, kind, added_at_call_id, added_at_round
                ) VALUES (?, ?, ?, ?)
                RETURNING id
                """,
                (name, kind, added_at_call_id, added_at_round),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def list_kg_schema(self) -> dict[str, list[str]]:
        """Read the canonical schema in the shape the agents consume:
        ``{"entity_types": [...], "relation_types": [...]}``.

        Filters out names that are also alias keys in the corresponding
        alias table — the schema-entry-uniqueness invariant says a name
        is either canonical OR an alias, never both — so the Mapper's
        candidate set never offers an aliased form as a canonical even if
        a stray ``kg_schema`` row leaked through.
        """

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ks.name, ks.kind
                FROM kg_schema ks
                WHERE NOT (
                    ks.kind = 'relation_type' AND EXISTS (
                        SELECT 1 FROM predicate_aliases pa
                        WHERE pa.alias = ks.name COLLATE NOCASE
                    )
                ) AND NOT (
                    ks.kind = 'entity_type' AND EXISTS (
                        SELECT 1 FROM entity_type_aliases ea
                        WHERE ea.alias = ks.name COLLATE NOCASE
                    )
                )
                ORDER BY ks.id ASC
                """
            ).fetchall()
        out: dict[str, list[str]] = {
            "entity_types": [], "relation_types": [], "property_types": [],
        }
        _bucket = {
            "entity_type": "entity_types",
            "relation_type": "relation_types",
            "property_type": "property_types",
        }
        for row in rows:
            out[_bucket[row["kind"]]].append(row["name"])
        return out

    # ── high-level kg-data ops ────────────────────────────────────────

    def count_entities(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]

    def count_relations(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM kg_relations").fetchone()[0]

    def count_schema_types(self, kind: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM kg_schema WHERE kind = ?", (kind,)
            ).fetchone()[0]

    def list_kg_entities(self) -> list[dict[str, Any]]:
        """Full enumerate for agents that iterate (e.g. KGReviewer)."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, canonical_name, entity_type, mentions, degree,
                       first_seen_round, last_seen_round
                FROM kg_entities
                ORDER BY mentions DESC, id ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def predicate_counts(self) -> list[tuple[str, int]]:
        """Predicate → relation count, descending. Powers KGReviewer's
        relation-type rollup."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT predicate, COUNT(*) AS n
                FROM kg_relations
                GROUP BY predicate
                ORDER BY n DESC
                """
            ).fetchall()
        return [(r["predicate"], r["n"]) for r in rows]

    def property_counts(self) -> list[tuple[str, int]]:
        """Canonical property → observation count, descending.
        The attribute-lane analogue of ``predicate_counts``; powers the Mapper's
        frequent-property candidate floor above the vocabulary cutoff."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT property, COUNT(*) AS n
                FROM kg_entity_properties
                GROUP BY property
                ORDER BY n DESC
                """
            ).fetchall()
        return [(r["property"], r["n"]) for r in rows]

    def merge_extraction(
        self,
        *,
        entities: list[tuple[str, str]],  # (name, type) pairs from one extraction
        relations: list[tuple[str, str, str]],  # (subject, predicate, object)
        source_doc_id: str,
        chunk_id: int | None,
        agent_call_id: int,
        round_number: int,
    ) -> dict[str, int]:
        """Merge one chunk's extraction (alias-aware, schema-canonicalized) into
        the canonical store.

        Not the live ingest path — ``merge_mapped_relations`` is. Kept as an
        alias-resolving merge primitive used by tests. All work
        in one transaction so a mid-chunk crash leaves no half-merged state.

        For each entity: case-insensitive upsert (the UNIQUE COLLATE
        NOCASE constraint catches case-different variants). Mention
        counter increments per appearance; type is set from the first
        non-empty extracted type. Records an entity_chunk_refs row if
        chunk_id is provided.

        For each relation: resolves endpoint names to row ids via
        case-insensitive lookup. Drops the relation if either endpoint
        is missing (dangling relations are filtered upstream but
        defensive here). INSERT OR IGNORE on the UNIQUE shape; degree
        bumps on both endpoints when a row is actually inserted.

        Returns ``{name: entity_id}`` mapping for the entities merged
        this call (handy for tests + callers wanting the IDs).
        """

        name_to_id: dict[str, int] = {}
        with self._txn() as conn:
            for raw_name, raw_type in entities:
                # Alias-aware rewrite: entity-name aliases route to the
                # survivor; entity-type aliases route to the canonical type.
                # The Extractor stays a raw-data compactor — the canonical
                # vocabulary is enforced here at write time.
                canonical_name = (
                    _resolve_entity_alias_in_conn(conn, raw_name) or raw_name
                )
                canonical_type = (
                    _resolve_kind_alias_in_conn(conn, "entity_type_aliases", raw_type)
                    or raw_type
                ) if raw_type else None
                entity_id = self._upsert_entity_case_insensitive(
                    conn,
                    canonical_name=canonical_name,
                    entity_type=canonical_type,
                    agent_call_id=agent_call_id,
                    round_number=round_number,
                )
                # Map both raw and canonical forms so relation-endpoint lookups
                # below hit regardless of which surface form the Extractor used.
                name_to_id[raw_name] = entity_id
                if canonical_name != raw_name:
                    name_to_id[canonical_name] = entity_id
                if chunk_id is not None:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO entity_chunk_refs (
                            entity_id, chunk_id, char_start, char_end,
                            confidence, agent_call_id
                        ) VALUES (?, ?, 0, 0, NULL, ?)
                        """,
                        (entity_id, chunk_id, agent_call_id),
                    )

            for subj_name, predicate, obj_name in relations:
                canonical_pred = (
                    _resolve_kind_alias_in_conn(conn, "predicate_aliases", predicate)
                    or predicate
                )
                subj_id = name_to_id.get(subj_name)
                if subj_id is None:
                    canon_subj = (
                        _resolve_entity_alias_in_conn(conn, subj_name) or subj_name
                    )
                    subj_id = name_to_id.get(canon_subj) or _lookup_entity_id(conn, canon_subj)
                obj_id = name_to_id.get(obj_name)
                if obj_id is None:
                    canon_obj = (
                        _resolve_entity_alias_in_conn(conn, obj_name) or obj_name
                    )
                    obj_id = name_to_id.get(canon_obj) or _lookup_entity_id(conn, canon_obj)
                if subj_id is None or obj_id is None:
                    continue
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO kg_relations (
                        subject_entity_id, predicate, object_entity_id,
                        source_doc_id, source_chunk_id, round_observed,
                        agent_call_id, created_by_call_id,
                        last_modified_at_round, last_modified_at_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (
                        subj_id, canonical_pred, obj_id,
                        source_doc_id, chunk_id, round_number,
                        agent_call_id, agent_call_id,
                        round_number, _utc_now(),
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    # New row — bump endpoint degree.
                    for eid in (subj_id, obj_id):
                        conn.execute(
                            """
                            UPDATE kg_entities SET
                                degree = degree + 1,
                                last_modified_by_call_id = ?,
                                last_modified_at_round = ?,
                                last_modified_at_ts = ?
                            WHERE id = ?
                            """,
                            (agent_call_id, round_number, _utc_now(), eid),
                        )
        return name_to_id

    def merge_mapped_relations(
        self,
        *,
        entities: list[tuple[str, str | None, int | None]],
        relations: list[MappedRelation],
        source_doc_id: str,
        mapper_call_id: int,
        round_number: int,
    ) -> dict[str, int]:
        """Merge one doc's gated, Mapper-canonicalized output.

        ``entities`` is the authoritative ``(canonical_name, canonical_type,
        source_chunk_id)`` list for every entity the doc mentions (triple
        endpoints + standalone); the merge upserts each and writes an
        ``entity_chunk_refs`` row when ``source_chunk_id`` is given. The first
        non-empty type for a name wins (the Mapper, seeing the whole doc, is
        expected to be self-consistent).

        ``relations`` carry the canonical predicate the gate chose plus the raw
        surface forms and per-observation scores. Endpoints resolve by canonical
        name; the relation is dropped if either endpoint is unknown. The UNIQUE
        ``(subj, predicate, obj, source_doc)`` shape dedups re-ingests; degree
        bumps fire only on a genuinely new row.

        Distinct from ``merge_extraction``: it does NOT re-resolve predicate or
        entity-type aliases (the Mapper already canonicalized), and it stamps
        ``raw_*`` + ``mapper_confidence`` + ``newness_score`` + ``mapper_call_id``
        + ``mapped_flag``. Returns ``{canonical_name: entity_id}``.
        """

        name_to_id: dict[str, int] = {}
        type_by_name: dict[str, str | None] = {}
        for name, etype, _chunk in entities:
            if name not in type_by_name or (
                type_by_name[name] in (None, "") and etype
            ):
                type_by_name[name] = etype

        with self._txn() as conn:
            # Entity-name alias grace path: snap a name the Resolver already
            # merged away (entity_aliases) to its survivor, so a later doc's
            # Mapper re-surfacing the merged surface form lands on the survivor
            # row instead of re-creating a duplicate the Resolver must re-merge.
            for name, etype in type_by_name.items():
                canonical_name = _resolve_entity_alias_in_conn(conn, name) or name
                entity_id = self._upsert_entity_case_insensitive(
                    conn,
                    canonical_name=canonical_name,
                    entity_type=etype,
                    agent_call_id=mapper_call_id,
                    round_number=round_number,
                )
                name_to_id[name] = entity_id
                if canonical_name != name:
                    name_to_id[canonical_name] = entity_id
            for name, _etype, chunk_id in entities:
                if chunk_id is None:
                    continue
                entity_id = name_to_id.get(name)
                if entity_id is None:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_chunk_refs (
                        entity_id, chunk_id, char_start, char_end,
                        confidence, agent_call_id
                    ) VALUES (?, ?, 0, 0, NULL, ?)
                    """,
                    (entity_id, chunk_id, mapper_call_id),
                )

            for rel in relations:
                subj_id = name_to_id.get(rel.canonical_subject)
                if subj_id is None:
                    canon = _resolve_entity_alias_in_conn(conn, rel.canonical_subject) \
                        or rel.canonical_subject
                    subj_id = name_to_id.get(canon) or _lookup_entity_id(conn, canon)
                obj_id = name_to_id.get(rel.canonical_object)
                if obj_id is None:
                    canon = _resolve_entity_alias_in_conn(conn, rel.canonical_object) \
                        or rel.canonical_object
                    obj_id = name_to_id.get(canon) or _lookup_entity_id(conn, canon)
                if subj_id is None or obj_id is None:
                    continue
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO kg_relations (
                        subject_entity_id, predicate, object_entity_id,
                        source_doc_id, source_chunk_id, round_observed,
                        agent_call_id, created_by_call_id,
                        last_modified_at_round, last_modified_at_ts,
                        mapper_call_id, raw_subject, raw_predicate, raw_object,
                        source_quote,
                        mapper_confidence, newness_score, mapped_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (
                        subj_id, rel.predicate, obj_id,
                        source_doc_id, rel.source_chunk_id, round_number,
                        mapper_call_id, mapper_call_id,
                        round_number, _utc_now(),
                        mapper_call_id, rel.raw_subject, rel.raw_predicate,
                        rel.raw_object, rel.source_quote, rel.mapper_confidence,
                        rel.newness_score, 1 if rel.mapped_flag else 0,
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    for eid in (subj_id, obj_id):
                        conn.execute(
                            """
                            UPDATE kg_entities SET
                                degree = degree + 1,
                                last_modified_by_call_id = ?,
                                last_modified_at_round = ?,
                                last_modified_at_ts = ?
                            WHERE id = ?
                            """,
                            (mapper_call_id, round_number, _utc_now(), eid),
                        )
        return name_to_id

    def merge_mapped_attributes(
        self,
        *,
        attributes: list[MappedEntityProperty],
        source_doc_id: str,
        mapper_call_id: int,
        round_number: int,
    ) -> int:
        """Merge one doc's gated, canonicalized ATTRIBUTES into
        ``kg_entity_properties``. Each attribute's ``entity``
        resolves to an existing ``kg_entities`` row by canonical name (+ the
        alias grace path); an attribute whose entity is unknown is skipped (the
        entity is expected to have been upserted by the relation/entity merge
        first). ``UNIQUE(entity_id, property, value_raw, condition,
        source_doc_id)`` dedups re-ingests. Returns the number of new rows."""

        written = 0
        with self._txn() as conn:
            for attr in attributes:
                ent_id = _lookup_entity_id(conn, attr.entity)
                if ent_id is None:
                    canon = _resolve_entity_alias_in_conn(conn, attr.entity) or attr.entity
                    ent_id = _lookup_entity_id(conn, canon)
                if ent_id is None:
                    continue
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO kg_entity_properties (
                        entity_id, property, value_num, value_raw, unit, condition,
                        raw_property, raw_value, source_quote, source_doc_id, source_chunk_id,
                        round_observed, mapper_call_id, created_by_call_id,
                        mapper_confidence, newness_score, mapped_flag,
                        last_modified_at_round, last_modified_at_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (
                        ent_id, attr.property, attr.value_num, attr.value_raw,
                        attr.unit, attr.condition, attr.raw_property, attr.raw_value,
                        attr.source_quote, source_doc_id, attr.source_chunk_id, round_number,
                        mapper_call_id, mapper_call_id, attr.mapper_confidence,
                        attr.newness_score, 1 if attr.mapped_flag else 0,
                        round_number, _utc_now(),
                    ),
                )
                if cur.fetchone() is not None:
                    written += 1
        return written

    def aggregated_edges(self) -> list[AggregatedEdge]:
        """Read-time edge roll-up over ``kg_relations``.

        Groups the per-observation rows by ``(canonical_subject, predicate,
        canonical_object)`` and reports the mean ``mapper_confidence``, the
        distinct-source corroboration count, and the raw observation count.
        Confidence is averaged, never stacked — corroboration is the separate
        trust axis. Rows with NULL confidence (Reconciler) are
        ignored by the average but still counted as observations.
        """

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.canonical_name AS subject_name,
                       kr.predicate AS predicate,
                       o.canonical_name AS object_name,
                       AVG(kr.mapper_confidence) AS avg_confidence,
                       COUNT(DISTINCT kr.source_doc_id) AS corroboration,
                       COUNT(*) AS observations
                FROM kg_relations kr
                JOIN kg_entities s ON s.id = kr.subject_entity_id
                JOIN kg_entities o ON o.id = kr.object_entity_id
                GROUP BY kr.subject_entity_id, kr.predicate, kr.object_entity_id
                ORDER BY corroboration DESC, observations DESC
                """
            ).fetchall()
        return [
            AggregatedEdge(
                subject_name=r["subject_name"],
                predicate=r["predicate"],
                object_name=r["object_name"],
                avg_confidence=(
                    float(r["avg_confidence"])
                    if r["avg_confidence"] is not None
                    else None
                ),
                corroboration=int(r["corroboration"]),
                observations=int(r["observations"]),
            )
            for r in rows
        ]

    def _upsert_entity_case_insensitive(
        self,
        conn: sqlite3.Connection,
        *,
        canonical_name: str,
        entity_type: str | None,
        agent_call_id: int,
        round_number: int,
    ) -> int:
        """Case-insensitive upsert. Returns the row id.

        On first insert: stamps creator + modifier; mentions=1.
        On hit: increments mentions, refreshes type if previously
        NULL/'other', updates modifier stamps. The first-seen casing
        of the name wins — re-emissions with different case still hit
        the same row but don't rename it.
        """

        row = conn.execute(
            """
            SELECT id, entity_type FROM kg_entities
            WHERE canonical_name = ? COLLATE NOCASE
            """,
            (canonical_name,),
        ).fetchone()
        if row is None:
            cur = conn.execute(
                """
                INSERT INTO kg_entities (
                    canonical_name, entity_type, mentions, degree,
                    first_seen_round, last_seen_round,
                    created_by_call_id, last_modified_by_call_id,
                    last_modified_at_round, last_modified_at_ts
                ) VALUES (?, ?, 1, 0, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    canonical_name, entity_type,
                    round_number, round_number,
                    agent_call_id, agent_call_id,
                    round_number, _utc_now(),
                ),
            )
            return cur.fetchone()[0]

        entity_id = row["id"]
        existing_type = row["entity_type"]
        # Let a later, more-specific type replace a generic placeholder — not just
        # NULL/'other' but the catch-alls a Mapper may still emit — so a stuck
        # `concept` doesn't outlive a real type seen in a later doc.
        refresh_type = (
            entity_type
            if (existing_type or "").strip().lower()
            in ("", "other", "concept", "thing", "entity", "object", "item")
            else None
        )
        conn.execute(
            """
            UPDATE kg_entities SET
                mentions = mentions + 1,
                entity_type = COALESCE(?, entity_type),
                last_seen_round = ?,
                last_modified_by_call_id = ?,
                last_modified_at_round = ?,
                last_modified_at_ts = ?
            WHERE id = ?
            """,
            (refresh_type, round_number, agent_call_id, round_number,
             _utc_now(), entity_id),
        )
        return entity_id

    # ── resolver-specific reads + writes ────────────────────────────────

    def entity_snapshot_for_resolver(
        self,
        incremental_names: set[str] | None = None,
    ) -> dict[str, EntitySnapshot]:
        """Materialise a small snapshot for Resolver's per-call work.

        Returns ``{canonical_name: EntitySnapshot}``. When
        ``incremental_names`` is provided, **all** entities are still
        returned — Resolver needs the full set to compute candidate
        pairs against the incremental subset. Per-call cost is one
        SELECT + one JOIN, fine at the scale of one topic's graph.

        source_doc_ids is computed via a JOIN on kg_relations (no
        denormalised column).
        """

        with self._lock:
            ent_rows = self._conn.execute(
                """
                SELECT id, canonical_name, entity_type, mentions, degree
                FROM kg_entities
                ORDER BY id ASC
                """
            ).fetchall()
            doc_rows = self._conn.execute(
                """
                SELECT subject_entity_id AS eid, source_doc_id FROM kg_relations
                WHERE source_doc_id IS NOT NULL
                UNION
                SELECT object_entity_id AS eid, source_doc_id FROM kg_relations
                WHERE source_doc_id IS NOT NULL
                """
            ).fetchall()
        docs_by_eid: dict[int, list[str]] = {}
        for r in doc_rows:
            docs_by_eid.setdefault(r["eid"], []).append(r["source_doc_id"])
        return {
            r["canonical_name"]: EntitySnapshot(
                id=r["id"],
                canonical_name=r["canonical_name"],
                entity_type=r["entity_type"],
                mentions=r["mentions"],
                degree=r["degree"],
                source_doc_ids=docs_by_eid.get(r["id"], []),
            )
            for r in ent_rows
        }

    def apply_merge(
        self,
        *,
        survivor_name: str,
        merged_name: str,
        agent_call_id: int | None,
        round_number: int,
    ) -> tuple[int, int, list[int]] | None:
        """Merge ``merged_name`` into ``survivor_name`` atomically.

        Returns ``(survivor_id, merged_id_before_delete, affected_relation_ids)``
        or ``None`` if either name no longer exists (defensive against
        racing tier-1 chains).

        Survivor absorbs the merged row's mentions; degree is the max
        of the two; entity_type fills in if survivor's was empty.
        Relations referencing the merged id get their endpoints
        rewritten; the merged kg_entities row is then deleted.
        """

        with self._txn() as conn:
            sv_row = conn.execute(
                "SELECT id, mentions, degree, entity_type FROM kg_entities "
                "WHERE canonical_name = ? COLLATE NOCASE",
                (survivor_name,),
            ).fetchone()
            mg_row = conn.execute(
                "SELECT id, mentions, degree, entity_type FROM kg_entities "
                "WHERE canonical_name = ? COLLATE NOCASE",
                (merged_name,),
            ).fetchone()
            if sv_row is None or mg_row is None or sv_row["id"] == mg_row["id"]:
                return None
            sv_id, mg_id = sv_row["id"], mg_row["id"]

            new_type = sv_row["entity_type"] or mg_row["entity_type"]
            conn.execute(
                """
                UPDATE kg_entities SET
                    mentions = mentions + ?,
                    degree = MAX(degree, ?),
                    entity_type = COALESCE(NULLIF(entity_type, ''), ?),
                    last_modified_by_call_id = ?,
                    last_modified_at_round = ?,
                    last_modified_at_ts = ?
                WHERE id = ?
                """,
                (mg_row["mentions"], mg_row["degree"], new_type,
                 agent_call_id, round_number, _utc_now(), sv_id),
            )

            # Rewrite endpoints row-by-row to handle the collision case:
            # if the rewritten tuple (subj, pred, obj, source_doc) would
            # duplicate an existing row (e.g. two relations from the same
            # doc that become equal after the merge), DELETE the now-
            # redundant row instead of UPDATE. The survivor row absorbs
            # the provenance via the existing one.
            affected_rows = conn.execute(
                """
                SELECT id, subject_entity_id, predicate, object_entity_id, source_doc_id
                FROM kg_relations
                WHERE subject_entity_id = ? OR object_entity_id = ?
                """,
                (mg_id, mg_id),
            ).fetchall()
            now = _utc_now()
            affected: list[int] = []
            for r in affected_rows:
                new_subj = sv_id if r["subject_entity_id"] == mg_id else r["subject_entity_id"]
                new_obj = sv_id if r["object_entity_id"] == mg_id else r["object_entity_id"]
                # Check for collision against any other existing row with
                # the same post-rewrite tuple.
                collision = conn.execute(
                    """
                    SELECT id FROM kg_relations
                    WHERE id != ? AND subject_entity_id = ? AND predicate = ?
                      AND object_entity_id = ?
                      AND (
                        (source_doc_id IS NULL AND ? IS NULL) OR source_doc_id = ?
                      )
                    """,
                    (r["id"], new_subj, r["predicate"], new_obj,
                     r["source_doc_id"], r["source_doc_id"]),
                ).fetchone()
                if collision is not None:
                    # Redundant after merge — delete; the original row stays.
                    conn.execute("DELETE FROM kg_relations WHERE id = ?", (r["id"],))
                    affected.append(r["id"])  # still log the mutation: this id was touched
                    continue
                conn.execute(
                    """
                    UPDATE kg_relations SET
                        subject_entity_id = ?,
                        object_entity_id = ?,
                        last_modified_by_call_id = ?,
                        last_modified_at_round = ?,
                        last_modified_at_ts = ?
                    WHERE id = ?
                    """,
                    (new_subj, new_obj, agent_call_id, round_number, now, r["id"]),
                )
                affected.append(r["id"])

            # entity_chunk_refs.entity_id was pointing at mg_id; UNIQUE
            # constraint (entity_id, chunk_id, char_start, char_end) means
            # re-pointing may collide with rows already present for sv_id.
            # Use INSERT OR IGNORE + DELETE pattern.
            conn.execute(
                """
                INSERT OR IGNORE INTO entity_chunk_refs (
                    entity_id, chunk_id, char_start, char_end, confidence, agent_call_id
                )
                SELECT ?, chunk_id, char_start, char_end, confidence, agent_call_id
                FROM entity_chunk_refs WHERE entity_id = ?
                """,
                (sv_id, mg_id),
            )
            conn.execute(
                "DELETE FROM entity_chunk_refs WHERE entity_id = ?", (mg_id,)
            )

            # Record the merged_name → survivor alias so future extractions
            # naming the merged form snap to the survivor without a Resolver
            # turn. Any pre-existing alias rows pointing at mg_id (acquired
            # via earlier merges into the now-merged row) get repointed to
            # sv_id in the same transaction. Skip when agent_call_id is
            # None — entity_aliases.created_by_call_id is NOT NULL.
            if agent_call_id is not None:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_aliases (
                        entity_id, alias_text, source_chunk_id,
                        created_by_call_id, created_at_round
                    ) VALUES (?, ?, NULL, ?, ?)
                    """,
                    (sv_id, merged_name, agent_call_id, round_number),
                )
                conn.execute(
                    "UPDATE entity_aliases SET entity_id = ? WHERE entity_id = ?",
                    (sv_id, mg_id),
                )

            conn.execute(
                "DELETE FROM embeddings WHERE target_kind = 'entity' AND target_id = ?",
                (str(mg_id),),
            )

            # kg_entity_properties.entity_id was pointing at mg_id (typed-
            # statements attribute lane). UNIQUE(entity_id, property, value_raw,
            # condition, source_doc_id) means re-pointing may collide with rows
            # already on sv_id — move the non-colliding ones, then drop the
            # leftover duplicates so the entity delete's FK is satisfied.
            conn.execute(
                "UPDATE OR IGNORE kg_entity_properties SET entity_id = ? WHERE entity_id = ?",
                (sv_id, mg_id),
            )
            conn.execute(
                "DELETE FROM kg_entity_properties WHERE entity_id = ?", (mg_id,)
            )

            conn.execute("DELETE FROM kg_entities WHERE id = ?", (mg_id,))

        return sv_id, mg_id, affected

    # ── alias tables: canonicalization layer ─────────────────────────

    def record_entity_alias_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        entity_id: int,
        alias_text: str,
        source_chunk_id: int | None,
        agent_call_id: int,
        round_number: int,
    ) -> int | None:
        """INSERT OR IGNORE one entity-alias row inside an existing txn.
        Callers from inside ``apply_merge`` use this path so the alias
        insert + survivor stamping commit atomically. Returns the new row id
        or ``None`` if a row already aliases ``alias_text``."""

        cur = conn.execute(
            """
            INSERT OR IGNORE INTO entity_aliases (
                entity_id, alias_text, source_chunk_id,
                created_by_call_id, created_at_round
            ) VALUES (?, ?, ?, ?, ?)
            RETURNING id
            """,
            (entity_id, alias_text, source_chunk_id, agent_call_id, round_number),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _record_kind_alias_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        alias: str,
        canonical: str,
        agent_call_id: int,
        round_number: int,
    ) -> tuple[str, int | None]:
        """Flatten-on-insert alias write for predicate_aliases /
        entity_type_aliases (same shape, different table). Returns
        ``(effective_canonical, row_id)``; ``row_id`` is None when the
        alias was already present (first-write-wins) OR when the row
        would have been a self-loop after dereference."""

        # Step 1: dereference canonical if it's itself an alias.
        row = conn.execute(
            f"SELECT canonical FROM {table} WHERE alias = ? COLLATE NOCASE",
            (canonical,),
        ).fetchone()
        effective = row[0] if row else canonical

        # Self-loop guard: be defensive — we don't want a row claiming X → X.
        if alias.lower() == effective.lower():
            return effective, None

        cur = conn.execute(
            f"""
            INSERT OR IGNORE INTO {table} (
                alias, canonical, created_by_call_id, created_at_round
            ) VALUES (?, ?, ?, ?)
            RETURNING id
            """,
            (alias, effective, agent_call_id, round_number),
        )
        new_row = cur.fetchone()

        # Step 2: any existing rows whose canonical points at the just-inserted
        # alias must be repointed to the effective canonical. Keeps the table
        # flat (no chains).
        conn.execute(
            f"UPDATE {table} SET canonical = ? WHERE canonical = ? COLLATE NOCASE",
            (effective, alias),
        )
        return effective, (new_row[0] if new_row else None)

    def record_predicate_alias(
        self,
        *,
        alias: str,
        canonical: str,
        agent_call_id: int,
        round_number: int,
    ) -> tuple[str, int | None]:
        """Flatten-on-insert predicate alias. Returns
        ``(effective_canonical, row_id)`` — ``effective_canonical`` may
        differ from ``canonical`` if the latter was itself an alias."""

        with self._txn() as conn:
            return self._record_kind_alias_in_conn(
                conn,
                table="predicate_aliases",
                alias=alias,
                canonical=canonical,
                agent_call_id=agent_call_id,
                round_number=round_number,
            )

    def record_entity_type_alias(
        self,
        *,
        alias: str,
        canonical: str,
        agent_call_id: int,
        round_number: int,
    ) -> tuple[str, int | None]:
        """Flatten-on-insert entity-type alias. Same semantics as
        :meth:`record_predicate_alias`."""

        with self._txn() as conn:
            return self._record_kind_alias_in_conn(
                conn,
                table="entity_type_aliases",
                alias=alias,
                canonical=canonical,
                agent_call_id=agent_call_id,
                round_number=round_number,
            )

    def record_property_alias(
        self,
        *,
        alias: str,
        canonical: str,
        agent_call_id: int,
        round_number: int,
    ) -> tuple[str, int | None]:
        """Flatten-on-insert property alias (attribute lane). Same semantics as
        :meth:`record_predicate_alias`."""

        with self._txn() as conn:
            return self._record_kind_alias_in_conn(
                conn,
                table="property_aliases",
                alias=alias,
                canonical=canonical,
                agent_call_id=agent_call_id,
                round_number=round_number,
            )

    def record_schema_parent(
        self,
        *,
        child: str,
        parent: str,
        kind: str,
        agent_call_id: int,
        round_number: int,
    ) -> int | None:
        """Record a child→parent specialization edge for the rollup hierarchy.
        First-write-wins per ``(child, kind)`` via INSERT OR IGNORE; a self-loop
        (``child == parent``) is skipped. Returns the new row id, or ``None`` if
        skipped / already present."""

        if child.strip().lower() == parent.strip().lower():
            return None
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO schema_parents (
                    child, parent, kind, created_by_call_id, created_at_round
                ) VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """,
                (child, parent, kind, agent_call_id, round_number),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def resolve_entity_alias(self, name: str) -> str | None:
        """Return the survivor's canonical name if ``name`` is a recorded
        entity alias, else ``None``. Used at extraction-merge time so a
        re-surfaced merged name snaps to the survivor without a Resolver
        turn."""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT ke.canonical_name
                FROM entity_aliases ea
                JOIN kg_entities ke ON ke.id = ea.entity_id
                WHERE ea.alias_text = ? COLLATE NOCASE
                """,
                (name,),
            ).fetchone()
        return row[0] if row else None

    def resolve_predicate_alias(self, predicate: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT canonical FROM predicate_aliases WHERE alias = ? COLLATE NOCASE",
                (predicate,),
            ).fetchone()
        return row[0] if row else None

    def resolve_entity_type_alias(self, entity_type: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT canonical FROM entity_type_aliases WHERE alias = ? COLLATE NOCASE",
                (entity_type,),
            ).fetchone()
        return row[0] if row else None

    def list_predicate_aliases(self) -> list[tuple[str, str]]:
        """All (alias, canonical) pairs for predicate_aliases. Read by
        :class:`KgSchemaView` so the Mapper's candidate set + the merge
        grace-path can resolve against canonicals ∪ alias keys."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT alias, canonical FROM predicate_aliases ORDER BY id ASC"
            ).fetchall()
        return [(r["alias"], r["canonical"]) for r in rows]

    def list_entity_type_aliases(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT alias, canonical FROM entity_type_aliases ORDER BY id ASC"
            ).fetchall()
        return [(r["alias"], r["canonical"]) for r in rows]

    def resolve_property_alias(self, property_name: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT canonical FROM property_aliases WHERE alias = ? COLLATE NOCASE",
                (property_name,),
            ).fetchone()
        return row[0] if row else None

    def list_property_aliases(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT alias, canonical FROM property_aliases ORDER BY id ASC"
            ).fetchall()
        return [(r["alias"], r["canonical"]) for r in rows]

    def cached_distinct_pairs(self) -> set[str]:
        """Memoized Resolver ``distinct`` verdict keys. The Resolver
        skips re-judging any candidate pair already cached here."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT pair_key FROM resolver_verdicts WHERE verdict = 'distinct'"
            ).fetchall()
        return {r[0] for r in rows}

    def record_resolver_verdicts(
        self, pair_keys: list[str], *, round_number: int, verdict: str = "distinct"
    ) -> None:
        """Cache resolver verdicts (first-write-wins per pair_key)."""

        if not pair_keys:
            return
        with self._txn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO resolver_verdicts (pair_key, verdict, created_at_round) "
                "VALUES (?, ?, ?)",
                [(k, verdict, round_number) for k in pair_keys],
            )

    def list_schema_parents(self, kind: str) -> list[tuple[str, str]]:
        """All (child, parent) specialization edges for ``kind``
        ('relation_type' | 'property_type'), the rollup substrate the
        hypothesis layer walks for query-time predicate/property expansion."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT child, parent FROM schema_parents WHERE kind = ? ORDER BY id ASC",
                (kind,),
            ).fetchall()
        return [(r["child"], r["parent"]) for r in rows]

    def apply_retroactive_predicate_rewrite(
        self,
        *,
        alias: str,
        canonical: str,
        agent_call_id: int,
        round_number: int,
    ) -> int:
        """Rewrite every ``kg_relations`` row whose predicate matches
        ``alias`` (case-insensitive) to ``canonical``. Handles collisions
        with existing canonical rows by deleting the now-redundant alias
        row (mirrors ``apply_merge``'s collision path). Each touched row
        gets a ``kg_mutations`` row with ``kind='schema_refactor_rewrite'``.
        Returns the count of rows touched (UPDATE + DELETE).
        """

        touched = 0
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT id, subject_entity_id, object_entity_id, source_doc_id
                FROM kg_relations
                WHERE predicate = ? COLLATE NOCASE
                """,
                (alias,),
            ).fetchall()
            now = _utc_now()
            for r in rows:
                # Check for collision against an existing row that already
                # carries the canonical predicate for the same endpoints.
                collision = conn.execute(
                    """
                    SELECT id FROM kg_relations
                    WHERE id != ? AND subject_entity_id = ? AND predicate = ?
                      AND object_entity_id = ?
                      AND (
                        (source_doc_id IS NULL AND ? IS NULL) OR source_doc_id = ?
                      )
                    """,
                    (r["id"], r["subject_entity_id"], canonical,
                     r["object_entity_id"], r["source_doc_id"], r["source_doc_id"]),
                ).fetchone()
                if collision is not None:
                    conn.execute("DELETE FROM kg_relations WHERE id = ?", (r["id"],))
                    conn.execute(
                        """
                        INSERT INTO kg_mutations (
                            kind, target_kind, target_id, before_json, after_json,
                            reason, confidence, agent_call_id, round, ts
                        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                        """,
                        (
                            "schema_refactor_rewrite",
                            "relation",
                            str(r["id"]),
                            json.dumps({"predicate": alias}, ensure_ascii=False),
                            json.dumps(
                                {"predicate": canonical, "collided_into": collision[0]},
                                ensure_ascii=False,
                            ),
                            f"predicate alias {alias!r} → {canonical!r}; row deleted (collided)",
                            agent_call_id,
                            round_number,
                            now,
                        ),
                    )
                    touched += 1
                    continue
                conn.execute(
                    """
                    UPDATE kg_relations SET
                        predicate = ?,
                        last_modified_by_call_id = ?,
                        last_modified_at_round = ?,
                        last_modified_at_ts = ?
                    WHERE id = ?
                    """,
                    (canonical, agent_call_id, round_number, now, r["id"]),
                )
                conn.execute(
                    """
                    INSERT INTO kg_mutations (
                        kind, target_kind, target_id, before_json, after_json,
                        reason, confidence, agent_call_id, round, ts
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        "schema_refactor_rewrite",
                        "relation",
                        str(r["id"]),
                        json.dumps({"predicate": alias}, ensure_ascii=False),
                        json.dumps({"predicate": canonical}, ensure_ascii=False),
                        f"predicate alias {alias!r} → {canonical!r}",
                        agent_call_id,
                        round_number,
                        now,
                    ),
                )
                touched += 1
        return touched

    def apply_retroactive_entity_type_rewrite(
        self,
        *,
        alias: str,
        canonical: str,
        agent_call_id: int,
        round_number: int,
    ) -> int:
        """Rewrite every ``kg_entities`` row whose ``entity_type`` matches
        ``alias`` to ``canonical``. No UNIQUE collisions to worry about —
        ``entity_type`` is not part of any unique constraint. Each touched
        row gets a ``kg_mutations`` row with ``kind='schema_refactor_rewrite'``.
        Returns the count of rows touched.
        """

        touched = 0
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT id FROM kg_entities WHERE entity_type = ? COLLATE NOCASE",
                (alias,),
            ).fetchall()
            now = _utc_now()
            for r in rows:
                conn.execute(
                    """
                    UPDATE kg_entities SET
                        entity_type = ?,
                        last_modified_by_call_id = ?,
                        last_modified_at_round = ?,
                        last_modified_at_ts = ?
                    WHERE id = ?
                    """,
                    (canonical, agent_call_id, round_number, now, r["id"]),
                )
                conn.execute(
                    """
                    INSERT INTO kg_mutations (
                        kind, target_kind, target_id, before_json, after_json,
                        reason, confidence, agent_call_id, round, ts
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        "schema_refactor_rewrite",
                        "entity",
                        str(r["id"]),
                        json.dumps({"entity_type": alias}, ensure_ascii=False),
                        json.dumps({"entity_type": canonical}, ensure_ascii=False),
                        f"entity_type alias {alias!r} → {canonical!r}",
                        agent_call_id,
                        round_number,
                        now,
                    ),
                )
                touched += 1
        return touched

    def update_agent_call_output_summary(
        self,
        agent_call_id: int,
        *,
        output_summary: str,
    ) -> None:
        """Overwrite ``output_summary`` on an existing ``agent_calls`` row.

        Used by deterministic apply steps that create the row before the
        work completes (so child ``kg_mutations`` rows can reference the
        agent_call_id), then refresh the summary once totals are known.
        """

        with self._txn() as conn:
            conn.execute(
                "UPDATE agent_calls SET output_summary = ? WHERE id = ?",
                (output_summary, agent_call_id),
            )

    # ── reads / writes: loop_state ───────────────────────────────────

    def write_loop_state(self, corpus_state_json: str) -> None:
        """Upsert the singleton loop_state row. Called at end-of-round
        with the serialised CorpusState; resume reads via
        :meth:`read_loop_state`."""

        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO loop_state (id, corpus_state_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    corpus_state_json = excluded.corpus_state_json,
                    updated_at = excluded.updated_at
                """,
                (corpus_state_json, _utc_now()),
            )

    def read_loop_state(self) -> str | None:
        """Return the singleton ``corpus_state_json`` blob, or ``None`` if
        no row exists yet (fresh kg.db)."""

        with self._lock:
            row = self._conn.execute(
                "SELECT corpus_state_json FROM loop_state WHERE id = 1"
            ).fetchone()
        return row["corpus_state_json"] if row else None

    # ── reads / writes: round_reviews (two-phase round) ──────────────

    def record_round_metrics(
        self,
        round_: RoundHandle,
        *,
        new_entities: int,
        new_relations: int,
        docs_fetched_total: int,
        already_ingested_total: int = 0,
        candidates_surfaced_total: int = 0,
        already_seen_by_topic: dict[str, list[int]] | None = None,
    ) -> None:
        """Write the Phase-1 ``round_reviews`` row: deterministic metrics with
        the review FK columns left NULL (the reviews-pending checkpoint).

        Tokens + repairs are summed from this round's ``agent_calls`` as they
        stand *now* — i.e. before the reviews run — so ``tokens_spent_total``
        is the round's *work* cost, not the review overhead. Idempotent upsert
        that never clobbers already-stamped review FKs."""

        with self._txn() as conn:
            per_source_rows = conn.execute(
                """
                SELECT source, COUNT(*) AS n FROM fetched_docs
                WHERE round_id = ? AND outcome = 'selected_and_fetched'
                GROUP BY source
                """,
                (round_.round_id,),
            ).fetchall()
            per_source = {r["source"]: r["n"] for r in per_source_rows}
            agg = conn.execute(
                """
                SELECT COALESCE(SUM(COALESCE(prompt_tokens, 0)
                                  + COALESCE(completion_tokens, 0)), 0) AS tokens,
                       COALESCE(SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END), 0)
                           AS repairs
                FROM agent_calls WHERE round_id = ?
                """,
                (round_.round_id,),
            ).fetchone()
            tokens_total = int(agg["tokens"])
            repairs = int(agg["repairs"])
            started = conn.execute(
                "SELECT started_at FROM rounds WHERE id = ?", (round_.round_id,)
            ).fetchone()
            round_started_at = started["started_at"] if started else _utc_now()
            conn.execute(
                """
                INSERT INTO round_reviews (
                    round_id, round_number, new_entities, new_relations,
                    docs_fetched_total, docs_fetched_per_source,
                    tokens_spent_total, tokens_per_new_entity,
                    tokens_per_new_relation, consolidation_repairs,
                    already_ingested_total, candidates_surfaced_total,
                    already_seen_by_topic, round_started_at, round_ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (round_id) DO UPDATE SET
                    new_entities = excluded.new_entities,
                    new_relations = excluded.new_relations,
                    docs_fetched_total = excluded.docs_fetched_total,
                    docs_fetched_per_source = excluded.docs_fetched_per_source,
                    tokens_spent_total = excluded.tokens_spent_total,
                    tokens_per_new_entity = excluded.tokens_per_new_entity,
                    tokens_per_new_relation = excluded.tokens_per_new_relation,
                    consolidation_repairs = excluded.consolidation_repairs,
                    already_ingested_total = excluded.already_ingested_total,
                    candidates_surfaced_total = excluded.candidates_surfaced_total,
                    already_seen_by_topic = excluded.already_seen_by_topic,
                    round_ended_at = excluded.round_ended_at
                """,
                (
                    round_.round_id,
                    round_.round_number,
                    new_entities,
                    new_relations,
                    docs_fetched_total,
                    json.dumps(per_source, ensure_ascii=False),
                    tokens_total,
                    (tokens_total / new_entities) if new_entities > 0 else None,
                    (tokens_total / new_relations) if new_relations > 0 else None,
                    repairs,
                    already_ingested_total,
                    candidates_surfaced_total,
                    json.dumps(already_seen_by_topic or {}, ensure_ascii=False),
                    round_started_at,
                    _utc_now(),
                ),
            )

    def stamp_round_reviews(
        self,
        round_: RoundHandle,
        *,
        kg_review_call_id: int | None,
        round_review_call_id: int | None,
    ) -> None:
        """Phase-2: attach the two review ``agent_calls`` to the round_reviews
        row. Stamping ``kg_review_call_id`` clears the reviews-pending marker."""

        with self._txn() as conn:
            conn.execute(
                """
                UPDATE round_reviews SET
                    kg_review_call_id = ?, round_review_call_id = ?
                WHERE round_id = ?
                """,
                (kg_review_call_id, round_review_call_id, round_.round_id),
            )

    def prime_round_sequence(self, round_: RoundHandle) -> None:
        """Seed the in-memory agent_call sequence counter for a round from the
        DB. Used on resume, when re-running Phase 2 for a round whose Phase-1
        calls already exist, so the new review calls don't reuse sequence
        numbers (the counter is reset on a fresh store open)."""

        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_in_round), 0) AS m "
                "FROM agent_calls WHERE round_id = ?",
                (round_.round_id,),
            ).fetchone()
            self._sequence_by_round[round_.round_id] = int(row["m"])

    def read_pending_round_review(self) -> dict[str, Any] | None:
        """The round_reviews row whose reviews never completed
        (``kg_review_call_id IS NULL``), or None. Non-None on resume means
        the KG committed but Phase 2 crashed → re-run reviews only."""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT round_id, round_number, new_entities, new_relations,
                       docs_fetched_total
                FROM round_reviews
                WHERE kg_review_call_id IS NULL
                ORDER BY round_number DESC LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def read_recent_round_metrics(self, limit: int) -> list[dict[str, Any]]:
        """Last ``limit`` round_reviews rows, oldest→newest, for trend
        derivation + the RoundReviewer prompt."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT round_number, new_entities, new_relations,
                       docs_fetched_total, docs_fetched_per_source,
                       tokens_spent_total, tokens_per_new_entity,
                       tokens_per_new_relation, consolidation_repairs,
                       already_ingested_total, candidates_surfaced_total,
                       already_seen_by_topic
                FROM round_reviews
                ORDER BY round_number DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            surfaced = d.get("candidates_surfaced_total") or 0
            d["already_ingested_rate"] = (
                (d.get("already_ingested_total") or 0) / surfaced if surfaced else None
            )
            try:
                d["already_seen_by_topic"] = json.loads(d.get("already_seen_by_topic") or "{}")
            except (TypeError, ValueError):
                d["already_seen_by_topic"] = {}
            out.append(d)
        return out

    def read_latest_reviews(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """``(kg_review_json, round_review_json)`` — the parsed outputs of the
        two review calls from the most recent fully-reviewed round. Either is
        None when the call or its parsed output is missing. The next round's
        RoundPlanner reads prior gaps + directives through here (DB-backed, so
        it survives resume)."""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT kg_review_call_id, round_review_call_id
                FROM round_reviews
                WHERE kg_review_call_id IS NOT NULL
                ORDER BY round_number DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None, None

            def _parsed(call_id: int | None) -> dict[str, Any] | None:
                if call_id is None:
                    return None
                r = self._conn.execute(
                    "SELECT parsed_output_json FROM agent_call_details "
                    "WHERE agent_call_id = ?",
                    (call_id,),
                ).fetchone()
                if r is None or r["parsed_output_json"] is None:
                    return None
                return json.loads(r["parsed_output_json"])

            return _parsed(row["kg_review_call_id"]), _parsed(
                row["round_review_call_id"]
            )

    def list_round_reviews(self, build_id: int) -> list[dict[str, Any]]:
        """Read-side: per-round metrics joined with the RoundReviewer's
        ``status`` / ``reason`` for a build. Powers the inspector. Trends are derived by the caller
        from these rows."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT rr.round_number, rr.new_entities, rr.new_relations,
                       rr.docs_fetched_total, rr.docs_fetched_per_source,
                       rr.tokens_spent_total, rr.tokens_per_new_entity,
                       rr.tokens_per_new_relation, rr.consolidation_repairs,
                       d.parsed_output_json AS review_json
                FROM round_reviews rr
                JOIN rounds r ON r.id = rr.round_id
                LEFT JOIN agent_call_details d
                    ON d.agent_call_id = rr.round_review_call_id
                WHERE r.build_id = ?
                ORDER BY rr.round_number
                """,
                (build_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            rec = dict(row)
            review_json = rec.pop("review_json", None)
            status = reason = None
            if review_json:
                try:
                    parsed = json.loads(review_json)
                    status = parsed.get("status")
                    reason = parsed.get("reason")
                except (json.JSONDecodeError, TypeError):
                    pass
            rec["status"] = status
            rec["reason"] = reason
            out.append(rec)
        return out

    # ── reads: for the inspector + UI ────────────────────────────────

    def list_builds(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, build_run_id, kg_id, base_query, sample_mode,
                       started_at, finished_at, stop_reason,
                       rounds_completed, total_docs, total_entities,
                       total_relations, schema_entity_types,
                       schema_relation_types, total_prompt_tokens,
                       total_completion_tokens
                FROM builds
                ORDER BY started_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_build(self, build_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM builds WHERE id = ?", (build_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_rounds(self, build_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM rounds WHERE build_id = ?
                ORDER BY round_number ASC
                """,
                (build_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_agent_calls(self, round_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, parent_call_id, triggered_by, sequence_in_round,
                       agent_name, action, target_agent, started_at,
                       finished_at, duration_ms, status, retry_count,
                       confidence, model_name, input_summary, output_summary,
                       prompt_tokens, completion_tokens, error_type,
                       error_message
                FROM agent_calls
                WHERE round_id = ?
                ORDER BY sequence_in_round ASC
                """,
                (round_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_call(self, agent_call_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT ac.*, acd.system_prompt, acd.user_prompt,
                       acd.raw_response, acd.parsed_output_json
                FROM agent_calls ac
                LEFT JOIN agent_call_details acd
                    ON acd.agent_call_id = ac.id
                WHERE ac.id = ?
                """,
                (agent_call_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_fetched_docs(self, round_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM fetched_docs WHERE round_id = ?
                ORDER BY id ASC
                """,
                (round_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_outcome_summary(self) -> dict[str, Any]:
        """Build-level rollup of every ``fetched_docs`` outcome (degrade
        visibility). Returns counts by outcome + ``fetch_failed`` broken out by
        source, so an unattended build's degradation surfaces in one
        place instead of only as scattered per-round log lines."""
        with self._lock:
            by_outcome = {
                r["outcome"]: r["n"]
                for r in self._conn.execute(
                    "SELECT outcome, COUNT(*) AS n FROM fetched_docs GROUP BY outcome"
                ).fetchall()
            }
            failed_by_source = {
                r["source"]: r["n"]
                for r in self._conn.execute(
                    "SELECT source, COUNT(*) AS n FROM fetched_docs "
                    "WHERE outcome = 'fetch_failed' GROUP BY source ORDER BY n DESC"
                ).fetchall()
            }
        fetched = by_outcome.get("selected_and_fetched", 0)
        failed = by_outcome.get("fetch_failed", 0)
        attempted = fetched + failed
        return {
            "by_outcome": by_outcome,
            "fetch_failed_by_source": failed_by_source,
            "fetched": fetched,
            "fetch_failed": failed,
            "fetch_failed_rate": round(failed / attempted, 3) if attempted else 0.0,
        }

    def list_schema_changes(self, build_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT sc.*, r.round_number
                FROM schema_changes sc
                JOIN rounds r ON r.id = sc.round_id
                WHERE r.build_id = ?
                ORDER BY r.round_number ASC, sc.id ASC
                """,
                (build_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_failures(self, build_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ac.id, ac.agent_name, ac.action, ac.status,
                       ac.error_type, ac.error_message, ac.started_at,
                       r.round_number
                FROM agent_calls ac
                JOIN rounds r ON r.id = ac.round_id
                WHERE r.build_id = ?
                  AND ac.status IN ('failed', 'sample_mode')
                ORDER BY ac.started_at DESC
                LIMIT ?
                """,
                (build_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def graph_data(
        self,
        *,
        min_mentions: int = 1,
        entity_type: str | None = None,
        predicate: str | None = None,
        limit_nodes: int = 500,
        include_isolated: bool = False,
    ) -> dict[str, Any]:
        """Build a vis-network–shaped node + edge payload for the graph view.

        Edge rows in ``kg_relations`` carry one row per provenance
        ``(subj, pred, obj, source_doc_id)``. The UI wants one edge per
        ``(subj, pred, obj)`` with a count of supporting docs — collapsing
        parallel provenance edges into a single weighted edge. We aggregate
        here so the client doesn't have to.

        Filters:
        * ``min_mentions`` — drop entities below mention threshold (noise
          control on big graphs).
        * ``entity_type`` — restrict to one entity type.
        * ``predicate`` — restrict edges to one predicate.
        * ``limit_nodes`` — cap top-N by mentions to keep vis-network
          responsive (~500 is its sweet spot).
        * ``include_isolated`` — when False (default), drop entities that
          end up with no surviving edges after predicate-filtering.
        """

        with self._lock:
            ent_rows = self._conn.execute(
                """
                SELECT id, canonical_name, entity_type, mentions, degree,
                       first_seen_round, last_seen_round
                FROM kg_entities
                WHERE mentions >= ?
                  AND (? IS NULL OR entity_type = ?)
                ORDER BY mentions DESC, degree DESC
                LIMIT ?
                """,
                (min_mentions, entity_type, entity_type, limit_nodes),
            ).fetchall()
            entities = [dict(r) for r in ent_rows]
            entity_ids = [e["id"] for e in entities]
            if not entity_ids:
                return {"nodes": [], "edges": [], "entity_types": []}

            placeholders = ",".join("?" * len(entity_ids))
            edge_rows = self._conn.execute(
                f"""
                SELECT subject_entity_id AS source,
                       object_entity_id  AS target,
                       predicate,
                       MIN(round_observed) AS first_round,
                       COUNT(*)            AS provenance_count,
                       MIN(source_doc_id)  AS sample_doc_id,
                       MIN(agent_call_id)  AS sample_agent_call_id
                FROM kg_relations
                WHERE subject_entity_id IN ({placeholders})
                  AND object_entity_id  IN ({placeholders})
                  AND (? IS NULL OR predicate = ?)
                GROUP BY subject_entity_id, object_entity_id, predicate
                ORDER BY provenance_count DESC
                """,
                (*entity_ids, *entity_ids, predicate, predicate),
            ).fetchall()
            edges = [dict(r) for r in edge_rows]

            type_rows = self._conn.execute(
                """
                SELECT entity_type, COUNT(*) AS n
                FROM kg_entities
                GROUP BY entity_type
                ORDER BY n DESC
                """
            ).fetchall()
            entity_types = [dict(r) for r in type_rows]

        if not include_isolated:
            connected = {e["source"] for e in edges} | {e["target"] for e in edges}
            entities = [e for e in entities if e["id"] in connected]

        nodes = [
            {
                "id": e["id"],
                "label": e["canonical_name"],
                "group": e["entity_type"] or "other",
                "value": e["mentions"],
                "title": (
                    f"{e['canonical_name']}\n"
                    f"type: {e['entity_type'] or '—'}\n"
                    f"mentions: {e['mentions']} · degree: {e['degree']}\n"
                    f"rounds: {e['first_seen_round']}–{e['last_seen_round']}"
                ),
            }
            for e in entities
        ]
        edges_out = [
            {
                "from": e["source"],
                "to": e["target"],
                "label": e["predicate"],
                "value": e["provenance_count"],
                "title": (
                    f"{e['predicate']}\n"
                    f"docs supporting: {e['provenance_count']}\n"
                    f"first round: {e['first_round']}"
                ),
                "sample_doc_id": e["sample_doc_id"],
                "sample_agent_call_id": e["sample_agent_call_id"],
            }
            for e in edges
        ]
        return {
            "nodes": nodes,
            "edges": edges_out,
            "entity_types": entity_types,
        }

    def entity_detail(self, entity_id: int) -> dict[str, Any] | None:
        """Full entity drill-down for the right pane: type, mention/degree,
        source docs, top outgoing + incoming relations, originating
        agent_calls. Used by the graph view's node-click handler."""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, canonical_name, entity_type, mentions, degree,
                       first_seen_round, last_seen_round
                FROM kg_entities
                WHERE id = ?
                """,
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            entity = dict(row)

            outgoing = self._conn.execute(
                """
                SELECT kr.predicate, kr.object_entity_id, ke.canonical_name AS object_name,
                       COUNT(*) AS provenance_count
                FROM kg_relations kr
                JOIN kg_entities ke ON ke.id = kr.object_entity_id
                WHERE kr.subject_entity_id = ?
                GROUP BY kr.predicate, kr.object_entity_id, ke.canonical_name
                ORDER BY provenance_count DESC
                LIMIT 25
                """,
                (entity_id,),
            ).fetchall()

            incoming = self._conn.execute(
                """
                SELECT kr.predicate, kr.subject_entity_id, ke.canonical_name AS subject_name,
                       COUNT(*) AS provenance_count
                FROM kg_relations kr
                JOIN kg_entities ke ON ke.id = kr.subject_entity_id
                WHERE kr.object_entity_id = ?
                GROUP BY kr.predicate, kr.subject_entity_id, ke.canonical_name
                ORDER BY provenance_count DESC
                LIMIT 25
                """,
                (entity_id,),
            ).fetchall()

            doc_ids = self._conn.execute(
                """
                SELECT DISTINCT source_doc_id
                FROM kg_relations
                WHERE (subject_entity_id = ? OR object_entity_id = ?)
                  AND source_doc_id IS NOT NULL
                LIMIT 25
                """,
                (entity_id, entity_id),
            ).fetchall()

            # Attribute lane: one row per distinct
            # (property, value, unit, condition), parallel provenance collapsed.
            properties = self._conn.execute(
                """
                SELECT property, value_num, value_raw, unit, condition,
                       COUNT(*)            AS provenance_count,
                       AVG(mapper_confidence) AS avg_confidence,
                       MIN(round_observed)    AS first_round
                FROM kg_entity_properties
                WHERE entity_id = ?
                GROUP BY property, value_raw, unit, condition
                ORDER BY provenance_count DESC, property ASC
                LIMIT 50
                """,
                (entity_id,),
            ).fetchall()

        entity["outgoing"] = [dict(r) for r in outgoing]
        entity["incoming"] = [dict(r) for r in incoming]
        entity["source_doc_ids"] = [r[0] for r in doc_ids]
        entity["properties"] = [dict(r) for r in properties]
        return entity

    def relation_detail(
        self, subject_entity_id: int, object_entity_id: int, predicate: str
    ) -> dict[str, Any] | None:
        """Drill-down for one graph edge: the ``(subject, predicate, object)``
        triple the UI collapses parallel provenance rows into. Returns the
        endpoint entities, an avg-confidence/count aggregate, every supporting
        ``kg_relations`` row (one per source doc), and the merged mutation
        history across those rows. Used by the graph view's edge-click handler.
        Returns None when the triple has no surviving rows."""

        with self._lock:
            prov_rows = self._conn.execute(
                """
                SELECT id, source_doc_id, source_chunk_id, round_observed,
                       agent_call_id, mapper_call_id, raw_subject, raw_predicate,
                       raw_object, mapper_confidence, newness_score, mapped_flag
                FROM kg_relations
                WHERE subject_entity_id = ?
                  AND object_entity_id  = ?
                  AND predicate = ?
                ORDER BY round_observed ASC, id ASC
                """,
                (subject_entity_id, object_entity_id, predicate),
            ).fetchall()
            if not prov_rows:
                return None
            provenance = [dict(r) for r in prov_rows]

            subj = self._conn.execute(
                "SELECT id, canonical_name, entity_type FROM kg_entities WHERE id = ?",
                (subject_entity_id,),
            ).fetchone()
            obj = self._conn.execute(
                "SELECT id, canonical_name, entity_type FROM kg_entities WHERE id = ?",
                (object_entity_id,),
            ).fetchone()

            relation_ids = [str(p["id"]) for p in provenance]
            placeholders = ",".join("?" * len(relation_ids))
            mut_rows = self._conn.execute(
                f"""
                SELECT id, kind, target_kind, target_id, before_json,
                       after_json, reason, confidence, agent_call_id,
                       parent_mutation_id, round, ts
                FROM kg_mutations
                WHERE target_kind = 'relation' AND target_id IN ({placeholders})
                ORDER BY id ASC
                """,
                relation_ids,
            ).fetchall()
            mutations = [dict(r) for r in mut_rows]

        confs = [p["mapper_confidence"] for p in provenance if p["mapper_confidence"] is not None]
        rounds = [p["round_observed"] for p in provenance if p["round_observed"] is not None]
        return {
            "predicate": predicate,
            "subject": dict(subj) if subj else {"id": subject_entity_id},
            "object": dict(obj) if obj else {"id": object_entity_id},
            "provenance_count": len(provenance),
            "avg_confidence": (sum(confs) / len(confs)) if confs else None,
            "first_round": min(rounds) if rounds else None,
            "last_round": max(rounds) if rounds else None,
            "provenance": provenance,
            "mutations": mutations,
        }

    def list_mutations_for_target(
        self,
        *,
        target_kind: str,
        target_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Per-row drill-down (lens 1): every mutation that touched one
        entity/relation/predicate, oldest first."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, kind, target_kind, target_id, before_json,
                       after_json, reason, confidence, agent_call_id,
                       parent_mutation_id, round, ts
                FROM kg_mutations
                WHERE target_kind = ? AND target_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (target_kind, str(target_id), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mutation_counts_by_round(self) -> list[dict[str, Any]]:
        """Overview (lens 2): mutation count per round, broken out by kind
        so the UI can stack the bars. Counts are cumulative across
        invocations (the graph is shared by every build of a kg_id)."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT round, kind, COUNT(*) AS n
                FROM kg_mutations
                WHERE round IS NOT NULL
                GROUP BY round, kind
                ORDER BY round ASC, kind ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def list_mutations_for_call(
        self, agent_call_id: int
    ) -> list[dict[str, Any]]:
        """Per-agent-call view (lens 3): every mutation written by one call.

        Parent merge rows appear before their endpoint-rewrite children
        because the parent row is always inserted first.
        """

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, kind, target_kind, target_id, before_json,
                       after_json, reason, confidence, agent_call_id,
                       parent_mutation_id, round, ts
                FROM kg_mutations
                WHERE agent_call_id = ?
                ORDER BY id ASC
                """,
                (agent_call_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def top_entities(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT canonical_name, entity_type, mentions, degree,
                       first_seen_round, last_seen_round
                FROM kg_entities
                ORDER BY mentions DESC, degree DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def _row_iter(cur: sqlite3.Cursor) -> Iterable[dict[str, Any]]:
    for row in cur:
        yield dict(row)


def _lookup_entity_id(conn: sqlite3.Connection, name: str) -> int | None:
    """Case-insensitive name → kg_entities.id lookup."""

    row = conn.execute(
        "SELECT id FROM kg_entities WHERE canonical_name = ? COLLATE NOCASE",
        (name,),
    ).fetchone()
    return row[0] if row else None


def _resolve_entity_alias_in_conn(
    conn: sqlite3.Connection, name: str
) -> str | None:
    """Case-insensitive entity-alias → survivor canonical_name lookup
    bound to an already-open transaction. Mirrors
    :meth:`TraceStore.resolve_entity_alias` but reuses ``conn`` so callers
    inside ``merge_extraction`` / ``apply_merge`` see writes from the same
    transaction."""

    row = conn.execute(
        """
        SELECT ke.canonical_name
        FROM entity_aliases ea
        JOIN kg_entities ke ON ke.id = ea.entity_id
        WHERE ea.alias_text = ? COLLATE NOCASE
        """,
        (name,),
    ).fetchone()
    return row[0] if row else None


def _resolve_kind_alias_in_conn(
    conn: sqlite3.Connection, table: str, value: str
) -> str | None:
    """Generic alias → canonical lookup against ``predicate_aliases`` /
    ``entity_type_aliases`` from inside an existing transaction."""

    row = conn.execute(
        f"SELECT canonical FROM {table} WHERE alias = ? COLLATE NOCASE",
        (value,),
    ).fetchone()
    return row[0] if row else None
