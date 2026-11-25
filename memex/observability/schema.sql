-- memex — per-build canonical KG + trace DB schema
--
-- One SQLite file per build at `kg/<kg_id>/kg.db`.
-- Shape:
--   * agent call graph     → agent_calls.parent_call_id + triggered_by + no
--                            uniqueness on (round, agent)
--   * chunked extraction   → chunks table populated by the chunker
--                            (paragraph-aware, ~2000 chars + 200 overlap);
--                            extractions.chunk_id + kg_relations.source_chunk_id
--                            carry per-chunk provenance. Reconciler-emitted
--                            relations have source_chunk_id=NULL (doc-level
--                            synthesis).
--   * entity ↔ chunk refs  → entity_chunk_refs populated per-chunk during
--                            extraction; char_start/end sentinel 0,0 for
--                            "whole-chunk mention" (idempotent re-write via
--                            UNIQUE + INSERT OR IGNORE).
--   * embeddings           → embeddings table stores inline float32 BLOBs
--                            (chunk, entity, predicate) via the
--                            /v1/embeddings API. Consumed by Resolver tier-3
--                            (entity similarity) and the Mapper's
--                            above-cutoff candidate retrieval (predicate
--                            similarity).
--
-- Status enum on agent_calls is deliberate:
--   success      — real LLM call returned a valid parsed result
--   sample_mode  — SAMPLE_MODE=true short-circuited the call (intentional,
--                  not a fallback from error)
--   failed       — LLM call raised; sample factory served the result as
--                  graceful degradation so the loop keeps moving
--   retried      — succeeded after one or more retries (retry_count > 0)
--   skipped      — agent body decided to short-circuit (e.g. the Mapper on an
--                  entity-only doc with no triples to canonicalize)

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS builds (
  id INTEGER PRIMARY KEY,
  build_run_id TEXT UNIQUE NOT NULL,
  kg_id TEXT NOT NULL,
  base_query TEXT NOT NULL,
  config_json TEXT NOT NULL,
  sources_json TEXT NOT NULL,
  sample_mode INTEGER NOT NULL,
  resume_from_round INTEGER,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  stop_reason TEXT,
  rounds_completed INTEGER NOT NULL DEFAULT 0,
  total_docs INTEGER NOT NULL DEFAULT 0,
  total_entities INTEGER NOT NULL DEFAULT 0,
  total_relations INTEGER NOT NULL DEFAULT 0,
  schema_entity_types INTEGER NOT NULL DEFAULT 0,
  schema_relation_types INTEGER NOT NULL DEFAULT 0,
  total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
  total_completion_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rounds (
  id INTEGER PRIMARY KEY,
  build_id INTEGER NOT NULL REFERENCES builds(id) ON DELETE CASCADE,
  round_number INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  duration_seconds REAL,
  docs_fetched INTEGER NOT NULL DEFAULT 0,
  docs_skipped INTEGER NOT NULL DEFAULT 0,
  docs_failed INTEGER NOT NULL DEFAULT 0,
  entities_before INTEGER NOT NULL,
  entities_after INTEGER,
  relations_before INTEGER NOT NULL,
  relations_after INTEGER,
  schema_entity_types_before INTEGER NOT NULL,
  schema_entity_types_after INTEGER,
  schema_relation_types_before INTEGER NOT NULL,
  schema_relation_types_after INTEGER,
  stop_criterion_fired TEXT,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  UNIQUE (build_id, round_number)
);

CREATE TABLE IF NOT EXISTS agent_calls (
  id INTEGER PRIMARY KEY,
  round_id INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
  parent_call_id INTEGER REFERENCES agent_calls(id),
  triggered_by TEXT NOT NULL,
  sequence_in_round INTEGER NOT NULL,
  agent_name TEXT NOT NULL,
  action TEXT NOT NULL,
  target_agent TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  duration_ms INTEGER,
  status TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  confidence REAL,
  model_name TEXT,
  input_summary TEXT NOT NULL,
  output_summary TEXT NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cached_tokens INTEGER,
  error_type TEXT,
  error_message TEXT
);

CREATE TABLE IF NOT EXISTS agent_call_details (
  agent_call_id INTEGER PRIMARY KEY REFERENCES agent_calls(id) ON DELETE CASCADE,
  system_prompt TEXT,
  user_prompt TEXT,
  raw_response TEXT,
  parsed_output_json TEXT
);

CREATE TABLE IF NOT EXISTS fetched_docs (
  id INTEGER PRIMARY KEY,
  round_id INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
  doc_id TEXT,
  source TEXT NOT NULL,
  url TEXT,
  title TEXT,
  query TEXT,
  ranker_position INTEGER,
  ranker_score REAL,
  outcome TEXT NOT NULL,
  outcome_detail TEXT,
  parse_method TEXT,
  cleaned_text_chars INTEGER,
  openalex_enriched INTEGER NOT NULL DEFAULT 0
);

-- documents: the canonical per-doc record. One row per doc_id (cumulative
-- across rounds/builds, like the kg-data tables). Holds the ``metadata``
-- (a serialized DocMetadata: title, authors, query provenance, excerpt,
-- ingestion + OpenAlex metadata) AND the full cleaned ``text``. Chunks are
-- offset-views into ``text`` (chunk text = text[char_start:char_end]); the
-- embed step + downstream citation / hypothesis consumers read here.
-- kg.db is the single canonical store — no per-doc files on disk.
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  metadata TEXT NOT NULL,
  text TEXT NOT NULL,
  chars INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

-- doc_assessments: the UsefulnessGate's drop-but-persist record. One row per
-- fetched doc per round (INCLUDING drops): the whole-doc usefulness verdict
-- (score + keep/skip + reason) plus, for keepers, the durable summary the
-- gate's second pass produced. Drops are kept so what the gate throws away can
-- be inspected; the summary grounds downstream chat / report consumers.
CREATE TABLE IF NOT EXISTS doc_assessments (
  id INTEGER PRIMARY KEY,
  round_id INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
  doc_id TEXT NOT NULL,
  source TEXT NOT NULL,
  usefulness REAL NOT NULL,
  decision TEXT NOT NULL,
  reason TEXT,
  summary TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  fetched_doc_id INTEGER NOT NULL REFERENCES fetched_docs(id) ON DELETE CASCADE,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  text_chars INTEGER NOT NULL,
  token_estimate INTEGER,
  UNIQUE (fetched_doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('chunk', 'entity', 'predicate', 'relation', 'attribute', 'property')),
  target_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (target_kind, target_id, provider, model)
);
CREATE INDEX IF NOT EXISTS ix_embeddings_target
  ON embeddings (target_kind, target_id);

CREATE TABLE IF NOT EXISTS extractions (
  id INTEGER PRIMARY KEY,
  agent_call_id INTEGER NOT NULL REFERENCES agent_calls(id) ON DELETE CASCADE,
  chunk_id INTEGER REFERENCES chunks(id),
  doc_id TEXT NOT NULL,
  entities_extracted INTEGER NOT NULL,
  relations_extracted INTEGER NOT NULL,
  relations_dropped_dangling INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_changes (
  id INTEGER PRIMARY KEY,
  round_id INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
  agent_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  event TEXT NOT NULL,
  type_kind TEXT NOT NULL,
  name TEXT NOT NULL,
  rationale TEXT,
  decision_reason TEXT
);

-- kg-data tables (kg_entities, kg_relations, kg_mutations,
-- entity_chunk_refs) are cumulative across run_pipeline invocations:
-- one kg.db file holds the canonical KG, regardless of how many times
-- the loop has been resumed/extended. Per-invocation lifecycle lives in
-- builds / rounds / agent_calls (FK-chain scoped); the kg-data rows
-- carry an agent_call_id stamp which transitively links them to a
-- builds row but does not denormalise build_id onto every kg-data row.
CREATE TABLE IF NOT EXISTS kg_entities (
  id INTEGER PRIMARY KEY,
  canonical_name TEXT NOT NULL,
  entity_type TEXT,
  mentions INTEGER NOT NULL DEFAULT 0,
  degree INTEGER NOT NULL DEFAULT 0,
  first_seen_round INTEGER,
  last_seen_round INTEGER,
  -- Provenance: every row is created by exactly one agent_calls row
  -- (Extractor, Reconciler, or a skipped row for deterministic paths).
  -- Modifier stays nullable for rows never modified after creation.
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  last_modified_by_call_id INTEGER REFERENCES agent_calls(id),
  last_modified_at_round INTEGER,
  last_modified_at_ts TEXT,
  -- COLLATE NOCASE so case-different variants ('Iridium' vs 'iridium')
  -- collapse on insert. The Resolver tier-1 normalisation handles
  -- punctuation + plural variants; the SQL unique handles raw case.
  UNIQUE (canonical_name COLLATE NOCASE)
);

CREATE TABLE IF NOT EXISTS kg_relations (
  id INTEGER PRIMARY KEY,
  subject_entity_id INTEGER NOT NULL REFERENCES kg_entities(id),
  predicate TEXT NOT NULL,
  object_entity_id INTEGER NOT NULL REFERENCES kg_entities(id),
  source_doc_id TEXT,
  source_chunk_id INTEGER REFERENCES chunks(id),
  round_observed INTEGER,
  agent_call_id INTEGER REFERENCES agent_calls(id),
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  last_modified_by_call_id INTEGER REFERENCES agent_calls(id),
  last_modified_at_round INTEGER,
  last_modified_at_ts TEXT,
  -- Mapper columns. The per-doc Mapper canonicalizes the
  -- extracted vocabulary at ingest: ``predicate`` holds its chosen canonical
  -- (or the raw surface form when ``mapped_flag=0`` — Mapper failed / unmapped,
  -- a backfill cache). ``raw_*`` retain the Extractor's original surface forms.
  -- ``mapper_confidence`` is per-observation (is THIS triple supported by the
  -- source?); edge trust is computed at read time by AVG over the per-source
  -- rows, never confidence-stacked. ``newness_score`` is 0.0 (mapped to an
  -- existing canonical) … 1.0 (net-new). All nullable so deterministic (Reconciler)
  -- paths stay valid.
  mapper_call_id INTEGER REFERENCES agent_calls(id),
  raw_subject TEXT,
  raw_predicate TEXT,
  raw_object TEXT,
  -- the verbatim source sentence this relation was drawn from (Extractor-
  -- emitted; '' until the prompt populates it). A deterministic locator resolves
  -- it to a precise char span within source_chunk_id at query time (degrades to
  -- the whole chunk). NOT an LLM-emitted char offset — store the quote, locate it.
  source_quote TEXT NOT NULL DEFAULT '',
  mapper_confidence REAL,
  newness_score REAL,
  mapped_flag INTEGER NOT NULL DEFAULT 0,
  -- Re-inserts of the same (subj, pred, obj, source_doc) tuple are a
  -- no-op via INSERT OR IGNORE. source_doc_id is part of the key so the
  -- same relation from two different docs persists as two rows
  -- (preserves provenance). The key is on the CANONICAL predicate, NOT
  -- mapper_call_id: a re-mapped re-ingest must collapse onto the same row,
  -- and confidence is per-row (per observation), aggregated at read time.
  UNIQUE (subject_entity_id, predicate, object_entity_id, source_doc_id)
);

-- kg_entity_properties: the ATTRIBUTE lane. A statement whose
-- object is a VALUE (not an entity) lands here instead of kg_relations — e.g.
-- "AISI Ni42-300 / overpotential / 491 / mV / 0.1 M KOH pH 7". The per-doc Mapper
-- canonicalizes ``property`` (its chosen canonical property name, or the raw
-- surface form when ``mapped_flag=0``) and parses ``value_num`` (the comparable
-- numeric, NULL when non-numeric) out of ``value_raw`` (the preserved surface).
-- ``unit`` / ``condition`` default '' so the UNIQUE dedups re-ingests. Scores are
-- per-observation; edge/property trust is aggregated at read time, never stacked.
-- Mirrors kg_relations' provenance + mapper columns so the same contracts hold.
CREATE TABLE IF NOT EXISTS kg_entity_properties (
  id INTEGER PRIMARY KEY,
  entity_id INTEGER NOT NULL REFERENCES kg_entities(id),
  property TEXT NOT NULL,
  value_num REAL,
  value_raw TEXT NOT NULL,
  unit TEXT NOT NULL DEFAULT '',
  condition TEXT NOT NULL DEFAULT '',
  raw_property TEXT,
  raw_value TEXT,
  -- verbatim source sentence this attribute was drawn from (see kg_relations).
  source_quote TEXT NOT NULL DEFAULT '',
  source_doc_id TEXT,
  source_chunk_id INTEGER REFERENCES chunks(id),
  round_observed INTEGER,
  mapper_call_id INTEGER REFERENCES agent_calls(id),
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  mapper_confidence REAL,
  newness_score REAL,
  mapped_flag INTEGER NOT NULL DEFAULT 0,
  last_modified_at_round INTEGER,
  last_modified_at_ts TEXT,
  UNIQUE (entity_id, property, value_raw, condition, source_doc_id)
);

-- kg_mutations: append-only log of every change to entities, relations, and
-- predicates after their initial creation. The mutation log makes the UI
-- questions "what did this Resolver call do this round?" and "show every
-- edit to entity X across the build" cheap.
--
-- target_kind ∈ {"entity", "relation", "predicate"}; target_id is the row
-- id for entity/relation, or the predicate name string for predicate-level
-- mutations (rename / drop). confidence is nullable (LLM-driven mutations
-- populate it; deterministic ones leave NULL). parent_mutation_id links
-- endpoint-rewrite rows to the parent merge row so one logical merge stays
-- groupable across its fan-out.
CREATE TABLE IF NOT EXISTS kg_mutations (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  target_kind TEXT NOT NULL,
  target_id TEXT NOT NULL,
  before_json TEXT,
  after_json TEXT,
  reason TEXT,
  confidence REAL,
  agent_call_id INTEGER REFERENCES agent_calls(id),
  parent_mutation_id INTEGER REFERENCES kg_mutations(id),
  round INTEGER,
  ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_chunk_refs (
  id INTEGER PRIMARY KEY,
  entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  char_start INTEGER,
  char_end INTEGER,
  confidence REAL,
  agent_call_id INTEGER REFERENCES agent_calls(id),
  UNIQUE (entity_id, chunk_id, char_start, char_end)
);

-- kg_schema: canonical entity/relation type vocabulary. One row per admitted
-- canonical; the Mapper gate writes rows when it admits a net-new
-- canonical the per-doc Mapper proposed. Mutation log references type-level
-- edits (rename/drop) via target_kind='predicate'.
CREATE TABLE IF NOT EXISTS kg_schema (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('entity_type', 'relation_type', 'property_type')),
  added_at_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  added_at_round INTEGER NOT NULL,
  UNIQUE (name, kind)
);

-- Alias tables: the canonicalization layer. Three separate tables because the
-- lifecycles differ:
--
--   entity_aliases       — populated by Resolver inside apply_merge; keyed by
--                          surviving entity_id. Read at merge time so a
--                          re-surfaced merged name snaps to the survivor.
--   predicate_aliases    — alias → canonical predicate rows (written by the
--                          VocabResolver). Read by the Mapper's candidate set,
--                          the merge grace-path (rewrites raw / unmapped
--                          relations) and query-time expansion. Ingest makes
--                          no retroactive rewrites.
--   entity_type_aliases  — same shape as predicate_aliases but for entity
--                          types. Separate because read per kg_entities row
--                          rather than per kg_relations row.
--
-- All three are flat (no chains): every row's canonical points at a true
-- canonical, not at another alias. Inserts dereference on the way in
-- (store.record_*_alias).
CREATE TABLE IF NOT EXISTS entity_aliases (
  id INTEGER PRIMARY KEY,
  entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
  alias_text TEXT NOT NULL,
  source_chunk_id INTEGER REFERENCES chunks(id),
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  created_at_round INTEGER NOT NULL,
  -- COLLATE NOCASE so case-different alias_text collapses on insert; matches
  -- the kg_entities canonical_name uniqueness rule.
  UNIQUE (alias_text COLLATE NOCASE)
);

CREATE TABLE IF NOT EXISTS predicate_aliases (
  id INTEGER PRIMARY KEY,
  alias TEXT NOT NULL,
  canonical TEXT NOT NULL,
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  created_at_round INTEGER NOT NULL,
  UNIQUE (alias COLLATE NOCASE)
);

CREATE TABLE IF NOT EXISTS entity_type_aliases (
  id INTEGER PRIMARY KEY,
  alias TEXT NOT NULL,
  canonical TEXT NOT NULL,
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  created_at_round INTEGER NOT NULL,
  UNIQUE (alias COLLATE NOCASE)
);

-- property_aliases: synonym → canonical map for the attribute-lane property
-- vocabulary (the attribute-lane analogue of predicate_aliases). Written by
-- the VocabResolver when a property cluster member is judged a synonym.
CREATE TABLE IF NOT EXISTS property_aliases (
  id INTEGER PRIMARY KEY,
  alias TEXT NOT NULL,
  canonical TEXT NOT NULL,
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  created_at_round INTEGER NOT NULL,
  UNIQUE (alias COLLATE NOCASE)
);

-- schema_parents: the specialization axis the alias tables can't express — a
-- canonical's broader-canonical PARENT for rollup (child stays its own
-- canonical; query-time expansion walks child → parent). Written by the
-- VocabResolver's child_of verdicts; one parent per (child, kind).
CREATE TABLE IF NOT EXISTS schema_parents (
  id INTEGER PRIMARY KEY,
  child TEXT NOT NULL,
  parent TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('relation_type', 'property_type')),
  created_by_call_id INTEGER NOT NULL REFERENCES agent_calls(id),
  created_at_round INTEGER NOT NULL,
  UNIQUE (child COLLATE NOCASE, kind)
);

-- resolver_verdicts: memoized Resolver tier-2 'distinct' verdicts.
-- Most Resolver LLM judgments in a long build re-judge stable non-matching
-- pairs (especially during the full sweep); caching the distinct verdict lets later
-- rounds + the sweep skip re-judging them. pair_key = sorted lowercased
-- "name_a||name_b". Conservative: only 'distinct' is cached + never expires
-- (distinct→merge flips are rare; a name change after a merge changes the key,
-- so the pair re-judges naturally).
CREATE TABLE IF NOT EXISTS resolver_verdicts (
  pair_key TEXT PRIMARY KEY,
  verdict TEXT NOT NULL,
  created_at_round INTEGER NOT NULL
);

-- loop_state: singleton row carrying the cumulative outer-loop accumulator
-- state (CorpusState) across invocations. Resume reads this; end-of-round
-- save writes it.
-- CHECK (id = 1) enforces the singleton at the SQL layer.
CREATE TABLE IF NOT EXISTS loop_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  corpus_state_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- round_reviews: one row per fully-worked round (1:1 with rounds). Written
-- in two stages by the orchestrator's two-phase round:
--   Phase 1 (work committed) → deterministic metric columns filled, the two
--     review FK columns left NULL. A row whose kg_review_call_id IS NULL is
--     the "reviews pending" resume checkpoint: a crash after the KG commit
--     re-runs reviews only, never the fetch/extract.
--   Phase 2 (reviews done) → kg_review_call_id + round_review_call_id stamped.
-- Trends (Δ vs previous round, last-N window) are derived at read-time from
-- these rows, not stored.
CREATE TABLE IF NOT EXISTS round_reviews (
  round_id INTEGER PRIMARY KEY REFERENCES rounds(id) ON DELETE CASCADE,
  round_number INTEGER NOT NULL,
  new_entities INTEGER NOT NULL,
  new_relations INTEGER NOT NULL,
  docs_fetched_total INTEGER NOT NULL,
  docs_fetched_per_source TEXT NOT NULL,   -- JSON {source: count}
  tokens_spent_total INTEGER NOT NULL,     -- round work tokens; excludes the reviews
  tokens_per_new_entity REAL,              -- NULL when new_entities = 0
  tokens_per_new_relation REAL,            -- NULL when new_relations = 0
  consolidation_repairs INTEGER NOT NULL DEFAULT 0,
  already_ingested_total INTEGER NOT NULL DEFAULT 0,       -- searched candidates already in the KG (dropped pre-Selector)
  candidates_surfaced_total INTEGER NOT NULL DEFAULT 0,    -- total candidates the searches surfaced this round
  already_seen_by_topic TEXT NOT NULL DEFAULT '{}',        -- JSON {topic: [surfaced, already_seen]} — per-theme saturation
  kg_review_call_id INTEGER REFERENCES agent_calls(id),    -- NULL ⇒ reviews pending
  round_review_call_id INTEGER REFERENCES agent_calls(id),
  round_started_at TEXT NOT NULL,
  round_ended_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_agent_calls_round
  ON agent_calls (round_id, sequence_in_round);
CREATE INDEX IF NOT EXISTS ix_agent_calls_status
  ON agent_calls (status);
CREATE INDEX IF NOT EXISTS ix_fetched_docs_round
  ON fetched_docs (round_id, outcome);
CREATE INDEX IF NOT EXISTS ix_doc_assessments_round
  ON doc_assessments (round_id, decision);
CREATE INDEX IF NOT EXISTS ix_kg_entities_mentions
  ON kg_entities (mentions DESC);
CREATE INDEX IF NOT EXISTS ix_kg_mutations_target
  ON kg_mutations (target_kind, target_id);
CREATE INDEX IF NOT EXISTS ix_kg_mutations_call
  ON kg_mutations (agent_call_id);
CREATE INDEX IF NOT EXISTS ix_kg_mutations_round
  ON kg_mutations (round);
CREATE INDEX IF NOT EXISTS ix_kg_mutations_kind
  ON kg_mutations (kind);
CREATE INDEX IF NOT EXISTS ix_kg_entity_properties_entity
  ON kg_entity_properties (entity_id);
CREATE INDEX IF NOT EXISTS ix_kg_entity_properties_property
  ON kg_entity_properties (property);
CREATE INDEX IF NOT EXISTS ix_entity_aliases_alias
  ON entity_aliases (alias_text COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS ix_entity_aliases_entity
  ON entity_aliases (entity_id);
CREATE INDEX IF NOT EXISTS ix_predicate_aliases_alias
  ON predicate_aliases (alias COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS ix_predicate_aliases_canonical
  ON predicate_aliases (canonical);
CREATE INDEX IF NOT EXISTS ix_entity_type_aliases_alias
  ON entity_type_aliases (alias COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS ix_entity_type_aliases_canonical
  ON entity_type_aliases (canonical);
