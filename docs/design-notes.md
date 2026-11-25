# Design notes

*Last updated: 2026-11-26*

Condensed notes from building memex, organized by topic rather than by date. For each part of
the system they record what was decided, why, what the measurements said, and what was tried
and rejected. The current architecture is described in
[architecture/build-pipeline.md](architecture/build-pipeline.md) and
[architecture/query-layer.md](architecture/query-layer.md); this file is the reasoning behind it.

Most of the work happened in May–June 2026. "Chat tier" means gpt-4.1, "reasoning tier" means
gpt-5.1, embeddings are text-embedding-3-large, and all LLM calls run at temperature 0.

---

## 1. Framing principles

**The graph is an oracle that agents query, not a perfect schema.** The build's job is
best-effort extraction into a navigable, durable memory. Perfect canonicalization is not
achievable, and much of the ambiguity (which sentence supports a claim, what an entity "is",
which predicates are synonyms) can be resolved at query time by an LLM agent working over
embeddings and raw provenance. Every proposed change to the stored graph was tested against
one question: *is this only knowable at build time, or can a query-time agent handle it?*
Build-time-only information (the exact source sentence, a measurement's condition and unit)
goes into the substrate. Everything else goes into the query layer.

**Build once, query many times.** A 24-round build over 88 documents takes about 6 hours and
about 4,900 LLM calls. Querying takes seconds to minutes. The two run separately and share
one SQLite file, `kg.db`.

**Quality over cost, within reason.** For a researcher, a few dollars of extra LLM calls per
build is negligible next to a wrong edge in the graph. That is why LLM judge agents run on the
reasoning tier, extraction sees the whole document, and ranking uses LLM judgments instead of cheap
heuristics. It is also why an early append-only schema rule was dropped: a schema frozen at
ingestion time slowly stops matching what the corpus says.

**Fragmented-but-correct beats dense-but-wrong.** A graph with three synonymous predicates is
still a correct reasoning substrate. A graph where `inhibits` was merged into `catalyzes` is
not, and a hallucinated edge becomes a hallucinated hypothesis. When in doubt, memex prefers
reversible read-time fixes to destructive build-time merges (section 7).

---

## 2. Corpus acquisition

### Iterative researcher loop, not one-shot query expansion

The first design expanded the user's query once into per-source queries. That fails whenever
the field's vocabulary is not in the model's training data: named catalysts, recent
techniques and lab-specific terms never enter the search. memex instead runs rounds the way a
researcher would (plan, search, read, let what was read suggest the next search), and the
rounds span sources, so a term found in a news article can become an arXiv query next round.
Sources are arXiv (papers), Wikipedia (encyclopedic) and GDELT (news): free, no
authentication, three registers of the same topic. OpenAlex adds metadata to arXiv documents
but is not a primary source.

### Round shape

- **Parallel fan-out (first version).** Nine queries per round fired concurrently. arXiv
  (about one request per 3 s) and GDELT (about one per 5 s) throttled at once, and retries
  backed off in lockstep and collided again; in one run both returned nothing for three
  rounds. Fix: per-adapter throttles, sequential queries within a source (sources still run
  concurrently), retry backoff no shorter than the throttle. arXiv retries in a smoke build
  went from about 20 to 0.
- **Depth-first, one document per round (designed, dropped).** A goal per round, an inner
  search-rank-refine loop and consolidation after every document. Dropped because the quality
  bottleneck was extraction and resolution, not selection; because arXiv and GDELT match
  keywords, so iterative refinement buys little; and because it would cost about 100 planner
  and reviewer calls for a 100-document build instead of about 12.
- **Per-document consolidation within a round (kept).** At first, every document in a round
  was extracted against the schema as it stood at the start of the round, and resolution ran
  once at the end. With a per-document loop, the second document in a smoke build saw 51
  entity types and 131 predicates instead of none, and proposed 12 schema additions instead
  of 229.

### Planning and selection

The current front end factors the loop by who knows what:

- **RoundPlanner** (strategy) emits a *ranked list of topics*. An earlier single prompt that
  wrote queries for every source could not take new sources cleanly.
- **Each adapter plans its own queries** in its own dialect (boolean keywords for arXiv,
  natural language elsewhere) and returns about 30 topic-tagged candidates. The planner also
  rates each source's relevance `low/medium/high`, and rate-limited sources act only on `high`,
  so throttled APIs are spared *before* search, which a post-hoc selector cannot do.
- **One global Selector** (chat tier) picks at most 10 documents across all sources, weighted
  by topic priority. Per-source quotas wasted budget: when arXiv was throttled, its slots
  stayed empty. The cap is a ceiling, not a target.
- **Deduplicate documents, not queries.** Removing the "skip queries already tried" filter
  lets a document that was truncated or never ingested resurface later.
- **No intra-round retry.** A designed "kick-back" (Selector rejects a weak pool, planner
  replans) was dropped. Lower-ranked topics fill the same fan-out, and if nothing survives the
  gate, the reviewers see zero progress. The multi-round loop plus downstream gates recover
  what any single pass misses.

### UsefulnessGate replaced the Reflector

The early Reflector summarized new documents from their titles and abstracts and later gained
a per-document relevance gate. On a fixed 20-document smoke recipe:

| Gate version | Skipped | Skips judged correct |
|---|---|---|
| "Is this about X?", chat tier | 9 of 27 | 6 of 9 |
| "Would this contribute to a KG about X?", chat tier | similar | better, still missed shared-materials cases |
| Same prompt, reasoning tier | 4 of 24 | 4 of 4 |

The reasoning-tier gate also let in useful adjacent material (`gray hydrogen`, `OER`,
`iridium`), and distinct predicates *fell* from 78 to 69 because the Extractor saw more
on-topic text. The gate had become a reasoning task while still routed to the chat tier;
routing was decided once and not re-checked when the agent's job grew.

The **UsefulnessGate** (reasoning tier) replaced the Reflector. It reads the *whole* fetched
document before extraction, in two passes sharing a cached prefix: a usefulness verdict that
gates extraction, then a summary and corpus-state reflection for documents that pass. Rejected
documents are dropped but persisted, so they stay inspectable and are never re-fetched. In the
first full reference build it kept 81 documents and skipped 1.

### Stop criteria and the Strategist

`target_docs`, a reviewer's `coverage_adequate` flag and a fixed-ratio "diminishing returns"
rule were all removed. A *complete* graph has no hypotheses left to find, so coverage is the
wrong goal; the right question is whether the build still earns its tokens. A fixed-ratio floor
also overrode better judgment.

Review is now two agents in sequence. **KGReviewer** finds structural gaps, including
"breadth" gaps (whole dimensions of the topic that are absent, found through low-degree
"breadcrumb" entities). **RoundReviewer**, labeled "Strategist" in traces, consumes those gaps
plus per-round velocity metrics computed in SQL, with deltas and a 3-round window derived at
read time. It decides `continuing / slowing / stalled / wasted`, writes per-theme directives
for the next round, and keeps an `abandoned_goals` list so the planner stops chasing dead ends.
A "dry round" signal (the share of search results already ingested, per round and per topic)
shows it saturation. Deterministic floors catch what the LLM might miss: N rounds with no new
entities *and* no new relations, and a `max_rounds` cost backstop. The third reference build
stopped itself after 24 rounds, when its search returned only documents it already had.

---

## 3. Parsing and chunking

- **pdfminer** reads PDFs as flat character streams: multi-column text interleaves, headers
  and captions leak into the body, paragraph breaks are unreliable. Its chunker is
  paragraph-greedy (2,000-character target, 200-character overlap).
- **GROBID** gives section-aligned chunks: adjacent whole sections merge toward the target
  size, a long section stands alone and is never split, there is no overlap, and every chunk
  starts with its `## header`, so the Extractor knows where in the paper it is. GROBID's CRF
  and deep-learning models segment sections equally well; the deep-learning advantage is only
  in parsing bibliographies and affiliations. It is optional, see [grobid.md](grobid.md).
- **References are metadata, not edges.** memex models topic knowledge, not a citation index,
  so there are no `paper cites paper` edges. Bibliography-seeded fetching is parked.
- **Cache fetched bytes, not search results.** Search is not reproducible (GDELT embeds the
  current time; planner phrasing drifts), so only `fetch()` responses are cached, keyed per
  adapter (versioned PDF URL for arXiv, canonical URL without tracking parameters for GDELT).

---

## 4. Extraction

### Structured outputs first

The first agents parsed JSON out of free text. In the first real build, 9 of 11 Extractor
calls failed to parse, losing about 80% of the yield, and what survived drifted off topic.
With schema-enforced structured output (PydanticAI `output_type`), the failure rate went to
0%, and the same query and budget gave 628 entities and 316 relations instead of 86 and 40.

### Chunked, then whole-document session extraction

Per-chunk extraction avoids attention dilution on long documents but cannot see a relation
whose endpoints are in different chunks. Two remedies were tried:

1. **Reconciler.** One extra call per multi-chunk document proposes cross-chunk relations from
   a synopsis and the per-chunk entity lists. It helped a little (about one relation per
   document in its first smoke). It replaced a "re-fire low-yield documents" heuristic that
   had misread the problem: yield tracks document genre, not extraction quality.
2. **Session extraction (the default).** Each chunk is extracted in its own request whose
   prompt is `[instructions + whole document]` (byte-identical across the document's chunks,
   so a cacheable prefix) followed by the chunk in focus. The first request warms the cache
   before the rest fan out, which gives about 91% prompt-cache hits. The document prefix is
   capped at 360k characters (about 90k tokens), so huge documents are truncated rather than
   failing. The Reconciler is skipped in this mode.

A controlled A/B on the same two documents found that session extraction gives 13–36% more
entities and about 2.5× more mechanism (structure–activity) relations, exactly the
cross-section links the chunked path cannot make. Chunked mode produced more relations in
*total*, but the extras were routine `material × instrument` enumeration. Session extraction
costs about 2.3–2.9× the input tokens with caching, not the naive 4×. `chunked` stays as an
explicit fallback. Two methodology lessons: at temperature 1 session extraction seemed to find
46% more relations, a difference that vanished at temperature 0 (sampling noise; every model
has been pinned to temperature 0 since); and the first session prompt ("only emit what this
section states") suppressed the very cross-section benefit under test.

### Model routing

At first every agent ran on the chat tier. Routing is now declared per agent
(`MODEL_PREFERENCE` in each agent module). Volume work (Extractor, RoundPlanner, Selector)
uses the chat tier. Agents where one wrong call does lasting damage (Mapper, Resolver,
UsefulnessGate, the reviewers, Reconciler and every query agent) use the reasoning tier. The
Reconciler was almost left on the chat tier, but a false cross-chunk relation is as permanent
as a bad merge.

---

## 5. From a reactive schema stack to a per-document Mapper

### What was tried first

The first canonicalization design was reactive. The Extractor saw the current schema in its
prompt; a SchemaProposer and a SchemaValidator negotiated additions in a revision loop; and a
SchemaCritic, from round 2 on, clustered predicates and types (by token overlap, later by
embeddings), wrote alias tables and rewrote existing rows. It worked mechanically: one
iteration made 99 retroactive rewrites, and three tightenings (sweep every alias on each
apply, delete a canonical row once re-pointed, hide aliased names from the schema view) halved
entity types from 180 to 90. But it was the wrong shape:

- It was always a step behind. The Extractor kept minting free-text verb phrases faster than
  the Critic could chase them; one build had 372 distinct predicates, 58.9% used only once.
- Embedding clusters group by *topic* and tend to merge antonyms. Direction inversions
  (`is_produced_by` → `produces`) got past an LLM validator told to reject them.
- The schema agents accounted for about 46% of build cost.

### What replaced it

Canonicalization became proactive and two-stage:

1. The **Extractor** emits *free-form* triples, optimized only for faithfulness, with no
   schema in its prompt.
2. A per-document **Mapper** (reasoning tier, one call per document, whole document as a
   cached prefix) maps vocabulary only: predicates and entity *types*, never entity identity.
   It sees a candidate slice of the existing vocabulary, scores each triple's `confidence` and
   `newness`, and names a `fallback_predicate` so a rejected new predicate lands on existing
   vocabulary instead of going in raw.
3. A **MappingReviewer** (same model, adversarial, given deterministic scores the Mapper
   lacked) flags suspect mappings (over-collapse, spurious new predicate, wrong direction,
   wrong type, unfaithful) for one re-map.
4. A deterministic **gate** handles form: lowercase new predicates, reject stranded-preposition
   tails, drop self-loops and identity-like predicates, and resolve fallbacks against the
   *full* vocabulary, not just the slice.

Mapping is a reasoning judgment, and a reasoning pass with the source in context keeps
opposites apart where embedding clustering does not. Raw subject, predicate and object strings
are kept on every relation, and triples the Mapper cannot map are committed raw (flagged)
rather than lost. The old stack was replaced outright, not put behind a flag: free-form
extraction has no other consumer, so a flag would have kept the whole old stack alive.

### Confidence and corroboration

Confidence is per observation. The same triple from N documents is N rows
(`UNIQUE(subject, predicate, object, source_doc_id)`), and edge trust is computed at read time
as average confidence plus the number of distinct sources; confidences are never stacked.
[`eval_confidence_calibration.py`](../experiments/eval_confidence_calibration.py) showed the
original `mapper_confidence` was inert (average 0.992 on a 6-document corpus, almost no
spread). After recalibration the first full reference build averaged 0.883, with 27% of
statements below 0.9.

### What the first full build revealed about sprawl

The first reference build (24 rounds, 82 documents, 11.6k entities, 17.4k relations, 10.1k
attributes) was healthy on identity: about 1,000 Resolver merges, real cross-topic hubs, no
fake measurement hubs, no casing or punctuation duplicates. Its one material defect was
sprawl: 2,202 distinct predicates (991 used once), `uses` / `used_in` / `used_for` covering
1,727 relations between them, and 2,874 distinct properties. The first diagnosis was
"candidate-slice starvation": the Mapper saw only 119 of 1,786 predicates (the 60 most frequent
plus 60 retrieved), so it could not see existing canonical forms. A later audit revised this.
The sprawl came mostly from a *missing rollup hierarchy* (near-synonyms with no parent to roll
up to) and weak form rules, and it was fixed by the cluster overlay in section 7 and stricter
form rules, not by enlarging the slice. Feeding live clusters back into the slice was deferred
because it pushes the Mapper toward over-collapse, its main failure mode.

---

## 6. Typed statements and grounding

### Relations versus attributes

Read triple by triple, a smoke build showed about 21% of "relations" were measurement leaves,
a value forced into the object slot (`AISI Ni42-300 shows_overpotential "491 mV"`). They
produced opaque degree-1 nodes and one fake hub of degree 153, and triples alone cannot
support property-driven hypotheses (ranking, outliers, comparing measurements across
families).

The basic unit became the **typed statement** (the RDF resource-versus-literal distinction).
A statement's object is either an entity, making it a **relation** (`kg_relations`), or a
value, making it an **attribute** (`kg_entity_properties`, with `value_num`, `value_raw`,
`unit` and `condition` columns). The Extractor classifies each statement on the chat tier. An
**AttributeMapper** (reasoning tier, same cached prefix) canonicalizes the property name,
parses number and unit, and normalizes the condition. Canonical relations and attributes are
also embedded as sentences, so the query layer can retrieve *facts* by meaning, not only names.

A/B on a fixed 6-document corpus ([`eval_corpus_build.py`](../experiments/eval_corpus_build.py)
+ [`eval_corpus_compare.py`](../experiments/eval_corpus_compare.py)):

| Metric | Triples only | Typed statements |
|---|---|---|
| Measurement-leaf relations | 2.9% | 0% |
| Relation purity (both endpoints real entities) | 97% | 100% |
| Distinct predicates | 473 | 222 |
| Near-duplicate predicate families | 5 | 0 |
| Structured attributes | 0 | 448 (63% numeric, 58% with a condition) |
| Fact embeddings | 0 | 955 relation + 448 attribute |

The orphan rate rose from 2.8% to 11%, an artifact: entities once connected only by
measurement edges now carry attributes but no relations. The run also caught a real bug,
entity merges not re-pointing rows in the new attribute table.

### Conditions, units and a value gate

Property-outlier hypotheses were comparing apples with oranges: stack efficiency against system
efficiency, regional figures against global ones. Each value was faithful to its source, so a
per-fact verifier could not catch it; the fix had to go into the substrate. The AttributeMapper
now captures the condition (the measurement's system boundary) and normalizes units, and a
deterministic gate drops vague or sentence-shaped values. On a document with few stated
conditions, condition fill on numeric attributes rose from 34% to 69%
([`eval_attr_mapper.py`](../experiments/eval_attr_mapper.py)); across full builds, from 59%
to 73%. The property lane compares values only within `(property, unit, entity_type)` and
down-ranks groups where most values lack a condition.

### Source-quote grounding

Grounding first used a three-level ladder: canonical triple, raw strings, chunk text. A chunk
is a paragraph, not a sentence, and a short value like "23%" is not unique within one. The
Extractor now emits the verbatim source sentence for every relation and attribute; it knows the
sentence at extraction time, which beats fuzzy relocation at query time. The LLM is *not* asked
for character offsets, because models cannot count characters; `locate_quote` finds the quote
deterministically. Quote coverage went from 0% to 100% in the next build, and every grounded
fact in a report shows its quote. Entities get no quote and no synthesized definition: an
entity's meaning is the union of its mentions, and "what is X?" is answered by reading its
chunks.

---

## 7. Canonicalization: identity merges versus cluster metadata

### The Resolver (entity identity)

The Resolver merges entities that are the same thing. It proposes candidates in three tiers
and has the LLM judge them (reasoning tier, verdicts memoized):

1. deterministic normalization (case, punctuation, morphology);
2. token overlap and acronyms, filtered to pairs that share a source;
3. embedding similarity (cosine ≥ 0.75) between this round's new entities and all entities.
   The motivating case was `Fe2O3` versus `iron oxide`, which share no tokens; in its first
   smoke tier 3 surfaced 66 candidates, 58 of which token overlap had missed.

Pairs above cosine 0.88 skip the shared-source filter, so cross-document spelling variants
(`electrolyser` / `electrolyzer`) can merge. This replaced a hard-coded morphology rule.

- **A judge prompt that was not shipped.** A known false merge ("green hydrogen" absorbing
  "grey hydrogen") prompted a contrastive-qualifier rule for the judge.
  [`eval_resolver.py`](../experiments/eval_resolver.py) tested it on curated easy pairs,
  curated hard subset/hypernym pairs and 24 real high-cosine pairs from a built graph: it
  changed 0 of 40 verdicts. The false merge had been a candidate-*routing* problem, already
  fixed by the bypass change. The harness stays as a regression test.
- **An over-merge that was accepted.** Both prompts merge `X carbon capture` with
  `X carbon capture and storage`. A same-model verifier shares the blind spot, and a rule such
  as "an extra substantive token blocks the merge" would also split legitimate merges
  (`on-grid grid-connected hydrogen system` / `on-grid hydrogen system`). The loss is bounded:
  the merged entity still carries different values under different conditions.

### Predicate sprawl is handled at read time

A retroactive, destructive predicate merge was designed and dropped. Two of the three
retrieval modes are already immune to sprawl (fact embeddings cluster synonyms regardless of
label; untyped topology ignores labels), and the symbolic mode recovers through *predicate
expansion*: query a predicate together with its cluster and embedding neighbours, and let
evidence verification filter the union. Expansion faces the same antonym hazard as merging,
but an over-broad expansion only adds candidates that get filtered, while a wrong merge
corrupts edges for good. A stricter Mapper form prompt
([`eval_mapper.py`](../experiments/eval_mapper.py)) showed no benefit and was not shipped.

### VocabResolver: a non-destructive cluster overlay

The **VocabResolver** clusters predicates and properties after the build: embeddings propose
candidates (cosine ≥ 0.60), the LLM judges each as `alias`, `child_of` or `unrelated`, and the
result is written as alias and parent tables. Edges are never rewritten, and query-time
expansion walks the overlay. Its first run took cluster coverage from 0% to 58% for predicates
and 43% for properties. The same hardening round had the gate resolve fallbacks against the
full vocabulary (raw commits fell from 4.3% to 1.3% of statements), and the cosine bypass
merged the `PEM electrolyzer` / `PEM electrolyser` hubs. At first the VocabResolver ran only
inside an evaluation harness, so a fresh build would have had no clusters at all. It now runs
as a finalization phase of `memex build`, and `scripts/kg_finalize_vocab.py --reset`
re-clusters the full vocabulary in one pass, which gives cleaner clusters than incremental
top-ups.

### The placement rule

> Run a resolver **in the round loop if and only if it mutates the graph**; run it **post-hoc
> if and only if it writes non-destructive metadata**.

The Resolver fuses nodes, and that propagates to everything downstream (next round's context,
the planner, topology, the reviews), so it belongs in the loop. The VocabResolver writes
metadata that nothing reads automatically. In the loop it would cost more, help nothing during
the build, and cluster worse, since an incremental pass judges a name before its neighbours
exist and freezes that verdict. Two alternatives were rejected. Letting clustering replace the
Mapper's predicate mapping would bring back antonym merges and lose document context. Feeding
live clusters into the Mapper would need an in-loop VocabResolver and push the Mapper toward
over-collapse. The Resolver stays because topology (paths, common neighbours, communities,
degree) needs one node per real entity; without it the bridge and gap lanes fragment. The
result has two layers: a **local best-effort base graph** (Extractor and Mapper reason inside
one document, which LLMs do well) and a **global non-destructive metagraph** built when the
whole graph is visible.

---

## 8. Embeddings and retrieval infrastructure

- **Storage.** text-embedding-3-large at its native 3,072 dimensions, stored as float32 blobs
  and searched with an in-memory numpy dot product over pre-normalized vectors. Chunks,
  entities, predicates, properties, relations and attributes are embedded. A merged entity's
  or aliased predicate's vector is deleted in the same transaction as the merge. Embeddings
  cost about $0.50 per build.
- **Matryoshka truncation to 1,024 dimensions.** Embeddings were 71% of a 975 MB database.
  Builds keep 3,072 dimensions so clustering and resolution stay sharp; a slim export
  (`scripts/kg_slim_export.py`) truncates and renormalizes to 1,024 and drops trace-only
  tables (975 → 280 MB in one case, 681 → 198 MB in another; the published demo graph keeps
  its trace tables, so it is larger). The query layer truncates the
  query vector to match. A full-length query matches its own truncated vector at cosine 1.0,
  and a chat answer to the same question was as rich at 1,024 dimensions as at 3,072. A
  float16 round-trip was also measured as lossless for retrieval (recall@20 about 1.0).
- **Local arXiv hybrid index (an experiment, not in this repository).** The reference builds
  searched arXiv through a self-hosted index of 3.05M papers (a Kaggle bulk snapshot plus
  incremental harvesting) on a local GPU machine: BM25 (SQLite FTS5) plus dense retrieval
  (bge-large-en-v1.5, 1,024 dimensions, with a query prefix), fused by reciprocal rank on the
  server. Dense retrieval did its job, lifting a paper BM25 ranked 22nd into the top 6 in one
  check and handling abbreviations (PEM → proton exchange membrane). Two upgrades were dropped:
  an OpenAlex concept/citation hybrid, because the dense half already rescued what BM25
  missed; and the stella embedding model, because the LLM Selector re-ranks anyway, so
  retrieval only needs recall@K, which is not where the benchmark gap lies. The index failed
  loudly instead of falling back, so an outage stopped a build instead of quietly degrading
  it. The machine has since been retired; this repository uses the arXiv Atom API.

---

## 9. Storage, orchestration and resilience

- **One SQLite file is the whole graph.** The first plan was NetworkX in memory with JSONL
  snapshots. SQLite came in for tracing, became the canonical store, and the in-memory model
  was deleted. Full document text lives in `documents`, and chunks are offsets into it, so
  citation, verification and "show me the surrounding paragraph" all work from one file.
  Edits to existing rows go to `kg_mutations`, and each entity and relation records the agent
  call that created it. Output excerpts follow per-source licensing: Wikipedia and arXiv show
  the cited chunk; GDELT is clipped to 200 characters.
- **Graph orchestration per round, plain loop outside.** Each round is a `pydantic_graph`
  built with `GraphBuilder` / `Fork` / `Join` / `Decision`; the older `BaseNode` runner was
  already deprecated. The loop over rounds, the stop criteria and cross-round feedback stayed
  plain Python, where graph form only added ceremony.
- **Two-phase atomic rounds.** Work commits first, then reviews run. A `round_reviews` row
  without review results is a resume checkpoint, so a crash in review re-runs only the
  reviews. A failed review stops the build instead of continuing on stub output.
- **Retries are not free.** The HTTP client had been assumed to retry 429s; it had zero
  retries, and a bare `except` silently swapped in offline stub output on the first throttle.
  Now: transport retries honour `Retry-After`, throttle errors are classified (including
  free-text quota messages), the build cools down and stops after three consecutive throttles,
  hung connections time out, and degraded rounds are reported in a summary.

---

## 10. The query layer

**Thin agents over a deterministic toolbox.** Retrieval (SQL, fact embeddings, NetworkX
topology), grounding, seed generation and scoring are pure functions in
[`memex/kg_query/`](../memex/kg_query/) with no LLM calls. Agents decide what to look at and
how to state conclusions. The flows are described in
[query-layer.md](architecture/query-layer.md).

**Chat.** A tool-using **Retriever** (ReAct, trajectory persisted) gathers evidence, an
**Answerer** writes strictly from it, and a **GroundingChecker** verifies each claim and can
force a revision. The first live answer had 9 cited claims, an explicit "not in the corpus"
section and one grounding-forced revision.

**Hypothesize: seeds.** Deterministic seeds bound the LLM's work and guarantee real premises.
There are five lanes ([`seeds.py`](../memex/kg_query/seeds.py)):

- **bridge**: A and C share a neighbour B, have no edge, and are *far apart* in embedding
  space (Swanson's ABC pattern);
- **gap**: the same structure, but A and C are *close* or in one community, an expected link
  that is missing;
- **property**: outliers within comparable `(property, unit, entity_type)` groups, plus shared
  values across entity types ("different materials, same value: shared mechanism?"), which
  prose retrieval over one entity at a time cannot reach;
- **attribute_contested**: one entity and property with different values under different
  conditions;
- **relation_contested**: two or more predicates between the same pair.

**Hypothesize: what changed and why.**

1. **The verifier must not require that the legs prove the claim.** A hypothesis goes beyond
   its premises by design. **VerifyEvidence** rejects a candidate only when a load-bearing leg
   is hallucinated or contradicted by its source; survivors on a 16-seed sweep went from 0 to
   4, with 8 still filtered.
2. **Verify against the whole chunk.** The verifier had read a clipped 500-character window
   and judged correctly grounded legs "unsupported". Reading the full chunk took survivors on a
   16-seed sweep from 0 to 13.
3. **Rank by judgment, not cheap heuristics.** Each lane fires on a cheap signal (cosine,
   predicate count, shared neighbours) that mixes real signal with noise, and the lane's own
   premise names the check that separates them. *Novelty:* cosine treats a synthesis as a
   restatement (it reuses its premises' words) and cannot tell contradiction from agreement, so
   a batched **NoveltyJudge** asks whether the *claim* is already asserted or entailed by the
   nearest facts; the cosine pre-filter (≥ 0.85 counts as a restatement) stays as a first pass
   and fallback. *Contested relations:* synonyms (`combined_with`, `integrates`) look
   contested by count, so a **ContestationJudge** keeps only genuine opposition (`increases`
   versus `replaces`). *Bridges:* legs that collapse into one source chunk are a co-mention,
   which contradicts the premise that A and C are distant. Before this, a near-definitional
   claim led one report and its only genuinely contested edge ranked seventh; afterwards the
   report led with that edge, conflicting numbers and concrete gaps.
4. **Ranking prefers no lane.** The structure score was a per-lane table (bridge 1.0 down to
   property 0.55), a built-in preference for one *kind* of hypothesis. It is now **graph
   dependence**, the number of distinct documents the legs span, which also covers the
   contested-sprawl and single-chunk-bridge cases (both reduce to "few sources"). Composite
   weights: novelty 0.25, plausibility 0.25, support 0.20, structure 0.30.
5. **An agentic Explorer replaced one-shot generation.** The one-shot generator saw compressed
   leg triples and partly reasoned from the model's priors, which the structure-first design
   exists to avoid. The **Explorer** is a tool-using agent that, per explore-worthy seed, reads
   source sentences (`read_source`), pulls numeric landscapes (`facts_by_property`), traces
   links, and then forms a grounded hypothesis or abandons the seed. It is anchored to its
   seed and capped by a usage budget. `seed_type` spends the whole budget on one lane; the
   user picks the lens and ranking within it stays neutral. One `attribute_contested` sweep
   produced 6 verified hypotheses in 120 s, including a data-integrity catch (one capacity
   figure credited to two technologies). The one-shot generator survives only as the baseline
   in [`eval_hypothesize.py`](../experiments/eval_hypothesize.py).
6. **Value filtering is not a type preference.** Some grounded, novel leads answered no
   scientific question: data-quality trivia, one company's business details, off-topic
   tangents. The Explorer now gets the research topic and a value bar, and explore-worthiness
   includes a mild topical-relevance factor. Neutral ranking means not preferring a *kind* of
   legitimate hypothesis; it does not mean accepting non-questions.
7. **The report always runs.** If nothing survives verification, the output is an honest
   exploration summary, not an empty result.

Hypothesis *content* varies from run to run; gates and scoring are deterministic. Trust them
for category-level quality, not exact rank order.

---

## 11. Evaluation method

Full builds are non-deterministic (live search, shifting corpora) and expensive, so they
cannot isolate one change. What worked:

- **Per-agent harnesses on fixed inputs.** Reuse the real agent, prompt builder and output
  schema; swap only the system prompt, model or threshold; read inputs from a prior build's
  stored raw columns: [`eval_mapper.py`](../experiments/eval_mapper.py),
  [`eval_attr_mapper.py`](../experiments/eval_attr_mapper.py),
  [`eval_resolver.py`](../experiments/eval_resolver.py),
  [`eval_vocab_resolver.py`](../experiments/eval_vocab_resolver.py),
  [`eval_chat.py`](../experiments/eval_chat.py),
  [`eval_hypothesize.py`](../experiments/eval_hypothesize.py). Cheap enough to validate every
  prompt change *before* a multi-hour build.
- **Fixed-corpus builds.** [`eval_corpus_build.py`](../experiments/eval_corpus_build.py) runs
  only the per-document knowledge loop over 6 pinned documents. At temperature 0 that is close
  to deterministic, so A/B comparisons mean something.
  [`eval_corpus_compare.py`](../experiments/eval_corpus_compare.py) computes the battery:
  measurement-leaf rate, relation purity, generic-type rate, near-duplicate predicate
  families, orphan rate, embedding coverage.
- **Rebuilds from an existing corpus.** [`eval_rebuild.py`](../experiments/eval_rebuild.py)
  re-runs extraction, mapping, embedding and resolution over every document of an earlier
  build, in the original order, with current code, and ends with a full Resolver sweep. It
  skips search and fetch, so it tests ingest fixes at full scale.

Lessons: raw counts mislead (more relations can be noise, fewer predicates can be
over-collapse), so measure form, meaning and grounding; drop changes whose metric does not move
(the Mapper form rule and the Resolver contrastive rule both went); validate on the real
distribution as well as curated gold sets; and re-audit model routing when an agent's job
changes.

### Results across the three reference builds

The builds are not volume-controlled. The second re-ran the first build's corpus through the
hardened pipeline in one pass; the third was a fresh, native multi-round build with a broader
topic mix. The demo graph (`memex fetch-demo`) is the third build with the 24 arXiv papers that
are not openly licensed removed, together with everything extracted from them.

| Metric | 1st: 24 rounds, 82 docs | 2nd: rebuild, same corpus | 3rd: native, 24 rounds, 88 docs |
|---|---|---|---|
| Entities / relations / attributes | 11.6k / 17.4k / 10.1k | — / 22.2k / — | 11.3k / 15.0k / 7.8k |
| Predicate / property cluster coverage | 0% / 0% | 58% / 43% | 55% / 50% |
| Generic entity-type rate | 7.5% | 2.8% | 7.4% |
| Relation purity | — | 99.8% | 97.3% |
| Measurement-leaf relations | ~0.2% | ~0% | 1.8% |
| Source-quote coverage | 0% | 0% | 100% |
| Condition fill, numeric attributes | — | 59% | 73% |
| Near-duplicate predicate families | — | 4 | 1 |
| Orphan entities | — | 11.2% | 14.4% |

The third build is a mixed result. Grounding, conditions and cluster quality improved; typing,
measurement leaves and orphans regressed. The causes: a corpus heavier in economics (266
`economic_quantity` objects), typing decided at extraction time, and no final full-corpus
Resolver sweep (the native build sweeps every 5 rounds). None of it is fixable after the fact:
a final sweep merged 0 of 1,622 orphans, 1,579 of which are mentioned once and 608 of which
still carry attributes. The measured functional impact is small. Chat ignores type and orphan
status; only the property lane groups by type, and generic `concept` types appear in 3.4% of
its comparison groups, labeled and filtered downstream. The levers for future builds are at
extraction time: a minimum-mention filter and a stronger typing prompt.

---

## 12. What did not work, or was dropped

- Covered above: one-shot query expansion, depth-first rounds, the kick-back retry, quotas,
  the Reflector, `coverage_adequate` and `target_docs` (section 2); JSON parsed from free text
  and re-firing low-yield documents (section 4); the reactive schema stack and the
  candidate-slice diagnosis (section 5); destructive predicate merges (rewrite code kept,
  unused) and two prompt changes that moved no metric (section 7); the OpenAlex hybrid and
  the stella upgrade (section 8); lane-keyed structure scores and cosine-only novelty
  (section 10).
- Allowing new vocabulary only when a document is judged useful enough ("reuse-only"
  otherwise): it made relations from adjacent documents generic and cost breadth, while the
  non-destructive rollup organized the sprawl without losing specificity.
- Entity glosses and embedding them: an entity's meaning is the union of its mentions.
- New sources: a general web adapter (search API, headless browser, boilerplate removal) was
  not worth its dependencies; chemRxiv's API and PDF host sit behind a JavaScript challenge,
  and its only clean path (Crossref) is abstract-only.
- A replay mode that re-runs a recorded build without LLM calls: querying a prebuilt `kg.db`
  directly made it unnecessary.

---

## 13. Open directions

- **Entity metagraph.** Entities are either *identical* (merged) or *unrelated*; nothing
  represents "related but distinct", such as `green hydrogen` and `grey hydrogen` as siblings
  under `hydrogen`. The plan is to give the VocabResolver an `entity` kind (embeddings propose
  neighbours, the LLM judges `alias` / `child_of` / `unrelated`, results stored as parent
  links in the existing `schema_parents` shape) and let query-time expansion reach entity
  families. Post-hoc and non-destructive, it runs on a finished graph without a rebuild and
  stays unwired if it does not help. It would also make a more conservative Resolver safe,
  since relatedness the Resolver declines to merge would come back as metadata. Open
  questions: clusters form a DAG, not a tree (`green hydrogen` under both `hydrogen` and
  `renewable energy`); the entity lane is about 6× larger than the predicate lane; tuning the
  Resolver down is a real precision/recall trade-off; and expansion should be wired in only
  once it measurably improves bridge, gap or chat results.
- **Use `locate_quote` in the grounding ladder.** Quotes are now fully populated, so a
  sentence-level step can sit before the chunk-level fallback.
- **Fix typing at extraction time** (minimum-mention filter, stronger typing prompt) for the
  generic-type and orphan regressions.
- **Support scoring.** Support weighs 0.20 and does not penalize thin single-source grounding
  enough; it is the next lever if ranking needs another push.
- **Bibliography-seeded fetching** from GROBID-parsed references.
- **Statement-level precision.** The MappingReviewer's `unfaithful` verdict is today's
  precision path; no calibrated confidence threshold for dropping statements has been set.
