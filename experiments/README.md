# Experiments

The A/B harnesses used while developing memex. Each one isolates a single agent
or stage so a prompt or threshold change can be measured without a full build.
They are kept as working examples of the evaluation method described in
[docs/design-notes.md](../docs/design-notes.md); their comments record the
verdicts of the experiments they ran.

Most read an existing `kg.db` and need `OPENAI_API_KEY`; `SAMPLE_MODE=true`
runs them against stubs to check the plumbing. Run them from the repo root,
e.g. `uv run python experiments/eval_mapper.py green-hydrogen --variant shipped`.
Defaults that name an earlier build (for example the pinned 6-document corpus in
`eval_corpus_build.py`) refer to builds that were not published; point them at
your own with the command-line options.

| Harness | What it measures |
|---|---|
| `eval_corpus_build.py` | Builds a graph from a pinned set of documents through the real extract → map → resolve loop, so two code versions see identical input. |
| `eval_corpus_compare.py` | Diffs two `kg.db` files on graph-quality metrics: measurement leaves, relation purity, generic types, near-duplicate predicates, attribute coverage, vocabulary clusters. |
| `eval_rebuild.py` | Re-ingests a whole earlier corpus with the current code (no search or fetch), for A/B on ingest-quality fixes. |
| `eval_mapper.py` | Mapper system-prompt variants on one document's fixed raw triples. |
| `eval_attr_mapper.py` | AttributeMapper prompt variants: condition capture and unit normalisation. |
| `eval_resolver.py` | The entity Resolver's merge judge on gold and real candidate pairs. |
| `eval_vocab_resolver.py` | VocabResolver clusters and alias / parent verdicts on an existing vocabulary (read-only). |
| `eval_confidence_calibration.py` | Whether `mapper_confidence` is calibrated, and which elicitation prompt spreads it. |
| `eval_chat.py` | One Q&A agent at a time on a deterministic evidence bundle. |
| `eval_hypothesize.py` | Hypothesis stages in isolation; `--seeds-only` needs no LLM. |
