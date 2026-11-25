# Knowledge-graph build — “green hydrogen production”

A deliberately tiny build (1 round, 1 document) to show one pass of the loop.

## Result
- **Rounds completed:** 1 (stop: `max_rounds`)
- **Documents ingested:** 1 from sources ['arxiv', 'gdelt', 'wikipedia']
- **Graph:** 308 entities · 283 relations · schema 77 entity-types / 63 relation-types

## The agent loop (see the JSONL trace for each call)
RoundPlanner → per-source QueryPlanner + search → Selector → UsefulnessGate → Extractor → Mapper / AttributeMapper → Resolver → KGReviewer → Strategist. Feedback loops: the Strategist's review feeds the next round's plan; the MappingReviewer can trigger a re-map; the Resolver merges across the round.
