# Examples

Outputs from memex on the green-hydrogen graph, with the agent trace of each run.

| File | Produced by |
|---|---|
| [chat.md](chat.md) | `memex chat` — a grounded answer with an evidence table and citations |
| [hypothesize_bridge.md](hypothesize_bridge.md) | `memex hypothesize --seed-type bridge` |
| [hypothesize_gap.md](hypothesize_gap.md) | `memex hypothesize --seed-type gap` |
| [hypothesize_property.md](hypothesize_property.md) | `memex hypothesize --seed-type property` |
| [hypothesize_attribute_contested.md](hypothesize_attribute_contested.md) | `memex hypothesize --seed-type attribute_contested` |
| [hypothesize_relation_contested.md](hypothesize_relation_contested.md) | `memex hypothesize --seed-type relation_contested` |
| [build.md](build.md) | a deliberately tiny `build` through the HTTP API (1 round, 1 document) |

- `traces/` holds the JSONL agent trace of each run: one line per agent action,
  including revision requests and the agent they were sent back to.
- `requests/` holds the same runs as `POST /run` bodies for `memex serve`.

The chat and hypothesis reports went through their review loops before being
written out, so some open with a line about the revision ("Below is a revised
answer…"). That text is the agents' own: the GroundingChecker or FinalReviewer
asked for changes and the writer says what it changed.

The chat and hypothesis runs used the original 88-document build. The published
demo graph drops 24 arXiv papers that are not openly licensed, so the same
commands on it give similar but not identical results.
