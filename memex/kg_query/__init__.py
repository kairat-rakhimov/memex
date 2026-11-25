"""Query layer over a built knowledge graph (`kg/<id>/kg.db`).

A shared, mostly-deterministic retrieval toolbox + two thin agent tracks
(Q&A — `memex chat`; hypothesis generation — `memex hypothesize`).
The toolbox (`retrieval` / `expansion` / `grounding` / `seeds` / `scoring`) is
pure Python over a READ-ONLY connection + numpy matrices + NetworkX — no LLM calls.
LLM cost lives only in the agents (`memex/kg_query/agents/`).
"""
