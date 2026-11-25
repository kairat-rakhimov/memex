"""``memex inspect`` — terminal views over a build's trace in ``kg.db``.

    memex inspect <kg_id>                  # build + rounds rollup
    memex inspect <kg_id> --build 2        # focus a build
    memex inspect <kg_id> --round 3        # round timeline
    memex inspect <kg_id> --agent Extractor
    memex inspect <kg_id> --failures
    memex inspect <kg_id> --call 17        # full prompts + response

Reads ``kg/<kg_id>/kg.db`` directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from memex.observability import TraceStore
from memex.observability.store import kg_db_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="memex inspect",
        description="Inspect a kg-loop build's trace DB.",
    )
    parser.add_argument("kg_id", help="kg_id (subdir of --kg-root).")
    parser.add_argument(
        "--kg-root", default="kg", help="Parent dir containing the kg_id (default: kg)."
    )
    parser.add_argument(
        "--build",
        type=int,
        help="Focus a single build id (default: latest).",
    )
    parser.add_argument(
        "--round", type=int, help="Show all agent calls in this round number."
    )
    parser.add_argument(
        "--agent",
        help="Filter agent calls to this agent name (e.g. Extractor).",
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="List recent failures + sample_mode rows for the build.",
    )
    parser.add_argument(
        "--call",
        type=int,
        help="Drill into a single agent_call id: full prompts + response + parsed output.",
    )
    parser.add_argument(
        "--schema-changes",
        action="store_true",
        help="List vocabulary-gate accept / reject / drop events across the build.",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="List fetched_docs outcomes for the selected round (or all rounds).",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Row limit for tabular views (default: 50)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit selected view as JSON instead of text (for piping into jq/UI).",
    )
    return parser.parse_args(argv)


# ── rendering helpers ──────────────────────────────────────────────────────


def _fmt_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) if i < len(cols) - 1 else c for i, (c, w) in enumerate(zip(cols, widths, strict=False)))


def _truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _header(title: str) -> str:
    return f"\n── {title} " + "─" * max(2, 70 - len(title))


def _fmt_status_count(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(empty)"
    return ", ".join(f"{r['status']}={r['c']}" for r in rows)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.2f}s"


def _fmt_tokens(prompt: int | None, completion: int | None) -> str:
    if not prompt and not completion:
        return "0/0"
    return f"{prompt or 0}/{completion or 0}"


# ── views ──────────────────────────────────────────────────────────────────


def _view_overview(store: TraceStore, build_id: int) -> str:
    build = store.get_build(build_id)
    if build is None:
        return f"no build with id={build_id}"

    rounds = store.list_rounds(build_id)

    out: list[str] = []
    out.append(_header(f"build #{build['id']} — {build['kg_id']}"))
    out.append(f"query        {build['base_query']!r}")
    out.append(f"build_run_id {build['build_run_id']}")
    out.append(
        f"started      {build['started_at']}    "
        f"finished     {build['finished_at'] or '(running)'}"
    )
    out.append(
        f"sample_mode  {'yes' if build['sample_mode'] else 'no'}    "
        f"stop_reason  {build['stop_reason'] or '—'}    "
        f"rounds       {build['rounds_completed']}"
    )
    out.append(
        f"docs         {build['total_docs']}    "
        f"entities     {build['total_entities']}    "
        f"relations    {build['total_relations']}    "
        f"schema       {build['schema_entity_types']}e / {build['schema_relation_types']}r"
    )
    out.append(
        f"tokens       prompt={build['total_prompt_tokens']}    "
        f"completion={build['total_completion_tokens']}"
    )

    out.append(_header("rounds"))
    if not rounds:
        out.append("  (no rounds)")
    else:
        widths = [3, 7, 8, 8, 8, 12, 12, 18, 24]
        out.append(_fmt_row(
            ["#", "dur", "fetched", "skipped", "failed",
             "Δentities", "Δrelations", "tokens (p/c)", "stop"],
            widths,
        ))
        out.append(_fmt_row(["-" * w for w in widths], widths))
        for r in rounds:
            d_ent = (r["entities_after"] or 0) - (r["entities_before"] or 0)
            d_rel = (r["relations_after"] or 0) - (r["relations_before"] or 0)
            out.append(_fmt_row(
                [
                    str(r["round_number"]),
                    _fmt_duration(r["duration_seconds"]),
                    str(r["docs_fetched"] or 0),
                    str(r["docs_skipped"] or 0),
                    str(r["docs_failed"] or 0),
                    f"+{d_ent}" if d_ent >= 0 else str(d_ent),
                    f"+{d_rel}" if d_rel >= 0 else str(d_rel),
                    _fmt_tokens(r["prompt_tokens"], r["completion_tokens"]),
                    r["stop_criterion_fired"] or "",
                ],
                widths,
            ))

    # round reviews — RoundReviewer trajectory + ROI metrics per round
    reviews = store.list_round_reviews(build_id)
    if reviews:
        out.append(_header("round reviews"))
        widths = [3, 12, 12, 10, 12, 8, 12]
        out.append(_fmt_row(
            ["#", "Δentities", "Δrelations", "tokens", "tok/ent", "repairs", "status"],
            widths,
        ))
        out.append(_fmt_row(["-" * w for w in widths], widths))
        for rv in reviews:
            tpe = rv["tokens_per_new_entity"]
            out.append(_fmt_row(
                [
                    str(rv["round_number"]),
                    f"+{rv['new_entities']}",
                    f"+{rv['new_relations']}",
                    str(rv["tokens_spent_total"]),
                    f"{tpe:.0f}" if isinstance(tpe, int | float) else "—",
                    str(rv["consolidation_repairs"]),
                    rv["status"] or "—",
                ],
                widths,
            ))

    # agent-call status rollup
    out.append(_header("agent calls"))
    by_status: dict[str, int] = {}
    by_agent: dict[str, dict[str, int]] = {}
    for r in rounds:
        for c in store.list_agent_calls(r["id"]):
            status = c["status"]
            agent = c["agent_name"]
            by_status[status] = by_status.get(status, 0) + 1
            agent_statuses = by_agent.setdefault(agent, {})
            agent_statuses[status] = agent_statuses.get(status, 0) + 1
    if not by_status:
        out.append("  (no agent calls)")
    else:
        out.append("status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
        out.append("")
        widths = [20, 16, 12, 12, 12, 12]
        out.append(_fmt_row(["agent", "success", "sample_mode", "failed", "skipped", "other"], widths))
        out.append(_fmt_row(["-" * w for w in widths], widths))
        for agent, statuses in sorted(by_agent.items()):
            other = sum(v for k, v in statuses.items() if k not in {"success", "sample_mode", "failed", "skipped"})
            out.append(_fmt_row(
                [
                    agent,
                    str(statuses.get("success", 0)),
                    str(statuses.get("sample_mode", 0)),
                    str(statuses.get("failed", 0)),
                    str(statuses.get("skipped", 0)),
                    str(other),
                ],
                widths,
            ))

    return "\n".join(out)


def _view_round(store: TraceStore, build_id: int, round_number: int, agent_filter: str | None) -> str:
    rounds = [r for r in store.list_rounds(build_id) if r["round_number"] == round_number]
    if not rounds:
        return f"no round_number={round_number} in build {build_id}"
    r = rounds[0]
    out: list[str] = []
    out.append(_header(f"round {round_number} (build #{build_id})"))
    out.append(
        f"duration {_fmt_duration(r['duration_seconds'])}    "
        f"fetched {r['docs_fetched']}    skipped {r['docs_skipped']}    failed {r['docs_failed']}    "
        f"stop {r['stop_criterion_fired'] or '—'}"
    )
    out.append(
        f"entities {r['entities_before']} → {r['entities_after']}    "
        f"relations {r['relations_before']} → {r['relations_after']}    "
        f"schema {r['schema_entity_types_before']}/{r['schema_relation_types_before']} → "
        f"{r['schema_entity_types_after']}/{r['schema_relation_types_after']}"
    )

    out.append(_header("agent calls (in sequence)"))
    calls = store.list_agent_calls(r["id"])
    if agent_filter:
        calls = [c for c in calls if c["agent_name"] == agent_filter]
    widths = [4, 5, 18, 26, 12, 9, 14, 18]
    out.append(_fmt_row(
        ["#", "id", "agent", "action", "status", "dur", "tokens (p/c)", "input"],
        widths,
    ))
    out.append(_fmt_row(["-" * w for w in widths], widths))
    for c in calls:
        out.append(_fmt_row(
            [
                str(c["sequence_in_round"]),
                str(c["id"]),
                _truncate(c["agent_name"], 18),
                _truncate(c["action"], 26),
                c["status"],
                _fmt_ms(c["duration_ms"]),
                _fmt_tokens(c["prompt_tokens"], c["completion_tokens"]),
                _truncate(c["input_summary"], 60),
            ],
            widths,
        ))
        if c["status"] == "failed" and c["error_message"]:
            out.append(f"        ↳ {c['error_type']}: {_truncate(c['error_message'], 100)}")
    out.append("")
    out.append("drill into one with: --call <id>")
    return "\n".join(out)


def _view_call(store: TraceStore, call_id: int) -> str:
    c = store.get_agent_call(call_id)
    if c is None:
        return f"no agent_call with id={call_id}"
    out: list[str] = []
    out.append(_header(f"agent_call #{c['id']} — {c['agent_name']}.{c['action']}"))
    out.append(
        f"status       {c['status']}    duration {_fmt_ms(c['duration_ms'])}    "
        f"retries {c['retry_count']}    confidence {c['confidence']}"
    )
    out.append(
        f"started      {c['started_at']}    finished {c['finished_at'] or '—'}"
    )
    out.append(
        f"triggered_by {c['triggered_by']}    target_agent {c['target_agent']}    "
        f"parent {c['parent_call_id'] or '—'}"
    )
    out.append(
        f"model        {c['model_name'] or '—'}    tokens "
        f"{_fmt_tokens(c['prompt_tokens'], c['completion_tokens'])}"
    )
    out.append(f"input_summary  {c['input_summary']}")
    out.append(f"output_summary {c['output_summary']}")
    if c["error_message"]:
        out.append(_header("error"))
        out.append(f"{c['error_type']}: {c['error_message']}")
    if c["system_prompt"]:
        out.append(_header("system_prompt"))
        out.append(c["system_prompt"])
    if c["user_prompt"]:
        out.append(_header("user_prompt"))
        out.append(c["user_prompt"])
    if c["raw_response"]:
        out.append(_header("raw_response"))
        out.append(c["raw_response"])
    if c["parsed_output_json"]:
        out.append(_header("parsed_output"))
        try:
            out.append(json.dumps(json.loads(c["parsed_output_json"]), indent=2, ensure_ascii=False))
        except Exception:
            out.append(c["parsed_output_json"])
    return "\n".join(out)


def _view_failures(store: TraceStore, build_id: int, limit: int) -> str:
    rows = store.recent_failures(build_id, limit=limit)
    out: list[str] = [_header(f"recent failures + sample_mode (build #{build_id})")]
    if not rows:
        out.append("  (none)")
        return "\n".join(out)
    widths = [6, 4, 18, 26, 12, 18, 30]
    out.append(_fmt_row(
        ["call", "rnd", "agent", "action", "status", "error_type", "error"],
        widths,
    ))
    out.append(_fmt_row(["-" * w for w in widths], widths))
    for r in rows:
        out.append(_fmt_row(
            [
                str(r["id"]),
                str(r["round_number"]),
                _truncate(r["agent_name"], 18),
                _truncate(r["action"], 26),
                r["status"],
                _truncate(r["error_type"] or "", 18),
                _truncate(r["error_message"] or "", 60),
            ],
            widths,
        ))
    return "\n".join(out)


def _view_schema_changes(store: TraceStore, build_id: int) -> str:
    rows = store.list_schema_changes(build_id)
    out: list[str] = [_header(f"schema evolution (build #{build_id})")]
    if not rows:
        out.append("  (no schema changes)")
        return "\n".join(out)
    widths = [4, 10, 14, 26, 12, 40]
    out.append(_fmt_row(
        ["rnd", "event", "kind", "name", "call", "reason / rationale"],
        widths,
    ))
    out.append(_fmt_row(["-" * w for w in widths], widths))
    for r in rows:
        out.append(_fmt_row(
            [
                str(r["round_number"]),
                r["event"],
                r["type_kind"],
                _truncate(r["name"], 26),
                str(r["agent_call_id"]),
                _truncate(r["decision_reason"] or r["rationale"] or "", 80),
            ],
            widths,
        ))
    return "\n".join(out)


def _view_docs(store: TraceStore, build_id: int, round_number: int | None) -> str:
    rounds = store.list_rounds(build_id)
    if round_number is not None:
        rounds = [r for r in rounds if r["round_number"] == round_number]
    out: list[str] = []
    for r in rounds:
        out.append(_header(f"fetched_docs — round {r['round_number']}"))
        docs = store.list_fetched_docs(r["id"])
        if not docs:
            out.append("  (no docs this round)")
            continue
        widths = [10, 5, 24, 38, 12, 8]
        out.append(_fmt_row(
            ["source", "rank", "outcome", "title", "parse", "chars"],
            widths,
        ))
        out.append(_fmt_row(["-" * w for w in widths], widths))
        for d in docs:
            out.append(_fmt_row(
                [
                    d["source"],
                    str(d["ranker_position"] or ""),
                    _truncate(d["outcome"], 24),
                    _truncate(d["title"] or "", 38),
                    _truncate(d["parse_method"] or "", 12),
                    str(d["cleaned_text_chars"] or ""),
                ],
                widths,
            ))
            if d["outcome_detail"]:
                out.append(f"        ↳ {_truncate(d['outcome_detail'], 100)}")
    return "\n".join(out) if out else "(no rounds)"


# ── entry point ────────────────────────────────────────────────────────────


def _resolve_build_id(store: TraceStore, requested: int | None) -> int | None:
    builds = store.list_builds()
    if not builds:
        return None
    if requested is None:
        return builds[0]["id"]
    for b in builds:
        if b["id"] == requested:
            return b["id"]
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kg_root = Path(args.kg_root) / args.kg_id
    db_path = kg_db_path(kg_root)
    if not db_path.exists():
        print(f"no kg.db at {db_path}", file=sys.stderr)
        return 1

    store = TraceStore.open(kg_root)
    try:
        # --call is its own view, doesn't need a build
        if args.call is not None:
            payload = store.get_agent_call(args.call)
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            else:
                print(_view_call(store, args.call))
            return 0

        build_id = _resolve_build_id(store, args.build)
        if build_id is None:
            print(f"no builds in {db_path}", file=sys.stderr)
            return 1

        if args.failures:
            if args.json:
                print(json.dumps(store.recent_failures(build_id, args.limit),
                                 indent=2, default=str))
            else:
                print(_view_failures(store, build_id, args.limit))
            return 0

        if args.schema_changes:
            if args.json:
                print(json.dumps(store.list_schema_changes(build_id),
                                 indent=2, default=str))
            else:
                print(_view_schema_changes(store, build_id))
            return 0

        if args.docs:
            if args.json:
                rounds = store.list_rounds(build_id)
                if args.round is not None:
                    rounds = [r for r in rounds if r["round_number"] == args.round]
                payload = {
                    str(r["round_number"]): store.list_fetched_docs(r["id"])
                    for r in rounds
                }
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(_view_docs(store, build_id, args.round))
            return 0

        if args.round is not None:
            if args.json:
                rounds = store.list_rounds(build_id)
                target = [r for r in rounds if r["round_number"] == args.round]
                if not target:
                    print(f"no round {args.round} in build {build_id}", file=sys.stderr)
                    return 1
                calls = store.list_agent_calls(target[0]["id"])
                if args.agent:
                    calls = [c for c in calls if c["agent_name"] == args.agent]
                print(json.dumps(calls, indent=2, default=str))
            else:
                print(_view_round(store, build_id, args.round, args.agent))
            return 0

        if args.json:
            print(json.dumps({
                "build": store.get_build(build_id),
                "rounds": store.list_rounds(build_id),
            }, indent=2, default=str))
        else:
            print(_view_overview(store, build_id))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
