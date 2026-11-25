"""Per-KG query-trace store: ``kg/<kg_id>/query_trace.db``.

Query runs (chat / hypothesize) do NOT fit the build trace DB (``agent_calls``
require a round). This store has its own small schema and — crucially — is a
**drop-in for** :class:`memex.kg_intelligence.agents.base.AgentCallSinks`: it
exposes ``begin_agent_call`` / ``finish_agent_call`` with the same shape as
:class:`TraceStore`, so ``execute_agent_call`` and ``run_tool_agent`` write to it
unchanged. A :class:`QueryRunHandle` stands in for ``RoundHandle``.

Beyond the call rows it stores the **inner-thinking trail**: per agent call, the
ordered ReAct trajectory (``query_agent_steps``: thinking / text / tool_call /
tool_return) that the inner-thinking viewer renders.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS query_runs (
  id INTEGER PRIMARY KEY,
  run_id TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,                 -- 'chat' | 'hypothesize'
  kg_id TEXT NOT NULL,
  question TEXT,
  config_json TEXT,
  sample_mode INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT,
  summary TEXT,
  total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
  total_completion_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_agent_calls (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES query_runs(id) ON DELETE CASCADE,
  parent_call_id INTEGER REFERENCES query_agent_calls(id),
  is_tool_call INTEGER NOT NULL DEFAULT 0,
  triggered_by TEXT NOT NULL,
  sequence_in_run INTEGER NOT NULL,
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

CREATE TABLE IF NOT EXISTS query_agent_call_details (
  agent_call_id INTEGER PRIMARY KEY REFERENCES query_agent_calls(id) ON DELETE CASCADE,
  system_prompt TEXT,
  user_prompt TEXT,
  raw_response TEXT,
  parsed_output_json TEXT
);

CREATE TABLE IF NOT EXISTS query_agent_steps (
  id INTEGER PRIMARY KEY,
  agent_call_id INTEGER NOT NULL REFERENCES query_agent_calls(id) ON DELETE CASCADE,
  step_index INTEGER NOT NULL,
  kind TEXT NOT NULL,                 -- 'thinking' | 'text' | 'tool_call' | 'tool_return'
  content TEXT,
  tool_name TEXT,
  args_json TEXT,
  result TEXT
);

CREATE INDEX IF NOT EXISTS ix_query_agent_calls_run
  ON query_agent_calls (run_id, sequence_in_run);
CREATE INDEX IF NOT EXISTS ix_query_agent_steps_call
  ON query_agent_steps (agent_call_id, step_index);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class QueryRunHandle:
    run_db_id: int
    run_id: str
    # round_number kept for AgentCallSinks compatibility (some callers read it).
    round_number: int = 0

    @property
    def round_id(self) -> int:  # AgentCallSinks/TraceStore shim
        return self.run_db_id


@dataclass(frozen=True)
class QueryAgentCallHandle:
    agent_call_id: int
    sequence_in_round: int


class QueryTraceStore:
    """Writer + reader over ``kg/<kg_id>/query_trace.db``."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._lock = threading.RLock()
        self._sequence_by_run: dict[int, int] = {}

    @classmethod
    def open(cls, kg_root: Path) -> QueryTraceStore:
        db_path = kg_root / "query_trace.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        return cls(conn, db_path)

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

    # ── runs ──────────────────────────────────────────────────────────

    def start_run(
        self,
        *,
        run_id: str,
        kind: str,
        kg_id: str,
        question: str | None,
        config: dict[str, Any],
        sample_mode: bool,
    ) -> QueryRunHandle:
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO query_runs (
                    run_id, kind, kg_id, question, config_json,
                    sample_mode, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    kind,
                    kg_id,
                    question,
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    1 if sample_mode else 0,
                    _utc_now(),
                ),
            )
            run_db_id = cur.lastrowid
            self._sequence_by_run[run_db_id] = 0
            return QueryRunHandle(run_db_id=run_db_id, run_id=run_id)

    def finish_run(
        self, run: QueryRunHandle, *, status: str, summary: str
    ) -> None:
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE query_runs SET
                    finished_at = ?, status = ?, summary = ?,
                    total_prompt_tokens = COALESCE((
                        SELECT SUM(prompt_tokens) FROM query_agent_calls WHERE run_id = ?
                    ), 0),
                    total_completion_tokens = COALESCE((
                        SELECT SUM(completion_tokens) FROM query_agent_calls WHERE run_id = ?
                    ), 0)
                WHERE id = ?
                """,
                (_utc_now(), status, summary, run.run_db_id, run.run_db_id, run.run_db_id),
            )

    # ── agent calls (AgentCallSinks-compatible) ───────────────────────

    def begin_agent_call(
        self,
        round_: QueryRunHandle,
        *,
        agent_name: str,
        action: str,
        triggered_by: str,
        target_agent: str | None = None,
        parent_call_id: int | None = None,
        input_summary: str = "",
        is_tool_call: bool = False,
    ) -> QueryAgentCallHandle:
        with self._lock:
            seq = self._sequence_by_run.get(round_.run_db_id, 0) + 1
            self._sequence_by_run[round_.run_db_id] = seq
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO query_agent_calls (
                    run_id, parent_call_id, is_tool_call, triggered_by,
                    sequence_in_run, agent_name, action, target_agent,
                    started_at, status, input_summary, output_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, '')
                """,
                (
                    round_.run_db_id,
                    parent_call_id,
                    1 if is_tool_call else 0,
                    triggered_by,
                    seq,
                    agent_name,
                    action,
                    target_agent,
                    _utc_now(),
                    input_summary,
                ),
            )
            return QueryAgentCallHandle(agent_call_id=cur.lastrowid, sequence_in_round=seq)

    def finish_agent_call(
        self,
        call: QueryAgentCallHandle,
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
                UPDATE query_agent_calls SET
                    finished_at = ?, duration_ms = ?, status = ?, retry_count = ?,
                    confidence = ?, model_name = ?, output_summary = ?,
                    prompt_tokens = ?, completion_tokens = ?, cached_tokens = ?,
                    error_type = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    _utc_now(), duration_ms, status, retry_count, confidence,
                    model_name, output_summary, prompt_tokens, completion_tokens,
                    cached_tokens, error_type, error_message, call.agent_call_id,
                ),
            )
            parsed_json = (
                json.dumps(parsed_output, ensure_ascii=False, default=str)
                if parsed_output is not None
                else None
            )
            conn.execute(
                """
                INSERT INTO query_agent_call_details (
                    agent_call_id, system_prompt, user_prompt,
                    raw_response, parsed_output_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (agent_call_id) DO UPDATE SET
                    system_prompt = excluded.system_prompt,
                    user_prompt = excluded.user_prompt,
                    raw_response = excluded.raw_response,
                    parsed_output_json = excluded.parsed_output_json
                """,
                (call.agent_call_id, system_prompt, user_prompt, raw_response, parsed_json),
            )

    # ── tool children + trajectory ────────────────────────────────────

    def record_tool_call(
        self,
        round_: QueryRunHandle,
        *,
        parent_call_id: int,
        tool_name: str,
        args: Any,
        result_summary: str,
        duration_ms: int = 0,
        status: str = "success",
    ) -> QueryAgentCallHandle:
        """One CHILD call row per Retriever tool invocation (``is_tool_call=1``,
        ``parent_call_id`` = the agent) so the ReAct fan-out shows as a
        tree."""

        handle = self.begin_agent_call(
            round_,
            agent_name=tool_name,
            action="tool_call",
            triggered_by="Retriever",
            parent_call_id=parent_call_id,
            input_summary=json.dumps(args, ensure_ascii=False, default=str)[:280],
            is_tool_call=True,
        )
        self.finish_agent_call(
            handle,
            status=status,
            duration_ms=duration_ms,
            output_summary=result_summary[:280],
        )
        return handle

    def record_steps(self, agent_call_id: int, steps: list[dict[str, Any]]) -> None:
        if not steps:
            return
        with self._txn() as conn:
            conn.executemany(
                """
                INSERT INTO query_agent_steps (
                    agent_call_id, step_index, kind, content, tool_name, args_json, result
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        agent_call_id,
                        i,
                        s.get("kind", "text"),
                        s.get("content"),
                        s.get("tool_name"),
                        json.dumps(s["args"], ensure_ascii=False, default=str)
                        if s.get("args") is not None
                        else None,
                        s.get("result"),
                    )
                    for i, s in enumerate(steps)
                ],
            )

    # ── reads (viewer) ────────────────────────────────────────────────

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM query_runs ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_db_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM query_runs WHERE id = ?", (run_db_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_agent_calls(self, run_db_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, parent_call_id, is_tool_call, triggered_by,
                       sequence_in_run, agent_name, action, target_agent,
                       started_at, finished_at, duration_ms, status, retry_count,
                       confidence, model_name, input_summary, output_summary,
                       prompt_tokens, completion_tokens, error_type, error_message
                FROM query_agent_calls WHERE run_id = ?
                ORDER BY sequence_in_run ASC
                """,
                (run_db_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_call(self, agent_call_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT ac.*, d.system_prompt, d.user_prompt, d.raw_response,
                       d.parsed_output_json
                FROM query_agent_calls ac
                LEFT JOIN query_agent_call_details d ON d.agent_call_id = ac.id
                WHERE ac.id = ?
                """,
                (agent_call_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_steps(self, agent_call_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT step_index, kind, content, tool_name, args_json, result
                FROM query_agent_steps WHERE agent_call_id = ?
                ORDER BY step_index ASC
                """,
                (agent_call_id,),
            ).fetchall()
        return [dict(r) for r in rows]
