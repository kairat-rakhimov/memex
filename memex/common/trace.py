"""JSONL agent-trace writer; mirrors every line to stdout."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from memex.common.schemas import AgentTraceEvent, TraceStatus

SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{12,}|Bearer\s+[A-Za-z0-9_\-\.]+)",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def redact(text: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", text or "")


def summarize(value: object, limit: int = 280) -> str:
    text = redact(value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False))
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


class TraceWriter:
    """Writes agent trace events to a per-request JSONL file and stdout.

    Use one TraceWriter per /run request. Pass it into the orchestration layer
    and let each agent record its actions via :meth:`record`.
    """

    def __init__(self, *, request_id: str, log_dir: Path):
        self.request_id = request_id
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"agent_trace_{request_id}.jsonl"
        self._events: list[AgentTraceEvent] = []

    def record(
        self,
        *,
        agent_name: str,
        action: str,
        input_summary: str,
        output_summary: str,
        target_agent: str | None = None,
        confidence: float | None = None,
        retry_count: int = 0,
        status: TraceStatus = "success",
    ) -> AgentTraceEvent:
        event = AgentTraceEvent(
            timestamp=utc_now_iso(),
            trace_id=self.request_id,
            agent_name=agent_name,
            action=action,
            input_summary=summarize(input_summary),
            output_summary=summarize(output_summary),
            target_agent=target_agent,
            confidence=confidence,
            retry_count=retry_count,
            status=status,
        )
        self._events.append(event)
        line = event.model_dump_json()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True, file=sys.stdout)
        return event

    @property
    def events(self) -> list[AgentTraceEvent]:
        return list(self._events)

    @property
    def relative_path(self) -> str:
        try:
            return str(self.path.relative_to(Path.cwd()))
        except ValueError:
            return str(self.path)
