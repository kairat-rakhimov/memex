"""Pydantic models for the /run request/response contract and trace events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TraceStatus = Literal[
    "started",
    "success",
    "needs_revision",
    "retry",
    "error",
    "failed",
]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: dict[str, Any] = Field(
        ...,
        description="Task payload: {\"task\": \"chat\" | \"hypothesize\" | \"build\", ...}.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional caller-provided request ID. Server generates one if omitted.",
    )


class AgentTraceEvent(BaseModel):
    """One JSONL trace event per agent action."""

    timestamp: str
    trace_id: str
    agent_name: str
    action: str
    input_summary: str
    output_summary: str
    target_agent: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    retry_count: int = Field(default=0, ge=0)
    status: TraceStatus = "success"


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    retryable: bool = False


class AgentSummary(BaseModel):
    name: str
    role: str
    status: Literal["completed", "skipped", "failed"] = "completed"


class RunResponse(BaseModel):
    status: Literal["success", "error"]
    request_id: str
    trace_id: str
    output: dict[str, Any] | None = None
    agents_used: list[str] = Field(default_factory=list)
    agents: list[AgentSummary] = Field(default_factory=list)
    runtime_seconds: float = 0.0
    trace_file: str | None = None
    sample_mode: bool = False
    error: ErrorInfo | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    port: int
    llm_configured: bool
    sample_mode: bool
