"""``run_tool_agent`` — the runtime for tool-using query agents.

Every build-loop agent is ``Agent[None, OutT]`` run once via ``execute_agent_call``
(one row, no tools). The Q&A Retriever is the first tool-using agent: an
``Agent[QueryDeps, OutT]`` whose ``@agent.tool`` wrappers call the pure toolbox.
This runner:

* bounds the ReAct loop with ``UsageLimits(request_limit, tool_calls_limit)`` —
  the main cost lever;
* reuses ``build_llm_model`` + the throttle classifier + the sample fallback;
* after the run, walks ``result.all_messages()`` to (a) persist the FULL ReAct
  trajectory as ordered ``query_agent_steps`` — the inner-thinking surface — and
  (b) emit one CHILD call row per tool invocation (``parent_call_id`` = the
  agent) so the fan-out is visible in the trace.

The non-tool query agents reuse ``execute_agent_call`` unchanged (the
``QueryTraceStore`` is drop-in for ``AgentCallSinks``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    LLMThrottleError,
    ModelPreference,
    build_llm_model,
)
from memex.kg_query.deps import QueryDeps
from memex.observability.query_store import QueryTraceStore

_log = logging.getLogger(__name__)

OutT = TypeVar("OutT", bound=BaseModel)

_THROTTLE_MARKERS = ("quota", "rate", "throttle", "429", "subscription limit")
_THROTTLE_HTTP = (429, 500, 502, 503, 504)
# pydantic_ai's structured-output mechanism surfaces as an internal tool call;
# it's not a real retrieval tool, so keep it out of the trajectory + child rows.
_OUTPUT_TOOL_NAMES = frozenset({"final_result"})


def _looks_like_throttle(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _THROTTLE_MARKERS)


@dataclass
class ToolAgentResult:
    output: Any
    handle: Any | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "success"


def _walk_messages(messages: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten ``all_messages()`` into (ordered trajectory steps, tool
    invocations). A tool invocation pairs a ``ToolCallPart`` with its
    ``ToolReturnPart`` by ``tool_call_id``."""

    steps: list[dict[str, Any]] = []
    returns: dict[str, str] = {}
    # first pass: collect tool returns (they live in later ModelRequests)
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    returns[part.tool_call_id] = _stringify(part.content)
    invocations: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ThinkingPart):
                steps.append({"kind": "thinking", "content": part.content})
            elif isinstance(part, TextPart):
                if part.content and part.content.strip():
                    steps.append({"kind": "text", "content": part.content})
            elif isinstance(part, ToolCallPart):
                if part.tool_name in _OUTPUT_TOOL_NAMES:
                    continue
                args = part.args_as_dict() if hasattr(part, "args_as_dict") else part.args
                result = returns.get(part.tool_call_id, "")
                steps.append(
                    {"kind": "tool_call", "tool_name": part.tool_name, "args": args}
                )
                steps.append(
                    {"kind": "tool_return", "tool_name": part.tool_name, "result": result}
                )
                invocations.append(
                    {"tool_name": part.tool_name, "args": args, "result": result}
                )
    return steps, invocations


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        import json

        return json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        return str(content)


async def run_tool_agent(
    *,
    agent: Agent[QueryDeps, OutT],
    user_prompt: str,
    system_prompt: str,
    deps: QueryDeps,
    sample_factory: Callable[[], OutT],
    log_prefix: str,
    settings: Settings,
    llm_model: Model | None,
    sinks: AgentCallSinks,
    agent_name: str,
    action: str,
    triggered_by: str,
    target_agent: str | None,
    input_summary: str,
    output_summary_fn: Callable[[OutT], str],
    confidence: float,
    request_limit: int = 8,
    tool_calls_limit: int | None = 12,
    model_preference: ModelPreference = "reasoning",
    parent_call_id: int | None = None,
) -> tuple[OutT, Any | None]:
    """Run a tool-using agent, bound by usage limits, and persist its full ReAct
    trajectory + per-tool child rows. Returns ``(output, parent_handle)``."""

    started = time.monotonic()
    store = sinks.store if isinstance(sinks.store, QueryTraceStore) else None
    run_handle = sinks.round_

    call = None
    if store is not None and run_handle is not None:
        call = store.begin_agent_call(
            run_handle,
            agent_name=agent_name,
            action=action,
            triggered_by=triggered_by,
            target_agent=target_agent,
            parent_call_id=parent_call_id,
            input_summary=input_summary,
        )

    def _finish(
        output: OutT,
        *,
        status: str,
        steps: list[dict[str, Any]] | None = None,
        invocations: list[dict[str, Any]] | None = None,
        usage: Any = None,
        model_name: str | None = None,
        raw_response: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        output_summary = output_summary_fn(output)
        sinks.trace.record(
            agent_name=agent_name,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            target_agent=target_agent,
            confidence=confidence,
            status="failed" if status == "failed" else "success",
        )
        if store is None or call is None or run_handle is None:
            return
        for inv in invocations or []:
            store.record_tool_call(
                run_handle,
                parent_call_id=call.agent_call_id,
                tool_name=inv["tool_name"],
                args=inv["args"],
                result_summary=inv["result"],
            )
            sinks.trace.record(
                agent_name=agent_name,
                action=f"tool:{inv['tool_name']}",
                input_summary=_stringify(inv["args"])[:200],
                output_summary=inv["result"][:200],
                target_agent=inv["tool_name"],
                status="success",
            )
        store.record_steps(call.agent_call_id, steps or [])
        store.finish_agent_call(
            call,
            status=status,
            duration_ms=int((time.monotonic() - started) * 1000),
            output_summary=output_summary,
            confidence=confidence,
            model_name=model_name,
            prompt_tokens=_safe_int(getattr(usage, "input_tokens", None)) if usage else None,
            completion_tokens=_safe_int(getattr(usage, "output_tokens", None)) if usage else None,
            cached_tokens=_safe_int(getattr(usage, "cache_read_tokens", None)) if usage else None,
            error_type=error_type,
            error_message=error_message,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_response,
            parsed_output=output.model_dump(mode="json") if hasattr(output, "model_dump") else None,
        )

    if settings.sample_mode:
        output = sample_factory()
        # Exercise the trajectory shape so the viewer + JSONL trace have content
        # even offline (retrieval quality is live-only — embeddings are zeros).
        stub_steps = [
            {"kind": "thinking", "content": "[SAMPLE_MODE] deterministic stub trajectory."},
            {"kind": "tool_call", "tool_name": "search_facts", "args": {"query": input_summary}},
            {"kind": "tool_return", "tool_name": "search_facts", "result": "[]"},
        ]
        _finish(
            output,
            status="sample_mode",
            steps=stub_steps,
            invocations=[{"tool_name": "search_facts", "args": {"query": input_summary}, "result": "[]"}],
        )
        return output, call

    try:
        model = llm_model or build_llm_model(settings, model_preference)
    except RuntimeError as exc:
        _log.warning("%s: %s; using sample", log_prefix, exc)
        output = sample_factory()
        _finish(output, status="failed", error_type=type(exc).__name__, error_message=str(exc))
        return output, call

    limits = UsageLimits(request_limit=request_limit, tool_calls_limit=tool_calls_limit)
    try:
        result = await agent.run(user_prompt, deps=deps, model=model, usage_limits=limits)
    except ModelHTTPError as exc:
        if exc.status_code in _THROTTLE_HTTP:
            raise LLMThrottleError(f"HTTP {exc.status_code}: {exc}") from exc
        output = sample_factory()
        _finish(output, status="failed", error_type=type(exc).__name__, error_message=str(exc))
        return output, call
    except (UnexpectedModelBehavior, Exception) as exc:
        if _looks_like_throttle(exc):
            raise LLMThrottleError(str(exc)) from exc
        _log.warning("%s: %s; using sample", log_prefix, exc)
        output = sample_factory()
        _finish(output, status="failed", error_type=type(exc).__name__, error_message=str(exc))
        return output, call

    steps, invocations = _walk_messages(result.all_messages())
    raw = result.all_messages_json().decode("utf-8") if hasattr(result, "all_messages_json") else None
    model_name = getattr(getattr(result, "response", None), "model_name", None)
    _finish(
        result.output,
        status="success",
        steps=steps,
        invocations=invocations,
        usage=result.usage,
        model_name=model_name,
        raw_response=raw,
    )
    return result.output, call


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
