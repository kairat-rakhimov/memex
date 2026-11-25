"""Shared PydanticAI plumbing for the kg-loop agents.

``execute_agent_call`` is the single entry point: it begins the DB row
before the model call (so a mid-call crash leaves a visible
"started, never finished" trace), runs the agent through
``run_agent_or_fallback``, and finishes the row on both sinks.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.settings import ModelSettings
from tenacity import (
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memex.common.config import Settings
from memex.common.http import http_timeout
from memex.common.trace import TraceWriter
from memex.observability import AgentCallHandle, RoundHandle, TraceStore

_log = logging.getLogger(__name__)

OutT = TypeVar("OutT", bound=BaseModel)

# "chat" → settings.chat_model (volume pool); "reasoning" → settings.reasoning_model.
# Only consulted when the caller passes llm_model=None.
ModelPreference: TypeAlias = Literal["chat", "reasoning"]


# ── LLM throttle classifier ────────────────────────────────────────────


class LLMThrottleError(Exception):
    """LLM API throttling persisted past transport-level retries.

    Triggers the pipeline circuit-break; does NOT fall through to
    sample-mode (sample-mode is the explicit offline path, not an
    outage substitute).
    """


class AgentHardFailure(Exception):
    """A ``fail_hard=True`` agent call failed (non-throttle) past its retries.

    Critical-path agents — the two end-of-round reviews — opt out of the
    sample-on-failure degradation: a broken review signal must halt the build
    (surfaced, resumable) rather than let it continue on a fabricated sample.
    """


# Provider throttling shows up both as clean HTTP 429s and as free-text
# "quota..." messages on other status codes. Match both.
_THROTTLE_TEXT_MARKERS = ("quota", "rate", "throttle", "429", "subscription limit")
_THROTTLE_HTTP_STATUSES = (429, 500, 502, 503, 504)


def _looks_like_llm_throttle(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _THROTTLE_TEXT_MARKERS)


# ── model build ────────────────────────────────────────────────────────────


def build_llm_model(
    settings: Settings, preference: ModelPreference = "chat"
) -> Model:
    """Build an OpenAI-compatible model pointed at ``settings.base_url``.

    Wrapped in ``AsyncTenacityTransport`` so 429 / 5xx get bounded
    retries (``Retry-After`` honoured if present). On exhaustion,
    ``run_agent_or_fallback`` raises ``LLMThrottleError``.
    """

    if not settings.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — cannot build the LLM model. "
            "Set the key, or enable SAMPLE_MODE for an offline run."
        )
    model_name = (
        settings.reasoning_model if preference == "reasoning" else settings.chat_model
    )
    # Every agent here is a faithfulness / structured task, not creative
    # writing — pin temperature 0 for determinism + precision. (gpt-5.1 accepts
    # temperature=0 through the chat-completions API; verified.)
    return _build_model(settings.api_key, settings.base_url, model_name, 0.0)


def _retry_log(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    _log.warning(
        "LLM HTTP retry (attempt %d): %s",
        retry_state.attempt_number,
        exc,
    )


@lru_cache(maxsize=4)
def _build_model(
    api_key: str, base_url: str, model_name: str, temperature: float | None = None
) -> Model:
    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(
                max_wait=120.0,
                fallback_strategy=wait_exponential(multiplier=2, min=2, max=60),
            ),
            stop=stop_after_attempt(4),
            before_sleep=_retry_log,
            reraise=True,
        ),
        validate_response=lambda r: r.raise_for_status(),
    )
    http_client = httpx.AsyncClient(transport=transport, timeout=http_timeout(300.0))
    provider = OpenAIProvider(base_url=base_url, api_key=api_key, http_client=http_client)
    settings = ModelSettings(temperature=temperature) if temperature is not None else None
    return OpenAIChatModel(model_name, provider=provider, settings=settings)


# ── agent run + capture ────────────────────────────────────────────────────


@dataclass
class AgentRunResult(Generic[OutT]):
    output: OutT
    status: str  # "success" | "sample_mode" | "failed"
    duration_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    raw_response: str | None = None
    model_name: str | None = None
    retry_count: int = 0
    error_type: str | None = None
    error_message: str | None = None


async def run_agent_or_fallback(
    agent: Agent[None, OutT],
    user_prompt: str,
    *,
    settings: Settings,
    llm_model: Model | None,
    sample_factory: Callable[[], OutT],
    log_prefix: str,
    model_preference: ModelPreference = "chat",
    fail_hard: bool = False,
) -> AgentRunResult[OutT]:
    """Run ``agent`` against the LLM API; on failure fall back to ``sample_factory``.

    Throttle (429/5xx after transport retries, or text-matched
    "quota"/"rate" on any exception) raises ``LLMThrottleError``
    instead of falling back — sample-mode is the explicit offline
    path, not an outage substitute.

    ``fail_hard=True`` opts out of the sample fallback on a real failure:
    the call raises ``AgentHardFailure`` so a critical-path caller (the
    reviews) can halt. SAMPLE_MODE still serves a sample (it's the explicit
    offline path, not a failure).
    """

    started_monotonic = time.monotonic()

    if settings.sample_mode:
        sample = sample_factory()
        return AgentRunResult(
            output=sample,
            status="sample_mode",
            duration_ms=_elapsed_ms(started_monotonic),
            raw_response=_dump_sample(sample),
        )

    def _degrade(exc: Exception, *, model_name: str | None) -> AgentRunResult[OutT]:
        if fail_hard:
            raise AgentHardFailure(f"{log_prefix}: {exc}") from exc
        _log.warning("%s: %s; using sample", log_prefix, exc)
        return AgentRunResult(
            output=sample_factory(),
            status="failed",
            duration_ms=_elapsed_ms(started_monotonic),
            model_name=model_name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    try:
        model = llm_model or build_llm_model(settings, model_preference)
    except RuntimeError as exc:
        return _degrade(exc, model_name=None)

    intended_model = getattr(model, "model_name", None)

    try:
        run = await agent.run(user_prompt, model=model)
    except (UnexpectedModelBehavior, ValidationError) as exc:
        return _degrade(exc, model_name=intended_model)
    except ModelHTTPError as exc:
        if exc.status_code in _THROTTLE_HTTP_STATUSES:
            _log.warning(
                "%s: LLM HTTP %d after retries; raising LLMThrottleError",
                log_prefix,
                exc.status_code,
            )
            raise LLMThrottleError(f"HTTP {exc.status_code}: {exc}") from exc
        return _degrade(exc, model_name=intended_model)
    except Exception as exc:
        if _looks_like_llm_throttle(exc):
            _log.warning(
                "%s: LLM throttle (text-matched); raising LLMThrottleError",
                log_prefix,
            )
            raise LLMThrottleError(str(exc)) from exc
        return _degrade(exc, model_name=intended_model)

    return AgentRunResult(
        output=run.output,
        status="success",
        duration_ms=_elapsed_ms(started_monotonic),
        prompt_tokens=_safe_int(getattr(run.usage, "input_tokens", None)),
        completion_tokens=_safe_int(getattr(run.usage, "output_tokens", None)),
        cached_tokens=_safe_int(getattr(run.usage, "cache_read_tokens", None)),
        raw_response=_dump_response(run),
        model_name=getattr(getattr(run, "response", None), "model_name", None) or intended_model,
    )


# ── orchestrated run: writes to both trace sinks ───────────────────────────


@dataclass
class AgentCallSinks:
    trace: TraceWriter
    store: TraceStore | None = None
    round_: RoundHandle | None = None


async def execute_agent_call(
    *,
    agent: Agent[None, OutT],
    user_prompt: str,
    system_prompt: str,
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
    parent_call_id: int | None = None,
    model_preference: ModelPreference = "chat",
    fail_hard: bool = False,
) -> tuple[OutT, AgentCallHandle | None, AgentRunResult[OutT]]:
    """Begin → run → finish across both sinks.

    Returns ``(output, store_handle, run_result)`` — the handle lets
    callers attach side rows (extractions, schema-changes) to the
    same agent_call row.

    ``fail_hard=True`` re-raises ``AgentHardFailure`` on a real failure
    instead of degrading to the sample. The "started" agent_call row is
    left as a visible tombstone (never finished) for the crashed call.
    """

    call: AgentCallHandle | None = None
    if sinks.store is not None and sinks.round_ is not None:
        call = sinks.store.begin_agent_call(
            sinks.round_,
            agent_name=agent_name,
            action=action,
            triggered_by=triggered_by,
            target_agent=target_agent,
            parent_call_id=parent_call_id,
            input_summary=input_summary,
        )

    run_result = await run_agent_or_fallback(
        agent,
        user_prompt,
        settings=settings,
        llm_model=llm_model,
        sample_factory=sample_factory,
        log_prefix=log_prefix,
        model_preference=model_preference,
        fail_hard=fail_hard,
    )
    output_summary = output_summary_fn(run_result.output)

    sinks.trace.record(
        agent_name=agent_name,
        action=action,
        input_summary=input_summary,
        output_summary=output_summary,
        target_agent=target_agent,
        confidence=confidence,
        status=_jsonl_status(run_result.status),
        retry_count=run_result.retry_count,
    )

    if call is not None and sinks.store is not None:
        parsed_dump = (
            run_result.output.model_dump(mode="json", by_alias=True)
            if hasattr(run_result.output, "model_dump")
            else None
        )
        sinks.store.finish_agent_call(
            call,
            status=run_result.status,
            duration_ms=run_result.duration_ms,
            output_summary=output_summary,
            confidence=confidence,
            retry_count=run_result.retry_count,
            model_name=run_result.model_name,
            prompt_tokens=run_result.prompt_tokens,
            completion_tokens=run_result.completion_tokens,
            cached_tokens=run_result.cached_tokens,
            error_type=run_result.error_type,
            error_message=run_result.error_message,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=run_result.raw_response,
            parsed_output=parsed_dump,
        )

    return run_result.output, call, run_result


def record_skipped_agent_call(
    *,
    sinks: AgentCallSinks,
    agent_name: str,
    action: str,
    triggered_by: str,
    target_agent: str | None,
    input_summary: str,
    output_summary: str,
    confidence: float = 1.0,
) -> AgentCallHandle | None:
    """Emit a ``skipped`` row for short-circuit calls (no model invocation)."""

    sinks.trace.record(
        agent_name=agent_name,
        action=action,
        input_summary=input_summary,
        output_summary=output_summary,
        target_agent=target_agent,
        confidence=confidence,
        status="success",
    )
    if sinks.store is None or sinks.round_ is None:
        return None
    call = sinks.store.begin_agent_call(
        sinks.round_,
        agent_name=agent_name,
        action=action,
        triggered_by=triggered_by,
        target_agent=target_agent,
        input_summary=input_summary,
    )
    sinks.store.finish_agent_call(
        call,
        status="skipped",
        duration_ms=0,
        output_summary=output_summary,
        confidence=confidence,
    )
    return call


# ── helpers ────────────────────────────────────────────────────────────────


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dump_response(run: Any) -> str | None:
    response = getattr(run, "response", None)
    if response is None:
        return None
    dump = getattr(response, "model_dump_json", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return None
    return None


def _dump_sample(sample: Any) -> str | None:
    if hasattr(sample, "model_dump_json"):
        try:
            return sample.model_dump_json()
        except Exception:
            return None
    try:
        return json.dumps(sample, default=str)
    except TypeError:
        return None


def _jsonl_status(store_status: str) -> str:
    # JSONL trace only carries success/failed; sample_mode surfaces as success.
    if store_status == "failed":
        return "failed"
    return "success"
