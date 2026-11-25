"""FastAPI app: ``POST /run`` (chat / hypothesize / build) and ``GET /health``.

``POST /run`` enforces ``RUN_TIMEOUT_SECONDS``, returns structured
success/error responses, and writes one JSONL trace per request.
``GET /health`` is the liveness check (no LLM call).
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from memex.common.config import Settings
from memex.common.schemas import (
    ErrorInfo,
    HealthResponse,
    RunRequest,
    RunResponse,
)
from memex.common.trace import redact
from memex.run_router import run_workflow

SERVICE_NAME = "memex"
SERVICE_VERSION = "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="memex",
        description="Multi-agent knowledge-graph system for hypothesis generation.",
        version=SERVICE_VERSION,
    )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "message": "POST /run to chat, hypothesize or build; GET /health for liveness.",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            port=settings.port,
            llm_configured=settings.llm_configured,
            sample_mode=settings.sample_mode,
        )

    @app.post("/run", response_model=RunResponse)
    async def run_endpoint(run_request: RunRequest) -> JSONResponse:
        return await _execute_run(run_request, settings)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            request_id=str(uuid.uuid4()),
            http_status=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Request body must include a valid JSON 'input' object.",
            details={"errors": exc.errors()},
            settings=settings,
            runtime=0.0,
            retryable=False,
        )

    return app


async def _execute_run(run_request: RunRequest, settings: Settings) -> JSONResponse:
    started = time.monotonic()
    request_id = run_request.request_id or str(uuid.uuid4())

    try:
        outcome = await asyncio.wait_for(
            run_workflow(run_request=run_request, settings=settings),
            timeout=settings.run_timeout_seconds,
        )

        response = RunResponse(
            status="success",
            request_id=outcome.request_id,
            trace_id=outcome.request_id,
            output=outcome.output,
            agents_used=outcome.agents_used,
            agents=outcome.agents,
            runtime_seconds=outcome.runtime_seconds,
            trace_file=outcome.trace_path,
            sample_mode=settings.sample_mode,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response.model_dump(),
        )

    except TimeoutError:
        return _error_response(
            request_id=request_id,
            http_status=status.HTTP_408_REQUEST_TIMEOUT,
            code="TIMEOUT",
            message="Execution exceeded RUN_TIMEOUT_SECONDS.",
            details={"limit_seconds": settings.run_timeout_seconds},
            settings=settings,
            runtime=round(time.monotonic() - started, 3),
            retryable=True,
        )

    except ValidationError as exc:
        return _error_response(
            request_id=request_id,
            http_status=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Request body failed validation.",
            details={"errors": exc.errors()},
            settings=settings,
            runtime=round(time.monotonic() - started, 3),
            retryable=False,
        )

    except (ValueError, FileNotFoundError) as exc:
        return _error_response(
            request_id=request_id,
            http_status=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message=redact(str(exc)),
            details={},
            settings=settings,
            runtime=round(time.monotonic() - started, 3),
            retryable=False,
        )

    except Exception as exc:
        return _error_response(
            request_id=request_id,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="The agent workflow failed during execution.",
            details={"error": redact(str(exc))},
            settings=settings,
            runtime=round(time.monotonic() - started, 3),
            retryable=False,
        )


def _error_response(
    *,
    request_id: str,
    http_status: int,
    code: str,
    message: str,
    details: dict[str, object],
    settings: Settings,
    runtime: float,
    retryable: bool,
) -> JSONResponse:
    response = RunResponse(
        status="error",
        request_id=request_id,
        trace_id=request_id,
        output=None,
        agents_used=[],
        agents=[],
        runtime_seconds=runtime,
        sample_mode=settings.sample_mode,
        error=ErrorInfo(code=code, message=message, details=details, retryable=retryable),
    )
    return JSONResponse(
        status_code=http_status,
        content=response.model_dump(),
    )


# Module-level ASGI app for uvicorn ("memex.api:app").
app = create_app()
