"""Unified read-only trace UI — one server (port 8001) over the build-loop trace
and the query-reasoning trace.

Mounts the two existing apps under ``/build`` and ``/query`` and serves a shared
top-nav to switch between them. Both sub-apps discover per-kg_id subdirs under the
same ``kg_root`` (default ``kg/``), so every built graph shows up in both
views. ``/`` redirects to the build view.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse


def create_app(kg_root: Path | str = "kg") -> FastAPI:
    # Imported lazily: the sub-apps import memex.ui.nav, so a module-level import
    # here would be circular (memex.ui.__init__ → memex.ui.app → trace_ui → memex.ui.nav).
    from memex.ui.query_ui import create_app as create_query_app
    from memex.ui.trace_ui import create_app as create_trace_app

    app = FastAPI(title="memex · agent traces")
    app.mount("/build", create_trace_app(kg_root))
    app.mount("/query", create_query_app(kg_root))

    @app.get("/")
    def index() -> RedirectResponse:
        return RedirectResponse(url="build/")

    return app


__all__ = ["create_app"]
