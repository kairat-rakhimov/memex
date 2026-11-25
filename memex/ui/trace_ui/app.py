"""FastAPI app backing the trace UI.

Discovers builds via ``<kg_root>/<kg_id>/kg.db``; exposes a read-only
REST surface plus the static SPA shell. Reads open short-lived
TraceStore instances per request to sidestep WAL writer/reader
ordering during an in-flight build.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from memex.observability import TraceStore
from memex.observability.store import kg_db_path
from memex.ui.nav import render_nav

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_INCLUDE_RE = re.compile(r"{%\s*include\s+\"([^\"]+)\"\s*%}")


def _render_index() -> str:
    """Expand ``{% include "..." %}`` directives in ``index.html``; one
    pass, no nesting. Called once at boot, not per-request."""

    base = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

    def _resolve(match: re.Match[str]) -> str:
        partial_path = TEMPLATES_DIR / match.group(1)
        return partial_path.read_text(encoding="utf-8")

    return _INCLUDE_RE.sub(_resolve, base)


def _discover_kg_ids(kg_root: Path) -> list[str]:
    if not kg_root.exists():
        return []
    out: list[str] = []
    for entry in sorted(kg_root.iterdir()):
        if not entry.is_dir():
            continue
        if kg_db_path(entry).exists():
            out.append(entry.name)
    return out


def _store(kg_root: Path, kg_id: str) -> TraceStore:
    kg_dir = kg_root / kg_id
    if not kg_db_path(kg_dir).exists():
        raise HTTPException(status_code=404, detail=f"no trace DB for kg_id={kg_id}")
    return TraceStore.open(kg_dir)


def _load_parsed(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    raw = row.get("parsed_output_json")
    if raw:
        try:
            row["parsed_output"] = json.loads(raw)
        except Exception:
            row["parsed_output"] = raw
    return row


def create_app(kg_root: Path | str = "kg") -> FastAPI:
    """``kg_root`` is the directory containing per-build kg_id subdirs."""

    root = Path(kg_root).resolve()
    app = FastAPI(title="memex kg-loop trace UI")

    # Compose index.html at app-creation time. Partials are stable for the
    # life of the process; the per-request cost is one preassembled string.
    index_html = _render_index()

    # ── routes: API ───────────────────────────────────────────────────────

    @app.get("/api/kg_ids")
    def list_kg_ids() -> JSONResponse:
        return JSONResponse({"kg_ids": _discover_kg_ids(root), "kg_root": str(root)})

    @app.get("/api/{kg_id}/builds")
    def list_builds(kg_id: str) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            return JSONResponse({"kg_id": kg_id, "builds": store.list_builds()})
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}")
    def get_build(kg_id: str, build_id: int) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            build = store.get_build(build_id)
            if build is None:
                raise HTTPException(status_code=404, detail="build not found")
            return JSONResponse({
                "kg_id": kg_id,
                "build": build,
                "rounds": store.list_rounds(build_id),
                "schema_changes": store.list_schema_changes(build_id),
                "top_entities": store.top_entities(limit=30),
            })
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}/failures")
    def list_failures(kg_id: str, build_id: int, limit: int = 50) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            return JSONResponse({
                "kg_id": kg_id,
                "build_id": build_id,
                "failures": store.recent_failures(build_id, limit=limit),
            })
        finally:
            store.close()

    @app.get("/api/{kg_id}/rounds/{round_id}/agent_calls")
    def list_agent_calls(kg_id: str, round_id: int) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            return JSONResponse({
                "kg_id": kg_id,
                "round_id": round_id,
                "agent_calls": store.list_agent_calls(round_id),
                "fetched_docs": store.list_fetched_docs(round_id),
            })
        finally:
            store.close()

    @app.get("/api/{kg_id}/agent_calls/{agent_call_id}")
    def get_agent_call(kg_id: str, agent_call_id: int) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            call = _load_parsed(store.get_agent_call(agent_call_id))
            if call is None:
                raise HTTPException(status_code=404, detail="agent_call not found")
            return JSONResponse({"kg_id": kg_id, "agent_call": call})
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}/graph")
    def get_graph(
        kg_id: str,
        build_id: int,
        min_mentions: int = 1,
        entity_type: str | None = None,
        predicate: str | None = None,
        limit_nodes: int = 500,
        include_isolated: bool = False,
    ) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            data = store.graph_data(
                min_mentions=min_mentions,
                entity_type=entity_type,
                predicate=predicate,
                limit_nodes=limit_nodes,
                include_isolated=include_isolated,
            )
            return JSONResponse({"kg_id": kg_id, "build_id": build_id, **data})
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}/entities/{entity_id}")
    def get_entity(kg_id: str, build_id: int, entity_id: int) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            entity = store.entity_detail(entity_id)
            if entity is None:
                raise HTTPException(status_code=404, detail="entity not found")
            return JSONResponse({"kg_id": kg_id, "build_id": build_id, "entity": entity})
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}/relation")
    def get_relation(
        kg_id: str,
        build_id: int,
        subject_id: int,
        object_id: int,
        predicate: str,
    ) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            relation = store.relation_detail(subject_id, object_id, predicate)
            if relation is None:
                raise HTTPException(status_code=404, detail="relation not found")
            return JSONResponse({"kg_id": kg_id, "build_id": build_id, "relation": relation})
        finally:
            store.close()

    # ── routes: mutation log (three lenses) ──────────────

    @app.get("/api/{kg_id}/builds/{build_id}/mutations")
    def list_mutations_for_target(
        kg_id: str,
        build_id: int,
        target_kind: str,
        target_id: str,
        limit: int = 100,
    ) -> JSONResponse:
        """Lens 1: full mutation history for one entity / relation /
        predicate, ordered oldest → newest."""

        store = _store(root, kg_id)
        try:
            rows = store.list_mutations_for_target(
                target_kind=target_kind,
                target_id=target_id,
                limit=limit,
            )
            return JSONResponse({
                "kg_id": kg_id,
                "build_id": build_id,
                "target_kind": target_kind,
                "target_id": target_id,
                "mutations": rows,
            })
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}/mutations_by_round")
    def mutations_by_round(kg_id: str, build_id: int) -> JSONResponse:
        """Lens 2: mutation count per round, broken out by kind."""

        store = _store(root, kg_id)
        try:
            return JSONResponse({
                "kg_id": kg_id,
                "build_id": build_id,
                "counts": store.mutation_counts_by_round(),
            })
        finally:
            store.close()

    @app.get("/api/{kg_id}/builds/{build_id}/agent_calls/{agent_call_id}/mutations")
    def list_mutations_for_call(
        kg_id: str, build_id: int, agent_call_id: int
    ) -> JSONResponse:
        """Lens 3: every mutation linked to one agent call. Parent merge
        rows come before their endpoint-rewrite children."""

        store = _store(root, kg_id)
        try:
            return JSONResponse({
                "kg_id": kg_id,
                "build_id": build_id,
                "agent_call_id": agent_call_id,
                "mutations": store.list_mutations_for_call(agent_call_id),
            })
        finally:
            store.close()

    # ── routes: static SPA ────────────────────────────────────────────────

    @app.get("/")
    def index(request: Request) -> HTMLResponse:
        root_path = request.scope.get("root_path", "")
        html = (
            index_html.replace("/static/", f"{root_path}/static/")
            .replace("__API_BASE__", root_path)
            .replace("__NAV__", render_nav("build", root_path))
        )
        return HTMLResponse(html)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


__all__ = ["create_app"]
