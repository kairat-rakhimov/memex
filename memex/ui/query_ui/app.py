"""FastAPI app backing the query-run inner-thinking viewer.

Discovers query traces via ``<kg_root>/<kg_id>/query_trace.db``; read-only REST
surface + a self-contained single-page shell (Alpine + Tailwind via CDN). Short-
lived ``QueryTraceStore`` per request (WAL-safe against an in-flight run).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from memex.observability.query_store import QueryTraceStore
from memex.ui.nav import render_nav


def _has_query_db(kg_dir: Path) -> bool:
    return (kg_dir / "query_trace.db").exists()


def _discover_kg_ids(kg_root: Path) -> list[str]:
    if not kg_root.exists():
        return []
    return [e.name for e in sorted(kg_root.iterdir()) if e.is_dir() and _has_query_db(e)]


def _store(kg_root: Path, kg_id: str) -> QueryTraceStore:
    kg_dir = kg_root / kg_id
    if not _has_query_db(kg_dir):
        raise HTTPException(status_code=404, detail=f"no query_trace.db for kg_id={kg_id}")
    return QueryTraceStore.open(kg_dir)


def _parse_call(call: dict[str, Any] | None) -> dict[str, Any] | None:
    if call is None:
        return None
    raw = call.get("parsed_output_json")
    if raw:
        try:
            call["parsed_output"] = json.loads(raw)
        except Exception:
            call["parsed_output"] = raw
    return call


def create_app(kg_root: Path | str = "kg") -> FastAPI:
    root = Path(kg_root).resolve()
    app = FastAPI(title="memex query-run viewer")

    @app.get("/api/kg_ids")
    def list_kg_ids() -> JSONResponse:
        return JSONResponse({"kg_ids": _discover_kg_ids(root), "kg_root": str(root)})

    @app.get("/api/{kg_id}/runs")
    def list_runs(kg_id: str) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            return JSONResponse({"kg_id": kg_id, "runs": store.list_runs()})
        finally:
            store.close()

    @app.get("/api/{kg_id}/runs/{run_db_id}")
    def get_run(kg_id: str, run_db_id: int) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            run = store.get_run(run_db_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            return JSONResponse({"kg_id": kg_id, "run": run, "calls": store.list_agent_calls(run_db_id)})
        finally:
            store.close()

    @app.get("/api/{kg_id}/calls/{call_id}")
    def get_call(kg_id: str, call_id: int) -> JSONResponse:
        store = _store(root, kg_id)
        try:
            call = _parse_call(store.get_agent_call(call_id))
            if call is None:
                raise HTTPException(status_code=404, detail="call not found")
            return JSONResponse({"kg_id": kg_id, "call": call, "steps": store.list_steps(call_id)})
        finally:
            store.close()

    @app.get("/")
    def index(request: Request) -> HTMLResponse:
        root_path = request.scope.get("root_path", "")
        html = _INDEX_HTML.replace("__API_BASE__", root_path).replace(
            "__NAV__", render_nav("query", root_path)
        )
        return HTMLResponse(html)

    return app


__all__ = ["create_app"]


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>memex · query-run inner-thinking viewer</title>
<script>window.API_BASE = "__API_BASE__";</script>
<script src="https://cdn.tailwindcss.com"></script>
<script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
<style>
  /* theme palette */
  :root{
    --bg:#0A0E15; --bg-alt:#0C111B; --surface:#131B29; --surface-2:#172234;
    --ink:#EAF0F8; --muted:#8C9AB0; --faint:#5E6B80;
    --accent:#57E6CE; --line:rgba(120,150,190,.18); --line-strong:rgba(120,150,190,.30);
  }
  .trail-thinking { color:var(--muted); font-style:italic; }
  .trail-text { color:var(--ink); }
  .trail-tool_call { color:var(--accent); }
  .trail-tool_return { color:#34D399; }
  pre { white-space:pre-wrap; word-break:break-word; }

  /* retheme: remap the slate-blue utilities onto the theme tokens (loaded after
     the Tailwind CDN; !important guards against its runtime injection order) */
  body { background-color:var(--bg) !important; color:var(--ink) !important; }
  *, ::before, ::after { border-color:var(--line) !important; }
  ::selection { background:rgba(87,230,206,.18); }
  .bg-slate-900 { background-color:var(--bg) !important; }
  .bg-slate-800 { background-color:var(--surface) !important; }
  .bg-slate-700 { background-color:var(--surface-2) !important; }
  .bg-slate-950 { background-color:#070A10 !important; }
  .bg-slate-800\/60 { background-color:rgba(19,27,41,.6) !important; }
  .hover\:bg-slate-800:hover { background-color:var(--surface-2) !important; }
  .border-slate-700 { border-color:var(--line-strong) !important; }
  .border-slate-800 { border-color:var(--line) !important; }
  .text-slate-100 { color:var(--ink) !important; }
  .text-slate-400 { color:#A7B4C8 !important; }
  .text-slate-500 { color:var(--muted) !important; }
  .text-slate-600 { color:var(--faint) !important; }
  .text-blue-300 { color:var(--accent) !important; }
  select, option { background-color:var(--surface) !important; color:var(--ink) !important; }
</style>
</head>
<body class="bg-slate-900 text-slate-100 h-screen overflow-hidden flex flex-col" x-data="viewer()" x-init="init()">
__NAV__
<div class="flex flex-1 min-h-0">

  <!-- left: kg + runs -->
  <aside class="w-72 border-r border-slate-700 flex flex-col">
    <div class="p-3 border-b border-slate-700">
      <div class="text-xs uppercase tracking-wide text-slate-400 mb-1">knowledge graph</div>
      <select x-model="kgId" @change="loadRuns()" class="w-full bg-slate-800 rounded p-1 text-sm">
        <template x-for="k in kgIds" :key="k"><option :value="k" x-text="k"></option></template>
      </select>
    </div>
    <div class="overflow-y-auto flex-1">
      <template x-for="r in runs" :key="r.id">
        <button @click="loadRun(r.id)" class="w-full text-left px-3 py-2 border-b border-slate-800 hover:bg-slate-800"
                :class="run && run.id===r.id ? 'bg-slate-800' : ''">
          <div class="flex justify-between items-center">
            <span class="text-xs px-1.5 py-0.5 rounded"
                  :class="r.kind==='chat' ? 'bg-sky-700' : 'bg-fuchsia-700'" x-text="r.kind"></span>
            <span class="text-[10px] text-slate-500" x-text="r.status"></span>
          </div>
          <div class="text-sm mt-1 line-clamp-2" x-text="r.question || '(no question)'"></div>
          <div class="text-[10px] text-slate-500 mt-1" x-text="r.started_at"></div>
        </button>
      </template>
      <div x-show="runs.length===0" class="p-3 text-sm text-slate-500">No query runs yet. Run `memex chat` / `memex hypothesize`.</div>
    </div>
  </aside>

  <!-- middle: call graph -->
  <section class="w-96 border-r border-slate-700 overflow-y-auto">
    <div class="p-3 border-b border-slate-700 text-xs uppercase tracking-wide text-slate-400">agent calls</div>
    <template x-for="c in tree" :key="c.id">
      <button @click="loadCall(c.id)" class="w-full text-left px-2 py-1.5 border-b border-slate-800 hover:bg-slate-800 flex items-start gap-1"
              :class="call && call.id===c.id ? 'bg-slate-800' : ''" :style="`padding-left:${0.5 + c.depth*1.1}rem`">
        <span class="text-slate-600" x-text="c.is_tool_call ? '🔧' : '▸'"></span>
        <span class="flex-1">
          <span class="text-sm" :class="c.is_tool_call ? 'text-blue-300' : 'font-medium'" x-text="c.agent_name"></span>
          <span class="text-[10px] text-slate-500 ml-1" x-text="c.action"></span>
          <span class="text-[10px] ml-1 px-1 rounded"
                :class="statusColor(c.status)" x-text="c.status"></span>
        </span>
      </button>
    </template>
  </section>

  <!-- right: detail + inner-thinking trail -->
  <main class="flex-1 overflow-y-auto p-4">
    <template x-if="!call"><div class="text-slate-500">Select an agent call to inspect its inner thinking.</div></template>
    <template x-if="call">
      <div>
        <div class="flex items-center gap-2 mb-2">
          <span class="text-lg font-semibold" x-text="call.agent_name"></span>
          <span class="text-sm text-slate-400" x-text="call.action"></span>
          <span class="text-xs px-1.5 py-0.5 rounded" :class="statusColor(call.status)" x-text="call.status"></span>
        </div>
        <div class="text-xs text-slate-500 mb-3">
          <span x-text="'model: ' + (call.model_name || '—')"></span> ·
          <span x-text="'tokens: ' + (call.prompt_tokens||0) + '/' + (call.completion_tokens||0)"></span> ·
          <span x-text="(call.duration_ms||0) + 'ms'"></span>
          <template x-if="call.target_agent"><span x-text="' · → ' + call.target_agent"></span></template>
        </div>

        <!-- reasoning field (first-class inner thinking) -->
        <template x-if="reasoning()">
          <div class="bg-amber-950/40 border border-amber-800/50 rounded p-3 mb-3">
            <div class="text-xs uppercase tracking-wide text-amber-400 mb-1">💭 reasoning</div>
            <div class="text-sm text-amber-100" x-text="reasoning()"></div>
          </div>
        </template>

        <!-- inner-thinking trail (ReAct trajectory) -->
        <template x-if="steps.length">
          <div class="bg-slate-800/60 rounded p-3 mb-3">
            <div class="text-xs uppercase tracking-wide text-slate-400 mb-2">inner-thinking trail</div>
            <template x-for="(s,i) in steps" :key="i">
              <div class="mb-1.5 text-sm" :class="'trail-'+s.kind">
                <template x-if="s.kind==='thinking'"><span x-text="'🧠 '+s.content"></span></template>
                <template x-if="s.kind==='text'"><span x-text="s.content"></span></template>
                <template x-if="s.kind==='tool_call'"><span x-text="'🔧 '+s.tool_name+'('+(s.args_json||'')+')'"></span></template>
                <template x-if="s.kind==='tool_return'"><span x-text="'↳ '+(s.result||'')"></span></template>
              </div>
            </template>
          </div>
        </template>

        <!-- collapsible raw panes -->
        <details class="mb-2"><summary class="cursor-pointer text-xs text-slate-400">parsed output (JSON)</summary>
          <pre class="text-xs bg-slate-950 rounded p-2 mt-1" x-text="pretty(call.parsed_output)"></pre></details>
        <details class="mb-2"><summary class="cursor-pointer text-xs text-slate-400">user prompt</summary>
          <pre class="text-xs bg-slate-950 rounded p-2 mt-1" x-text="call.user_prompt||'—'"></pre></details>
        <details class="mb-2"><summary class="cursor-pointer text-xs text-slate-400">system prompt</summary>
          <pre class="text-xs bg-slate-950 rounded p-2 mt-1" x-text="call.system_prompt||'—'"></pre></details>
      </div>
    </template>
  </main>
</div>

<script>
function viewer(){
  return {
    kgIds:[], kgId:'', runs:[], run:null, calls:[], tree:[], call:null, steps:[],
    api(u){ return (window.API_BASE||'') + u; },
    async init(){
      const r = await (await fetch(this.api('/api/kg_ids'))).json();
      this.kgIds = r.kg_ids; if(this.kgIds.length){ this.kgId=this.kgIds[0]; await this.loadRuns(); }
    },
    async loadRuns(){
      this.run=null; this.calls=[]; this.tree=[]; this.call=null; this.steps=[];
      const r = await (await fetch(this.api(`/api/${this.kgId}/runs`))).json(); this.runs=r.runs;
      if(this.runs.length) await this.loadRun(this.runs[0].id);
    },
    async loadRun(id){
      const r = await (await fetch(this.api(`/api/${this.kgId}/runs/${id}`))).json();
      this.run=r.run; this.calls=r.calls; this.tree=this.buildTree(r.calls); this.call=null; this.steps=[];
      if(this.tree.length) await this.loadCall(this.tree[0].id);
    },
    buildTree(calls){
      const byId={}; calls.forEach(c=>byId[c.id]=c);
      const depth=c=>{ let d=0,p=c.parent_call_id; while(p&&byId[p]){d++;p=byId[p].parent_call_id;} return d; };
      // order: depth-first by parent, preserving sequence
      const children={}; calls.forEach(c=>{const k=c.parent_call_id||0;(children[k]=children[k]||[]).push(c);});
      const out=[]; const walk=(pid)=>{(children[pid]||[]).forEach(c=>{c.depth=depth(c);out.push(c);walk(c.id);});};
      walk(0); return out;
    },
    async loadCall(id){
      const r = await (await fetch(this.api(`/api/${this.kgId}/calls/${id}`))).json();
      this.call=r.call; this.steps=r.steps;
    },
    reasoning(){ const p=this.call&&this.call.parsed_output; return p&&typeof p==='object'?(p.reasoning||''):''; },
    pretty(o){ try{return JSON.stringify(o,null,2);}catch(e){return String(o);} },
    statusColor(s){ return {success:'bg-emerald-800',sample_mode:'bg-slate-700',failed:'bg-red-800',started:'bg-yellow-800'}[s]||'bg-slate-700'; },
  };
}
</script>
</body>
</html>"""
