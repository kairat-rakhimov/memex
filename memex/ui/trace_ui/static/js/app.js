// Alpine SPA entry — registers the `traceUI` component via the
// `alpine:init` event. Composes the component from helper modules
// (formatters / mutations / graph) loaded earlier in HTML order; each
// exposes a plain object on `window` that we spread into the factory
// return.
//
// The `alpine:init` listener pattern is intentional: it sidesteps the
// race between Alpine.start() firing on DOMContentLoaded and our
// listener registering. Alpine fires `alpine:init` synchronously inside
// start(); as long as we register the listener before Alpine loads, our
// factory gets registered before Alpine walks the DOM.

document.addEventListener("alpine:init", () => {
  Alpine.data("traceUI", () => ({
    state: {
      kgRoot: "",
      kgIds: [],
      kgId: null,
      builds: [],
      buildId: null,
      build: null,
      rounds: [],
      schemaChanges: [],
      topEntities: [],
      roundId: null,
      calls: [],
      fetchedDocs: [],
      callId: null,
      call: null,
      failures: [],
      showFailures: false,
      // middleView: 'overview' | 'graph'
      middleView: "overview",
      // rightView: 'call' | 'entity' | 'relation' | 'empty'
      rightView: "empty",
      entity: null,
      relation: null,
      graph: {
        loading: false,
        fullscreen: false,
        minMentions: 1,
        entityType: "",
        limitNodes: 500,
        includeIsolated: false,
        types: [],
        nodeCount: 0,
        edgeCount: 0,
      },
      // three mutation lenses (see mutations.js)
      mutations: {
        forTarget: [],          // lens 1: per entity / relation / predicate
        byRound: [],            // lens 2: build overview rollup
        forCall: [],            // lens 3: per agent call
      },
    },

    // vis-network handles, kept off Alpine's reactive proxy by underscore
    // prefix so its internal mutations don't trigger reactivity storms.
    _network: null,
    _nodes: null,
    _edges: null,
    _edgeMap: null,

    // ── lifecycle ───────────────────────────────────────────────────────

    async bootstrap() {
      const j = await fetchJson("/api/kg_ids");
      this.state.kgRoot = j.kg_root;
      this.state.kgIds = j.kg_ids;
      if (this.state.kgIds.length === 0) return;
      if (!this.state.kgId || !this.state.kgIds.includes(this.state.kgId)) {
        this.state.kgId = this.state.kgIds[0];
      }
      await this.onKgIdChange();
    },

    async onKgIdChange() {
      this.resetSelections();
      const j = await fetchJson(`/api/${this.state.kgId}/builds`);
      this.state.builds = j.builds;
      if (this.state.builds.length) {
        await this.selectBuild(this.state.builds[0].id);
      }
    },

    resetSelections() {
      this.state.buildId = null;
      this.state.build = null;
      this.state.rounds = [];
      this.state.roundId = null;
      this.state.calls = [];
      this.state.callId = null;
      this.state.call = null;
      this.state.entity = null;
      this.state.relation = null;
      this.state.showFailures = false;
      this.state.middleView = "overview";
      this.state.rightView = "empty";
    },

    // ── selection flow ──────────────────────────────────────────────────

    async selectBuild(id) {
      this.state.buildId = id;
      this.state.roundId = null;
      this.state.callId = null;
      this.state.call = null;
      this.state.entity = null;
      this.state.relation = null;
      this.state.calls = [];
      this.state.fetchedDocs = [];
      this.state.showFailures = false;
      this.state.rightView = "empty";
      this.state.mutations.forTarget = [];
      this.state.mutations.forCall = [];
      const j = await fetchJson(`/api/${this.state.kgId}/builds/${id}`);
      this.state.build = j.build;
      this.state.rounds = j.rounds;
      this.state.schemaChanges = j.schema_changes;
      this.state.topEntities = j.top_entities;
      await this.loadMutationsByRound();
      if (this.state.middleView === "graph") {
        await this.loadGraph();
      }
    },

    async selectRound(id) {
      this.state.roundId = id;
      this.state.callId = null;
      this.state.call = null;
      this.state.entity = null;
      this.state.relation = null;
      this.state.showFailures = false;
      this.state.rightView = "empty";
      this.state.middleView = "overview";
      const j = await fetchJson(
        `/api/${this.state.kgId}/rounds/${id}/agent_calls`
      );
      this.state.calls = j.agent_calls;
      this.state.fetchedDocs = j.fetched_docs;
    },

    async selectCall(id) {
      this.state.callId = id;
      this.state.entity = null;
      this.state.relation = null;
      this.state.rightView = "call";
      this.state.mutations.forCall = [];
      const j = await fetchJson(`/api/${this.state.kgId}/agent_calls/${id}`);
      this.state.call = j.agent_call;
      this.state.mutations.forCall = await this._fetchMutationsForCall(id);
    },

    async selectEntity(entityId) {
      this.state.call = null;
      this.state.callId = null;
      this.state.relation = null;
      this.state.rightView = "entity";
      this.state.mutations.forTarget = [];
      const j = await fetchJson(
        `/api/${this.state.kgId}/builds/${this.state.buildId}/entities/${entityId}`
      );
      this.state.entity = j.entity;
      this.state.mutations.forTarget = await this._fetchMutationsForEntity(
        entityId,
        j.entity?.canonical_name
      );
    },

    async selectRelation(subjectId, objectId, predicate) {
      this.state.call = null;
      this.state.callId = null;
      this.state.entity = null;
      this.state.rightView = "relation";
      this.state.relation = null;
      const params = new URLSearchParams({
        subject_id: subjectId,
        object_id: objectId,
        predicate,
      });
      const j = await fetchJson(
        `/api/${this.state.kgId}/builds/${this.state.buildId}/relation?` + params
      );
      this.state.relation = j.relation;
    },

    async loadFailures() {
      this.state.showFailures = true;
      this.state.roundId = null;
      this.state.middleView = "overview";
      const j = await fetchJson(
        `/api/${this.state.kgId}/builds/${this.state.buildId}/failures`
      );
      this.state.failures = j.failures;
    },

    // ── derived state ───────────────────────────────────────────────────

    currentRound() {
      return this.state.rounds.find((r) => r.id === this.state.roundId);
    },

    middleTitle() {
      if (this.state.middleView === "graph") return "Knowledge graph";
      if (this.state.showFailures) return "Failures + sample_mode";
      if (this.state.roundId) {
        const r = this.currentRound();
        return "Round " + (r ? r.round_number : "") + " · agent calls";
      }
      return "Build overview";
    },

    setMiddleView(view) {
      this.state.middleView = view;
      this.state.showFailures = false;
      if (view === "graph") {
        // defer to next tick so the canvas div is rendered before vis-network
        // measures its container size
        requestAnimationFrame(() => this.loadGraph());
      }
    },

    // ── helpers spread from sibling modules ─────────────────────────────
    // formatters.js → statusClass / formatDur / formatMs / prettyJson
    // mutations.js  → loadMutationsByRound / _fetchMutationsForCall /
    //                 _fetchMutationsForEntity / roundMutationSummary
    // graph.js      → loadGraph / renderGraph / onGraphClick /
    //                 focusGraphNode / fitGraph

    ...window.traceUiFormatters,
    ...window.traceUiMutations,
    ...window.traceUiGraph,
  }));
});
