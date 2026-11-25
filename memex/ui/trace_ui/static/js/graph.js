// vis-network KG graph view. Methods read/write `this.state.graph.*` and
// `this._network` / `this._nodes` / `this._edges` / `this._edgeMap`
// (kept off Alpine's reactive proxy by underscore-prefixed names — Alpine
// doesn't trap them, so vis-network's internal mutations don't trigger
// reactivity storms).
//
// Spread into the Alpine factory in app.js.

window.traceUiGraph = {
  async loadGraph() {
    if (!this.state.buildId) return;
    if (typeof vis === "undefined") {
      console.error("vis-network not loaded");
      return;
    }
    this.state.graph.loading = true;
    const params = new URLSearchParams({
      min_mentions: this.state.graph.minMentions,
      limit_nodes: this.state.graph.limitNodes,
      include_isolated: this.state.graph.includeIsolated,
    });
    if (this.state.graph.entityType) {
      params.set("entity_type", this.state.graph.entityType);
    }
    const url =
      `/api/${this.state.kgId}/builds/${this.state.buildId}/graph?` + params;
    try {
      const j = await fetchJson(url);
      this.state.graph.types = j.entity_types || [];
      this.state.graph.nodeCount = j.nodes.length;
      this.state.graph.edgeCount = j.edges.length;
      this.renderGraph(j.nodes, j.edges);
    } finally {
      this.state.graph.loading = false;
    }
  },

  renderGraph(rawNodes, rawEdges) {
    const container = document.getElementById("graph-canvas");
    if (!container) return;

    // Color palette indexed by group (entity_type) — stable per build.
    const groupColors = {};
    const palette = [
      "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
      "#ec4899", "#14b8a6", "#f97316", "#0ea5e9", "#84cc16",
      "#a855f7", "#22d3ee", "#eab308", "#f43f5e", "#3b82f6",
    ];
    let nextColorIdx = 0;
    const colorFor = (group) => {
      if (!groupColors[group]) {
        groupColors[group] = palette[nextColorIdx % palette.length];
        nextColorIdx++;
      }
      return groupColors[group];
    };

    const nodes = rawNodes.map((n) => ({
      id: n.id,
      label: n.label,
      group: n.group,
      title: n.title,
      value: n.value,
      color: { background: colorFor(n.group), border: "#475569" },
      font: { size: 12, color: "#EAF0F8" },
    }));
    this._edgeMap = new Map();
    const edges = rawEdges.map((e, i) => {
      this._edgeMap.set(i, {
        from: e.from,
        to: e.to,
        predicate: e.label,
        sample_doc_id: e.sample_doc_id,
        sample_agent_call_id: e.sample_agent_call_id,
      });
      return {
        id: i,
        from: e.from,
        to: e.to,
        label: e.label,
        title: e.title,
        value: e.value,
        arrows: "to",
        color: { color: "#5b6b85", highlight: "#57E6CE" },
        font: { size: 10, color: "#8C9AB0", strokeWidth: 0 },
        smooth: { type: "dynamic" },
      };
    });

    this._nodes = new vis.DataSet(nodes);
    this._edges = new vis.DataSet(edges);

    const options = {
      nodes: {
        shape: "dot",
        scaling: { min: 8, max: 40, label: { enabled: true, min: 10, max: 18 } },
        borderWidth: 1,
      },
      edges: {
        scaling: { min: 1, max: 6 },
        arrows: { to: { scaleFactor: 0.5 } },
      },
      physics: {
        enabled: true,
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -50,
          centralGravity: 0.01,
          springLength: 100,
          springConstant: 0.08,
          damping: 0.4,
          avoidOverlap: 0.5,
        },
        stabilization: { iterations: 200, fit: true },
      },
      interaction: {
        hover: true,
        tooltipDelay: 150,
        navigationButtons: false,
        keyboard: true,
      },
    };

    if (this._network) this._network.destroy();
    this._network = new vis.Network(container, { nodes: this._nodes, edges: this._edges }, options);

    this._network.on("click", (params) => this.onGraphClick(params));
    this._network.once("stabilizationIterationsDone", () => {
      this._network.setOptions({ physics: false });
    });
  },

  onGraphClick(params) {
    if (params.nodes && params.nodes.length > 0) {
      this.selectEntity(params.nodes[0]);
      return;
    }
    if (params.edges && params.edges.length > 0) {
      const meta = this._edgeMap?.get(params.edges[0]);
      if (meta && meta.from != null && meta.to != null) {
        this.selectRelation(meta.from, meta.to, meta.predicate);
      }
    }
  },

  focusGraphNode(entityId) {
    if (this._network) {
      try {
        this._network.focus(entityId, { scale: 1.2, animation: true });
        this._network.selectNodes([entityId]);
      } catch (_) {
        // node may not be in the current filter view; just load the detail
      }
    }
    this.selectEntity(entityId);
  },

  fitGraph() {
    if (this._network) this._network.fit({ animation: true });
  },

  toggleGraphFullscreen() {
    const el = document.getElementById("graph-view");
    if (!el) return;
    // bind once: vis-network needs a redraw+fit after the container resizes,
    // both entering and exiting fullscreen.
    if (!this._fsBound) {
      document.addEventListener("fullscreenchange", () => {
        this.state.graph.fullscreen = !!document.fullscreenElement;
        setTimeout(() => {
          try { this._network?.redraw(); this._network?.fit({ animation: false }); } catch (_) {}
        }, 150);
      });
      this._fsBound = true;
    }
    if (document.fullscreenElement) document.exitFullscreen?.();
    else el.requestFullscreen?.();
  },
};
