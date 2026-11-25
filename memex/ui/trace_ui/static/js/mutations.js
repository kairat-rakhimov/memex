// kg_mutations lens loaders + round-detail rollup.
//
// Three lenses:
//   1. forTarget — entity / relation / predicate history (right pane)
//   2. byRound   — build overview rollup
//   3. forCall   — every mutation emitted by one agent call
//
// Spread into the Alpine factory in app.js; reads `this.state.mutations.*`
// and `this.state.kgId`.

window.traceUiMutations = {
  async loadMutationsByRound() {
    if (!this.state.buildId) return;
    try {
      const j = await fetchJson(
        `/api/${this.state.kgId}/builds/${this.state.buildId}/mutations_by_round`
      );
      this.state.mutations.byRound = j.counts || [];
    } catch (_) {
      this.state.mutations.byRound = [];
    }
  },

  async _fetchMutationsForCall(callId) {
    if (!this.state.buildId) return [];
    try {
      const mj = await fetchJson(
        `/api/${this.state.kgId}/builds/${this.state.buildId}/agent_calls/${callId}/mutations`
      );
      return mj.mutations || [];
    } catch (_) {
      return [];
    }
  },

  // Resolver writes kg_mutations rows keyed on canonical_name (the
  // survivor entity's name at merge time), not row id. Look up both
  // keys for robustness — older mutation rows may use either form. We
  // de-dupe by id.
  async _fetchMutationsForEntity(entityId, entityName) {
    const targets = [entityName, String(entityId)].filter(Boolean);
    const merged = [];
    for (const tid of targets) {
      try {
        const mj = await fetchJson(
          `/api/${this.state.kgId}/builds/${this.state.buildId}/mutations?` +
            new URLSearchParams({ target_kind: "entity", target_id: tid })
        );
        for (const m of mj.mutations || []) {
          if (!merged.some((x) => x.id === m.id)) merged.push(m);
        }
      } catch (_) {
        /* ignore — keep whatever we found */
      }
    }
    return merged;
  },

  // Rollup for the currently-selected round. Derives from the already-loaded
  // byRound counts (no extra fetch). Empty when no round is selected or no
  // mutations fired in this round.
  roundMutationSummary() {
    const r = this.currentRound();
    if (!r) return { total: 0, breakdown: "" };
    const here = (this.state.mutations?.byRound || []).filter(
      (m) => m.round === r.round_number
    );
    const total = here.reduce((acc, m) => acc + m.n, 0);
    const breakdown = here.map((m) => `${m.n} ${m.kind}`).join(", ");
    return { total, breakdown };
  },
};
