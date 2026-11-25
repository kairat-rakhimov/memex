// Pure display helpers. Spread into the Alpine.data factory in app.js so
// they're available as `this.statusClass()` etc. inside templates.

window.traceUiFormatters = {
  statusClass(status) {
    switch (status) {
      case "success":
        return "bg-emerald-100 text-emerald-800";
      case "sample_mode":
        return "bg-amber-100 text-amber-800";
      case "failed":
        return "bg-rose-100 text-rose-800";
      case "skipped":
        return "bg-slate-100 text-slate-600";
      case "started":
        return "bg-indigo-100 text-indigo-700";
      default:
        return "bg-slate-100 text-slate-700";
    }
  },

  formatDur(seconds) {
    if (seconds == null) return "—";
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m${String(s).padStart(2, "0")}s`;
  },

  formatMs(ms) {
    if (ms == null) return "—";
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  },

  prettyJson(value) {
    if (value == null) return "";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch (_) {
      return String(value);
    }
  },
};
