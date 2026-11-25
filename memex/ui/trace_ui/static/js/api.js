// Shared HTTP helper. Exposes `window.fetchJson` so every other module
// (graph, mutations, app) can call it without imports. Plain script load,
// no ES modules — see index.html for the script-load-order comment.

window.fetchJson = async function fetchJson(url) {
  // window.API_BASE is the mount prefix ("" standalone, "/build" under the
  // unified UI) so absolute /api/... paths resolve to the right sub-app.
  const r = await fetch((window.API_BASE || "") + url);
  if (!r.ok) {
    throw new Error(`${url} → ${r.status}`);
  }
  return r.json();
};
