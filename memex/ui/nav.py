"""Shared top-nav for the unified trace UI (build-loop + query-reasoning).

Leaf module — imported by both sub-apps; imports nothing from them, so the
parent app can compose them without a circular import. The nav is injected only
when the sub-app runs mounted under the unified app (a non-empty ``root_path``);
standalone launches of the sub-apps render no bar.
"""

from __future__ import annotations

_LINKS = [("build", "/build/", "Build loop"), ("query", "/query/", "Query reasoning")]


def render_nav(active: str, root_path: str) -> str:
    """Top bar with one link per view, the active one highlighted. Empty string
    when ``root_path`` is empty (standalone — there is no sibling view to reach)."""

    if not root_path:
        return ""
    base = root_path.rsplit("/", 1)[0]  # drop the mounted segment (/build or /query)
    items = []
    for key, suffix, label in _LINKS:
        on = key == active
        color = "background:#57E6CE;color:#06231F;" if on else "color:#8C9AB0;"
        items.append(
            f'<a href="{base}{suffix}" style="padding:5px 12px;border-radius:6px;'
            f'text-decoration:none;font:600 13px system-ui;{color}">{label}</a>'
        )
    return (
        '<nav style="display:flex;gap:8px;align-items:center;background:#0A0E15;'
        'padding:7px 14px;border-bottom:1px solid rgba(120,150,190,.18);flex-shrink:0;">'
        '<span style="color:#8C9AB0;font:700 13px system-ui;margin-right:8px;">'
        "memex · agent traces</span>" + "".join(items) + "</nav>"
    )
