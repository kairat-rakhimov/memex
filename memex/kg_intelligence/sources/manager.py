"""``AdapterManager`` — registry of active source plugins.

Candidates are probed once at process start; those whose ``probe()``
passes register under their ``name``. The pipeline iterates
``manager.active`` and never names a source directly, so adding a source
is a registration change, not an orchestrator change.
"""

from __future__ import annotations

import logging

from memex.kg_intelligence.sources.common.base import SourceAdapter

_log = logging.getLogger(__name__)


class AdapterManager:
    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}
        self._declined: dict[str, str] = {}

    def probe_and_register(self, candidates: list[SourceAdapter]) -> None:
        """Probe each candidate once; register the ready ones. A failed
        probe records a reason for the startup summary rather than raising."""

        for adapter in candidates:
            try:
                ready = adapter.probe()
            except Exception as exc:  # defensive: a probe must never crash startup
                ready = False
                self._declined[adapter.name] = f"{type(exc).__name__}: {exc}"
                _log.info("source declined (probe error): %s — %s", adapter.name, exc)
                continue
            if ready:
                self._adapters[adapter.name] = adapter
                _log.info("source registered: %s", adapter.name)
            else:
                self._declined[adapter.name] = "probe returned False"
                _log.info("source declined (not ready): %s", adapter.name)

    @property
    def active(self) -> list[SourceAdapter]:
        return list(self._adapters.values())

    def get(self, name: str) -> SourceAdapter:
        return self._adapters[name]

    def as_dict(self) -> dict[str, SourceAdapter]:
        """Active adapters keyed by name — for the source-keyed helpers
        (``_plan_search_and_rank`` / ``_fetch_and_persist``)."""

        return dict(self._adapters)

    def manifest(self) -> dict[str, str]:
        """Active source name → description summary; recorded in the build trace."""

        return {a.name: a.description.summary for a in self.active}

    def summary(self) -> str:
        active = ", ".join(sorted(self._adapters)) or "(none)"
        declined = (
            "; ".join(f"{n} ({why})" for n, why in sorted(self._declined.items()))
            or "(none)"
        )
        return f"active: {active} | declined: {declined}"


__all__ = ["AdapterManager"]
