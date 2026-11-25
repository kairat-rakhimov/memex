"""Inner-thinking viewer for query runs (chat + hypothesize).

A separate, self-contained read-only viewer over ``kg/<kg_id>/query_trace.db``
(distinct from the build trace UI). Reuses the agent-call tree
(``parent_call_id``) + prompt/response/parsed panes, drops the build/round/doc
chrome, and adds the headline **inner-thinking trail**: the Retriever's ReAct
trajectory + every agent's ``reasoning`` field.
"""

from memex.ui.query_ui.app import create_app

__all__ = ["create_app"]
