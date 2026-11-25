"""Query-layer agents: the tool-using Retriever + Explorer (both via
:func:`runtime.run_tool_agent`) and the single-shot agents (Answerer,
GroundingChecker, ContestationJudge, VerifyEvidence, NoveltyJudge, ReportWriter,
FinalReviewer), which reuse the build loop's ``execute_agent_call`` unchanged.
The one-shot ``hypothesis_generator`` is retained only as the eval baseline.
"""
