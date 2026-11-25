"""Output schemas for the query-layer agents (chat + hypothesis).

Every schema carries an explicit ``reasoning`` field: the inner-thinking viewer
renders it as first-class captured data, not something inferred from I/O. Fact
references are ``"kind:id"`` strings (``relation:1037`` / ``attribute:42``) — the
only ids the toolbox handed the agent; grounding rejects anything else.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ── Q&A: Retriever ────────────────────────────────────────────────────────────


class RetrieverResult(BaseModel):
    """The agentic Retriever's evidence bundle: the fact refs it judged relevant
    after tool exploration, plus its strategy reasoning + observed gaps."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    selected: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Q&A: Answerer ─────────────────────────────────────────────────────────────


class Claim(BaseModel):
    """One assertion in the answer. ``support`` are 1-based indices into the
    grounded evidence bundle the Answerer was shown."""

    model_config = ConfigDict(extra="forbid")

    text: str
    support: list[int] = Field(default_factory=list)


class AnswerResult(BaseModel):
    """Grounded answer: prose (with inline ``[n]`` citations) + the structured
    claim → evidence map + honest ``gaps`` ("not in corpus")."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# ── Q&A: GroundingChecker ─────────────────────────────────────────────────────


class ClaimVerdict(BaseModel):
    """``claim_index`` is 1-based into ``AnswerResult.claims``."""

    model_config = ConfigDict(extra="forbid")

    claim_index: int = Field(ge=1)
    supported: bool
    reasoning: str = ""


class GroundingVerdict(BaseModel):
    """Adversarial per-claim grounding check. ``revision_request`` is the
    guidance handed back to the Answerer when any claim is unsupported."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    revision_request: str = ""

    def all_supported(self) -> bool:
        return all(v.supported for v in self.verdicts) if self.verdicts else True

    def unsupported_indices(self) -> list[int]:
        return [v.claim_index for v in self.verdicts if not v.supported]


# ── Hypothesis: Generator ─────────────────────────────────────────────────────


class HypothesisCandidate(BaseModel):
    """One articulated hypothesis from a deterministic seed. ``seed_index`` ties
    it back to the seed (and its grounded legs). ``worthwhile=False`` rejects a
    dull seed. ``leg_refs`` are the seed fact refs ("kind:id") the mechanism
    leans on."""

    model_config = ConfigDict(extra="forbid")

    seed_index: int = Field(ge=0)
    worthwhile: bool = True
    statement: str = ""
    mechanism: str = ""
    leg_refs: list[str] = Field(default_factory=list)
    hypothesis_class: str = ""
    plausibility: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


class GeneratorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    candidates: list[HypothesisCandidate] = Field(default_factory=list)


# ── Hypothesis: Explorer (tool-using, one seed investigated deeply) ────────────


class ExploredHypothesis(BaseModel):
    """The Explorer's verdict after investigating ONE seed with graph tools.
    ``worthwhile=False`` means the investigation found nothing genuine (a dud —
    bailed early). ``leg_refs`` are the fact refs ("kind:id") it ACTUALLY saw via
    tools and leans on — grounding rejects anything else."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    worthwhile: bool = True
    statement: str = ""
    mechanism: str = ""
    leg_refs: list[str] = Field(default_factory=list)
    plausibility: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Hypothesis: VerifyEvidence ────────────────────────────────────────────────


class LegVerdict(BaseModel):
    """``ref`` echoes a candidate leg ref ("kind:id")."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    supported: bool
    note: str = ""


class VerifyVerdict(BaseModel):
    """Per-candidate verification: re-read each leg's source prose, drop
    hallucinated/unsupported legs, and judge whether the hypothesis survives
    (a broken load-bearing leg downgrades it)."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    leg_verdicts: list[LegVerdict] = Field(default_factory=list)
    holds: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Hypothesis: NoveltyJudge ──────────────────────────────────────────────────


class NoveltyJudgment(BaseModel):
    """Assertional-novelty verdict for ONE candidate. ``index`` is its 0-based
    position in the batch. ``novelty`` ∈ [0,1]: ~0 the conclusion is already
    asserted or trivially entailed by the nearest facts; ~1 a genuine inferential
    leap stated nowhere among them."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    novelty: float = Field(ge=0.0, le=1.0)
    why: str = ""


class NoveltyBatch(BaseModel):
    """One batched pass scoring meaning-novelty for all surviving candidates —
    what embedding cosine ("about the same things") cannot tell from "this claim
    is already asserted/entailed"."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    judgments: list[NoveltyJudgment] = Field(default_factory=list)


# ── Hypothesis: ContestationJudge ─────────────────────────────────────────────


class ContestationVerdict(BaseModel):
    """Whether a relation_contested seed's predicate set is a GENUINE conflict.
    ``index`` is the 0-based position in the batch. ``genuine=True`` ⇒ the
    predicates actually oppose/tension each other (e.g. ``increases`` vs
    ``replaces``); ``False`` ⇒ synonyms or non-conflicting facets of one
    relationship (predicate sprawl masquerading as contestation)."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    genuine: bool
    why: str = ""


class ContestationBatch(BaseModel):
    """One batched pass classifying every relation_contested seed — the LLM
    judgment the lane needs because embeddings place antonyms adjacent and so
    can't separate genuine opposition from synonymy."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    verdicts: list[ContestationVerdict] = Field(default_factory=list)


# ── Hypothesis: ReportWriter + FinalReviewer ──────────────────────────────────


class HypothesisWrite(BaseModel):
    """The ReportWriter's prose for ONE ranked hypothesis, in rank order. Scores,
    citations, and the grounded-evidence list are added deterministically by the
    orchestration — do NOT emit them here."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    hypothesis: str = ""
    rationale: str = ""
    how_to_test: str = ""


class ReportResult(BaseModel):
    """Structured report: an intro + one prose block per ranked hypothesis (in
    order) + a conclusion. The orchestration renders the headings (with scores)
    and the grounded-evidence blocks deterministically around these."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    intro: str = ""
    hypotheses: list[HypothesisWrite] = Field(default_factory=list)
    conclusion: str = ""


class FinalReview(BaseModel):
    """Coverage / over-claim / citation audit of the report. ``approved=False``
    sends one revision back to the ReportWriter."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    issues: list[str] = Field(default_factory=list)
    approved: bool = True
