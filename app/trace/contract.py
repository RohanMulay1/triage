"""The Triage trace/data contract (schema v1.0.0).

Checkpoint 0 found that request-level telemetry (``app/telemetry/db.py``) records
only what was *served*. It can never say what an unchosen action would have
produced, so it cannot identify action-conditioned repair value. This module
defines the data the research plan actually needs:

    Action          a typed, costed intervention the policy may take
    StateSnapshot   the observed response state before/after an action
    ActionOutcome   what one action did to that state, and what it cost
    Budget          the remaining resource envelope (immutable updates)
    PolicyDecision  which action was chosen, from which feasible set, and why
    Trajectory      one ordered branch of (decision, outcome) steps for one item

Two properties are load-bearing and easy to lose:

  * ``PolicyDecision.propensity`` -- the behaviour policy's probability of the
    chosen action. Without it, IPS/DR off-policy estimation is impossible and
    the "action support and missingness" diagnostic cannot be computed.
  * ``ActionOutcome.provider_label`` -- ``LLMClient.generate`` silently falls
    back to the mock provider on any live-provider error. An outcome that does
    not record its provider can smuggle synthetic text into a dataset of real
    model behaviour.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import SignalSet

SCHEMA_VERSION = "1.2.0"
# 1.1.0 widened CostSource. 1.2.0 added OutcomeStatus, cumulative cost, action
# execution detail, and behaviour-policy provenance on PolicyDecision -- all
# additive with defaults, so 1.0.0 and 1.1.0 traces read back unchanged. The
# reader accepts known versions explicitly rather than coercing an unknown one.
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0", "1.1.0", "1.2.0"})

# Where a cost number came from. The distinction that matters for a results
# table is between a figure someone was actually invoiced for and a figure
# computed from a price list:
#
#   billed_api    taken from a provider billing record
#   listed_price  measured token counts x the price in config/models.yaml at
#                 collection time (snapshotted in the run manifest). Reproducible
#                 and honest, but it is an estimate, not an invoice.
#   local_gpu     measured GPU-seconds on a controlled local serving lane
#   token_proxy   the hardware-agnostic compute-unit proxy; never dollars
#   free          no cost incurred (a local vector search, the mock provider)
#   mixed         a sum spanning more than one of the above
CostSource = Literal[
    "billed_api", "listed_price", "local_gpu", "token_proxy", "free", "mixed"
]
# Sources whose dollar figure means something. A single action costed by the
# token proxy, or served for free, may never carry a price.
DOLLAR_SOURCES = frozenset({"billed_api", "listed_price", "local_gpu", "mixed"})
Split = Literal["train", "calib", "test", "pilot"]


class SchemaVersionError(ValueError):
    """A trace was written by a schema version this build does not understand."""


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def text_hash(text: str) -> str:
    return _sha(text or "")


# --------------------------------------------------------------------------- #
# Action
# --------------------------------------------------------------------------- #
class ActionKind(str, Enum):
    """The intervention vocabulary.

    The research plan names RESAMPLE / RETRIEVE / VERIFY / TOOL / STRONGER_MODEL
    / SPECIALIST_MODEL / ABSTAIN. Three more are required for the accounting to
    be correct:

      ANSWER      the root generation -- there is no state to score without it.
      STOP        the null action. Its value is the baseline every marginal
                  value estimate is measured against, so it must be a
                  first-class member of the feasible set, not an implicit
                  default.
      SELF_CHECK  the contradiction probe (``app/signals/proxy.py``). It is a
                  separately billed LLM call; folding its cost into RESAMPLE or
                  VERIFY would silently bias those two actions' value estimates.
    """

    ANSWER = "answer"
    RESAMPLE = "resample"
    SELF_CHECK = "self_check"
    RETRIEVE = "retrieve"
    VERIFY = "verify"
    TOOL = "tool"
    STRONGER_MODEL = "stronger_model"
    SPECIALIST_MODEL = "specialist_model"
    ABSTAIN = "abstain"
    STOP = "stop"


class OutcomeStatus(str, Enum):
    """What actually happened to an action. These are NOT interchangeable.

    Collapsing them is how a dataset silently lies: an action that was never
    tried, one that was tried and failed, and one that has no implementation at
    all would otherwise all look like "no benefit observed", which would bias
    every value estimate toward zero for exactly the actions that are hardest to
    execute.

      CHOSEN              executed on the served path
      COUNTERFACTUAL      executed as an unchosen branch from the same state
      UNCHOSEN            feasible and considered, not executed (no outcome)
      ATTEMPTED           executed but produced nothing usable (empty answer)
      UNAVAILABLE         no real integration exists for this action here
      FAILED              executed and raised or returned a provider error
      SYNTHETIC_FALLBACK  a live call failed and the mock provider answered
    """

    CHOSEN = "chosen"
    COUNTERFACTUAL = "counterfactual"
    UNCHOSEN = "unchosen"
    ATTEMPTED = "attempted"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    SYNTHETIC_FALLBACK = "synthetic_fallback"


#: Statuses whose outcome reflects a real execution, so a value estimate may use
#: it. UNCHOSEN and UNAVAILABLE carry no observation; ATTEMPTED/FAILED/
#: SYNTHETIC_FALLBACK carry an observation that is not about model capability.
OBSERVED_STATUSES = frozenset({OutcomeStatus.CHOSEN, OutcomeStatus.COUNTERFACTUAL})


class Action(BaseModel):
    """One intervention. Frozen: an action recorded in a trace is history."""

    model_config = ConfigDict(frozen=True)

    kind: ActionKind
    model_id: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Canonical grouping key, stable across dict ordering.

        Per-action value heads are fit by grouping outcomes across traces, so
        two records of the same intervention must produce the same string
        whatever order their params happened to be built in.
        """
        bits = [self.kind.value]
        if self.model_id:
            bits.append(self.model_id)
        if self.params:
            bits.append(",".join(f"{k}={canonical(self.params[k])}" for k in sorted(self.params)))
        return ":".join(bits)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.key


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
class ActionCost(BaseModel):
    """Measured cost of one action.

    ``source`` is mandatory. The strategy is explicit that
    ``compute_units = tokens_in + 10*tokens_out`` is a transparent proxy, not
    FLOPs, energy or GPU-seconds. Recording where each number came from is what
    stops a results table from quietly averaging a measured bill together with
    an estimate.
    """

    model_config = ConfigDict(frozen=True)

    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    est_cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    compute_units: float = 0.0
    source: CostSource = "free"

    @model_validator(mode="after")
    def _dollars_need_a_source_that_can_have_them(self) -> "ActionCost":
        if self.source not in DOLLAR_SOURCES and self.est_cost_usd > 0:
            raise ValueError(
                f"est_cost_usd={self.est_cost_usd} with source={self.source!r}: "
                f"dollar figures require one of {sorted(DOLLAR_SOURCES)}"
            )
        return self

    @classmethod
    def zero(cls, source: CostSource = "free") -> "ActionCost":
        return cls(source=source)

    def __add__(self, other: "ActionCost") -> "ActionCost":
        return ActionCost(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            llm_calls=self.llm_calls + other.llm_calls,
            est_cost_usd=self.est_cost_usd + other.est_cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
            ttft_ms=max(self.ttft_ms, other.ttft_ms),
            compute_units=self.compute_units + other.compute_units,
            source=merge_sources(self.source, other.source),
        )


def merge_sources(a: CostSource, b: CostSource) -> CostSource:
    """Provenance of a sum.

    Adding a free component changes nothing. Adding components of genuinely
    different provenance produces "mixed", which still permits a dollar figure
    (part of it was really priced) while making it impossible for a results
    table to present the total as a clean invoice.

    Returning the *weaker* source instead would be worse than useless here: it
    used to make ``measured + proxy`` claim source="token_proxy" while carrying
    dollars, which the validator then rejected -- so summing two legitimately
    costed actions raised instead of producing a total.
    """
    if a == b:
        return a
    if a == "free":
        return b
    if b == "free":
        return a
    return "mixed"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class StateSnapshot(BaseModel):
    """The response state a decision is conditioned on. Frozen, content-addressed."""

    model_config = ConfigDict(frozen=True)

    state_id: str
    item_id: str
    step: int
    answer: str = ""
    answer_hash: str = ""
    signals: SignalSet = Field(default_factory=SignalSet)
    evidence_count: int = 0
    model_id: str = ""
    actions_taken: list[str] = Field(default_factory=list)
    spent: ActionCost = Field(default_factory=ActionCost)
    prompt_features: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        item_id: str,
        step: int,
        answer: str,
        signals: SignalSet,
        model_id: str,
        actions_taken: list[str],
        spent: Optional[ActionCost] = None,
        evidence_count: int = 0,
        prompt_features: Optional[dict[str, Any]] = None,
    ) -> "StateSnapshot":
        """Content-addressed construction: identical content yields an identical id.

        Deterministic ids are what let two sibling branches share a parent, and
        what makes a trace verifiable after the fact.
        """
        ah = text_hash(answer)
        sig = canonical(signals.model_dump(exclude={"detail"}))
        sid = _sha(item_id, str(step), ah, sig, model_id, canonical(actions_taken))
        return cls(
            state_id=sid, item_id=item_id, step=step, answer=answer, answer_hash=ah,
            # Deep copy: the router mutates one SignalSet in place across a
            # request, so storing the reference would let later mutations
            # retroactively rewrite snapshots already recorded.
            signals=signals.model_copy(deep=True),
            evidence_count=evidence_count, model_id=model_id,
            actions_taken=list(actions_taken), spent=spent or ActionCost(),
            prompt_features=prompt_features or {},
        )


# --------------------------------------------------------------------------- #
# Outcome
# --------------------------------------------------------------------------- #
class ActionOutcome(BaseModel):
    """What one action did. Recorded whether or not it was on the served path.

    ``cost`` is the **incremental** cost of this action alone. ``cumulative_cost``
    is the trajectory's cost through this step. Both are stored because a
    marginal-value estimate needs the increment while a budget check needs the
    running total, and deriving one from the other after the fact is where
    accounting mistakes hide. For STRONGER_MODEL specifically, ``cost`` is the
    big-model pass only -- the small-model trajectory was already paid for and is
    not re-charged.
    """

    model_config = ConfigDict(frozen=True)

    action: Action
    parent_state_id: str
    result_state_id: str
    status: OutcomeStatus = OutcomeStatus.CHOSEN
    cost: ActionCost = Field(default_factory=ActionCost)
    cumulative_cost: ActionCost = Field(default_factory=ActionCost)
    answer_changed: bool = False
    answer_similarity: Optional[float] = None
    signal_delta: dict[str, float] = Field(default_factory=dict)
    provider_label: str = "unknown"
    error: Optional[str] = None
    is_counterfactual: bool = False
    seed: Optional[int] = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_and_flag_agree(self) -> "ActionOutcome":
        if (self.status == OutcomeStatus.COUNTERFACTUAL) != self.is_counterfactual:
            raise ValueError(
                f"status={self.status.value!r} contradicts "
                f"is_counterfactual={self.is_counterfactual}"
            )
        if self.status == OutcomeStatus.SYNTHETIC_FALLBACK and self.provider_label != "mock":
            raise ValueError(
                "SYNTHETIC_FALLBACK requires provider_label='mock'; "
                f"got {self.provider_label!r}"
            )
        return self

    @property
    def is_synthetic(self) -> bool:
        """True when this outcome came from the offline mock provider.

        ``LLMClient.generate`` falls back to mock on any live-provider error, so
        this can be true inside an otherwise-live run.
        """
        return self.provider_label == "mock"

    @property
    def is_observed(self) -> bool:
        """True when this outcome is a real execution a value estimate may use."""
        return self.status in OBSERVED_STATUSES


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
class Budget(BaseModel):
    """Remaining resource envelope. Updates return a new Budget; never mutate."""

    model_config = ConfigDict(frozen=True)

    max_usd: Optional[float] = None
    max_llm_calls: Optional[int] = None
    max_wall_ms: Optional[float] = None
    max_compute_units: Optional[float] = None
    max_escalations: Optional[int] = None

    spent_usd: float = 0.0
    spent_llm_calls: int = 0
    spent_wall_ms: float = 0.0
    spent_compute_units: float = 0.0
    escalations_used: int = 0

    def can_afford(self, cost: ActionCost, *, escalation: bool = False) -> bool:
        """False if ANY dimension would be exceeded -- the constraint is a conjunction."""
        checks = [
            (self.max_usd, self.spent_usd + cost.est_cost_usd),
            (self.max_llm_calls, self.spent_llm_calls + cost.llm_calls),
            (self.max_wall_ms, self.spent_wall_ms + cost.latency_ms),
            (self.max_compute_units, self.spent_compute_units + cost.compute_units),
            (self.max_escalations, self.escalations_used + (1 if escalation else 0)),
        ]
        return all(cap is None or after <= cap for cap, after in checks)

    def charge(self, cost: ActionCost, *, escalation: bool = False) -> "Budget":
        return self.model_copy(update={
            "spent_usd": self.spent_usd + cost.est_cost_usd,
            "spent_llm_calls": self.spent_llm_calls + cost.llm_calls,
            "spent_wall_ms": self.spent_wall_ms + cost.latency_ms,
            "spent_compute_units": self.spent_compute_units + cost.compute_units,
            "escalations_used": self.escalations_used + (1 if escalation else 0),
        })

    def remaining(self) -> dict[str, Optional[float]]:
        def left(cap, spent):
            return None if cap is None else cap - spent

        return {
            "usd": left(self.max_usd, self.spent_usd),
            "llm_calls": left(self.max_llm_calls, self.spent_llm_calls),
            "wall_ms": left(self.max_wall_ms, self.spent_wall_ms),
            "compute_units": left(self.max_compute_units, self.spent_compute_units),
            "escalations": left(self.max_escalations, self.escalations_used),
        }


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
class ActionScore(BaseModel):
    """A policy's estimate of one action's marginal value."""

    model_config = ConfigDict(frozen=True)

    delta_hat: float = 0.0
    sigma_hat: float = 0.0
    lcb: float = 0.0

    @field_validator("sigma_hat")
    @classmethod
    def _sigma_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("sigma_hat must be non-negative")
        return v


class PolicyDecision(BaseModel):
    """Which action was chosen, out of which feasible set, under which policy.

    ``propensity`` is the behaviour policy's probability of ``chosen`` given the
    state. The deterministic heuristic router records 1.0 -- which is honest,
    and is exactly the evidence the support diagnostic uses to show that
    heuristic-only traces carry no information about unchosen actions.
    """

    model_config = ConfigDict(frozen=True)

    state_id: str
    feasible: list[Action]
    chosen: Action
    policy_id: str
    policy_version: str = "0"
    #: How `feasible` was derived, so a support denominator computed under one
    #: action-space definition is never silently compared with another.
    feasible_set_definition: str = "unspecified"
    exploration_seed: Optional[int] = None
    propensity: float = 1.0
    scores: dict[str, ActionScore] = Field(default_factory=dict)
    stop_reason: Optional[str] = None
    rationale: dict[str, Any] = Field(default_factory=dict)

    @field_validator("propensity")
    @classmethod
    def _propensity_in_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"propensity must lie in (0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _chosen_is_feasible(self) -> "PolicyDecision":
        keys = {a.key for a in self.feasible}
        if self.chosen.key not in keys:
            raise ValueError(
                f"chosen action {self.chosen.key!r} is not in the feasible set {sorted(keys)}"
            )
        return self


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
class TraceStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: PolicyDecision
    outcome: ActionOutcome

    @model_validator(mode="after")
    def _decision_matches_outcome(self) -> "TraceStep":
        if self.decision.state_id != self.outcome.parent_state_id:
            raise ValueError(
                f"decision state {self.decision.state_id!r} != outcome parent "
                f"{self.outcome.parent_state_id!r}"
            )
        if self.decision.chosen.key != self.outcome.action.key:
            raise ValueError("outcome records a different action than the decision chose")
        return self


class TerminalRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    served_answer: str = ""
    served_answer_hash: str = ""
    status: str = "OK"
    abstained: bool = False
    tier: int = 0
    models_used: list[str] = Field(default_factory=list)
    label: Optional[float] = None
    label_source: Optional[str] = None


class Trajectory(BaseModel):
    """One branch: an ordered chain of (decision, outcome) steps for one item."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    trajectory_id: str
    run_id: str
    item_id: str
    dataset: str = "unknown"
    split: Split = "pilot"
    branch_id: str = "served"
    parent_trajectory_id: Optional[str] = None
    root_state_id: str
    # The states themselves, root first. Steps reference them by id rather than
    # embedding them, so a state shared by sibling branches is stored once. A
    # trajectory that kept only ids would be useless for fitting value models:
    # the state features are the input those models condition on.
    states: list[StateSnapshot] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    terminal: TerminalRecord = Field(default_factory=TerminalRecord)
    total_cost: ActionCost = Field(default_factory=ActionCost)
    budget: Budget = Field(default_factory=Budget)
    seed: int = 0
    created_ts: float = 0.0

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(
                f"trace schema {v!r} is not supported by this build "
                f"(known: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
            )
        return v

    @model_validator(mode="after")
    def _chain_is_intact(self) -> "Trajectory":
        """Every step must hang off the root or off an earlier step's result."""
        reachable = {self.root_state_id}
        for i, step in enumerate(self.steps):
            parent = step.outcome.parent_state_id
            if parent not in reachable:
                raise ValueError(
                    f"step {i} ({step.outcome.action.key}) hangs off unknown state "
                    f"{parent!r}; reachable states are {sorted(reachable)}"
                )
            reachable.add(step.outcome.result_state_id)
        if self.states:
            stored = {s.state_id for s in self.states}
            missing = reachable - stored
            if missing:
                raise ValueError(
                    f"trajectory references {len(missing)} state(s) it does not store: "
                    f"{sorted(missing)}"
                )
        return self

    @property
    def has_synthetic_outcomes(self) -> bool:
        return any(s.outcome.is_synthetic for s in self.steps)

    def state(self, state_id: str) -> Optional[StateSnapshot]:
        for s in self.states:
            if s.state_id == state_id:
                return s
        return None

    def summed_cost(self) -> ActionCost:
        total = ActionCost.zero()
        for s in self.steps:
            total = total + s.outcome.cost
        return total

    def redacted(self) -> "Trajectory":
        """Drop every answer string, keep every hash.

        For corpora whose licence forbids redistributing model outputs. Ids are
        content hashes of the original text, so the trace stays verifiable and
        joinable without carrying the text itself.
        """
        states = [s.model_copy(update={"answer": ""}) for s in self.states]
        terminal = self.terminal.model_copy(update={"served_answer": ""})
        return self.model_copy(update={"states": states, "terminal": terminal})


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class TraceManifest(BaseModel):
    """The immutable provenance record the checkpoint protocol requires."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    schema_version: str = SCHEMA_VERSION
    created_ts: float = 0.0
    git_sha: str = "unknown"
    router_config_hash: str = ""
    models_config_hash: str = ""
    model_snapshot: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    force_mock: bool = False
    dataset: str = "unknown"
    #: The run's split POLICY, not a split: "mixed" when items are assigned by
    #: group hash, or a split name when the run forced one. The authoritative
    #: per-item split is Trajectory.split, which stays strictly typed.
    split: str = "pilot"
    command: str = ""
    env: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
