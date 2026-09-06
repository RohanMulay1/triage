"""Action-level research traces: the counterfactual data request telemetry cannot hold.

See ``contract.py`` for the schema, ``adapter.py`` for the legacy-router bridge,
``store.py`` for persistence and ``support.py`` for the support diagnostics.
"""
from .contract import (
    SCHEMA_VERSION,
    Action,
    ActionCost,
    ActionKind,
    ActionOutcome,
    ActionScore,
    Budget,
    OBSERVED_STATUSES,
    OutcomeStatus,
    PolicyDecision,
    SchemaVersionError,
    StateSnapshot,
    TerminalRecord,
    TraceManifest,
    TraceStep,
    Trajectory,
)

__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "ActionCost",
    "ActionKind",
    "ActionOutcome",
    "ActionScore",
    "Budget",
    "OBSERVED_STATUSES",
    "OutcomeStatus",
    "PolicyDecision",
    "SchemaVersionError",
    "StateSnapshot",
    "TerminalRecord",
    "TraceManifest",
    "TraceStep",
    "Trajectory",
]
