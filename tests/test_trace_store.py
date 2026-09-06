"""Trace persistence and the support diagnostics.

conftest redirects TRIAGE_TRACE_DIR into a throwaway temp dir, so these never
write into the committed data/ tree.
"""
import json

import pytest

from app.schemas import SignalSet
from app.trace import store
from app.trace.contract import (
    Action,
    ActionCost,
    ActionKind,
    ActionOutcome,
    PolicyDecision,
    SchemaVersionError,
    StateSnapshot,
    TerminalRecord,
    TraceStep,
    Trajectory,
)
from app.trace.support import action_support

ANSWER = Action(kind=ActionKind.ANSWER, model_id="small-1")
VERIFY = Action(kind=ActionKind.VERIFY, model_id="small-1")
STOP = Action(kind=ActionKind.STOP)


def _trajectory(run_id: str, item_id: str, chosen: Action = ANSWER,
                feasible=None, cost: ActionCost = None) -> Trajectory:
    cost = cost or ActionCost(tokens_in=3, tokens_out=4, llm_calls=1)
    feasible = feasible or [ANSWER, VERIFY, STOP]
    root = StateSnapshot.build(item_id, 0, "", SignalSet(), "small-1", [])
    result = StateSnapshot.build(item_id, 1, "answer text", SignalSet(), "small-1", [chosen.key])
    step = TraceStep(
        decision=PolicyDecision(state_id=root.state_id, feasible=feasible,
                                chosen=chosen, policy_id="heuristic_triage"),
        outcome=ActionOutcome(action=chosen, parent_state_id=root.state_id,
                              result_state_id=result.state_id, cost=cost,
                              provider_label="live-provider"),
    )
    return Trajectory(
        trajectory_id=f"traj-{item_id}", run_id=run_id, item_id=item_id,
        root_state_id=root.state_id, states=[root, result], steps=[step],
        terminal=TerminalRecord(served_answer="answer text"), total_cost=cost,
    )


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def test_append_and_read_round_trips_in_order():
    run_id = store.new_run_id("roundtrip")
    written = [_trajectory(run_id, f"item-{i}") for i in range(3)]
    for t in written:
        store.append(t)
    read_back = store.read_run(run_id)
    assert [t.item_id for t in read_back] == ["item-0", "item-1", "item-2"]
    assert read_back == written


def test_reading_a_run_with_no_trajectories_returns_empty():
    assert store.read_run("no-such-run") == []


def test_listed_runs_include_a_written_one():
    run_id = store.new_run_id("listed")
    store.append(_trajectory(run_id, "item-0"))
    assert run_id in store.list_runs()


# --------------------------------------------------------------------------- #
# Manifest immutability
# --------------------------------------------------------------------------- #
def test_manifest_rewrite_with_identical_content_is_allowed():
    run_id = store.new_run_id("manifest-same")
    m = store.build_manifest(run_id, dataset="unit", split="pilot")
    store.write_manifest(m)
    store.write_manifest(m)  # idempotent
    assert store.read_manifest(run_id).run_id == run_id


def test_manifest_rewrite_with_different_content_is_refused():
    """A run's provenance must not drift away from the traces it describes."""
    run_id = store.new_run_id("manifest-diff")
    store.write_manifest(store.build_manifest(run_id, dataset="unit", notes="first"))
    with pytest.raises(store.ManifestConflictError):
        store.write_manifest(store.build_manifest(run_id, dataset="unit", notes="second"))


def test_manifest_records_the_force_mock_flag():
    run_id = store.new_run_id("manifest-mock")
    store.write_manifest(store.build_manifest(run_id))
    # conftest forces the mock provider, so a run collected here is never
    # evidence about real model behaviour and the manifest must say so.
    assert store.read_manifest(run_id).force_mock is True


def test_missing_manifest_raises():
    with pytest.raises(FileNotFoundError):
        store.read_manifest("run-that-does-not-exist")


# --------------------------------------------------------------------------- #
# Run ids are single-use
# --------------------------------------------------------------------------- #
def test_create_run_refuses_a_run_id_that_already_holds_traces():
    """Appending to an existing run would hide two collections behind one manifest."""
    run_id = store.new_run_id("reuse")
    store.create_run(store.build_manifest(run_id, dataset="unit"))
    store.append(_trajectory(run_id, "item-0"))
    with pytest.raises(store.RunExistsError):
        store.create_run(store.build_manifest(run_id, dataset="unit"))


def test_create_run_is_fine_on_a_fresh_id():
    run_id = store.new_run_id("fresh")
    store.create_run(store.build_manifest(run_id, dataset="unit"))
    assert store.read_manifest(run_id).run_id == run_id
    assert store.read_run(run_id) == []


def test_schema_version_is_current_and_older_traces_still_read():
    """Widening the schema must not orphan traces written by an earlier version."""
    from app.trace.contract import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

    assert SCHEMA_VERSION == "1.1.0"
    assert "1.0.0" in SUPPORTED_SCHEMA_VERSIONS

    run_id = store.new_run_id("oldschema")
    path = store.append(_trajectory(run_id, "item-0"))
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    raw["schema_version"] = "1.0.0"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    assert len(store.read_run(run_id)) == 1


# --------------------------------------------------------------------------- #
# Schema gate
# --------------------------------------------------------------------------- #
def test_unknown_schema_version_on_disk_is_refused_not_coerced():
    run_id = store.new_run_id("badversion")
    path = store.append(_trajectory(run_id, "item-0"))
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    raw["schema_version"] = "2.0.0"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        store.read_run(run_id)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_run_reports_cost_conservation_failure():
    run_id = store.new_run_id("badcost")
    t = _trajectory(run_id, "item-0")
    broken = t.model_copy(update={"total_cost": ActionCost(tokens_in=999)})
    store.append(broken)
    store.write_manifest(store.build_manifest(run_id))
    report = store.validate_run(run_id)
    assert report["cost_conservation_ok"] is False
    assert any("tokens_in" in e for e in report["errors"])


def test_validate_run_refuses_analysis_grade_for_a_mock_run():
    run_id = store.new_run_id("mockrun")
    root = StateSnapshot.build("item-0", 0, "", SignalSet(), "small-1", [])
    result = StateSnapshot.build("item-0", 1, "a", SignalSet(), "small-1", [ANSWER.key])
    step = TraceStep(
        decision=PolicyDecision(state_id=root.state_id, feasible=[ANSWER, STOP],
                                chosen=ANSWER, policy_id="p"),
        outcome=ActionOutcome(action=ANSWER, parent_state_id=root.state_id,
                              result_state_id=result.state_id, provider_label="mock"),
    )
    store.append(Trajectory(trajectory_id="t", run_id=run_id, item_id="item-0",
                            root_state_id=root.state_id, states=[root, result], steps=[step]))
    store.write_manifest(store.build_manifest(run_id))
    report = store.validate_run(run_id)
    assert report["synthetic_trajectories"] == 1
    assert report["analysis_grade"] is False


def test_validate_run_reports_a_missing_manifest():
    run_id = store.new_run_id("nomanifest")
    store.append(_trajectory(run_id, "item-0"))
    assert store.validate_run(run_id)["manifest_present"] is False


# --------------------------------------------------------------------------- #
# Support diagnostics
# --------------------------------------------------------------------------- #
def test_support_is_one_for_the_taken_action_and_zero_for_unchosen_ones():
    """The result that motivates randomized branch collection.

    A deterministic policy that always picks ANSWER leaves VERIFY and STOP
    feasible-but-never-taken, so nothing in the data says what they would have
    done. This test pins that as a measurement rather than an assertion.
    """
    run_id = store.new_run_id("support")
    for i in range(4):
        store.append(_trajectory(run_id, f"item-{i}"))
    sup = action_support(store.read_run(run_id))

    assert sup["decisions"] == 4
    assert sup["per_action"][ANSWER.key]["coverage"] == 1.0
    assert sup["per_action"][VERIFY.key]["coverage"] == 0.0
    assert sup["per_action"][STOP.key]["coverage"] == 0.0
    assert set(sup["unsupported_actions"]) == {VERIFY.key, STOP.key}
    assert sup["off_policy_ready"] is False


def test_support_records_the_minimum_propensity_seen():
    run_id = store.new_run_id("propensity")
    store.append(_trajectory(run_id, "item-0"))
    sup = action_support(store.read_run(run_id))
    # The heuristic router is deterministic, so every chosen action has
    # propensity 1.0 -- which is precisely why it cannot support IPS/DR.
    assert sup["per_action"][ANSWER.key]["min_propensity"] == 1.0


def test_support_becomes_ready_once_every_feasible_action_is_taken():
    run_id = store.new_run_id("covered")
    for i, act in enumerate([ANSWER, VERIFY, STOP]):
        store.append(_trajectory(run_id, f"item-{i}", chosen=act))
    sup = action_support(store.read_run(run_id))
    assert sup["unsupported_actions"] == []
    assert sup["off_policy_ready"] is True
