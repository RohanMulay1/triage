import pytest

from app.trace import store
from app.trace.information_value import c2_report
from test_trace_measurement_repairs import collect


@pytest.fixture
def run():
    run_id = store.new_run_id('c2-test')
    store.create_run(store.build_manifest(run_id, dataset='unit'))
    for t in collect(run_id=run_id, depth=2):
        store.append(t)
    return run_id


def test_mock_is_refused_without_explicit_diagnostic(run):
    report = c2_report(run)
    assert report['verdict'] == 'REFUSED'
    assert 'per_action' not in report


def test_zero_paired_interval_moves_to_c4_without_claim(run):
    report = c2_report(run, diagnostic=True)
    assert report['not_evidence']
    assert report['next_candidate'] == 'C4'
    for row in report['per_action'].values():
        assert row['depth1']['point'] == 0
        ci = row['paired_depth2_minus_depth1_ci95']
        assert ci['n'] == 1 and ci['lo'] <= 0 <= ci['hi']
        assert not ci['excludes_zero']
    assert report['c4']['served_projection']['status'] == 'REFUSED'
    assert report['c4']['fitting_status'] == 'NOT_RUN'


def test_incomplete_depth2_does_not_impute_zero(run, monkeypatch):
    from app.trace import information_value as iv
    traces = store.read_run(run)
    monkeypatch.setattr(iv, 'read_run', lambda _: [t for t in traces if t.decision_depth == 1])
    report = c2_report(run, diagnostic=True)
    assert all(r['status'] == 'REFUSED_INCOMPLETE_PAIRS' for r in report['per_action'].values())


def test_matched_continuation_separates_repair_from_information(run, monkeypatch):
    from app.trace import information_value as iv
    traces = store.read_run(run)
    # Controlled test labels: VERIFY repairs regardless of information.
    changed = []
    for t in traces:
        if t.terminal.label is not None:
            label = float(t.steps[0].decision.chosen.kind.value == 'verify')
            t = t.model_copy(update={'terminal': t.terminal.model_copy(update={'label': label})})
        changed.append(t)
    monkeypatch.setattr(iv, 'read_run', lambda _: changed)
    report = c2_report(run, diagnostic=True)
    for row in report['per_action'].values():
        assert row['paired_depth2_minus_depth1_ci95']['point'] == 1
        assert row['matched_information_ci95']['point'] == 0
    assert report['next_candidate'] == 'C4'


def test_outcome_selected_continuation_is_refused(run):
    with pytest.raises(ValueError):
        c2_report(run, continuation='best')
