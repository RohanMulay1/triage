import asyncio
import pytest
from app.config import load_router_config
from app.trace import store
from app.trace.branching import collect_fanout, RunBudget, BudgetExceeded
from app.trace.contract import Budget, ActionKind
from app.trace.support import marginal_value_table, SupportThresholds


def collect(**kwargs):
    args = dict(item_id='item', question='What is the capital of France?',
                cfg=load_router_config(), small_id='llama-3.1-8b', big_id='gpt-4o',
                run_id='measurement', dataset='unit', split='train', seed=0,
                labeler=lambda answer, abstained: float('Paris' in answer and not abstained),
                label_source='unit')
    args.update(kwargs)
    return asyncio.run(collect_fanout(**args))


def test_d7_prompt_control_is_nondegenerate():
    prompts = ['What is the capital of France?',
               'Compute the square root of 123456 and explain why step by step.']
    features = [collect(question=q)[0].states[0].prompt_features for q in prompts]
    for key in ('message_chars', 'approx_prompt_tokens', 'predicted_difficulty'):
        assert len({f.get(key, 0.0) for f in features}) > 1


def test_d8_child_has_information_parent_and_local_baseline():
    traces = collect(depth=2)
    parent = next(t for t in traces if t.branch_id.startswith('retrieve') and t.decision_depth == 1)
    child = next(t for t in traces if t.decision_depth == 2
                 and t.baseline_branch_id == parent.branch_id
                 and t.steps[0].decision.chosen.kind == ActionKind.VERIFY)
    assert child.parent_trajectory_id == parent.trajectory_id
    assert child.root_state_id == parent.steps[0].outcome.result_state_id
    assert child.steps[-1].decision.chosen.kind == ActionKind.STOP
    # Deliberately distinct unit-test labels expose a wrong root comparison.
    labels = {'stop': 0.0, parent.branch_id: 1.0, child.branch_id: 0.0}
    traces = [t.model_copy(update={"terminal": t.terminal.model_copy(
        update={"label": labels[t.branch_id]})}) if t.branch_id in labels else t
        for t in traces]
    table = marginal_value_table(traces, SupportThresholds(allow_synthetic=True))
    assert table['per_action'][child.branch_id]['mean_delta'] == -1.0
    assert table['per_action'][child.branch_id]['baseline_branch_id'] == parent.branch_id


def test_depth2_persistence_and_cost_conservation():
    run_id = store.new_run_id('depth2-test')
    store.create_run(store.build_manifest(run_id, dataset='unit'))
    budget = RunBudget(Budget())
    traces = collect(depth=2, run_id=run_id, budget=budget)
    for t in traces:
        store.append(t)
    validation = store.validate_run(run_id)
    assert validation['chain_ok'] and validation['cost_conservation_ok']
    assert not validation['errors']
    assert not validation['analysis_grade']
    assert budget.state.spent_llm_calls == sum(t.total_cost.llm_calls for t in traces)
    assert any(t.decision_depth == 2 for t in store.read_run(run_id))


def test_depth2_budget_preserves_completed_branches():
    first = collect()
    cap = sum(t.total_cost.llm_calls for t in first)
    budget = RunBudget(Budget(max_llm_calls=cap))
    with pytest.raises(BudgetExceeded) as exc:
        collect(depth=2, budget=budget)
    assert budget.state.spent_llm_calls == cap
    assert sum(t.total_cost.llm_calls for t in exc.value.trajectories) == cap


def test_depth2_analysis_uses_parent_state_and_refuses_wrong_parent(monkeypatch):
    from app.trace import analysis
    from app.trace.support import SupportError
    traces = collect(depth=2)
    parent = next(t for t in traces if t.branch_id.startswith('retrieve') and t.decision_depth == 1)
    child = next(t for t in traces if t.decision_depth == 2
                 and t.baseline_branch_id == parent.branch_id
                 and t.steps[0].decision.chosen.kind == ActionKind.VERIFY)
    monkeypatch.setattr(analysis, 'read_run', lambda _: traces)
    thresholds = SupportThresholds(allow_synthetic=True)
    rows = analysis.assemble('measurement', thresholds)
    row = next(r for r in rows if r['baseline_branch_id'] == parent.branch_id)
    assert row['features'] == analysis.state_features(child.states[0], load_router_config())
    assert row['deltas'][child.branch_id] == child.terminal.label - parent.terminal.label
    traces[traces.index(child)] = child.model_copy(update={'parent_trajectory_id': 'wrong'})
    with pytest.raises(SupportError, match='informational parent'):
        analysis.assemble('measurement', thresholds)
