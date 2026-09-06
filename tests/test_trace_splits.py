"""Leakage-safe splits, and the guarantee that tests never touch committed data.

Counterfactual branches of one item share a prefix state and an answer. Splitting
rows at random would put one branch in train and its sibling in test, where the
test label is nearly determined by the training row. Splits are therefore
assigned to a group, never to a row.
"""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from app.config import ROOT
from app.trace.splits import (
    DEFAULT_RATIOS,
    GROUP_KEY_DEFINITION,
    GroupKey,
    SplitManifest,
    assign_split,
    check_no_leakage,
    prompt_fingerprint,
)


def _key(item_id="i-0", **kw):
    base = dict(item_id=item_id, dataset="unit", task_family="gsm8k",
                model_pair="small->big", retriever_version="4",
                prompt_template="unit", prompt_fingerprint="what is the capital")
    base.update(kw)
    return GroupKey(**base)


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def test_assignment_is_deterministic():
    assert assign_split(_key()) == assign_split(_key())


def test_every_axis_of_the_group_key_changes_the_group():
    base = _key().digest()
    for field, value in [("dataset", "other"), ("task_family", "mmlu"),
                         ("model_pair", "a->b"), ("retriever_version", "8"),
                         ("prompt_template", "other"),
                         ("prompt_fingerprint", "something else")]:
        assert _key(**{field: value}).digest() != base, field


def test_a_salt_changes_the_assignment_without_changing_the_group():
    key = _key()
    splits = {assign_split(key, salt=str(s)) for s in range(40)}
    assert len(splits) > 1, "salting must be able to reshuffle assignments"


def test_all_ratios_are_reachable():
    seen = {assign_split(_key(item_id=f"i-{i}", prompt_fingerprint=f"p{i}"))
            for i in range(300)}
    assert seen == set(DEFAULT_RATIOS)


def test_ratios_must_be_positive():
    with pytest.raises(ValueError):
        assign_split(_key(), ratios={"train": 0.0})


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_near_duplicate_prompts_share_a_fingerprint():
    """Exact matching would let a paraphrase of a train prompt sit in test."""
    a = prompt_fingerprint("What is the Capital of France?")
    b = prompt_fingerprint("what is the capital of france")
    assert a == b


def test_fingerprint_separates_genuinely_different_prompts():
    assert prompt_fingerprint("What is the capital of France?") != \
        prompt_fingerprint("Who wrote Hamlet?")


def test_every_item_in_a_group_lands_in_the_same_split():
    manifest = SplitManifest(run_id="r")
    # Same fingerprint and axes, different item ids: one group.
    splits = {manifest.add(_key(item_id=f"i-{i}")) for i in range(20)}
    assert len(splits) == 1
    assert len(manifest.groups) == 1


def test_check_no_leakage_flags_a_group_spanning_two_splits():
    key = _key()
    assert check_no_leakage([(key, "train"), (key, "test")])
    assert check_no_leakage([(key, "train"), (key, "train")]) == []


def test_counts_track_items_not_groups():
    manifest = SplitManifest(run_id="r")
    for i in range(5):
        manifest.add(_key(item_id=f"i-{i}"))
    assert sum(manifest.counts().values()) == 5
    assert len(manifest.groups) == 1


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def test_manifest_round_trips(tmp_path):
    manifest = SplitManifest(run_id="r", salt="s")
    for i in range(6):
        manifest.add(_key(item_id=f"i-{i}", prompt_fingerprint=f"p{i}"))
    manifest.write(tmp_path)
    reloaded = SplitManifest.read(tmp_path)
    assert reloaded.run_id == "r"
    assert reloaded.salt == "s"
    assert reloaded.counts() == manifest.counts()
    assert reloaded.definition == GROUP_KEY_DEFINITION


def test_manifest_refuses_to_overwrite_a_different_key_definition(tmp_path):
    """Splits computed under different recipes are not comparable."""
    first = SplitManifest(run_id="r")
    first.add(_key())
    first.write(tmp_path)

    second = SplitManifest(run_id="r", definition="group_key_v2:something_else")
    second.add(_key())
    with pytest.raises(ValueError, match="not comparable"):
        second.write(tmp_path)


def test_manifest_records_the_definition_it_used(tmp_path):
    manifest = SplitManifest(run_id="r")
    manifest.add(_key())
    path = manifest.write(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["definition"] == GROUP_KEY_DEFINITION
    assert raw["group_count"] == 1


# --------------------------------------------------------------------------- #
# The committed receipts are never touched by a test run
# --------------------------------------------------------------------------- #
def _receipt_hashes() -> dict[str, str]:
    data = Path(ROOT) / "data"
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data.glob("*.json"))
    }


def test_state_written_during_tests_lands_outside_the_repository():
    """conftest redirects every writable path; this asserts it actually worked."""
    from app.config import get_settings

    settings = get_settings()
    repo_data = (Path(ROOT) / "data").resolve()
    for path in (Path(settings.telemetry_db), Path(settings.trace_dir)):
        assert repo_data not in path.resolve().parents, f"{path} is inside {repo_data}"


def test_routing_does_not_mutate_a_committed_benchmark_receipt():
    """The published numbers in data/*.json must survive any run."""
    from app.router.router import route_and_answer
    from app.schemas import ChatRequest

    before = _receipt_hashes()
    assert before, "expected committed receipts in data/"
    asyncio.run(route_and_answer(ChatRequest(message="What is the capital of Japan?")))
    assert _receipt_hashes() == before
