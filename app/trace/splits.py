"""Leakage-safe dataset splits for trace runs.

Splitting trace rows at random leaks in ways that are easy to miss and fatal to
the result. Counterfactual branches of one item share a prefix state and an
answer; putting one branch in train and its sibling in test means the test label
is nearly determined by the training row. The same applies across near-duplicate
prompts, across a task family the model has effectively memorised, and across
runs that differ only in model pair or retriever version.

So splits are assigned to a **group**, never to a row, and every row in a group
lands in the same split. The group key spans every axis the plan names as a shift
dimension: item, task family, model pair, retriever version and prompt template.
The manifest records the key definition, so a split computed under one definition
is never silently compared with another.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .contract import Split, canonical

#: Bump when the group-key recipe changes; a manifest carrying a different
#: definition must not be reused.
GROUP_KEY_DEFINITION = "group_key_v2:family+modelpair+retriever+template+fingerprint"

DEFAULT_RATIOS: dict[str, float] = {"train": 0.5, "calib": 0.25, "test": 0.25}

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]")


def prompt_fingerprint(prompt: str, *, head_words: int = 12) -> str:
    """A coarse fingerprint that collapses near-duplicate prompts into one group.

    Exact-match grouping would let a paraphrase of a training prompt sit in test.
    This normalises case and punctuation and keys on the leading content words,
    which is crude but errs toward over-grouping -- the safe direction.
    """
    text = _NONWORD.sub(" ", (prompt or "").lower())
    words = _WS.sub(" ", text).strip().split()
    return " ".join(words[:head_words])


@dataclass(frozen=True)
class GroupKey:
    """Every axis a split must not leak across."""

    item_id: str
    dataset: str
    task_family: str = "default"
    model_pair: str = "unknown"
    retriever_version: str = "unknown"
    prompt_template: str = "default"
    prompt_fingerprint: str = ""

    #: Fields that define the GROUP. `item_id` is deliberately excluded: it
    #: identifies a member, not a group, and including it made every item its own
    #: group -- which silently defeated the leakage prevention this class exists
    #: for, letting two near-duplicate prompts land in train and test.
    GROUPING_FIELDS = ("dataset", "task_family", "model_pair",
                       "retriever_version", "prompt_template", "prompt_fingerprint")

    def grouping(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in self.GROUPING_FIELDS}

    def digest(self) -> str:
        return hashlib.sha256(canonical(self.grouping()).encode("utf-8")).hexdigest()


def assign_split(
    key: GroupKey,
    ratios: Optional[dict[str, float]] = None,
    salt: str = "",
) -> Split:
    """Deterministic hash assignment. Same group and salt always give the same split."""
    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")
    h = hashlib.sha256((key.digest() + "|" + salt).encode("utf-8")).hexdigest()
    # 52 bits is ample and keeps the value exact in a float.
    position = (int(h[:13], 16) / float(1 << 52)) * total
    upto = 0.0
    for name in ("train", "calib", "test", "pilot"):
        if name not in ratios:
            continue
        upto += ratios[name]
        if position < upto:
            return name  # type: ignore[return-value]
    return "test"  # pragma: no cover - only reachable on float edge


@dataclass
class SplitManifest:
    """The record that makes a split reproducible and auditable."""

    run_id: str
    definition: str = GROUP_KEY_DEFINITION
    ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    salt: str = ""
    created_ts: float = field(default_factory=time.time)
    #: group digest -> {"split": ..., "key": {...}, "items": [...]}
    groups: dict[str, Any] = field(default_factory=dict)

    def add(self, key: GroupKey) -> Split:
        digest = key.digest()
        entry = self.groups.get(digest)
        if entry is None:
            split = assign_split(key, self.ratios, self.salt)
            self.groups[digest] = {"split": split, "key": key.grouping(),
                                   "items": [key.item_id]}
            return split
        if key.item_id not in entry["items"]:
            entry["items"].append(key.item_id)
        return entry["split"]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in self.groups.values():
            out[entry["split"]] = out.get(entry["split"], 0) + len(entry["items"])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "definition": self.definition,
            "ratios": self.ratios, "salt": self.salt, "created_ts": self.created_ts,
            "group_count": len(self.groups), "counts": self.counts(),
            "groups": self.groups,
        }

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "splits.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("definition") != self.definition:
                raise ValueError(
                    f"existing split manifest at {path} uses definition "
                    f"{existing.get('definition')!r}, not {self.definition!r}; "
                    "splits under different definitions are not comparable"
                )
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, directory: Path) -> "SplitManifest":
        raw = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
        m = cls(run_id=raw["run_id"], definition=raw["definition"],
                ratios=raw["ratios"], salt=raw.get("salt", ""),
                created_ts=raw.get("created_ts", 0.0))
        m.groups = raw.get("groups", {})
        return m


def check_no_leakage(assignments: Iterable[tuple[GroupKey, str]]) -> list[str]:
    """Return a violation per group that landed in more than one split."""
    seen: dict[str, set[str]] = {}
    for key, split in assignments:
        seen.setdefault(key.digest(), set()).add(split)
    return [f"group {digest} spans splits {sorted(splits)}"
            for digest, splits in seen.items() if len(splits) > 1]
