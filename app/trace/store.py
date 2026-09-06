"""Append-only trace store: one directory per run, JSONL trajectories.

    <trace_dir>/<run_id>/manifest.json        provenance, written once
    <trace_dir>/<run_id>/trajectories.jsonl   one Trajectory per line

Two rules exist because the checkpoint protocol depends on them:

  * The manifest is immutable. Rewriting it with different content raises, so a
    run's recorded provider versions, config hashes and seeds cannot drift away
    from the traces they describe.
  * ``read_run`` refuses an unknown ``schema_version`` instead of coercing it.
    A silently-upgraded trace is a silently-wrong dataset.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from ..config import CONFIG_DIR, ROOT, get_settings, load_models
from .contract import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaVersionError,
    TraceManifest,
    Trajectory,
)

MANIFEST_NAME = "manifest.json"
TRAJECTORIES_NAME = "trajectories.jsonl"

# Cost conservation tolerance: float addition over many steps, not a real gap.
_COST_TOL = 1e-9


def trace_root() -> Path:
    return Path(get_settings().trace_dir)


def run_dir(run_id: str) -> Path:
    return trace_root() / run_id


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return "unknown"


def model_snapshot() -> dict[str, Any]:
    """Model ids with the provider and prices in force at collection time.

    Prices and provider routing change under us; a cost table built from traces
    is only defensible if it says which prices produced it.
    """
    from ..providers.registry import get_registry

    reg = get_registry()
    snap: dict[str, Any] = {}
    for m in load_models():
        _adapter, provider_model, label = reg.resolve(m["id"])
        snap[m["id"]] = {
            "live_provider": label,
            "live_model": provider_model,
            "cost_in": m.get("cost_in", 0.0),
            "cost_out": m.get("cost_out", 0.0),
        }
    return snap


def build_manifest(
    run_id: str,
    dataset: str = "unknown",
    split: str = "pilot",
    seeds: Optional[list[int]] = None,
    command: str = "",
    notes: str = "",
) -> TraceManifest:
    import pydantic

    return TraceManifest(
        run_id=run_id,
        created_ts=time.time(),
        git_sha=_git_sha(),
        router_config_hash=_file_hash(CONFIG_DIR / "router.yaml"),
        models_config_hash=_file_hash(CONFIG_DIR / "models.yaml"),
        model_snapshot=model_snapshot(),
        seeds=seeds or [],
        force_mock=get_settings().force_mock,
        dataset=dataset,
        split=split,
        command=command or " ".join(sys.argv),
        env={"python": sys.version.split()[0], "pydantic": pydantic.VERSION},
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
class ManifestConflictError(RuntimeError):
    """An existing manifest for this run differs from the one being written."""


class RunExistsError(RuntimeError):
    """A run id already holds trajectories. Run ids are single-use."""


def create_run(manifest: TraceManifest) -> Path:
    """Open a new run. Refuses to reuse a run id that already holds traces.

    Appending to an existing run would silently mix two collections -- possibly
    under different code, prices or policies -- behind one manifest. The
    manifest is what makes a run interpretable, so a run id gets exactly one.
    """
    d = run_dir(manifest.run_id)
    existing = d / TRAJECTORIES_NAME
    if existing.exists() and existing.stat().st_size > 0:
        raise RunExistsError(
            f"run {manifest.run_id!r} already holds trajectories at {existing}; "
            "run ids are single-use, choose a new one"
        )
    write_manifest(manifest)
    return d


def write_manifest(manifest: TraceManifest) -> Path:
    d = run_dir(manifest.run_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / MANIFEST_NAME
    payload = manifest.model_dump_json(indent=2)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if json.loads(existing) != json.loads(payload):
            raise ManifestConflictError(
                f"manifest for run {manifest.run_id!r} already exists with different "
                "content; a run's provenance is immutable, start a new run instead"
            )
        return path
    path.write_text(payload, encoding="utf-8")
    return path


def append(trajectory: Trajectory) -> Path:
    """Append one trajectory. Creates the run directory if needed."""
    d = run_dir(trajectory.run_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / TRAJECTORIES_NAME
    with open(path, "a", encoding="utf-8") as f:
        f.write(trajectory.model_dump_json() + "\n")
    return path


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def read_manifest(run_id: str) -> TraceManifest:
    path = run_dir(run_id) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no manifest for run {run_id!r} at {path}")
    return TraceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def iter_trajectories(run_id: str) -> Iterator[Trajectory]:
    path = run_dir(run_id) / TRAJECTORIES_NAME
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Version-gate BEFORE validating so an unknown schema fails with a
            # clear error rather than a wall of field-level complaints.
            version = raw.get("schema_version")
            if version not in SUPPORTED_SCHEMA_VERSIONS:
                raise SchemaVersionError(
                    f"{path}:{lineno} was written by trace schema {version!r}; "
                    f"this build understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
                )
            yield Trajectory.model_validate(raw)


def read_run(run_id: str) -> list[Trajectory]:
    return list(iter_trajectories(run_id))


def list_runs() -> list[str]:
    root = trace_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- #
# Validate
# --------------------------------------------------------------------------- #
def validate_run(run_id: str) -> dict[str, Any]:
    """Structural check plus the honesty checks the strategy requires.

    ``analysis_grade`` is False whenever any outcome came from the mock provider.
    Such a run is still readable and useful for testing the pipeline; it is not
    evidence about model behaviour.
    """
    errors: list[str] = []
    trajectories: list[Trajectory] = []
    try:
        trajectories = read_run(run_id)
    except (SchemaVersionError, ValueError) as e:
        errors.append(str(e))

    manifest = None
    try:
        manifest = read_manifest(run_id)
    except FileNotFoundError as e:
        errors.append(str(e))

    synthetic = 0
    cost_mismatches: list[str] = []
    for t in trajectories:
        if t.has_synthetic_outcomes:
            synthetic += 1
        summed = t.summed_cost()
        for field in ("tokens_in", "tokens_out", "llm_calls", "est_cost_usd", "compute_units"):
            a, b = getattr(t.total_cost, field), getattr(summed, field)
            if abs(a - b) > _COST_TOL:
                cost_mismatches.append(f"{t.trajectory_id}.{field}: total={a} vs steps={b}")

    structural_errors = [e for e in errors if "hangs off unknown state" in e
                         or "does not store" in e or "schema" in e.lower()]
    errors.extend(cost_mismatches)
    force_mock = bool(manifest.force_mock) if manifest else False
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "trajectories": len(trajectories),
        "manifest_present": manifest is not None,
        "force_mock": force_mock,
        "synthetic_trajectories": synthetic,
        "cost_conservation_ok": not cost_mismatches,
        "chain_ok": not structural_errors,
        # A run is evidence about model behaviour only if it validates, contains
        # no mock-provider outcome, and was not collected under forced mock.
        "analysis_grade": bool(trajectories) and not errors and synthetic == 0 and not force_mock,
        "errors": errors[:20],
    }
