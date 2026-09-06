"""Invalid live caps must fail before provider/configuration imports."""
import builtins
import runpy
import sys
from pathlib import Path

import pytest


def run_cli(args, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_traces.py"
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    monkeypatch.setattr(sys, "path", list(sys.path))
    real_import = builtins.__import__

    def intercept(name, *a, **kw):
        if name == "app" or name.startswith("app."):
            raise RuntimeError("arguments accepted before app initialization")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", intercept)
    runpy.run_path(str(script), run_name="__main__")


@pytest.mark.parametrize("cap", [None, "nan", "NaN", "inf", "-inf", "0", "-1"])
def test_live_requires_finite_positive_cap_before_app_import(cap, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_traces.py"
    argv = [str(script), "--live"]
    if cap is not None:
        argv.append(f"--max-usd={cap}")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "path", list(sys.path))
    real_import = builtins.__import__

    def no_app_import(name, *args, **kwargs):
        if name == "app" or name.startswith("app."):
            pytest.fail("invalid cap reached application/provider initialization")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_app_import)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 2
    assert "--max-usd" in capsys.readouterr().err


def test_live_served_is_refused_before_app_initialization(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(["--live", "--mode=served", "--max-usd=1"], monkeypatch)
    assert exc.value.code == 2
    assert "served mode has no cost control" in capsys.readouterr().err


def test_finite_positive_cap_is_accepted_without_executing_live_code(monkeypatch):
    with pytest.raises(RuntimeError, match="arguments accepted"):
        run_cli(["--live", "--max-usd=1"], monkeypatch)


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan"])
def test_invalid_sample_sizes_refused_before_initialization(value, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_cli([f"--n={value}"], monkeypatch)
    assert exc.value.code == 2
