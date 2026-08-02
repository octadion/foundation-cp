"""Structured report writing (AGENTS.md §3.4, §12).

Every notebook / experiment writes reports/<name>.json containing: the
hypothesis, the pass criteria WRITTEN BEFORE RUNNING, full config, package
versions, seed, wall-clock time, and results with CIs, plus an explicit
pass/fail conclusion.

`write_report` refuses to write a result that has no pre-registered pass
criteria, because AGENTS.md §12 says such a result is unusable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _pkg_versions(pkgs=("torch", "torchvision", "timm", "numpy", "scipy",
                        "sklearn", "pandas")) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in pkgs:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = "not-installed"
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def environment_stamp() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "git_commit": _git_commit(),
    }


def write_report(
    reports_dir: str | Path,
    name: str,
    *,
    hypothesis: str,
    pass_criteria: str,
    config: dict[str, Any],
    seed: int,
    results: dict[str, Any],
    conclusion: str,
    started_at: float | None = None,
) -> Path:
    """Write reports/<name>.json. Raises if hypothesis/pass_criteria/conclusion
    are empty — a report without pre-registered criteria is unusable (§12).
    """
    if not hypothesis.strip():
        raise ValueError("hypothesis is required (AGENTS.md §12)")
    if not pass_criteria.strip():
        raise ValueError(
            "pass_criteria must be written BEFORE running (AGENTS.md §12); "
            "a result with no pre-registered criteria cannot be used."
        )
    if not conclusion.strip():
        raise ValueError("explicit pass/fail conclusion is required (AGENTS.md §12)")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "hypothesis": hypothesis,
        "pass_criteria": pass_criteria,
        "config": config,
        "seed": seed,
        "environment": environment_stamp(),
        "runtime_seconds": None if started_at is None else round(time.time() - started_at, 2),
        "results": results,
        "conclusion": conclusion,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = reports_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
