# -*- coding: utf-8 -*-
"""Prove that notebook 13 can resume after a session crash.

This is not a re-implementation of the resume logic: the runner cell is lifted out of the
.ipynb and executed, so what is tested is the code that actually runs in Colab. The driver
is stubbed with a counter, which is the only way to tell "read from cache" apart from
"recomputed and got the same answer" -- the distinction the whole mechanism exists for.

Four properties, each one a way the mechanism has already failed or could fail:
  * a finished run lands in the Drive cache immediately, not at the end of a phase
  * an identical re-run reads it and does NOT recompute
  * a run whose config changed is detected as stale and IS recomputed (notebook 12 served
    a seed cached at the old MAX_ROWS_COMP next to a fresh one, a 9x discrepancy)
  * a corrupt cache entry is recomputed rather than crashing the phase
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[2] / "notebooks" / "13_review_addenda.ipynb"


def _runner_source():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "def run_one(" in src:
            return src
    raise AssertionError("runner cell not found in " + str(NB))


class _StubDriver:
    """Stands in for phase2_pcc, counting how often a configuration is really computed."""

    HYPOTHESIS = "stub"
    PASS_CRITERIA = "stub"

    def __init__(self):
        self.calls = []

    def run(self, x):
        self.calls.append(dict(vars(x)))
        return {
            "n_classes": 10, "n_seen": 7, "n_heldout": 3,
            "pcc": {"lambda": 0.1, "n_star": None, "offset": 0.0,
                    "features": ["a"], "provenance": {}},
            "table_1_seen": {"n_classes": 7, "primary_stat": "worst",
                             "delta": {"worst": 0.01}},
            "table_2_heldout": {"n_classes": 3, "primary_stat": "worst",
                                "delta": {"worst": 0.05}},
        }

    def verdict(self, r, stat):
        return "LULUS"


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    """Execute the notebook's runner cell against a temporary cache directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pcc" / "reports").mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()

    # `time`, `shutil`, `os` and `np` are imported by earlier cells in the notebook, so the
    # runner cell relies on them being present. Supplying them here mirrors a "Run all",
    # which is how the notebook is meant to be executed.
    import shutil
    import time

    import numpy as np

    ns = {
        "CACHE_DIR": str(cache),
        "LTC_DIR": str(tmp_path / "no_such_repo"),
        "PRIMARY_S": "primary_scores.npy",
        "os": os, "time": time, "shutil": shutil, "np": np,
    }
    exec(compile(_runner_source(), "<nb13 runner>", "exec"), ns)

    stub = _StubDriver()
    ns["drv"] = stub
    ns["_stub"] = stub
    ns["CACHE_DIR"] = str(cache)      # the cell may re-import os but not reassign this
    return ns, stub, cache


def _call(ns, **over):
    base = dict(scores="primary_scores.npy", labels="y.npy", dataset="synth",
                reports_dir="pcc/reports", alpha=0.10, n_cal=10, max_rows=250_000,
                seed=0)
    base.update(over)
    return ns["run_one"]("R", base.pop("tag", "reuse"), **base)


def test_a_finished_run_is_cached_immediately(harness):
    ns, stub, cache = harness
    _call(ns)
    assert len(stub.calls) == 1
    files = list(cache.glob("*.json"))
    assert len(files) == 1, "nothing was written to the Drive cache"
    assert files[0].name == "nb13_R_reuse_s0.json"
    assert len(ns["RESULTS"]) == 1
    assert ns["is_cached"]("R", "reuse", 0) is True


def test_an_identical_rerun_reads_the_cache_and_does_not_recompute(harness):
    ns, stub, cache = harness
    _call(ns)
    _call(ns)
    assert len(stub.calls) == 1, "the second call recomputed instead of resuming"
    # both attempts must appear in RESULTS, else the summary cell loses half its rows
    assert len(ns["RESULTS"]) == 2
    assert ns["RESULTS"][0]["res"] == ns["RESULTS"][1]["res"]


def test_a_changed_config_is_detected_as_stale_and_recomputed(harness):
    """Notebook 12 served a seed cached at the old MAX_ROWS_COMP beside a fresh one at the
    new value, and the two differed by 9x. The key is (phase, tag, seed), so only the
    stored config can catch it."""
    ns, stub, cache = harness
    _call(ns)
    _call(ns, max_rows=500_000)
    assert len(stub.calls) == 2, "a config change was silently served from cache"
    assert stub.calls[1]["max_rows"] == 500_000


@pytest.mark.parametrize("field,value", [
    ("alpha", 0.05), ("n_cal", 25), ("frac_recal", 0.25),
    ("dist_metric", "euclidean"), ("drop_features", ("w_bias",)),
])
def test_every_axis_this_notebook_sweeps_invalidates_the_cache(harness, field, value):
    """A new axis that is NOT in the comparison list is the dangerous case: one descriptor
    arm would be served the cached report of another."""
    ns, stub, _ = harness
    _call(ns)
    _call(ns, **{field: value})
    assert len(stub.calls) == 2, "%s did not invalidate the cache" % field


def test_a_corrupt_cache_entry_is_recomputed_not_fatal(harness):
    ns, stub, cache = harness
    _call(ns)
    (cache / "nb13_R_reuse_s0.json").write_text("{not json at all" + "x" * 300)
    _call(ns)
    assert len(stub.calls) == 2, "a corrupt entry should be recomputed"


def test_missing_max_rows_on_the_primary_dump_is_refused(harness):
    """The 2026-08-19 crash: three phases omitted max_rows, loaded the full 4.61 GB dump,
    and would have produced numbers measured at 524 calibration rows per class instead of
    177 -- incomparable with every table already in the paper."""
    ns, stub, _ = harness
    with pytest.raises(AssertionError, match="max_rows"):
        ns["build"](scores="primary_scores.npy", labels="y.npy")
    assert stub.calls == []


def test_a_non_primary_dump_may_omit_max_rows(harness):
    """The guard must not fire for the backbone or long-tail dumps, which are small."""
    ns, _, _ = harness
    x = ns["build"](scores="some_other_dump.npy", labels="y.npy")
    assert x.max_rows is None
