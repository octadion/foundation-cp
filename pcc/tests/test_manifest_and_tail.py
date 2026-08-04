"""Tests for extraction manifests (resume safety) and tail stratified evaluation.

Runnable with pytest OR directly: `python pcc/tests/test_manifest_and_tail.py`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from pcc.data import manifest as mf
from pcc.eval.tail import (prevalence_strata, macro_coverage_of,
                           evaluate_by_stratum, compare_by_stratum)


# ------------------------------- manifest -------------------------------------

def _make_shards(d, n_shards=3, per=100):
    m = mf.init_manifest(d, provenance={"backbone": "test", "ckpt_sha": "deadbeef"})
    for i in range(n_shards):
        f = f"shard_{i:04d}.npz"
        np.savez(os.path.join(d, f),
                 embeddings=np.zeros((per, 8), np.float32),
                 labels=np.zeros(per, int))
        m = mf.add_shard(d, m, f, n_samples=per)
    return m


def test_manifest_roundtrip_and_resume_point():
    d = tempfile.mkdtemp()
    _make_shards(d)
    assert mf.verify_manifest(d)["ok"]
    assert mf.resume_point(d) == 300


def test_corrupt_shard_truncates_resume_point():
    """A killed session can leave a half-written shard. The resume point must stop
    at the last GOOD shard, otherwise the re-run skips a hole and the embeddings
    silently disagree with the labels."""
    d = tempfile.mkdtemp()
    _make_shards(d)
    with open(os.path.join(d, "shard_0001.npz"), "ab") as fh:
        fh.write(b"garbage")
    v = mf.verify_manifest(d)
    assert not v["ok"] and v["corrupt"] == ["shard_0001.npz"]
    assert mf.resume_point(d) == 100, "resume must stop before the corrupt shard"
    res = mf.truncate_to_valid_prefix(d)
    assert res == {"dropped": 2, "kept": 1}
    assert mf.resume_point(d) == 100


def test_missing_shard_detected():
    d = tempfile.mkdtemp()
    _make_shards(d)
    os.remove(os.path.join(d, "shard_0000.npz"))
    v = mf.verify_manifest(d)
    assert not v["ok"] and v["missing"] == ["shard_0000.npz"]
    assert mf.resume_point(d) == 0


def test_unreadable_manifest_is_treated_as_absent():
    d = tempfile.mkdtemp()
    _make_shards(d)
    with open(mf.manifest_path(d), "w", encoding="utf-8") as f:
        f.write("{ truncated json")
    assert mf.load_manifest(d) is None
    assert mf.resume_point(d) == 0, "a truncated manifest must not crash the resume"


# --------------------------------- tail ---------------------------------------

def _longtail_counts(K=200, seed=0, n_empty=20):
    rng = np.random.default_rng(seed)
    counts = np.zeros(K, int)
    counts[n_empty:] = np.clip((rng.pareto(1.3, K - n_empty) * 3 + 1).astype(int), 1, 500)
    return counts


def test_strata_partition_evaluable_classes_exactly_once():
    counts = _longtail_counts()
    strata, unevaluable = prevalence_strata(counts, 4, min_count=1)
    members = np.concatenate(list(strata.values()))
    assert len(members) == len(set(members.tolist())), "a class appears in two strata"
    assert set(members.tolist()) == set(np.where(counts >= 1)[0].tolist())
    assert set(unevaluable.tolist()) == set(np.where(counts < 1)[0].tolist())
    # unevaluable classes must NEVER be inside a stratum
    assert not (set(members.tolist()) & set(unevaluable.tolist()))


def test_strata_are_ordered_rarest_first():
    counts = _longtail_counts()
    strata, _ = prevalence_strata(counts, 4, min_count=1)
    means = [counts[c].mean() for c in strata.values()]
    assert means == sorted(means), f"strata not ordered by prevalence: {means}"


def test_macro_coverage_ignores_classes_with_no_eval_samples():
    """Macro-coverage is an unweighted mean over classes that HAVE eval samples;
    a class with none must be excluded rather than counted as 0 coverage."""
    K = 10
    labels = np.array([0, 0, 1, 1])          # classes 2..9 absent from eval
    sets = np.ones((4, K), bool)
    cov, n = macro_coverage_of(sets, labels, np.arange(K), K)
    assert n == 2, f"expected 2 contributing classes, got {n}"
    assert abs(cov - 1.0) < 1e-12


def test_compare_by_stratum_reports_both_arms_and_never_pools():
    counts = _longtail_counts(K=120, seed=1, n_empty=10)
    labels = np.repeat(np.arange(len(counts)), counts)
    n, K = len(labels), len(counts)
    rng = np.random.default_rng(2)
    S = rng.random((n, K))
    S[np.arange(n), labels] *= 0.3                      # true label scores lower
    res = compare_by_stratum(S, labels, K, counts, 0.5, rng.normal(0, .02, K), seed=3)
    assert set(res) == {"uncorrected", "corrected", "change"}
    for arm in ("uncorrected", "corrected"):
        assert "_unevaluable" in res[arm], "unevaluable classes must be reported"
        strata = [k for k in res[arm] if not k.startswith("_")]
        assert len(strata) == 4, "strata must stay separate (never pooled)"
        for k in strata:
            assert "macro_coverage" in res[arm][k] and "avg_set_size" in res[arm][k]
    assert res["uncorrected"]["_unevaluable"]["n_classes"] == 10


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} manifest/tail tests passed")
