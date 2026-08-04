"""Resume correctness for sharded extraction.

The property that matters: a killed-and-resumed extraction must produce output
IDENTICAL to a single clean pass. If it does not, embeddings desynchronise from
labels and every downstream number is quietly wrong — and nothing visibly breaks.

Runnable with pytest OR directly: `python pcc/tests/test_extract_resume.py`.
"""

from __future__ import annotations

import tempfile

import numpy as np

from pcc.data.manifest import resume_point
from pcc.extract.forward import (extract_sharded, load_extracted,
                                 per_class_quota_indices)

N, D, K = 5000, 16, 10
_DATA = np.arange(N)[:, None] * np.ones((1, D))   # order-revealing on purpose
_LABELS = np.arange(N) % K
_PROV = {"backbone": "stub", "ckpt_sha": "x"}


def _make_iter(stop_after=None):
    def it(start):
        emitted = 0
        for b in range(start, N, 64):
            e = min(b + 64, N)
            if stop_after is not None and emitted >= stop_after:
                raise RuntimeError("SESSION KILLED")
            yield {"embeddings": _DATA[b:e].astype(np.float32), "labels": _LABELS[b:e]}
            emitted += e - b
    return it


def _clean_pass():
    d = tempfile.mkdtemp()
    extract_sharded(_make_iter(), N, d, provenance=_PROV, shard_size=500, verbose=False)
    return load_extracted(d)


def test_single_pass_preserves_order():
    ref = _clean_pass()
    assert ref["embeddings"].shape == (N, D)
    assert np.array_equal(ref["labels"], _LABELS)
    assert np.array_equal(ref["embeddings"][:, 0], np.arange(N))


def test_kill_and_resume_is_identical_to_one_pass():
    ref = _clean_pass()
    d = tempfile.mkdtemp()
    try:
        extract_sharded(_make_iter(stop_after=1700), N, d, provenance=_PROV,
                        shard_size=500, verbose=False)
        raise AssertionError("the stub should have raised mid-run")
    except RuntimeError:
        pass
    partial = resume_point(d)
    assert 0 < partial < N, f"expected a partial extraction, got {partial}"

    extract_sharded(_make_iter(), N, d, provenance=_PROV, shard_size=500, verbose=False)
    got = load_extracted(d)
    assert np.array_equal(got["embeddings"], ref["embeddings"]), \
        "resumed embeddings differ from a clean pass"
    assert np.array_equal(got["labels"], ref["labels"]), \
        "resumed labels differ from a clean pass"


def test_completed_extraction_is_a_noop():
    d = tempfile.mkdtemp()
    extract_sharded(_make_iter(), N, d, provenance=_PROV, shard_size=500, verbose=False)
    again = extract_sharded(_make_iter(), N, d, provenance=_PROV, shard_size=500,
                            verbose=False)
    assert again["n_written"] == 0 and again["complete"]


def test_provenance_drift_is_refused():
    """Appending shards from a DIFFERENT model to an existing directory would mix
    two models' embeddings into one array with nothing to reveal it."""
    d = tempfile.mkdtemp()
    extract_sharded(_make_iter(), N, d, provenance=_PROV, shard_size=500, verbose=False)
    try:
        extract_sharded(_make_iter(), N, d,
                        provenance={"backbone": "OTHER", "ckpt_sha": "y"},
                        shard_size=500, verbose=False)
        raise AssertionError("provenance drift was not refused")
    except ValueError as e:
        assert "provenance drift" in str(e)


def test_per_class_quota_is_deterministic_and_capped():
    labels = np.repeat(np.arange(20), np.arange(1, 21))   # 1..20 samples per class
    a = per_class_quota_indices(labels, 5, seed=7)
    b = per_class_quota_indices(labels, 5, seed=7)
    assert np.array_equal(a, b), "quota selection must be deterministic (resume needs it)"
    assert np.array_equal(a, np.sort(a)), "indices must be sorted for a stable order"
    counts = np.bincount(labels[a], minlength=20)
    assert counts.max() <= 5
    for y in range(20):
        assert counts[y] == min(5, y + 1)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} extract-resume tests passed")
