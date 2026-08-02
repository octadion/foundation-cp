"""§6.4 set-size translation + coverage-cost visibility.

Runnable with pytest OR directly: `python pcc/tests/test_setsize.py`.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import conformal_quantile
from pcc.eval.setsize import compare_setsize, setsize_translation_holdout


def _synthetic(n=12000, n_classes=40, seed=0):
    """Model where some classes are systematically harder (need a larger
    threshold). A well-chosen per-class offset should shrink sets on the easy
    classes without losing coverage overall."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n)
    # class difficulty: half the classes are 'easy' (peaked), half 'hard'
    hard = np.zeros(n_classes, dtype=bool); hard[n_classes // 2:] = True
    logits = rng.normal(size=(n, n_classes))
    boost = np.where(hard[labels], 1.0, 3.0)  # easy classes get a bigger boost
    logits[np.arange(n), labels] += boost
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    return 1.0 - probs, labels, hard


def test_compare_setsize_reports_both_and_coverage_cost():
    S, labels, hard = _synthetic()
    n_classes = S.shape[1]
    alpha = 0.1
    n = len(labels)
    idx = np.random.default_rng(1).permutation(n)
    cal, ev = idx[: n // 2], idx[n // 2:]
    from pcc.targets.delta import delta_y
    cal_true = S[cal, labels[cal]]
    q_global = conformal_quantile(cal_true, alpha)
    delta = delta_y(cal_true, labels[cal], n_classes, alpha)

    res = compare_setsize(S[ev], labels[ev], n_classes, alpha, q_global, delta)
    # both metric bundles present, with coverage reported for each (the §9 cost)
    assert "marginal_coverage" in res["uncorrected"]
    assert "marginal_coverage" in res["corrected"]
    # per-class offset should not destroy marginal coverage
    assert res["corrected"]["marginal_coverage"] > (1 - alpha) - 0.03


def test_holdout_translation_runs_and_flags_direction():
    S, labels, hard = _synthetic(seed=2)
    n_classes = S.shape[1]
    alpha = 0.1
    n = len(labels)
    idx = np.random.default_rng(3).permutation(n)
    cal, ev = idx[: n // 2], idx[n // 2:]
    from pcc.targets.delta import delta_y
    cal_true = S[cal, labels[cal]]
    q_global = conformal_quantile(cal_true, alpha)
    delta = delta_y(cal_true, labels[cal], n_classes, alpha)
    held = np.arange(n_classes // 2)  # pretend the 'easy' classes are held out
    res = setsize_translation_holdout(S[ev], labels[ev], n_classes, alpha,
                                      q_global, delta, held)
    assert "reduces_size" in res and isinstance(res["reduces_size"], bool)
    assert res["n_holdout_points"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} set-size tests passed")
