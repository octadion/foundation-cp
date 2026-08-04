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




# ---------------------- §6.4 resolved design (Amendment 4) ----------------------

def _heldout_fixture(K=100, npc=400, boost=3.5, seed=0):
    from pcc.eval.decomposition import group_quantile
    rng = np.random.default_rng(seed)
    n = npc * K
    labels = np.repeat(np.arange(K), npc)
    shift = rng.normal(0, 1.2, K)[labels]
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), labels] += boost + shift
    e = np.exp(logits - logits.max(1, keepdims=True))
    S = 1 - e / e.sum(1, keepdims=True)
    idx = rng.permutation(n)
    cal, ev = idx[: n // 2], idx[n // 2:]
    ct = S[cal, labels[cal]]
    qg = group_quantile(ct, 0.1, "empirical")
    delta = np.array([np.quantile(ct[labels[cal] == y], 0.9) - qg for y in range(K)])
    return S[ev], labels[ev], qg, delta, np.arange(K // 2), rng


def test_sec64_oracle_positive_and_null_negative():
    """The §6.4 metric must reward real class structure and punish its null."""
    from pcc.eval.setsize import setsize_translation_heldout_space as f
    S, lab, qg, delta, held, rng = _heldout_fixture()
    oracle = f(S, lab, 0.1, qg, delta, held)["gap"]
    shuffled = delta.copy()
    shuffled[held] = rng.permutation(delta[held])
    null = f(S, lab, 0.1, qg, shuffled, held)["gap"]
    assert oracle > 0, f"oracle gap {oracle:+.3f} not positive"
    assert null < oracle - 1.0, f"shuffled null {null:+.3f} not clearly worse than oracle {oracle:+.3f}"


def test_sec64_constant_delta_is_a_noop():
    """A constant δ̂ is a pure global shift and must carry NO credit. Before
    deflation was allowed it scored −16.27; before the label-space restriction it
    scored +15.54 and BEAT the oracle."""
    from pcc.eval.setsize import setsize_translation_heldout_space as f
    S, lab, qg, delta, held, _ = _heldout_fixture()
    oracle = f(S, lab, 0.1, qg, delta, held)["gap"]
    const = f(S, lab, 0.1, qg, np.full(len(delta), 0.116), held)["gap"]
    shifted = f(S, lab, 0.1, qg, delta + 0.5, held)["gap"]
    assert abs(const) < 0.5, f"constant δ̂ gap {const:+.3f} should be ~0"
    assert abs(shifted - oracle) < 0.1, \
        f"adding a constant changed the oracle result ({oracle:+.3f} -> {shifted:+.3f})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} set-size tests passed")
