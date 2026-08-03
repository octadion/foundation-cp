"""Phase-0 §5 decomposition instrument: positive AND negative controls.

The §5 gate asks whether a per-CLASS offset closes a substantially larger
efficiency gap than a global temperature or a per-sample energy-indexed offset.
An instrument that always favours the per-class component would make the gate
meaningless, so both directions are tested:

  * class structure PRESENT  -> per-class offset must dominate
  * class structure ABSENT   -> per-class offset must NOT dominate

Runnable with pytest OR directly: `python pcc/tests/test_phase0_decomposition.py`.
"""

from __future__ import annotations

import numpy as np

from pcc.eval import decomposition as dc


def _make(n=20000, K=50, *, class_structure: bool, seed=0):
    """Logits whose true-class boost varies either BY CLASS or BY SAMPLE."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, n)
    logits = rng.normal(size=(n, K))
    if class_structure:
        shift = rng.normal(0, 1.5, K)[labels]      # difficulty indexed by CLASS
    else:
        shift = rng.normal(0, 1.5, n)              # difficulty indexed by SAMPLE
    logits[np.arange(n), labels] += 2.0 + shift
    idx = rng.permutation(n)
    return logits, labels, idx[: n // 2], idx[n // 2:]


def _components(logits, labels, cal, ev, K, alpha=0.1):
    S = 1 - dc.temperature_softmax(logits, 1.0)
    temp = dc.gap_from_global_temperature(logits, labels, alpha, cal, ev)
    cls = dc.gap_from_per_class_offset(S, labels, K, alpha, cal, ev)
    sweep = dc.phase0_energy_bin_sweep(logits, S, labels, alpha, cal, ev,
                                       bin_grid=(2, 10, 50))
    energy_best = max(v["gap_closed"] for v in sweep.values())
    return temp["gap_closed"], energy_best, cls["gap_closed"], cls["coverage"]


def test_class_offset_dominates_when_class_structure_present():
    logits, labels, cal, ev = _make(class_structure=True, seed=0)
    t, e, c, cov = _components(logits, labels, cal, ev, K=50)
    assert c > 2 * max(t, e), f"class offset {c:.3f} should dominate temp {t:.3f} / energy {e:.3f}"
    assert cov > 0.87, f"coverage {cov:.3f} collapsed"


def test_class_offset_does_not_dominate_without_class_structure():
    """NEGATIVE CONTROL — the gate must be able to say 'no'."""
    logits, labels, cal, ev = _make(class_structure=False, seed=1)
    t, e, c, cov = _components(logits, labels, cal, ev, K=50)
    assert c <= max(t, e), (
        f"class offset {c:.3f} beat temp {t:.3f} / energy {e:.3f} even though "
        f"difficulty was per-SAMPLE — the §5 instrument is biased toward its own hypothesis")


def test_energy_more_bins_is_not_free_gap():
    """More free parameters must not trivially close more gap (estimation noise
    dominates), so the §5 comparison is not just a parameter-count artefact."""
    logits, labels, cal, ev = _make(class_structure=True, seed=2)
    S = 1 - dc.temperature_softmax(logits, 1.0)
    sweep = dc.phase0_energy_bin_sweep(logits, S, labels, 0.1, cal, ev,
                                       bin_grid=(2, 50))
    assert sweep[50]["gap_closed"] <= sweep[2]["gap_closed"], \
        "more energy bins closed more gap — check for calibration leakage"


def _realistic(n_per_class=100, K=100, boost=3.5, seed=0):
    """The regime that actually broke: HIGH accuracy (~0.75-0.85) and only ~100
    samples/class, i.e. ~50 calibration samples/class after the split. The
    original tests used 400/class with a weak model, which is why they missed it.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * K
    labels = np.repeat(np.arange(K), n_per_class)
    shift = rng.normal(0, 1.2, K)[labels]
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), labels] += boost + shift
    idx = rng.permutation(n)
    return logits, labels, idx[: n // 2], idx[n // 2:]


def test_class_offset_positive_at_realistic_accuracy_and_sample_count():
    """REGRESSION for the level-mismatch bug (reports/protocol_amendments.md).

    With class structure present, the per-class offset must close a POSITIVE gap.
    Using the finite-sample `conformal` quantile makes it strongly NEGATIVE
    (-5.98 on this fixture; -5.6 on real CIFAR-100) because a 50-sample class group
    targets the ~98th percentile while the pooled global group targets the ~95th.
    """
    logits, labels, cal, ev = _realistic()
    S = 1 - dc.temperature_softmax(logits, 1.0)
    emp = dc.gap_from_per_class_offset(S, labels, 100, 0.05, cal, ev,
                                       estimator="empirical")
    assert emp["gap_closed"] > 0, (
        f"per-class gap {emp['gap_closed']:+.3f} is not positive even with the "
        f"level-matched estimator and real class structure present")

    con = dc.gap_from_per_class_offset(S, labels, 100, 0.05, cal, ev,
                                       estimator="conformal")
    assert con["gap_closed"] < emp["gap_closed"], (
        "the conformal estimator should be MORE conservative here; if this fails, "
        "group_quantile no longer distinguishes the two levels")


def test_group_quantile_levels_differ_with_n():
    """The mechanism itself: conformal level depends on n, empirical does not."""
    rng = np.random.default_rng(0)
    big = rng.beta(1.2, 8.0, 5000)
    small = rng.beta(1.2, 8.0, 50)
    # empirical targets the same percentile regardless of n
    e_big = dc.group_quantile(big, 0.05, "empirical")
    e_small = dc.group_quantile(small, 0.05, "empirical")
    assert abs(e_big - e_small) < 0.15, "empirical quantiles should track each other"
    # conformal on the small group is pushed to a higher percentile
    c_small = dc.group_quantile(small, 0.05, "conformal")
    assert c_small >= e_small, "conformal at n=50 must be >= the plain 1-alpha quantile"


def _cc(class_structure, n_per_class, K=100, boost=3.5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_class * K
    labels = np.repeat(np.arange(K), n_per_class)
    shift = (rng.normal(0, 1.2, K)[labels] if class_structure
             else rng.normal(0, 1.2, n))
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), labels] += boost + shift
    idx = rng.permutation(n)
    return dc.phase0_cc_decomposition(logits, labels, K, 0.1,
                                      idx[: n // 2], idx[n // 2:],
                                      bin_grid=(2, 10, 50))


def test_cc_metric_class_dominates_with_structure_and_abundant_data():
    """ADOPTED §5 metric (Amendment 3): with real class structure AND enough
    calibration data, the class-indexed correction must be the cheapest way to
    reach worst-class coverage >= 1-alpha."""
    r = _cc(True, 2000)
    rivals = max(v["gap_vs_global"] for k, v in r.items()
                 if k not in ("class", "global"))
    assert r["class"]["gap_vs_global"] > rivals, (
        f"class gap {r['class']['gap_vs_global']:+.2f} did not beat best rival {rivals:+.2f}")


def test_cc_metric_negative_control_no_class_structure():
    """NEGATIVE CONTROL — without class structure the class mechanism must NOT be
    the cheapest. This is what the original marginal-coverage metric could not do
    in reverse (it could never reward the class mechanism at all)."""
    r = _cc(False, 2000)
    rivals = max(v["gap_vs_global"] for k, v in r.items()
                 if k not in ("class", "global"))
    assert r["class"]["gap_vs_global"] <= rivals, (
        f"class gap {r['class']['gap_vs_global']:+.2f} beat rivals {rivals:+.2f} "
        f"with NO class structure — metric is biased toward its own hypothesis")


def test_cc_metric_cannot_resolve_at_cifar100_sample_count():
    """Documents WHY CIFAR-100 cannot answer Phase 0: at ~50 calibration samples
    per class, even strong real structure does not make the class mechanism win."""
    r = _cc(True, 100)      # 100/class -> ~50 calibration samples/class
    rivals = max(v["gap_vs_global"] for k, v in r.items()
                 if k not in ("class", "global"))
    assert r["class"]["gap_vs_global"] < rivals, (
        "class mechanism unexpectedly won at 50 cal samples/class; if this starts "
        "passing, re-examine the CIFAR-100 'debug only' justification")


def test_min_size_reaches_target_coverage():
    rng = np.random.default_rng(3)
    n, K = 4000, 50
    labels = rng.integers(0, K, n)
    S = rng.random((n, K))
    sz, w, inf = dc.min_size_at_worst_class_coverage(S, labels, K,
                                                    np.full(K, 0.5), 0.9)
    assert w >= 0.9 - 1e-9, f"worst coverage {w:.4f} below target"
    assert 0 <= sz <= K


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} phase-0 decomposition tests passed")
