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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} phase-0 decomposition tests passed")
