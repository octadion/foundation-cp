"""Phase 0 (§5) — decompose the efficiency gap on REAL logits.

Measures how much of the average-set-size gap is closed, SEPARATELY, by:
  (1) a single global temperature,
  (2) per-sample energy reweighting (free energy of the pre-softmax logits),
  (3) a per-class offset fit on ABUNDANT data (not a realistic budget).

Pass criterion (§5): (3) must close a gap SUBSTANTIALLY larger than (1) and (2),
with non-overlapping CIs. If not, "structure lives at the class level" does not
hold on real data — STOP, do not enter Phase 1 (§5).

Why this exists: the motivating evidence came from synthetic injection-and-
recovery, which is circular. This must replicate on real logits (ImageNet,
iNaturalist) first.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import build_sets, conformal_quantile, coverage, set_sizes


def temperature_softmax(logits, T):
    z = np.asarray(logits, float) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def thr_scores_from_softmax(probs):
    """THR/LAC nonconformity (repo convention, higher = worse)."""
    return 1.0 - probs


def free_energy(logits):
    """Free energy of the pre-softmax logits: E(x) = -logsumexp(logits)."""
    L = np.asarray(logits, float)
    m = L.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(L - m).sum(axis=1)))


def efficiency(score_matrix, labels, alpha, cal_idx, eval_idx, threshold=None):
    """Avg set size at target coverage. Calibrates the marginal threshold on
    cal_idx (unless `threshold` given) and reports size + coverage on eval_idx."""
    cal_true = score_matrix[cal_idx, labels[cal_idx]]
    q = conformal_quantile(cal_true, alpha) if threshold is None else threshold
    sets = build_sets(score_matrix[eval_idx], q)
    return {"avg_set_size": float(set_sizes(sets).mean()),
            "coverage": coverage(sets, labels[eval_idx]), "threshold": float(q)}


def gap_from_global_temperature(logits, labels, alpha, cal_idx, eval_idx,
                                Ts=None):
    """Best global temperature by eval avg-set-size (component 1). Returns the
    baseline (T=1) size and the best-T size; the gap closed is baseline - best."""
    Ts = Ts if Ts is not None else np.linspace(0.5, 3.0, 26)
    base = efficiency(thr_scores_from_softmax(temperature_softmax(logits, 1.0)),
                      labels, alpha, cal_idx, eval_idx)
    best = base
    best_T = 1.0
    for T in Ts:
        e = efficiency(thr_scores_from_softmax(temperature_softmax(logits, T)),
                       labels, alpha, cal_idx, eval_idx)
        if e["avg_set_size"] < best["avg_set_size"]:
            best, best_T = e, T
    return {"baseline_size": base["avg_set_size"], "best_size": best["avg_set_size"],
            "gap_closed": base["avg_set_size"] - best["avg_set_size"], "best_T": best_T}


def gap_from_per_class_offset(score_matrix, labels, n_classes, alpha,
                              cal_idx, eval_idx):
    """Per-class offset fit on ABUNDANT calibration data (component 3): apply
    classwise δ_y estimated on cal_idx, measure eval avg set size. This is the
    'structure at the class level' upper bound (not a realistic budget)."""
    from pcc.targets.delta import delta_y
    from pcc.eval.setsize import corrected_thresholds

    cal_true = score_matrix[cal_idx, labels[cal_idx]]
    q_global = conformal_quantile(cal_true, alpha)
    delta = delta_y(cal_true, labels[cal_idx], n_classes, alpha)
    base = efficiency(score_matrix, labels, alpha, cal_idx, eval_idx, threshold=q_global)
    sets = build_sets(score_matrix[eval_idx], corrected_thresholds(q_global, delta))
    size = float(set_sizes(sets).mean())
    return {"baseline_size": base["avg_set_size"], "offset_size": size,
            "gap_closed": base["avg_set_size"] - size,
            "coverage": coverage(sets, labels[eval_idx])}


def gap_from_energy_reweighting(*args, **kwargs):
    """Component (2): per-sample energy reweighting. Left as a documented
    interface: the exact reweighting scheme (how free_energy modulates the score
    or the calibration weighting) must be pinned down in the Phase-0 design and
    reproduced, not guessed — a wrong scheme would make the §5 comparison
    meaningless. `free_energy()` above provides the energy input.
    """
    raise NotImplementedError(
        "energy reweighting scheme: specify in Phase-0 design (§5 component 2) "
        "before implementing; do not guess.")
