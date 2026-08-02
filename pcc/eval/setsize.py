"""§6.4 — translate predicted δ̂_y into prediction-set size.

δ_y is a PROXY; what is actually cared about is set-size reduction, and the
δ_y → set-size relation is not simply monotone across classes (§6.1). So gate B/C
predictability that does NOT translate into smaller held-out sets means the wrong
target was chosen — report it and propose an alternative (§6.4, §6.5).

Correction acts on the true-class score: s'(x,y) = s(x,y) − δ̂_y. Label k is in
the set iff s'(x,k) ≤ q̂, i.e. s(x,k) ≤ q̂ + δ̂_k. So the corrected rule is a
PER-CLASS threshold `q̂ + δ̂_k`. Marginal coverage validity is preserved for any
δ̂ (see tests/test_coverage_validity.py); the question here is efficiency.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import build_sets
from pcc.eval.metrics import summary


def corrected_thresholds(q_global: float, delta_hat: np.ndarray) -> np.ndarray:
    """Per-class threshold q̂ + δ̂_k. NaN δ̂ (no prediction) falls back to q̂."""
    d = np.array(delta_hat, float)
    d[~np.isfinite(d)] = 0.0
    return q_global + d


def compare_setsize(score_matrix, labels, n_classes, alpha, q_global, delta_hat,
                    *, group_of_class=None):
    """Uncorrected (global q̂) vs corrected (q̂ + δ̂_k) prediction sets.

    Returns the full §9 metric bundle for BOTH, so any set-size gain is shown
    together with its coverage cost (§9). `group_of_class` enables the
    seen/held-out and head/tail breakdowns that §6.4 requires (never merge them).
    """
    sets_unc = build_sets(score_matrix, q_global)
    sets_cor = build_sets(score_matrix, corrected_thresholds(q_global, delta_hat))
    m_unc = summary(sets_unc, labels, n_classes, alpha, group_of_class=group_of_class)
    m_cor = summary(sets_cor, labels, n_classes, alpha, group_of_class=group_of_class)
    return {
        "uncorrected": m_unc,
        "corrected": m_cor,
        "avg_set_size_delta": m_cor["avg_set_size"] - m_unc["avg_set_size"],
        "marginal_coverage_delta": m_cor["marginal_coverage"] - m_unc["marginal_coverage"],
    }


def setsize_translation_holdout(score_matrix, labels, n_classes, alpha, q_global,
                                delta_hat, held_out_classes):
    """§6.4 gate: does δ̂_y reduce set size ON HELD-OUT CLASSES?

    Restricts the size/coverage comparison to points whose true label is a
    held-out class. Returns corrected vs uncorrected avg set size + coverage on
    that subset, and a `reduces_size` flag (corrected < uncorrected).
    """
    held = set(int(c) for c in held_out_classes)
    mask = np.array([int(y) in held for y in labels])
    if not mask.any():
        raise ValueError("no eval points fall in held_out_classes")
    res = compare_setsize(score_matrix[mask], labels[mask], n_classes, alpha,
                          q_global, delta_hat)
    res["n_holdout_points"] = int(mask.sum())
    res["reduces_size"] = bool(res["avg_set_size_delta"] < 0)
    return res
