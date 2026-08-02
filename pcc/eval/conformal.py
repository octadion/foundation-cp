"""Generic split-conformal primitives. Phase-agnostic — used by every phase and
by the baselines. Nothing here depends on g_θ or on any PCC-specific method.

The security property (AGENTS.md §1): for a modified score
`s'(x, y) = s(x, y) − δ̂_y`, marginal split-CP coverage holds with NO assumption
on g_θ, because s' is still a measurable function of (x, y). These primitives
build sets from an arbitrary score matrix, so they can be fed either raw scores
or corrected scores identically — which is what makes the coverage-validity
test (tests/test_coverage_validity.py) meaningful.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-α))/n empirical
    quantile of the calibration nonconformity scores.

    Convention: `cal_scores[i]` is the nonconformity score of the *true* label
    of calibration point i. Higher score = more nonconforming.
    """
    n = len(cal_scores)
    if n == 0:
        return np.inf  # no calibration data -> infinite threshold (see fallback_policy.md)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return np.inf
    return float(np.quantile(cal_scores, level, method="higher"))


def build_sets(score_matrix: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Boolean prediction-set membership.

    `score_matrix[i, k]` = nonconformity score of label k for point i. A label is
    included iff its score <= threshold. `threshold` may be a scalar (marginal)
    or a per-class vector of length n_classes (class-conditional / corrected).
    """
    return score_matrix <= threshold


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Empirical marginal coverage: fraction of points whose true label is in
    the predicted set."""
    return float(sets[np.arange(len(labels)), labels].mean())


def set_sizes(sets: np.ndarray) -> np.ndarray:
    """Per-point prediction-set sizes."""
    return sets.sum(axis=1)
