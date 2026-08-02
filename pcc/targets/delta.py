"""δ_y target and its reliability estimator (AGENTS.md §6.1, §6.2).

δ_y = q̂_y − q̂_global, where q̂_y is the classwise conformal quantile and
q̂_global the marginal quantile, per (backbone × dataset × base-score).

δ_y is a PROXY. What is actually cared about is set-size reduction, and the
δ_y → set-size relation is not simply monotone across classes — so §6.4
(translate to set size) is mandatory, not optional.

Reliability (§6.2, the real gate A): split-half reliability of δ_y —
split each class's samples into two random halves, compute δ_y independently on
each, correlate across classes, Spearman-Brown correct, repeat >=100 times.
This sets the CEILING on any achievable predictive R²; reporting R² without this
ceiling is an analysis error.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import conformal_quantile


def delta_y(cal_scores_true, class_of_sample, n_classes, alpha):
    """δ_y for every class from calibration nonconformity scores.

    cal_scores_true[i] = nonconformity of the true label of calibration sample i.
    class_of_sample[i] = its class. Returns array [n_classes] of δ_y (NaN for
    classes with no calibration samples).
    """
    cal_scores_true = np.asarray(cal_scores_true, float)
    class_of_sample = np.asarray(class_of_sample)
    q_global = conformal_quantile(cal_scores_true, alpha)
    delta = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = class_of_sample == c
        if mask.any():
            q_c = conformal_quantile(cal_scores_true[mask], alpha)
            delta[c] = q_c - q_global
    return delta


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman_brown(r):
    """Spearman-Brown correction for split-half (doubling) reliability.

    The two halves each use ~half the per-class samples; SB estimates the
    reliability of the full-length measurement: r_sb = 2r / (1 + r).
    """
    if np.isnan(r):
        return np.nan
    return float(2 * r / (1 + r)) if (1 + r) != 0 else np.nan


def _delta_on_subset(scores, classes, idx, n_classes, alpha, min_per_class):
    """δ_y computed on a subset of sample indices; NaN for classes with fewer
    than `min_per_class` samples in the subset."""
    s = scores[idx]
    c = classes[idx]
    q_global = conformal_quantile(s, alpha)
    delta = np.full(n_classes, np.nan)
    for y in range(n_classes):
        m = c == y
        if m.sum() >= min_per_class:
            delta[y] = conformal_quantile(s[m], alpha) - q_global
    return delta


def split_half_reliability(cal_scores_true, class_of_sample, n_classes, alpha, *,
                           n_splits=100, min_per_class=2, seed=42,
                           group_of_class=None):
    """Gate A (§6.2): split-half reliability of δ_y.

    For each of `n_splits` random splits, halve EACH class's samples, compute δ_y
    independently on each half, correlate the two δ_y vectors across classes
    (classes present in both halves), and Spearman-Brown correct. Returns the
    distribution of SB-corrected reliabilities (mean + CI-ready array).

    `group_of_class` (optional): class_id -> group (e.g. 'head'/'tail'); when
    given, reliability is also reported per group. Gate A threshold: reliability
    >= 0.3 in the realistic-sample regime. This is the CEILING on any predictive
    R² (§6.3) — report R² normalized by it, never raw.
    """
    cal_scores_true = np.asarray(cal_scores_true, float)
    class_of_sample = np.asarray(class_of_sample)
    rng = np.random.default_rng(seed)

    overall = []
    per_group = {}
    for _ in range(n_splits):
        # split each class's samples into two halves
        h1_mask = np.zeros(len(class_of_sample), dtype=bool)
        for y in range(n_classes):
            idx_y = np.where(class_of_sample == y)[0]
            if len(idx_y) < 2 * min_per_class:
                continue
            rng.shuffle(idx_y)
            h1_mask[idx_y[: len(idx_y) // 2]] = True
        idx1 = np.where(h1_mask)[0]
        idx2 = np.where(~h1_mask)[0]
        d1 = _delta_on_subset(cal_scores_true, class_of_sample, idx1, n_classes, alpha, min_per_class)
        d2 = _delta_on_subset(cal_scores_true, class_of_sample, idx2, n_classes, alpha, min_per_class)
        both = ~np.isnan(d1) & ~np.isnan(d2)
        overall.append(spearman_brown(_pearson(d1[both], d2[both])))

        if group_of_class is not None:
            groups = np.array([group_of_class.get(y, None) for y in range(n_classes)])
            for g in set(v for v in groups if v is not None):
                sel = both & (groups == g)
                per_group.setdefault(g, []).append(
                    spearman_brown(_pearson(d1[sel], d2[sel])))

    overall = np.array(overall, float)
    out = {"reliability_mean": float(np.nanmean(overall)),
           "reliability_splits": overall,
           "n_splits": n_splits}
    if group_of_class is not None:
        out["by_group"] = {g: {"mean": float(np.nanmean(v)),
                               "splits": np.array(v, float)}
                           for g, v in per_group.items()}
    return out
