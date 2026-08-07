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


def delta_y_matched_n(cal_scores_true, class_of_sample, n_classes, alpha, *,
                      n_cal, seed=42, estimator="empirical"):
    """δ_y estimated from a MATCHED number of calibration samples per class
    (Amendment 2, reports/protocol_amendments.md).

    WHY: the quantile estimator's bias depends on the group size n, and n_y ∝
    prevalence, so δ_y correlates MECHANICALLY with prevalence even when no class
    structure exists. Measured on a control where every class had an identical
    score distribution (true δ_y = 0 for all):

        corr(δ_y, log n_y) = -0.287 (conformal) / +0.227 (empirical)   [artefact]
        matched n_cal = 20                        -> +0.016             [removed]

    This matters because §6.3 gate C asks whether geometry beats log-prevalence,
    and §6.5 stops the project if prevalence explains most of δ_y. A spurious
    prevalence↔δ_y link would therefore kill the project for an estimator artefact.

    Classes with fewer than `n_cal` samples are returned as NaN — report how many
    were dropped, since dropping them is itself prevalence-linked selection.
    Returns (delta [n_classes], kept_mask [n_classes]).

    `estimator` DEFAULTS TO "empirical" (level-matched), and that default matters.
    Matching n removes the dependence of the bias on n_y, but a `conformal`
    quantile ALSO carries a level mismatch between the small per-class group and
    the large pooled group: at n_cal=25 and α=0.1 the class group is evaluated at
    level 0.96 while the pooled group sits at 0.9004. Measured on a control where
    the true δ_y is 0 for every class:

        conformal : mean δ_y = +0.1158  (90% of classes positive)   <- artefact
        empirical : mean δ_y = -0.0093  (39% positive)              <- correct

    On a [0,1] score scale a spurious +0.116 offset inflates every threshold and
    made §6.4 report sets GROWING by ~12.7 labels. Use `estimator="conformal"`
    only for a deployment-valid correction, never for the prediction target.
    """
    from pcc.eval.decomposition import group_quantile

    s = np.asarray(cal_scores_true, float)
    c = np.asarray(class_of_sample)
    rng = np.random.default_rng(seed)

    kept = np.zeros(n_classes, dtype=bool)
    picks = []
    for y in range(n_classes):
        idx = np.where(c == y)[0]
        if len(idx) >= n_cal:
            picks.append(rng.choice(idx, n_cal, replace=False))
            kept[y] = True
    if not picks:
        return np.full(n_classes, np.nan), kept
    pool = np.concatenate(picks)
    q_global = group_quantile(s[pool], alpha, estimator)

    delta = np.full(n_classes, np.nan)
    for y, sel in zip(np.where(kept)[0], picks):
        qy = group_quantile(s[sel], alpha, estimator)
        delta[y] = qy - q_global
    return delta, kept


def prevalence_null(cal_scores_true, class_of_sample, n_classes, alpha, *,
                    n_cal, n_reps=50, seed=42, estimator="conformal"):
    """MANDATORY null control for gate C (Amendment 2).

    Destroys any real class structure while PRESERVING each class's sample count,
    by permuting the class labels of the calibration scores. Returns the
    distribution of corr(δ_y, log n_y) under that null, so gate-C conclusions can
    be judged against the null rather than against zero.

    A prevalence effect that does not exceed this null is an estimator artefact.
    """
    s = np.asarray(cal_scores_true, float)
    c = np.asarray(class_of_sample)
    counts = np.bincount(c, minlength=n_classes)
    rng = np.random.default_rng(seed)

    corrs = []
    for r in range(n_reps):
        c_perm = rng.permutation(c)          # same class sizes, structure destroyed
        d, kept = delta_y_matched_n(s, c_perm, n_classes, alpha, n_cal=n_cal,
                                    seed=seed + 1 + r, estimator=estimator)
        ok = kept & np.isfinite(d) & (counts > 0)
        if ok.sum() > 3:
            corrs.append(_pearson(d[ok], np.log(counts[ok])))
    corrs = np.array([x for x in corrs if not np.isnan(x)], float)
    if len(corrs) == 0:
        # Happens when class counts have no variance (a BALANCED dataset such as
        # CIFAR-100): log n_y is constant, so corr(δ_y, log n_y) is undefined and
        # the prevalence ablation cannot be tested at all. Keys are kept identical
        # so callers never hit a KeyError on this path.
        undefined = bool(np.ptp(counts[counts > 0]) == 0) if (counts > 0).any() else True
        return {"null_mean": np.nan, "null_sd": np.nan, "null_abs_p95": np.nan,
                "n_reps": 0,
                "undefined_reason": ("class counts have no variance (balanced dataset) "
                                     "-> prevalence ablation undefined"
                                     if undefined else
                                     "too few classes retained at this n_cal")}
    return {"null_mean": float(corrs.mean()), "null_sd": float(corrs.std(ddof=1)),
            "null_abs_p95": float(np.percentile(np.abs(corrs), 95)),
            "n_reps": int(len(corrs)), "undefined_reason": None}


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


def class_quantile_reliability(s_true, class_of_sample, n_classes, alpha, *,
                               n_splits=30, min_per_class=4, seed=0,
                               estimator="empirical"):
    """Split-half reliability of the per-CLASS quantile q̂_y on a GIVEN score scale.

    Same machinery as gate A, but applied to q̂_y rather than δ_y (they differ by the
    pooled constant q̂_global, which does not affect a correlation across classes) and
    exposed so Phase 0 can ask it of a TRANSFORMED score matrix.

    Phase 0 (§5) needs exactly this question twice: does class-level structure exist
    above sampling noise on the raw scores, and does it SURVIVE the best global
    temperature? A global temperature cannot itself produce class-level variance, so
    scoring it by class-level R² would rig the comparison; asking whether it REMOVES
    the structure is the question a temperature can actually answer.
    """
    s_true = np.asarray(s_true, float)
    lab = np.asarray(class_of_sample)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        h1 = np.zeros(len(lab), dtype=bool)
        for y in range(n_classes):
            idx = np.where(lab == y)[0]
            if len(idx) < 2 * min_per_class:
                continue
            rng.shuffle(idx)
            h1[idx[: len(idx) // 2]] = True
        q1 = _class_quantiles(s_true, lab, n_classes, alpha, h1, min_per_class,
                              estimator)
        q2 = _class_quantiles(s_true, lab, n_classes, alpha, ~h1, min_per_class,
                              estimator)
        m = np.isfinite(q1) & np.isfinite(q2)
        out.append(spearman_brown(_pearson(q1[m], q2[m])))
    out = np.array(out, float)
    n_valid = int(np.isfinite(out).sum())
    return {"reliability_mean": float(np.nanmean(out)) if n_valid else float("nan"),
            "reliability_splits": out, "n_splits_with_a_value": n_valid}


def _class_quantiles(s_true, lab, n_classes, alpha, mask, min_per_class, estimator):
    from pcc.eval.decomposition import group_quantile
    q = np.full(n_classes, np.nan)
    for y in range(n_classes):
        v = s_true[mask & (lab == y)]
        if len(v) >= min_per_class:
            q[y] = group_quantile(v, alpha, estimator)
    return q


def _delta_on_subset(scores, classes, idx, n_classes, alpha, min_per_class,
                     estimator="empirical"):
    """δ_y computed on a subset of sample indices; NaN for classes with fewer
    than `min_per_class` samples in the subset.

    Uses the LEVEL-MATCHED estimator by default, for the same reason as Amendment 1:
    the conformal quantile's level depends on n, so a small half-class group targets
    a different percentile than the pooled group, and it returns `inf` outright when
    n < ceil(1/α) − 1. On Pl@ntNet (cal median 2) that made gate A return NaN for
    every split even though 523 classes were nominally eligible.
    """
    from pcc.eval.decomposition import group_quantile
    s = scores[idx]
    c = classes[idx]
    q_global = group_quantile(s, alpha, estimator)
    delta = np.full(n_classes, np.nan)
    for y in range(n_classes):
        m = c == y
        if m.sum() >= min_per_class:
            delta[y] = group_quantile(s[m], alpha, estimator) - q_global
    return delta


def split_half_reliability(cal_scores_true, class_of_sample, n_classes, alpha, *,
                           n_splits=100, min_per_class=2, seed=42,
                           group_of_class=None, estimator="empirical",
                           class_subset=None):
    """Gate A (§6.2): split-half reliability of δ_y.

    For each of `n_splits` random splits, halve EACH class's samples, compute δ_y
    independently on each half, correlate the two δ_y vectors across classes
    (classes present in both halves), and Spearman-Brown correct. Returns the
    distribution of SB-corrected reliabilities (mean + CI-ready array).

    `group_of_class` (optional): class_id -> group (e.g. 'head'/'tail'); when
    given, reliability is also reported per group. Gate A threshold: reliability
    >= 0.3 in the realistic-sample regime. This is the CEILING on any predictive
    R² (§6.3) — report R² normalized by it, never raw.

    `class_subset` restricts the CORRELATION to a subset of classes while leaving the
    pooled q̂_global untouched, so the ceiling can be computed at the same granularity as
    the R² it normalizes. This matters and the omission was a real defect: reliability is
    itself a variance ratio, so in a stratum where δ_y barely varies the ceiling is much
    lower than the pooled 0.754 — and normalizing that stratum's R² by the POOLED ceiling
    systematically understates it. §6.3 already requires normalizing by the ceiling; it was
    being applied at the wrong level of aggregation.
    """
    cal_scores_true = np.asarray(cal_scores_true, float)
    class_of_sample = np.asarray(class_of_sample)
    rng = np.random.default_rng(seed)

    overall = []
    per_group = {}
    n_contrib = []
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
        d1 = _delta_on_subset(cal_scores_true, class_of_sample, idx1, n_classes,
                              alpha, min_per_class, estimator)
        d2 = _delta_on_subset(cal_scores_true, class_of_sample, idx2, n_classes,
                              alpha, min_per_class, estimator)
        # isfinite, NOT ~isnan: conformal quantiles can be +inf for tiny classes and
        # an inf slips through a NaN-only filter, turning the correlation into NaN.
        both = np.isfinite(d1) & np.isfinite(d2)
        if class_subset is not None:
            keep = np.zeros(n_classes, dtype=bool)
            keep[np.asarray(class_subset, int)] = True
            both = both & keep
        overall.append(spearman_brown(_pearson(d1[both], d2[both])))

        if group_of_class is not None:
            groups = np.array([group_of_class.get(y, None) for y in range(n_classes)])
            for g in set(v for v in groups if v is not None):
                sel = both & (groups == g)
                per_group.setdefault(g, []).append(
                    spearman_brown(_pearson(d1[sel], d2[sel])))
        n_contrib.append(int(both.sum()))

    overall = np.array(overall, float)
    # How many classes could contribute at all: a split-half needs >= 2*min_per_class
    # samples in the class. On Pl@ntNet the cal median is 2, so this can be tiny and
    # the reliability then comes back NaN. Report the count so a NaN is diagnosable
    # instead of mysterious.
    counts_all = np.bincount(class_of_sample, minlength=n_classes)
    if class_subset is not None:
        keep = np.zeros(n_classes, dtype=bool)
        keep[np.asarray(class_subset, int)] = True
        counts_all = np.where(keep, counts_all, 0)
    n_eligible = int((counts_all >= 2 * min_per_class).sum())
    n_valid = int(np.isfinite(overall).sum())
    out = {"reliability_mean": float(np.nanmean(overall)) if n_valid else float("nan"),
           "reliability_splits": overall,
           "n_splits": n_splits,
           "n_classes_eligible": n_eligible,
           "n_classes_contributing_mean": (float(np.mean(n_contrib)) if n_contrib else 0.0),
           "n_splits_with_a_value": n_valid,
           "undefined_reason": (None if n_valid else
                                f"no split produced a correlation; only {n_eligible} "
                                f"classes have >= {2*min_per_class} calibration samples")}
    if group_of_class is not None:
        out["by_group"] = {g: {"mean": float(np.nanmean(v)),
                               "splits": np.array(v, float)}
                           for g, v in per_group.items()}
    return out
