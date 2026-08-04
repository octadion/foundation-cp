"""Tail evaluation via macro-coverage aggregation.

WHY THIS MODULE EXISTS (measured, 2026-08-04). On Pl@ntNet's released `cal` split
the median class has **2** calibration samples and 105 classes have none, so ~72%
of classes cannot even receive a finite classwise conformal threshold at α=0.1.
Two consequences:

1. There is **no ground-truth δ_y** for a 2-sample class, so gate B (R² against
   δ_y) is necessarily restricted to an estimable, prevalence-selected subset.
2. But the tail is exactly where the claim is supposed to pay off — a predicted
   correction needs no calibration samples for its target class, while every
   incumbent per-class method is undefined there and falls back to global.

So the tail must still be EVALUATED, just not used as a regression target. A single
2-sample class yields a useless coverage estimate (0/2, 1/2 or 2/2), but the
**unweighted mean of per-class coverages over hundreds of tail classes** is
estimable. That is macro-coverage (Bhattacharyya, Ding & Barber, arXiv 2606.28598),
whose reference implementation defines it as `class_cov[valid].mean()` and also
formalizes restricting it to a subset of classes (`macro_cov_plus`).

Everything here therefore reports per-STRATUM macro-coverage with a bootstrap CI
over classes, never a per-class number, and never pooled with the head.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import build_sets
from pcc.eval.metrics import per_class_coverage


def prevalence_strata(counts, n_strata: int = 4, *, min_count: int = 1):
    """Split classes into prevalence strata (quartiles by default).

    `counts` is the per-class calibration/train count. Classes with fewer than
    `min_count` samples cannot be evaluated at all and are returned separately —
    they must be reported, not dropped silently (§6.2 head-vs-tail, and the
    prevalence-linked-selection warning in descriptor_stability_findings.md).

    Returns (strata dict name -> class-id array, unevaluable class-id array).
    Stratum 0 is the rarest.
    """
    counts = np.asarray(counts)
    n_classes = len(counts)
    unevaluable = np.where(counts < min_count)[0]
    ok = np.where(counts >= min_count)[0]
    if len(ok) == 0:
        return {}, unevaluable
    order = ok[np.argsort(counts[ok], kind="stable")]
    parts = np.array_split(order, n_strata)
    strata = {}
    for i, p in enumerate(parts):
        lo, hi = (int(counts[p].min()), int(counts[p].max())) if len(p) else (0, 0)
        strata[f"q{i}_n{lo}-{hi}"] = p
    return strata, unevaluable


def macro_coverage_of(sets, labels, class_subset, n_classes):
    """Unweighted mean per-class coverage over `class_subset` (macro_cov_plus).

    Classes in the subset with no EVAL samples are NaN and excluded from the mean;
    the count of contributing classes is returned so thin strata are visible.
    """
    cov = per_class_coverage(sets, labels, n_classes)
    sel = cov[np.asarray(class_subset, int)]
    valid = ~np.isnan(sel)
    if not valid.any():
        return np.nan, 0
    return float(sel[valid].mean()), int(valid.sum())


def _bootstrap_ci_over_classes(per_class_values, *, n_boot=2000, seed=42, alpha=0.05):
    """CI for a mean over CLASSES (resample classes, not samples).

    Resampling classes is the right unit here: macro-coverage is an average over
    classes, so its uncertainty comes from which classes we happened to observe.
    """
    v = np.asarray(per_class_values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_classes": int(len(v))}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = v[rng.integers(0, len(v), len(v))].mean()
    return {"mean": float(v.mean()),
            "ci_low": float(np.percentile(boot, 100 * alpha / 2)),
            "ci_high": float(np.percentile(boot, 100 * (1 - alpha / 2))),
            "n_classes": int(len(v))}


def evaluate_by_stratum(score_matrix, labels, n_classes, thresholds, counts, *,
                        n_strata=4, min_count=1, seed=42):
    """Per-stratum macro-coverage and set size for one threshold vector.

    `thresholds` is scalar (global) or a per-class vector (e.g. q̂ + δ̂_y).
    Returns {stratum: {macro_coverage(+CI), avg_set_size, n_classes, n_points}}
    plus an `unevaluable` record. Strata are NEVER pooled (§9, §7 two-tables rule).
    """
    sets = build_sets(score_matrix, thresholds)
    cov = per_class_coverage(sets, labels, n_classes)
    sizes = sets.sum(axis=1)
    strata, unevaluable = prevalence_strata(counts, n_strata, min_count=min_count)

    out = {}
    for name, cls in strata.items():
        mask = np.isin(labels, cls)
        ci = _bootstrap_ci_over_classes(cov[cls], seed=seed)
        out[name] = {"macro_coverage": ci["mean"],
                     "macro_ci_low": ci["ci_low"], "macro_ci_high": ci["ci_high"],
                     "n_classes_contributing": ci["n_classes"],
                     "n_classes_in_stratum": int(len(cls)),
                     "n_eval_points": int(mask.sum()),
                     "avg_set_size": float(sizes[mask].mean()) if mask.any() else np.nan}
    out["_unevaluable"] = {"n_classes": int(len(unevaluable)),
                           "note": f"fewer than {min_count} calibration samples; "
                                   f"no per-class coverage is defined for these"}
    return out


def compare_by_stratum(score_matrix, labels, n_classes, counts, q_global, delta_hat,
                       *, n_strata=4, min_count=1, seed=42):
    """Uncorrected global threshold vs corrected q̂ + δ̂_y, PER STRATUM.

    This is the tail-facing evaluation the claim needs: it works even where δ_y is
    unmeasurable, because it only requires coverage aggregated over many classes.
    Reports both arms so any set-size change is shown together with its coverage
    consequence (§9) — and separately for the rare and common strata, never merged.
    """
    d = np.array(delta_hat, float)
    d[~np.isfinite(d)] = 0.0
    unc = evaluate_by_stratum(score_matrix, labels, n_classes, q_global, counts,
                              n_strata=n_strata, min_count=min_count, seed=seed)
    cor = evaluate_by_stratum(score_matrix, labels, n_classes, q_global + d, counts,
                              n_strata=n_strata, min_count=min_count, seed=seed)
    delta = {}
    for k in unc:
        if k.startswith("_"):
            continue
        delta[k] = {
            "macro_coverage_change": cor[k]["macro_coverage"] - unc[k]["macro_coverage"],
            "avg_set_size_change": cor[k]["avg_set_size"] - unc[k]["avg_set_size"],
        }
    return {"uncorrected": unc, "corrected": cor, "change": delta}
