"""Descriptor stability vs. images-per-class (AGENTS.md §3.3) — MANDATORY before
Phase 1.

Why this must run first: if the per-class image quota is too small, the class
mean and within-class covariance are noisy, and that input noise depresses R² in
Phase 1. A negative Phase-1 result would then be **ambiguous** — no signal, or a
bad descriptor? This measurement removes the ambiguity by fixing the quota at a
point where the descriptor has already stabilized.

Method: for each quota q, draw TWO DISJOINT subsets of q images per class,
build φ(y) independently on each, and correlate each descriptor feature across
classes. High correlation = the descriptor is reproducible at that q. Repeat over
several draws and report mean + CI. Requires >= 2q images per class available.
"""

from __future__ import annotations

import numpy as np

from pcc.descriptors.phi import build_descriptors


def _disjoint_halves(classes, n_classes, q, rng):
    """Two disjoint index sets with q samples per class each (classes with fewer
    than 2q available are skipped in both)."""
    a, b = [], []
    for y in range(n_classes):
        idx = np.where(classes == y)[0]
        if len(idx) < 2 * q:
            continue
        pick = rng.permutation(idx)[: 2 * q]
        a.extend(pick[:q].tolist())
        b.extend(pick[q:].tolist())
    return np.array(a, int), np.array(b, int)


def _corr_per_feature(P1, P2):
    """Pearson correlation across classes, per descriptor column."""
    out = np.full(P1.shape[1], np.nan)
    ok = np.isfinite(P1).all(axis=1) & np.isfinite(P2).all(axis=1)
    for j in range(P1.shape[1]):
        a, b = P1[ok, j], P2[ok, j]
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            out[j] = np.corrcoef(a, b)[0, 1]
    return out


# Features that are CONSTANT BY CONSTRUCTION in this study: the design forces
# exactly q samples per class, so sample-count features carry no cross-class
# variance here and their "correlation" is undefined/meaningless. In the real
# descriptor these DO vary (long tail) — they must be computed from the FULL
# train counts via build_descriptors(log_prevalence_from=...), not from the quota.
QUOTA_DETERMINED = ("n_eff", "log_prevalence")


def descriptor_stability(features, logits, classes, n_classes, *,
                         quotas=(10, 25, 50, 100), n_reps=5, seed=42,
                         stable_threshold=0.9):
    """Stability of φ(y) as a function of images-per-class.

    Returns per-quota mean/CI of the per-feature correlation between two disjoint
    draws, the per-feature detail, and `recommended_quota`: the smallest q whose
    mean correlation reaches `stable_threshold` (None if none does — that itself
    is the reportable finding).
    """
    features = np.asarray(features, float)
    classes = np.asarray(classes)
    logits = None if logits is None else np.asarray(logits, float)
    rng = np.random.default_rng(seed)

    per_quota = {}
    names = None
    for q in quotas:
        reps = []
        n_classes_used = 0
        for _ in range(n_reps):
            ia, ib = _disjoint_halves(classes, n_classes, q, rng)
            if len(ia) == 0:
                break
            P1, names = build_descriptors(features[ia], None if logits is None else logits[ia],
                                          classes[ia], n_classes)
            P2, _ = build_descriptors(features[ib], None if logits is None else logits[ib],
                                      classes[ib], n_classes)
            reps.append(_corr_per_feature(P1, P2))
            n_classes_used = int(np.isfinite(P1).all(axis=1).sum())
        if not reps:
            per_quota[q] = {"insufficient_data": True}
            continue
        R = np.vstack(reps)                       # [n_reps, n_features]
        with np.errstate(invalid="ignore"):
            per_feature = np.array([np.nanmean(R[:, j]) if np.isfinite(R[:, j]).any()
                                    else np.nan for j in range(R.shape[1])])
            # aggregate over informative features only (exclude quota-determined)
            keep = [j for j, nm in enumerate(names) if nm not in QUOTA_DETERMINED]
            overall = np.array([np.nanmean(R[i, keep]) if np.isfinite(R[i, keep]).any()
                                else np.nan for i in range(R.shape[0])])
        per_quota[q] = {
            "mean_corr": float(np.nanmean(overall)),
            "se": float(np.nanstd(overall, ddof=1) / np.sqrt(len(overall)))
            if len(overall) > 1 else float("nan"),
            "per_feature": {names[j]: float(per_feature[j]) for j in range(len(names))},
            "excluded_from_aggregate": list(QUOTA_DETERMINED),
            "n_classes_used": n_classes_used,
            "n_reps": len(reps),
        }

    recommended = None
    for q in sorted(k for k in per_quota if not per_quota[k].get("insufficient_data")):
        if per_quota[q]["mean_corr"] >= stable_threshold:
            recommended = q
            break

    return {"by_quota": per_quota, "feature_names": names,
            "stable_threshold": stable_threshold,
            "recommended_quota": recommended}
