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


def _bootstrap_pair(classes, n_classes, q, rng):
    """Two INDEPENDENT bootstrap resamples of q images per class (with replacement).

    Needed because disjoint halves require 2q images per class, which the rare
    strata of a long-tailed dataset simply do not have — measured on a Pl@ntNet-like
    simulation, the two rarest quartiles were unmeasurable even at q=5. Without an
    estimator that works at q <= n_y, the head-vs-tail descriptor-quality gap (the
    gate-C contamination risk) cannot be quantified at all.

    CAVEAT, and it matters: the two resamples SHARE samples, so this correlation is
    biased UPWARD relative to disjoint halves. Bootstrap numbers are therefore NOT
    comparable to disjoint-half numbers. They ARE comparable ACROSS STRATA as long
    as the same method and the same q are used everywhere — which is exactly what
    the head/tail comparison needs.
    """
    a, b = [], []
    for y in range(n_classes):
        idx = np.where(classes == y)[0]
        if len(idx) == 0:
            continue
        k = min(q, len(idx))
        a.extend(rng.choice(idx, k, replace=True).tolist())
        b.extend(rng.choice(idx, k, replace=True).tolist())
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


def descriptor_stability_by_stratum(features, logits, classes, n_classes, counts, *,
                                    quotas=(10, 25, 50), n_reps=3, seed=42,
                                    stable_threshold=0.9, n_strata=4,
                                    method="bootstrap"):
    """Descriptor stability computed SEPARATELY per prevalence stratum.

    THE RISK THIS GUARDS (descriptor_stability_findings.md, "Open issue 2"). On a
    long-tailed dataset, head classes can supply many images per class and tail
    classes cannot. If descriptors are noisier for rare classes, then **descriptor
    quality correlates with prevalence** — and §6.3 gate C asks whether geometry
    beats log-prevalence. Geometry would then look predictive exactly where
    prevalence is high, i.e. gate C could PASS for a spurious reason. §3.3 states
    the rule directly: never use descriptors of differing quality between head and
    tail without reporting it.

    A stratum whose classes cannot supply `2*q` images is reported as
    `insufficient_data` rather than silently dropped — that absence IS the finding.

    Returns {stratum_name: {quota: {...}}} plus `spread`: the head-minus-tail gap in
    mean stability at each quota. A large positive spread is the contamination
    warning; it must be reported alongside any gate-C conclusion.
    """
    from pcc.eval.tail import prevalence_strata

    strata, unevaluable = prevalence_strata(counts, n_strata, min_count=1)
    out = {}
    for name, cls in strata.items():
        sub = np.isin(classes, cls)
        if sub.sum() == 0:
            out[name] = {"insufficient_data": True, "n_classes": int(len(cls))}
            continue
        res = descriptor_stability(
            features[sub], None if logits is None else logits[sub], classes[sub],
            n_classes, quotas=quotas, n_reps=n_reps, seed=seed,
            stable_threshold=stable_threshold, method=method)
        res["n_classes_in_stratum"] = int(len(cls))
        out[name] = res

    names = [k for k in out if not out[k].get("insufficient_data")]
    spread = {}
    if len(names) >= 2:
        tail, head = names[0], names[-1]
        for q in quotas:
            t = out[tail]["by_quota"].get(q, {})
            h = out[head]["by_quota"].get(q, {})
            if not t.get("insufficient_data") and not h.get("insufficient_data"):
                spread[q] = {"tail": t["mean_corr"], "head": h["mean_corr"],
                             "head_minus_tail": h["mean_corr"] - t["mean_corr"]}
            else:
                spread[q] = {"tail": None if t.get("insufficient_data") else t.get("mean_corr"),
                             "head": None if h.get("insufficient_data") else h.get("mean_corr"),
                             "head_minus_tail": None,
                             "note": "a stratum could not supply 2*q images per class"}
    return {"by_stratum": out, "spread": spread, "method": method,
            "unevaluable_classes": int(len(unevaluable))}


def descriptor_stability(features, logits, classes, n_classes, *,
                         quotas=(10, 25, 50, 100), n_reps=5, seed=42,
                         stable_threshold=0.9, method="disjoint"):
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
            pair = _disjoint_halves if method == "disjoint" else _bootstrap_pair
            ia, ib = pair(classes, n_classes, q, rng)
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
            "stable_threshold": stable_threshold, "method": method,
            "recommended_quota": recommended}
