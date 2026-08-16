"""Predicted Class Correction (PCC) — the Phase 2 method, `g_θ` and its threshold rule.

Unblocked by the Phase 1 gate; see `README.md` in this directory for the linked
pre-registration, the result table, and why this is built on the **head-weight**
descriptor family rather than the output-space one.

WHAT PCC CLAIMS, AND WHAT IT CANNOT
-----------------------------------
At `n_y = 0` there is no per-class sample to take a quantile of, so **no** method can
carry a finite-sample class-conditional coverage guarantee there. PCC therefore does
**not** claim classwise validity. It claims exactly two things:

1. **Marginal coverage is restored by construction** — a single scalar offset refit on
   a calibration slice (`recalibrate_marginal`). One degree of freedom, so the usual
   split-conformal argument still applies to the marginal rate.
2. **Empirical class-conditional equity improves at matched set size** — worst-class
   coverage, measured against the pegged targets in
   `reports/baseline_reproduction.md`.

Any wording stronger than that is false, and the docstrings here exist partly to keep
it out of the paper.

THREE THINGS THE GATE EVIDENCE FORCED INTO THE DESIGN
-----------------------------------------------------
- **Shrinkage is part of the method, not tuning.** Raw δ̂ at λ=1 *harms* worst-class
  equity (measured: −0.508 for head φ, −0.234 for output-space φ). λ is selected on
  TRAIN classes only, by `pcc.eval.setsize.select_shrinkage`.
- **Shrinkage applies only to the PREDICTED part.** Where δ_y is observed with enough
  samples it is already a direct estimate; shrinking it toward the global threshold
  would throw away information the data actually contains.
- **The data-threshold rule (§6.7) must be derived, not picked.** `data_threshold`
  crosses the sampling noise of the empirical class quantile against g_θ's own
  prediction error, both measured on TRAIN classes only.

HOW `n_star` SELECTION ARRIVED HERE (2026-08-13) — see README.md
----------------------------------------------------------------
Three flaws, found in order, each by measurement rather than review:

1. **Wrong currency.** `data_threshold` crossed mean squared errors while the objective is
   worst-class equity. Same mistake Amendment 8 records for λ. Kept as a reported
   secondary; it no longer decides.
2. **In-sample optimism.** `select_n_star` scored the observed correction on the very rows
   δ_obs came from, so a null world still lost 0.05–0.17 worst-class coverage on seen
   classes. `select_n_star_oos` scores both arms on held-out rows instead, and is the
   default.
3. **Degenerate candidates.** A candidate no class has the rows to support leaves both arms
   identical, so `>=` held and a meaningless `n_star` was reported (50, on a slice with ~42
   rows per class). Such candidates are now excluded and listed.

After all three: in a null world both tables are exactly +0.0000 — the machinery neither
invents structure nor damages what it cannot improve — and in a signal world Table 1 is
+0.43…+0.57 and Table 2 +0.31…+0.50.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pcc.eval.conformal import restrict_to_classes
from pcc.eval.predictability import ridge_fit, ridge_predict
from pcc.eval.setsize import (avg_set_size_at_shift, corrected_thresholds,
                              equity_at_matched_size, select_shrinkage)

# Below this many rows per class on the FIT slice, a per-class MINIMUM is noise rather
# than signal, so lambda is selected on a lower-tail quantile (p25) instead. Same
# threshold as reports/prereg_metrics_per_dataset.md uses for reporting, applied to
# selection -- see `fit_pcc` for why not applying it there was a real error.
SELECT_MIN_ROWS_PER_CLASS = 30

__all__ = [
    "SELECT_MIN_ROWS_PER_CLASS",
    "GTheta",
    "PCCModel",
    "fit_gtheta",
    "gtheta_cv_mse",
    "quantile_noise_at_n",
    "data_threshold",
    "select_n_star",
    "select_n_star_oos",
    "blend_delta",
    "recalibrate_marginal",
    "fit_pcc",
]


# --------------------------------------------------------------------------- g_θ
@dataclass
class GTheta:
    """φ(y) → δ̂_y. A ridge on standardized descriptors, fit on TRAIN classes only."""

    model: dict
    feature_names: tuple
    cols: tuple
    ridge_alpha: float
    n_train_classes: int

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        X = np.asarray(Phi, float)[:, list(self.cols)]
        out = np.full(len(X), np.nan)
        ok = np.isfinite(X).all(axis=1)
        if ok.any():
            out[ok] = ridge_predict(self.model, X[ok])
        return out


def fit_gtheta(Phi, delta_obs, train_classes, feature_names, *,
               features: Optional[Sequence[str]] = None,
               ridge_alpha: float = 1.0) -> GTheta:
    """Fit g_θ on TRAIN classes only.

    `train_classes` is the label space g_θ may see. Fitting on any class whose
    coverage is later reported would leak the answer into the predictor, which is the
    failure mode Amendments 4 and 8 exist to prevent.
    """
    names = list(feature_names)
    use = list(features) if features is not None else names
    missing = [f for f in use if f not in names]
    if missing:
        raise ValueError("feature(s) not in feature_names: " + repr(missing))
    cols = [names.index(f) for f in use]

    Phi = np.asarray(Phi, float)
    d = np.asarray(delta_obs, float)
    tr = np.asarray(train_classes, int)
    ok = np.isfinite(d[tr]) & np.isfinite(Phi[tr][:, cols]).all(axis=1)
    tr = tr[ok]
    if len(tr) < len(cols) + 2:
        raise ValueError(
            "too few usable TRAIN classes ({}) for {} features".format(len(tr), len(cols)))
    return GTheta(model=ridge_fit(Phi[tr][:, cols], d[tr], ridge_alpha),
                  feature_names=tuple(use), cols=tuple(cols),
                  ridge_alpha=float(ridge_alpha), n_train_classes=int(len(tr)))


def gtheta_cv_mse(Phi, delta_obs, train_classes, feature_names, *,
                  features: Optional[Sequence[str]] = None,
                  ridge_alpha: float = 1.0, n_folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold MSE of g_θ, measured **within TRAIN classes**.

    This is the "prediction error" side of the §6.7 rule. It must be out-of-fold: the
    in-sample residual would understate the error and push `n_star` down, biasing the
    rule toward the predictor.
    """
    tr = np.asarray(train_classes, int)
    d = np.asarray(delta_obs, float)
    tr = tr[np.isfinite(d[tr])]
    if len(tr) < max(2 * n_folds, 6):
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tr))
    folds = np.array_split(order, n_folds)
    se, n = 0.0, 0
    for f in folds:
        held = tr[f]
        fit = tr[np.setdiff1d(order, f, assume_unique=False)]
        if len(fit) < 4:
            continue
        try:
            g = fit_gtheta(Phi, d, fit, feature_names,
                           features=features, ridge_alpha=ridge_alpha)
        except ValueError:
            continue
        pred = g.predict(np.asarray(Phi, float)[held])
        m = np.isfinite(pred)
        se += float(np.sum((pred[m] - d[held][m]) ** 2))
        n += int(m.sum())
    return se / n if n else float("nan")


# ---------------------------------------------------- §6.7 data-threshold rule
def quantile_noise_at_n(score_matrix, labels, train_classes, alpha, q_global, n, *,
                        n_rep: int = 20, min_full: int = 40, seed: int = 0) -> float:
    """MSE of the empirical class correction δ_y estimated from only `n` samples.

    Estimated by subsampling within TRAIN classes that have enough data, and comparing
    the `n`-sample estimate against that class's own full-sample estimate. This is the
    "sampling noise" side of the §6.7 rule, and it is deliberately measured on the same
    scale as `gtheta_cv_mse` so the two can be crossed.
    """
    S = np.asarray(score_matrix)
    y = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    lo = max(1, 1.0 - float(alpha))
    se, cnt = 0.0, 0
    for c in np.asarray(train_classes, int):
        idx = np.where(y == c)[0]
        if len(idx) < max(min_full, n + 1):
            continue
        s_all = S[idx, c]
        full = float(np.quantile(s_all, lo)) - float(q_global)
        for _ in range(n_rep):
            sub = rng.choice(s_all, n, replace=False)
            est = float(np.quantile(sub, lo)) - float(q_global)
            se += (est - full) ** 2
            cnt += 1
    return se / cnt if cnt else float("nan")


def data_threshold(score_matrix, labels, train_classes, alpha, q_global, gtheta_mse, *,
                   candidates=(5, 10, 15, 20, 25, 30, 40, 50, 75, 100),
                   n_rep: int = 20, seed: int = 0) -> dict:
    """§6.7: the smallest `n` at which the OBSERVED δ_y beats the PREDICTED δ̂_y.

    Below `n_star` the empirical class quantile is noisier than g_θ's prediction, so
    the prediction is the better estimate even for a class that *has* a few samples.
    Above it, the data wins and should be used directly.

    Returns `n_star = None` when no candidate is good enough — meaning g_θ is more
    accurate than the empirical estimate at every tested `n`, and δ̂ should be used
    throughout. That is reported, not silently turned into a number.

    Everything here is computed on TRAIN classes only; no evaluation class is touched.
    """
    curve = {}
    for n in candidates:
        curve[int(n)] = quantile_noise_at_n(score_matrix, labels, train_classes, alpha,
                                            q_global, int(n), n_rep=n_rep, seed=seed)
    n_star = None
    if np.isfinite(gtheta_mse):
        for n in sorted(curve):
            v = curve[n]
            if np.isfinite(v) and v <= gtheta_mse:
                n_star = int(n)
                break
    return {"n_star": n_star, "gtheta_mse": float(gtheta_mse), "noise_curve": curve,
            "rule": "use observed delta_y when n_y >= n_star, else lambda * delta_hat"}


BIN_MIN_SCORE_ROWS = 200


def prevalence_bins(counts, classes, y_score, min_rows=BIN_MIN_SCORE_ROWS):
    """Classes sorted by prevalence, accumulated until each bin holds `min_rows` scoring
    rows. Same construction `reports/prereg_metrics_per_dataset.md` fixes for reporting.
    A short final bin merges backwards so no half-filled bin can dominate the minimum."""
    cls = np.asarray(classes, int)
    order = cls[np.argsort(np.asarray(counts, float)[cls], kind="stable")]
    per = np.bincount(np.asarray(y_score, int), minlength=int(cls.max()) + 1)
    bins, cur, n = [], [], 0
    for c in order:
        cur.append(int(c))
        n += int(per[c])
        if n >= min_rows:
            bins.append(cur)
            cur, n = [], 0
    if cur:
        (bins[-1].extend(cur) if bins else bins.append(cur))
    return bins


def select_shrinkage_binned(score_matrix, labels, n_classes, q_global, delta_hat,
                            target_size, bins, *,
                            lams=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)) -> dict:
    """λ chosen on POOLED prevalence bins instead of any per-class quantile.

    Why this and not p25. The 2026-08-15 Pl@ntNet curve was

        λ: 0.0 -> 0.5,  0.05 -> 0.0,  0.1 -> 0.0,  ...,  1.0 -> 0.0

    a cliff, not a slope. 0.5 and 0.0 are the signature of a class holding TWO
    calibration rows, where coverage can only be 0, 0.5 or 1: a handful of such classes
    decide the objective, and any perturbation at all drops one of them to zero. A
    lower-tail quantile does not escape that — the discreteness reaches through it, which
    is why the p25 downgrade failed too. Pooling rows into bins is the only move that
    removes the discreteness rather than stepping around it, and it is the construction
    the pre-registration already fixed for REPORTING in regime B.
    """
    from pcc.eval.setsize import shift_to_size

    S = np.asarray(score_matrix)
    y = np.asarray(labels, int)
    d = np.where(np.isfinite(np.asarray(delta_hat, float)), delta_hat, 0.0)
    base = np.full(int(n_classes), float(q_global))
    hit_rows = np.arange(len(y))
    masks = [np.isin(y, np.asarray(b, int)) for b in bins]
    masks = [m for m in masks if m.any()]
    if not masks:
        raise ValueError("no prevalence bin contains any scoring row")

    curve, best_lam, best_val = {}, 0.0, -np.inf
    for lam in lams:
        t = base + float(lam) * d
        t = t + shift_to_size(S, t, target_size)
        hit = S[hit_rows, y] <= t[y]
        v = float(min(hit[m].mean() for m in masks))
        curve[float(lam)] = v
        if v > best_val:
            best_lam, best_val = float(lam), v
    return {"lambda": best_lam, "value": best_val, "curve": curve, "stat": "bin_worst",
            "n_bins": len(masks),
            "rows_per_bin_min": int(min(int(m.sum()) for m in masks))}


def select_n_star(score_matrix, labels, train_classes, alpha, q_global,
                  delta_obs, delta_hat, n_per_class, lam, *,
                  candidates=(5, 10, 15, 20, 25, 30, 40, 50, 75, 100),
                  stat: str = "worst") -> dict:
    """§6.7 selected by the OBJECTIVE, on the TRAIN label space only.

    This replaces an MSE crossing that was the wrong currency. The objective is
    worst-class equity at matched set size, and a worst-class objective is governed by
    the LARGEST error, not the mean squared one — the same lesson Amendment 8 records
    for λ. Measured consequence of getting it wrong: in a null world λ correctly went to
    0 and held-out classes were untouched, yet SEEN classes lost 0.05–0.17 worst-class
    coverage, because by the MSE test their noisy observed δ_y really was the more
    *accurate* estimate — it just was not the better one for this objective.

    `n_star=None` is a genuine candidate and means "never prefer the observed value".
    Selection uses only TRAIN classes, so it cannot tune itself on reported classes.
    """
    S_tr, y_tr, K_tr, ids = restrict_to_classes(np.asarray(score_matrix),
                                                np.asarray(labels, int), train_classes)
    base = np.full(K_tr, float(q_global))
    target = avg_set_size_at_shift(S_tr, base, 0.0)
    curve, best_ns, best_val = {}, None, -np.inf
    for ns in [None] + [int(c) for c in candidates]:
        bl = blend_delta(delta_obs, delta_hat, n_per_class, ns, lam)
        e = equity_at_matched_size(S_tr, y_tr, K_tr, base + bl["delta"][ids], target)
        curve["none" if ns is None else str(ns)] = float(e[stat])
        if e[stat] > best_val:
            best_ns, best_val = ns, float(e[stat])
    return {"n_star": best_ns, "value": best_val, "curve": curve, "stat": stat,
            "selected_by": "objective (" + stat + ") on TRAIN label space"}


def select_n_star_oos(score_matrix, labels, train_classes, alpha, q_global,
                      delta_hat, lam, *, candidates=(5, 10, 20, 30, 50, 75),
                      n_rep: int = 3, stat: str = "worst", min_score_rows: int = 8,
                      seed: int = 0) -> dict:
    """§6.7 selected by the objective **out of sample**. This is the default rule.

    `select_n_star` scored the observed correction on the very rows δ_obs was estimated
    from, so it looked better than it is — measured: a null world stayed at −0.05…−0.17
    worst-class coverage on seen classes even after the currency was fixed, because the
    optimism only disappears on evaluation data.

    Here each candidate `n` is measured for what it actually claims: estimate δ_y from
    `n` rows of a class, then score coverage equity on that class's *remaining* rows.
    `n_star` is the smallest `n` whose observed correction beats simply predicting — and
    if none does, `None`, meaning the prediction is never worse and should always be used.

    Costs `len(candidates) * n_rep` matched-size searches on the TRAIN label space, which
    is why the default candidate grid is coarser than the in-sample rule's.
    """
    S = np.asarray(score_matrix)
    y = np.asarray(labels, int)
    tr = np.asarray(train_classes, int)
    d_hat = np.where(np.isfinite(np.asarray(delta_hat, float)), delta_hat, 0.0)
    rng = np.random.default_rng(seed)
    lo = 1.0 - float(alpha)

    by_class = {int(c): np.where(y == c)[0] for c in tr}
    min_est_classes = max(5, int(0.10 * len(tr)))
    curve_obs, curve_pred, skipped = {}, {}, {}

    for n in candidates:
        obs_vals, pred_vals, est_counts = [], [], []
        for _ in range(n_rep):
            est, sc = {}, []
            for c, idx in by_class.items():
                if len(idx) < n + min_score_rows:
                    sc.append(idx)              # too small to split: scoring only
                    continue
                p = rng.permutation(idx)
                est[c] = p[:n]
                sc.append(p[n:])
            est_counts.append(len(est))
            score_rows = np.concatenate(sc) if sc else np.array([], int)
            if len(score_rows) < min_score_rows or len(est) < min_est_classes:
                continue

            d_n = np.full(len(d_hat), np.nan)
            for c, rows in est.items():
                d_n[c] = float(np.quantile(S[rows, c], lo)) - float(q_global)

            S_sc, y_sc, K_sc, ids = restrict_to_classes(S[score_rows], y[score_rows], tr)
            base = np.full(K_sc, float(q_global))
            target = avg_set_size_at_shift(S_sc, base, 0.0)

            d_use = np.where(np.isfinite(d_n), d_n, lam * d_hat)
            obs_vals.append(equity_at_matched_size(S_sc, y_sc, K_sc,
                                                   base + d_use[ids], target)[stat])
            pred_vals.append(equity_at_matched_size(S_sc, y_sc, K_sc,
                                                    base + lam * d_hat[ids],
                                                    target)[stat])
        if obs_vals:
            curve_obs[str(int(n))] = float(np.mean(obs_vals))
            curve_pred[str(int(n))] = float(np.mean(pred_vals))
        else:
            # A candidate no class has the rows to support is NOT evidence that the
            # observed correction ties the prediction — with no estimate the two arms are
            # literally the same vector, so `>=` would hold degenerately and report a
            # meaningless n_star. Excluded and recorded instead.
            skipped[str(int(n))] = {
                "reason": "fewer than {} classes could spare {} estimation rows".format(
                    min_est_classes, int(n)),
                "max_classes_splittable": int(max(est_counts)) if est_counts else 0}

    n_star = None
    for k in sorted(curve_obs, key=lambda s: int(s)):
        if curve_obs[k] >= curve_pred[k]:
            n_star = int(k)
            break
    return {"n_star": n_star, "stat": stat, "n_rep": int(n_rep),
            "curve_observed": curve_obs, "curve_predicted_only": curve_pred,
            "candidates_not_evaluable": skipped,
            "min_est_classes_required": int(min_est_classes),
            "selected_by": "objective (" + stat + "), OUT OF SAMPLE, TRAIN label space",
            "value": curve_obs.get(str(n_star)) if n_star is not None else None}


def blend_delta(delta_obs, delta_hat, n_per_class, n_star, lam) -> dict:
    """Combine the observed and predicted corrections per the §6.7 rule.

    Shrinkage `lam` multiplies **only** the predicted part. Where δ_y is observed with
    `n_y >= n_star` it is used as-is: it is a direct estimate, and shrinking it toward
    the global threshold would discard information the data does contain.
    """
    d_obs = np.asarray(delta_obs, float)
    d_hat = np.asarray(delta_hat, float)
    n = np.asarray(n_per_class, int)
    if not (len(d_obs) == len(d_hat) == len(n)):
        raise ValueError("delta_obs, delta_hat and n_per_class must be the same length")

    use_obs = np.isfinite(d_obs)
    if n_star is None:
        use_obs[:] = False            # g_theta wins everywhere; say so, don't fake an n*
    else:
        use_obs &= n >= int(n_star)

    out = np.where(use_obs, d_obs, float(lam) * d_hat)
    out = np.where(np.isfinite(out), out, 0.0)   # no estimate at all -> global threshold
    return {"delta": out, "used_observed": use_obs,
            "n_observed": int(use_obs.sum()),
            "n_predicted": int(np.sum(~use_obs & np.isfinite(d_hat))),
            "n_fallback_global": int(np.sum(~use_obs & ~np.isfinite(d_hat)))}


# ------------------------------------------------------------------- validity
def recalibrate_marginal(score_matrix, labels, thresholds, alpha, *,
                         lo=-1.0, hi=1.0, iters: int = 60) -> float:
    """One scalar offset added to every threshold so MARGINAL coverage hits 1−α.

    This is the step that keeps PCC's marginal claim honest. It spends a single degree
    of freedom on a calibration slice, so it must be fit on data disjoint from the
    evaluation slice — the caller is responsible for that split, and
    `fit_pcc` documents which slice it used.
    """
    S = np.asarray(score_matrix)
    y = np.asarray(labels, int)
    t = np.asarray(thresholds, float)
    target = 1.0 - float(alpha)
    rows = np.arange(len(y))

    def cov(off):
        return float(np.mean(S[rows, y] <= t[y] + off))

    if cov(lo) > target:
        return float(lo)
    if cov(hi) < target:
        return float(hi)
    a, b = float(lo), float(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if cov(m) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# ----------------------------------------------------------------- the facade
@dataclass
class PCCModel:
    """A fitted PCC: everything needed to produce per-class thresholds, plus the
    provenance of every free parameter so a reader can check none of it saw the
    evaluation classes."""

    gtheta: GTheta
    q_global: float
    alpha: float
    lam: float
    n_star: Optional[int]
    offset: float
    threshold_rule: dict
    lambda_selection: dict
    blend: dict = field(repr=False, default_factory=dict)
    provenance: dict = field(repr=False, default_factory=dict)

    def thresholds(self) -> np.ndarray:
        return corrected_thresholds(self.q_global, self.blend["delta"]) + self.offset

    def prediction_sets(self, score_matrix) -> np.ndarray:
        return np.asarray(score_matrix) <= self.thresholds()[None, :]


def fit_pcc(Phi, feature_names, delta_obs, n_per_class, q_global, alpha, *,
            score_matrix_fit, labels_fit, train_classes,
            features: Optional[Sequence[str]] = None,
            ridge_alpha: float = 1.0, target_size: Optional[float] = None,
            stat: str = "worst", n_folds: int = 5, n_star_rule: str = "oos",
            class_counts: Optional[np.ndarray] = None,
            lam_override: Optional[float] = None,
            recalibrate: bool = True, seed: int = 0) -> PCCModel:
    """Fit the whole method on the FIT slice and the TRAIN label space only.

    `score_matrix_fit` / `labels_fit` must be disjoint from whatever slice the results
    are later reported on. Every free parameter — g_θ's weights, λ, `n_star`, and the
    marginal offset — is chosen here, and `provenance` records that so the claim can be
    audited instead of trusted.
    """
    n_classes = len(np.asarray(delta_obs, float))
    tr = np.asarray(train_classes, int)

    g = fit_gtheta(Phi, delta_obs, tr, feature_names,
                   features=features, ridge_alpha=ridge_alpha)
    d_hat = g.predict(np.asarray(Phi, float))

    # The MSE crossing is still computed and reported — it is informative about where the
    # empirical estimate becomes accurate — but it does NOT choose n_star. See
    # `select_n_star` for why that currency was wrong.
    mse = gtheta_cv_mse(Phi, delta_obs, tr, feature_names, features=features,
                        ridge_alpha=ridge_alpha, n_folds=n_folds, seed=seed)
    mse_rule = data_threshold(score_matrix_fit, labels_fit, tr, alpha, q_global, mse,
                              seed=seed)

    # λ is selected on the TRAIN LABEL SPACE ONLY. Selecting it over all classes would
    # let a free parameter tune itself on the very classes the result is read from —
    # the failure mode Amendment 8 exists to prevent. `restrict_to_classes` reindexes
    # the score matrix to the train classes, exactly as `setsize_translation_shrunk`
    # does, so the two agree by construction rather than by comment.
    d_safe = np.where(np.isfinite(d_hat), d_hat, 0.0)
    S_tr, y_tr, K_tr, ids_tr = restrict_to_classes(
        np.asarray(score_matrix_fit), np.asarray(labels_fit, int), tr)
    if target_size is None:
        target_size = avg_set_size_at_shift(S_tr, np.full(K_tr, float(q_global)), 0.0)

    # The SELECTION statistic must be measurable on the slice it is selected on. This was
    # a real design error, found on the 2026-08-13 long-tail run: lambda was chosen by
    # worst-class coverage on the CAL slice, where Pl@ntNet has a handful of rows per
    # class and iNat about two. A minimum over hundreds of classes whose coverage can only
    # be 0 or 1 is unmovable noise, so NO lambda could ever improve it and lambda
    # collapsed to 0 on both datasets -- which meant delta_hat never entered the
    # thresholds at all and the head family was never actually exercised there.
    # `prereg_metrics_per_dataset.md` already fixed this rule for REPORTING; it applies
    # with equal force to SELECTION, and not applying it there was the oversight.
    rows_per_train_class = np.bincount(y_tr, minlength=K_tr)
    med_fit = float(np.median(rows_per_train_class)) if K_tr else 0.0
    thin = med_fit < SELECT_MIN_ROWS_PER_CLASS
    sel_stat = stat if not thin else "bin_worst"
    if thin and class_counts is not None:
        # Regime B: pool into prevalence bins. A per-class quantile — even p25 — cannot
        # escape the discreteness of 2-row classes; see `select_shrinkage_binned`.
        bins_tr = prevalence_bins(np.asarray(class_counts, float)[ids_tr],
                                  np.arange(K_tr), y_tr)
        lam_sel = select_shrinkage_binned(S_tr, y_tr, K_tr, q_global, d_safe[ids_tr],
                                          float(target_size), bins_tr)
    else:
        if thin:
            sel_stat = "p25"      # no prevalence counts supplied; fall back, and say so
        lam_sel = select_shrinkage(S_tr, y_tr, K_tr, q_global, d_safe[ids_tr],
                                   float(target_size), stat=sel_stat)
    lam_sel["selection_stat"] = sel_stat
    lam_sel["median_fit_rows_per_train_class"] = med_fit
    lam_sel["selection_stat_downgraded"] = bool(sel_stat != stat)
    # WHICH SLICE MAY lambda BE CHOSEN ON? nb05's Sec 6.4 scored it on EVAL rows of the
    # train classes, fit_pcc scores it on CAL, and the two disagreed (+0.1173 vs 0).
    # Only CAL is DEPLOYABLE -- at deployment there are no evaluation labels -- so CAL is
    # the answer and nb05's figure was a measurement convenience, not a deployable one.
    # lambda multiplies delta_hat ONLY, so it carries no in-sample optimism with respect
    # to delta_obs; its residual optimism is that delta_hat is in-sample for g_theta,
    # which row splitting cannot fix and `gtheta_cv_mse` reports instead.
    lam_sel["scored_on"] = "CAL rows; lambda touches delta_hat only"

    # WHY lambda = 0? Two very different answers, and until 2026-08-15 the output could
    # not tell them apart -- three long-tail runs reported lambda = 0 with no way to know
    # which had happened:
    #
    #   curve DECREASING -> delta_hat carries signal but HURTS. A real finding about phi.
    #   curve FLAT       -> delta_hat is degenerate (near-constant across classes), so
    #                       after size matching it shifts nothing and every lambda scores
    #                       identically. select_shrinkage then returns the first candidate,
    #                       which is 0.0, because the comparison is a strict `>`. That is
    #                       an estimation limit, NOT evidence about phi.
    #
    # A near-constant delta_hat is exactly what to expect when g_theta can only be fitted
    # on the head classes that reach n_cal, and is then extrapolated to a long tail far
    # outside that range. So the distinction is diagnosed here rather than left to be
    # guessed from a lost log.
    _cv = np.array(list(lam_sel["curve"].values()), float)
    _rng_curve = float(np.nanmax(_cv) - np.nanmin(_cv)) if _cv.size else float("nan")
    _dh_tr = d_hat[tr][np.isfinite(d_hat[tr])]
    lam_sel["curve_range"] = _rng_curve
    lam_sel["sd_delta_hat_train"] = float(np.std(_dh_tr)) if _dh_tr.size else float("nan")
    lam_sel["sd_delta_obs_train"] = float(
        np.nanstd(np.asarray(delta_obs, float)[tr])) if len(tr) else float("nan")
    if lam_sel["lambda"] == 0.0:
        flat = np.isfinite(_rng_curve) and _rng_curve < 1e-9
        lam_sel["zero_lambda_reason"] = (
            "DEGENERATE delta_hat: the lambda curve is flat, so no lambda changes the "
            "objective and 0.0 is returned by default. This is an estimation limit, not "
            "evidence that phi carries no signal."
            if flat else
            "delta_hat carries signal but every positive lambda scored WORSE than 0 on "
            "the train label space. This is evidence about phi, not a degeneracy.")
    else:
        lam_sel["zero_lambda_reason"] = None

    if n_star_rule == "oos":
        # `sel_stat`, not `stat`: the thin-slice downgrade reached lambda but not n_star,
        # so on a long-tail dump the two interdependent parameters were being chosen by
        # two DIFFERENT objectives. Found by reading the 2026-08-15 curve, which reported
        # stat='worst' for n_star while lambda had already been downgraded to p25.
        # `bin_worst` has no EQUITY_STATS entry, so n_star cannot yet be scored on bins.
        # It falls back to the lower-tail per-class quantile, and the mismatch is RECORDED
        # rather than hidden -- lambda and n_star being chosen on different objectives is
        # exactly the bug found on 2026-08-15, and it must stay visible until bin-level
        # n_star exists.
        ns_stat = "p25" if sel_stat == "bin_worst" else sel_stat
        ns_sel = select_n_star_oos(score_matrix_fit, labels_fit, tr, alpha, q_global,
                                   d_hat, lam_sel["lambda"], stat=ns_stat, seed=seed)
        ns_sel["stat_differs_from_lambda"] = bool(ns_stat != sel_stat)
        if ns_stat != sel_stat:
            ns_sel["mismatch_note"] = (
                "lambda was selected on " + sel_stat + " but n_star on " + ns_stat +
                "; bin-level n_star is not implemented yet.")
    elif n_star_rule == "objective":
        ns_sel = select_n_star(score_matrix_fit, labels_fit, tr, alpha, q_global,
                               delta_obs, d_hat, n_per_class, lam_sel["lambda"],
                               stat=stat)
    elif n_star_rule == "mse":
        ns_sel = {"n_star": mse_rule["n_star"], "selected_by": "mse crossing (legacy)",
                  "curve": {}, "stat": stat, "value": float("nan")}
    else:
        raise ValueError("n_star_rule must be 'oos', 'objective' or 'mse'")

    # ABLATION hook. lambda = 0 removes the correction entirely (the method reduces to
    # the global threshold on held-out classes); lambda = 1 is the raw, unshrunk delta_hat
    # that Amendment 8 measured as HARMFUL. Both are needed to show shrinkage is part of
    # the method rather than a tuning detail, and the override is recorded so a report can
    # never be mistaken for a normal run.
    if lam_override is not None:
        lam_sel = dict(lam_sel, lambda_selected=lam_sel["lambda"],
                       ablation_override=float(lam_override))
        lam_sel["lambda"] = float(lam_override)

    blend = blend_delta(delta_obs, d_hat, n_per_class, ns_sel["n_star"],
                        lam_sel["lambda"])

    # The marginal offset is also fit on TRAIN-class rows only, and that is not a
    # convenience: in deployment a class with n_y = 0 contributes no calibration rows
    # at all, so those rows do not exist to calibrate on. The consequence must be
    # stated in the paper rather than glossed — the marginal guarantee is over the
    # SEEN-class distribution, not the full population.
    offset = 0.0
    if recalibrate:
        t_all = corrected_thresholds(q_global, blend["delta"])
        offset = recalibrate_marginal(S_tr, y_tr, t_all[ids_tr], alpha)

    return PCCModel(
        gtheta=g, q_global=float(q_global), alpha=float(alpha),
        lam=float(lam_sel["lambda"]), n_star=ns_sel["n_star"], offset=float(offset),
        threshold_rule={"selected": ns_sel, "mse_crossing_secondary": mse_rule},
        lambda_selection=lam_sel, blend=blend,
        provenance={
            "n_star_rule": n_star_rule,
            "n_star_selected_on": "TRAIN label space only",
            "fit_slice_rows": int(len(np.asarray(labels_fit))),
            "train_classes": int(len(tr)),
            "n_classes": int(n_classes),
            "target_size_for_lambda": float(target_size),
            "everything_fit_on": "FIT slice + TRAIN label space only",
            "lambda_selected_on": "TRAIN label space only (restrict_to_classes)",
            "lambda_selection_stat": sel_stat,
            "lambda_selection_stat_downgraded": bool(sel_stat != stat),
            "marginal_offset_recalibrated": bool(recalibrate),
            "marginal_offset_fit_on": "TRAIN-class rows only",
            "claims": ("marginal coverage over the SEEN-class distribution by "
                       "construction; empirical class equity at matched size. NOT "
                       "classwise validity — impossible at n_y=0."),
        })
