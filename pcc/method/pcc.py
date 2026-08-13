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

OPEN FLAW IN `n_star` SELECTION (2026-08-13) — see README.md
------------------------------------------------------------
The wrong-currency flaw is fixed: `select_n_star` now chooses by the objective, not by an
MSE crossing. A second flaw remains. δ_obs is estimated on the CAL slice and `n_star` is
selected on that **same** slice, so the observed correction is scored on its own training
data and looks better than it is; the optimism disappears on EVAL. Measured: in a null
world Table 1 stays at −0.05…−0.17 worst-class coverage. The fix is out-of-sample
selection within CAL (estimate δ_obs on one half, score on the other, swap, average),
which is a change to the method and so gets its own measurement.

**Until then, Table 1 (seen-class) numbers must not be read as the method's performance.**
Table 2 (`n_y = 0`) is unaffected: no observed δ_y exists there, so `n_star` never fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pcc.eval.conformal import restrict_to_classes
from pcc.eval.predictability import ridge_fit, ridge_predict
from pcc.eval.setsize import (avg_set_size_at_shift, corrected_thresholds,
                              equity_at_matched_size, select_shrinkage)

__all__ = [
    "GTheta",
    "PCCModel",
    "fit_gtheta",
    "gtheta_cv_mse",
    "quantile_noise_at_n",
    "data_threshold",
    "select_n_star",
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
            stat: str = "worst", n_folds: int = 5, n_star_rule: str = "objective",
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
    lam_sel = select_shrinkage(S_tr, y_tr, K_tr, q_global, d_safe[ids_tr],
                               float(target_size), stat=stat)

    if n_star_rule == "objective":
        ns_sel = select_n_star(score_matrix_fit, labels_fit, tr, alpha, q_global,
                               delta_obs, d_hat, n_per_class, lam_sel["lambda"],
                               stat=stat)
    elif n_star_rule == "mse":
        ns_sel = {"n_star": mse_rule["n_star"], "selected_by": "mse crossing (legacy)",
                  "curve": {}, "stat": stat, "value": float("nan")}
    else:
        raise ValueError("n_star_rule must be 'objective' or 'mse'")

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
            "marginal_offset_recalibrated": bool(recalibrate),
            "marginal_offset_fit_on": "TRAIN-class rows only",
            "claims": ("marginal coverage over the SEEN-class distribution by "
                       "construction; empirical class equity at matched size. NOT "
                       "classwise validity — impossible at n_y=0."),
        })
