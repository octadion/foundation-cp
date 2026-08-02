"""Gate B/C (AGENTS.md §6.3, §6.5): is δ_y predictable from class geometry, and
does geometry beat the trivial predictors?

The predictor here is a SIMPLE, FIXED ridge — the gate's *measurement
instrument*, not the Phase-2 g_θ. Architecture search on g_θ is forbidden before
the gate passes (§10); this module only measures whether signal exists.

Splits are at the CLASS level (𝒴_train / 𝒴_held-out), never sample level
(§8.2) — the caller supplies classes via pcc.eval.leakguard.class_level_split.
Held-out R² is reported NORMALIZED by the reliability ceiling from gate A (§6.3):
raw R² without the ceiling is an analysis error.

Gate B: held-out normalized R² clearly > 0, CI excludes 0.
Gate C: the full descriptor beats log-prevalence-only and nearest-class-distance-
only (Fargion-style) predictors — if log-prevalence alone explains most of δ_y,
the space is already occupied (§6.5C). This is the criterion most likely to fail;
run it early.
"""

from __future__ import annotations

import numpy as np


def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def ridge_fit(X, y, lam=1.0):
    """Closed-form ridge with intercept, on standardized features."""
    mu, sd = _standardize_fit(X)
    Xs = (X - mu) / sd
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    d = Xb.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0  # don't penalize intercept
    w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return {"w": w, "mu": mu, "sd": sd}


def ridge_predict(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    return Xb @ model["w"]


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1 - ss_res / ss_tot)


def _percentile_ci(arr, alpha=0.05):
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "ci_low": float(np.percentile(a, 100 * alpha / 2)),
            "ci_high": float(np.percentile(a, 100 * (1 - alpha / 2))),
            "n": int(len(a))}


def predictability(Phi, delta, feature_names, *, reliability=None,
                   n_splits=100, frac_train=0.5, lam=1.0, seed=42,
                   prevalence_col="log_prevalence", distance_col="cos_knn_1"):
    """Class-level split predictability of δ_y with gate B/C summaries.

    Only classes with a finite δ_y and finite descriptor row are used. For each
    split: partition those classes 𝒴_train/𝒴_held-out, fit ridge on 𝒴_train,
    evaluate held-out R² for the FULL descriptor and for the ablation predictors
    (log-prevalence only, nearest-class-distance only, both). Differences are
    paired within split.
    """
    Phi = np.asarray(Phi, float)
    delta = np.asarray(delta, float)
    valid = np.where(np.isfinite(delta) & np.isfinite(Phi).all(axis=1))[0]
    if len(valid) < 10:
        raise ValueError(f"too few usable classes ({len(valid)}) for predictability")

    names = list(feature_names)
    col = {n: i for i, n in enumerate(names)}
    prev_i = col.get(prevalence_col)
    dist_i = col.get(distance_col)
    ablations = {}
    if prev_i is not None:
        ablations["log_prevalence_only"] = [prev_i]
    if dist_i is not None:
        ablations["distance_only"] = [dist_i]
    if prev_i is not None and dist_i is not None:
        ablations["prevalence+distance"] = [prev_i, dist_i]

    rng = np.random.default_rng(seed)
    predictors = {"full": None, **ablations}
    r2 = {k: [] for k in predictors}

    for _ in range(n_splits):
        perm = rng.permutation(valid)
        cut = int(round(frac_train * len(perm)))
        tr, te = perm[:cut], perm[cut:]
        if len(te) < 2:
            continue
        for name, cols in predictors.items():
            Xtr = Phi[tr] if cols is None else Phi[tr][:, cols]
            Xte = Phi[te] if cols is None else Phi[te][:, cols]
            model = ridge_fit(Xtr, delta[tr], lam)
            pred = ridge_predict(model, Xte)
            r2[name].append(r2_score(delta[te], pred))

    r2 = {k: np.array(v, float) for k, v in r2.items()}
    summ = {k: _percentile_ci(v) for k, v in r2.items()}

    # normalized held-out R² (§6.3): divide by reliability ceiling
    norm = None
    if reliability is not None and reliability > 0:
        norm_arr = r2["full"] / reliability
        norm = _percentile_ci(norm_arr)

    # gate B: (normalized) full R² CI excludes 0
    ref = norm if norm is not None else summ["full"]
    gate_B = bool(ref["ci_low"] > 0)

    # gate C: full beats each simple predictor (paired diff CI excludes 0)
    gate_C_detail = {}
    gate_C = True
    for name in ablations:
        diff = r2["full"] - r2[name]
        d = _percentile_ci(diff)
        beats = bool(d["ci_low"] > 0)
        gate_C_detail[name] = {"paired_diff": d, "full_beats_it": beats}
        gate_C = gate_C and beats
    if not ablations:
        gate_C = None

    return {
        "n_classes_used": int(len(valid)),
        "r2_by_predictor": summ,
        "reliability_ceiling": reliability,
        "normalized_full_r2": norm,
        "gate_B_pass": gate_B,
        "gate_C_pass": gate_C,
        "gate_C_detail": gate_C_detail,
    }
