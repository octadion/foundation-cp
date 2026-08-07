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


def predictability_by_stratum(Phi, delta, feature_names, counts, *, reliability=None,
                              n_splits=100, frac_train=0.5, lam=1.0, seed=42,
                              n_strata=4, min_classes=20,
                              reliability_fn=None, **kw):
    """Gate B/C computed WITHIN prevalence strata — the PRIMARY form of gate C
    (Amendment 5, reports/protocol_amendments.md).

    THE CONFOUND THIS BREAKS (measured on Pl@ntNet, reports/
    descriptor_stability_findings.md): descriptor stability rises monotonically with
    prevalence — 0.684 in the rarest quartile versus 0.922 in the densest, a spread
    of **0.238** that no quota choice can remove, because the rare classes only hold
    2–7 images. So geometry descriptors are *most accurate exactly where prevalence
    is highest*. Pooled across all classes, "geometry beats log-prevalence" is
    therefore confounded: geometry could look predictive because of
    prevalence-linked descriptor quality rather than despite prevalence.

    The standard remedy for a confound is to CONDITION on it. Within one prevalence
    quartile, prevalence barely varies (so it cannot explain much) and descriptor
    quality is roughly uniform (so the quality gradient is held fixed). Geometry that
    still predicts δ_y inside strata is not a prevalence artefact.

    `reliability_fn(class_ids) -> float` supplies a PER-STRATUM reliability ceiling.
    Pass it whenever normalized R² is compared across strata. Reliability is a variance
    ratio, so a stratum whose δ_y barely varies has a much lower ceiling than the pooled
    one — normalizing every stratum by the pooled ceiling therefore understates exactly the
    strata where the target is most compressed. `sd_delta` is returned alongside as the
    direct diagnostic for that compression.

    Returns {stratum: predictability(...)} plus `summary` giving, per stratum, the
    held-out R² and whether the full descriptor beat each ablation. Strata are never
    pooled; a stratum with fewer than `min_classes` usable classes is reported as
    skipped rather than silently merged.
    """
    from pcc.eval.tail import prevalence_strata

    # Stratify over the classes that ACTUALLY HAVE a delta_y, not over every class.
    # Measured on Pl@ntNet: matched n_cal=25 leaves ~152 classes, all necessarily
    # head classes, so stratifying the full 1081 put 3 of 4 strata at zero usable
    # classes and the stratification could not do its job. The retained classes still
    # span 25..616 calibration samples, and descriptor quality varies across that
    # range too, so stratifying WITHIN them tests the same confound.
    delta_arr = np.asarray(delta, float)
    have_delta = np.isfinite(delta_arr) & np.isfinite(np.asarray(Phi, float)).all(axis=1)
    counts_masked = np.where(have_delta, np.asarray(counts), 0)
    strata, unevaluable = prevalence_strata(counts_masked, n_strata, min_count=1)
    out, summary = {}, {}
    for name, cls in strata.items():
        sub = np.asarray(cls, int)
        d_sub = np.full(len(delta), np.nan)
        d_sub[sub] = np.asarray(delta, float)[sub]
        usable = int(np.sum(np.isfinite(d_sub) & np.isfinite(Phi).all(axis=1)))
        if usable < min_classes:
            out[name] = {"skipped": True, "n_usable_classes": usable,
                         "reason": f"fewer than {min_classes} usable classes"}
            continue
        rel_s = reliability
        if reliability_fn is not None:
            try:
                rel_s = reliability_fn(sub)
            except Exception:
                rel_s = reliability
        try:
            res = predictability(Phi, d_sub, feature_names, reliability=rel_s,
                                 n_splits=n_splits, frac_train=frac_train, lam=lam,
                                 seed=seed, **kw)
        except ValueError as e:
            out[name] = {"skipped": True, "n_usable_classes": usable, "reason": str(e)}
            continue
        res["n_classes_in_stratum"] = int(len(sub))
        res["prevalence_range"] = [int(counts[sub].min()), int(counts[sub].max())]
        fin = d_sub[np.isfinite(d_sub)]
        res["sd_delta"] = float(fin.std()) if len(fin) > 1 else float("nan")
        res["reliability_used"] = (float(rel_s) if rel_s is not None
                                   else float("nan"))
        out[name] = res
        summary[name] = {
            "n_classes": res["n_classes_used"],
            "prevalence_range": res["prevalence_range"],
            "r2_full": res["r2_by_predictor"]["full"]["mean"],
            "r2_full_ci": [res["r2_by_predictor"]["full"]["ci_low"],
                           res["r2_by_predictor"]["full"]["ci_high"]],
            "gate_B_pass": res["gate_B_pass"],
            "sd_delta": res["sd_delta"],
            "reliability_used": res["reliability_used"],
            # NaN, not None: a stratum whose ceiling comes back non-finite yields no
            # normalized R2, and callers build float arrays from these fields --
            # np.array([None], float) raises, np.array([nan], float) does not.
            "r2_normalized": (res["normalized_full_r2"]["mean"]
                              if res["normalized_full_r2"] else float("nan")),
            "r2_normalized_ci": ([res["normalized_full_r2"]["ci_low"],
                                  res["normalized_full_r2"]["ci_high"]]
                                 if res["normalized_full_r2"] else
                                 [float("nan"), float("nan")]),
            "underpowered": res["underpowered"],
            "gate_C_pass": res["gate_C_pass"],
            "gate_C_pass_prereg": res["gate_C_pass_prereg"],
            "distance_baseline_is_nested_in_full": res["distance_baseline_is_nested_in_full"],
            "n_train_classes": res["n_train_classes"],
            "n_features_full": res["n_features_full"],
            "beats": {k: v["full_beats_it"] for k, v in res["gate_C_detail"].items()},
        }
    n_pass_B = sum(1 for v in summary.values() if v["gate_B_pass"])
    ablations = sorted({a for v in summary.values() for a in v["beats"]})
    n_beats = {a: sum(1 for v in summary.values() if v["beats"].get(a)) for a in ablations}
    return {"by_stratum": out, "summary": summary,
            "n_strata_reported": len(summary), "n_strata_gate_B_pass": n_pass_B,
            "n_strata_full_beats_ablation": n_beats,
            "unevaluable_classes": int(len(unevaluable))}


def predictability(Phi, delta, feature_names, *, reliability=None,
                   n_splits=100, frac_train=0.5, lam=1.0, seed=42,
                   feature_subset=None,
                   prevalence_col="log_prevalence",
                   distance_col=("cos_knn_5", "cos_knn_1", "cos_knn_10")):
    """Class-level split predictability of δ_y with gate B/C summaries.

    Only classes with a finite δ_y and finite descriptor row are used. For each
    split: partition those classes 𝒴_train/𝒴_held-out, fit ridge on 𝒴_train,
    evaluate held-out R² for the FULL descriptor and for the ablation predictors
    (log-prevalence only, nearest-class-distance only, both). Differences are
    paired within split.

    `feature_subset` (names) selects the columns of the "full" model. **Always pass
    the COMPLETE Phi and feature_names and restrict via this argument**, never a
    pre-sliced Phi. Gate C's ablations are BASELINES, not features of the full model:
    slicing Phi down to the stable set ['cos_knn_5', 'logit_margin'] removed both
    `log_prevalence` and the distance column, so `ablations` came back empty and
    **gate C silently returned None for the primary feature set** -- the gate most
    likely to fail was never run (observed on Pl@ntNet, 2026-08-05).

    `distance_col` is a tuple of CANDIDATES, first present wins, so the distance
    baseline survives a stability screen that drops `cos_knn_1`.

    Also returns `n_train_classes`, `n_features_full` and `underpowered`. A ridge with
    p features fit on n_train classes cannot be compared fairly against a 1-2 feature
    baseline when n_train is a small multiple of p: on Pl@ntNet the 15-feature `full`
    model had 19 training classes per stratum and scored NEGATIVE held-out R² while the
    2-feature stable set reached +0.45. `underpowered` marks that regime so a FAIL
    there is never read as evidence of no signal.
    """
    Phi = np.asarray(Phi, float)
    delta = np.asarray(delta, float)
    names = list(feature_names)
    col = {n: i for i, n in enumerate(names)}

    if feature_subset is None:
        full_cols = list(range(Phi.shape[1]))
    else:
        missing = [n for n in feature_subset if n not in col]
        if missing:
            raise ValueError(f"feature_subset names not in feature_names: {missing}")
        full_cols = [col[n] for n in feature_subset]
    if not full_cols:
        raise ValueError("feature_subset selected no columns")

    prev_i = col.get(prevalence_col)
    cands = [n for n in np.atleast_1d(distance_col) if n in col]
    dist_name = cands[0] if cands else None
    dist_i = col.get(dist_name) if dist_name is not None else None
    ablations = {}
    if prev_i is not None:
        ablations["log_prevalence_only"] = [prev_i]
    if dist_i is not None:
        ablations["distance_only"] = [dist_i]
    if prev_i is not None and dist_i is not None:
        ablations["prevalence+distance"] = [prev_i, dist_i]

    # NESTING WARNING, and a second distance baseline because of it.
    # The stability screen makes the preferred distance baseline `cos_knn_5`, which on
    # Pl@ntNet is ALSO one of the two features in the PRIMARY feature set. The ablation
    # is then a nested SUBMODEL of the full model, so "full beats distance_only" tests
    # "does logit_margin add anything beyond cos_knn_5" -- strictly HARDER than the
    # pre-registered §6.5C question ("does our geometry beat a Fargion-style
    # nearest-class-distance baseline"). A FAIL under a harder-than-registered criterion
    # is not the same finding as a FAIL under the registered one, so the pre-registered
    # baseline is computed alongside and both verdicts are returned.
    nested = bool(dist_i is not None and dist_i in full_cols)
    prereg_name = None
    PREREG = ("distance_only_prereg", "prevalence+distance_prereg")
    if nested:
        # Only build the parallel baseline when the primary one is actually nested;
        # otherwise the two questions coincide and a duplicate would just confuse.
        prereg_name = next((n for n in np.atleast_1d(distance_col)[1:]
                            if n in col and col[n] not in full_cols), None)
        if prereg_name is not None:
            ablations["distance_only_prereg"] = [col[prereg_name]]
            if prev_i is not None:
                ablations["prevalence+distance_prereg"] = [prev_i, col[prereg_name]]

    # A class must have finite values in every column any predictor will touch --
    # including the ablation columns, which may sit outside `feature_subset`.
    used = sorted(set(full_cols) | {i for c in ablations.values() for i in c})
    valid = np.where(np.isfinite(delta) & np.isfinite(Phi[:, used]).all(axis=1))[0]
    if len(valid) < 10:
        raise ValueError(f"too few usable classes ({len(valid)}) for predictability")

    rng = np.random.default_rng(seed)
    predictors = {"full": full_cols, **ablations}
    r2 = {k: [] for k in predictors}
    n_tr = 0

    for _ in range(n_splits):
        perm = rng.permutation(valid)
        cut = int(round(frac_train * len(perm)))
        tr, te = perm[:cut], perm[cut:]
        if len(te) < 2:
            continue
        n_tr = len(tr)
        for name, cols in predictors.items():
            Xtr = Phi[tr][:, cols]
            Xte = Phi[te][:, cols]
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
    gate_C_prereg = True
    for name in ablations:
        diff = r2["full"] - r2[name]
        d = _percentile_ci(diff)
        beats = bool(d["ci_low"] > 0)
        gate_C_detail[name] = {"paired_diff": d, "full_beats_it": beats}
        # `distance_only` (possibly nested) drives the primary verdict; the
        # pre-registered baseline drives its own, so the two are never conflated.
        # `prevalence+distance` carries the SAME nested column, so it belongs to the
        # primary verdict only -- letting it into the prereg verdict would re-introduce
        # the nesting the parallel baseline exists to avoid.
        if name not in PREREG:
            gate_C = gate_C and beats
        if name not in ("distance_only", "prevalence+distance"):
            gate_C_prereg = gate_C_prereg and beats
    if not ablations:
        gate_C = gate_C_prereg = None
    elif prereg_name is None:
        gate_C_prereg = None      # no independent baseline available -> not a verdict

    return {
        "n_classes_used": int(len(valid)),
        "n_train_classes": int(n_tr),
        "n_features_full": int(len(full_cols)),
        "underpowered": bool(n_tr < 3 * len(full_cols)),
        "distance_col_used": dist_name,
        "r2_by_predictor": summ,
        "reliability_ceiling": reliability,
        "normalized_full_r2": norm,
        "gate_B_pass": gate_B,
        "gate_C_pass": gate_C,
        "gate_C_pass_prereg": gate_C_prereg,
        "distance_baseline_is_nested_in_full": nested,
        "distance_col_prereg": prereg_name,
        "gate_C_detail": gate_C_detail,
    }
