"""Phase 2 experiment: PCC on classes with ZERO labelled calibration samples.

This is the primary execution path (§4, §12). Notebooks are thin runners over it, so
that varying seed / α / n_cal / dataset is a loop rather than nine hand-edited cells.

    python -m pcc.experiments.phase2_pcc \
        --scores /content/ccc_npy/imagenet/scores.npy \
        --labels /content/ccc_npy/imagenet/labels.npy \
        --dataset imagenet --alpha 0.10 --heldout-frac 0.3 --seed 0

WHAT THIS MEASURES
------------------
A fraction of classes is chosen as **held-out**: their rows are deleted from the
calibration slice entirely, so `n_y = 0` for them — not "few", *none*. That is the
deployment situation the whole project is about, and it is also why no method can carry
a finite-sample class-conditional guarantee there.

Two tables, never averaged into one (§7):

- **Table 1, seen classes** — PCC must at least tie. If it wins on held-out classes but
  loses here, the honest conclusion is that it trades data-rich classes for data-poor
  ones, and that is what gets written.
- **Table 2, held-out classes** — where the claim lives. Per the frozen
  `reports/fallback_policy.md`, every class-level baseline degenerates to the marginal
  threshold at `n_y = 0`, so the `uncorrected` row *is* the baseline row there.

Both tables use `restrict_to_classes`, so prediction sets span only the label space being
scored. Restricting samples but not score columns is the confound that made three §6.4
designs unusable (Amendment 4); it is not repeated here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pcc.descriptors.head_weights import build_head_descriptors
from pcc.descriptors.output_space import build_output_descriptors
from pcc.eval.conformal import restrict_to_classes
from pcc.eval.metrics import per_class_coverage
from pcc.eval.setsize import avg_set_size_at_shift, equity_at_matched_size
from pcc.eval.stats import mean_ci
from pcc.method.pcc import fit_pcc
from pcc.scores.base import thr_lac
from pcc.utils.io import write_report
from pcc.utils.seed import set_seed

HYPOTHESIS = (
    "For classes with ZERO labelled calibration samples, a correction delta_hat_y "
    "predicted from class-level geometric descriptors phi(y) improves empirical "
    "class-conditional coverage equity at matched average set size, relative to the "
    "marginal threshold that every class-level baseline falls back to there."
)

PASS_CRITERIA = (
    "PRE-REGISTERED, see reports/baseline_reproduction.md ('Target PCC'). "
    "TABLE 2 (held-out, n_y=0): worst-class coverage delta > 0 at matched set size, "
    "with the mean over seeds CI-low > 0. TABLE 1 (seen): PCC must NOT be worse than "
    "the uncorrected marginal threshold by more than 0.01 worst-class coverage; and "
    "when baselines are available it must tie Clustered CP (max_gap <= 0.233 at size "
    "<= 2.568 on ImageNet THR alpha=0.10). Winning Table 2 alone is NOT a pass: every "
    "competitor is undefined at n_y=0 by construction, so a Table 1 loss means the "
    "method trades data-rich classes for data-poor ones and must be reported as such."
)


# --------------------------------------------------------------------------- data
def _load(scores_path, labels_path, max_rows, seed):
    S_mm = np.load(scores_path, mmap_mode="r")
    y_full = np.load(labels_path).astype(int)
    K = int(S_mm.shape[1])
    cnt_full = np.bincount(y_full, minlength=K)

    if max_rows is not None and len(y_full) > max_rows:
        keep = max_rows / len(y_full)
        rng = np.random.default_rng(seed)
        sel = []
        for c in range(K):
            idx = np.where(y_full == c)[0]
            if not len(idx):
                continue
            k = max(1, int(round(keep * len(idx))))
            sel.append(rng.choice(idx, k, replace=False) if k < len(idx) else idx)
        sel = np.sort(np.concatenate(sel))
    else:
        sel = np.arange(len(y_full))

    S = np.ascontiguousarray(S_mm[sel]).astype(np.float32, copy=False)
    y = y_full[sel]
    del S_mm
    if not np.allclose(S[:200].sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("rows do not sum to 1; this driver expects softmax scores")
    return S, y, K, cnt_full, int(len(y_full))


def _two_way_split(y, K, frac_desc, seed):
    """DESC / CAL only — used when EVAL comes from a separate released dump."""
    rng = np.random.default_rng(seed)
    role = np.empty(len(y), dtype="<U4")
    for c in range(K):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_d = int(round(frac_desc * len(idx)))
        role[idx[:n_d]] = "desc"
        role[idx[n_d:]] = "cal"
    return np.where(role == "desc")[0], np.where(role == "cal")[0]


def _three_way_split(y, K, frac_desc, frac_cal, seed):
    rng = np.random.default_rng(seed)
    role = np.empty(len(y), dtype="<U4")
    for c in range(K):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_d = int(round(frac_desc * n))
        n_c = int(round(frac_cal * n))
        role[idx[:n_d]] = "desc"
        role[idx[n_d:n_d + n_c]] = "cal"
        role[idx[n_d + n_c:]] = "eval"
    return (np.where(role == "desc")[0], np.where(role == "cal")[0],
            np.where(role == "eval")[0])


# ----------------------------------------------- measurability (pre-registered)
# reports/prereg_metrics_per_dataset.md, written before any Phase 2 run. Thresholds are
# choices, fixed here so they cannot be moved once results are visible.
PER_CLASS_MIN_EVAL = 30      # below this median, a per-class coverage estimate is noise
BIN_MIN_EVAL_ROWS = 200      # pooled evaluation rows required per prevalence bin
PER_CLASS_STATS = ("worst", "max_gap", "p05", "p10")


def measurability(y_eval, classes) -> dict:
    """Can per-class coverage be estimated at all on this evaluation slice?

    With 2 evaluation samples a class's coverage can only be 0, 0.5 or 1, so `min` over
    thousands of such classes is ~0 for ANY method including an oracle — it measures
    sampling noise, not the method. Pl@ntNet's median is 3 and iNat's is 2, so this is not
    hypothetical.
    """
    per = np.bincount(np.asarray(y_eval, int),
                      minlength=int(max(classes)) + 1)[np.asarray(classes, int)]
    med = float(np.median(per)) if len(per) else 0.0
    ok = bool(med >= PER_CLASS_MIN_EVAL)
    return {"median_eval_per_class": med, "min_eval_per_class": int(per.min()) if len(per) else 0,
            "classes_with_zero_eval": int((per == 0).sum()),
            "coverage_granularity": (1.0 / med) if med else float("inf"),
            "per_class_stats_reportable": ok,
            "regime": "A (per-class)" if ok else "B (prevalence bins)",
            "primary_stat": "worst" if ok else "bin_worst",
            "borderline": bool(25 <= med <= 35),
            "threshold": PER_CLASS_MIN_EVAL}


def prevalence_bins(y_eval, classes, counts, min_rows=BIN_MIN_EVAL_ROWS):
    """Classes sorted by prevalence, accumulated greedily until each bin holds `min_rows`
    evaluation rows. A short final bin is merged backwards, so no half-filled bin can
    dominate the minimum."""
    cls = np.asarray(classes, int)
    order = cls[np.argsort(np.asarray(counts, float)[cls], kind="stable")]
    per = np.bincount(np.asarray(y_eval, int), minlength=int(cls.max()) + 1)
    bins, cur, cur_n = [], [], 0
    for c in order:
        cur.append(int(c))
        cur_n += int(per[c])
        if cur_n >= min_rows:
            bins.append(cur)
            cur, cur_n = [], 0
    if cur:
        if bins:
            bins[-1].extend(cur)
        else:
            bins.append(cur)
    return bins


def _bin_coverage(S_sub, y_sub, thresholds, bins, id_of_class):
    """Pooled coverage inside each bin, in the RESTRICTED label space."""
    hit = S_sub[np.arange(len(y_sub)), y_sub] <= np.asarray(thresholds, float)[y_sub]
    out = []
    for b in bins:
        idx = np.array([id_of_class[c] for c in b if c in id_of_class], int)
        m = np.isin(y_sub, idx)
        if m.any():
            out.append(float(hit[m].mean()))
    return out


# ------------------------------------------------------------------- evaluation
def _one_table(S_ev, y_ev, classes, q_global, thresholds_full, stat, counts=None,
               alpha=0.10):
    """Equity of PCC vs the marginal threshold, inside one label space, matched size.

    Per-class statistics that the pre-registration says are unmeasurable on this slice are
    moved to `withheld_unmeasurable`: still computed, never allowed to decide anything.
    """
    S_sub, y_sub, K_sub, ids = restrict_to_classes(S_ev, y_ev, classes)
    base = np.full(K_sub, float(q_global))
    target = avg_set_size_at_shift(S_sub, base, 0.0)
    t_pcc = np.asarray(thresholds_full, float)[ids]
    e_base = equity_at_matched_size(S_sub, y_sub, K_sub, base, target)
    e_pcc = equity_at_matched_size(S_sub, y_sub, K_sub, t_pcc, target)

    meas = measurability(y_sub, np.arange(K_sub))
    # THE TWO NUMBERS A CP READER LOOKS FOR FIRST, and neither was being reported.
    #
    # `macro` is the mean of PER-CLASS coverage; on an imbalanced label space that is not
    # marginal coverage, which is the fraction of all test points covered and the thing
    # the split-conformal guarantee is about. And `neg_covgap` measures spread around the
    # OBSERVED mean, not distance from the TARGET 1-alpha -- so neither of the stats we
    # had answers "is this valid?". Both are added here, absolute, for both arms.
    tgt = 1.0 - float(alpha)
    rows_i = np.arange(len(y_sub))

    def _abs(thr, e):
        t = np.asarray(thr, float) + e["shift"]
        sets = np.asarray(S_sub) <= t[None, :]
        hit = S_sub[rows_i, y_sub] <= t[y_sub]
        cov = per_class_coverage(sets, y_sub, K_sub)
        e["marginal_cov"] = float(np.mean(hit))
        e["cov_gap_vs_target"] = float(np.nanmean(np.abs(cov - tgt)))

        # %classes below target -- reported by the old UM-TTA paper. Dropping a metric a
        # previous submission carried reads as regression to the same reviewers.
        e["frac_classes_below_target"] = float(np.nanmean(cov < tgt))

        # SSCV, standard since RAPS (Angelopoulos 2021): stratify TEST POINTS by the size
        # of the set they receive, then take the worst deviation from target across
        # strata. A method can hold marginal coverage while systematically under-covering
        # exactly the points it gives small sets to, and only this metric shows it.
        sz = sets.sum(axis=1)
        edges = [(1, 1), (2, 3), (4, 10), (11, 100), (101, int(K_sub))]
        viol, strat = [], {}
        for lo, hi in edges:
            m = (sz >= lo) & (sz <= hi)
            if m.sum() >= 30:                      # below this a stratum is noise
                c = float(hit[m].mean())
                strat["{}-{}".format(lo, hi)] = {"cov": c, "n": int(m.sum())}
                viol.append(abs(c - tgt))
        e["sscv"] = float(max(viol)) if viol else float("nan")
        e["size_strata"] = strat

        # Worst-slab over contiguous prevalence-ordered class windows: a coarser,
        # measurable cousin of worst-class that does not collapse on thin classes.
        order = np.argsort(np.asarray(counts, float)[ids] if counts is not None
                           else np.arange(K_sub), kind="stable")
        w = max(5, K_sub // 20)
        slabs = [float(np.nanmean(cov[order[i:i + w]]))
                 for i in range(0, K_sub - w + 1, max(1, w // 2))]
        slabs = [v for v in slabs if np.isfinite(v)]
        e["worst_slab"] = float(min(slabs)) if slabs else float("nan")
        return e

    e_base = _abs(base, e_base)
    e_pcc = _abs(t_pcc, e_pcc)

    # ORACLE CEILING -- the number that gives +0.0249 a scale. It uses EVAL labels, so it
    # is unachievable by construction; it exists only to say how much of the available
    # room the method took. Without it the headline has no denominator, and nb05 showed
    # the room can be near zero, in which case any figure is unreadable.
    s_true = S_sub[rows_i, y_sub]
    q_or = np.full(K_sub, float(q_global))
    for k in range(K_sub):
        m = y_sub == k
        if m.sum() >= 5:
            q_or[k] = float(np.quantile(s_true[m], tgt))
    e_or = equity_at_matched_size(S_sub, y_sub, K_sub, q_or, target)
    e_or = _abs(q_or, e_or)

    out = {
        "n_classes": int(K_sub), "n_rows": int(len(y_sub)),
        "target_avg_set_size": float(target), "alpha": float(alpha),
        "measurability": meas,
        "uncorrected": e_base, "pcc": e_pcc, "oracle": e_or,
        "delta": {k: e_pcc[k] - e_base[k] for k in e_base
                  if k not in ("shift", "size_strata")},
        "delta_oracle": {k: e_or[k] - e_base[k] for k in e_base
                         if k not in ("shift", "size_strata")},
        "size_matched": bool(abs(e_pcc["avg_set_size"] - e_base["avg_set_size"]) < 1e-2),
    }

    if not meas["per_class_stats_reportable"] and counts is not None:
        id_of_class = {int(c): i for i, c in enumerate(ids)}
        bins = prevalence_bins(y_ev, classes, counts)
        sh_base = e_base["shift"]
        sh_pcc = e_pcc["shift"]
        cb = _bin_coverage(S_sub, y_sub, base + sh_base, bins, id_of_class)
        cp = _bin_coverage(S_sub, y_sub, t_pcc + sh_pcc, bins, id_of_class)
        out["bins"] = {"n_bins": len(bins),
                       "classes_per_bin": [len(b) for b in bins],
                       "uncorrected_bin_coverage": cb,
                       "pcc_bin_coverage": cp}
        if cb and cp:
            out["uncorrected"]["bin_worst"] = float(min(cb))
            out["pcc"]["bin_worst"] = float(min(cp))
            out["delta"]["bin_worst"] = float(min(cp) - min(cb))
        out["withheld_unmeasurable"] = {
            k: {"uncorrected": e_base.get(k), "pcc": e_pcc.get(k),
                "delta": (e_pcc.get(k, np.nan) - e_base.get(k, np.nan))}
            for k in PER_CLASS_STATS if k in e_base}

    primary = stat if meas["per_class_stats_reportable"] else meas["primary_stat"]
    out["primary_stat"] = primary
    out["pass"] = (bool(out["delta"][primary] > 0) if primary in out["delta"] else None)
    return out


def _baselines_table1(ccc_root, S_cal, y_cal, S_ev, y_ev, seen, alpha, seed):
    """Author-implemented baselines, restricted to the SEEN label space.

    Only Table 1: at n_y = 0 every one of these falls back to the marginal threshold
    (frozen in reports/fallback_policy.md), so Table 2's `uncorrected` row already IS
    their row and calling them there would only restate it.
    """
    import importlib
    import sys
    if ccc_root not in sys.path:
        sys.path.insert(0, ccc_root)
    importlib.invalidate_caches()
    cu = importlib.import_module("utils.conformal_utils")

    Sc, yc, Kc, _ = restrict_to_classes(S_cal, y_cal, seen)
    Se, ye, Ke, _ = restrict_to_classes(S_ev, y_ev, seen)
    out = {}
    calls = {
        "standard_conformal": lambda: cu.standard_conformal(Sc, yc, Se, ye, alpha),
        "classwise_conformal": lambda: cu.classwise_conformal(Sc, yc, Se, ye, alpha, Kc),
        "clustered_conformal": lambda: cu.clustered_conformal(
            Sc, yc, alpha, val_scores_all=Se, val_labels=ye, seed=seed),
    }
    for nm, fn in calls.items():
        try:
            parts = list(fn())
        except Exception as e:                                   # noqa: BLE001
            out[nm] = {"error": type(e).__name__ + ": " + str(e)}
            continue
        rec = {}
        for part in parts:
            if not isinstance(part, dict):
                continue
            is_cov = any("cov" in str(k).lower() for k in part)
            for k, v in part.items():
                kl = str(k).lower()
                if isinstance(v, (int, float, np.floating, np.integer)):
                    if "cov" in kl and "gap" in kl:
                        rec["mean_class_cov_gap"] = float(v)
                    elif "max_gap" in kl:
                        rec["max_gap"] = float(v)
                    elif "undercovered" in kl:
                        rec["very_undercovered"] = float(v)
                    elif kl == "marginal_cov":
                        rec["marginal_cov"] = float(v)
                    elif kl == "mean" and not is_cov:
                        rec["avg_set_size"] = float(v)
        out[nm] = rec
    return out


# ------------------------------------------------------------------------- main
def run(args) -> dict:
    set_seed(args.seed)
    S_all, y_all, K, cnt_full, n_dump = _load(args.scores, args.labels,
                                              args.max_rows, args.seed)

    # LTC releases SEPARATE calibration and test dumps. Splitting the test dump three
    # ways would leave ~1 evaluation row per class on Pl@ntNet — worse than the regime-B
    # threshold this project pre-registered. When an eval dump is supplied, the main dump
    # provides DESC+CAL and the whole eval dump is EVAL, which is also how LTC itself
    # evaluates. See reports/prereg_metrics_per_dataset.md.
    S_eval_sep = None
    if args.eval_scores:
        i_desc, i_cal = _two_way_split(y_all, K, args.frac_desc, args.seed)
        i_eval = np.array([], int)
        S_e, y_e, K_e, _, _ = _load(args.eval_scores, args.eval_labels, None, args.seed)
        if K_e != K:
            raise ValueError("eval dump has {} classes, cal dump has {}".format(K_e, K))
        S_eval_sep = (S_e, y_e)
    else:
        i_desc, i_cal, i_eval = _three_way_split(y_all, K, args.frac_desc,
                                                 args.frac_cal, args.seed)

    # --- held-out classes: their calibration rows are DELETED, not thinned ------
    rng = np.random.default_rng(args.seed + 10_000)
    n_ho = int(round(args.heldout_frac * K))
    heldout = np.sort(rng.choice(K, n_ho, replace=False)) if n_ho else np.array([], int)
    seen = np.setdiff1d(np.arange(K), heldout)
    keep_cal = ~np.isin(y_all[i_cal], heldout)
    i_cal_seen = i_cal[keep_cal]

    S_desc = thr_lac(S_all[i_desc])
    S_cal = thr_lac(S_all[i_cal_seen])
    y_cal = y_all[i_cal_seen]
    used_separate_eval = S_eval_sep is not None
    if S_eval_sep is None:
        S_ev = thr_lac(S_all[i_eval])
        y_ev = y_all[i_eval]
    else:
        # thr_lac copies, so the raw eval dump must be released immediately -- on iNat it
        # is 1.5 GB and keeping both alive is the difference between fitting in Colab's
        # standard RAM and not.
        S_ev = thr_lac(S_eval_sep[0])
        y_ev = S_eval_sep[1]
        S_eval_sep = (None, y_ev)

    # φ is built on DESC only. Held-out classes keep their descriptors -- that is the
    # whole point: φ needs no labels, so it exists for a class with no calibration data.
    # A k-NN descriptor needs k neighbours to exist. K varies by dataset (1000 ImageNet,
    # 1081 Pl@ntNet, 8142 iNat, 100 CIFAR-100), so the k list is derived from K rather
    # than assumed — and which k were dropped is recorded, not silently swallowed.
    knn_ks = tuple(k for k in (1, 5, 10, 50) if k <= K - 1)
    knn_dropped = tuple(k for k in (1, 5, 10, 50) if k > K - 1)
    if not knn_ks:
        raise ValueError("K={} is too small for any k-NN descriptor".format(K))

    if args.phi == "head":
        W = np.load(args.head_weights)
        b = np.load(args.head_bias) if args.head_bias else None
        if W.shape[0] != K:
            raise ValueError("head has {} classes, dump has {}".format(W.shape[0], K))
        Phi, names = build_head_descriptors(W, b, knn_ks=knn_ks)
        # The head family carries no prevalence feature, so splice in the TRUE dump
        # counts -- otherwise the prevalence ablation cannot be tested at all.
        Phi = np.column_stack([Phi, np.log(np.maximum(cnt_full, 1))])
        names = list(names) + ["log_prevalence"]
    else:
        # build_output_descriptors expects SOFTMAX rows, not the THR/LAC transform, and
        # it already emits log_prevalence -- appending another would duplicate a column.
        Phi, names = build_output_descriptors(S_all[i_desc], y_all[i_desc], K,
                                             knn_ks=knn_ks, log_prevalence_from=cnt_full)
        names = list(names)

    q_global = float(np.quantile(S_cal[np.arange(len(y_cal)), y_cal], 1 - args.alpha))
    n_per_class = np.bincount(y_cal, minlength=K)
    delta_obs = np.full(K, np.nan)
    for c in seen:
        m = y_cal == c
        if m.sum() >= args.n_cal:
            delta_obs[c] = float(np.quantile(S_cal[m, c], 1 - args.alpha)) - q_global

    # PRE-FLIGHT. g_theta needs classes whose delta_y is observable, and on a long-tail
    # released dump that can be ZERO: Pl@ntNet's calibration dump has a median of 2 rows
    # per class, so n_cal=25 is unreachable there by construction. Without this check the
    # failure surfaces much deeper as "too few usable TRAIN classes (0)", which says
    # nothing about the cause. Reported with the n_cal that IS achievable -- but never
    # applied automatically, because silently lowering n_cal changes the criterion.
    if args.distance_holdout not in names:
        raise ValueError("--distance-holdout {!r} is not a descriptor of the {!r} family; "
                         "available: {}".format(args.distance_holdout, args.phi, names))
    feats = [f for f in names if f != args.distance_holdout]

    seen_counts = np.sort(n_per_class[seen])[::-1]
    n_trainable = int(np.isfinite(delta_obs[seen]).sum())
    min_trainable = len(feats) + 2
    if n_trainable < min_trainable:
        achievable = [int(seen_counts[i]) for i in
                      (min_trainable - 1, min(len(seen_counts), 30) - 1)
                      if i < len(seen_counts)]
        raise ValueError(
            "only {} of {} seen classes reach n_cal={} in the CAL slice, but g_theta needs "
            "at least {} (p+2 for {} features). CAL rows per class: median {}, max {}. "
            "n_cal values that WOULD work: <= {} for {} classes, <= {} for {} classes. "
            "Lower --n-cal explicitly -- it is a pre-registered criterion, so this driver "
            "will not lower it for you.".format(
                n_trainable, len(seen), args.n_cal, min_trainable, len(feats),
                int(np.median(n_per_class[seen])), int(seen_counts[0]),
                achievable[0] if achievable else "n/a", min_trainable,
                achievable[-1] if achievable else "n/a", min(len(seen_counts), 30)))
    # cnt_full lets fit_pcc build prevalence bins when the fit slice is too thin for any
    # per-class statistic -- the regime-B rule, now applied to SELECTION and not only to
    # reporting. Without it fit_pcc falls back to p25 and says so.
    model = fit_pcc(Phi, names, delta_obs, n_per_class, q_global, args.alpha,
                    score_matrix_fit=S_cal, labels_fit=y_cal, train_classes=seen,
                    features=feats, stat=args.stat, class_counts=cnt_full,
                    seed=args.seed)
    t = model.thresholds()

    res = {
        "n_classes": K, "n_seen": int(len(seen)), "n_heldout": int(len(heldout)),
        "split_sizes": {"desc": int(len(i_desc)), "cal_seen": int(len(i_cal_seen)),
                        "eval": int(len(y_ev))},
        "eval_from_separate_dump": bool(used_separate_eval),
        "subsample_frac_of_dump": float(len(y_all) / n_dump),
        "q_global": q_global,
        "knn_ks_used": list(knn_ks), "knn_ks_dropped_K_too_small": list(knn_dropped),
        "delta_obs_defined": int(np.isfinite(delta_obs).sum()),
        "cal_rows_per_seen_class": {"median": float(np.median(n_per_class[seen])),
                                    "min": int(n_per_class[seen].min()),
                                    "max": int(n_per_class[seen].max())},
        "pcc": {"lambda": model.lam, "n_star": model.n_star, "offset": model.offset,
                "n_star_selection": model.threshold_rule["selected"],
                "n_star_mse_crossing_secondary":
                    model.threshold_rule["mse_crossing_secondary"],
                "blend": {k: v for k, v in model.blend.items() if k != "delta"
                          and k != "used_observed"},
                "lambda_curve_train": model.lambda_selection["curve"],
                "features": list(model.gtheta.feature_names),
                "provenance": model.provenance},
        "table_1_seen": _one_table(S_ev, y_ev, seen, q_global, t, args.stat,
                                   counts=cnt_full, alpha=args.alpha),
    }
    if len(heldout):
        res["table_2_heldout"] = _one_table(S_ev, y_ev, heldout, q_global, t, args.stat,
                                            counts=cnt_full, alpha=args.alpha)

    if args.ccc_root:
        try:
            res["baselines_table_1"] = _baselines_table1(
                args.ccc_root, S_cal, y_cal, S_ev, y_ev, seen, args.alpha, args.seed)
        except Exception as e:                                   # noqa: BLE001
            res["baselines_table_1"] = {"error": type(e).__name__ + ": " + str(e)}
    return res


def verdict(res: dict, stat: str) -> str:
    """The verdict may only use a statistic the pre-registration calls measurable on this
    slice (reports/prereg_metrics_per_dataset.md). Each table carries its own
    `primary_stat`, so a long-tail dataset is judged on prevalence bins rather than on a
    per-class minimum that is pure sampling noise there."""
    t2 = res.get("table_2_heldout")
    t1 = res["table_1_seen"]
    if t2 is None:
        return "TIDAK DAPAT DINILAI (tidak ada kelas held-out)"
    s1 = t1.get("primary_stat", stat)
    s2 = t2.get("primary_stat", stat)
    if s1 not in t1["delta"] or s2 not in t2["delta"]:
        return "TIDAK DAPAT DINILAI (statistik primer tidak terukur di slice ini)"
    won_2 = t2["delta"][s2] > 0
    kept_1 = t1["delta"][s1] > -0.01
    if won_2 and kept_1:
        return "LULUS"
    if won_2 and not kept_1:
        return "MENUKAR: menang held-out, kalah pada kelas terlihat"
    return "GAGAL"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--reports-dir", default="pcc/reports")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n-cal", type=int, default=25,
                   help="min CAL samples for delta_y to count as OBSERVED")
    p.add_argument("--heldout-frac", type=float, default=0.30)
    p.add_argument("--frac-desc", type=float, default=0.40)
    p.add_argument("--frac-cal", type=float, default=0.30)
    p.add_argument("--eval-scores", default=None,
                   help="separate EVAL dump (LTC releases cal and test apart). When given, "
                        "the main dump supplies DESC+CAL and this whole dump is EVAL.")
    p.add_argument("--eval-labels", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--phi", choices=("output", "head"), default="head")
    p.add_argument("--head-weights", default=None)
    p.add_argument("--head-bias", default=None)
    p.add_argument("--distance-holdout", default=None,
                   help="descriptor withheld so the distance baseline is not nested in "
                        "the full model. Defaults to the pre-registered choice for the "
                        "chosen phi family: w_cos_knn_1 (head) / prof_knn_1 (output).")
    p.add_argument("--stat", default="worst")
    p.add_argument("--ccc-root", default=None,
                   help="cloned class-conditional-conformal root, for Table 1 baselines")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--print-json", action="store_true")
    a = p.parse_args(argv)

    if a.phi == "head" and not a.head_weights:
        p.error("--phi head requires --head-weights")
    if a.distance_holdout is None:
        a.distance_holdout = "w_cos_knn_1" if a.phi == "head" else "prof_knn_1"
    if bool(a.eval_scores) != bool(a.eval_labels):
        p.error("--eval-scores and --eval-labels must be given together")

    t0 = time.time()
    res = run(a)
    concl = verdict(res, a.stat)
    name = a.name or "phase2_pcc_{}_{}_a{}_ho{}_s{}".format(
        a.dataset, a.phi, a.alpha, a.heldout_frac, a.seed)
    path = write_report(a.reports_dir, name, hypothesis=HYPOTHESIS,
                        pass_criteria=PASS_CRITERIA, config=vars(a), seed=a.seed,
                        results=res, conclusion=concl, started_at=t0)

    t1, t2 = res["table_1_seen"], res.get("table_2_heldout")
    print("== {} | phi={} | alpha={} | held-out {}/{} kelas ==".format(
        a.dataset, a.phi, a.alpha, res["n_heldout"], res["n_classes"]))
    print("  lambda {:.3f} | n_star {} | offset {:+.4f} | delta_y terukur {}".format(
        res["pcc"]["lambda"], res["pcc"]["n_star"], res["pcc"]["offset"],
        res["delta_obs_defined"]))
    print("  blend:", res["pcc"]["blend"])
    for tag, tb in (("TABEL 1 kelas terlihat", t1), ("TABEL 2 kelas held-out", t2)):
        if tb is None:
            continue
        m = tb["measurability"]
        print("  {} ({} kelas, size {:.3f}) | rezim {} | median eval/kelas {:.0f}".format(
            tag, tb["n_classes"], tb["target_avg_set_size"], m["regime"],
            m["median_eval_per_class"]))
        s = tb["primary_stat"]
        if s in tb["delta"]:
            print("    {:10s} global {:+.4f} -> PCC {:+.4f}  delta {:+.4f}{}".format(
                s, tb["uncorrected"][s], tb["pcc"][s], tb["delta"][s],
                "" if tb["size_matched"] else "   [UKURAN TIDAK COCOK — jangan dibaca]"))
        else:
            print("    {} tidak terukur di slice ini".format(s))
        print("    macro      global {:+.4f} -> PCC {:+.4f}  delta {:+.4f}".format(
            tb["uncorrected"]["macro"], tb["pcc"]["macro"], tb["delta"]["macro"]))
        if "withheld_unmeasurable" in tb:
            print("    DITAHAN (tak terukur, granularitas {:.2f}): {}".format(
                m["coverage_granularity"], sorted(tb["withheld_unmeasurable"])))
    print("  VERDICT:", concl)
    print("  laporan:", path)
    if a.print_json:
        print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
