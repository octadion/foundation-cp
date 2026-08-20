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
from pcc.eval.setsize import (avg_set_size_at_shift, equity_at_matched_size,
                              select_shrinkage)
from pcc.eval.stats import mean_ci
from pcc.method.pcc import fit_pcc
from pcc.scores.base import SCORE_FNS, score_matrix, thr_lac
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
               alpha=0.10, competitors=None):
    """Equity of PCC vs the marginal threshold, inside one label space, matched size.

    Per-class statistics that the pre-registration says are unmeasurable on this slice are
    moved to `withheld_unmeasurable`: still computed, never allowed to decide anything.

    `competitors` maps a published method's name to a list of `(label, thresholds_full)`
    candidates. Every one of them is a PER-CLASS THRESHOLD VECTOR, which is the whole
    reason they can be compared honestly: they go through the same size-matching shift and
    the same statistics as PCC, instead of having their own heterogeneous metric dicts
    scraped for keys that happen to contain "cov". When a method offers several
    hyperparameters, the BEST candidate is taken -- oracle-tuning the competitor, which is
    the conservative direction for us, and recorded as such.
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
        # Empty-set rate. Under distribution shift a threshold calibrated on clean data
        # can put NOTHING in the set: the ImageNet-C phase produced average set sizes
        # below 1.0, which is only possible if a large share of rows get an empty set.
        # Coverage and set size both look merely "low" in that regime; this is the number
        # that says the predictor abstained rather than guessed narrowly.
        e["frac_empty_sets"] = float(np.mean(sets.sum(axis=1) == 0))
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

    # G4: WHICH classes the correction helps, not just by how much.
    #
    # A single worst-class number cannot say whether the method lifts the rare tail or
    # merely trades one unlucky class for another, and that distinction is the whole
    # story for a long-tail claim. Per-class coverage change is summarised by prevalence
    # quintile, and the identity of the worst class is recorded before and after -- if
    # the worst class is the SAME one both times, PCC lifted it; if it moved, PCC traded.
    def _who(thr_a, thr_b, sh_a, sh_b):
        ca = per_class_coverage(np.asarray(S_sub) <= (np.asarray(thr_a, float) + sh_a)[None, :],
                                y_sub, K_sub)
        cb = per_class_coverage(np.asarray(S_sub) <= (np.asarray(thr_b, float) + sh_b)[None, :],
                                y_sub, K_sub)
        d = cb - ca
        prev = np.asarray(counts, float)[ids] if counts is not None else np.arange(K_sub)
        order = np.argsort(prev, kind="stable")            # rarest first
        q = np.array_split(order, 5)
        out = {"by_prevalence_quintile": [
                   {"quintile": i + 1, "n_classes": int(len(g_)),
                    "median_prevalence": float(np.median(prev[g_])),
                    "mean_coverage_change": float(np.nanmean(d[g_])),
                    "frac_improved": float(np.nanmean(d[g_] > 0))}
                   for i, g_ in enumerate(q) if len(g_)],
               "frac_classes_improved": float(np.nanmean(d > 0)),
               "frac_classes_worsened": float(np.nanmean(d < 0))}
        if np.isfinite(ca).any() and np.isfinite(cb).any():
            wa, wb = int(np.nanargmin(ca)), int(np.nanargmin(cb))
            out.update({
                "worst_class_before": {"class_id": int(ids[wa]), "coverage": float(ca[wa]),
                                       "prevalence": float(prev[wa])},
                "worst_class_after": {"class_id": int(ids[wb]), "coverage": float(cb[wb]),
                                      "prevalence": float(prev[wb])},
                # same class lifted, or a different class now at the bottom
                "worst_class_is_same": bool(wa == wb),
                "coverage_of_old_worst_after": float(cb[wa])})
        return out

    e_base = _abs(base, e_base)
    e_pcc = _abs(t_pcc, e_pcc)

    # ORACLE CEILING -- and it has to be SHRUNK to actually be a ceiling.
    #
    # The first version took per-class quantiles of the true-class scores straight from
    # the EVAL labels and called that the ceiling. It is not one: PCC measured +0.0588
    # against an "oracle" of -0.0058 on the same run, and a method beating its own upper
    # bound means the bound was mislabelled. The reason is that raw per-class quantiles
    # ARE lambda=1, and lambda=1 is catastrophic for worst-class at matched size -- the
    # lam1 ablation measures -0.58 on this very dump. So the unshrunk oracle is not
    # "perfect delta", it is "perfect delta used in the worst possible way".
    #
    # The honest ceiling for this family is the BEST shrinkage of a perfect delta. It
    # still uses EVAL labels twice over (for delta and for lambda), so it remains
    # unachievable and remains a ceiling rather than a method -- just one that PCC
    # cannot exceed by construction, which is what a ceiling has to mean.
    s_true = S_sub[rows_i, y_sub]
    q_raw = np.full(K_sub, float(q_global))
    for k in range(K_sub):
        m = y_sub == k
        if m.sum() >= 5:
            q_raw[k] = float(np.quantile(s_true[m], tgt))
    d_or = q_raw - float(q_global)
    # select_shrinkage cannot optimise a BIN statistic, so in regime B the ceiling's
    # lambda is chosen on per-class `worst` even though the arms are read on bin_worst.
    # That is acceptable for a ceiling -- it is not a method and carries no guarantee --
    # but which statistic chose lambda has to be on the record, not inferred.
    or_stat = stat if meas["per_class_stats_reportable"] else "worst"
    or_sel = select_shrinkage(S_sub, y_sub, K_sub, q_global, d_or, target, stat=or_stat)
    q_or = float(q_global) + or_sel["lambda"] * d_or
    e_or = equity_at_matched_size(S_sub, y_sub, K_sub, q_or, target)
    e_or = _abs(q_or, e_or)
    e_or["oracle_lambda"] = float(or_sel["lambda"])
    e_or["oracle_lambda_curve"] = or_sel["curve"]
    e_or["oracle_lambda_stat"] = or_stat
    # the unshrunk version is kept because it is what the lam1 ablation predicts, and
    # seeing the two side by side is the clearest statement of why shrinkage exists
    e_unsh = _abs(q_raw, equity_at_matched_size(S_sub, y_sub, K_sub, q_raw, target))

    # UNMATCHED thresholds, taken exactly as the method emits them.
    #
    # Every number above is computed after a scalar shift that forces both arms to the
    # same average set size -- which is the point (Amendment 8: match the resource, read
    # the benefit), but it also means a CONSTANT added to every threshold is cancelled
    # exactly. PCC's marginal recalibration is precisely such a constant, so the E3
    # ablation ("recalibration on/off") is invisible in the matched tables by
    # construction, and the dry-run made that visible by returning deltas of exactly
    # 0.0000. The offset is not doing nothing; it is doing something this view cannot
    # see. So the raw view is recorded too: it is where marginal validity actually lives.
    def _raw(thr):
        t = np.asarray(thr, float)
        sets = np.asarray(S_sub) <= t[None, :]
        hit = S_sub[rows_i, y_sub] <= t[y_sub]
        cov = per_class_coverage(sets, y_sub, K_sub)
        return {"marginal_cov": float(np.mean(hit)),
                "avg_set_size": float(sets.sum(axis=1).mean()),
                "cov_gap_vs_target": float(np.nanmean(np.abs(cov - tgt))),
                "frac_classes_below_target": float(np.nanmean(cov < tgt)),
                "worst": float(np.nanmin(cov)) if np.isfinite(cov).any() else float("nan")}

    raw_base, raw_pcc = _raw(base), _raw(t_pcc)

    out = {
        "n_classes": int(K_sub), "n_rows": int(len(y_sub)),
        "target_avg_set_size": float(target), "alpha": float(alpha),
        "measurability": meas,
        "uncorrected": e_base, "pcc": e_pcc, "oracle": e_or,
        "delta": {k: e_pcc[k] - e_base[k] for k in e_base
                  if k not in ("shift", "size_strata")},
        "delta_oracle": {k: e_or[k] - e_base[k] for k in e_base
                         if k not in ("shift", "size_strata")},
        # the unshrunk oracle: perfect delta at lambda=1, i.e. the lam1 ablation with a
        # perfect predictor. Reported so "why shrink?" is answered by a number.
        "delta_oracle_unshrunk": {k: e_unsh[k] - e_base[k] for k in e_base
                                  if k not in ("shift", "size_strata")},
        "size_matched": bool(abs(e_pcc["avg_set_size"] - e_base["avg_set_size"]) < 1e-2),
        "raw_unmatched": {"uncorrected": raw_base, "pcc": raw_pcc,
                          "delta": {k: raw_pcc[k] - raw_base[k] for k in raw_base}},
        "who_is_helped": _who(base, t_pcc, e_base["shift"], e_pcc["shift"]),
    }

    if not meas["per_class_stats_reportable"] and counts is not None:
        id_of_class = {int(c): i for i, c in enumerate(ids)}
        bins = prevalence_bins(y_ev, classes, counts)
        sh_base = e_base["shift"]
        sh_pcc = e_pcc["shift"]
        cb = _bin_coverage(S_sub, y_sub, base + sh_base, bins, id_of_class)
        cp = _bin_coverage(S_sub, y_sub, t_pcc + sh_pcc, bins, id_of_class)
        # The oracle needs the SAME statistic as the arms it is supposed to bound. Without
        # this the ceiling has no bin_worst, so "how much of the available room did we
        # take" is unanswerable in regime B -- which is Pl@ntNet and iNat, two of the
        # three datasets. The ceiling was only ever readable on ImageNet.
        co = _bin_coverage(S_sub, y_sub, q_or + e_or["shift"], bins, id_of_class)
        cu = _bin_coverage(S_sub, y_sub, q_raw + e_unsh["shift"], bins, id_of_class)
        out["bins"] = {"n_bins": len(bins),
                       "classes_per_bin": [len(b) for b in bins],
                       "uncorrected_bin_coverage": cb,
                       "pcc_bin_coverage": cp,
                       "oracle_bin_coverage": co}
        if cb and cp:
            out["uncorrected"]["bin_worst"] = float(min(cb))
            out["pcc"]["bin_worst"] = float(min(cp))
            out["delta"]["bin_worst"] = float(min(cp) - min(cb))
        if cb and co:
            out["oracle"]["bin_worst"] = float(min(co))
            out["delta_oracle"]["bin_worst"] = float(min(co) - min(cb))
        if cb and cu:
            out["delta_oracle_unshrunk"]["bin_worst"] = float(min(cu) - min(cb))
        out["withheld_unmeasurable"] = {
            k: {"uncorrected": e_base.get(k), "pcc": e_pcc.get(k),
                "delta": (e_pcc.get(k, np.nan) - e_base.get(k, np.nan))}
            for k in PER_CLASS_STATS if k in e_base}

    primary = stat if meas["per_class_stats_reportable"] else meas["primary_stat"]
    out["primary_stat"] = primary

    # --- published competitors, on identical footing -----------------------------
    if competitors:
        out["competitors"], out["delta_competitors"] = {}, {}
        for nm, cands in competitors.items():
            best = None
            for label, thr_full in cands:
                t_c = np.asarray(thr_full, float)[ids]
                n_undef = int((~np.isfinite(t_c)).sum())
                if n_undef:
                    # a non-finite threshold means "include every label", which is what
                    # classwise CP and RC3P emit for a class with no calibration rows.
                    # The frozen fallback policy sends those to the marginal threshold;
                    # substituting it here is that policy, applied per class.
                    t_c = np.where(np.isfinite(t_c), t_c, float(q_global))
                e_c = _abs(t_c, equity_at_matched_size(S_sub, y_sub, K_sub, t_c, target))
                if primary in e_c:
                    e_c["selected_by"] = primary
                elif "worst" in e_c:
                    e_c["selected_by"] = "worst"
                key = (e_c.get(e_c["selected_by"], -np.inf), -n_undef)
                e_c["hyperparameter"] = label
                # THE number that makes the held-out comparison legible: how many classes
                # the method could not give a threshold to at all. Worst-class coverage is
                # decided by the classes a method fails on, not the ones it handles, so a
                # method can hand 12 of 18 held-out classes an informative threshold and
                # still score exactly 0.0000.
                e_c["n_classes_undefined"] = n_undef
                e_c["frac_classes_undefined"] = float(n_undef) / max(1, K_sub)
                if best is None or key > best[0]:
                    best = (key, e_c)
            if best is None:
                continue
            e_c = best[1]
            out["competitors"][nm] = e_c
            out["delta_competitors"][nm] = {
                k: e_c[k] - e_base[k] for k in e_base
                if k not in ("shift", "size_strata") and k in e_c}
            out["competitors"][nm]["n_candidates"] = len(cands)
    out["pass"] = (bool(out["delta"][primary] > 0) if primary in out["delta"] else None)
    return out


# Bandwidth is a TUNING AXIS for fuzzy classwise CP, not a constant: the release sweeps
# 1e-30 ... 1000 and picks off a Pareto plot. A small geometric subset is swept here and
# the best is kept, which hands the competitor its best shot -- the conservative direction.
FUZZY_BANDWIDTHS = (1e-5, 1e-3, 1e-2, 1e-1, 10.0)

# MEASURED, not estimated: one fuzzy_classwise_CP fit takes ~99 s at K=1000 with 122k
# calibration rows, because it walks all K classes and builds the per-row weight vector
# with a Python list comprehension over every calibration row. Fifteen candidates (five
# bandwidths x three projections) is therefore ~25 min PER RUN at that scale, and cost
# scales with calibration rows -- on the ImageNet-C slices (20 rows/class) it is ~3 min.
#
# So competitors are opt-in per run rather than always on. Running them in every ablation
# cell would cost tens of hours to restate the same comparison; they belong in the tables
# that actually compare against published work.
COMPETITOR_COST = ("~25 min/run at 122k cal rows and K=1000; ~3 min at 14k rows. "
                   "Dominated by fuzzy_classwise_CP: 15 candidates x ~99 s.")

# Ding et al. sweep tau over {0, .25, .5, .75, .9, .95, .975, .99, .999, 1}. We keep the
# end points and the values where their own figures show the size/coverage trade-off
# actually turning; as with the fuzzy bandwidth the BEST is kept, which favours them.
INTERP_Q_WEIGHTS = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


def _competitor_thresholds(ltc_root, S_cal, y_cal, K, alpha, seed, P_cal=None,
                           class_counts=None, tmp_dir=None):
    """Per-class threshold vectors from the published methods, over the FULL label space.

    The calibration slice contains rows only for SEEN classes, but the score matrix keeps
    all K columns, so each method decides for itself what a class with zero calibration
    rows gets. That is the point: `reports/fallback_policy.md` was frozen on 2026-07-24
    predicting exactly what each one would do, and the released code agrees --

      * classwise CP  -> qhat = +inf (set = all labels); policy sends it to q_global
      * clustered CP  -> null cluster -> the method's own marginal fallback
      * RC3P          -> warns "no calibration examples", qhat = +inf; policy -> q_global
      * fuzzy CP      -> similarity-weighted over seen classes, sd = bandwidth/(n_k+1),
                         whose "+1 to account for classes with 0 examples" is the
                         authors' own comment. This is the ONE published method that is
                         genuinely defined at n_y = 0, which is why AGENTS.md §7 flags it
                         as the most dangerous competitor and why it is the real test.

    Non-finite entries are left as they come; `_one_table` applies the frozen policy.
    """
    import importlib
    import sys
    if ltc_root not in sys.path:
        sys.path.insert(0, ltc_root)
    importlib.invalidate_caches()
    cu = importlib.import_module("utils.conformal_utils")

    out, errs = {}, {}
    # The released code prints a per-class line for every class it cannot fit -- with
    # K=1000 and 300 held-out classes that is thousands of lines per candidate, tens of
    # thousands per run. It drowns the notebook log and truncated the one that had to be
    # read to diagnose phase 3. Captured rather than discarded: the lines are counted and
    # the first few kept, so nothing new goes unnoticed.
    import contextlib
    import io as _io
    chatter = {}

    def _try(name, fn, label="-"):
        buf = _io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                q = np.atleast_1d(np.asarray(fn(), dtype=float)).ravel()
        except Exception as e:                                       # noqa: BLE001
            errs[name] = type(e).__name__ + ": " + str(e)[:200]
            return
        finally:
            lines = [l for l in buf.getvalue().splitlines() if l.strip()]
            if lines:
                c = chatter.setdefault(name, {"n_lines": 0, "sample": []})
                c["n_lines"] += len(lines)
                for l in lines[:3]:
                    if l not in c["sample"] and len(c["sample"]) < 3:
                        c["sample"].append(l)
        if q.size == 1:
            q = np.full(K, float(q[0]))
        if q.size != K:
            errs[name] = "returned {} thresholds, expected {}".format(q.size, K)
            return
        out.setdefault(name, []).append((label, q))

    _try("standard_conformal",
         lambda: cu.compute_qhat(S_cal, y_cal, alpha))
    _try("classwise_conformal",
         lambda: cu.compute_class_specific_qhats(S_cal, y_cal, K, alpha))
    _try("clustered_conformal",
         lambda: cu.clustered_conformal(S_cal, y_cal, alpha, seed=seed))
    if P_cal is not None:
        _try("rc3p",
             lambda: cu.compute_rc3p_params(P_cal, S_cal, y_cal, alpha)[0])
        del P_cal


    # The `rarity` projection embeds a class by its TRAIN prevalence, which exists without
    # any calibration row -- so it is the one projection that can give a held-out class an
    # INFORMATIVE position rather than a degenerate one. With use_train=False it would read
    # counts off the calibration labels, where every held-out class is zero and the
    # embedding collapses. Materialising the dump's true counts is therefore what makes
    # this competitor strong, and a strong version is the one worth beating: it is the
    # direct analogue of PCC's own log_prevalence feature, which the E4 ablation says
    # carries no signal on its own. That makes this a falsifiable prediction, not a
    # formality.
    rarity_path = None
    if class_counts is not None and tmp_dir is not None:
        cnt = np.asarray(class_counts, int)
        rarity_path = str(Path(tmp_dir) / "ltc_train_labels.npy")
        np.save(rarity_path, np.repeat(np.arange(len(cnt)), cnt))

    # INTERP-Q, Ding et al.'s second proposed method. Transcribed from `interpQ` in their
    # released example.ipynb: the classwise quantile with its infinities replaced by one --
    # the maximum of the softmax score -- then linearly blended with the marginal quantile.
    # Their own compute_qhat does both quantiles, so this is their arithmetic on their
    # estimates. Unlike classwise CP it is DEFINED at n_y = 0, which is why the review is
    # right that it belongs in the table: the cap gives every empty class the value one and
    # the blend pulls it back towards q_hat.
    #
    # cw and std do not depend on the weight, so they are computed once and the six weights
    # are six vector combinations. Recomputing them per weight would cost K quantile calls
    # each and turn a second into minutes at K=8142.
    def _interp_q_parts():
        std = float(np.atleast_1d(np.asarray(
            cu.compute_qhat(S_cal, y_cal, alpha), dtype=float)).ravel()[0])
        cw = np.full(K, np.inf)
        for k in range(K):
            m = y_cal == k
            if not m.any():
                continue
            cw[k] = float(np.atleast_1d(np.asarray(
                cu.compute_qhat(S_cal[m], y_cal[m], alpha), dtype=float)).ravel()[0])
            del m
        cw[~np.isfinite(cw)] = 1.0
        return cw, std

    # Same stdout treatment as every other competitor. This loop calls the released
    # compute_qhat K times, and K is 1000 here and 8142 on iNaturalist, so leaving it
    # outside the capture is how a single configuration buries the log.
    _buf = _io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            _cw, _std = _interp_q_parts()
    except Exception as e:                                           # noqa: BLE001
        errs["interp_q"] = type(e).__name__ + ": " + str(e)[:200]
        _cw = None
    finally:
        _lines = [l for l in _buf.getvalue().splitlines() if l.strip()]
        if _lines:
            _c = chatter.setdefault("interp_q", {"n_lines": 0, "sample": []})
            _c["n_lines"] += len(_lines)
            for _l in _lines[:3]:
                if _l not in _c["sample"] and len(_c["sample"]) < 3:
                    _c["sample"].append(_l)
    if _cw is not None:
        for tau in INTERP_Q_WEIGHTS:
            _try("interp_q", (lambda t=tau: t * _cw + (1.0 - t) * _std),
                 label="tau={}".format(tau))

    projections = ["quantile", "random"] + (["rarity"] if rarity_path else [])
    for bw in FUZZY_BANDWIDTHS:
        for proj in projections:
            par = {"bandwidth": bw}
            if proj == "rarity":
                # "dataset" is required even though train_labels_path is supplied: their
                # default argument is an f-string that reads params["dataset"], and Python
                # evaluates it before .get() can ignore it.
                par.update({"use_train": True, "train_labels_path": rarity_path,
                            "dataset": "pcc"})

            def _call(p=proj, pa=par):
                # their rarity branch jitters the embedding with the LEGACY global RNG,
                # so seeding it here is what makes this reproducible at all
                np.random.seed(seed)
                return cu.fuzzy_classwise_CP(S_cal, y_cal, alpha, projection=p,
                                             mode="weight", params=pa)[0]

            _try("fuzzy_classwise_" + proj, _call,
                 label="bandwidth={}".format(bw))
    # rc3p first, then release the softmax copy. It is the only method that needs raw
    # softmax, and it is also the heaviest: compute_ranks builds two int64 (n_cal x K)
    # matrices and loops over every row in Python -- measured 68 s and +1.85 GB at 70k
    # rows. Holding P_cal through the fifteen fuzzy fits afterwards would keep another
    # 0.28 GB resident (0.7 GB on the full slice) for nothing.
    # kept OUT of `errs`: that field means "a competitor failed", and noisy stdout is
    # not a failure. Conflating them would make a clean run look broken.
    return out, errs, chatter


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

    # CALIBRATION DEPTH -- the sweep axis. Four settings disagreed about whether PCC works
    # and the only thing that separated them was rows per class (76 vs 20 vs 12 vs 3), but
    # that was confounded with dataset and backbone. Capping depth inside ONE dump turns
    # an accidental cross-dataset observation into a designed curve.
    #
    # A cap ABOVE the rows the slice actually holds is not a sweep point -- it is the
    # same run under a different label. The first sweep asked for depths 100 and 200 on
    # a slice with 75 rows per class and got two bit-identical results that read as "the
    # curve has saturated" when they meant "nothing was cut". So how many classes the cap
    # actually bound is recorded, and a cap that binds nothing is reported as such.
    cap_bound = 0
    if getattr(args, "cal_depth", None):
        rr = np.random.default_rng(args.seed + 777)
        keep = []
        for c in seen:
            idx = i_cal_seen[y_all[i_cal_seen] == c]
            if len(idx) > args.cal_depth:
                cap_bound += 1
                keep.append(rr.choice(idx, int(args.cal_depth), replace=False))
            else:
                keep.append(idx)
        i_cal_seen = np.sort(np.concatenate(keep)) if keep else i_cal_seen

    # SCORE FUNCTION AS AN AXIS. Until now the driver hardcoded THR/LAC, so every PCC
    # number ever produced used one score -- and delta_y is defined on the score
    # distribution, so "does the correction survive a different score?" is a question a
    # reviewer will ask before they ask for a sixteenth corruption. Same seed for all
    # three slices: APS/RAPS/SAPS are randomized, and drawing fresh uniforms per slice
    # would make calibration and evaluation disagree about the same point.
    score = getattr(args, "score", "thr")
    if score not in SCORE_FNS and score != "pas":
        raise ValueError("unknown --score {!r}; available: {}".format(
            score, sorted(SCORE_FNS) + ["pas"]))

    # PAS is Ding et al.'s prevalence-adjusted softmax, s = -p(y|x)/p(y). Its prior comes
    # from the TRAIN label counts, which exist for a class with no calibration rows, so it
    # is one of the few published scores that is defined in our regime -- and a reviewer
    # asked for exactly this comparison. The counts are the same ones that feed
    # log_prevalence, so the two constructions see identical information.
    _priors = None
    if score == "pas":
        c = np.asarray(cnt_full, float)
        if c.sum() <= 0:
            raise ValueError("--score pas needs class counts; this dump reports none")
        _priors = np.maximum(c, 1.0) / max(c.sum(), 1.0)

    def _sc(P):
        return score_matrix(P, score, seed=args.seed, priors=_priors)

    # EVALUATION DEPTH -- the mirror of cal_depth, and the axis every failure so far
    # points at. CCC works with 75-175 evaluation rows per class; the torchvision-backbone
    # dumps fail with 35; Pl@ntNet and iNat fail with 2-3. Calibration depth was ruled out
    # by its own sweep (10 rows still works on CCC), so evaluation depth is the remaining
    # candidate -- and capping it INSIDE one dump is the only way to test it without
    # confounding dataset and backbone all over again.
    if getattr(args, "eval_depth", None) and len(i_eval):
        re = np.random.default_rng(args.seed + 999)
        keep_e = []
        for c in range(K):
            idx = i_eval[y_all[i_eval] == c]
            keep_e.append(re.choice(idx, int(args.eval_depth), replace=False)
                          if len(idx) > args.eval_depth else idx)
        i_eval = np.sort(np.concatenate(keep_e)) if keep_e else i_eval

    # PCC-SPLIT. Proposition 1 needs the rows behind delta_tilde and the rows behind the
    # marginal offset c to be disjoint; with --frac-recal 0 they are the same rows, which
    # is what every run before 2026-08-19 did and what reports/phase2_results.md measures
    # the optimism of. A positive fraction carves the second slice PER CLASS, so no class
    # loses its whole share, and the fit slice shrinks accordingly -- the cost is real and
    # tab:depth says roughly what halving calibration depth costs.
    i_recal = np.array([], int)
    frac_recal = float(getattr(args, "frac_recal", 0.0) or 0.0)
    if frac_recal > 0:
        if not 0 < frac_recal < 1:
            raise ValueError("--frac-recal must lie in (0,1), got {}".format(frac_recal))
        rr2 = np.random.default_rng(args.seed + 4242)
        fit_keep, rec_keep = [], []
        for c in seen:
            idx = i_cal_seen[y_all[i_cal_seen] == c]
            rr2.shuffle(idx)
            n_r = int(round(frac_recal * len(idx)))
            rec_keep.append(idx[:n_r])
            fit_keep.append(idx[n_r:])
        i_recal = np.sort(np.concatenate(rec_keep)) if rec_keep else i_recal
        i_cal_seen = np.sort(np.concatenate(fit_keep)) if fit_keep else i_cal_seen
        if len(i_recal) < 2 or len(i_cal_seen) < 2:
            raise ValueError(
                "--frac-recal {} leaves {} fit rows and {} recalibration rows; both need "
                "at least 2".format(frac_recal, len(i_cal_seen), len(i_recal)))

    S_cal = _sc(S_all[i_cal_seen])
    y_cal = y_all[i_cal_seen]
    S_recal = _sc(S_all[i_recal]) if len(i_recal) else None
    y_recal = y_all[i_recal] if len(i_recal) else None
    used_separate_eval = S_eval_sep is not None
    if S_eval_sep is None:
        S_ev = _sc(S_all[i_eval])
        y_ev = y_all[i_eval]
    else:
        # the score transform copies, so the raw eval dump must be released immediately
        # -- on iNat it is 1.5 GB and keeping both alive is the difference between
        # fitting in Colab's standard RAM and not.
        S_ev = _sc(S_eval_sep[0])
        y_ev = S_eval_sep[1]
        S_eval_sep = (None, y_ev)

    # φ is built on DESC only. Held-out classes keep their descriptors -- that is the
    # whole point: φ needs no labels, so it exists for a class with no calibration data.
    # A k-NN descriptor needs k neighbours to exist. K varies by dataset (1000 ImageNet,
    # 1081 Pl@ntNet, 8142 iNat, 100 CIFAR-100), so the k list is derived from K rather
    # than assumed — and which k were dropped is recorded, not silently swallowed.
    ks_req = getattr(args, "knn_ks", None) or (1, 5, 10, 50)
    ks_req = tuple(int(k) for k in ks_req)
    knn_ks = tuple(k for k in ks_req if k <= K - 1)
    knn_dropped = tuple(k for k in ks_req if k > K - 1)
    if not knn_ks:
        raise ValueError("K={} is too small for any k-NN descriptor".format(K))

    if args.phi == "head":
        W = np.load(args.head_weights)
        b = np.load(args.head_bias) if args.head_bias else None
        if W.shape[0] != K:
            raise ValueError("head has {} classes, dump has {}".format(W.shape[0], K))
        Phi, names = build_head_descriptors(
            W, b, knn_ks=knn_ks, metric=getattr(args, "dist_metric", "cosine"))
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

    # S_all has served its purpose: every slice that needs it has been taken, and on the
    # primary dump it is a full gigabyte. Released here rather than at function exit,
    # because everything below -- the shrinkage search, both tables, the competitors --
    # allocates, and this is the largest single array that no longer has a reader.
    del S_all

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
    # --drop-features answers the review's question about individual descriptors (does the
    # head bias matter?) without inventing a new feature_group for each combination. An
    # unknown name is an error: silently dropping nothing would make the ablation look
    # like it ran.
    for d in (getattr(args, "drop_features", None) or ()):
        if d not in names:
            raise ValueError("--drop-features {!r} is not a descriptor; available: {}"
                             .format(d, names))
        feats = [f for f in feats if f != d]
    grp = getattr(args, "feature_group", "all")
    if grp == "distance":                 # ablasi E4: hanya jarak
        feats = [f for f in feats if "knn" in f]
    elif grp == "prevalence":
        feats = [f for f in feats if f == "log_prevalence"]
    elif grp == "no_prevalence":
        feats = [f for f in feats if f != "log_prevalence"]
    if not feats:
        raise ValueError("feature_group {!r} left no features from {}".format(grp, names))

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
                    score_matrix_recal=S_recal, labels_recal=y_recal,
                    features=feats, stat=args.stat, class_counts=cnt_full,
                    lam_override=getattr(args, "lam_override", None),
                    n_star_rule=getattr(args, "n_star_rule", "oos"),
                    recalibrate=not getattr(args, "no_recalibrate", False),
                    seed=args.seed)
    t = model.thresholds()

    # --dump-fit: the arrays behind the method figure, saved from THIS run so the figure and
    # the tables cannot disagree about the split, the seed or the fit.
    dump_fit = getattr(args, "dump_fit", None)
    if dump_fit:
        d_hat_all = model.gtheta.predict(np.asarray(Phi, float))
        np.savez_compressed(
            dump_fit,
            delta_obs=np.asarray(delta_obs, float),
            delta_hat=np.asarray(d_hat_all, float),
            Phi=np.asarray(Phi, float),
            # NOT dtype=object: that pickles, and a dump written under numpy 2
            # then fails to load under numpy 1 with 'No module named numpy._core'
            feature_names=np.array([str(n) for n in names]),
            seen=np.asarray(seen, int),
            heldout=np.asarray(heldout, int),
            n_per_class=np.asarray(n_per_class, int),
            class_counts=np.asarray(cnt_full, float),
            q_global=float(q_global),
            lam=float(model.lam),
            offset=float(model.offset),
            n_cal=int(args.n_cal),
            alpha=float(args.alpha),
        )
        print("  dump-fit -> {} ({} kelas, {} fitur)".format(
            dump_fit, len(delta_obs), len(names)), flush=True)

    # RESTRICT TO CLASSES WHERE THE METRIC IS MEASURABLE AT ALL.
    #
    # The evaluation-depth sweep showed the oracle ceiling itself falls to zero below about
    # 35 evaluation rows per class: no method, not even one holding the test labels, can
    # improve worst-class coverage there. On Pl@ntNet the MEDIAN is 3 -- but a median is not
    # a floor, and its head classes have far more. Keeping only classes that clear the
    # threshold turns "the method fails on long-tail data" into a question that can actually
    # be answered on long-tail data, and it is the falsifiable half of that explanation: if
    # the effect appears here, evaluation depth was the boundary; if it does not, the
    # explanation is incomplete and that is what gets written.
    n_dropped_thin = 0
    if getattr(args, "min_eval_rows", None):
        per_ev = np.bincount(y_ev, minlength=K)
        thick = np.where(per_ev >= int(args.min_eval_rows))[0]
        n_dropped_thin = int(K - len(thick))
        seen = np.intersect1d(seen, thick)
        heldout = np.intersect1d(heldout, thick)
        if not len(heldout):
            raise ValueError(
                "min_eval_rows={} leaves no held-out class: {} of {} classes have that "
                "many evaluation rows. Lower it, or accept that this slice cannot measure "
                "per-class coverage at all.".format(
                    args.min_eval_rows, len(thick), K))

    res = {
        "n_classes": K, "n_seen": int(len(seen)), "n_heldout": int(len(heldout)),
        "min_eval_rows": getattr(args, "min_eval_rows", None),
        "classes_dropped_too_thin_to_measure": n_dropped_thin,
        "frac_recal": frac_recal,
        "recal_rows": int(len(i_recal)),
        "dist_metric": getattr(args, "dist_metric", "cosine"),
        "knn_ks_requested": list(ks_req),
        "dropped_features": list(getattr(args, "drop_features", None) or ()),
        "split_sizes": {"desc": int(len(i_desc)), "cal_seen": int(len(i_cal_seen)),
                        "eval": int(len(y_ev))},
        "eval_from_separate_dump": bool(used_separate_eval),
        "subsample_frac_of_dump": float(len(y_all) / n_dump),
        "q_global": q_global, "score": score,
        "knn_ks_used": list(knn_ks), "knn_ks_dropped_K_too_small": list(knn_dropped),
        "ablation": {"cal_depth": getattr(args, "cal_depth", None),
                     "cal_depth_classes_capped": int(cap_bound),
                     "eval_depth": getattr(args, "eval_depth", None),
                     "cal_depth_binding": bool(cap_bound > 0.5 * len(seen)),
                     "lam_override": getattr(args, "lam_override", None),
                     "n_star_rule": getattr(args, "n_star_rule", "oos"),
                     "recalibrate": not getattr(args, "no_recalibrate", False),
                     "feature_group": getattr(args, "feature_group", "all")},
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
    }

    # Published competitors, fitted ONCE over the full label space and then read inside
    # both tables. Fitting per table would let each of them see a different calibration
    # slice, which is not what any of them does in deployment.
    comps, comp_errs, comp_chatter = {}, {}, {}
    want_comp = getattr(args, "competitors", False)
    if args.ccc_root and want_comp:
        try:
            comps, comp_errs, comp_chatter = _competitor_thresholds(
                args.ccc_root, S_cal, y_cal, K, args.alpha, args.seed,
                P_cal=(1.0 - S_cal) if score == "thr" else None,
                class_counts=cnt_full, tmp_dir=args.reports_dir)
        except Exception as e:                                       # noqa: BLE001
            comp_errs = {"_import": type(e).__name__ + ": " + str(e)[:300]}
            comp_chatter = {}
    res["competitor_errors"] = comp_errs
    res["competitor_stdout"] = {k: {"n_lines": v["n_lines"], "sample": v["sample"]}
                                for k, v in comp_chatter.items()}
    res["competitors_enabled"] = bool(args.ccc_root and want_comp)
    res["competitor_candidates"] = {k: [lb for lb, _ in v] for k, v in comps.items()}
    # RC3P needs the raw softmax as well as the score matrix, and softmax is only
    # recoverable from the score under THR. Said out loud rather than skipped in silence.
    if score != "thr":
        res["competitor_notes"] = ("rc3p omitted: it needs raw softmax, which is not "
                                   "recoverable from the {} score".format(score))

    res["table_1_seen"] = _one_table(S_ev, y_ev, seen, q_global, t, args.stat,
                                     counts=cnt_full, alpha=args.alpha,
                                     competitors=comps)
    if len(heldout):
        res["table_2_heldout"] = _one_table(S_ev, y_ev, heldout, q_global, t, args.stat,
                                            counts=cnt_full, alpha=args.alpha,
                                            competitors=comps)

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
    p.add_argument("--min-eval-rows", type=int, default=None,
                   help="keep only classes with this many EVALUATION rows; the falsifiable "
                        "half of the evaluation-depth explanation")
    p.add_argument("--eval-depth", type=int, default=None,
                   help="cap EVALUATION rows per class; the mirror of --cal-depth")
    p.add_argument("--competitors", action="store_true",
                   help="fit the published competitors too. " + COMPETITOR_COST)
    p.add_argument("--score", default="thr", choices=sorted(SCORE_FNS),
                   help="nonconformity score; delta_y is defined on ITS distribution")
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
    p.add_argument("--cal-depth", type=int, default=None,
                   help="cap CAL rows per SEEN class -- the calibration-depth sweep axis")
    p.add_argument("--lam-override", type=float, default=None,
                   help="ablation: force lambda (0 = no correction, 1 = raw delta_hat)")
    p.add_argument("--n-star-rule", default="oos", choices=("oos", "objective", "mse"))
    p.add_argument("--no-recalibrate", action="store_true",
                   help="ablation: drop the marginal offset")
    p.add_argument("--feature-group", default="all",
                   choices=("all", "distance", "prevalence", "no_prevalence"))
    p.add_argument("--frac-recal", type=float, default=0.0,
                   help="PCC-split: fraction of each SEEN class's calibration rows held "
                        "back so the marginal offset c is fit on rows disjoint from "
                        "g_theta and lambda. 0 reuses the rows (the default everywhere "
                        "before 2026-08-19) and does NOT satisfy Proposition 1.")
    p.add_argument("--dist-metric", default="cosine", choices=("cosine", "euclidean"),
                   help="metric for the neighbour descriptors")
    p.add_argument("--knn-ks", default=None,
                   help="comma-separated neighbour counts, e.g. 1,5,10,50")
    p.add_argument("--drop-features", default=None,
                   help="comma-separated descriptor names to remove, e.g. w_bias")
    p.add_argument("--dump-fit", default=None,
                   help="write delta_obs, Phi, g_theta's predictions and the class split to "
                        "this .npz, for the method figure")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--print-json", action="store_true")
    a = p.parse_args(argv)

    if a.phi == "head" and not a.head_weights:
        p.error("--phi head requires --head-weights")
    a.knn_ks = (tuple(int(x) for x in a.knn_ks.split(",") if x.strip())
                if a.knn_ks else None)
    a.drop_features = (tuple(x.strip() for x in a.drop_features.split(",") if x.strip())
                       if a.drop_features else ())
    if a.distance_holdout is None:
        # the held-out feature must exist in whatever ks were asked for, else the very
        # first check in run() rejects the configuration
        k0 = min(a.knn_ks) if a.knn_ks else 1
        a.distance_holdout = ("w_cos_knn_{}".format(k0) if a.phi == "head"
                              else "prof_knn_{}".format(k0))
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
