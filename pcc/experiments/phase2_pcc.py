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


# ------------------------------------------------------------------- evaluation
def _one_table(S_ev, y_ev, classes, q_global, thresholds_full, stat):
    """Equity of PCC vs the marginal threshold, inside one label space, matched size."""
    S_sub, y_sub, K_sub, ids = restrict_to_classes(S_ev, y_ev, classes)
    base = np.full(K_sub, float(q_global))
    target = avg_set_size_at_shift(S_sub, base, 0.0)
    e_base = equity_at_matched_size(S_sub, y_sub, K_sub, base, target)
    e_pcc = equity_at_matched_size(S_sub, y_sub, K_sub,
                                   np.asarray(thresholds_full, float)[ids], target)
    return {
        "n_classes": int(K_sub), "n_rows": int(len(y_sub)),
        "target_avg_set_size": float(target),
        "uncorrected": e_base, "pcc": e_pcc,
        "delta": {k: e_pcc[k] - e_base[k] for k in e_base if k != "shift"},
        "size_matched": bool(abs(e_pcc["avg_set_size"] - e_base["avg_set_size"]) < 1e-2),
        "pass": bool(e_pcc[stat] - e_base[stat] > 0),
    }


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
    S_ev = thr_lac(S_all[i_eval])
    y_ev = y_all[i_eval]

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

    if args.distance_holdout not in names:
        raise ValueError("--distance-holdout {!r} is not a descriptor of the {!r} family; "
                         "available: {}".format(args.distance_holdout, args.phi, names))
    feats = [f for f in names if f != args.distance_holdout]
    model = fit_pcc(Phi, names, delta_obs, n_per_class, q_global, args.alpha,
                    score_matrix_fit=S_cal, labels_fit=y_cal, train_classes=seen,
                    features=feats, stat=args.stat, seed=args.seed)
    t = model.thresholds()

    res = {
        "n_classes": K, "n_seen": int(len(seen)), "n_heldout": int(len(heldout)),
        "split_sizes": {"desc": int(len(i_desc)), "cal_seen": int(len(i_cal_seen)),
                        "eval": int(len(i_eval))},
        "subsample_frac_of_dump": float(len(y_all) / n_dump),
        "q_global": q_global,
        "knn_ks_used": list(knn_ks), "knn_ks_dropped_K_too_small": list(knn_dropped),
        "delta_obs_defined": int(np.isfinite(delta_obs).sum()),
        "pcc": {"lambda": model.lam, "n_star": model.n_star, "offset": model.offset,
                "gtheta_mse": model.threshold_rule["gtheta_mse"],
                "noise_curve": model.threshold_rule["noise_curve"],
                "blend": {k: v for k, v in model.blend.items() if k != "delta"
                          and k != "used_observed"},
                "lambda_curve_train": model.lambda_selection["curve"],
                "features": list(model.gtheta.feature_names),
                "provenance": model.provenance},
        "table_1_seen": _one_table(S_ev, y_ev, seen, q_global, t, args.stat),
    }
    if len(heldout):
        res["table_2_heldout"] = _one_table(S_ev, y_ev, heldout, q_global, t, args.stat)

    if args.ccc_root:
        try:
            res["baselines_table_1"] = _baselines_table1(
                args.ccc_root, S_cal, y_cal, S_ev, y_ev, seen, args.alpha, args.seed)
        except Exception as e:                                   # noqa: BLE001
            res["baselines_table_1"] = {"error": type(e).__name__ + ": " + str(e)}
    return res


def verdict(res: dict, stat: str) -> str:
    t2 = res.get("table_2_heldout")
    t1 = res["table_1_seen"]
    if t2 is None:
        return "TIDAK DAPAT DINILAI (tidak ada kelas held-out)"
    won_2 = t2["delta"][stat] > 0
    kept_1 = t1["delta"][stat] > -0.01
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
        print("  {} ({} kelas, size {:.3f})".format(
            tag, tb["n_classes"], tb["target_avg_set_size"]))
        print("    {:6s} global {:+.4f} -> PCC {:+.4f}  delta {:+.4f}{}".format(
            a.stat, tb["uncorrected"][a.stat], tb["pcc"][a.stat], tb["delta"][a.stat],
            "" if tb["size_matched"] else "   [UKURAN TIDAK COCOK — jangan dibaca]"))
    print("  VERDICT:", concl)
    print("  laporan:", path)
    if a.print_json:
        print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
