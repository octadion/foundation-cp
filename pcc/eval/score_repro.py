"""Permutation-invariant checkpoint->score reproduction gate (Phase 0).

The released LTC scores are in an unrecoverable shuffled order (loaders use
shuffle=True), so we compare our regenerated scores to the released ones by
order-independent statistics. Criteria are pre-registered in
reports/phase0_checkpoint_gate.md — this module implements them and returns a
PASS/FAIL verdict; it does not decide anything on its own.

G1 accuracy, G3 true-prob curve, G4 label-multiset are invariant to a consistent
class-index relabeling. G2 nearest-neighbor row matching is NOT — it needs the
score COLUMNS to share a convention (guaranteed for iNaturalist category_ids;
verify for Pl@ntNet).
"""

from __future__ import annotations

import hashlib

import numpy as np


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def top1_accuracy(softmax: np.ndarray, labels: np.ndarray) -> float:
    return float((softmax.argmax(axis=1) == labels).mean())


def true_class_prob_curve(softmax, labels, n_grid: int = 512):
    """Sorted per-sample probability assigned to the true class, resampled to a
    fixed grid so two sets of different N are comparable."""
    p_true = softmax[np.arange(len(labels)), labels]
    p_true = np.sort(p_true)
    grid = np.linspace(0, 1, n_grid)
    return np.interp(grid, np.linspace(0, 1, len(p_true)), p_true)


def nn_match_distances(sample_rows: np.ndarray, reference: np.ndarray,
                       tol: float = 1e-4, cand_block: int = 4096) -> np.ndarray:
    """For each row of `sample_rows`, the min L-infinity distance to any row of
    `reference` — EXACT for distances <= tol, and memory-bounded.

    A naive `abs(reference[None] - chunk[:, None]).max(2)` materialises a
    `[block, N_ref, C]` array. On Pl@ntNet that is 256 x 21,783 x 1,081 float64 =
    **48 GB**, which crashed the Colab session. The blocking was on the wrong axis.

    Pruning bound used instead: the row max is 1-Lipschitz in L-infinity, i.e.
    `|max(a) - max(b)| <= max_c |a_c - b_c|`. So any reference row whose max
    probability differs from the query's by more than `tol` CANNOT be within `tol`.
    Sorting the reference by row max and binary-searching the window
    `[q_max - tol, q_max + tol]` therefore prunes without false negatives.

    GUARANTEE, stated precisely: the returned value is EXACT whenever it is
    <= tol. When it is > tol, it is only guaranteed to BE > tol — it is not a bound
    in either direction (an empty window yields a Lipschitz lower bound; a
    non-empty window that misses the true nearest neighbour yields an upper bound).
    That is exactly what G2 needs, since G2 asks what FRACTION of rows lie within
    tol. Verified against brute force: identical within/outside-tol classification,
    and bit-exact values for the within-tol rows.
    """
    ref_max = reference.max(axis=1)
    order = np.argsort(ref_max, kind="stable")
    sorted_max = ref_max[order]
    n_ref = len(reference)

    def _best_within(x, r):
        """Min L-inf over reference rows whose row-max is within r of x's."""
        qm = float(x.max())
        lo = int(np.searchsorted(sorted_max, qm - r, side="left"))
        hi = int(np.searchsorted(sorted_max, qm + r, side="right"))
        cand = order[lo:hi]
        if len(cand) == 0:
            return np.inf, 0
        best = np.inf
        for s in range(0, len(cand), cand_block):
            blk = reference[cand[s:s + cand_block]]
            best = min(best, float(np.abs(blk - x[None, :]).max(axis=1).min()))
        return best, len(cand)

    out = np.empty(len(sample_rows))
    for i, x in enumerate(sample_rows):
        # PROGRESSIVE WIDENING WITH A CORRECTNESS CERTIFICATE.
        #
        # A single window of radius `tol` was WRONG and produced a badly misleading
        # result on real data (2026-08-04): the true twin of a row differed in
        # row-max by ~3e-4, i.e. THREE TIMES the 1e-4 window, so the twin was pruned
        # away for every non-saturated row and a large distance was reported. Only
        # saturated rows (row-max ~1.0 on both sides) survived the window, which is
        # exactly the "19.5% matched, all of them confident" pattern observed.
        #
        # Certificate: if searching radius r yields best distance d <= r, then every
        # row outside the window has |max difference| > r >= d, and since
        # L-inf >= |max difference|, its distance exceeds d. So d is the TRUE
        # minimum. Widening until that condition holds gives exact distances at
        # bounded cost.
        best = np.inf
        for r in (tol, 10 * tol, 100 * tol, 1e-2, 1e-1, np.inf):
            best, n_cand = _best_within(x, r)
            if np.isfinite(best) and best <= r:
                break                      # certified: this is the true minimum
            if not np.isfinite(r) or n_cand >= n_ref:
                break                      # searched everything
        out[i] = best
    return out


def label_multiset_equal(y_mine: np.ndarray, y_rel: np.ndarray) -> bool:
    vm = np.bincount(y_mine)
    vr = np.bincount(y_rel)
    n = max(len(vm), len(vr))
    vm = np.pad(vm, (0, n - len(vm)))
    vr = np.pad(vr, (0, n - len(vr)))
    return bool(np.array_equal(vm, vr))


def evaluate_gate(mine_softmax, mine_labels, rel_softmax, rel_labels, *,
                  nn_subsample: int = 1000, seed: int = 42,
                  tol_acc: float = 0.002, tol_nn_linf: float = 1e-4,
                  tol_nn_median: float = 1e-5, tol_curve: float = 1e-3,
                  check_nn: bool = True, acc_sigma: float = 3.0,
                  full_mine_labels=None):
    """Return a dict with per-criterion results and an overall 'PASS'/'FAIL'.

    check_nn=False skips G2 (use when the class-column convention is not known to
    match, e.g. Pl@ntNet before confirmation); G2 is then reported 'skipped' and
    does not gate — G1/G3/G4 still decide.
    """
    acc_mine = top1_accuracy(mine_softmax, mine_labels)
    acc_rel = top1_accuracy(rel_softmax, rel_labels)

    # G1 TOLERANCE CORRECTION (2026-08-04). The pre-registered 0.002 was
    # MIS-DERIVED: it assumed a like-for-like comparison, but we compare a
    # SUBSAMPLE accuracy against the full released set, so binomial sampling noise
    # alone is sqrt(p(1-p)/n_sub) — at n=3000, p=0.8 that is 0.0073, i.e. larger
    # than the whole tolerance. A correct checkpoint would have failed most of the
    # time. The tolerance is therefore derived from the subsample size rather than
    # fixed: max(tol_acc, acc_sigma * SE). This is a specification fix, not a
    # loosening to pass: the failure modes G1 guards (preprocessing drift,
    # normalization mismatch, wrong checkpoint) move accuracy by whole percentage
    # points, far beyond 3 SE.
    n_sub = len(mine_labels)
    se_acc = float(np.sqrt(max(acc_mine * (1 - acc_mine), 1e-12) / max(n_sub, 1)))
    tol_acc_eff = max(tol_acc, acc_sigma * se_acc)
    g1 = abs(acc_mine - acc_rel) <= tol_acc_eff

    curve_mine = true_class_prob_curve(mine_softmax, mine_labels)
    curve_rel = true_class_prob_curve(rel_softmax, rel_labels)
    curve_maxdiff = float(np.abs(curve_mine - curve_rel).max())
    g3 = curve_maxdiff <= tol_curve

    # G4 MUST use the FULL reconstructed subset, not the forward-passed subsample.
    # `mine_labels` is a SUBSAMPLE (e.g. 3000 of 21,783), so its per-class counts can
    # never equal the released ones and G4 would fail even for a perfect checkpoint.
    # The label multiset needs NO forward pass — it comes straight from the split
    # reconstruction — so pass `full_mine_labels` and G4 becomes a real test of
    # whether ltc_cal_val_indices reproduced LTC's cal membership.
    g4_labels = mine_labels if full_mine_labels is None else np.asarray(full_mine_labels)
    g4_scope = "subsample (NOT meaningful)" if full_mine_labels is None else "full reconstructed subset"
    g4 = label_multiset_equal(g4_labels, rel_labels)

    result = {
        "G1_accuracy": {"acc_mine": acc_mine, "acc_rel": acc_rel,
                        "abs_diff": abs(acc_mine - acc_rel),
                        "n_subsample": int(n_sub), "se_binomial": se_acc,
                        "tol_requested": tol_acc, "tol_effective": tol_acc_eff,
                        "diff_in_sigmas": abs(acc_mine - acc_rel) / max(se_acc, 1e-12),
                        "pass": bool(g1)},
        "G3_true_prob_curve": {"max_abs_diff": curve_maxdiff, "tol": tol_curve,
                               "pass": bool(g3)},
        "G4_label_multiset": {"pass": bool(g4), "scope": g4_scope,
                             "n_mine": int(len(g4_labels)), "n_released": int(len(rel_labels))},
    }

    gates = [g1, g3, g4]
    if check_nn and mine_softmax.shape[1] == rel_softmax.shape[1]:
        rng = np.random.default_rng(seed)
        k = min(nn_subsample, len(mine_softmax))
        idx = rng.choice(len(mine_softmax), k, replace=False)
        dists = nn_match_distances(mine_softmax[idx], rel_softmax, tol=tol_nn_linf)
        frac_ok = float((dists <= tol_nn_linf).mean())
        median = float(np.median(dists))
        g2 = (frac_ok >= 0.99) and (median <= tol_nn_median)
        result["G2_nn_match"] = {"frac_within_tol": frac_ok,
                                 "median_dist": median, "tol_linf": tol_nn_linf,
                                 "tol_median": tol_nn_median, "k": k,
                                 "pass": bool(g2)}
        gates.append(g2)
    else:
        result["G2_nn_match"] = {"pass": None, "note": "skipped (convention unconfirmed or shape mismatch)"}

    result["verdict"] = "PASS" if all(gates) else "FAIL"
    return result
