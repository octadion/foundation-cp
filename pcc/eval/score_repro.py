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
                       block: int = 256) -> np.ndarray:
    """For each row in `sample_rows`, the min L-inf distance to any row in
    `reference`. Blocked to bound memory."""
    out = np.empty(len(sample_rows))
    for s in range(0, len(sample_rows), block):
        chunk = sample_rows[s:s + block]                       # [b, C]
        # L-inf over classes, min over reference rows
        d = np.abs(reference[None, :, :] - chunk[:, None, :]).max(axis=2)
        out[s:s + block] = d.min(axis=1)
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
                  check_nn: bool = True):
    """Return a dict with per-criterion results and an overall 'PASS'/'FAIL'.

    check_nn=False skips G2 (use when the class-column convention is not known to
    match, e.g. Pl@ntNet before confirmation); G2 is then reported 'skipped' and
    does not gate — G1/G3/G4 still decide.
    """
    acc_mine = top1_accuracy(mine_softmax, mine_labels)
    acc_rel = top1_accuracy(rel_softmax, rel_labels)
    g1 = abs(acc_mine - acc_rel) <= tol_acc

    curve_mine = true_class_prob_curve(mine_softmax, mine_labels)
    curve_rel = true_class_prob_curve(rel_softmax, rel_labels)
    curve_maxdiff = float(np.abs(curve_mine - curve_rel).max())
    g3 = curve_maxdiff <= tol_curve

    g4 = label_multiset_equal(mine_labels, rel_labels)

    result = {
        "G1_accuracy": {"acc_mine": acc_mine, "acc_rel": acc_rel,
                        "abs_diff": abs(acc_mine - acc_rel), "tol": tol_acc,
                        "pass": bool(g1)},
        "G3_true_prob_curve": {"max_abs_diff": curve_maxdiff, "tol": tol_curve,
                               "pass": bool(g3)},
        "G4_label_multiset": {"pass": bool(g4)},
    }

    gates = [g1, g3, g4]
    if check_nn and mine_softmax.shape[1] == rel_softmax.shape[1]:
        rng = np.random.default_rng(seed)
        k = min(nn_subsample, len(mine_softmax))
        idx = rng.choice(len(mine_softmax), k, replace=False)
        dists = nn_match_distances(mine_softmax[idx], rel_softmax)
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
