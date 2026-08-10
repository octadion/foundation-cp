"""φ(y) from a SCORE MATRIX only — no images, no GPU, no forward pass.

WHY THIS EXISTS. The embedding descriptors in `pcc.descriptors.phi` need a forward pass
over each class's images. That makes every gate-C evaluation cost an extraction run, and
on Pl@ntNet the gate turned out to be power-limited rather than signal-limited: 152
classes have a δ_y at n_cal=25, so a 4-way stratification leaves 38 classes per cell and
R² CIs up to 1.38 wide. No descriptor improvement can fix a class count.

The released score dumps are far richer in samples-per-class than the long-tail val
splits they were derived from:

    CCC imagenet          115,301 x 1000   ->  ~115 / class
    CCC cifar-100          30,000 x  100   ->   300 / class
    CCC places365         183,996 x  365   ->   504 / class
    CCC inaturalist-2021 1,324,900 x  633   -> 2,093 / class
    LTC plantnet cal       21,783 x 1081   -> median 3 / class

So a score-only descriptor family lets gate C be evaluated at 1000 classes instead of 38,
in minutes, with no download of images at all.

WHAT IT IS AND IS NOT. These are descriptors of the class's position in the OUTPUT
geometry — its typical confidence, its confusion neighbourhood, the cosine geometry of
mean softmax profiles — not of the embedding geometry. That is a genuinely different
φ(y), so a result here does NOT transfer automatically to the embedding descriptors; it
answers "does class-level geometry predict δ_y once the class count is adequate", which
is the question Pl@ntNet cannot answer at any power.

LEAK GUARD. Everything is computed from a DESCRIPTOR split that must be disjoint from the
calibration split producing δ_y and from the evaluation split. Computing φ from the same
rows that produce δ_y makes gate B circular. `build_output_descriptors` takes only the
descriptor rows and asserts nothing about the others — the caller owns the split, exactly
as `pcc.extract` does for the embedding path (descriptors from TRAIN, δ_y from val).
"""

from __future__ import annotations

import numpy as np

# Features whose value is fixed by the split construction rather than estimated from the
# class's samples, so they carry no sampling noise and must not be credited with
# stability the estimated features have to earn.
QUOTA_DETERMINED = ("n_eff", "log_prevalence")


def _row_entropy(P):
    Q = np.clip(P, 1e-12, 1.0)
    return -(Q * np.log(Q)).sum(axis=1)


def _top2_margin(P):
    part = np.partition(P, -2, axis=1)
    return part[:, -1] - part[:, -2]


def _true_rank(P, labels):
    """Rank of the true class, 0 = argmax. Counts strictly-greater scores, so ties do not
    inflate the rank."""
    true_score = P[np.arange(len(labels)), labels]
    return (P > true_score[:, None]).sum(axis=1)


def class_mean_profiles(P, labels, n_classes):
    """Mean softmax profile per class, [n_classes, n_classes]. NaN rows for absent
    classes so the caller can see them rather than silently getting zeros."""
    P = np.asarray(P, float)
    labels = np.asarray(labels, int)
    prof = np.full((n_classes, P.shape[1]), np.nan)
    counts = np.bincount(labels, minlength=n_classes)
    present = np.where(counts > 0)[0]
    # np.add.at is the vectorised scatter-add; a python loop over 1000 classes with
    # boolean masks over 115k rows is ~1000x more work.
    acc = np.zeros((n_classes, P.shape[1]))
    np.add.at(acc, labels, P)
    prof[present] = acc[present] / counts[present, None]
    return prof, counts


def profile_knn_distances(prof, ks=(1, 5, 10, 50)):
    """Cosine distance from each class's mean softmax profile to its k nearest OTHER
    class profiles. The output-space analogue of `phi.cosine_knn_matrix`."""
    prof = np.asarray(prof, float)
    ok = np.isfinite(prof).all(axis=1)
    X = prof[ok]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)          # exclude self
    dist = 1.0 - sim
    dist_sorted = np.sort(dist, axis=1)
    out = {}
    idx = np.where(ok)[0]
    for k in ks:
        col = np.full(len(prof), np.nan)
        kk = min(k, dist_sorted.shape[1])
        col[idx] = dist_sorted[:, :kk].mean(axis=1)
        out[f"prof_knn_{k}"] = col
    return out


def build_output_descriptors(P, labels, n_classes, *, knn_ks=(1, 5, 10, 50),
                             log_prevalence_from=None):
    """φ(y) for every class, from the descriptor split's score matrix alone.

    `P` is [n, n_classes] softmax rows, `labels` their true classes. `log_prevalence_from`
    supplies the counts that log-prevalence should reflect (normally the TRAIN class
    counts, not the descriptor split's own counts — otherwise log-prevalence just encodes
    the quota and the gate-C prevalence ablation tests nothing).

    Returns (Phi [n_classes, d], feature_names). Rows for absent classes are NaN, which
    downstream code already filters on.
    """
    P = np.asarray(P, float)
    labels = np.asarray(labels, int)
    if P.ndim != 2 or P.shape[1] != n_classes:
        raise ValueError(f"score matrix is {P.shape}, expected (n, {n_classes})")
    if len(labels) != len(P):
        raise ValueError("labels and score rows disagree in length")

    prof, counts = class_mean_profiles(P, labels, n_classes)
    ent = _row_entropy(P)
    marg = _top2_margin(P)
    rank = _true_rank(P, labels)
    conf = P[np.arange(len(labels)), labels]
    top1_hit = (rank == 0).astype(float)

    feat = {}

    def per_class_mean(v, name):
        acc = np.zeros(n_classes)
        np.add.at(acc, labels, v)
        col = np.full(n_classes, np.nan)
        present = counts > 0
        col[present] = acc[present] / counts[present]
        feat[name] = col
        return col

    conf_mean = per_class_mean(conf, "conf_mean")
    per_class_mean(ent, "entropy_mean")
    per_class_mean(marg, "margin_mean")
    per_class_mean(top1_hit, "frac_top1")
    per_class_mean(np.log1p(rank), "mean_log1p_rank")

    # spread of self-confidence within the class: a class can be easy on average but
    # bimodal, and δ_y is a TAIL statistic, so the spread is the more relevant moment.
    sq = np.zeros(n_classes)
    np.add.at(sq, labels, conf ** 2)
    sd = np.full(n_classes, np.nan)
    present = counts > 1
    sd[present] = np.sqrt(np.maximum(
        sq[present] / counts[present] - conf_mean[present] ** 2, 0.0))
    feat["conf_sd"] = sd

    # confusion geometry: how much probability mass leaks to OTHER classes, and how
    # concentrated that leakage is. A class with one strong confuser is a different
    # geometric situation from one that leaks diffusely, and δ_y should differ.
    off = np.array(prof)
    rows = np.arange(n_classes)
    finite = np.isfinite(prof).all(axis=1)
    off[rows[finite], rows[finite]] = 0.0
    leak_max = np.full(n_classes, np.nan)
    leak_top5 = np.full(n_classes, np.nan)
    leak_ent = np.full(n_classes, np.nan)
    if finite.any():
        sub = off[finite]
        srt = np.sort(sub, axis=1)
        leak_max[finite] = srt[:, -1]
        leak_top5[finite] = srt[:, -5:].sum(axis=1)
        tot = sub.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        leak_ent[finite] = _row_entropy(sub / tot)
    feat["leak_max"] = leak_max
    feat["leak_top5"] = leak_top5
    feat["leak_entropy"] = leak_ent

    feat.update(profile_knn_distances(prof, ks=knn_ks))

    feat["n_eff"] = counts.astype(float)
    src = counts if log_prevalence_from is None else np.asarray(log_prevalence_from)
    feat["log_prevalence"] = np.log(np.maximum(src, 1)).astype(float)

    names = sorted(feat)
    Phi = np.column_stack([feat[n] for n in names])
    return Phi, names


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, n = 60, 6000
    y = rng.integers(0, K, n)
    logits = rng.normal(0, 1, (n, K))
    logits[np.arange(n), y] += rng.uniform(1, 5, K)[y]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    P = e / e.sum(axis=1, keepdims=True)

    Phi, names = build_output_descriptors(P, y, K)
    assert Phi.shape == (K, len(names)), Phi.shape
    assert np.isfinite(Phi).all(), "unexpected NaN with every class present"
    print(f"  {len(names)} features: {names}")

    # the planted per-class difficulty must show up in conf_mean, or the descriptors
    # are not measuring anything about the class
    mu = rng.uniform(1, 5, K)  # not the same draw; use the realised accuracy instead
    acc = Phi[:, names.index("frac_top1")]
    cm = Phi[:, names.index("conf_mean")]
    assert np.corrcoef(acc, cm)[0, 1] > 0.8, "conf_mean should track class accuracy"
    print(f"  corr(frac_top1, conf_mean) = {np.corrcoef(acc, cm)[0,1]:+.3f}  OK")

    # absent classes must be NaN, not zero
    keep = y != 7
    Phi2, _ = build_output_descriptors(P[keep], y[keep], K)
    assert np.isnan(Phi2[7]).any(), "absent class must be NaN"
    print("  absent class -> NaN (not silently 0)  OK")

    # order independence: shuffling rows must not change the descriptors
    perm = rng.permutation(n)
    Phi3, _ = build_output_descriptors(P[perm], y[perm], K)
    assert np.allclose(Phi, Phi3, equal_nan=True, atol=1e-10), "row order changed Phi"
    print("  row-order independent  OK")
    print("all output-space descriptor tests passed")
