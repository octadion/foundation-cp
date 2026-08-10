"""φ(y) from the CLASSIFIER HEAD's weight vector — zero sampling noise, no images.

WHY THIS EXISTS: it is the only descriptor family so far that is NOT derived from the
score distribution that defines δ_y.

The output-space family (`pcc.descriptors.output_space`) passed gate B/C on ImageNet with
R² = 0.497, but that result cannot establish the project's geometric claim, and the reason
is structural rather than statistical: φ there is built from the score matrix, and δ_y is
built from the score matrix of the *same model* on disjoint samples. `conf_mean` on the
descriptor split and `q̂_y` on the calibration split are two estimates of the same class's
score distribution. Predicting one from the other is near-guaranteed whenever a class has
any stable score distribution at all — closer to distributional estimation than to
geometric extrapolation. Measured on that run: the class-similarity feature `prof_knn_1`
alone explained only ~0.155 of the 0.497; the rest came from direct score summaries.

A classifier head's row `w_y` breaks that circularity on three counts:

1. **It is a PARAMETER, not a sample estimate.** No sampling noise, so reliability is 1.0
   by construction — see EXACT_BY_CONSTRUCTION below for why that must not be read as
   descriptor quality the way an estimated feature's stability is.
2. **It needs no images and no forward pass.** A `(K, d)` matrix from a checkpoint.
3. **It can come from a DIFFERENT model than the one that produced the scores.** Using an
   independent encoder makes φ genuinely exogenous to δ_y, which is exactly what the
   geometric claim requires. On ImageNet the released CCC scores come from SimCLRv2 +
   linear probe while a torchvision ResNet-50 head is a separate model entirely.

WHAT IT IS NOT. `w_y` describes the class's position in the *decision* geometry of its own
classifier, not the distribution of images in feature space. It is exogenous to δ_y when
the two models differ, but it is still a model artefact, not a property of the data. State
which checkpoint it came from in every table.
"""

from __future__ import annotations

import numpy as np

# Features that are exact functions of model parameters. A split-half stability screen
# returns 1.0 for these no matter how the data is split, because they do not depend on the
# data at all. That 1.0 is a tautology, not evidence of a well-estimated descriptor, so it
# must be excluded from any r_phi that is meant to bound achievable R² — pooling it in
# would inflate the ceiling and make the normalized R² look worse than it is.
EXACT_BY_CONSTRUCTION = ("w_norm", "w_bias", "w_cos_to_mean", "w_cos_knn_1",
                         "w_cos_knn_5", "w_cos_knn_10", "w_cos_knn_50",
                         "w_neigh_spread", "w_margin_nearest")


def _unit_rows(W):
    n = np.linalg.norm(W, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return W / n


def head_cos_knn(W, ks=(1, 5, 10, 50)):
    """Mean cosine distance from each class's weight vector to its k nearest OTHER class
    weight vectors. The decision-geometry analogue of `phi.cosine_knn_matrix`.

    Returned alongside `w_neigh_spread` (sd over the 50 nearest) because a class with one
    close competitor is a different geometric situation from one uniformly crowded, and
    δ_y should differ between them — a mean alone cannot separate those.
    """
    W = np.asarray(W, float)
    Wn = _unit_rows(W)
    sim = Wn @ Wn.T
    np.fill_diagonal(sim, -np.inf)          # exclude self
    dist = 1.0 - sim
    order = np.sort(dist, axis=1)
    out = {}
    for k in ks:
        kk = min(k, order.shape[1])
        out[f"w_cos_knn_{k}"] = order[:, :kk].mean(axis=1)
    kk = min(50, order.shape[1])
    out["w_neigh_spread"] = order[:, :kk].std(axis=1)
    out["w_margin_nearest"] = order[:, 0]   # distance to the single closest rival
    return out


def build_head_descriptors(W, b=None, *, knn_ks=(1, 5, 10, 50)):
    """φ(y) from a classifier head.

    `W` is `(n_classes, d)` — e.g. `state_dict['fc.weight']` of a torchvision ResNet-50,
    or `model.fc.weight.detach().cpu().numpy()`. `b` is the optional bias `(n_classes,)`.

    Returns `(Phi [n_classes, n_features], feature_names)`. Every column is an exact
    function of `W`/`b`, so there are no NaN rows and no per-class sample requirement:
    a class with zero labelled examples anywhere still gets a full descriptor row. That
    is the property the extrapolation claim needs.
    """
    W = np.asarray(W, float)
    if W.ndim != 2:
        raise ValueError(f"W must be (n_classes, d), got {W.shape}")
    K = W.shape[0]

    feat = {"w_norm": np.linalg.norm(W, axis=1)}
    feat["w_bias"] = (np.zeros(K) if b is None else np.asarray(b, float).reshape(K))

    mean_w = W.mean(axis=0)
    mn = np.linalg.norm(mean_w)
    if mn == 0:
        feat["w_cos_to_mean"] = np.zeros(K)
    else:
        feat["w_cos_to_mean"] = 1.0 - (_unit_rows(W) @ (mean_w / mn))

    feat.update(head_cos_knn(W, ks=knn_ks))

    names = sorted(feat)
    Phi = np.column_stack([feat[n] for n in names])
    if not np.isfinite(Phi).all():
        raise ValueError("non-finite head descriptor — check W for NaN/inf rows")
    return Phi, names


def load_torchvision_resnet50_head():
    """`(W, b)` from torchvision's ImageNet ResNet-50. ~100 MB download, no GPU.

    Deliberately a supervised model INDEPENDENT of the SimCLRv2 + linear probe that
    produced the released CCC ImageNet scores: that independence is the point, since it
    makes φ exogenous to δ_y.
    """
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    W = m.fc.weight.detach().cpu().numpy()
    b = m.fc.bias.detach().cpu().numpy() if m.fc.bias is not None else None
    del m
    return W, b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, d = 200, 64

    # planted structure: classes on a ring have close neighbours, scattered ones do not
    W = rng.normal(0, 1, (K, d))
    W[:50] = W[0] + rng.normal(0, 0.05, (50, d))     # a tight cluster
    Phi, names = build_head_descriptors(W, b=rng.normal(0, 0.1, K))
    assert Phi.shape == (K, len(names)), Phi.shape
    assert np.isfinite(Phi).all()
    print(f"  {len(names)} features: {names}")

    j = names.index("w_cos_knn_1")
    clustered = Phi[:50, j].mean()
    scattered = Phi[50:, j].mean()
    assert clustered < scattered, (clustered, scattered)
    print(f"  w_cos_knn_1: klaster {clustered:.4f} < tersebar {scattered:.4f}  OK")

    # exactness: no data split enters, so recomputation is bit-identical
    Phi2, _ = build_head_descriptors(W, b=None)
    j2 = names.index("w_norm")
    assert np.array_equal(Phi[:, j2], Phi2[:, j2]), "w_norm must not depend on bias"
    print("  w_norm tidak bergantung bias; deskriptor eksak  OK")

    # every class gets a row even with no data anywhere — the extrapolation property
    assert Phi.shape[0] == K and np.isfinite(Phi).all()
    print(f"  {K}/{K} kelas punya baris deskriptor lengkap tanpa satu pun sampel  OK")

    # spread must separate "one close rival" from "uniformly crowded"
    W2 = rng.normal(0, 1, (K, d))
    W2[1] = W2[0] + rng.normal(0, 0.01, d)           # class 0 has exactly one twin
    P2, n2 = build_head_descriptors(W2)
    s = P2[:, n2.index("w_neigh_spread")]
    assert s[0] > np.median(s), (s[0], np.median(s))
    print(f"  w_neigh_spread: satu-kembar {s[0]:.4f} > median {np.median(s):.4f}  OK")
    print("all head-weight descriptor tests passed")
