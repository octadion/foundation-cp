"""Synthetic tests for the Phase-1 gate instruments (§6.2, §6.3).

Runnable with pytest OR directly: `python pcc/tests/test_phase1_gate.py`.
Uses only numpy so it runs anywhere.

Design: plant known structure, assert the instruments recover it, AND assert the
negative controls read ~0 — a gate that can't return a negative is useless.
"""

from __future__ import annotations

import numpy as np

from pcc.targets.delta import delta_y, split_half_reliability
from pcc.descriptors.phi import build_descriptors
from pcc.eval.predictability import predictability


# ----------------------------- reliability (gate A) -----------------------------

def _make_scores(n_classes, n_per, signal_std, seed):
    """Scores with a per-class location shift mu_y (the stable signal). δ_y should
    track mu_y. signal_std=0 -> no signal (negative control)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, signal_std, n_classes)
    scores, classes = [], []
    for y in range(n_classes):
        scores.append(rng.normal(mu[y], 1.0, n_per))
        classes.append(np.full(n_per, y))
    return np.concatenate(scores), np.concatenate(classes), mu


def test_reliability_high_when_signal_present():
    s, c, mu = _make_scores(60, 300, signal_std=1.0, seed=0)
    out = split_half_reliability(s, c, 60, alpha=0.1, n_splits=60, seed=1)
    assert out["reliability_mean"] > 0.5, out["reliability_mean"]


def test_reliability_near_zero_without_signal():
    s, c, mu = _make_scores(60, 300, signal_std=0.0, seed=2)
    out = split_half_reliability(s, c, 60, alpha=0.1, n_splits=60, seed=3)
    # pure estimation noise -> essentially no reproducible per-class signal
    assert out["reliability_mean"] < 0.2, out["reliability_mean"]


# ----------------------------- descriptors -----------------------------

def test_build_descriptors_shapes_and_finiteness():
    rng = np.random.default_rng(4)
    n_classes, d, n_per = 30, 16, 40
    centers = rng.normal(0, 3, (n_classes, d))
    feats, cls, logits = [], [], []
    for y in range(n_classes):
        f = rng.normal(centers[y], 1.0, (n_per, d))
        feats.append(f); cls.append(np.full(n_per, y))
        lg = f @ centers.T  # crude logits: similarity to each center
        logits.append(lg)
    Phi, names = build_descriptors(np.vstack(feats), np.vstack(logits),
                                   np.concatenate(cls), n_classes)
    assert Phi.shape[0] == n_classes and Phi.shape[1] == len(names)
    assert np.isfinite(Phi).all(), "descriptor rows must be finite for present classes"
    for expect in ["mean_norm", "cos_knn_1", "cov_trace", "log_prevalence",
                   "logit_margin"]:
        assert expect in names, expect


# ----------------------------- predictability (gate B/C) -----------------------------

def _planted_delta(Phi, names, driver, seed, noise=0.3):
    rng = np.random.default_rng(seed)
    col = names.index(driver)
    x = Phi[:, col]
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    return 1.5 * x + rng.normal(0, noise, len(x))


def _descriptors_for_pred(seed=5, n_classes=120, d=16, n_per=40):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (n_classes, d))
    feats, cls, logits = [], [], []
    for y in range(n_classes):
        f = rng.normal(centers[y], 1.0, (n_per, d))
        feats.append(f); cls.append(np.full(n_per, y))
        logits.append(f @ centers.T)
    return build_descriptors(np.vstack(feats), np.vstack(logits),
                             np.concatenate(cls), n_classes)


def test_gateB_pass_when_delta_predictable_from_geometry():
    Phi, names = _descriptors_for_pred()
    delta = _planted_delta(Phi, names, "mean_norm", seed=6)
    out = predictability(Phi, delta, names, reliability=0.8, n_splits=80, seed=7)
    assert out["gate_B_pass"], out["r2_by_predictor"]["full"]
    # geometry driver (mean_norm) should beat log-prevalence-only -> gate C
    assert out["gate_C_detail"]["log_prevalence_only"]["full_beats_it"]


def test_gateB_fail_when_delta_is_noise():
    Phi, names = _descriptors_for_pred(seed=8)
    rng = np.random.default_rng(9)
    delta = rng.normal(0, 1, Phi.shape[0])  # pure noise, unrelated to geometry
    out = predictability(Phi, delta, names, reliability=0.8, n_splits=80, seed=10)
    assert not out["gate_B_pass"], out["normalized_full_r2"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"all {len(fns)} phase-1 gate tests passed")
