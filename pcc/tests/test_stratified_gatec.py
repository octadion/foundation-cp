"""Stratified gate C must break the prevalence confound (Amendment 5).

On Pl@ntNet, descriptor stability rises monotonically with prevalence (0.684 in the
rarest quartile vs 0.922 in the densest, spread 0.238) because rare classes hold only
2-7 images. Geometry descriptors are therefore most accurate exactly where prevalence
is highest, so a POOLED "geometry beats log-prevalence" comparison is confounded.

Conditioning on the confound is the remedy: inside one prevalence quartile, prevalence
barely varies and descriptor quality is roughly uniform.

The test plants each causal story and checks the stratified verdict follows it, while
confirming the POOLED R2 cannot tell them apart.

Runnable with pytest OR directly: `python pcc/tests/test_stratified_gatec.py`.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.predictability import predictability, predictability_by_stratum

NAMES = ["geom_a", "geom_b", "log_prevalence", "cos_knn_1"]


def _fixture(driver, K=800, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.clip((rng.pareto(1.1, K) * 8 + 2).astype(int), 2, 400)
    lp = np.log(counts)
    g, g2, ck = (rng.normal(0, 1, K) for _ in range(3))
    Phi = np.column_stack([g, g2, lp, ck])
    if driver == "prevalence":
        delta = 1.5 * (lp - lp.mean()) / lp.std() + rng.normal(0, 0.4, K)
    else:
        delta = 1.5 * (g - g.mean()) / g.std() + rng.normal(0, 0.4, K)
    return Phi, delta, counts


def _n_strata_beating_prevalence(Phi, delta, counts):
    r = predictability_by_stratum(Phi, delta, NAMES, counts, n_splits=60, seed=1,
                                  n_strata=4)
    return r["n_strata_full_beats_ablation"].get("log_prevalence_only", 0), \
        r["n_strata_reported"]


def test_stratified_rejects_a_prevalence_only_story():
    """delta driven ONLY by prevalence: geometry must beat prevalence in NO stratum."""
    Phi, delta, counts = _fixture("prevalence")
    n_beat, n_rep = _n_strata_beating_prevalence(Phi, delta, counts)
    assert n_rep >= 3, f"expected several strata, got {n_rep}"
    assert n_beat == 0, f"geometry 'beat' prevalence in {n_beat}/{n_rep} strata for a "
    f"purely prevalence-driven delta"


def test_stratified_accepts_a_geometry_story():
    """delta driven by geometry, independent of prevalence: geometry must beat
    prevalence in essentially every stratum."""
    Phi, delta, counts = _fixture("geometry")
    n_beat, n_rep = _n_strata_beating_prevalence(Phi, delta, counts)
    assert n_beat >= n_rep - 1, f"geometry only beat prevalence in {n_beat}/{n_rep} strata"


def test_pooled_analysis_cannot_distinguish_the_two_stories():
    """This is WHY stratification is the primary form: pooled R2 is nearly identical
    for the two causal stories, so it carries almost no information about which is
    true."""
    r2 = {}
    for driver in ("prevalence", "geometry"):
        Phi, delta, counts = _fixture(driver)
        res = predictability(Phi, delta, NAMES, n_splits=60, seed=1)
        r2[driver] = res["r2_by_predictor"]["full"]["mean"]
    assert abs(r2["prevalence"] - r2["geometry"]) < 0.10, (
        f"pooled R2 separated the stories ({r2}); if this starts failing the "
        f"justification for stratifying should be revisited")


def test_thin_strata_are_skipped_not_merged():
    Phi, delta, counts = _fixture("geometry", K=120)
    r = predictability_by_stratum(Phi, delta, NAMES, counts, n_splits=20, seed=1,
                                  n_strata=4, min_classes=1000)
    assert all(v.get("skipped") for v in r["by_stratum"].values())
    assert r["n_strata_reported"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} stratified gate-C tests passed")
