"""Phase-0 v2 (Amendment 6) must recover a PLANTED mechanism, and gate C must
actually RUN for a restricted feature set.

The Phase-0 test is the pre-registration guard: the set-size criterion it replaced
named `temperature` the winner in 4 of 4 planted worlds, so any Phase-0 verdict is
only readable if the metric first proves it can tell the worlds apart.
"""
import numpy as np

from pcc.eval.decomposition import phase0_explain_class_level
from pcc.eval.predictability import predictability


def make_world(world, n_per_class=160, K=60, seed=0):
    """Logits carrying exactly one planted mechanism."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n_per_class)
    L = rng.normal(0, 1, (len(y), K))
    if world == "class":
        L[np.arange(len(y)), y] += rng.uniform(1.0, 5.0, K)[y]
    elif world == "global":
        L[np.arange(len(y)), y] += 3.0
        L *= 0.45
    elif world == "energy":
        L[np.arange(len(y)), y] += 3.0
        L *= rng.normal(1.0, 0.35, len(y)).clip(0.3, 3.0)[:, None]
    elif world == "none":
        L[np.arange(len(y)), y] += 3.0
    return L, y


def _split(y, seed=0):
    rng = np.random.default_rng(seed)
    cal = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cal[idx[: len(idx) // 2]] = True
    return np.where(cal)[0], np.where(~cal)[0]


def _verdict(r):
    return (r["reliability_identity"] > 0.30 and r["reliability_after_T"] > 0.30
            and r["class_beats_rivals"])


def test_recovers_planted_class_structure():
    L, y = make_world("class")
    ci, ei = _split(y)
    r = phase0_explain_class_level(L, y, 60, 0.1, ci, ei)
    assert _verdict(r), f"planted class structure not recovered: {r['r2']}"
    assert r["r2"]["class"] > r["r2"]["global"]
    assert r["target_sd"] > 0.02, "planted world should show real class-level spread"
    print(f"  class world : R2 class {r['r2']['class']:+.3f} > best rival "
          f"{r['best_rival']} {max(v for k, v in r['r2'].items() if k not in ('class','global')):+.3f}"
          f"  sd(q*)={r['target_sd']:.4f}  OK")


def test_rejects_worlds_without_class_structure():
    for world in ("global", "energy", "none"):
        L, y = make_world(world)
        ci, ei = _split(y)
        r = phase0_explain_class_level(L, y, 60, 0.1, ci, ei)
        assert not _verdict(r), f"{world} world wrongly called class-level: {r}"
        assert r["target_sd"] < 0.02, f"{world}: unexpected class-level spread"
        print(f"  {world:7s} world: reliability {r['reliability_identity']:+.3f} "
              f"R2 class {r['r2']['class']:+.3f}  sd(q*)={r['target_sd']:.4f}  correctly rejected")


def test_capacity_is_punished_not_rewarded():
    """Nested energy bins must not improve by adding noise parameters."""
    L, y = make_world("none")
    ci, ei = _split(y)
    r = phase0_explain_class_level(L, y, 60, 0.1, ci, ei, bin_grid=(2, 10, 50))
    assert r["r2"]["energy_b50"] < 0.05 and r["r2"]["energy_b10"] < 0.05
    print(f"  no-structure world: energy R2 b2 {r['r2']['energy_b2']:+.3f} "
          f"b10 {r['r2']['energy_b10']:+.3f} b50 {r['r2']['energy_b50']:+.3f} "
          "-> capacity punished")


def test_gate_C_runs_for_a_restricted_feature_set():
    """The bug this locks down: slicing Phi to the stable set removed the ablation
    columns, so gate C returned None for the PRIMARY feature set."""
    rng = np.random.default_rng(0)
    K = 120
    names = ["cos_knn_5", "logit_margin", "log_prevalence", "cos_knn_1", "junk"]
    Phi = rng.normal(0, 1, (K, len(names)))
    delta = 0.8 * Phi[:, 0] + 0.5 * Phi[:, 1] + rng.normal(0, 0.3, K)

    res = predictability(Phi, delta, names,
                         feature_subset=["cos_knn_5", "logit_margin"], n_splits=40)
    assert res["gate_C_pass"] is not None, "gate C must not silently vanish"
    assert set(res["gate_C_detail"]) == {"log_prevalence_only", "distance_only",
                                         "prevalence+distance",
                                         "distance_only_prereg",
                                         "prevalence+distance_prereg"}
    assert res["n_features_full"] == 2
    assert res["distance_col_used"] == "cos_knn_5"
    assert res["gate_B_pass"] and res["gate_C_pass"], "planted signal should pass B and C"
    print(f"  restricted set: gate B {res['gate_B_pass']} gate C {res['gate_C_pass']} "
          f"ablations {sorted(res['gate_C_detail'])} distance={res['distance_col_used']}  OK")


def test_nested_distance_baseline_is_flagged_and_prereg_reported_separately():
    """The Amendment-7 side effect: the stability screen picked `cos_knn_5` as the
    distance baseline, and on Pl@ntNet that feature is ALSO in the primary feature set.
    The ablation is then a nested submodel, which makes gate C strictly harder than the
    pre-registered §6.5C question. Both verdicts must be reported, never conflated."""
    rng = np.random.default_rng(5)
    K = 140
    names = ["cos_knn_5", "logit_margin", "log_prevalence", "cos_knn_1"]
    Phi = rng.normal(0, 1, (K, len(names)))
    Phi[:, 3] = 0.3 * Phi[:, 0] + rng.normal(0, 0.9, K)   # a weaker distance proxy
    delta = 0.9 * Phi[:, 0] + rng.normal(0, 0.5, K)       # signal lives in cos_knn_5 ONLY

    nested = predictability(Phi, delta, names,
                            feature_subset=["cos_knn_5", "logit_margin"], n_splits=60)
    assert nested["distance_baseline_is_nested_in_full"], "nesting must be flagged"
    assert nested["distance_col_prereg"] == "cos_knn_1"
    # Signal is entirely in cos_knn_5, so adding logit_margin buys nothing: the NESTED
    # test must fail while the pre-registered (independent) baseline is still beaten.
    assert not nested["gate_C_pass"], "nested test should not pass when the extra feature is inert"
    assert nested["gate_C_pass_prereg"], "the pre-registered baseline should still be beaten"

    apart = predictability(Phi, delta, names,
                           feature_subset=["cos_knn_5", "logit_margin"],
                           distance_col=("cos_knn_1",), n_splits=60)
    assert not apart["distance_baseline_is_nested_in_full"]
    print(f"  nested={nested['distance_baseline_is_nested_in_full']} -> "
          f"gate_C {nested['gate_C_pass']} but gate_C_prereg "
          f"{nested['gate_C_pass_prereg']}; the two are not the same question  OK")


def test_underpowered_flag_marks_the_overfit_regime():
    rng = np.random.default_rng(1)
    K, p = 38, 15
    names = [f"f{i}" for i in range(p - 1)] + ["log_prevalence"]
    Phi = rng.normal(0, 1, (K, p))
    delta = rng.normal(0, 1, K)
    res = predictability(Phi, delta, names, n_splits=30)
    assert res["underpowered"], "15 features on 19 training classes must be flagged"
    wide = predictability(Phi, delta, names, feature_subset=["f0", "f1"], n_splits=30)
    assert not wide["underpowered"]
    print(f"  p=15 n_train={res['n_train_classes']} -> underpowered={res['underpowered']}; "
          f"p=2 -> underpowered={wide['underpowered']}  OK")


if __name__ == "__main__":
    test_recovers_planted_class_structure()
    test_rejects_worlds_without_class_structure()
    test_capacity_is_punished_not_rewarded()
    test_gate_C_runs_for_a_restricted_feature_set()
    test_nested_distance_baseline_is_flagged_and_prereg_reported_separately()
    test_underpowered_flag_marks_the_overfit_regime()
    print("all 6 phase-0-v2 / gate-C tests passed")
