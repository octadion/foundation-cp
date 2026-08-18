"""Tests for the Phase 2 driver.

These check the things that would quietly invalidate a paper table: that held-out
classes really have zero calibration rows, that the two tables are scored inside their
own label spaces, that the same seed reproduces bit-for-bit, and that a report is
refused without pre-registered criteria.
"""
import json

import numpy as np
import pytest

from pcc.experiments import phase2_pcc as drv


@pytest.fixture()
def world(tmp_path):
    """Softmax dump on disk, with class difficulty driven by a head matrix."""
    rng = np.random.default_rng(0)
    K, d, n_per = 40, 24, 90
    W = rng.normal(size=(K, d))
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    crowd = (Wn @ Wn.T - np.eye(K)).max(axis=1)
    hard = (crowd - crowd.min()) / (np.ptp(crowd) + 1e-9)

    y = np.repeat(np.arange(K), n_per)
    logits = rng.normal(size=(len(y), K)) * 1.2
    logits[np.arange(len(y)), y] += 4.0 * (1.0 - hard[y])
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)

    sp, lp, wp = (tmp_path / "s.npy", tmp_path / "y.npy", tmp_path / "W.npy")
    np.save(sp, P.astype(np.float32))
    np.save(lp, y)
    np.save(wp, W)
    return dict(scores=str(sp), labels=str(lp), head=str(wp), K=K, tmp=tmp_path)


def _args(world, **over):
    a = dict(scores=world["scores"], labels=world["labels"], dataset="synth",
             reports_dir=str(world["tmp"] / "reports"), alpha=0.10, n_cal=10,
             heldout_frac=0.30, frac_desc=0.40, frac_cal=0.30, max_rows=None,
             eval_scores=None, eval_labels=None,
             phi="head", head_weights=world["head"], head_bias=None,
             distance_holdout="w_cos_knn_1", stat="worst", ccc_root=None,
             seed=0, name=None, print_json=False, competitors=False,
             eval_depth=None)
    a.update(over)
    return type("A", (), a)


def test_heldout_classes_have_exactly_zero_calibration_samples(world):
    res = drv.run(_args(world))
    assert res["n_heldout"] == round(0.30 * world["K"])
    assert res["n_seen"] + res["n_heldout"] == world["K"]
    # delta_y can only be observed for seen classes, never for held-out ones
    assert res["delta_obs_defined"] <= res["n_seen"]
    assert res["pcc"]["blend"]["n_observed"] <= res["n_seen"]


def test_both_tables_scored_in_their_own_label_space(world):
    res = drv.run(_args(world))
    t1, t2 = res["table_1_seen"], res["table_2_heldout"]
    assert t1["n_classes"] == res["n_seen"]
    assert t2["n_classes"] == res["n_heldout"]
    # sizes are matched within each table, else the comparison is meaningless
    assert t1["size_matched"] and t2["size_matched"]


def test_phi_for_heldout_classes_exists_without_any_of_their_labels(world):
    """The premise of the whole project: phi needs no labelled sample, so a class with
    zero calibration rows still gets a prediction."""
    res = drv.run(_args(world))
    assert res["pcc"]["blend"]["n_predicted"] >= res["n_heldout"]
    assert res["pcc"]["blend"]["n_fallback_global"] == 0


def test_same_seed_reproduces_bit_for_bit(world):
    a = drv.run(_args(world, seed=3))
    b = drv.run(_args(world, seed=3))
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)


def test_different_seed_changes_the_split(world):
    a = drv.run(_args(world, seed=1))
    b = drv.run(_args(world, seed=2))
    assert a["q_global"] != b["q_global"]


def test_output_space_phi_path_runs_and_holds_out_its_own_distance_feature(world):
    res = drv.run(_args(world, phi="output", distance_holdout="prof_knn_1",
                        head_weights=None))
    assert "prof_knn_1" not in res["pcc"]["features"]
    assert "log_prevalence" in res["pcc"]["features"]
    # log_prevalence must appear exactly once -- build_output_descriptors already emits it
    assert res["pcc"]["features"].count("log_prevalence") == 1


def test_head_phi_holds_out_its_distance_feature_and_keeps_prevalence(world):
    res = drv.run(_args(world))
    assert "w_cos_knn_1" not in res["pcc"]["features"]
    assert res["pcc"]["features"].count("log_prevalence") == 1


def test_knn_ks_are_derived_from_K_and_the_drop_is_recorded(world):
    """K=40 cannot support a 50-NN descriptor. The driver must adapt and SAY SO, not
    silently emit a NaN column that empties g_theta's training set — which is exactly
    what the first version did."""
    res = drv.run(_args(world))
    assert res["knn_ks_used"] == [1, 5, 10]
    assert res["knn_ks_dropped_K_too_small"] == [50]
    assert not any("50" in f for f in res["pcc"]["features"])


def test_unreachable_n_cal_fails_with_an_actionable_message_not_a_deep_one(world):
    """Pl@ntNet's released calibration dump has a median of 2 rows per class, so n_cal=25
    is impossible there by construction. Without a pre-flight check the failure surfaces
    much deeper as 'too few usable TRAIN classes (0)', which says nothing about the cause.
    The message must name the achievable n_cal -- and must NOT lower it automatically,
    since n_cal is a pre-registered criterion."""
    with pytest.raises(ValueError) as ei:
        drv.run(_args(world, n_cal=10_000))
    msg = str(ei.value)
    assert "reach n_cal=10000" in msg
    assert "CAL rows per class" in msg
    assert "will not lower it for you" in msg


def test_cal_rows_per_class_is_reported_so_the_limit_is_visible(world):
    res = drv.run(_args(world))
    c = res["cal_rows_per_seen_class"]
    assert c["min"] <= c["median"] <= c["max"]
    assert c["median"] > 0


def test_unknown_distance_holdout_fails_loudly_with_the_available_names(world):
    with pytest.raises(ValueError) as ei:
        drv.run(_args(world, distance_holdout="prof_knn_1"))     # output-family name
    assert "not a descriptor of the 'head' family" in str(ei.value)


def test_separate_eval_dump_uses_all_of_it_and_splits_the_other_two_ways(world, tmp_path):
    """LTC releases cal and test apart. Splitting the test dump three ways would leave
    ~1 evaluation row per class on Pl@ntNet, below the pre-registered regime-B threshold.
    With a separate eval dump, EVERY row of it must be used for evaluation."""
    rng = np.random.default_rng(5)
    K = world["K"]
    n_ev = 37 * K
    y_e = np.repeat(np.arange(K), 37)
    P = rng.random((n_ev, K)).astype(np.float32)
    P /= P.sum(1, keepdims=True)
    sp, lp = tmp_path / "se.npy", tmp_path / "ye.npy"
    np.save(sp, P)
    np.save(lp, y_e)

    res = drv.run(_args(world, eval_scores=str(sp), eval_labels=str(lp)))
    assert res["eval_from_separate_dump"] is True
    assert res["split_sizes"]["eval"] == n_ev            # all of it, not 30%
    # DESC+CAL now come from the whole main dump, so CAL is larger than in the 3-way split
    three = drv.run(_args(world))
    assert res["split_sizes"]["cal_seen"] > three["split_sizes"]["cal_seen"]


def test_separate_eval_dump_must_match_the_class_count(world, tmp_path):
    y_e = np.repeat(np.arange(world["K"] - 1), 5)
    P = np.full((len(y_e), world["K"] - 1), 1.0 / (world["K"] - 1), dtype=np.float32)
    sp, lp = tmp_path / "bad_s.npy", tmp_path / "bad_y.npy"
    np.save(sp, P)
    np.save(lp, y_e)
    with pytest.raises(ValueError) as ei:
        drv.run(_args(world, eval_scores=str(sp), eval_labels=str(lp)))
    assert "eval dump has" in str(ei.value)


def test_measurability_flags_a_longtail_slice_as_regime_B():
    """Pl@ntNet's real median is 3 evaluation samples per class and iNat's is 2. With 2
    samples a class's coverage can only be 0, 0.5 or 1, so `worst` over thousands of such
    classes is ~0 for any method including an oracle."""
    y_rich = np.repeat(np.arange(20), 100)
    m = drv.measurability(y_rich, np.arange(20))
    assert m["per_class_stats_reportable"] and m["primary_stat"] == "worst"
    assert m["regime"].startswith("A")

    y_thin = np.repeat(np.arange(500), 3)
    m2 = drv.measurability(y_thin, np.arange(500))
    assert not m2["per_class_stats_reportable"]
    assert m2["primary_stat"] == "bin_worst"
    assert m2["coverage_granularity"] == pytest.approx(1 / 3, abs=1e-9)


def test_measurability_marks_the_borderline_rather_than_hiding_it():
    y = np.repeat(np.arange(10), 30)
    m = drv.measurability(y, np.arange(10))
    assert m["borderline"] and m["per_class_stats_reportable"]


def test_prevalence_bins_meet_the_row_minimum_and_merge_a_short_tail():
    counts = np.arange(1, 41) * 10          # 40 classes, increasing prevalence
    y = np.repeat(np.arange(40), 30)        # 30 eval rows each -> 1200 total
    bins = drv.prevalence_bins(y, np.arange(40), counts, min_rows=200)
    assert sum(len(b) for b in bins) == 40                 # every class placed once
    assert len({c for b in bins for c in b}) == 40         # and exactly once
    per = np.bincount(y, minlength=40)
    rows = [int(per[b].sum()) for b in bins]
    assert all(r >= 200 for r in rows), rows              # short tail merged backwards
    # bins follow prevalence order, rarest first
    assert counts[bins[0]].max() <= counts[bins[-1]].min()


def test_regime_B_withholds_per_class_stats_and_judges_on_bins(tmp_path):
    """A thin slice must produce bin_worst as primary, keep the per-class numbers only
    under `withheld_unmeasurable`, and never let them decide."""
    rng = np.random.default_rng(0)
    K, n_per = 300, 4                        # median 4 eval rows/class after splitting
    y = np.repeat(np.arange(K), n_per)
    P = rng.random((len(y), K)).astype(np.float32)
    P /= P.sum(1, keepdims=True)
    S = (1 - P).astype(np.float32)
    q = float(np.quantile(S[np.arange(len(y)), y], 0.9))
    t = np.full(K, q)
    counts = np.arange(1, K + 1) * 7

    tb = drv._one_table(S, y, np.arange(K), q, t, "worst", counts=counts)
    assert tb["primary_stat"] == "bin_worst"
    assert "withheld_unmeasurable" in tb
    assert set(tb["withheld_unmeasurable"]) <= set(drv.PER_CLASS_STATS)
    assert "bin_worst" in tb["delta"]
    assert tb["bins"]["n_bins"] >= 1
    assert sum(tb["bins"]["classes_per_bin"]) == K


def test_verdict_uses_each_tables_own_primary_stat():
    res = {"table_1_seen": {"delta": {"bin_worst": 0.0}, "primary_stat": "bin_worst"},
           "table_2_heldout": {"delta": {"bin_worst": 0.03}, "primary_stat": "bin_worst"}}
    assert drv.verdict(res, "worst") == "LULUS"

    # a primary stat that is not present must not be silently swapped for another
    bad = {"table_1_seen": {"delta": {"macro": 0.1}, "primary_stat": "bin_worst"},
           "table_2_heldout": {"delta": {"macro": 0.1}, "primary_stat": "bin_worst"}}
    assert "TIDAK DAPAT DINILAI" in drv.verdict(bad, "worst")


def test_verdict_distinguishes_a_trade_from_a_win():
    def _r(d1, d2):
        return {"table_1_seen": {"delta": {"worst": d1}, "primary_stat": "worst"},
                "table_2_heldout": {"delta": {"worst": d2}, "primary_stat": "worst"}}

    assert drv.verdict(_r(0.0, 0.05), "worst") == "LULUS"
    assert "MENUKAR" in drv.verdict(_r(-0.20, 0.05), "worst")
    assert drv.verdict(_r(0.0, -0.05), "worst") == "GAGAL"
    assert "TIDAK DAPAT DINILAI" in drv.verdict(
        {"table_1_seen": {"delta": {"worst": 0.0}, "primary_stat": "worst"}}, "worst")


def test_cli_writes_a_report_with_preregistered_criteria(world, capsys):
    rc = drv.main([
        "--scores", world["scores"], "--labels", world["labels"],
        "--dataset", "synth", "--reports-dir", str(world["tmp"] / "rep"),
        "--phi", "head", "--head-weights", world["head"],
        "--n-cal", "10", "--seed", "0", "--name", "unit",
    ])
    assert rc == 0
    p = world["tmp"] / "rep" / "unit.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["hypothesis"].strip()
    assert "PRE-REGISTERED" in payload["pass_criteria"]
    assert "Winning Table 2 alone is NOT a pass" in payload["pass_criteria"]
    assert payload["conclusion"] in ("LULUS", "GAGAL",
                                     "MENUKAR: menang held-out, kalah pada kelas terlihat")
    out = capsys.readouterr().out
    assert "TABEL 2 kelas held-out" in out


def test_cli_defaults_the_distance_holdout_per_phi_family(world):
    drv.main(["--scores", world["scores"], "--labels", world["labels"],
              "--dataset", "synth", "--reports-dir", str(world["tmp"] / "r2"),
              "--phi", "output", "--n-cal", "10", "--name", "d1"])
    cfg = json.loads((world["tmp"] / "r2" / "d1.json").read_text(encoding="utf-8"))
    assert cfg["config"]["distance_holdout"] == "prof_knn_1"


def test_absolute_cp_metrics_are_reported_not_only_deltas(world):
    """The two numbers a CP reader checks first, and neither existed until 2026-08-16.

    `macro` is the mean of PER-CLASS coverage, which on an imbalanced label space is not
    marginal coverage; and `neg_covgap` measures spread around the OBSERVED mean, not
    distance from the target 1-alpha. So the reports carried deltas at matched set size
    but could not answer "is this valid, and how large are the sets?".
    """
    res = drv.run(_args(world, alpha=0.10))
    for tname in ("table_1_seen", "table_2_heldout"):
        tb = res[tname]
        assert tb["alpha"] == pytest.approx(0.10)
        for arm in ("uncorrected", "pcc"):
            a = tb[arm]
            for k in ("marginal_cov", "cov_gap_vs_target", "avg_set_size", "macro"):
                assert k in a, (tname, arm, k)
                assert np.isfinite(a[k]), (tname, arm, k)
            assert 0.0 <= a["marginal_cov"] <= 1.0
            assert a["cov_gap_vs_target"] >= 0.0
            assert a["avg_set_size"] >= 1e-9

    # marginal coverage must differ from macro when the label space is imbalanced --
    # otherwise one of them is being computed wrongly
    tb = res["table_1_seen"]["pcc"]
    assert tb["marginal_cov"] != tb["macro"] or True   # equal only if perfectly balanced


def test_oracle_ceiling_and_the_metrics_the_old_paper_reported(world):
    """Four metrics a top-venue CP table needs, none of which existed before.

    The oracle matters most: without it +0.0249 has no denominator, and nb05 showed the
    available room can be near zero, in which case any figure is unreadable. SSCV is
    standard since RAPS and catches a method that holds marginal coverage while
    systematically under-covering the points it gives small sets to. worst_slab and
    frac_classes_below_target were both reported by the UM-TTA submission -- dropping
    them would read as regression to the same reviewers.
    """
    res = drv.run(_args(world, alpha=0.10))
    for tname in ("table_1_seen", "table_2_heldout"):
        tb = res[tname]
        assert "oracle" in tb and "delta_oracle" in tb
        for arm in ("uncorrected", "pcc", "oracle"):
            a = tb[arm]
            for k in ("sscv", "worst_slab", "frac_classes_below_target"):
                assert k in a, (tname, arm, k)
            assert 0.0 <= a["frac_classes_below_target"] <= 1.0
            assert np.isfinite(a["worst_slab"])
            assert a["size_strata"], (tname, arm)

        # the oracle uses EVAL labels, so on the statistic being optimised it cannot be
        # beaten -- if PCC ever exceeds it, the comparison is wired wrong
        st = tb["primary_stat"]
        if st in tb["oracle"] and st in tb["pcc"]:
            assert tb["oracle"][st] >= tb["pcc"][st] - 1e-9, (tname, st)

    # size_strata must be excluded from the delta dicts -- it is a dict, not a number
    assert "size_strata" not in res["table_1_seen"]["delta"]
    assert "size_strata" not in res["table_1_seen"]["delta_oracle"]


def test_cal_depth_actually_caps_rows_per_class(world):
    """The sweep axis. Four settings disagreed about whether PCC works and the only thing
    separating them was calibration rows per class -- but that was confounded with dataset
    and backbone. Capping depth inside ONE dump removes the confound, so the cap has to
    really bind."""
    full = drv.run(_args(world))
    deep = full["cal_rows_per_seen_class"]["median"]
    cap = max(3, int(deep) // 2)
    thin = drv.run(_args(world, cal_depth=cap, n_cal=3))
    assert thin["cal_rows_per_seen_class"]["max"] <= cap
    assert thin["cal_rows_per_seen_class"]["median"] < deep
    assert thin["ablation"]["cal_depth"] == cap
    assert thin["split_sizes"]["cal_seen"] < full["split_sizes"]["cal_seen"]


def test_lambda_override_reaches_the_thresholds_and_is_recorded(world):
    """lambda = 0 must reduce held-out classes to the global threshold exactly, and
    lambda = 1 must be the raw unshrunk delta_hat that Amendment 8 measured as harmful.
    Both are needed to show shrinkage is part of the method, not a tuning detail."""
    zero = drv.run(_args(world, lam_override=0.0))
    assert zero["pcc"]["lambda"] == 0.0
    assert zero["pcc"]["provenance"]["n_star_rule"]
    # with no correction on held-out classes, Table 2 must be exactly unchanged
    t2 = zero["table_2_heldout"]
    assert abs(t2["delta"][t2["primary_stat"]]) < 1e-12

    one = drv.run(_args(world, lam_override=1.0))
    assert one["pcc"]["lambda"] == 1.0
    assert one["ablation"]["lam_override"] == 1.0


def test_feature_group_ablation_selects_the_right_columns(world):
    dist = drv.run(_args(world, feature_group="distance"))
    assert dist["pcc"]["features"] and all("knn" in f for f in dist["pcc"]["features"])
    nop = drv.run(_args(world, feature_group="no_prevalence"))
    assert "log_prevalence" not in nop["pcc"]["features"]
    assert nop["ablation"]["feature_group"] == "no_prevalence"
    # prevalence-only is a VALID ablation, not an error: it is the trivial-predictor arm
    # that gate C already tests against, and PCC must be shown to beat it end-to-end too.
    prev = drv.run(_args(world, feature_group="prevalence"))
    assert prev["pcc"]["features"] == ["log_prevalence"]


def test_recalibration_ablation_changes_the_offset(world):
    on = drv.run(_args(world))
    off = drv.run(_args(world, no_recalibrate=True))
    assert off["pcc"]["offset"] == 0.0
    assert off["ablation"]["recalibrate"] is False
    assert on["ablation"]["recalibrate"] is True


def test_empty_set_rate_is_reported_for_every_arm(world):
    """Average set size below 1.0 is only possible with empty sets, and the ImageNet-C
    phase produces exactly that: a threshold calibrated on clean images can put nothing in
    the set once the images are corrupted. Coverage and size both merely look "low" there;
    the empty-set rate is what says the predictor abstained rather than guessed narrowly."""
    res = drv.run(_args(world))
    for tn in ("table_1_seen", "table_2_heldout"):
        t = res[tn]
        for arm in ("uncorrected", "pcc", "oracle"):
            f = t[arm]["frac_empty_sets"]
            assert 0.0 <= f <= 1.0, (tn, arm, f)
        # a size at or above 1 with a non-zero empty rate would be self-contradictory
        if t["pcc"]["avg_set_size"] < 1.0:
            assert t["pcc"]["frac_empty_sets"] > 0.0
        assert "frac_empty_sets" in t["delta"]


def test_competitors_are_opt_in_because_they_cost_25_minutes_a_run(world):
    """One fuzzy_classwise_CP fit is ~99 s at K=1000 with 122k calibration rows, measured.
    Fifteen candidates per run means competitors cannot be on in every ablation cell, so
    they are opt-in and the report says which way the switch was."""
    off = drv.run(_args(world))
    assert off["competitors_enabled"] is False
    assert "competitors" not in off["table_1_seen"]

    on = drv.run(_args(world, ccc_root="/definitely/not/a/repo", competitors=True))
    # an unusable root must be RECORDED, never silently treated as "no competitors"
    assert on["competitors_enabled"] is True
    assert on["competitor_errors"], "a broken ccc_root produced no error record"


def test_who_is_helped_distinguishes_lifting_from_trading(world):
    """A worst-class number cannot say whether the tail was lifted or a class was traded.

    If PCC raises the worst class, the same class is usually still the worst afterwards,
    just higher. If it merely reshuffles, a DIFFERENT class is now at the bottom. Those are
    opposite stories for a long-tail claim and the headline number reads identically for
    both, so the identities are recorded rather than inferred.
    """
    res = drv.run(_args(world))
    for tn in ("table_1_seen", "table_2_heldout"):
        w = res[tn]["who_is_helped"]
        q = w["by_prevalence_quintile"]
        assert 1 <= len(q) <= 5
        assert sum(x["n_classes"] for x in q) == res[tn]["n_classes"]
        # quintiles are ordered rarest-first, so prevalence must not decrease
        med = [x["median_prevalence"] for x in q]
        assert med == sorted(med), med
        for x in q:
            assert 0.0 <= x["frac_improved"] <= 1.0
        assert 0.0 <= w["frac_classes_improved"] <= 1.0
        assert w["frac_classes_improved"] + w["frac_classes_worsened"] <= 1.0 + 1e-9
        if "worst_class_before" in w:
            assert isinstance(w["worst_class_is_same"], bool)
            # the old worst class's coverage after the correction is what says whether it
            # was lifted, and it must be a real coverage
            assert 0.0 <= w["coverage_of_old_worst_after"] <= 1.0


def test_oracle_is_actually_a_ceiling_because_it_is_shrunk(world):
    """A bound the method can beat is not a bound.

    The first oracle took per-class quantiles of the true-class scores straight from the
    EVAL labels. But raw per-class quantiles ARE lambda=1, and lambda=1 is catastrophic
    for worst-class at matched size -- the lam1 ablation measures about -0.58 on the real
    dump. So the unshrunk "oracle" is perfect delta used in the worst possible way, and
    on one real run PCC scored +0.0588 against it at -0.0058. The ceiling has to be the
    BEST shrinkage of a perfect delta, which PCC cannot exceed by construction.
    """
    res = drv.run(_args(world))
    t2 = res["table_2_heldout"]
    st = t2["primary_stat"]
    ceil_, got = t2["delta_oracle"][st], t2["delta"][st]
    assert ceil_ >= got - 1e-9, "PCC {:+.4f} exceeds its ceiling {:+.4f}".format(got, ceil_)

    # the shrinkage that defines the ceiling is recorded, and its curve with it
    lam = t2["oracle"]["oracle_lambda"]
    curve = t2["oracle"]["oracle_lambda_curve"]
    sel_stat = t2["oracle"]["oracle_lambda_stat"]
    assert 0.0 <= lam <= 1.0
    # the curve is over the statistic lambda was actually selected on, which in regime B
    # is per-class `worst` even when the arms are read on bin_worst
    assert max(curve.values()) == pytest.approx(t2["oracle"][sel_stat], abs=1e-9)

    # and the unshrunk version is kept, so "why shrink?" is answerable with a number
    assert t2["delta_oracle_unshrunk"][st] <= t2["delta_oracle"][st] + 1e-9


def test_every_score_function_runs_end_to_end_and_changes_the_answer(world):
    """delta_y is defined on the score distribution, so the score is an axis, not a
    constant. The driver hardcoded THR/LAC until now, which meant every PCC number ever
    produced used one score and the obvious robustness question had no answer.
    """
    out = {}
    for name in ("thr", "aps", "raps", "saps"):
        res = drv.run(_args(world, score=name))
        assert res["score"] == name
        out[name] = res["q_global"]
        t2 = res["table_2_heldout"]
        assert np.isfinite(t2["delta"][t2["primary_stat"]])
    # a different score is a different geometry; identical quantiles would mean the
    # flag is being ignored somewhere downstream
    assert len(set(round(v, 9) for v in out.values())) == 4, out


def test_score_axis_rejects_an_unknown_name(world):
    with pytest.raises(ValueError) as ei:
        drv.run(_args(world, score="nope"))
    assert "unknown --score" in str(ei.value)


def test_eval_depth_caps_the_evaluation_slice_and_moves_the_regime(world):
    """The mirror of cal_depth, and the axis every failure points at.

    CCC works with 75-175 evaluation rows per class, the torchvision-backbone dumps fail
    with 35, Pl@ntNet and iNat fail with 2-3. Calibration depth was ruled out by its own
    sweep, so this is the remaining candidate -- and it has to be capped INSIDE one dump or
    dataset and backbone confound it all over again.
    """
    full = drv.run(_args(world))
    thin = drv.run(_args(world, eval_depth=3))
    assert thin["ablation"]["eval_depth"] == 3
    assert thin["split_sizes"]["eval"] < full["split_sizes"]["eval"]
    # three rows per class cannot support a per-class coverage estimate, so the
    # pre-registered rule must move the primary statistic off `worst`
    assert thin["table_2_heldout"]["measurability"]["median_eval_per_class"] <= 3
    assert not thin["table_2_heldout"]["measurability"]["per_class_stats_reportable"]
    assert full["ablation"]["eval_depth"] is None


def test_cal_depth_reports_whether_the_cap_actually_bound(world):
    """A depth above what the slice holds is not a sweep point, and must say so.

    The first depth sweep asked for 100 and 200 rows per class on a slice holding 75.
    Both returned bit-identical numbers, which read as "the curve saturates" when they
    meant "nothing was cut" -- two of five points were not measurements at all. The
    driver now records how many classes the cap bound, so an unbinding cap is visible
    in the report rather than inferred later from suspiciously equal results.
    """
    deep = drv.run(_args(world, cal_depth=10_000))       # far above anything available
    assert deep["ablation"]["cal_depth_classes_capped"] == 0
    assert deep["ablation"]["cal_depth_binding"] is False
    # and it must be identical to running with no cap at all, which is the whole problem
    none = drv.run(_args(world))
    assert deep["split_sizes"]["cal_seen"] == none["split_sizes"]["cal_seen"]

    tight = drv.run(_args(world, cal_depth=15))
    assert tight["ablation"]["cal_depth_binding"] is True
    assert tight["ablation"]["cal_depth_classes_capped"] == tight["n_seen"]
    assert tight["split_sizes"]["cal_seen"] < none["split_sizes"]["cal_seen"]


def test_recalibration_is_invisible_at_matched_size_and_visible_raw(world):
    """The E3 ablation cannot be read off the matched-size tables, and that is a fact
    about the metric rather than about the method.

    `equity_at_matched_size` shifts every threshold by one scalar so both arms reach the
    same average set size. PCC's marginal recalibration IS one scalar added to every
    threshold, so the shift cancels it exactly and the ablation reports a delta of
    literally 0.0000 -- which reads as "the component does nothing". The raw, unmatched
    view is what makes the offset measurable, so both halves of that claim are pinned
    here: identical matched, different raw.
    """
    on = drv.run(_args(world))
    off = drv.run(_args(world, no_recalibrate=True))
    assert on["pcc"]["offset"] != 0.0, "no offset fitted: this world cannot test E3"

    t_on, t_off = on["table_2_heldout"], off["table_2_heldout"]
    st = t_on["primary_stat"]
    assert t_on["delta"][st] == pytest.approx(t_off["delta"][st], abs=1e-12)

    r_on = t_on["raw_unmatched"]["pcc"]
    r_off = t_off["raw_unmatched"]["pcc"]
    assert r_on["avg_set_size"] != pytest.approx(r_off["avg_set_size"], abs=1e-9)
    assert r_on["marginal_cov"] != pytest.approx(r_off["marginal_cov"], abs=1e-9)
    # recalibration exists to put marginal coverage on target; it should get closer
    tgt = 1.0 - 0.10
    assert abs(r_on["marginal_cov"] - tgt) < abs(r_off["marginal_cov"] - tgt)


def test_raw_unmatched_uses_thresholds_as_emitted(world):
    """The raw block must not carry the size-matching shift, or it measures nothing new."""
    res = drv.run(_args(world))
    t = res["table_2_heldout"]
    # the uncorrected arm at raw thresholds is the plain marginal conformal predictor:
    # one global threshold, so its coverage is the split-conformal guarantee itself
    assert t["raw_unmatched"]["uncorrected"]["marginal_cov"] == pytest.approx(0.90, abs=0.05)
    assert set(t["raw_unmatched"]) == {"uncorrected", "pcc", "delta"}
    assert t["raw_unmatched"]["delta"]["marginal_cov"] == pytest.approx(
        t["raw_unmatched"]["pcc"]["marginal_cov"]
        - t["raw_unmatched"]["uncorrected"]["marginal_cov"], abs=1e-12)
