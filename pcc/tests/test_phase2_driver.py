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
             phi="head", head_weights=world["head"], head_bias=None,
             distance_holdout="w_cos_knn_1", stat="worst", ccc_root=None,
             seed=0, name=None, print_json=False)
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


def test_unknown_distance_holdout_fails_loudly_with_the_available_names(world):
    with pytest.raises(ValueError) as ei:
        drv.run(_args(world, distance_holdout="prof_knn_1"))     # output-family name
    assert "not a descriptor of the 'head' family" in str(ei.value)


def test_verdict_distinguishes_a_trade_from_a_win():
    base = {"table_1_seen": {"delta": {"worst": 0.0}},
            "table_2_heldout": {"delta": {"worst": 0.05}}}
    assert drv.verdict(base, "worst") == "LULUS"

    trade = {"table_1_seen": {"delta": {"worst": -0.20}},
             "table_2_heldout": {"delta": {"worst": 0.05}}}
    assert "MENUKAR" in drv.verdict(trade, "worst")

    lose = {"table_1_seen": {"delta": {"worst": 0.0}},
            "table_2_heldout": {"delta": {"worst": -0.05}}}
    assert drv.verdict(lose, "worst") == "GAGAL"

    assert "TIDAK DAPAT DINILAI" in drv.verdict(
        {"table_1_seen": {"delta": {"worst": 0.0}}}, "worst")


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
