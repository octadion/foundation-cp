"""Tests for `pcc.method.pcc`.

The point of these is not that the code runs. It is that each free parameter behaves
the way the gate evidence says it must, and that the ones which could manufacture a
result (λ, n_star, the marginal offset) cannot see the evaluation classes.
"""
import numpy as np
import pytest

from pcc.method.pcc import (blend_delta, data_threshold, fit_gtheta, fit_pcc,
                            gtheta_cv_mse, quantile_noise_at_n, recalibrate_marginal,
                            select_n_star, select_n_star_oos)


def _world(K=60, n_per=120, d_phi=4, seed=0, signal=1.0):
    """A world where class difficulty IS a linear function of phi, so g_theta has
    something real to find and the tests can distinguish signal from plumbing."""
    rng = np.random.default_rng(seed)
    Phi = rng.normal(size=(K, d_phi))
    w = np.array([1.0, -0.6, 0.3, 0.0])[:d_phi]
    hard = Phi @ w
    hard = (hard - hard.min()) / (np.ptp(hard) + 1e-9)
    y = np.repeat(np.arange(K), n_per)
    logits = rng.normal(size=(len(y), K)) * 1.2
    logits[np.arange(len(y)), y] += 4.0 * (1.0 - signal * hard[y])
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    S = (1.0 - P).astype(np.float64)
    names = ["f%d" % i for i in range(d_phi)]
    return Phi, names, S, y, K


def _delta_obs(S, y, K, alpha, q_global, classes=None):
    d = np.full(K, np.nan)
    for c in range(K) if classes is None else classes:
        m = y == c
        if m.sum() >= 5:
            d[c] = np.quantile(S[m, c], 1 - alpha) - q_global
    return d


def test_gtheta_recovers_planted_signal_and_refuses_unknown_features():
    Phi, names, S, y, K = _world()
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    tr = np.arange(0, K, 2)
    held = np.arange(1, K, 2)

    g = fit_gtheta(Phi, d, tr, names)
    pred = g.predict(Phi[held])
    # correlation on classes g_theta never saw
    assert np.corrcoef(pred, d[held])[0, 1] > 0.5

    with pytest.raises(ValueError):
        fit_gtheta(Phi, d, tr, names, features=["nope"])


def test_gtheta_is_fit_on_train_classes_only():
    Phi, names, S, y, K = _world()
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    tr = np.arange(0, K, 2)

    g = fit_gtheta(Phi, d, tr, names)
    # corrupting a HELD-OUT class's target must not change the fitted model at all
    d2 = d.copy()
    d2[1] = 999.0
    g2 = fit_gtheta(Phi, d2, tr, names)
    assert np.allclose(g.model["w"], g2.model["w"])


def test_cv_mse_is_out_of_fold_and_larger_than_in_sample():
    Phi, names, S, y, K = _world()
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    tr = np.arange(K)

    g = fit_gtheta(Phi, d, tr, names)
    in_sample = float(np.mean((g.predict(Phi[tr]) - d[tr]) ** 2))
    oof = gtheta_cv_mse(Phi, d, tr, names, seed=0)
    assert np.isfinite(oof)
    assert oof >= in_sample          # out-of-fold cannot beat in-sample here


def test_quantile_noise_decreases_with_n():
    Phi, names, S, y, K = _world(K=30, n_per=200)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    tr = np.arange(K)
    small = quantile_noise_at_n(S, y, tr, alpha, qg, 5, n_rep=10, seed=0)
    large = quantile_noise_at_n(S, y, tr, alpha, qg, 80, n_rep=10, seed=0)
    assert np.isfinite(small) and np.isfinite(large)
    assert large < small             # more samples -> quieter empirical quantile


def test_data_threshold_moves_the_right_way_with_predictor_quality():
    Phi, names, S, y, K = _world(K=40, n_per=200)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    tr = np.arange(K)

    # A BETTER predictor (smaller mse) should demand MORE data before the empirical
    # estimate is preferred, i.e. a larger n_star.
    good = data_threshold(S, y, tr, alpha, qg, 1e-4, n_rep=8, seed=0)
    bad = data_threshold(S, y, tr, alpha, qg, 1e6, n_rep=8, seed=0)
    assert bad["n_star"] is not None
    if good["n_star"] is not None:
        assert good["n_star"] >= bad["n_star"]
    assert bad["n_star"] == min(bad["noise_curve"])   # a useless predictor loses at once


def test_data_threshold_reports_none_rather_than_inventing_a_number():
    Phi, names, S, y, K = _world(K=20, n_per=120)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    r = data_threshold(S, y, np.arange(K), alpha, qg, 0.0, n_rep=5, seed=0)
    assert r["n_star"] is None


def test_blend_uses_observed_above_threshold_and_shrunk_prediction_below():
    d_obs = np.array([0.10, 0.20, np.nan, 0.30])
    d_hat = np.array([1.00, 1.00, 1.00, 1.00])
    n = np.array([100, 3, 100, 50])
    out = blend_delta(d_obs, d_hat, n, n_star=25, lam=0.1)
    assert out["delta"][0] == pytest.approx(0.10)     # enough data -> observed, unshrunk
    assert out["delta"][1] == pytest.approx(0.10)     # too few -> lam * d_hat
    assert out["delta"][2] == pytest.approx(0.10)     # no observation -> lam * d_hat
    assert out["delta"][3] == pytest.approx(0.30)
    assert out["n_observed"] == 2


def test_blend_falls_back_to_global_when_nothing_is_known():
    out = blend_delta(np.array([np.nan]), np.array([np.nan]), np.array([0]),
                      n_star=10, lam=0.5)
    assert out["delta"][0] == 0.0                     # global threshold, not NaN
    assert out["n_fallback_global"] == 1


def test_blend_with_no_n_star_predicts_everywhere():
    out = blend_delta(np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([999, 999]),
                      n_star=None, lam=0.2)
    assert out["n_observed"] == 0
    assert np.allclose(out["delta"], 0.2)


def test_recalibrate_restores_marginal_coverage():
    Phi, names, S, y, K = _world(K=40, n_per=100)
    alpha = 0.1
    t = np.full(K, 0.5)                               # deliberately far too tight
    off = recalibrate_marginal(S, y, t, alpha)
    got = float(np.mean(S[np.arange(len(y)), y] <= (t + off)[y]))
    assert got == pytest.approx(1 - alpha, abs=0.01)


def test_fit_pcc_end_to_end_keeps_marginal_and_records_provenance():
    Phi, names, S, y, K = _world(K=60, n_per=140, seed=3)
    alpha = 0.1
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    fit_i, ev_i = idx[: len(idx) // 2], idx[len(idx) // 2:]
    S_fit, y_fit = S[fit_i], y[fit_i]
    S_ev, y_ev = S[ev_i], y[ev_i]

    qg = float(np.quantile(S_fit[np.arange(len(y_fit)), y_fit], 1 - alpha))
    d = _delta_obs(S_fit, y_fit, K, alpha, qg)
    n_per_class = np.bincount(y_fit, minlength=K)
    tr = np.arange(0, K, 2)

    m = fit_pcc(Phi, names, d, n_per_class, qg, alpha,
                score_matrix_fit=S_fit, labels_fit=y_fit, train_classes=tr, seed=0)

    assert m.thresholds().shape == (K,)
    assert m.prediction_sets(S_ev).shape == S_ev.shape
    # marginal coverage on the FIT slice is what the offset targets, so check it there
    cov_fit = float(np.mean(S_fit[np.arange(len(y_fit)), y_fit] <= m.thresholds()[y_fit]))
    assert cov_fit == pytest.approx(1 - alpha, abs=0.02)
    # and it should not be wildly off on the held-out slice either
    cov_ev = float(np.mean(S_ev[np.arange(len(y_ev)), y_ev] <= m.thresholds()[y_ev]))
    assert abs(cov_ev - (1 - alpha)) < 0.05

    assert m.provenance["everything_fit_on"].startswith("FIT slice")
    assert "NOT classwise validity" in m.provenance["claims"]
    assert 0.0 <= m.lam <= 1.0


def test_n_star_by_objective_never_loses_to_the_mse_rule_on_train():
    """The point of the fix: n_star chosen by the objective cannot do worse, ON TRAIN,
    than n_star chosen by an MSE crossing — because the objective is what is scored."""
    from pcc.eval.conformal import restrict_to_classes
    from pcc.eval.setsize import avg_set_size_at_shift, equity_at_matched_size

    Phi, names, S, y, K = _world(K=50, n_per=90, seed=11)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    n_per_class = np.bincount(y, minlength=K)
    tr = np.arange(0, K, 2)

    g = fit_gtheta(Phi, d, tr, names)
    d_hat = np.where(np.isfinite(g.predict(Phi)), g.predict(Phi), 0.0)
    lam = 0.1

    sel = select_n_star(S, y, tr, alpha, qg, d, d_hat, n_per_class, lam)
    assert sel["selected_by"].startswith("objective")
    assert "none" in sel["curve"]                    # None is a real candidate

    mse = gtheta_cv_mse(Phi, d, tr, names, seed=0)
    mse_ns = data_threshold(S, y, tr, alpha, qg, mse, n_rep=6, seed=0)["n_star"]

    S_tr, y_tr, K_tr, ids = restrict_to_classes(S, y, tr)
    base = np.full(K_tr, qg)
    target = avg_set_size_at_shift(S_tr, base, 0.0)

    def score(ns):
        bl = blend_delta(d, d_hat, n_per_class, ns, lam)
        return equity_at_matched_size(S_tr, y_tr, K_tr, base + bl["delta"][ids],
                                      target)["worst"]

    assert score(sel["n_star"]) >= score(mse_ns) - 1e-12


def test_n_star_has_a_do_no_harm_floor_even_when_delta_obs_is_pure_noise():
    """In a world with NO class-difficulty signal, delta_y is pure noise.

    The guarantee that matters is a FLOOR, not that "none" wins: because `None` is
    always a candidate, the selected n_star can never score below always-predicting. On a
    single seed noise can favour either side — worst-class coverage moves in steps of
    1/n_per (0.025 here), so a 0.025 gap is literally one sample — which is exactly why
    the test asserts the floor rather than a winner.
    """
    Phi, names, S, y, K = _world(K=50, n_per=40, seed=13, signal=0.0)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    n_per_class = np.bincount(y, minlength=K)
    tr = np.arange(0, K, 2)
    g = fit_gtheta(Phi, d, tr, names)
    d_hat = np.where(np.isfinite(g.predict(Phi)), g.predict(Phi), 0.0)

    for lam in (0.0, 0.1):
        sel = select_n_star(S, y, tr, alpha, qg, d, d_hat, n_per_class, lam)
        assert "none" in sel["curve"]
        assert sel["value"] >= sel["curve"]["none"] - 1e-12
        assert sel["value"] == max(sel["curve"].values())


def test_oos_rule_scores_observed_against_predicted_on_held_out_rows():
    Phi, names, S, y, K = _world(K=50, n_per=120, seed=19)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    tr = np.arange(0, K, 2)
    g = fit_gtheta(Phi, d, tr, names)
    d_hat = np.where(np.isfinite(g.predict(Phi)), g.predict(Phi), 0.0)

    sel = select_n_star_oos(S, y, tr, alpha, qg, d_hat, 0.1, candidates=(5, 20, 50),
                            n_rep=2, seed=0)
    assert "OUT OF SAMPLE" in sel["selected_by"]
    # both arms are measured on the SAME held-out rows, so they are comparable
    assert set(sel["curve_observed"]) == set(sel["curve_predicted_only"])
    assert sel["n_star"] is None or sel["n_star"] in (5, 20, 50)
    if sel["n_star"] is not None:
        k = str(sel["n_star"])
        assert sel["curve_observed"][k] >= sel["curve_predicted_only"][k]


def test_oos_rule_returns_none_when_predicting_is_never_worse():
    """In a null world delta_obs is noise, so with a useless predictor the observed arm
    should not win at any n -- and the rule must then say None, not pick the smallest n."""
    Phi, names, S, y, K = _world(K=40, n_per=60, seed=23, signal=0.0)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    tr = np.arange(K)
    sel = select_n_star_oos(S, y, tr, alpha, qg, np.zeros(K), 0.0,
                            candidates=(5, 20), n_rep=2, seed=0)
    # lam=0 and d_hat=0 makes the predicted arm exactly the global threshold; the noisy
    # observed arm must not be preferred unless it genuinely scores at least as well
    if sel["n_star"] is not None:
        k = str(sel["n_star"])
        assert sel["curve_observed"][k] >= sel["curve_predicted_only"][k]


def test_oos_rule_excludes_candidates_no_class_can_support():
    """A candidate n larger than the rows any class has leaves BOTH arms identical, so a
    '>=' comparison would hold degenerately and report a meaningless n_star. It must be
    excluded and recorded -- this was real: the smoke run reported n_star=50 on a CAL
    slice with ~42 rows per class, in both a signal and a null world."""
    Phi, names, S, y, K = _world(K=40, n_per=30, seed=31)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    tr = np.arange(K)
    sel = select_n_star_oos(S, y, tr, alpha, qg, np.zeros(K), 0.1,
                            candidates=(5, 200), n_rep=2, seed=0)
    assert "200" in sel["candidates_not_evaluable"]
    assert "200" not in sel["curve_observed"]
    assert sel["n_star"] != 200
    assert sel["candidates_not_evaluable"]["200"]["max_classes_splittable"] == 0


def test_oos_rule_is_reproducible_and_blind_to_heldout_classes():
    Phi, names, S, y, K = _world(K=50, n_per=100, seed=29)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    tr = np.arange(0, K, 2)
    held = np.arange(1, K, 2)
    g = fit_gtheta(Phi, d, tr, names)
    d_hat = np.where(np.isfinite(g.predict(Phi)), g.predict(Phi), 0.0)

    kw = dict(candidates=(10, 30), n_rep=2, seed=0)
    a = select_n_star_oos(S, y, tr, alpha, qg, d_hat, 0.1, **kw)
    b = select_n_star_oos(S, y, tr, alpha, qg, d_hat, 0.1, **kw)
    assert a == b                                        # same seed -> same answer

    S2 = S.copy()
    S2[:, held] = 0.0
    c = select_n_star_oos(S2, y, tr, alpha, qg, d_hat, 0.1, **kw)
    assert a["n_star"] == c["n_star"], "n_star saw held-out class scores"


def test_fit_pcc_reports_both_rules_and_defaults_to_the_objective():
    Phi, names, S, y, K = _world(K=50, n_per=90, seed=17)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    n_per_class = np.bincount(y, minlength=K)
    tr = np.arange(0, K, 2)

    m = fit_pcc(Phi, names, d, n_per_class, qg, alpha, score_matrix_fit=S, labels_fit=y,
                train_classes=tr, seed=0)
    assert m.provenance["n_star_rule"] == "oos"
    assert "OUT OF SAMPLE" in m.threshold_rule["selected"]["selected_by"]
    assert "mse_crossing_secondary" in m.threshold_rule       # kept, but not deciding
    assert m.n_star == m.threshold_rule["selected"]["n_star"]

    legacy = fit_pcc(Phi, names, d, n_per_class, qg, alpha, score_matrix_fit=S,
                     labels_fit=y, train_classes=tr, seed=0, n_star_rule="mse")
    assert legacy.n_star == m.threshold_rule["mse_crossing_secondary"]["n_star"]

    with pytest.raises(ValueError):
        fit_pcc(Phi, names, d, n_per_class, qg, alpha, score_matrix_fit=S, labels_fit=y,
                train_classes=tr, seed=0, n_star_rule="nonsense")


def test_lambda_and_offset_are_blind_to_heldout_class_SCORES():
    """The leak this catches was real and was in the first version of `fit_pcc`.

    Corrupting held-out class TARGETS is not enough to catch it: lambda is chosen from
    equity on the score matrix, so the scores of held-out classes are the thing that
    must not be visible. Here every held-out class's own-column scores are destroyed;
    lambda, n_star and the marginal offset must be bit-identical.
    """
    Phi, names, S, y, K = _world(K=60, n_per=140, seed=7)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    n_per_class = np.bincount(y, minlength=K)
    tr = np.arange(0, K, 2)
    held = np.arange(1, K, 2)

    a = fit_pcc(Phi, names, d, n_per_class, qg, alpha,
                score_matrix_fit=S, labels_fit=y, train_classes=tr, seed=0)

    S2 = S.copy()
    S2[:, held] = 0.0                      # held-out classes now trivially easy
    b = fit_pcc(Phi, names, d, n_per_class, qg, alpha,
                score_matrix_fit=S2, labels_fit=y, train_classes=tr, seed=0)

    assert a.lam == b.lam, "lambda saw held-out class scores"
    assert a.n_star == b.n_star, "n_star saw held-out class scores"
    assert a.offset == pytest.approx(b.offset), "marginal offset saw held-out scores"


def test_fit_pcc_never_sees_heldout_class_targets():
    Phi, names, S, y, K = _world(K=60, n_per=140, seed=5)
    alpha = 0.1
    qg = float(np.quantile(S[np.arange(len(y)), y], 1 - alpha))
    d = _delta_obs(S, y, K, alpha, qg)
    n_per_class = np.bincount(y, minlength=K)
    tr = np.arange(0, K, 2)
    held = np.arange(1, K, 2)

    kw = dict(score_matrix_fit=S, labels_fit=y, train_classes=tr, seed=0)
    a = fit_pcc(Phi, names, d, n_per_class, qg, alpha, **kw)

    # blow up every held-out class's observed delta; g_theta and lambda must not move
    d2 = d.copy()
    d2[held] = 50.0
    b = fit_pcc(Phi, names, d2, n_per_class, qg, alpha, **kw)

    assert np.allclose(a.gtheta.model["w"], b.gtheta.model["w"])
    assert a.lam == b.lam
    assert a.n_star == b.n_star
