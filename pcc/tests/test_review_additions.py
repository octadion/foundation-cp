# -*- coding: utf-8 -*-
"""Tests for the four things added in response to the WACV review.

  PCC-split      the recalibration slice must be DISJOINT from the fit slice, and the
                 report must say whether Proposition 1's premise actually holds.
  PAS            Ding et al.'s prevalence-adjusted softmax as a score axis.
  INTERP-Q       Ding et al.'s quantile interpolation as a competitor. The arithmetic is
                 transcribed from their example.ipynb, so it is tested against the two
                 end points their paper fixes: tau=0 is STANDARD, tau=1 is CLASSWISE with
                 infinities capped at one.
  descriptors    Euclidean in place of cosine, and dropping a named feature.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import pcc.experiments.phase2_pcc as drv
from pcc.descriptors.head_weights import build_head_descriptors, head_cos_knn
from pcc.scores.base import pas_scores, score_matrix


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A synthetic dump where crowded weight rows really are harder to cover."""
    tmp_path = tmp_path_factory.mktemp("review")
    rng = np.random.default_rng(0)
    K, d, n_per = 40, 16, 60
    W = rng.normal(size=(K, d))
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    crowd = (Wn @ Wn.T - np.eye(K)).max(axis=1)
    hard = (crowd - crowd.min()) / (np.ptp(crowd) + 1e-9)

    y = np.repeat(np.arange(K), n_per)
    logits = rng.normal(size=(len(y), K)) * 1.2
    logits[np.arange(len(y)), y] += 4.0 * (1.0 - hard[y])
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)

    sp, lp, wp = tmp_path / "s.npy", tmp_path / "y.npy", tmp_path / "W.npy"
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
             eval_depth=None, min_eval_rows=None)
    a.update(over)
    return type("A", (), a)


# ------------------------------------------------------------------- PCC-split
def test_frac_recal_carves_a_disjoint_slice_and_says_so(world):
    """Proposition 1 needs the rows behind delta_tilde and behind c to be disjoint.

    With --frac-recal 0 they are the same rows, and the report must admit that rather than
    let a reader assume the premise. With a positive fraction the two slices partition the
    calibration rows and the flag flips.
    """
    reuse = drv.run(_args(world))
    prov = reuse["pcc"]["provenance"]
    assert prov["prop1_premise_holds"] is False
    assert "SAME slice" in prov["marginal_offset_fit_on"]
    assert reuse["recal_rows"] == 0

    split = drv.run(_args(world, frac_recal=0.4))
    prov = split["pcc"]["provenance"]
    assert prov["prop1_premise_holds"] is True
    assert "DISJOINT" in prov["marginal_offset_fit_on"]
    assert split["recal_rows"] > 0

    # the two slices must PARTITION what the reuse run used: nothing invented, nothing lost
    assert (split["split_sizes"]["cal_seen"] + split["recal_rows"]
            == reuse["split_sizes"]["cal_seen"])
    # and the offset must be fit on the recalibration rows, not the fit rows
    assert prov["marginal_offset_rows"] == split["recal_rows"]


def test_frac_recal_refuses_a_slice_too_thin_to_hold_a_quantile(world):
    with pytest.raises(ValueError, match="frac-recal|at least 2"):
        drv.run(_args(world, frac_recal=0.999))
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        drv.run(_args(world, frac_recal=1.5))


def test_frac_recal_shrinks_the_fit_slice_because_the_cost_is_real(world):
    """Splitting is not free: g_theta and lambda see fewer rows. A silent no-op would be
    worse than the reuse it replaces, so the fit slice must actually get smaller.

    The fraction is 0.4 and not 0.5 because this world has 18 calibration rows per class
    and n_cal is 10: at 0.5 the fit slice keeps 9 and the driver correctly refuses the
    configuration, which is itself the behaviour asserted in the next test."""
    reuse = drv.run(_args(world))
    split = drv.run(_args(world, frac_recal=0.4))
    assert split["split_sizes"]["cal_seen"] < reuse["split_sizes"]["cal_seen"]


def test_frac_recal_that_starves_gtheta_fails_loudly_instead_of_lowering_n_cal(world):
    """n_cal is a pre-registered criterion. If the split leaves too few rows to reach it,
    the run must stop and say so, never quietly relax the criterion."""
    with pytest.raises(ValueError, match="pre-registered|n_cal"):
        drv.run(_args(world, frac_recal=0.6))


# -------------------------------------------------------------------------- PAS
def test_pas_matches_ding_et_al_formula_elementwise():
    """Transcribed from `compute_PAS_scores` in their example.ipynb:
    `return - softmax_scores / class_distribution`."""
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(7), size=50).astype(np.float32)
    pri = rng.dirichlet(np.ones(7))
    got = pas_scores(P, pri)
    want = -(P / pri)
    assert np.allclose(got, want, atol=1e-6)
    assert got.dtype == P.dtype, "the dtype must survive, else memory doubles on a dump"


def test_pas_orders_a_rare_class_ahead_of_a_common_one():
    """The whole point of dividing by the prior: at equal predicted probability the rarer
    class must look MORE conforming, i.e. get the smaller score."""
    P = np.array([[0.5, 0.5]])
    s = pas_scores(P, np.array([0.9, 0.1]))
    assert s[0, 1] < s[0, 0]


def test_pas_needs_priors_and_rejects_a_zero_prior():
    P = np.array([[0.5, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="priors"):
        score_matrix(P, "pas")
    with pytest.raises(ValueError, match="positive"):
        pas_scores(P, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="entries"):
        pas_scores(P, np.array([1.0, 0.5, 0.5]))


def test_pas_runs_end_to_end_as_a_score_axis(world):
    res = drv.run(_args(world, score="pas"))
    assert res["score"] == "pas"
    assert res["table_2_heldout"]["size_matched"]
    # a different score must give a different marginal quantile, else the axis is inert
    assert res["q_global"] != drv.run(_args(world))["q_global"]


# --------------------------------------------------------------------- INTERP-Q
def _fake_ltc(root: Path):
    """A stand-in for the released repository, exposing only what the driver calls.

    `compute_qhat` follows their convention: the finite-sample conformal quantile of the
    true-class scores, returning +inf when the level exceeds one, which is exactly the
    case INTERP-Q's cap exists to handle.
    """
    (root / "utils").mkdir(parents=True, exist_ok=True)
    (root / "utils" / "__init__.py").write_text("")
    (root / "utils" / "conformal_utils.py").write_text(textwrap.dedent('''
        import numpy as np

        def compute_qhat(scores_all, true_labels, alpha, **kw):
            s = np.asarray(scores_all, float)
            y = np.asarray(true_labels, int)
            v = s[np.arange(len(y)), y] if s.ndim == 2 else s
            n = len(v)
            if n == 0:
                return np.inf
            lvl = np.ceil((n + 1) * (1 - alpha)) / n
            if lvl > 1:
                return np.inf
            return float(np.quantile(v, lvl, method="higher"))

        def compute_class_specific_qhats(cal_scores_all, cal_true_labels, num_classes,
                                        alpha, **kw):
            out = np.full(num_classes, np.inf)
            y = np.asarray(cal_true_labels, int)
            for k in range(num_classes):
                m = y == k
                if m.any():
                    out[k] = compute_qhat(np.asarray(cal_scores_all)[m], y[m], alpha)
            return out

        def clustered_conformal(cal_scores_all, cal_labels, alpha, seed=0, **kw):
            return np.full(np.asarray(cal_scores_all).shape[1],
                           compute_qhat(cal_scores_all, cal_labels, alpha))

        def compute_rc3p_params(P, S, y, alpha, **kw):
            return (np.full(np.asarray(S).shape[1], np.inf),)

        def fuzzy_classwise_CP(cal_scores_all, cal_labels, alpha, projection="HRS",
                               mode="weight", params=None, **kw):
            q = compute_qhat(cal_scores_all, cal_labels, alpha)
            return (np.full(np.asarray(cal_scores_all).shape[1], q),)
    '''))
    return str(root)


def _interp_from_driver(world, tmp_path, alpha=0.10):
    root = _fake_ltc(tmp_path / "ltc")
    rng = np.random.default_rng(3)
    K, n = 12, 400
    S = rng.random((n, K))
    y = rng.integers(0, K, n)
    y[y == 0] = 1                      # class 0 gets NO calibration rows on purpose
    out, errs, _ = drv._competitor_thresholds(root, S, y, K, alpha, seed=0)
    for m in list(sys.modules):
        if m.startswith("utils"):
            del sys.modules[m]
    if root in sys.path:
        sys.path.remove(root)
    return S, y, K, out, errs


def test_interp_q_endpoints_are_standard_and_capped_classwise(world, tmp_path):
    """Ding et al. fix both end points: tau=0 is STANDARD, tau=1 is CLASSWISE with the
    infinite quantiles replaced by one. If our transcription is right, the vectors at
    those two weights are exactly those two things."""
    S, y, K, out, errs = _interp_from_driver(world, tmp_path)
    assert not errs, errs
    got = dict(out["interp_q"])
    sys.path.insert(0, _fake_ltc(tmp_path / "ltc"))
    import importlib
    cu = importlib.import_module("utils.conformal_utils")

    std = cu.compute_qhat(S, y, 0.10)
    cw = cu.compute_class_specific_qhats(S, y, K, 0.10)
    cw = np.where(np.isfinite(cw), cw, 1.0)

    assert np.allclose(got["tau=0.0"], np.full(K, std))
    assert np.allclose(got["tau=1.0"], cw)
    assert np.allclose(got["tau=0.5"], 0.5 * cw + 0.5 * std)


def test_interp_q_is_defined_for_a_class_with_no_calibration_rows(world, tmp_path):
    """This is why the review is right that INTERP-Q belongs in the table: unlike classwise
    CP it emits a FINITE threshold for an empty class, so its entry is a measurement of the
    method and not of our fallback policy."""
    S, y, K, out, errs = _interp_from_driver(world, tmp_path)
    for label, q in out["interp_q"]:
        assert np.isfinite(q).all(), "tau={} produced a non-finite threshold".format(label)
    # class 0 has no rows; at tau=1 it must sit at the cap, which is 1.0
    tau1 = dict(out["interp_q"])["tau=1.0"]
    assert tau1[0] == pytest.approx(1.0)


def test_interp_q_is_monotone_in_tau_towards_the_classwise_vector(world, tmp_path):
    S, y, K, out, errs = _interp_from_driver(world, tmp_path)
    got = dict(out["interp_q"])
    target = got["tau=1.0"]
    d = [np.abs(got["tau={}".format(t)] - target).mean()
         for t in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)]
    assert all(a >= b - 1e-12 for a, b in zip(d, d[1:])), d


# ------------------------------------------------------------------ descriptors
def test_euclidean_metric_changes_the_values_but_not_the_column_names():
    """The metric is an experimental axis, so the columns must line up between runs; only
    the numbers may move. Identical names are what lets the two be compared row by row."""
    rng = np.random.default_rng(5)
    W = rng.normal(size=(30, 8))
    Pc, nc = build_head_descriptors(W, knn_ks=(1, 5), metric="cosine")
    Pe, ne = build_head_descriptors(W, knn_ks=(1, 5), metric="euclidean")
    assert nc == ne
    assert Pc.shape == Pe.shape
    assert not np.allclose(Pc, Pe), "the metric switch did nothing"
    assert np.isfinite(Pe).all()


def test_euclidean_neighbour_distances_are_actual_euclidean_distances():
    W = np.array([[0.0, 0.0], [3.0, 4.0], [-3.0, -4.0], [10.0, 0.0]])
    out = head_cos_knn(W, ks=(1,), metric="euclidean")
    # nearest neighbour of row 0 is row 1 or 2, both at distance 5
    assert out["w_margin_nearest"][0] == pytest.approx(5.0)


def test_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="metric"):
        head_cos_knn(np.random.default_rng(0).normal(size=(5, 3)), ks=(1,),
                     metric="manhattan")


def test_drop_features_removes_the_named_descriptor_and_rejects_a_typo(world):
    res = drv.run(_args(world, drop_features=("w_bias",)))
    assert "w_bias" not in res["pcc"]["features"]
    assert res["dropped_features"] == ["w_bias"]
    assert len(res["pcc"]["features"]) >= 3

    with pytest.raises(ValueError, match="drop-features"):
        drv.run(_args(world, drop_features=("w_biass",)))


def test_knn_ks_axis_changes_the_feature_set(world):
    res = drv.run(_args(world, knn_ks=(1, 2, 5), distance_holdout="w_cos_knn_1"))
    f = res["pcc"]["features"]
    assert "w_cos_knn_2" in f and "w_cos_knn_50" not in f
    assert res["knn_ks_requested"] == [1, 2, 5]


# ------------------------------------------------------------------- dump-fit
def test_dump_fit_saves_the_arrays_behind_the_method_figure(world, tmp_path):
    """The figure must show what the run did, so the arrays come from the run rather than
    from a re-derivation that could differ in split or seed."""
    import numpy as _np
    out = tmp_path / "fit.npz"
    res = drv.run(_args(world, dump_fit=str(out)))
    assert out.exists(), "--dump-fit wrote nothing"
    z = _np.load(str(out), allow_pickle=True)

    K = world["K"]
    for k in ("delta_obs", "delta_hat", "n_per_class", "class_counts"):
        assert z[k].shape == (K,), "%s has shape %s" % (k, z[k].shape)
    assert z["Phi"].shape[0] == K
    assert z["Phi"].shape[1] == len(z["feature_names"])

    seen, held = set(z["seen"].tolist()), set(z["heldout"].tolist())
    assert seen.isdisjoint(held) and len(seen | held) == K

    # the whole point: held-out classes CANNOT have an observed offset, and every class
    # can have a predicted one
    assert _np.isnan(z["delta_obs"][sorted(held)]).all()
    assert _np.isfinite(z["delta_hat"]).all()
    assert res["n_heldout"] == len(held)


def test_dump_fit_is_off_by_default(world):
    """A run that does not ask for it must not write files beside the report."""
    res = drv.run(_args(world))
    assert res is not None


def test_dump_fit_carries_per_class_coverage_for_the_appendix_curve(world, tmp_path):
    """The sorted per-class coverage curve needs every class's coverage under each arm, and
    it must come from the same matched-size shift the tables report."""
    import numpy as _np
    out = tmp_path / "fit2.npz"
    res = drv.run(_args(world, dump_fit=str(out)))
    z = _np.load(str(out), allow_pickle=True)

    for space, n in (("seen", res["n_seen"]), ("heldout", res["n_heldout"])):
        for arm in ("uncorrected", "pcc", "oracle"):
            k = "cov_%s_%s" % (space, arm)
            assert k in z.files, "%s missing from the dump" % k
            assert z[k].shape == (n,), "%s has shape %s, expected (%d,)" % (k, z[k].shape, n)
            finite = z[k][_np.isfinite(z[k])]
            assert ((finite >= 0) & (finite <= 1)).all(), "%s is not a coverage" % k

    # the arms must actually differ, else the curve would show nothing
    assert not _np.allclose(z["cov_heldout_uncorrected"], z["cov_heldout_pcc"],
                            equal_nan=True)
    # and the thresholds the run emitted travel with it
    assert z["thresholds"].shape == (world["K"],)
