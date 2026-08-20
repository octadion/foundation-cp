# -*- coding: utf-8 -*-
"""Tests for the alternative g_theta heads, added to answer the review's question about
nonlinear predictors.

The point of the option is to find out whether the linear head is what limits the method, so
the tests have to establish that the nonlinear heads can actually fit something the linear one
cannot -- otherwise a null result would say nothing. They also pin the two ways this could go
wrong quietly: the default path must stay exactly what it was, and the p+2 floor must count
the coefficients each head really fits, not the raw feature count.
"""
from __future__ import annotations

import numpy as np
import pytest

import pcc.experiments.phase2_pcc as drv
from pcc.tests.test_review_additions import _args, world  # noqa: F401
from pcc.method.pcc import GTHETA_KINDS, fit_gtheta


NAMES = ["f%d" % i for i in range(6)]


def _world(seed=0, n=300, curved=True):
    """Classes whose offset depends on a product and a square of the descriptors, which no
    linear head can represent."""
    rng = np.random.default_rng(seed)
    Phi = rng.normal(size=(n, len(NAMES)))
    if curved:
        d = 0.4 * Phi[:, 0] + 0.9 * Phi[:, 1] ** 2 - 0.7 * Phi[:, 2] * Phi[:, 3]
    else:
        d = 0.4 * Phi[:, 0] + 0.9 * Phi[:, 1] - 0.7 * Phi[:, 2]
    d = d + 0.05 * rng.normal(size=n)
    train = np.arange(0, n, 2)
    held = np.arange(1, n, 2)
    obs = d.copy()
    obs[held] = np.nan          # held-out classes have no observed offset, as in a real run
    return Phi, d, obs, train, held


def _oos_mse(kind, **kw):
    Phi, truth, obs, train, held = _world()
    g = fit_gtheta(Phi, obs, train, NAMES, kind=kind, **kw)
    return float(np.mean((g.predict(Phi)[held] - truth[held]) ** 2))


# ------------------------------------------------------------------ the default
def test_the_default_head_is_linear_and_unchanged():
    """Every table in the paper was produced without this flag, so the default path must
    give bit-identical predictions to naming it explicitly."""
    Phi, _, obs, train, _ = _world()
    a = fit_gtheta(Phi, obs, train, NAMES)
    b = fit_gtheta(Phi, obs, train, NAMES, kind="linear")
    assert a.kind == "linear"
    assert np.array_equal(a.predict(Phi), b.predict(Phi))
    # and the linear model keeps the flat shape other code reads directly
    assert set(("w", "mu", "sd")).issubset(a.model)


def test_a_linear_target_is_not_made_worse_by_the_nonlinear_heads():
    """On data the linear head can fit, the extra capacity must not cost much, or the
    ablation would only be measuring overfitting."""
    Phi, truth, obs, train, held = _world(curved=False)
    base = None
    for kind in GTHETA_KINDS:
        g = fit_gtheta(Phi, obs, train, NAMES, kind=kind)
        mse = float(np.mean((g.predict(Phi)[held] - truth[held]) ** 2))
        if kind == "linear":
            base = mse
        else:
            assert mse < base + 0.25, (kind, mse, base)


# --------------------------------------------------------------- the nonlinear heads
def test_the_nonlinear_heads_fit_curvature_the_linear_one_cannot():
    lin = _oos_mse("linear")
    assert _oos_mse("quadratic") < 0.25 * lin
    assert _oos_mse("kernel") < lin


def test_the_kernel_head_respects_its_width():
    """A very wide kernel flattens towards the mean, a very narrow one cannot generalise;
    if gamma did nothing the flag would be silently ignored."""
    wide = _oos_mse("kernel", gamma=1e-4)
    mid = _oos_mse("kernel", gamma=1.0)
    narrow = _oos_mse("kernel", gamma=1e4)
    assert mid < wide
    assert mid < narrow


def test_predictions_are_finite_for_classes_with_no_observed_offset():
    """The whole point is predicting for held-out classes, so they must come back with
    real numbers even though their target is NaN."""
    Phi, _, obs, train, held = _world()
    for kind in GTHETA_KINDS:
        p = fit_gtheta(Phi, obs, train, NAMES, kind=kind).predict(Phi)
        assert np.isfinite(p[held]).all(), kind


# ------------------------------------------------------------------- the floor
def test_the_class_floor_counts_the_coefficients_each_head_actually_fits():
    """Six features become 27 coefficients under the quadratic head. Checking the floor
    against six would let it run underdetermined and return whatever the solver settled
    on, which is the kind of number that looks like a result."""
    Phi, _, obs, _, _ = _world(n=60)
    # only even indices carry an observed offset in this world, and the floor counts
    # USABLE classes, so the train sets below are drawn from those
    fit_gtheta(Phi, obs, np.arange(0, 20, 2), NAMES, kind="linear")   # 10 for 6 features
    with pytest.raises(ValueError, match="too few usable TRAIN classes"):
        fit_gtheta(Phi, obs, np.arange(0, 20, 2), NAMES, kind="quadratic")

    with pytest.raises(ValueError, match="27 coefficients"):
        fit_gtheta(Phi, obs, np.arange(0, 56, 2), NAMES, kind="quadratic")   # 28, one short
    fit_gtheta(Phi, obs, np.arange(0, 58, 2), NAMES, kind="quadratic")       # 29, enough


def test_an_unknown_head_is_refused_rather_than_silently_linear():
    Phi, _, obs, train, _ = _world()
    with pytest.raises(ValueError, match="unknown g_theta kind"):
        fit_gtheta(Phi, obs, train, NAMES, kind="mlp")


# --------------------------------------------------------------------- the flag
def test_the_driver_refuses_a_head_it_cannot_fit():
    """argparse rejects an unknown head before any file is touched, so a typo cannot run a
    linear fit under a nonlinear name. A valid head gets past the parser and fails later on
    the missing dump, which is a different failure."""
    base = ["--scores", "s.npy", "--labels", "y.npy", "--dataset", "d", "--phi", "output"]
    with pytest.raises(SystemExit):
        drv.main(base + ["--gtheta", "randomforest"])
    with pytest.raises(Exception) as e:
        drv.main(base + ["--gtheta", "kernel"])
    assert not isinstance(e.value, SystemExit)


def test_run_threads_the_head_through_to_the_fitted_model(world):
    """The flag has to reach fit_pcc. It reads the attribute off the args object, so an
    args object without it must still work and must still be linear."""
    assert drv.run(_args(world))["pcc"]["gtheta"] == "linear"
    assert drv.run(_args(world, gtheta="kernel"))["pcc"]["gtheta"] == "kernel"
