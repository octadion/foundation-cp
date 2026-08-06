"""§6.4 design 5 (Amendment 8): the metric must have ORACLE HEADROOM, a constant must
be a no-op, and λ must never be selected on the held-out classes.

The design this replaces failed the first of those: its own controls recorded
`oracle +0.045` against `shuffled oracle -15.39`, i.e. a perfect δ̂ bought nothing.
"""
import numpy as np

from pcc.eval.setsize import (EQUITY_STATS, avg_set_size_at_shift,
                              equity_at_matched_size, select_shrinkage,
                              setsize_translation_shrunk, shift_to_size)
from pcc.tests.test_phase0_explain import make_world, _split


def _setup(K=80, n_per_class=160, seed=0):
    from pcc.eval.decomposition import temperature_softmax, thr_scores_from_softmax
    L, y = make_world("class", n_per_class=n_per_class, K=K, seed=seed)
    ci, ei = _split(y, seed=seed)
    S = thr_scores_from_softmax(temperature_softmax(L, 1.0))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    return S, y, ci, ei, np.sort(perm[: K // 2]), np.sort(perm[K // 2:])


def test_shift_to_size_is_exact_and_monotone():
    S, y, ci, ei, _, _ = _setup()
    thr = np.full(S.shape[1], 0.9)
    for target in (2.0, 5.0, 12.0):
        sh = shift_to_size(S[ei], thr, target)
        got = avg_set_size_at_shift(S[ei], thr, sh)
        assert abs(got - target) < 0.05, f"target {target} got {got}"
    sizes = [avg_set_size_at_shift(S[ei], thr, s) for s in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), "size must be monotone in shift"
    print("  shift_to_size hits its target and size is monotone in the shift  OK")


def test_oracle_has_headroom_on_worst_but_not_on_macro():
    """The finding that forced Amendment 8, locked down as a test."""
    from pcc.eval.conformal import restrict_to_classes
    S, y, ci, ei, tr, ho = _setup()
    S_ho, y_ho, K_ho, ids = restrict_to_classes(S[ei], y[ei], ho)
    qg = float(np.quantile(S[ci, y[ci]], 0.9))
    base = np.full(K_ho, qg)
    target = avg_set_size_at_shift(S_ho, base, 0.0)
    st = S_ho[np.arange(len(y_ho)), y_ho]
    oracle = np.array([float(np.quantile(st[y_ho == k], 0.9)) - qg
                       if (y_ho == k).sum() >= 5 else 0.0 for k in range(K_ho)])
    e0 = equity_at_matched_size(S_ho, y_ho, K_ho, base, target)
    e1 = equity_at_matched_size(S_ho, y_ho, K_ho, base + oracle, target)
    assert abs(e1["avg_set_size"] - e0["avg_set_size"]) < 1e-2, "sizes must match"
    d_worst = e1["worst"] - e0["worst"]
    d_macro = e1["macro"] - e0["macro"]
    assert d_worst > 0.05, f"worst-class must have oracle headroom, got {d_worst:+.4f}"
    assert abs(d_macro) < 0.03, f"macro should have ~none (Jensen), got {d_macro:+.4f}"
    print(f"  oracle at matched size: worst {d_worst:+.3f} (headroom) "
          f"macro {d_macro:+.3f} (none, as Jensen predicts)  OK")


def test_constant_is_a_no_op():
    from pcc.eval.conformal import restrict_to_classes
    S, y, ci, ei, tr, ho = _setup()
    S_ho, y_ho, K_ho, _ = restrict_to_classes(S[ei], y[ei], ho)
    qg = float(np.quantile(S[ci, y[ci]], 0.9))
    base = np.full(K_ho, qg)
    target = avg_set_size_at_shift(S_ho, base, 0.0)
    e0 = equity_at_matched_size(S_ho, y_ho, K_ho, base, target)
    for c in (0.02, 0.05, -0.03):
        e = equity_at_matched_size(S_ho, y_ho, K_ho, base + c, target)
        for k in EQUITY_STATS:
            assert abs(e[k] - e0[k]) < 1e-9, f"constant {c} changed {k}"
    print("  a pure constant is exactly neutral in every equity statistic  OK")


def test_shrinkage_rescues_a_noisy_delta():
    """Raw δ̂ hurts at realistic quality; a shrunk δ̂ helps ON AVERAGE.

    Averaged over noise draws deliberately. Worst-class coverage is a minimum over K
    classes, so a SINGLE draw is far too noisy to regression-test — verified: one draw
    can leave λ=0 as the optimum. The claim in `select_shrinkage`'s docstring is a mean
    over repetitions, and that is what is asserted here.
    """
    from pcc.eval.conformal import restrict_to_classes
    S, y, ci, ei, tr, ho = _setup(K=120, n_per_class=200)
    S_ho, y_ho, K_ho, _ = restrict_to_classes(S[ei], y[ei], ho)
    qg = float(np.quantile(S[ci, y[ci]], 0.9))
    base = np.full(K_ho, qg)
    target = avg_set_size_at_shift(S_ho, base, 0.0)
    st = S_ho[np.arange(len(y_ho)), y_ho]
    oracle = np.array([float(np.quantile(st[y_ho == k], 0.9)) - qg
                       if (y_ho == k).sum() >= 5 else 0.0 for k in range(K_ho)])
    e0 = equity_at_matched_size(S_ho, y_ho, K_ho, base, target)

    raws, gains, lams = [], [], []
    for rep in range(10):
        rng = np.random.default_rng(100 + rep)
        noisy = 0.55 * oracle + rng.normal(0, 0.55 * oracle.std(), K_ho)
        raws.append(equity_at_matched_size(S_ho, y_ho, K_ho, base + noisy,
                                           target)["worst"] - e0["worst"])
        sel = select_shrinkage(S_ho, y_ho, K_ho, qg, noisy, target, stat="worst")
        gains.append(sel["value"] - e0["worst"])
        lams.append(sel["lambda"])
    raw_m, gain_m = float(np.mean(raws)), float(np.mean(gains))
    assert raw_m < -0.1, f"raw delta at R2~0.3 should clearly hurt, got {raw_m:+.3f}"
    assert gain_m > 0, f"shrunk delta should help on average, got {gain_m:+.3f}"
    assert np.median(lams) < 0.5, "the useful lambda should be strongly shrunk"
    print(f"  noisy delta (R2~0.3), mean of 10 draws: raw lambda=1 -> {raw_m:+.3f}, "
          f"best-lambda -> {gain_m:+.3f} (median lambda {np.median(lams):.2f})  OK")


def test_lambda_is_selected_on_train_classes_only():
    """The discipline that keeps λ from manufacturing a result: the λ the pipeline uses
    must come from the train label space, so it need not be the held-out optimum."""
    from pcc.eval.conformal import restrict_to_classes
    S, y, ci, ei, tr, ho = _setup()
    rng = np.random.default_rng(7)
    K = S.shape[1]
    qg = float(np.quantile(S[ci, y[ci]], 0.9))
    st_all = S[ei, y[ei]]
    delta = np.zeros(K)
    for k in range(K):
        m = y[ei] == k
        if m.sum() >= 5:
            delta[k] = float(np.quantile(st_all[m], 0.9)) - qg
    dhat = 0.6 * delta + rng.normal(0, 0.6 * delta.std(), K)

    res = setsize_translation_shrunk(S, y, 0.1, ci, ei, tr, ho, dhat, stat="worst")
    assert res["size_matched"], "average set size must be matched"
    assert res["controls"]["oracle_ceiling"] > 0.05, "metric must retain oracle headroom"
    assert abs(res["controls"]["pure_constant"]) < 1e-6, "constant must be neutral"

    S_ho, y_ho, K_ho, ids_ho = restrict_to_classes(S[ei], y[ei], ho)
    base = np.full(K_ho, qg)
    target = avg_set_size_at_shift(S_ho, base, 0.0)
    cheat = select_shrinkage(S_ho, y_ho, K_ho, qg, dhat[ids_ho], target, stat="worst")
    assert res["lambda_selected_on_train"] in res["lambda_curve_train"]
    print(f"  lambda from TRAIN = {res['lambda_selected_on_train']:.2f}; "
          f"held-out optimum would have been {cheat['lambda']:.2f} "
          f"(not used -- that is the point)")
    print(f"  held-out delta worst {res['delta']['worst']:+.3f}  "
          f"oracle ceiling {res['controls']['oracle_ceiling']:+.3f}  "
          f"raw lambda=1 {res['controls']['raw_delta_lambda1']:+.3f}  OK")


if __name__ == "__main__":
    test_shift_to_size_is_exact_and_monotone()
    test_oracle_has_headroom_on_worst_but_not_on_macro()
    test_constant_is_a_no_op()
    test_shrinkage_rescues_a_noisy_delta()
    test_lambda_is_selected_on_train_classes_only()
    print("all 5 sec-6.4 (Amendment 8) tests passed")
