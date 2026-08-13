# `pcc/method/` — g_θ

> **GATE STATUS: PASSED (2026-08-11). This directory is UNBLOCKED.**
>
> Required by §10 before the first line of `g_θ`: the pre-registered criteria and
> the gate report, linked. Both below. The blocking text is kept underneath —
> unchanged — because it states the standard this evidence had to meet.

## Evidence that unblocks this directory

- **Pre-registration:** [`reports/prereg_imagenet_gate.md`](../reports/prereg_imagenet_gate.md)
  (Amendments 10–11), criteria written before the run.
- **Result:** [`reports/05_imagenet_gate_ccc_imagenet.json`](../reports/05_imagenet_gate_ccc_imagenet.json)
  — CCC ImageNet dump, 1000 classes, 1,153,051 rows, produced by
  [`notebooks/05_imagenet_gate.ipynb`](../../notebooks/05_imagenet_gate.ipynb) and **committed
  2026-08-13** (`written_at 2026-08-13T02:11:07`, run at `git_commit e63f73e`). Every number in the
  table below is checkable against that file; the earlier gap — evidence existing only as pasted run
  output — is closed.
- **Held-out-class fallback, frozen before results:** [`reports/fallback_policy.md`](../reports/fallback_policy.md)

| Criterion | Required | Output-space φ | **Head-weight φ (`w_y`)** |
|---|---|---|---|
| A — reliability `r_δ` | ≥ 0.3 | 0.829 [0.827, 0.831] | same target |
| B — normalized R², class-level CI | CI low > 0 | +0.4975 [+0.461, +0.534] | **+0.3880** [+0.345, +0.439] |
| C — beats log-prevalence | Holm-corrected | p ≤ 0.001 (102σ) | p ≤ 0.001 (73σ) |
| C — beats distance baseline | Holm-corrected | p ≤ 0.001 (84σ) | p ≤ 0.001 (83σ) |
| §6.4 — δ̂ buys worst-class equity | CI low > 0, > null | +0.1173, 76% of oracle | **+0.0643** [+0.0536, +0.0751], 42% of oracle |

## Which descriptor family `g_θ` is built on, and why

**The head-weight family, not output-space.** Output-space φ passes with a larger
R², but φ and δ_y are both derived from the *same* score matrix, so that result is
closer to distributional estimation than to geometric extrapolation — measured, not
suspected: the distance baseline alone explains ~0.155 of its R² 0.497. Head weights
break that: they are *parameters*, carry zero sampling noise, exist for classes with
**no labeled samples at all**, need no images and no GPU — and on ImageNet they came
from a *different* model than the one that produced the scores.

Two constraints this evidence puts on the implementation:

1. **Shrinkage is part of the method, not a tuning detail.** Raw δ̂ at λ=1 *harms*
   worst-class equity (−0.508 for head φ, −0.234 for output-space). λ must be selected
   on training classes only.
2. **The honest number is 42%, not 76%.** An exogenous descriptor buys less than a
   circular one. Both belong in the paper, side by side.

## Implementation — [`pcc.py`](pcc.py), 13 tests

| Piece | What it is |
|---|---|
| `fit_gtheta` | ridge φ → δ̂, fit on TRAIN classes only |
| `gtheta_cv_mse` | out-of-fold error of g_θ, within TRAIN classes |
| `quantile_noise_at_n` | MSE of the empirical δ_y from only `n` samples |
| `data_threshold` | **§6.7**: smallest `n` where the observed δ_y beats the predicted one |
| `blend_delta` | observed above `n_star`, `λ·δ̂` below, global threshold when neither exists |
| `recalibrate_marginal` | one scalar offset restoring marginal coverage |
| `fit_pcc` | the facade, plus a `provenance` dict recording where every free parameter was fit |

**§6.7 is derived, not picked.** `n_star` is where the sampling noise of the empirical
class quantile crosses g_θ's own out-of-fold error — both measured on TRAIN classes.
When g_θ wins at every tested `n`, `n_star` is `None` and that is reported, not turned
into a number.

**Shrinkage applies only to the predicted part.** Where δ_y is observed with enough
samples it is a direct estimate; shrinking it toward the global threshold would discard
information the data does contain.

### `n_star`: one flaw fixed, a second one found and NOT yet fixed (2026-08-13)

**Flaw 1 — wrong currency. FIXED.** `data_threshold` chose `n_star` by crossing mean
squared errors, while the objective is worst-class equity at matched set size. Same
mistake Amendment 8 records for λ: a worst-class objective is governed by the *largest*
error, not the mean squared one. `select_n_star` now chooses by the objective on the
TRAIN label space, with `None` ("never prefer the observed value") as a real candidate,
so the selection has a do-no-harm floor by construction. The MSE crossing is still
computed and reported as a secondary, because it is informative about where the empirical
estimate becomes accurate — it just no longer decides.

Measured effect on the smoke world where φ genuinely predicts difficulty: Table 1
worst-class delta rose from **+0.26…+0.36 to +0.43…+0.57**, and the rule selected
`n_star = None` — prediction beats the observed quantile even for classes that have data,
which is itself a result worth reporting.

**Flaw 2 — in-sample optimism. FIXED.** Fixing the currency did not help the null world
at all (Table 1 stayed at −0.05…−0.17), and the reason turned out to be a second,
independent mistake: δ_obs was estimated on the CAL slice and `n_star` was selected on
that *same* slice, so the observed correction was scored on its own training data. The
optimism only disappeared on EVAL, which is where Table 1 is read.

`select_n_star_oos` (now the default, `n_star_rule="oos"`) measures each candidate `n` for
what it actually claims: estimate δ_y from `n` rows of a class, then score equity on that
class's **remaining** rows, with the predict-only arm scored on the *same* held-out rows so
the two are comparable. `n_star` is the smallest `n` whose observed arm is at least as good
as predicting; if none is, it returns `None`.

**Flaw 3 — degenerate candidates read as ties. FIXED in the same pass.** The first version
reported `n_star = 50` in both a signal and a null world. It was meaningless: CAL held ~42
rows per class, so at `n = 50` no class could be split, both arms became the identical
vector, and `>=` held degenerately. Candidates that fewer than `max(5, 10% of train
classes)` classes can support are now excluded and listed under
`candidates_not_evaluable`, with the maximum number of splittable classes recorded.

### Where that leaves the three smoke worlds

| world | λ | `n_star` | Table 1 (seen) | Table 2 (`n_y=0`) | verdict |
|---|---|---|---|---|---|
| φ predicts difficulty | 0.20–0.30 | `None` | **+0.43 … +0.57** | **+0.31 … +0.50** | LULUS |
| null, φ irrelevant | **0.000** | `None` | **+0.0000** | **+0.0000** | GAGAL |

The do-no-harm floor now holds on *both* tables in the null world, and the signal world
lost nothing to the fix. `n_star = None` there is a real result, not a fallback: on ~42
calibration rows per class, the observed per-class quantile never beat the prediction at
any evaluable `n`.

Nothing previously reported is affected: on real ImageNet with ~76 calibration samples per
class, observed per-class quantiles clearly help (`reports/baseline_reproduction.md`:
classwise `max_gap 0.173` vs standard `0.420`), so these synthetic worlds bracket the
noise range rather than describe it.

### A leak that was in the first version, found before it produced a result

`fit_pcc` initially selected λ over **all** classes, so a free parameter could tune
itself on the very classes the result is read from — exactly what Amendment 8 exists to
prevent, and what `setsize_translation_shrunk` avoids via `restrict_to_classes`. It was
not hypothetical: on the test world, corrupting held-out class scores moved the selected
λ from **0.3 to 0.5**, and the clean-data λ differed too. Fixed by restricting to the
TRAIN label space with the same helper, so the two code paths agree by construction
rather than by comment. `test_lambda_and_offset_are_blind_to_heldout_class_SCORES`
pins it, and was verified to fail against the old path.

The marginal offset is fit on TRAIN-class rows only for the same reason — and because
in deployment a class with `n_y = 0` contributes no calibration rows at all. The
consequence is stated rather than glossed: **the marginal guarantee is over the
seen-class distribution, not the full population.**

## Open, and not to be papered over

- ImageNet is **balanced**; the long-tail application claim is unresolved. The gate
  needed high-`n` classes (≥84 samples); the method targets `n_y = 0` classes — those
  are different populations, so Pl@ntNet's gate failure does **not** disqualify it for
  the method experiment.
- The ImageNet model mismatch (ResNet-50 head predicting SimCLR+probe thresholds) is a
  substantive claim about the label space and must be argued as one.

---

## Original blocking text (kept verbatim — it is the standard the above had to meet)

# DO NOT BUILD BEFORE THE GATE PASSES

This directory holds the predictive model `g_θ` that maps class descriptors
`φ(y)` to the correction `δ̂_y`. It is the Phase 2 method.

**Per `AGENTS.md` §0.2, §6.5, and §10: nothing here may be implemented until
Phase 1's gate passes** — that means criteria A (reliability ≥ 0.3), B (held-out
normalized R² clearly > 0), C (beats log-prevalence / Fargion-distance / fuzzy
baselines), **and** §6.4 (predicted δ̂_y actually shrinks set size on held-out
classes) are all satisfied, each with CIs that don't include the failing value.

> "Kapasitas model bukan masalahnya; keberadaan sinyal yang jadi masalah."
> (Model capacity is not the problem; the existence of signal is.)

An agent may **not** decide on its own to skip a failed gate. If the gate fails,
the deliverable is a clear report of that failure (`reports/`), not a workaround.

When the gate passes, the pre-registered pass criteria and the gate report must
be linked here before the first line of `g_θ` is written.
