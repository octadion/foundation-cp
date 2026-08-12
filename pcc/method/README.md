# `pcc/method/` — g_θ

> **GATE STATUS: PASSED (2026-08-11). This directory is UNBLOCKED.**
>
> Required by §10 before the first line of `g_θ`: the pre-registered criteria and
> the gate report, linked. Both below. The blocking text is kept underneath —
> unchanged — because it states the standard this evidence had to meet.

## Evidence that unblocks this directory

- **Pre-registration:** [`reports/prereg_imagenet_gate.md`](../reports/prereg_imagenet_gate.md)
  (Amendments 10–11), criteria written before the run.
- **Result:** `reports/05_imagenet_gate_ccc_imagenet.json`, produced by
  [`notebooks/05_imagenet_gate.ipynb`](../../notebooks/05_imagenet_gate.ipynb) — CCC ImageNet dump,
  1000 classes, 1,153,051 rows.
  **⚠ NOT YET IN THE REPO.** The notebook writes it on Colab; no run artifact has been committed
  back. Right now the gate evidence exists only in run output pasted into a chat, which is not a
  reproducible record. The JSON from the run that unblocked this directory must be committed before
  any table cites it. Numbers in the table below are transcribed from that run output and are
  therefore **unverifiable from this repo alone** until it lands.
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
