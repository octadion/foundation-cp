# PCC — Predicted Class Correction

Testing one claim: **can the class-level conformal correction `δ_y` be *extrapolated*
to classes that have no labeled calibration samples at all**, from geometric
descriptors of the class in representation space?

Every existing class-level method (Mondrian/classwise CP, Clustered CP, RC3P,
prevalence-adjusted softmax, Fuzzy Classwise CP, CaCT, kernel similarity)
*estimates* the correction from labeled samples of the class (or a cluster that
contains it). This repo asks whether the correction can instead be *predicted*
for held-out classes. Estimation-vs-extrapolation is the only claim that
separates this work — see `AGENTS.md`.

## Non-negotiables (read `AGENTS.md` in full)

- **The repo starts from zero.** Do not import code from prior research repos.
- **There is a GATE at the end of Phase 1 (`AGENTS.md` §6).** Nothing under
  `pcc/method/` (the `g_θ` model) may be built before criteria A, B, C **and**
  §6.4 all pass. A failed gate is a *reportable finding*, not a thing to route
  around.
- **Every reported number carries a CI / standard error** from the multi-split
  protocol (`AGENTS.md` §8).
- **Coverage validity is a unit test** — marginal coverage stays ≈ 1−α even for a
  deliberately bad `g_θ`, because `s'(x,y) = s(x,y) − δ̂_y` is still a measurable
  function of (x,y). The method is allowed to fail *predictively* without
  becoming statistically invalid.
- **Class-level splits for the extrapolation claim** — never sample-level.
  Descriptors never touch the calibration split; this is enforced by assertion
  (`pcc/eval/leakguard.py`), not convention.
- **No "label-free" claim.** Descriptors come from *labeled training data*. The
  correct claim is "no labeled calibration samples for the target class."

## Execution environment: Google Colab

Sessions are short-lived and disk is ephemeral. Everything long-running is
resumable and checkpoints to Google Drive. **Never store images — store
embeddings** (stream → forward pass → write embedding → discard image). See
`AGENTS.md` §3.

## Layout (`AGENTS.md` §4)

```
pcc/
  data/            # dataset loaders, manifests, checksums
  extract/         # forward pass -> logits + penultimate embeddings
  scores/          # THR/LAC, APS, RAPS, SAPS  (base score functions)
  descriptors/     # phi(y): class geometry descriptors  (TRAINING data only)
  targets/         # delta_y and its reliability estimators
  baselines/       # MANDATORY comparators (AGENTS.md §7)
  method/          # g_theta -- DO NOT TOUCH BEFORE THE GATE PASSES
  eval/            # multi-split protocol, bootstrap, multiple-testing, leakguard
  experiments/     # one numbered script per experiment, deterministic
  reports/         # structured output, one file per experiment
  tests/
notebooks/         # Colab runners only; logic lives in pcc/*.py
refs/              # cloned reference repos (gitignored) -- see reports/release_audit.md
```

## Phase order (do not skip)

0. **Phase 0** (`AGENTS.md` §5) — replicate the efficiency-gap decomposition on
   *real* logits (ImageNet, iNaturalist). If per-class offset does not beat
   global-temperature and energy-reweighting with non-overlapping CIs, stop.
1. **Phase 1 / GATE** (`AGENTS.md` §6) — reliability (A), predictability from
   geometry (B), adversarial ablation vs. log-prevalence / Fargion distance (C),
   and translation to set size (§6.4). `notebooks/01_descriptor_stability` is
   **mandatory before** this — a negative gate is uninterpretable without it (§3.3).
2. **Phase 2+** — only after the gate passes.

## Status

Bootstrapping. See `reports/` for the release audit and, once written, the
frozen fallback policy and per-experiment pass/fail criteria.
