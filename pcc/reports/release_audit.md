# §2.3 Release audit — what the Ding repos actually publish

**Date:** 2026-07-24. **Author:** coding agent.
**Repos audited** (shallow-cloned to `refs/`, gitignored):

- `class-conditional-conformal` (**CCC** — Clustered CP, Ding et al. NeurIPS 2023, arXiv 2306.09335)
- `long-tail-conformal` (**LTC** — Ding, Fermanian, Salmon, ICLR 2026, arXiv 2507.06867)

This audit answers the §2.3 questions: *what is released* (checkpoints / scores /
code), and whether the **same checkpoint** that produced the released scores is
available so penultimate embeddings can be extracted from it. Embeddings are the
main descriptor input and are **not** in any released score dump (§2.3.3).

---

## Bottom line

| Dataset (source repo) | Softmax scores + labels | Backbone checkpoint | Embeddings recoverable from the *same* model? | Blocker? |
|---|---|---|---|---|
| **iNaturalist-2018 (LTC)** | ✅ released (gdown) | ✅ **released** ResNet-50 (`models.zip`) | ✅ yes — extract from the released ResNet-50 | **No** |
| **Pl@ntNet-300K (LTC)** | ✅ released (gdown) | ✅ **released** ResNet-50 (`models.zip`) | ✅ yes — extract from the released ResNet-50 | **No** |
| **ImageNet-1k (CCC)** | ✅ released (gdown) | ⚠️ backbone = **public SimCLRv2 `r152_3x_sk1`** (via `download.py`); trained linear head not released | ✅ yes — SimCLR *representation* = the penultimate embedding, from the public encoder | **No**, but see deviation D1 |
| **CIFAR-100 (CCC)** | ✅ released (gdown, 0.01 GB) | ❌ training code only (`cifar-100.ipynb`) | ⚠️ only by retraining → different model than released scores | **No** (non-primary; §2.1 fast-iteration only) |
| **Places365 (CCC)** | ✅ released (gdown, 0.54 GB) | ❌ training code only (SLURM) | ⚠️ retraining only | Optional dataset; defer |
| **iNaturalist-2021, 633-class (CCC)** | ✅ released (gdown, 6.72 GB) | ❌ training code only | ⚠️ retraining only | See comparability trap C1 |

**Headline:** the two most important primary datasets — **iNaturalist-2018 and
Pl@ntNet-300K — have both released scores AND the exact ResNet-50 checkpoints.**
This is the ideal §2.3.3 hybrid path (released scores for baselines, own
embedding extraction from the *same* model) and it works with **no blocker**.
Start here.

---

## What each repo releases

### LTC (long-tail-conformal)
- **Scores + labels:** `scripts/download_softmax_and_labels.sh` (gdown) → `cal`
  and `test` softmax scores + labels + `train` labels, for `plantnet`,
  `inaturalist`, their `-trunc` variants, and **focal-loss** variants.
- **Checkpoints: YES.** `train_models/README.md` → `gdown 1tS-M-4IYyCGMeIxxyrgx2-XCZgdvw18S; unzip models.zip`
  gives weights for all six ResNet-50 models. These are the exact models that
  produced the released scores (20-epoch ResNet-50, best-val-acc epoch).
- **Baselines, ready-made:** `utils/conformal_utils.py` + `example.ipynb`
  implement **Standard CP with PAS**, **Fuzzy Classwise CP** (raw +
  reconformalized), and **Interp-Q** — three of the Tier-2 comparators in
  `AGENTS.md` §7, standalone. `utils/clustering_utils.py` is shared with CCC.
- **iNaturalist level:** `train.py` documents the 2018 taxonomy sizes
  `# 8142, 4412, 1120, 273, 57, 25, 6` (species → kingdom). LTC trains/evaluates
  at **8142 species** — matches the `AGENTS.md` §2.1 assumption.

### CCC (class-conditional-conformal)
- **Scores + labels:** `download_data.sh` (gdown) → arrays for `imagenet`
  `(115301, 1000)`, `cifar-100` `(30000, 100)`, `places365` `(183996, 365)`,
  `inaturalist` `(1324900, 633)`.
- **Checkpoints: NO.** Only `generate_scores/` training code.
  - **ImageNet** scores = **SimCLRv2 `r152_3x_sk1`** features + a trained linear
    head. The encoder is public (`download.py r152_3x_sk1`, weights converted via
    `convert.py`, code from `Separius/SimCLRv2-Pytorch`), and
    `get_simclr_representations.py` extracts the representation. So ImageNet
    **embeddings are reproducible from the same public encoder**; only the linear
    head that produced the exact scores is unreleased (and it is not needed for
    embeddings).
  - **CIFAR-100 / Places365 / iNaturalist-2021**: training code only.
- **Baselines, ready-made:** `utils/conformal_utils.py` + `utils/clustering_utils.py`
  implement **Clustered CP** and classwise/standard CP; `example.ipynb` shows usage.

---

## Deviations from `AGENTS.md` §2.2 to record (each is a reviewer question)

- **D1 — ImageNet backbone.** `AGENTS.md` §2.2 lists ResNet-50 + ViT-B/16. But
  to be **bit-level comparable to Clustered CP on ImageNet**, the backbone is
  **SimCLRv2 ResNet-152 (3×, sk1) + linear probe**, not a supervised ResNet-50.
  Options: (a) use released CCC scores + SimCLR embeddings for the Clustered-CP
  head-to-head (bit-level comparable, but backbone ≠ §2.2); (b) also run our own
  supervised ResNet-50 / ViT-B/16 ImageNet extraction for the §2.2 backbone
  sweep, comparing against baselines we recompute on that backbone (not against
  the released CCC numbers). Recommend doing both and labeling tables by backbone.
- **D2 — Pl@ntNet checkpoint provenance.** `AGENTS.md` §2.2 says "official
  Pl@ntNet-300K checkpoint." LTC provides its **own** ResNet-50 weights (that
  produced its released scores). For head-to-head with LTC (PAS/Fuzzy/Interp-Q),
  use LTC's weights — that is the comparable model. The Zenodo "official"
  checkpoint is a *different* model; only use it if comparing to something that used it.

## Comparability traps to flag (each can invalidate a merged table)

- **C1 — two different iNaturalists.** CCC ships **iNat-2021 at 633 classes**;
  LTC ships **iNat-2018 at 8142 species**. They are different datasets *and*
  different taxonomic levels. Never merge their numbers. Per `AGENTS.md` §2.1,
  pick the level of the specific paper being compared and record it per table.
  For PCC's own primary runs, prefer **LTC's iNat-2018 / 8142** (checkpoint +
  scores + train labels all released, finest level → most training units for g_θ).
- **C2 — do not recompute released scores.** `AGENTS.md` §2.3.2: use released
  scores directly for baselines; recomputing risks preprocessing drift that
  produces indefensible small deltas. Our own extraction is for **embeddings
  only**, from the same checkpoint.

---

## VERIFIED on Colab — Pl@ntNet Pre-check A PASSED (2026-08-04)

Zero-image artefact check, `notebooks/00_verify_checkpoint.ipynb` cells 1–5:

| item | value | verdict |
|---|---|---|
| released split used | `cal` | the release ships **cal + test only** — there is no `val` split |
| released rows `N` | **21,783** | **matches our reconstruction exactly** (see below) |
| released classes `C` | **1081** | matches `NUM_CLASSES['plantnet']` |
| checkpoint `fc.out_features` | **1081** | head matches released classes → correct checkpoint/arch |
| released self-accuracy | **0.7945** | LTC's Pl@ntNet model top-1 on `cal`; reference number |
| checkpoint sha256 | `4b82e4aa1a97d281…` | recorded |
| `labels[min,max]` | **[0, 1075]** | see the zero-sample-class note below |

**Split reconstruction independently confirmed.** LTC builds `cal` as a 70% subset
of the `val` directory via `np.random.seed(0); np.random.shuffle(indices)`, taking
the first 30% as proper-val. For a 31,118-image val directory that gives
`floor(0.3 × 31118) = 9,335` proper-val and `31118 − 9335 = 21,783` cal — exactly
the released row count. So `ltc_cal_val_indices` reproduces their membership, which
is what makes the G4 label-multiset check meaningful rather than a guaranteed
mismatch. (Row ORDER is still unrecoverable — loaders use `shuffle=True` — so the
gate remains permutation-invariant.)

**G2 is now ENABLED for Pl@ntNet.** LTC's `PlantNet` subclasses
`torchvision.datasets.ImageFolder`, so its class-index convention IS ImageFolder's
sorted-folder-name order. The earlier G2 uncertainty is resolved; all of G1–G4 apply.

**Zero-sample classes exist already.** `labels.max() = 1075` with `C = 1081` means
at least 5 classes have **no calibration samples at all**, so δ_y is undefined for
them and `fallback_policy.md` applies immediately on real data. Cell 5b
(Pre-check A2) quantifies this from the released labels alone — no images needed —
and its output is the input to the **`n_cal` human decision** required by
Amendment 2.

## Open items requiring on-Colab verification (cannot be done here)

The multi-GB gdown artifacts were **not** downloaded in this environment. Before
Phase 0, on Colab:

1. Download `models.zip` (LTC) and confirm the ResNet-50 architecture + that a
   forward pass reproduces the released softmax scores **bit-for-bit** on a few
   val samples (this is the §2.3 "same checkpoint" proof). Record checksums.
2. Confirm released score array shapes/label ranges (esp. iNat class counts:
   expect 8142 for LTC-2018, 633 for CCC-2021).
3. For ImageNet, confirm the SimCLR `r152_3x_sk1` download + conversion path
   still resolves, and that extracted representations reproduce the released
   ImageNet scores through a re-fit linear probe (sanity, not for the baseline).

## DECISIONS — approved 2026-07-25

**Decision 1 — backbone provenance (APPROVED).** LTC's released ResNet-50 weights
(`models.zip`) are the **canonical** backbone for iNat-2018 and Pl@ntNet-300K.
Rationale (recorded per instruction): the correct checkpoint is *the one that
produced the comparison scores*, not the most "official" one. The Zenodo
Pl@ntNet checkpoint is a **different model** and is used only if a specific
comparator used it — **none currently do**, so it is excluded. Provenance to
carry into every table: `backbone = resnet50 (LTC best-{dataset}-model), source
= tiffanyding/long-tail-conformal models.zip (gdown 1tS-M-4IYyCGMeIxxyrgx2-XCZgdvw18S)`.

**Decision 2 — ImageNet backbones (APPROVED, "do both", not either/or).** Run
**both** paths; they answer different questions and neither replaces the other:
1. **SimCLRv2-R152 + released CCC scores** — the *only* bit-level head-to-head
   with Clustered CP on ImageNet. Required for the Tier-2 comparison.
2. **Supervised ResNet-50 + ViT-B/16** (own extraction) — tests whether δ_y
   predictability from geometry **survives across backbones** or is an artifact
   of one representation.

   Reframing (recorded): self-supervised (SimCLR) vs. supervised (ResNet-50/ViT)
   is a **feature**, not a deviation to apologize for. If predictability holds
   across both representation *types*, the core claim is much stronger. This is
   **free backbone variance for Phase 3**, not an apology. (D1 above stands as a
   provenance note, not a caveat.)

**HARD RULE (both decisions).** Never compare supervised-backbone runs against
the released CCC numbers. **Baselines for backbone B are recomputed on backbone
B.** Every table is labeled by backbone. Never merge across backbones.

## Approved execution order (REVISED 2026-07-25 — cheap-first, iNat deferred)

Rationale (recorded): the Phase-1 gate answers *does the signal exist*, and that
is as visible at Pl@ntNet's 1,081 classes as at iNat's 8,142. iNat only adds
**training units for g_θ** — statistical power, not signal existence. So pay the
most expensive cost (iNat, 120 GB) **only after** cheap results prove it worth it.

1. **CIFAR-100 — pipeline debug first (0.01 GB).** Exercise the full Phase-0 +
   Phase-1 analysis code end-to-end on 100 classes before spending on anything
   bigger. **No released checkpoint exists** (CCC ships training code only; LTC
   doesn't cover CIFAR-100), so we **self-train a ResNet-50** (§2.2: fine-tune
   from IMAGENET1K_V2, the Ding/CFCP procedure) and take **scores + embeddings
   from our own model** — self-consistent. The checkpoint→score *reproduction*
   gate is **N/A here** (we own the whole stack); a lighter self-consistency
   check replaces it. Baselines recomputed on our model, never vs CCC's numbers.
   - **Model decision (approved 2026-07-25):** fine-tune / transfer-learn from
     IMAGENET1K_V2 (the faithful §2.2 procedure — same cost as a throwaway model,
     more useful and optionally CFCP-comparable).
   - **THIN-REGIME CAVEAT (binding):** 100 classes ⇒ a 50/50 class-level split is
     only 50 fit / 50 held-out classes for g_θ — too thin for the extrapolation
     claim (cf. §2.1's CIFAR-10 exclusion). CIFAR-100 is therefore **strictly a
     pipeline debug and never the gate verdict.** The Phase-1 gate decision
     (A/B/C + §6.4) is made on **Pl@ntNet**, not here. A CIFAR-100 pass does not
     unlock iNat; a CIFAR-100 fail does not kill the project — it debugs the code.
2. **Pl@ntNet-300K via LTC — the scientific result at this stage.** Released
   checkpoint + scores (feasible ~31 GB download). Run the checkpoint gate, then
   **Phase 0 and Phase 1 in full here.**
3. **iNaturalist-2018 (8142) — ONLY IF all of:** Pl@ntNet checkpoint gate passes,
   Phase 1 passes on Pl@ntNet (criteria A/B/C), **and** §6.4 (translation to set
   size) is positive. **If the gate dies on Pl@ntNet, the project stops without
   ever touching iNat.** Do not request the 120 GB machine decision up front.
4. **ImageNet** — SimCLR path (Tier-2 head-to-head) **and** supervised R50/ViT
   path (cross-backbone); tables kept separate per the hard rule. Sequenced with
   Phase 2+/Phase 3, after the Pl@ntNet gate outcome.

## PROMOTED TO A HARD GATE — checkpoint → score reproduction (Phase 0 prerequisite)

Per the 2026-07-25 instruction, the bit-for-bit checkpoint→score check is **no
longer an open item — it is a hard Phase-0 prerequisite.** If a forward pass on
the released checkpoint does **not** reproduce the released softmax scores on val
samples (preprocessing drift, normalization mismatch, wrong checkpoint version),
**STOP and report — do not proceed to embedding extraction.** This failure is
*silent*: extraction still runs, descriptors still compute, Phase 1 still emits
numbers — all meaningless. The reproduction check is the only alarm. Record
SHA256 checksums of the `.pth` and the released `.npy` when it passes.

Criteria and method: `reports/phase0_checkpoint_gate.md`. Runner:
`notebooks/00_verify_checkpoint.ipynb`. Extraction (`01_setup_extract`) refuses
to run unless the gate report on Drive says PASS.

### IMPORTANT operational finding — "bit-for-bit" must be permutation-invariant

`get_dataloaders` uses **`shuffle=True` for val/cal/test**, and
`get_softmax_and_labels` writes rows in loader order. So the released
`*_softmax.npy` / `*_labels.npy` are in an **unrecoverable shuffled order** (the
loader shuffle is not seeded reproducibly). Consequences:
- Exact per-row alignment of a released row to a specific source image is **not
  reconstructable**. Literal "bit-for-bit per row" is therefore not the operable
  test.
- The gate is instead **permutation-invariant** and tuned so that the real
  failure modes (which cause *gross* differences — accuracy collapse, argmax
  disagreement, diffs ≫ float noise) trip it, while GPU/cuDNN/version float noise
  does not. See the gate doc for the exact tolerances.
- iNaturalist val order **is** deterministic from `val2018.json["images"]`, so we
  regenerate our scores in that deterministic order (`shuffle=False`) and compare
  to the released set by permutation-invariant statistics + nearest-neighbor
  row matching.

## Data-acquisition logistics (needs a plan for iNaturalist val)

- **Pl@ntNet-300K:** ~31 GB on Zenodo (record 5645731), folder-structured; val/
  test are small subsets — feasible to download and use directly on Colab.
- **iNaturalist-2018 images — DEFERRED until the Pl@ntNet gate passes; do NOT
  decide the 120 GB machine now.** For the record, when the time comes:
  - The 24,426 val images live **only inside the monolithic
    `train_val2018.tar.gz` (~120 GB)** with train — there is no val-only tarball.
  - **Do not stream it on Colab.** gzip must be decompressed **sequentially**, so
    reaching the val entries means streaming the entire 120 GB — a multi-hour job
    that a Colab session will kill. This is not a tuning problem; it is the
    archive format.
  - **Do it once on a disk-backed machine** (ideally an `us-east-1` instance near
    the competition S3 bucket), then **mirror the ~2 GB subset to Drive** so every
    later run skips the 120 GB entirely.
  - **One pass, not two.** In that single tar traversal, extract
    **(all val) ∪ (N-per-class from train)** — val for the gate, the per-class
    train quota for descriptors (§6.3). Never traverse the tar twice.

## VERIFIED on Colab — Pl@ntNet Pre-check A PASSED (2026-08-04)

Zero-image artefact check, `notebooks/00_verify_checkpoint.ipynb` cells 1–5:

| item | value | verdict |
|---|---|---|
| released split used | `cal` | the release ships **cal + test only** — there is no `val` split |
| released rows `N` | **21,783** | **matches our reconstruction exactly** (see below) |
| released classes `C` | **1081** | matches `NUM_CLASSES['plantnet']` |
| checkpoint `fc.out_features` | **1081** | head matches released classes → correct checkpoint/arch |
| released self-accuracy | **0.7945** | LTC's Pl@ntNet model top-1 on `cal`; reference number |
| checkpoint sha256 | `4b82e4aa1a97d281…` | recorded |
| `labels[min,max]` | **[0, 1075]** | see the zero-sample-class note below |

**Split reconstruction independently confirmed.** LTC builds `cal` as a 70% subset
of the `val` directory via `np.random.seed(0); np.random.shuffle(indices)`, taking
the first 30% as proper-val. For a 31,118-image val directory that gives
`floor(0.3 × 31118) = 9,335` proper-val and `31118 − 9335 = 21,783` cal — exactly
the released row count. So `ltc_cal_val_indices` reproduces their membership, which
is what makes the G4 label-multiset check meaningful rather than a guaranteed
mismatch. (Row ORDER is still unrecoverable — loaders use `shuffle=True` — so the
gate remains permutation-invariant.)

**G2 is now ENABLED for Pl@ntNet.** LTC's `PlantNet` subclasses
`torchvision.datasets.ImageFolder`, so its class-index convention IS ImageFolder's
sorted-folder-name order. The earlier G2 uncertainty is resolved; all of G1–G4 apply.

**Zero-sample classes exist already.** `labels.max() = 1075` with `C = 1081` means
at least 5 classes have **no calibration samples at all**, so δ_y is undefined for
them and `fallback_policy.md` applies immediately on real data. Cell 5b
(Pre-check A2) quantifies this from the released labels alone — no images needed —
and its output is the input to the **`n_cal` human decision** required by
Amendment 2.

## Open items requiring on-Colab verification (cannot be done here)

The multi-GB gdown/Zenodo artifacts were **not** downloaded in this environment.
On Colab, before Phase 0 extraction:

1. Run `00_verify_checkpoint.ipynb` on **Pl@ntNet** → PASS (record checksums).
2. Confirm released array shapes / label ranges (Pl@ntNet 1081; iNat 8142).
3. For ImageNet, confirm the SimCLR `r152_3x_sk1` download + conversion path
   resolves, and that extracted representations re-fit a linear probe reproducing
   the released ImageNet scores (sanity, not a baseline).
