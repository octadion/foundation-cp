# Descriptor stability — findings (CIFAR-100, first real measurement)

**Run:** 2026-07-25, `notebooks/02_descriptor_stability.ipynb`, self-trained
ResNet-50 (82.6% test acc), 20,000 train embeddings (200/class, 2048-dim).
Machine-readable output: `reports/02_descriptor_stability_cifar100.json`.

## Measured curve (cross-draw correlation of φ(y), disjoint halves)

| quota q | mean_corr | SE |
|---|---|---|
| 10 | 0.669 | 0.009 |
| 25 | 0.780 | 0.002 |
| 50 | 0.852 | 0.007 |
| 100 | **0.903** | 0.003 |

`recommended_quota = 100` at threshold 0.90.

**Reality check:** real embeddings are far noisier than the synthetic used to
develop the module (q=10: 0.669 real vs 0.933 synthetic). This is exactly why
§3.3 mandates the measurement rather than assuming a quota.

## Per-feature stability

| feature | q=10 | q=25 | q=50 | q=100 | verdict |
|---|---|---|---|---|---|
| cos_knn_5 | 0.866 | 0.928 | 0.967 | 0.983 | **best behaved** |
| cos_knn_1 | 0.838 | 0.916 | 0.958 | 0.977 | strong |
| cos_knn_10 | 0.852 | 0.917 | 0.960 | 0.981 | strong |
| cos_knn_50 | 0.844 | 0.912 | 0.958 | 0.979 | strong |
| cov_trace | 0.768 | 0.894 | 0.943 | 0.975 | good |
| logit_margin | 0.765 | 0.893 | 0.942 | 0.973 | good |
| mean_norm | 0.708 | 0.859 | 0.918 | 0.959 | ok |
| softmax_entropy | 0.591 | 0.789 | 0.871 | 0.938 | ok at q≥100 |
| cov_eig_0 | 0.528 | 0.700 | 0.839 | 0.923 | marginal |
| cov_eig_2 | 0.628 | 0.711 | 0.790 | 0.848 | **weak** |
| cov_eig_1 | 0.568 | 0.669 | 0.730 | 0.831 | **weak** |
| true_rank | 0.070 | 0.166 | 0.348 | 0.468 | **NOISE — removed** |
| n_eff / log_prevalence | — | — | — | — | constant by construction, excluded |

Cosine-to-neighbour features are the most reliable geometry signal; higher-order
covariance eigenvalues are the least (estimating the 2nd/3rd eigenvalue of a
2048-dim covariance from ~100 samples is intrinsically hard).

## Action taken — `true_rank` replaced (recorded, measured)

Raw mean rank was essentially noise (<0.5 even at the largest testable quota).
Cause: at ~83% accuracy most samples have rank 0, so the mean is dominated by a
few long-tail misranked samples. Simulation with realistic per-class difficulty:

| q | mean_rank | mean_log1p_rank | frac_top1 |
|---|---|---|---|
| 10 | 0.367 | 0.505 | 0.556 |
| 25 | 0.584 | 0.710 | 0.751 |
| 50 | 0.751 | 0.842 | 0.868 |
| 100 | 0.843 | 0.907 | 0.925 |

`logit_stats` now emits **`frac_top1`** and **`mean_log1p_rank`** instead. Both
remain "rank distribution" features per §6.3. Rationale: feeding a near-noise
descriptor into the gate-B/C ridge depresses R² and makes a negative Phase-1
result ambiguous — the exact failure §3.3 exists to prevent.

## Open issue 1 — the curve has NOT plateaued

0.90 is crossed only at q=100, which is also the **largest q testable** here
(disjoint halves need 2q, and only 200/class were extracted). The curve is still
rising steeply (0.852 → 0.903), so the true plateau is unknown. CIFAR-100 has 500
train images/class, so extracting **400/class** would let us test q up to 200 and
locate the plateau. Cost ~2 min. Until then, `recommended_quota = 100` should be
read as "the smallest tested q that crosses threshold", **not** "the point where
the descriptor stops improving".

## Open issue 2 — BLOCKING RISK for Pl@ntNet (decide before Phase 1 there)

If descriptors need ~100 images/class to reach 0.90 stability, then on a
**long-tailed** dataset the tail classes cannot reach it. Pl@ntNet-300K has many
species with only a handful of images. Consequence:

> **Descriptor quality would correlate with class prevalence.** Head classes get
> stable descriptors, tail classes get noisy ones.

This is precisely the trap §3.3 warns about ("Jangan diam-diam memakai deskriptor
berkualitas berbeda antara kelas head dan tail; itu akan menghasilkan korelasi
palsu antara prevalence dan kualitas prediksi"), and it attacks the most
important criterion: **gate C**. The log-prevalence ablation asks whether geometry
beats prevalence alone. If geometry descriptors are accurate exactly where
prevalence is high, geometry could look predictive *because of* prevalence-linked
noise, not despite it — a spurious gate-C pass.

CIFAR-100 cannot reveal this (every class has 500 images, perfectly balanced), so
it must be handled explicitly on Pl@ntNet. Options to decide **before** running
Phase 1 there:

1. **Report descriptor stability per prevalence quartile** and treat it as a
   covariate — mandatory regardless of which option is chosen.
2. **Cap the quota at what tail classes can supply** (uniform descriptor quality,
   lower stability everywhere) — removes the confound at the cost of noise.
3. **Restrict Phase 1 to classes meeting the quota**, reporting explicitly which
   classes were excluded and what fraction of the tail is lost.
4. **Model the noise**: include per-class descriptor SE as a feature/weight so the
   ridge can discount unstable rows.

Recommendation: **(1) always, plus (2) as the primary analysis and (3) as a
sensitivity check.** Do not proceed to Pl@ntNet gate C without this settled — a
gate C "pass" produced under prevalence-linked descriptor noise would be exactly
the kind of result this repo exists to avoid.
