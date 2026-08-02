# `pcc/baselines/` — MANDATORY comparators (`AGENTS.md` §7)

A baseline may appear in a table **only after it is reproduced** against the
authors' reported numbers on ≥1 setting (§7 "Reproduksi"). Prefer the official
author implementation. Record every deviation.

Held-out-class behavior for every Tier-2 method is fixed in
[`reports/fallback_policy.md`](../reports/fallback_policy.md) and must not change
after results are seen.

## Tier 1 — floor (score functions) — report, not the bar
- THR/LAC (Sadinle 2019) — `pcc/scores/base.py:thr_lac` (done)
- APS (Romano 2020), RAPS (Angelopoulos 2021), SAPS (Huang 2024) — stubs in
  `pcc/scores/base.py`, implement + reproduce before use.

## Tier 2 — the real bar (post-hoc class-level, frozen model) — head-to-head
| Method | Source | Code available |
|---|---|---|
| Mondrian / classwise CP | Vovk 2012 | `refs/ccc/utils/conformal_utils.py` |
| Clustered CP | Ding 2023 (2306.09335) | `refs/ccc/utils/{conformal,clustering}_utils.py` ✅ |
| PAS + Interp-Q | Ding–Fermanian–Salmon 2026 (2507.06867) | `refs/ltc/utils/conformal_utils.py`, `example.ipynb` ✅ |
| Fuzzy Classwise CP | same paper, App. A | `refs/ltc/.../fuzzy_classwise_CP` ✅ **run early (§7)** |
| Macro-coverage / label-weighted CP | Bhattacharyya–Ding–Barber 2026 (2606.28598) | released — fetch |
| Class similarity score | Fargion–Dabah–Tirer (2511.19359) | github.com/ariel361/CP_via_CS |
| RC3P | Shi et al. 2024 (2406.06818) | github.com/YuanjieSh/RC3P |
| TACP / sTACP | Liu–Huang–Ong 2026 (2508.11345) | — |
| CFCP | Lavi–Shapira–Rappoport (2605.24872) | — |

> **Fuzzy Classwise and Class Similarity are the most dangerous comparators**
> (§7): both use hand-specified class similarity. If PCC's improvement over them
> is within noise, that is the prior failure pattern. Run them **early**.

## Tier 3 — position in related work, NOT in the main table
CaCT (2601.09522), ConfTr + DPSM/CUT/InfoCTr, C-Adapter (2410.09408). Need
retraining or a different setting → not head-to-head. Silence on these reads as
not knowing them; position in `reports/` and related work.

## Two tables, always (§7) — never averaged into one:
1. **Seen classes** — PCC must at least tie Tier 2.
2. **Held-out classes** — where the claim lives; fallback policy applies.
