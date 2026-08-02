# Held-out-class fallback policy (FROZEN)

**Written:** 2026-07-24 (before any experiment was run).
**Author:** coding agent, per `AGENTS.md` §7 ("TETAPKAN SEBELUM MELIHAT DATA").

This file fixes, in advance, what each Tier-2 baseline does on a **held-out
class with `n_y = 0` labeled calibration samples**, where most class-level
methods are *undefined*. This must be decided before seeing any results and
**must not be revised after results are visible**. If a revision is ever forced,
the reason is recorded here with a timestamp and BOTH versions are reported.

## Why this file exists (the double-edged sword, §7)

The held-out-class table is where the PCC claim lives or dies. It is *also* the
easiest place to build a strawman: if PCC "wins" only because every competitor
is undefined at `n_y = 0`, a reviewer will (correctly) reject the comparison.
So each baseline gets the **fairest defensible** behavior — which is the
behavior it would actually exhibit in deployment.

## Default rule

**Fall back to the global (marginal) split-conformal threshold `q̂_global`.**
Rationale: that is what these methods *actually do* in deployment when a class
has no calibration data — there is no per-class quantile to compute, so the
marginal quantile is used. This keeps marginal coverage valid and is the honest
comparator.

## Per-baseline fallback (Tier 2, `AGENTS.md` §7)

| Baseline | Behavior at `n_y = 0` (held-out class) | Frozen fallback |
|---|---|---|
| **Mondrian / classwise CP** | per-class quantile → +∞ threshold ⇒ set = full label space | **global `q̂_global`** |
| **Clustered CP** | class assigned to no cluster / null cluster | **global `q̂_global`** (equivalently: the "null cluster → marginal" rule the method itself specifies) |
| **Prevalence-adjusted softmax + Interp-Q** | needs class prevalence, which needs labels | **global `q̂_global`** with prevalence term dropped |
| **Fuzzy Classwise CP** | similarity-weighted over *seen* classes; weights to held-out class unavailable | similarity-weighted quantile using only **seen** classes' calibration data (this is the method's natural generalization; record it as such, not as global) |
| **Macro-coverage / label-weighted CP** | grouping given externally; held-out class not in any calibrated group | **global `q̂_global`** |
| **Class similarity score (Fargion)** | score uses distance to class means; threshold still needs a per-class or global quantile | similarity score retained, **global `q̂_global`** for the threshold |
| **RC3P** | rank-calibrated per-class; undefined without class labels | **global `q̂_global`** |
| **TACP / sTACP** | threshold-adjusted per class | **global `q̂_global`** |
| **CFCP** | per-embedding-cluster label frequency; held-out class may still fall in a populated cluster | **use the cluster it falls into if populated; else global `q̂_global`** (record which per class) |

Note: Fuzzy Classwise and CFCP get their *natural* generalization to held-out
classes rather than a bare global fallback, precisely because they *have* one —
using global for them would understate the strongest conceptual competitors and
invite the reverse strawman charge. Both are flagged in `AGENTS.md` §7 as the
most dangerous comparators; run them early.

## Two tables, always (§7)

1. **Seen classes** — PCC must at least tie Tier 2 here. Losing here means
   predicted correction is worse than estimated correction, which kills the claim.
2. **Held-out classes** — the claim lives or dies here; this fallback policy
   applies. Never average the two into one number.

## Revision log

_(empty — no revisions. Any entry here requires a timestamp, the reason, and
re-reporting of both the pre- and post-revision versions.)_
