# -*- coding: utf-8 -*-
"""Generate the paper's figures as PDFs, straight from the run reports.

Numbers in a figure should not be retyped any more than numbers in a table, so the two data
charts read the same JSON the tables do. The overview figure is a schematic and says so in
its caption.

Output goes to the paper directory as fig_*.pdf, vector, no external fonts.
"""
from __future__ import print_function
import glob
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = r"c:/jagr/foundation-cp"
D = sorted(glob.glob(os.path.join(REPO, "nb12_*latest*")))[-1]
OUT = os.path.join(REPO, "wacv-2027-author-kit-template",
                   "wacv-2027-author-kit-template", "figures")
if not os.path.isdir(OUT):
    os.makedirs(OUT)

# one WACV column is 3.25in; keep every figure inside it
COL = 3.25
INK = "#1a1a1a"
BLUE = "#3670a8"
ORANGE = "#c8672a"
GREY = "#9a9a9a"

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(tag):
    out = []
    for p in sorted(glob.glob(os.path.join(D, tag + "_s*.json"))):
        out.append(json.load(io.open(p, encoding="utf-8")))
    return out


def stat(ds):
    """(mean, lo, hi, ceiling, median eval per class) for one configuration."""
    t2 = [d["results"]["table_2_heldout"] for d in ds]
    st = t2[0]["primary_stat"]
    v = np.array([t["delta"][st] for t in t2], float)
    c = np.array([t["delta_oracle"].get(st, np.nan) for t in t2], float)
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return (v.mean(), v.mean() - h, v.mean() + h, np.nanmean(c),
            t2[0]["measurability"].get("median_eval_per_class"))


# ============================================================ 1. evaluation depth
TAGS = [("nb12_F_evaldepth_eval3", 3), ("nb12_F_evaldepth_eval10", 10),
        ("nb12_F_evaldepth_eval35", 35), ("nb12_F_evaldepth_eval75", 75),
        ("nb12_F_evaldepth_evalNone", 76)]
xs, mu, lo, hi, ceil = [], [], [], [], []
for tag, x in TAGS:
    ds = load(tag)
    if not ds:
        continue
    m, l, h, c, _ = stat(ds)
    xs.append(x); mu.append(m); lo.append(l); hi.append(h); ceil.append(c)
xs, mu, lo, hi, ceil = map(np.array, (xs, mu, lo, hi, ceil))

fig, ax = plt.subplots(figsize=(COL, 1.95))
ax.axhspan(-0.05, 0.30, xmin=0, xmax=(np.log10(35) - np.log10(2.4)) /
           (np.log10(95) - np.log10(2.4)), color="#f0f0f0", zorder=0)
ax.text(4.0, 0.20, "not measurable", fontsize=6.2, color=GREY, style="italic")
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (3, 3)), zorder=1)
ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.16, lw=0, zorder=2)
ax.plot(xs, ceil, color=ORANGE, marker="s", ms=3.0, ls="--", zorder=3,
        label="oracle ceiling")
ax.plot(xs, mu, color=BLUE, marker="o", ms=3.2, zorder=4, label="PCC")
for x, y, lab in ((35, 0.0171, "backbones"), (3, -0.0060, "Pl@ntNet"),
                  (2.6, -0.0060, "iNat")):
    if x < 2.8:
        continue
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(2, -9),
                fontsize=5.8, color=INK)
ax.set_xscale("log")
ax.set_xticks([3, 10, 35, 75])
ax.set_xticklabels(["3", "10", "35", "75"])
ax.set_xlim(2.4, 95)
ax.set_ylim(-0.05, 0.30)
ax.set_xlabel("evaluation examples per class")
ax.set_ylabel(r"$\Delta$ worst-class")
ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderpad=0.2)
fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "fig_evaldepth.pdf"))
plt.close(fig)
print("fig_evaldepth.pdf  depth %s" % list(xs))

# ============================================================== 2. held-out sweep
HTAGS = [("nb12_H_heldout_ho0p1", 10), ("nb12_H_heldout_ho0p3", 30),
         ("nb12_H_heldout_ho0p5", 50), ("nb12_H_heldout_ho0p7", 70),
         ("nb12_H_heldout_ho0p9", 90)]
hx, hm, hl, hh, hc, nlab = [], [], [], [], [], []
for tag, x in HTAGS:
    ds = load(tag)
    if not ds:
        continue
    m, l, h, c, _ = stat(ds)
    hx.append(x); hm.append(m); hl.append(l); hh.append(h); hc.append(c)
    nlab.append(ds[0]["results"]["n_seen"])
hx, hm, hl, hh = map(np.array, (hx, hm, hl, hh))

fig, ax = plt.subplots(figsize=(COL, 1.85))
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (3, 3)))
ax.fill_between(hx, hl, hh, color=BLUE, alpha=0.16, lw=0)
ax.plot(hx, hm, color=BLUE, marker="o", ms=3.2)
ax.axvline(99.0, color=ORANGE, lw=0.8, ls=":")
ax.text(97.5, 0.115, r"floor: $p{+}2$ classes", fontsize=5.8, color=ORANGE,
        rotation=90, ha="right", va="top")
for x, y, n in zip(hx, hm, nlab):
    ax.annotate("%d" % n, (x, y), textcoords="offset points", xytext=(0, 5),
                fontsize=5.6, color=INK, ha="center")
ax.set_xlim(5, 100)
ax.set_xticks([10, 30, 50, 70, 90])
ax.set_xlabel("% of classes with no calibration data")
ax.set_ylabel(r"$\Delta$ worst-class")
fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "fig_heldout.pdf"))
plt.close(fig)
print("fig_heldout.pdf    labelled classes %s" % nlab)

# =============================================================== 3. overview
fig = plt.figure(figsize=(COL, 2.10))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.78], hspace=0.10)

# --- (a) what each method gives a class with no calibration data
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0, 9.6, "(a) what each method gives a class with $n_y = 0$",
        fontsize=7.0, color=INK)
rows = [("classwise CP", r"$q_y=\infty$: every label", ORANGE),
        ("clustered, fuzzy", "one shared threshold", ORANGE),
        ("PCC", r"$\hat q + \lambda g_\theta(\varphi(y)) + c$", BLUE)]
for i, (name, what, col) in enumerate(rows):
    y = 7.0 - 2.6 * i
    ax.text(0, y, name, fontsize=6.6, color=INK, va="center")
    ax.add_patch(FancyArrowPatch((3.1, y), (4.2, y), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=GREY))
    ax.text(4.5, y, what, fontsize=6.6, color=col, va="center")

# --- (b) the method in three steps
ax = fig.add_subplot(gs[1])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0, 9.4, "(b) the correction", fontsize=7.2, color=INK)
steps = [(0.0, "observe " + r"$\delta_y$", "labelled classes"),
         (3.5, "fit " + r"$g_\theta(\varphi(y))$", "head weights only"),
         (7.0, "shrink by " + r"$\lambda$", "refit offset $c$")]
for x, top, bot in steps:
    ax.add_patch(FancyBboxPatch((x, 1.4), 2.7, 5.6,
                                boxstyle="round,pad=0.10,rounding_size=0.30",
                                fc="#f4f7fa", ec=BLUE, lw=0.8))
    ax.text(x + 1.35, 5.2, top, fontsize=6.3, ha="center", va="center", color=INK)
    ax.text(x + 1.35, 3.0, bot, fontsize=5.5, ha="center", va="center", color=GREY)
for x in (2.75, 6.25):
    ax.add_patch(FancyArrowPatch((x, 4.2), (x + 0.7, 4.2), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.9, color=BLUE))
fig.savefig(os.path.join(OUT, "fig_overview.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("fig_overview.pdf")

print()
for f in sorted(os.listdir(OUT)):
    print("  %-22s %6.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024.0))
