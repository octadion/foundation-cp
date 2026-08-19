# -*- coding: utf-8 -*-
"""Regenerate the paper's charts in the idiom of Ding et al. (ICLR 2026).

What their figures do, read off the PDF rather than remembered: multi-panel grids that share
a y-axis and span the full text width, log axes where the data spans decades, several series
per panel, a small frameless legend, and no shaded regions, callout boxes or in-plot prose.
Everything explanatory lives in the caption.

What the previous version did wrong: single panels, a grey "not measurable" band with an
italic label inside the axes, coloured annotations pointing at individual points, and a
flow-chart of rounded boxes. Ding et al. explain their procedure with a pseudocode box
instead, which is why fig_overview is deleted here and replaced by Algorithm 1 in the text.

The two panels are plotted from the same JSON the tables read, so they cannot drift.
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

REPO = r"c:/jagr/foundation-cp"
D = sorted(glob.glob(os.path.join(REPO, "nb12_*latest*")))[-1]
OUT = os.path.join(REPO, "wacv-2027-author-kit-template",
                   "wacv-2027-author-kit-template", "figures")
if not os.path.isdir(OUT):
    os.makedirs(OUT)

TEXTWIDTH = 6.875          # WACV, both columns
INK = "#000000"
BLUE = "#2f6fad"
RED = "#c03a2b"
GREY = "#8c8c8c"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.1, "lines.markersize": 3.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(tag):
    return [json.load(io.open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(D, tag + "_s*.json")))]


def series(tags):
    xs, mu, lo, hi, ceil = [], [], [], [], []
    for tag, x in tags:
        ds = load(tag)
        if not ds:
            continue
        t2 = [d["results"]["table_2_heldout"] for d in ds]
        st = t2[0]["primary_stat"]
        v = np.array([t["delta"][st] for t in t2], float)
        c = np.array([t["delta_oracle"].get(st, np.nan) for t in t2], float)
        h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        xs.append(x); mu.append(v.mean()); lo.append(v.mean() - h)
        hi.append(v.mean() + h); ceil.append(np.nanmean(c))
    return map(np.array, (xs, mu, lo, hi, ceil))


fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.15), sharey=True)

# ------------------------------------------------------------ (a) evaluation depth
xs, mu, lo, hi, ceil = series([
    ("nb12_F_evaldepth_eval3", 3), ("nb12_F_evaldepth_eval10", 10),
    ("nb12_F_evaldepth_eval35", 35), ("nb12_F_evaldepth_eval75", 75),
    ("nb12_F_evaldepth_evalNone", 76)])
ax = axes[0]
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=BLUE, marker="o",
            capsize=1.8, elinewidth=0.7, zorder=3, label="PCC")
ax.plot(xs, ceil, color=RED, marker="s", ls="--", zorder=2, label="oracle ceiling")
ax.set_xscale("log")
ax.set_xticks([3, 10, 35, 75])
ax.set_xticklabels(["3", "10", "35", "75"])
ax.set_xlim(2.5, 92)
ax.set_ylim(-0.06, 0.20)
ax.set_xlabel("evaluation examples per class")
ax.set_ylabel(r"$\Delta$ worst-class coverage")
h, l = ax.get_legend_handles_labels()
o = [l.index('PCC'), l.index('oracle ceiling')]
ax.legend([h[i] for i in o], [l[i] for i in o], frameon=False, loc='upper left',
          handlelength=1.8, borderaxespad=0.2)
ax.set_title('(a) evaluation depth', fontsize=8, pad=4)

# --------------------------------------------------------------- (b) held-out sweep
hx, hm, hlo, hhi, hceil = series([
    ("nb12_H_heldout_ho0p1", 10), ("nb12_H_heldout_ho0p3", 30),
    ("nb12_H_heldout_ho0p5", 50), ("nb12_H_heldout_ho0p7", 70),
    ("nb12_H_heldout_ho0p9", 90)])
ax = axes[1]
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
ax.errorbar(hx, hm, yerr=[hm - hlo, hhi - hm], color=BLUE, marker="o",
            capsize=1.8, elinewidth=0.7, zorder=3, label="PCC")
ax.plot(hx, hceil, color=RED, marker="s", ls="--", zorder=2, label="oracle ceiling")
ax.set_xticks([10, 30, 50, 70, 90])
ax.set_xlim(4, 96)
ax.set_xlabel(r"classes with no calibration data (\%)"
              if plt.rcParams["text.usetex"] else
              "classes with no calibration data (%)")
h, l = ax.get_legend_handles_labels()
o = [l.index('PCC'), l.index('oracle ceiling')]
ax.legend([h[i] for i in o], [l[i] for i in o], frameon=False, loc='upper left',
          handlelength=1.8, borderaxespad=0.2)
ax.set_title('(b) amount of supervision', fontsize=8, pad=4)

fig.tight_layout(pad=0.35, w_pad=1.4)
fig.savefig(os.path.join(OUT, "fig_sweeps.pdf"))
plt.close(fig)

for f in ("fig_overview.pdf", "fig_evaldepth.pdf", "fig_heldout.pdf"):
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        os.remove(p)
        print("dihapus:", f)
print("fig_sweeps.pdf  depth %s  heldout %s" % (list(xs), list(hx)))
print("  %.1f KB" % (os.path.getsize(os.path.join(OUT, "fig_sweeps.pdf")) / 1024.0))
