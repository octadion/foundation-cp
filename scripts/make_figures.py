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

# ========================================================== method figure (nb14)
FIT = os.path.join(REPO, "fit_primary.npz")
if not os.path.exists(FIT):
    print("lewati fig_method.pdf: %s belum ada (jalankan notebooks/14_class_counts.ipynb)"
          % os.path.basename(FIT))
else:
    z = np.load(FIT, allow_pickle=True)
    obs, hat = np.asarray(z["delta_obs"], float), np.asarray(z["delta_hat"], float)
    seen, held = np.asarray(z["seen"], int), np.asarray(z["heldout"], int)
    lam = float(z["lam"])

    ok = seen[np.isfinite(obs[seen])]          # labelled classes with an observed offset
    fig, ax = plt.subplots(figsize=(3.25, 2.45))

    lim = np.percentile(np.concatenate([obs[ok], hat[seen], hat[held]]), [1, 99])
    pad = 0.12 * (lim[1] - lim[0])
    lo, hi = lim[0] - pad, lim[1] + pad

    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.scatter(hat[ok], obs[ok], s=5, color=BLUE, alpha=0.45, linewidths=0,
               zorder=3, label="labelled class ($n_y \\geq n_{\\mathrm{cal}}$)")
    # held-out classes have no observed offset; they exist only on the horizontal axis
    ax.plot(hat[held], np.full(len(held), lo + 0.035 * (hi - lo)), "|",
            color=RED, ms=4, mew=0.7, zorder=4,
            label="held out ($n_y = 0$)")

    r = np.corrcoef(hat[ok], obs[ok])[0, 1]
    ax.text(0.03, 0.96, "$r = {:.2f}$,  $\\hat\\lambda = {:.2f}$".format(r, lam),
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"predicted offset $g_\theta(\varphi(y))$")
    ax.set_ylabel(r"observed offset $\delta_y$")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4,
              borderaxespad=0.3, labelspacing=0.35, scatterpoints=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_method.pdf"))
    plt.close(fig)
    print("fig_method.pdf")
    print("  %d kelas berlabel dengan offset teramati, %d held-out, r = %.2f, lambda = %.2f"
          % (len(ok), len(held), r, lam))

# ======================================================= motivating figure (nb14)
# Per-class evaluation counts are not in the run reports, so this block needs
# class_counts.json from notebook 14. Without it the rest of the script still runs.
CC = os.path.join(REPO, "class_counts.json")
if not os.path.exists(CC):
    print("lewati fig_measurable.pdf: %s belum ada (jalankan notebooks/14_class_counts.ipynb)"
          % os.path.basename(CC))
else:
    cc = json.load(io.open(CC, encoding="utf-8"))
    # name in the JSON -> label in the figure, in the order we want them drawn
    WANT = [("ccc_imagenet", "ImageNet", "-"),
            ("ltc_plantnet", "Pl@ntNet-300K", "--"),
            ("ltc_inaturalist", "iNaturalist-2018", ":")]
    THRESHOLD = 30

    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    drawn = []
    for key, label, ls in WANT:
        if key not in cc:
            continue
        c = np.asarray(cc[key]["counts"], float)
        c = np.sort(c[c > 0])[::-1]
        x = np.arange(1, len(c) + 1) / float(len(c)) * 100.0
        ax.plot(x, c, ls=ls, color=BLUE if key == "ccc_imagenet" else
                (RED if key.endswith("plantnet") else GREY), label=label)
        below = 100.0 * float((c < THRESHOLD).mean())
        drawn.append((label, len(c), below))
    ax.axhline(THRESHOLD, color="#000000", lw=0.7, ls=(0, (5, 3)))
    ax.text(99, THRESHOLD * 1.30, "worst-class coverage measurable above",
            fontsize=6.6, ha="right")
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("classes, sorted by count (\%)" if plt.rcParams.get("text.usetex")
                  else "classes, sorted by count (%)")
    ax.set_ylabel("evaluation examples per class")
    ax.legend(frameon=False, loc="upper right", handlelength=2.0,
              borderaxespad=0.3, labelspacing=0.35)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_measurable.pdf"))
    plt.close(fig)
    print("fig_measurable.pdf")
    for label, k, below in drawn:
        print("  %-18s %5d kelas, %.0f%% di bawah %d contoh" % (label, k, below, THRESHOLD))

for f in ("fig_overview.pdf", "fig_evaldepth.pdf", "fig_heldout.pdf"):
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        os.remove(p)
        print("dihapus:", f)
print("fig_sweeps.pdf  depth %s  heldout %s" % (list(xs), list(hx)))
print("  %.1f KB" % (os.path.getsize(os.path.join(OUT, "fig_sweeps.pdf")) / 1024.0))
