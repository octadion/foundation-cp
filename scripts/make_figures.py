# -*- coding: utf-8 -*-
"""Generate the paper's figures as PDFs, from the run reports and the notebook 14 dumps.

Numbers in a figure should not be retyped any more than numbers in a table, so everything
here is read from the same JSON and NPZ the tables and the text use.

Style follows Ding et al. (ICLR 2026), read off their PDF: multi-panel grids spanning the full
text width with a shared y-axis where the units allow it, log axes where the data spans
decades, small frameless legends, and nothing inside the axes except data and one reference
line. Explanation lives in the caption.

  fig_sweeps.pdf   three panels: how much evaluation data each dataset has, what the
                   statistic does as evaluation depth changes, and how little supervision
                   the method needs. Needs class_counts.json for panel (a).
  fig_method.pdf   observed offset against predicted offset. Needs fit_primary.npz.

Both extra inputs come from notebooks/14_class_counts.ipynb; each block is guarded, so the
script runs without them.
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
COL = 3.25                 # WACV, one column
INK = "#000000"
BLUE = "#2f6fad"
RED = "#c03a2b"
GREEN = "#3d7a52"
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

CHECK_PNG = os.environ.get("PCC_FIG_PNG")     # set to a directory to also write PNGs


def _also_png(fig, name):
    if CHECK_PNG:
        fig.savefig(os.path.join(CHECK_PNG, name + ".png"), dpi=190)


def load(tag):
    return [json.load(io.open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(D, tag + "_s*.json")))]


def series(tags):
    """(x, mean, lo, hi, ceiling) for a list of (report tag, x value)."""
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


def sweep_panel(ax, xs, mu, lo, hi, ceil, title, xlabel, log=False, ticks=None):
    ax.axhline(0, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=BLUE, marker="o",
                capsize=1.8, elinewidth=0.7, zorder=3, label="PCC")
    ax.plot(xs, ceil, color=RED, marker="s", ls="--", zorder=2, label="oracle ceiling")
    if log:
        ax.set_xscale("log")
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8, pad=4)
    h, l = ax.get_legend_handles_labels()
    order = [l.index("PCC"), l.index("oracle ceiling")]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False,
              loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)


# ================================================================= fig_sweeps
fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25))
ax_a, ax_b, ax_c = axes

# --------------------------------------- (a) how much evaluation data exists
CC = os.path.join(REPO, "class_counts.json")
THRESHOLD = 30
if os.path.exists(CC):
    cc = json.load(io.open(CC, encoding="utf-8"))
    for key, label, ls, col in (("ccc_imagenet", "ImageNet", "-", BLUE),
                                ("ltc_plantnet", "Pl@ntNet-300K", "--", RED),
                                ("ltc_inaturalist", "iNaturalist-2018", "-.", GREEN)):
        if key not in cc:
            continue
        c = np.asarray(cc[key]["counts"], float)
        c = np.sort(c[c > 0])[::-1]
        ax_a.plot(np.arange(1, len(c) + 1) / float(len(c)) * 100.0, c,
                  ls=ls, color=col, label=label)
        print("  %-18s %5d kelas, %2.0f%% di bawah %d contoh"
              % (label, len(c), 100.0 * float((c < THRESHOLD).mean()), THRESHOLD))
    ax_a.axhline(THRESHOLD, color=INK, lw=0.7, ls=(0, (5, 3)))
    ax_a.set_yscale("log")
    ax_a.set_xlim(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_ylim(0.7, 3e3)
    ax_a.set_xlabel("classes, sorted (%)")
    ax_a.set_ylabel("evaluation examples per class")
    # nudged up so the entries clear the threshold line
    ax_a.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.70),
                handlelength=1.8, borderaxespad=0.4, labelspacing=0.3)
    ax_a.set_title("(a) available evaluation data", fontsize=8, pad=4)
else:
    ax_a.axis("off")
    print("  panel (a) dilewati: class_counts.json belum ada")

# ------------------------------------------------------- (b) evaluation depth
xs, mu, lo, hi, ceil = series([
    ("nb12_F_evaldepth_eval3", 3), ("nb12_F_evaldepth_eval10", 10),
    ("nb12_F_evaldepth_eval35", 35), ("nb12_F_evaldepth_eval75", 75),
    ("nb12_F_evaldepth_evalNone", 76)])
sweep_panel(ax_b, xs, mu, lo, hi, ceil, "(b) evaluation depth",
            "evaluation examples per class", log=True, ticks=[3, 10, 35, 75])
ax_b.set_xlim(2.5, 92)
ax_b.set_ylim(-0.06, 0.20)
ax_b.set_ylabel(r"$\Delta$ worst-class coverage")

# ---------------------------------------------------- (c) held-out fraction
hx, hm, hlo, hhi, hceil = series([
    ("nb12_H_heldout_ho0p1", 10), ("nb12_H_heldout_ho0p3", 30),
    ("nb12_H_heldout_ho0p5", 50), ("nb12_H_heldout_ho0p7", 70),
    ("nb12_H_heldout_ho0p9", 90)])
sweep_panel(ax_c, hx, hm, hlo, hhi, hceil, "(c) amount of supervision",
            "classes with no calibration data (%)", ticks=[10, 30, 50, 70, 90])
ax_c.set_xlim(4, 96)
ax_c.set_ylim(-0.06, 0.20)

fig.tight_layout(pad=0.35, w_pad=1.6)
fig.savefig(os.path.join(OUT, "fig_sweeps.pdf"))
_also_png(fig, "chk_sweeps")
plt.close(fig)
print("fig_sweeps.pdf  depth %s  heldout %s" % (list(xs), list(hx)))

# ================================================================= fig_method
FIT = os.path.join(REPO, "fit_primary.npz")
if not os.path.exists(FIT):
    print("lewati fig_method.pdf: %s belum ada (jalankan notebooks/14_class_counts.ipynb)"
          % os.path.basename(FIT))
else:
    z = np.load(FIT, allow_pickle=True)
    obs, hat = np.asarray(z["delta_obs"], float), np.asarray(z["delta_hat"], float)
    seen, held = np.asarray(z["seen"], int), np.asarray(z["heldout"], int)
    lam = float(z["lam"])
    ok = seen[np.isfinite(obs[seen])]

    fig, ax = plt.subplots(figsize=(COL, 2.70))
    lim = np.percentile(np.concatenate([obs[ok], hat[seen], hat[held]]), [1, 99])
    pad = 0.12 * (lim[1] - lim[0])
    lo_, hi_ = lim[0] - pad, lim[1] + pad

    ax.plot([lo_, hi_], [lo_, hi_], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.scatter(hat[ok], obs[ok], s=5, color=BLUE, alpha=0.45, linewidths=0, zorder=3,
               label=r"labelled class ($n_y \geq n_{\mathrm{cal}}$)")
    ax.plot(hat[held], np.full(len(held), lo_ + 0.035 * (hi_ - lo_)), "|",
            color=RED, ms=4, mew=0.7, zorder=4, label=r"held out ($n_y = 0$)")

    r = np.corrcoef(hat[ok], obs[ok])[0, 1]
    ax.text(0.03, 0.96, r"$r = {:.2f}$,  $\hat\lambda = {:.2f}$".format(r, lam),
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlim(lo_, hi_)
    ax.set_ylim(lo_, hi_)
    ax.set_xlabel(r"predicted offset $g_\theta(\varphi(y))$")
    ax.set_ylabel(r"observed offset $\delta_y$")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, handlelength=1.2, columnspacing=1.4, handletextpad=0.4,
              borderaxespad=0.0, scatterpoints=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_method.pdf"))
    _also_png(fig, "chk_method")
    plt.close(fig)
    print("fig_method.pdf")
    print("  %d kelas berlabel dengan offset teramati, %d held-out, r = %.2f, lambda = %.2f"
          % (len(ok), len(held), r, lam))

# ================================================================= fig_teaser
# The body figure. A class with zero calibration examples still gets a threshold of its own,
# read off where its row sits among the others in the classifier head. The class shown is
# fixed by a rule stated before any image was looked at (notebooks/15): the held-out class
# with the lowest coverage under the marginal threshold. Every number is read from the dump,
# including the emitted threshold, which is q + lam*delta_hat + c and not q + lam*delta_hat.
ASSETS = os.path.join(REPO, "figassets")
META = os.path.join(ASSETS, "meta.json")
if not (os.path.exists(FIT) and os.path.exists(META)):
    print("lewati fig_teaser.pdf: butuh fit_primary.npz dan figassets/ "
          "(jalankan notebooks/15_figure_assets.ipynb)")
else:
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyArrowPatch, Rectangle

    mt = json.load(io.open(META, encoding="utf-8"))
    z = np.load(FIT, allow_pickle=True)
    tgt, nbrs = int(mt["target"]), [int(c) for c in mt["neighbours"]]
    held = np.asarray(z["heldout"], int)
    i = int(np.where(held == tgt)[0][0])
    c_unc = float(z["cov_heldout_uncorrected"][i])
    c_pcc = float(z["cov_heldout_pcc"][i])
    c_orc = float(z["cov_heldout_oracle"][i])
    q = float(z["q_global"])
    q_y = float(np.asarray(z["thresholds"], float)[tgt])   # includes the recalibration scalar
    alpha = float(z["alpha"])

    lam = float(z["lam"])
    d_hat = float(np.asarray(z["delta_hat"], float)[tgt])

    fig = plt.figure(figsize=(COL, 2.60))
    gs_top = fig.add_gridspec(1, 4, left=0.02, right=0.98, top=0.912, bottom=0.592,
                              wspace=0.09)
    gs_bot = fig.add_gridspec(1, 1, left=0.09, right=0.97, top=0.340, bottom=0.205)

    for j_, c in enumerate([tgt] + nbrs):
        ax = fig.add_subplot(gs_top[0, j_])
        ax.imshow(mpimg.imread(os.path.join(ASSETS, "cls%d.jpg" % c)))
        ax.set_xticks([]); ax.set_yticks([])
        edge, lw = (RED, 1.5) if c == tgt else (GREY, 0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(edge); sp.set_linewidth(lw); sp.set_visible(True)
        ax.set_xlabel(mt["names"][str(c)], fontsize=5.7, labelpad=1.6,
                      color=INK if c == tgt else GREY)

    fig.text(0.135, 0.949, r"held out, $n_y = 0$", ha="center", va="bottom",
             fontsize=6.2, color=RED)
    fig.text(0.645, 0.949, "its three nearest rows in the head",
             ha="center", va="bottom", fontsize=6.2, color=GREY)

    # the step between the two rows, which is the whole method
    fig.text(0.5, 0.512,
             r"$\varphi(y)$ from the head weights  $\rightarrow$  "
             r"$g_\theta$  $\rightarrow$  $\lambda\,\hat\delta_y = %+.3f$" % (lam * d_hat),
             ha="center", va="center", fontsize=6.4, color=INK)
    fig.text(0.5, 0.430,
             r"threshold  $\hat{q} = %.4f \;\rightarrow\; %.4f$" % (q, q_y),
             ha="center", va="center", fontsize=6.4, color=INK)

    # what the new threshold bought, on the coverage axis
    ax = fig.add_subplot(gs_bot[0, 0])
    ax.set_xlim(0.60, 0.95)
    ax.set_ylim(-1.05, 1.35)
    ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([0.65, 0.75, 0.85, 0.90])
    ax.set_xlabel(r"coverage of class %d, %s" % (tgt, mt["names"][str(tgt)]),
                  fontsize=6.8, labelpad=1.5)
    ax.axvline(1.0 - alpha, color=INK, lw=0.7, ls=(0, (3, 2.5)), zorder=1)
    ax.text(1.0 - alpha - 0.004, 1.30, r"target $1-\alpha$", fontsize=6.0, ha="right",
            va="top", color=INK)

    ax.add_patch(FancyArrowPatch((c_unc, 0.30), (c_pcc, 0.30),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=BLUE, shrinkA=1.5, shrinkB=1.5, zorder=3))
    ax.plot([c_unc], [0.30], "o", ms=4.0, color=GREY, zorder=4)
    ax.plot([c_pcc], [0.30], "o", ms=4.6, color=BLUE, zorder=4)
    ax.plot([c_orc], [0.30], "o", ms=4.6, mfc="none", mew=0.9, color=GREEN, zorder=4)
    for x, col, lab in ((c_unc, GREY, "marginal"), (c_pcc, BLUE, "PCC"),
                        (c_orc, GREEN, "oracle")):
        ax.text(x, 0.98, "%.3f" % x, fontsize=6.4, ha="center", color=col)
        ax.text(x, -0.34, lab, fontsize=6.2, ha="center", va="top", color=col)

    fig.savefig(os.path.join(OUT, "fig_teaser.pdf"))
    _also_png(fig, "chk_teaser")
    plt.close(fig)
    print("fig_teaser.pdf")
    print("  kelas %d (%s), tetangga %s" % (tgt, mt["names"][str(tgt)], nbrs))
    print("  coverage %.4f -> %.4f (oracle %.4f); ambang %.4f -> %.4f"
          % (c_unc, c_pcc, c_orc, q, q_y))

# =============================================================== fig_coverage
# The appendix figure. Worst-class coverage is a minimum, and a minimum says nothing about
# the shape it was taken from; the sorted curve does. The two label spaces are drawn apart
# because the paper never averages them together.
if not os.path.exists(FIT):
    print("lewati fig_coverage.pdf: fit_primary.npz belum ada")
else:
    z = np.load(FIT, allow_pickle=True)
    alpha = float(z["alpha"])
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.35), sharey=True)
    for ax, sp, ttl in ((axes[0], "seen", "labelled classes"),
                        (axes[1], "heldout", "held-out classes")):
        arms = (("uncorrected", GREY, "marginal threshold", (0, (3, 2))),
                ("pcc", BLUE, "PCC", "-"),
                ("oracle", GREEN, "oracle", (0, (1.2, 1.6))))
        n, mins = None, []
        for arm, col, lab, ls in arms:
            v = np.asarray(z["cov_%s_%s" % (sp, arm)], float)
            v = np.sort(v[np.isfinite(v)])
            n = len(v)
            mins.append((lab, col, v[0]))
            # rank, not percentile: worst-class coverage is decided by the first few
            # classes, and a linear percentile axis compresses them into one pixel
            ax.plot(np.arange(1, n + 1), v, color=col, ls=ls, lw=1.0, label=lab)
        ax.axhline(1.0 - alpha, color=INK, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(1, n)
        ax.set_ylim(0.55, 1.015)
        ax.set_xlabel("class rank by coverage")
        ax.set_title("%s ($%d$)" % (ttl, n), fontsize=8, pad=3)
        # placed where no curve runs: past rank 20 every arm is well above 0.66
        for k, (lab, col, mv) in enumerate(mins):
            ax.text(22.0, 0.578 + 0.036 * (len(mins) - 1 - k), "%.3f" % mv,
                    fontsize=6.6, color=col, ha="left", va="center")
        ax.text(22.0, 0.578 + 0.036 * len(mins), "worst", fontsize=6.6, color=INK,
                ha="left", va="center")
    axes[0].set_ylabel("class-conditional coverage")
    axes[0].text(1.6, 1.0 - alpha + 0.010, r"$1-\alpha$", fontsize=7, va="bottom")
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.8,
                   borderaxespad=0.5, labelspacing=0.28)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_coverage.pdf"))
    _also_png(fig, "chk_coverage")
    plt.close(fig)
    print("fig_coverage.pdf")
    for sp in ("seen", "heldout"):
        cu = np.asarray(z["cov_%s_uncorrected" % sp], float)
        cp = np.asarray(z["cov_%s_pcc" % sp], float)
        print("  %-8s worst %.4f -> %.4f   median %.4f -> %.4f"
              % (sp, np.nanmin(cu), np.nanmin(cp), np.nanmedian(cu), np.nanmedian(cp)))

# figures from earlier drafts, removed so a stale PDF cannot be compiled by accident
for f in ("fig_overview.pdf", "fig_evaldepth.pdf", "fig_heldout.pdf", "fig_measurable.pdf"):
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        os.remove(p)
        print("dihapus:", f)

print()
for f in sorted(os.listdir(OUT)):
    print("  %-22s %6.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024.0))
