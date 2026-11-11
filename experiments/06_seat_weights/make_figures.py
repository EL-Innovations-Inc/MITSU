"""Draw the two README figures into figures/. Needs matplotlib.

    python3 make_figures.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from variants import mitsu_c

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e9e8e3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order: blue, orange, aqua, yellow
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)


def shapes():
    u = torch.linspace(-5, 12, 1701, dtype=torch.float64)
    cs = [(2.0, "c = 2  MITSU"), (0.25, "c = 0.25"), (1.0, "c = 1"), (8.0, "c = 8")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for (c, lab), col in zip(cs, SERIES):
        y = mitsu_c(u, c)
        axes[0].plot(u.numpy(), y.numpy(), color=col, linewidth=2.6 if c == 2.0 else 1.8, label=lab, solid_capstyle="round")
        d = np.gradient(y.numpy(), u.numpy())
        axes[1].plot(u.numpy(), d, color=col, linewidth=2.6 if c == 2.0 else 1.8, solid_capstyle="round")
    axes[0].set_xlim(-5, 12); axes[0].set_ylim(-0.6, 8); axes[1].set_xlim(-5, 12); axes[1].set_ylim(-0.25, 1.05)
    axes[1].axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].text(-4.8, 0.52, "half open", fontsize=8.5, color=MUTED, va="bottom")
    for ax, t in ((axes[0], "The activation  s(s − 1) / (s + c)"), (axes[1], "Its slope: where the curve opens")):
        ax.axhline(0, color=AXIS, linewidth=1); clean(ax); ax.set_xlabel("signal u")
        ax.set_title(t, loc="left", fontsize=10, color=INK, pad=8)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=INK2)
    fig.suptitle("How c moves the cut-off", x=0.065, ha="left", fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, "figures", "c_shapes.png"), dpi=200); plt.close(fig)


def dose():
    S = json.load(open(os.path.join(HERE, "results", "summary.json")))
    panels = [("mnist", "MNIST accuracy, points", 1), ("fashion", "Fashion-MNIST accuracy, points", 1), ("gpt", "char-GPT loss, nats", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    for ax, (key, title, _) in zip(axes, panels):
        rows = {r["name"]: r for r in S[key]}
        cs = [0.25, 0.5, 1, 2, 4, 8]
        names = ["c=0.25", "c=0.5", "c=1", "mitsu", "c=4", "c=8"]
        d = [rows[n]["delta"] for n in names]; se = [0 if n == "mitsu" else rows[n]["se"] for n in names]
        ax.axhline(0, color=INK2, linewidth=1)
        ax.errorbar(cs, d, yerr=se, color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURF,
                    markeredgewidth=1.2, capsize=3, elinewidth=1.2)
        ax.set_xscale("log", base=2); ax.set_xticks(cs); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.minorticks_off(); clean(ax); ax.set_xlabel("c   (2 = current MITSU)")
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
        lo, hi = ax.get_ylim(); m = max(abs(lo), abs(hi)); ax.set_ylim(-m, m)
        better = "better ↑" if key != "gpt" else "worse ↑"
        ax.text(0.98, 0.96, better, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    axes[0].set_ylabel("difference from c = 2  (± 1 s.e.)")
    fig.suptitle("Every value of c against current MITSU", x=0.05, ha="left", fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, "figures", "c_dose_response.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    shapes()
    if os.path.exists(os.path.join(HERE, "results", "summary.json")):
        dose()
    print("wrote figures/")
