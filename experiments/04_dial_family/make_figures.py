"""Draw the two README figures into figures/. Needs matplotlib.

    python3 make_figures.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from variants import dial

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e9e8e3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order: blue, orange, aqua, yellow
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def shapes():
    u = torch.linspace(-5, 5, 801, dtype=torch.float64)
    dials = [((1.0, 1.0), "(1, 1)  MITSU"), ((0.75, 0.75), "(0.75, 0.75)"), ((0.5, 0.5), "(0.5, 0.5)"), ((1.5, 1.5), "(1.5, 1.5)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True)
    for ax, lg, title, ylim in ((axes[0], False, "Plain: f − resting level", (-0.8, 5.0)), (axes[1], True, "Log version: log f − log resting level", (-0.8, 2.6))):
        for ((a, b), lab), c in zip(dials, SERIES):
            y = dial(u, a, b, lg).numpy()
            ax.plot(u.numpy(), y, color=c, linewidth=2, label=lab, solid_capstyle="round")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_ylim(*ylim); ax.set_xlim(-5, 5); clean(ax)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("signal u")
    axes[0].set_ylabel("output")
    axes[0].legend(title="dials (a, b)", frameon=False, loc="upper left", fontsize=8.5, title_fontsize=8.5, labelcolor=INK2)
    fig.suptitle("How the two dials reshape the curve", x=0.065, ha="left", fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, "figures", "dial_shapes.png"), dpi=200); plt.close(fig)


def gpt_bars():
    rows = [r for r in json.load(open(os.path.join(HERE, "results", "summary.json")))["gpt"] if r["name"] != "mitsu"]
    rows.sort(key=lambda r: r["delta"], reverse=True)
    names = [r["name"] for r in rows]; d = [r["delta"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    ax.axvspan(-0.05, 0.05, color="#eeede8", zorder=0)
    ax.barh(range(len(rows)), d, height=0.62, color=SERIES[0], zorder=2)
    ax.axvline(0, color=INK2, linewidth=1, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, color=INK2, fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3); ax.set_xlim(-0.06, 0.145)
    clean(ax); ax.grid(axis="y", visible=False)
    ax.set_xlabel("validation loss minus current MITSU, nats (3 paired seeds).  Right of zero = worse than MITSU")
    for i, r in enumerate(rows):
        if abs(r["delta"]) >= 0.09:
            ax.text(r["delta"] + 0.003, i, f"{r['delta']:+.3f}", va="center", fontsize=8.5, color=INK)
    ax.text(0.0, len(rows) - 0.25, "parity band ±0.05", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    fig.suptitle("Character-GPT: every variant against current MITSU", x=0.02, ha="left", fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(HERE, "figures", "char_gpt_delta.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    shapes(); gpt_bars(); print("wrote figures/dial_shapes.png, figures/char_gpt_delta.png")
