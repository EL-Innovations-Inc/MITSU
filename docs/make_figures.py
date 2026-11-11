"""Draw docs/figures/mitsu_curve.png (the README figure). Needs matplotlib.

    python3 docs/make_figures.py
"""
import math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mitsu import mitsu, mitsu_d  # noqa: E402

SURF, INK, INK2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e9e8e3"
COL = {"MITSU": "#2a78d6", "GELU": "#eb6834", "SiLU": "#1baf7a", "ReLU": "#eda100"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})

u = np.linspace(-6, 6, 2401)
sig = 1 / (1 + np.exp(-u))
Phi = 0.5 * (1 + np.vectorize(math.erf)(u / math.sqrt(2)))
phi = np.exp(-u * u / 2) / math.sqrt(2 * math.pi)
f = {"MITSU": mitsu(u), "GELU": u * Phi, "SiLU": u * sig, "ReLU": np.maximum(u, 0)}
d = {"MITSU": mitsu_d(u), "GELU": Phi + u * phi, "SiLU": sig * (1 + u * (1 - sig)), "ReLU": (u > 0).astype(float)}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
for ax, data, title in ((axes[0], f, "The activation"), (axes[1], d, "Its slope")):
    for name in ("ReLU", "SiLU", "GELU", "MITSU"):          # MITSU drawn last, on top
        ax.plot(u, data[name], color=COL[name], linewidth=2.6 if name == "MITSU" else 1.6, label=name,
                solid_capstyle="round")
    ax.axhline(0, color=AXIS, linewidth=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_xlim(-6, 6); ax.set_xlabel("input u")
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
axes[0].set_ylim(-0.6, 4.2); axes[1].set_ylim(-0.2, 1.25)
h, l = axes[0].get_legend_handles_labels()
axes[0].legend(h[::-1], l[::-1], frameon=False, loc="upper left", fontsize=8.5, labelcolor=INK2)
axes[1].axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
axes[1].text(-5.8, 1.03, "slope 1", fontsize=8.5, color=MUTED, va="bottom")
axes[0].plot([-0.5665], [-0.101], "o", color=COL["MITSU"], markersize=5, markeredgecolor=SURF, markeredgewidth=1.2, zorder=5)
axes[0].annotate("MITSU dip: −0.10 at u ≈ −0.57", xy=(-0.5665, -0.101), xytext=(0.7, -0.42), fontsize=8.5, color=INK2,
                 va="center", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkB=4))
fig.suptitle("MITSU beside the usual activations", x=0.065, ha="left", fontsize=12, fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.join(ROOT, "docs", "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "docs", "figures", "mitsu_curve.png"), dpi=200)
print("wrote docs/figures/mitsu_curve.png")
