#!/usr/bin/env python3
"""Generate polished publication-quality figures for the tool-drift paper.

Uses hardcoded values from the paper tables to avoid JSON-file dependency.
Run with: /path/to/tool-drift/.venv/bin/python generate_polished_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication settings for LNCS readability
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "Times"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "text.usetex": False,
})

# Color palette: accessible, distinguishable in grayscale, print-safe
COLORS = {
    "original": "#2166AC",   # strong blue
    "drifted":  "#D6604D",   # muted red
    "naive":    "#B2ABD2",   # light purple
    "repaired": "#1B7837",   # dark green
}

OUTPUT_DIR = Path(__file__).parent / "figures"


def figure_accuracy_bars():
    """Figure 2: Oracle-constrained upper-bound DICE sweep across 4 models."""
    # Data from oracle-constrained DICE sweep
    models = ["Qwen-9B", "Qwen-35B", "Llama-4", "Mistral"]
    original = [0.750, 0.773, 0.737, 0.740]
    drifted  = [0.713, 0.727, 0.690, 0.777]
    repaired = [0.827, 0.840, 0.777, 0.813]

    n = len(models)
    width = 0.24
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.bar(x - width, original, width,
           label="Original", color=COLORS["original"],
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x, drifted, width,
           label="Drifted", color=COLORS["drifted"],
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width, repaired, width,
           label="Repaired", color=COLORS["repaired"],
           edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.55, n - 0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0, linestyle="--")

    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.8",
              fancybox=False)

    fig.tight_layout()
    out = OUTPUT_DIR / "accuracy_bars.pdf"
    fig.savefig(str(out))
    print(f"Saved {out}")
    plt.close(fig)


def figure_drift_ablation():
    """Figure 3: Drift-type ablation on DICE-300 with Qwen3.5-9B."""
    # Data from Table 4
    drift_types = ["Desc.", "Schema", "Candidates", "Combined"]
    original = [0.760, 0.767, 0.757, 0.763]
    drifted  = [0.747, 0.760, 0.710, 0.713]
    repaired = [0.813, 0.830, 0.817, 0.813]

    n = len(drift_types)
    width = 0.24
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.bar(x - width, original, width,
           label="Original", color=COLORS["original"],
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x, drifted, width,
           label="Drifted", color=COLORS["drifted"],
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width, repaired, width,
           label="Repaired", color=COLORS["repaired"],
           edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(drift_types)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.55, n - 0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0, linestyle="--")

    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.8",
              fancybox=False)

    fig.tight_layout()
    out = OUTPUT_DIR / "drift_ablation.pdf"
    fig.savefig(str(out))
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_accuracy_bars()
    figure_drift_ablation()
    print("Done.")
