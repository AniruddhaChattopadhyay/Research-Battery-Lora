"""Generate all figures for the paper."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Paper-quality defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (8, 5),
})


def load_all_metrics(results_dir: str) -> dict:
    """Load all metrics JSON files into a nested structure.

    Returns: {model: {quant: {benchmark: metrics_dict}}}
    """
    all_data = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_metrics.json"):
            continue
        parts = fname.replace("_metrics.json", "").rsplit("_", 2)
        if len(parts) != 3:
            continue
        model, quant, bench = parts
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        all_data.setdefault(model, {}).setdefault(quant, {})[bench] = data
    return all_data


def plot_ece_heatmap(results_dir: str, plots_dir: str, uq_method: str = "msp"):
    """Heatmap: models (y) x quant levels (x), color = ECE."""
    all_data = load_all_metrics(results_dir)
    if not all_data:
        print("  No metrics found, skipping heatmap")
        return

    models = sorted(all_data.keys())
    quants = sorted({q for m in all_data.values() for q in m})
    benchmarks = sorted({b for m in all_data.values() for q in m.values() for b in q})

    # Average ECE across benchmarks
    matrix = np.full((len(models), len(quants)), np.nan)
    for i, model in enumerate(models):
        for j, quant in enumerate(quants):
            eces = []
            for bench in benchmarks:
                try:
                    ece = all_data[model][quant][bench][uq_method]["ece"]
                    if ece is not None and not np.isnan(ece):
                        eces.append(ece)
                except (KeyError, TypeError):
                    pass
            if eces:
                matrix[i, j] = np.mean(eces)

    fig, ax = plt.subplots(figsize=(max(6, len(quants) * 1.5), max(4, len(models) * 0.6)))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=quants, yticklabels=models,
        ax=ax, cbar_kws={"label": "ECE (lower = better)"},
    )
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Model")
    ax.set_title(f"Expected Calibration Error — {uq_method.upper()}")

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"ece_heatmap_{uq_method}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_reliability_diagrams(results_dir: str, plots_dir: str, uq_method: str = "msp"):
    """Grid of reliability diagrams: rows = models, columns = quant levels."""
    all_data = load_all_metrics(results_dir)
    if not all_data:
        return

    models = sorted(all_data.keys())
    quants = sorted({q for m in all_data.values() for q in m})

    fig, axes = plt.subplots(
        len(models), len(quants),
        figsize=(3 * len(quants), 2.5 * len(models)),
        squeeze=False,
    )

    for i, model in enumerate(models):
        for j, quant in enumerate(quants):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")

            # Average reliability across benchmarks
            all_accs = []
            all_confs = []
            for bench_data in all_data.get(model, {}).get(quant, {}).values():
                try:
                    rel = bench_data[uq_method]["reliability"]
                    all_accs.append(rel["bin_accuracies"])
                    all_confs.append(rel["bin_confidences"])
                except (KeyError, TypeError):
                    pass

            if all_accs:
                mean_acc = np.nanmean(all_accs, axis=0)
                mean_conf = np.nanmean(all_confs, axis=0)
                mask = np.isfinite(mean_acc) & np.isfinite(mean_conf)
                ax.bar(
                    np.array(mean_conf)[mask], np.array(mean_acc)[mask],
                    width=0.08, alpha=0.7, color="steelblue", edgecolor="navy",
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(quant, fontsize=9)
            if j == 0:
                ax.set_ylabel(model, fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("Confidence", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Reliability Diagrams — {uq_method.upper()}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"reliability_{uq_method}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_ece_delta_bars(results_dir: str, plots_dir: str, uq_method: str = "msp"):
    """Bar chart: ECE change from FP16 baseline, grouped by model."""
    all_data = load_all_metrics(results_dir)
    if not all_data:
        return

    models = sorted(all_data.keys())
    quants = [q for q in sorted({q for m in all_data.values() for q in m}) if q != "fp16"]
    benchmarks = sorted({b for m in all_data.values() for q in m.values() for b in q})

    x = np.arange(len(models))
    width = 0.8 / max(len(quants), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))

    for k, quant in enumerate(quants):
        deltas = []
        for model in models:
            baseline_eces = []
            quant_eces = []
            for bench in benchmarks:
                try:
                    baseline_eces.append(all_data[model]["fp16"][bench][uq_method]["ece"])
                except (KeyError, TypeError):
                    pass
                try:
                    quant_eces.append(all_data[model][quant][bench][uq_method]["ece"])
                except (KeyError, TypeError):
                    pass
            if baseline_eces and quant_eces:
                deltas.append(np.mean(quant_eces) - np.mean(baseline_eces))
            else:
                deltas.append(0)

        ax.bar(x + k * width, deltas, width, label=quant)

    ax.set_xlabel("Model")
    ax.set_ylabel("ECE Delta (vs FP16)")
    ax.set_title(f"Calibration Change from Quantization — {uq_method.upper()}")
    ax.set_xticks(x + width * (len(quants) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"ece_delta_{uq_method}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_auroc_comparison(results_dir: str, plots_dir: str):
    """AUROC across UQ methods and quant levels — which methods survive?"""
    all_data = load_all_metrics(results_dir)
    if not all_data:
        return

    models = sorted(all_data.keys())
    quants = sorted({q for m in all_data.values() for q in m})
    benchmarks = sorted({b for m in all_data.values() for q in m.values() for b in q})

    # Find all UQ methods
    methods = set()
    for m in all_data.values():
        for q in m.values():
            for b in q.values():
                methods.update(k for k in b if not k.startswith("_"))
    methods = sorted(methods)

    # Average AUROC: (method, quant) averaged over models and benchmarks
    matrix = np.full((len(methods), len(quants)), np.nan)
    for i, method in enumerate(methods):
        for j, quant in enumerate(quants):
            aurocs = []
            for model in models:
                for bench in benchmarks:
                    try:
                        val = all_data[model][quant][bench][method]["auroc"]
                        if val is not None and not np.isnan(val):
                            aurocs.append(val)
                    except (KeyError, TypeError):
                        pass
            if aurocs:
                matrix[i, j] = np.mean(aurocs)

    fig, ax = plt.subplots(figsize=(max(6, len(quants) * 1.5), max(3, len(methods) * 0.6)))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlGn",
        xticklabels=quants, yticklabels=methods, ax=ax,
        cbar_kws={"label": "AUROC (higher = better)"},
    )
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("UQ Method")
    ax.set_title("Selective Prediction AUROC by Method and Quantization")

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "auroc_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def generate_all_plots(results_dir: str, plots_dir: str):
    """Generate all paper figures."""
    print("Generating plots...")
    for method in ("msp", "entropy", "log_likelihood", "temp_scaled_msp"):
        plot_ece_heatmap(results_dir, plots_dir, method)
        plot_reliability_diagrams(results_dir, plots_dir, method)
        plot_ece_delta_bars(results_dir, plots_dir, method)
    plot_auroc_comparison(results_dir, plots_dir)
    print("Done generating plots.")
