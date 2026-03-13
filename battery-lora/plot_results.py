"""
Plotting Module
===============
Generates all figures needed for the paper.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_convergence_curves(experiment_dirs: dict, save_path: str = "plots"):
    """
    Plot training loss convergence for all methods.

    Args:
        experiment_dirs: {method_name: results_dir_path}
    """
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots()

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548", "#607D8B"]

    for i, (name, exp_dir) in enumerate(experiment_dirs.items()):
        with open(os.path.join(exp_dir, "round_metrics.json")) as f:
            metrics = json.load(f)

        rounds = [m["round"] for m in metrics]
        losses = [m["avg_train_loss"] for m in metrics]

        # Smooth with moving average
        window = 5
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            rounds_smooth = rounds[window - 1:]
        else:
            smoothed = losses
            rounds_smooth = rounds

        color = colors[i % len(colors)]
        ax.plot(rounds_smooth, smoothed, label=name, color=color, linewidth=2)
        # Light raw data in background
        ax.plot(rounds, losses, color=color, alpha=0.15, linewidth=1)

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_path, "convergence_curves.png"))
    plt.close(fig)
    print(f"Saved: {save_path}/convergence_curves.png")


def plot_battery_trajectories(results_dir: str, save_path: str = "plots"):
    """
    Plot battery level over time for all devices, colored by tier.
    """
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(results_dir, "device_stats.json")) as f:
        device_stats = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    tier_colors = {"high": "#2196F3", "mid": "#4CAF50", "low": "#FF9800"}
    tier_plotted = set()

    for cid, dev in device_stats.items():
        tier = dev["tier"]
        history = dev["battery_history"]
        color = tier_colors.get(tier, "gray")
        label = f"{tier.capitalize()} tier" if tier not in tier_plotted else None
        tier_plotted.add(tier)
        ax.plot(range(len(history)), history, color=color, alpha=0.3, linewidth=0.8, label=label)

    # Dropout threshold line
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Dropout threshold (10%)")

    ax.set_xlabel("Round")
    ax.set_ylabel("Battery Level (%)")
    ax.set_title("Device Battery Trajectories")
    ax.legend()
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_path, "battery_trajectories.png"))
    plt.close(fig)
    print(f"Saved: {save_path}/battery_trajectories.png")


def plot_energy_fairness(experiment_dirs: dict, save_path: str = "plots"):
    """
    Box plot comparing energy consumption distribution across methods.
    """
    os.makedirs(save_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []

    for name, exp_dir in experiment_dirs.items():
        with open(os.path.join(exp_dir, "device_stats.json")) as f:
            device_stats = json.load(f)
        energies = [d["total_energy_wh"] for d in device_stats.values()]
        data.append(energies)
        labels.append(name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Total Energy Consumed (Wh)")
    ax.set_title("Energy Consumption Distribution Across Devices")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    fig.savefig(os.path.join(save_path, "energy_fairness.png"))
    plt.close(fig)
    print(f"Saved: {save_path}/energy_fairness.png")


def plot_rank_distribution_over_time(results_dir: str, save_path: str = "plots"):
    """
    Stacked area chart showing rank distribution across rounds.
    """
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(results_dir, "round_metrics.json")) as f:
        metrics = json.load(f)

    all_ranks = [2, 4, 8, 16, 32]
    rounds = []
    rank_counts = {r: [] for r in all_ranks}

    for m in metrics:
        rounds.append(m["round"])
        total = m.get("num_clients_trained", 0)
        for r in all_ranks:
            count = m.get("ranks_used", {}).get(str(r), 0)
            rank_counts[r].append(count)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#0D47A1"]
    ax.stackplot(
        rounds,
        [rank_counts[r] for r in all_ranks],
        labels=[f"Rank {r}" for r in all_ranks],
        colors=colors,
        alpha=0.8,
    )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Number of Clients")
    ax.set_title("LoRA Rank Distribution Over Training Rounds")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_path, "rank_distribution.png"))
    plt.close(fig)
    print(f"Saved: {save_path}/rank_distribution.png")


def plot_dropout_comparison(experiment_dirs: dict, save_path: str = "plots"):
    """
    Bar chart comparing dropout rates across methods.
    """
    os.makedirs(save_path, exist_ok=True)

    fig, ax = plt.subplots()

    names = []
    dropout_rates = []

    for name, exp_dir in experiment_dirs.items():
        with open(os.path.join(exp_dir, "battery_stats.json")) as f:
            stats = json.load(f)
        names.append(name)
        dropout_rates.append(stats.get("dropout_rate", 0) * 100)

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    bars = ax.bar(names, dropout_rates, color=colors[: len(names)], alpha=0.7)

    # Add value labels on bars
    for bar, rate in zip(bars, dropout_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center",
            fontsize=10,
        )

    ax.set_ylabel("Client Dropout Rate (%)")
    ax.set_title("Client Dropout Rate Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    fig.savefig(os.path.join(save_path, "dropout_comparison.png"))
    plt.close(fig)
    print(f"Saved: {save_path}/dropout_comparison.png")


def plot_comparison_table(experiment_dirs: dict, save_path: str = "plots"):
    """
    Generate a summary comparison table as an image (for the paper).
    """
    from evaluate import compare_experiments

    os.makedirs(save_path, exist_ok=True)

    comparison = compare_experiments(list(experiment_dirs.values()))

    # Format as table
    fig, ax = plt.subplots(figsize=(14, len(comparison) * 0.6 + 2))
    ax.axis("off")

    columns = [
        "Method",
        "Final Loss",
        "Energy (Wh)",
        "Energy Std",
        "Jain Index",
        "Dropout %",
        "Comm (MB)",
    ]

    rows = []
    for name, metrics in comparison.items():
        rows.append([
            name,
            f"{metrics['final_loss']:.4f}" if metrics['final_loss'] else "N/A",
            f"{metrics['total_energy_wh']:.2f}",
            f"{metrics['energy_std_wh']:.2f}",
            f"{metrics['jain_fairness']:.4f}",
            f"{metrics['dropout_rate']:.1%}",
            f"{metrics['communication_mb']:.1f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#1E88E5")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.savefig(os.path.join(save_path, "comparison_table.png"))
    plt.close(fig)
    print(f"Saved: {save_path}/comparison_table.png")


def generate_all_plots(experiment_dirs: dict, our_method_dir: str):
    """Generate all plots for the paper."""
    save_path = "plots"
    os.makedirs(save_path, exist_ok=True)

    print("\nGenerating plots...")
    print("-" * 40)

    # 1. Convergence curves
    plot_convergence_curves(experiment_dirs, save_path)

    # 2. Battery trajectories (for our method)
    plot_battery_trajectories(our_method_dir, save_path)

    # 3. Energy fairness
    plot_energy_fairness(experiment_dirs, save_path)

    # 4. Rank distribution (for our method)
    plot_rank_distribution_over_time(our_method_dir, save_path)

    # 5. Dropout comparison
    plot_dropout_comparison(experiment_dirs, save_path)

    # 6. Comparison table
    plot_comparison_table(experiment_dirs, save_path)

    print("-" * 40)
    print(f"All plots saved to {save_path}/")
