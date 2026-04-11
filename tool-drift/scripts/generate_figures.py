"""Generate publication-quality figures for the tool-drift paper.

Produces:
1. Bar chart: Original / Drifted / Repaired accuracy across models
2. Drift-type ablation chart
3. Pipeline diagram (TikZ source)

Requires matplotlib: pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def figure_accuracy_bars(result_paths: list[tuple[str, Path]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.size"] = 11
    except ImportError:
        print("matplotlib not installed, skipping bar chart")
        return

    labels = []
    orig_scores = []
    drift_scores = []
    repair_scores = []
    naive_scores = []

    for label, path in result_paths:
        data = load_results(path)
        s = data["summary"]
        labels.append(label)
        orig_scores.append(s["original_score"])
        drift_scores.append(s["drifted_score"])
        repair_scores.append(s["repaired_score"])
        naive_scores.append(s.get("naive_retry_score"))

    n = len(labels)
    has_naive = any(v is not None for v in naive_scores)
    bar_count = 4 if has_naive else 3
    width = 0.8 / bar_count

    fig, ax = plt.subplots(figsize=(max(6, n * 2.2), 4.5))

    import numpy as np
    x = np.arange(n)
    offset = 0
    ax.bar(x + offset * width, orig_scores, width, label="Original", color="#4C72B0", edgecolor="white")
    offset += 1
    ax.bar(x + offset * width, drift_scores, width, label="Drifted", color="#DD8452", edgecolor="white")
    offset += 1
    if has_naive:
        naive_clean = [v if v is not None else 0 for v in naive_scores]
        ax.bar(x + offset * width, naive_clean, width, label="Naive Retry", color="#C4AD66", edgecolor="white")
        offset += 1
    ax.bar(x + offset * width, repair_scores, width, label="Repaired", color="#55A868", edgecolor="white")

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x + width * (bar_count - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Saved accuracy bar chart to {output_path}")
    plt.close(fig)


def figure_drift_ablation(result_paths: list[tuple[str, Path]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.size"] = 11
    except ImportError:
        print("matplotlib not installed, skipping drift ablation chart")
        return

    labels = []
    orig_scores = []
    drift_scores = []
    repair_scores = []

    for label, path in result_paths:
        data = load_results(path)
        s = data["summary"]
        labels.append(label)
        orig_scores.append(s["original_score"])
        drift_scores.append(s["drifted_score"])
        repair_scores.append(s["repaired_score"])

    n = len(labels)
    width = 0.25
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 4.5))
    ax.bar(x - width, orig_scores, width, label="Original", color="#4C72B0", edgecolor="white")
    ax.bar(x, drift_scores, width, label="Drifted", color="#DD8452", edgecolor="white")
    ax.bar(x + width, repair_scores, width, label="Repaired", color="#55A868", edgecolor="white")

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Saved drift ablation chart to {output_path}")
    plt.close(fig)


def generate_tikz_pipeline() -> str:
    return r"""\begin{figure}[t]
\centering
\begin{tikzpicture}[
    node distance=1.2cm,
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=0.8cm, align=center, font=\small},
    decision/.style={draw, diamond, aspect=2, minimum width=1.5cm, align=center, font=\small},
    arrow/.style={->, >=stealth, thick},
]
\node[box] (input) {User Context\\+ Drifted Tools};
\node[box, right=of input] (model) {Base Model\\$M(x, \Delta(I))$};
\node[decision, right=of model] (valid) {Valid?};
\node[box, above right=0.6cm and 1.2cm of valid] (pass) {Output\\(pass-through)};
\node[box, below right=0.6cm and 1.2cm of valid] (repair) {Canonical Card\\+ Validation Errors};
\node[box, right=of repair] (repairmodel) {Repair Call\\(one-shot)};
\node[box, right=of repairmodel] (output) {Repaired\\Output};

\draw[arrow] (input) -- (model);
\draw[arrow] (model) -- (valid);
\draw[arrow] (valid) -- node[above, font=\scriptsize] {Yes} (pass);
\draw[arrow] (valid) -- node[below, font=\scriptsize] {No} (repair);
\draw[arrow] (repair) -- (repairmodel);
\draw[arrow] (repairmodel) -- (output);
\end{tikzpicture}
\caption{SchemaShield-Lite pipeline. Invalid drifted tool calls are repaired using the canonical tool card and structured validation errors.}
\label{fig:pipeline}
\end{figure}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for tool-drift paper")
    parser.add_argument("--results", nargs="+", help="label:path pairs for accuracy bar chart (e.g., 'Qwen-9B:path/to/results.json')")
    parser.add_argument("--drift-ablation", nargs="+", help="label:path pairs for drift ablation chart")
    parser.add_argument("--output-dir", default="figures", help="Directory to save figures")
    parser.add_argument("--tikz", action="store_true", help="Print TikZ pipeline diagram source")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tikz:
        print(generate_tikz_pipeline())

    if args.results:
        pairs = []
        for item in args.results:
            parts = item.split(":", 1)
            if len(parts) == 2:
                pairs.append((parts[0], Path(parts[1])))
            else:
                pairs.append((Path(parts[0]).stem, Path(parts[0])))
        figure_accuracy_bars(pairs, output_dir / "accuracy_bars.pdf")

    if args.drift_ablation:
        pairs = []
        for item in args.drift_ablation:
            parts = item.split(":", 1)
            if len(parts) == 2:
                pairs.append((parts[0], Path(parts[1])))
            else:
                pairs.append((Path(parts[0]).stem, Path(parts[0])))
        figure_drift_ablation(pairs, output_dir / "drift_ablation.pdf")


if __name__ == "__main__":
    main()
