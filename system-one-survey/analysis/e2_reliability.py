"""E2: reliability diagrams of the top answer for four systems and two task groups.

  uv run python analysis/e2_reliability.py

Writes paper1/figures/reliability.pdf. 10 equal-width bins; the size of a point shows
the number of answers in the bin.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT, SYSTEM_LABEL, TASK_GROUP, confidences, load_system  # noqa: E402

SYSTEMS = [("typesafe/jev-1.14", ("e2",)), ("perplexity/pplx-decider-v1.1-27b", ("e2",)),
           ("openai/gpt-6-luna-decisions", ("e2",)), ("anthropic/claude-sonnet-5.5", ("b1",))]
GROUPS = [("classification", "Classification tasks"), ("judging", "Judging tasks")]


def main():
    fig, axes = plt.subplots(len(GROUPS), len(SYSTEMS), figsize=(7.0, 3.6), sharex=True, sharey=True)
    for j, (model, tags) in enumerate(SYSTEMS):
        ans = load_system(model, tags)
        for i, (group, title) in enumerate(GROUPS):
            rows = [a for (t, iid, q), a in ans.items()
                    if TASK_GROUP[t] == group and "#" not in iid and a["dist"] is not None and a["gold"] is not None]
            conf, correct = confidences(rows)
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--")
            idx = np.minimum((conf * 10).astype(int), 9)
            xs, ys, ns = [], [], []
            for b in range(10):
                m = idx == b
                if m.sum() >= 5:
                    xs.append(conf[m].mean())
                    ys.append(correct[m].mean())
                    ns.append(m.sum())
            ax.plot(xs, ys, "-", color="C0", lw=0.8)
            ax.scatter(xs, ys, s=[6 + 40 * n / max(ns) for n in ns], color="C0", zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=5.5)
            if i == 0:
                ax.set_title(SYSTEM_LABEL[model], fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{title}\naccuracy", fontsize=6)
            if i == len(GROUPS) - 1:
                ax.set_xlabel("confidence", fontsize=6)
            ax.text(0.04, 0.9, f"n={len(rows)}", fontsize=5.5, transform=ax.transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "paper1" / "figures" / "reliability.pdf")
    print("written")


if __name__ == "__main__":
    main()
