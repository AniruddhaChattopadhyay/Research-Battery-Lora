"""Make the result figures and the results table for the meeting brief.

Usage: uv run python analysis/make_figures.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from pageindex_traces import trace  # noqa: E402
from summarize import TYPES, summarize  # noqa: E402

OUT = os.path.join(ROOT, "results")
PILOT = os.path.join(ROOT, "runs", "pilot")
PRICE_IN, PRICE_OUT = 2.0, 10.0  # gpt-5.6-sol, dollars per million tokens (OpenRouter list price)

# reference palette (light): first three categorical slots, fixed order; neutral gray for "no operators"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SYSTEMS = [  # (key, label, file, color)
    ("pageindex", "PageIndex (tree index)", "pageindex.jsonl", "#2a78d6"),
    ("fullcontext", "Full context (perfect retrieval)", "fullcontext.jsonl", "#eb6834"),
    ("graph", "Dual-view graph agent (ours, POC)", "graph_agent.jsonl", "#1baf7a"),
]
NEUTRAL = "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def load(name):
    return [json.loads(line) for line in open(os.path.join(PILOT, name))]


def cost_per_question(key, recs):
    if key == "pageindex":
        usage = json.load(open(os.path.join(PILOT, "pageindex.jsonl.usage.json")))
        p = sum(v["prompt"] for v in usage.values())
        c = sum(v["completion"] for v in usage.values())
    else:
        p = sum(r["usage"]["prompt"] for r in recs if r.get("usage"))
        c = sum(r["usage"]["completion"] for r in recs if r.get("usage"))
    n = len(recs)
    return {"input_tokens": p / n, "output_tokens": c / n, "dollars": (p * PRICE_IN + c * PRICE_OUT) / 1e6 / n,
            "seconds": sum(r["seconds"] for r in recs) / n}


def fig_accuracy(summ):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.26
    for k, (key, label, _, color) in enumerate(SYSTEMS):
        xs = [i + (k - 1) * w for i in range(len(TYPES))]
        ys = [summ[key][t]["per_question"] for t in TYPES]
        bars = ax.bar(xs, ys, w, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 1.2, "%.1f" % y, ha="center", va="bottom", fontsize=10, color=INK)
    n = summ["fullcontext"]
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels(["%s\n(n=%d)" % (t.capitalize(), n[t]["n"]) for t in TYPES])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Per-question accuracy (%)")
    ax.set_title("Same model, same questions: only the evidence access changes", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_accuracy_by_type.png"), dpi=200)
    plt.close(fig)


def fig_cost(costs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = [s[1] for s in SYSTEMS]
    for ax, (metric, title, fmt) in zip(axes, (("dollars", "Cost per question (US$, list price)", "$%.3f"),
                                               ("seconds", "Time per question (s)", "%.1f s"))):
        ys = [costs[s[0]][metric] for s in SYSTEMS]
        bars = ax.barh(range(len(SYSTEMS)), ys, 0.6, color=[s[3] for s in SYSTEMS], edgecolor=SURFACE, linewidth=2)
        for b, y in zip(bars, ys):
            ax.text(y + max(ys) * 0.02, b.get_y() + b.get_height() / 2, fmt % y, va="center", fontsize=10.5, color=INK)
        ax.set_yticks(range(len(SYSTEMS)))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_xlim(0, max(ys) * 1.3)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_cost_time.png"), dpi=200)
    plt.close(fig)


def fig_operators(ops):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    groups = ["predictive", "counterfactual"]
    series = [("observed_graph", "Causal graph of observed events only", NEUTRAL),
              ("with_operators", "+ intervention + world simulation", "#1baf7a")]
    w = 0.36
    for k, (key, label, color) in enumerate(series):
        xs = [i + (k - 0.5) * w for i in range(len(groups))]
        ys = [ops[key][g][1] for g in groups]
        bars = ax.bar(xs, ys, w, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 1.2, "%.1f" % y, ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["%s\n(n=%d)" % (g.capitalize(), ops["observed_graph"][g][2]) for g in groups])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Per-question accuracy (%)")
    ax.set_title("No LLM: graph operators on 3,000 held-out videos", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_operators_heldout.png"), dpi=200)
    plt.close(fig)


def fig_pages(pi_recs):
    pages = json.load(open(os.path.join(PILOT, "plant_log_30_runs_pages.json")))
    run_len = sum(b - a + 1 for a, b in pages.values()) / len(pages)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ys = []
    for t in TYPES:
        rs = [trace(r)["pages"] for r in pi_recs if r["type"] == t]
        ys.append(sum(rs) / len(rs))
    bars = ax.barh(range(len(TYPES)), ys, 0.6, color="#2a78d6", edgecolor=SURFACE, linewidth=2)
    for b, y in zip(bars, ys):
        ax.text(y + 0.15, b.get_y() + b.get_height() / 2, "%.1f pages" % y, va="center", fontsize=10.5, color=INK)
    ax.axvline(run_len, color=INK2, linewidth=1.5, linestyle="--")
    ax.text(run_len - 0.15, -0.55, "one run = %.1f pages" % run_len, ha="right", va="center", fontsize=10.5, color=INK2)
    ax.set_yticks(range(len(TYPES)))
    ax.set_yticklabels([t.capitalize() for t in TYPES])
    ax.set_ylim(len(TYPES) - 0.5, -0.9)
    ax.set_xlim(0, run_len * 1.15)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("PageIndex reads only a few pages of the right run", loc="left")
    ax.set_xlabel("Average pages read per question")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_pageindex_pages_read.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    recs = {key: load(f) for key, _, f, _ in SYSTEMS}
    summ = {key: summarize(os.path.join(PILOT, f)) for key, _, f, _ in SYSTEMS}
    costs = {key: cost_per_question(key, recs[key]) for key in recs}
    ops = json.load(open(os.path.join(ROOT, "runs", "operators_eval.json")))
    fig_accuracy(summ)
    fig_cost(costs)
    fig_operators(ops)
    fig_pages(recs["pageindex"])
    traces = [trace(r) for r in recs["pageindex"]]
    results = {"summary": summ, "cost": costs, "operators": ops,
               "pageindex_reading": {"summary_only_share": sum(t["summary_only"] for t in traces) / len(traces),
                                     "avg_pages": sum(t["pages"] for t in traces) / len(traces),
                                     "wrong_run_reads": sum(t["other_run_pages"] > 0 for t in traces)}}
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(results, indent=1)[:3000])
