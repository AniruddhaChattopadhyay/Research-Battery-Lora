"""Draw the dual-view graph of one run (the data that the graph agent's tools query).

Usage: uv run python analysis/plot_graph.py 10475
Reads results/graph_data.json (made by src/graph_export.py); writes results/graph_run<id>.png.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COLORS = {"gray": "#8a8a8a", "red": "#d03b3b", "blue": "#2a78d6", "green": "#1baf7a", "brown": "#8b5a2b",
          "purple": "#7a4fc9", "cyan": "#17a9b8", "yellow": "#e0a800"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main(sid):
    data = json.load(open(os.path.join(ROOT, "results", "graph_data.json")))
    run = next(r for r in data["runs"] if r["sid"] == sid)
    objs = run["objects"]
    lane = {o["id"]: len(objs) - 1 - k for k, o in enumerate(objs)}
    last = run["end"] + run["horizon"]
    fig, ax = plt.subplots(figsize=(13, 1.3 + 0.8 * len(objs)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(run["end"] + 0.5, last, color="#efeeea", zorder=0)
    ax.text(run["end"] + 3, len(objs) - 0.35, "after the video:\nworld simulation", fontsize=9, color=INK2, va="top")
    # data view: state segments on each object lane
    for s in run["segments"]:
        y, c = lane[s["obj"]], COLORS[objs[s["obj"]]["color"]]
        if not s["in_view"]:
            ax.plot([s["start"], s["end"] + 1], [y, y], ls=":", lw=1.2, color="#b8b7b2", zorder=1)
        elif s["state"] == "moving":
            ax.plot([s["start"], s["end"] + 1], [y, y], lw=7, color=c, solid_capstyle="butt", zorder=2)
        else:
            ax.plot([s["start"], s["end"] + 1], [y, y], lw=2.2, color=c, alpha=0.55, zorder=2)
    # process view: causal edges (previous event of an object -> its next event)
    ev = {e["id"]: e for e in run["events"]}
    for ed in run["causal_edges"]:
        a, b, y = ev[ed["from"]], ev[ed["to"]], lane[ed["obj"]]
        ax.add_patch(FancyArrowPatch((a["frame"], y + 0.12), (b["frame"], y + 0.12), connectionstyle="arc3,rad=-0.25",
                                     arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK2, zorder=3))
    # process view: event nodes
    for e in run["events"]:
        ys = [lane[o] for o in e["objects"]]
        if e["kind"] == "collision":
            ax.plot([e["frame"]] * 2, [min(ys), max(ys)], color=INK, lw=1.2, zorder=4)
            ax.scatter([e["frame"]] * 2, ys, s=70, color=INK, zorder=5)
            ax.text(e["frame"] + 1.2, max(ys) + 0.2, "collision\nframe %d" % e["frame"], fontsize=8.5, color=INK, va="bottom")
        else:
            ax.scatter([e["frame"]], ys, s=80, marker=">" if e["kind"] == "enter" else "s", color=INK, zorder=5)
            ax.text(e["frame"], ys[0] - 0.3, "%s %d" % (e["kind"], e["frame"]), fontsize=8.5, color=INK, ha="center", va="top")
    # operator output: simulated future collisions
    for e in run["future"]:
        ys = [lane[o] for o in e["objects"]]
        ax.plot([e["frame"]] * 2, [min(ys), max(ys)], color=INK, lw=1.2, ls="--", zorder=4)
        ax.scatter([e["frame"]] * 2, ys, s=70, facecolor=SURFACE, edgecolor=INK, linestyle="--", zorder=5)
        ax.text(e["frame"] + 1.2, max(ys) + 0.2, "simulated\nframe %d" % e["frame"], fontsize=8.5, color=INK, va="bottom")
    ax.set_yticks([lane[o["id"]] for o in objs])
    ax.set_yticklabels(["%d  %s" % (o["id"], o["name"]) for o in objs], fontsize=10.5)
    for tick, o in zip(ax.get_yticklabels(), objs):
        tick.set_color(COLORS[o["color"]] if o["color"] not in ("yellow", "gray") else INK)
    ax.set_xlim(-2, last)
    ax.set_ylim(-0.8, len(objs) - 0.2)
    ax.set_xlabel("frame", color=INK2)
    ax.tick_params(colors=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Run %d: the graph behind the graph agent's tools" % sid, loc="left", fontsize=13, fontweight="bold", color=INK)
    handles = [Line2D([], [], lw=7, color=INK2, label="moving, in view (data view)"),
               Line2D([], [], lw=2.2, color=INK2, alpha=0.55, label="stationary, in view"),
               Line2D([], [], lw=1.2, ls=":", color="#b8b7b2", label="out of view"),
               Line2D([], [], marker="o", ls="", color=INK, label="event (process view)"),
               Line2D([], [], marker=r"$\curvearrowright$", ls="", markersize=14, color=INK2, label="causal edge"),
               Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK, label="simulated event (operator)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=6, frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "graph_run%d.png" % sid)
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main(int(sys.argv[1]))
