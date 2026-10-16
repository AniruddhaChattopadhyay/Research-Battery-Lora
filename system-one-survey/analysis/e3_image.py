"""E3, exploratory (added on 2026-10-16): the same input as text and as an image.

  uv run python analysis/e3_image.py --tag image-pilot
  uv run python analysis/e3_image.py --tag e3-image-s2 --ref b1

src/run_image.py sends each item twice in the same run: as text (<id>#text) and as an
image of the same text (<id>#image). For each system and task:
  flip     the top answer of the image request differs from the top answer of the text request
  drift    the text request of this run against the original request of E2 (or B1 for the
           System Two models; the same text, sent earlier). This shows the noise and the
           version changes over the gap.
  dist     predicate: mean |p - p'|; choice: mean total variation distance
  acc      accuracy of the text and of the image requests
  sce      signed calibration error (mean confidence minus accuracy) of each
  list     the text as a one-element list (<id>#list) against the text request: a control
           for the list form of the image state (only if the run has these requests)
Also the pooled flip rate over all answers of a system (the decision rule of the pilot).
Writes results/<tag>.csv (one row for each system and task) and results/<tag>_summary.csv
(one row for each system: macro means over the tasks, a 95% bootstrap interval of the flip
rate with the items resampled within each task, and the noise floor of E3 on the same tasks).
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

import datetime

from common import ROOT, SYSTEM_LABEL, answers, file_name, records, top


def distance(a, b):
    if "true" in a and "true" in b:
        return abs(a["true"] - b["true"])
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in a.keys() | b.keys())


def bootstrap(by_item, n=500, seed=0):
    """95% interval of the macro mean over tasks. by_item: task -> base_id -> list of flips."""
    rng = np.random.default_rng(seed)
    groups = [list(items.values()) for items in by_item.values()]
    stats = []
    for _ in range(n):
        means = []
        for g in groups:
            pick = rng.integers(0, len(g), len(g))
            means.append(np.mean([x for i in pick for x in g[i]]))
        stats.append(np.mean(means))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def floors():
    """(system, task) -> the mean flip rate of the 4 identical repeats of E3."""
    path = ROOT / "results" / "e3_rows.csv"
    out = defaultdict(list)
    if path.exists():
        for r in csv.DictReader(open(path)):
            if r["family"] == "repeat":
                out[(r["system"], r["task"])].append(float(r["flip"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="image-pilot")
    p.add_argument("--ref", default="e2", help="the run with the original requests (e2, or b1 for System Two)")
    a = p.parse_args()
    models = [line.strip() for line in open(ROOT / "runs" / a.tag / "models.txt") if line.strip()]
    rows, summary = [], []
    floor = floors()
    for model in models:
        run = answers(ROOT / "runs" / a.tag / file_name(model))
        e2 = answers(ROOT / "runs" / a.ref / file_name(model))
        pairs = defaultdict(dict)
        for (task, item_id, qname), x in run.items():
            m = x["meta"] or {}
            if x["dist"] is not None:
                pairs[(task, m["base_id"], qname)][m["variant"]] = x
        by_task = defaultdict(lambda: defaultdict(list))
        by_item = defaultdict(lambda: defaultdict(list))
        for (task, base_id, qname), v in pairs.items():
            if "text" not in v or "image" not in v:
                continue
            t, i = v["text"], v["image"]
            c = by_task[task]
            c["flip"].append(top(t["dist"]) != top(i["dist"]))
            by_item[task][base_id].append(top(t["dist"]) != top(i["dist"]))
            c["dist"].append(distance(t["dist"], i["dist"]))
            c["acc_text"].append(top(t["dist"]) == t["gold"])
            c["acc_image"].append(top(i["dist"]) == i["gold"])
            c["conf_text"].append(max(t["dist"].values()))
            c["conf_image"].append(max(i["dist"].values()))
            o = e2.get((task, base_id, qname))
            if o is not None and o["dist"] is not None:
                c["drift"].append(top(t["dist"]) != top(o["dist"]))
            if "list" in v:
                c["flip_list"].append(top(t["dist"]) != top(v["list"]["dist"]))
        for task, c in by_task.items():
            rows.append({
                "system": model, "label": SYSTEM_LABEL.get(model, model), "task": task, "n": len(c["flip"]),
                "flip": float(np.mean(c["flip"])), "drift": float(np.mean(c["drift"])) if c["drift"] else "",
                "dist": float(np.mean(c["dist"])),
                "acc_text": float(np.mean(c["acc_text"])), "acc_image": float(np.mean(c["acc_image"])),
                "sce_text": float(np.mean(c["conf_text"]) - np.mean(c["acc_text"])),
                "sce_image": float(np.mean(c["conf_image"]) - np.mean(c["acc_image"])),
                "flips": int(np.sum(c["flip"])),
                "flip_list": float(np.mean(c["flip_list"])) if c["flip_list"] else "",
                "n_list": len(c["flip_list"]),
            })
        mine = [r for r in rows if r["system"] == model]
        # the time between the original request and the text request of this run (median, hours)
        times = {"run": [], "ref": []}
        ref_recs = records(ROOT / "runs" / a.ref / file_name(model))
        for (task, item_id), r in records(ROOT / "runs" / a.tag / file_name(model)).items():
            if r["meta"]["variant"] == "text" and (task, r["meta"]["base_id"]) in ref_recs:
                times["run"].append(datetime.datetime.fromisoformat(r["time_utc"]).timestamp())
                times["ref"].append(datetime.datetime.fromisoformat(ref_recs[(task, r["meta"]["base_id"])]["time_utc"]).timestamp())
        gap = (np.median(times["run"]) - np.median(times["ref"])) / 3600 if times["run"] else float("nan")
        if mine:
            lo, hi = bootstrap(by_item)
            fl = [floor[(model, r["task"])] for r in mine if (model, r["task"]) in floor]
            summary.append({
                "system": model, "label": SYSTEM_LABEL.get(model, model), "tasks": len(mine),
                "n": sum(r["n"] for r in mine), "flip": float(np.mean([r["flip"] for r in mine])), "lo": lo, "hi": hi,
                "floor": float(np.mean(fl)) if len(fl) == len(mine) else "",
                "drift": float(np.mean([r["drift"] for r in mine if r["drift"] != ""])) if any(r["drift"] != "" for r in mine) else "",
                "dacc": float(np.mean([r["acc_image"] - r["acc_text"] for r in mine])),
                "acc_text": float(np.mean([r["acc_text"] for r in mine])),
                "acc_image": float(np.mean([r["acc_image"] for r in mine])),
                "sce_text": float(np.mean([r["sce_text"] for r in mine])),
                "sce_image": float(np.mean([r["sce_image"] for r in mine])),
                "gap_hours": float(gap),
                "flip_list": float(np.mean([r["flip_list"] for r in mine])) if all(r["flip_list"] != "" for r in mine) else "",
            })
    with open(ROOT / "results" / f"{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "results" / f"{a.tag}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    tasks = sorted({r["task"] for r in rows})
    print(f"{a.tag}: text against image of the same input, same run. Values in %.")
    print("| System | Task | n | Flip | Drift (text vs E2) | Acc. text | Acc. image | SCE text | SCE image |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        drift = f"{100 * r['drift']:.1f}" if r["drift"] != "" else "-"
        print(f"| {r['label']} | {r['task']} | {r['n']} | {100 * r['flip']:.1f} | {drift} | {100 * r['acc_text']:.1f} | "
              f"{100 * r['acc_image']:.1f} | {100 * r['sce_text']:+.1f} | {100 * r['sce_image']:+.1f} |")
    print("\nPer system: pooled flip over all answers; macro flip; accuracy change (macro, points)")
    big_flip = big_acc = 0
    for model in models:
        rs = [r for r in rows if r["system"] == model]
        if not rs:
            continue
        pooled = sum(r["flips"] for r in rs) / sum(r["n"] for r in rs)
        macro = np.mean([r["flip"] for r in rs])
        dacc = np.mean([r["acc_image"] - r["acc_text"] for r in rs])
        big_flip += pooled > 0.05
        big_acc += abs(dacc) > 0.05
        print(f"  {SYSTEM_LABEL.get(model, model):26s} pooled {100 * pooled:5.1f}  macro {100 * macro:5.1f}  "
              f"acc change {100 * dacc:+5.1f}")
    print(f"\nSystems with pooled flip > 5%: {big_flip}; with |accuracy change| > 5 points: {big_acc}")
    print("Decision rule: go if at least 3 systems in either count.")
    print("\nMacro flip [95% interval] against the E3 floor of the same tasks")
    for r in summary:
        floor_text = f"{100 * r['floor']:.1f}" if r["floor"] != "" else "-"
        print(f"  {r['label']:26s} {100 * r['flip']:5.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]  floor {floor_text}")


if __name__ == "__main__":
    main()
