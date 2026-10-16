"""E5: what does decision training add to the same backbone? (H4)

  uv run python analysis/e5_training.py

Each pair is a System One model and its backbone on the same items:
  4B      Tev1-4B (Together)                   Qwen3.5-4B, local MLX readout (runs/b2-local)
  9B      Microsoft-Decision-1                 Qwen3.5-9B readout (runs/b2)
  27B     pplx-decider v1.1                    Qwen3.8-27B readout (runs/b2)
  closed  OpenAI Decisions API (gpt-6-luna)    gpt-6-luna with structured outputs, low reasoning (runs/b1)
          and without reasoning (runs/b3-none, exploratory)

BANKING77 is not in the comparison: the letter readout permits 26 options or fewer.
All columns are means over tasks (each task counts once):
  acc, gain   accuracy of the top answer; the interval resamples the items within each task
  ece         top-label ECE with 15 bins
  ece_ts      the same after temperature scaling: one temperature for each system and task,
              fitted by NLL on one half of the items and tested on the other half (2 folds)
Writes results/e5_pairs.csv.
"""
import csv
import zlib
from collections import defaultdict

import numpy as np

from common import ROOT, SYSTEM_LABEL, TASK_ORDER, confidences, ece, load_system, top

# (size, System One model, its runs, backbone model, its runs, backbone key for the tables)
PAIRS = [
    ("4B", "togethercomputer/tev1-4b-experimental", ("e2",), "local/qwen3.5-4b", ("b2-local",), "local/qwen3.5-4b"),
    ("9B", "microsoft/microsoft-decision-1", ("e2",), "qwen/qwen3.5-9b", ("b2",), "qwen/qwen3.5-9b"),
    ("27B", "perplexity/pplx-decider-v1.1-27b", ("e2",), "qwen/qwen3.8-27b", ("b2",), "qwen/qwen3.8-27b"),
    ("closed", "openai/gpt-6-luna-decisions", ("e2",), "openai/gpt-6-luna", ("b1",), "openai/gpt-6-luna"),
    ("closed", "openai/gpt-6-luna-decisions", ("e2",), "openai/gpt-6-luna", ("b3-none",), "openai/gpt-6-luna@none"),
]
TASKS = [t for t in TASK_ORDER if t != "banking77"]
TEMPS = np.exp(np.linspace(np.log(0.1), np.log(20), 80))


def scaled(dist, t):
    p = {k: max(v, 1e-4) ** (1 / t) for k, v in dist.items()}
    z = sum(p.values())
    return {k: v / z for k, v in p.items()}


def nll(rows, t):
    return -np.mean([np.log(scaled(r["dist"], t).get(r["gold"], 1e-4)) for r in rows])


def ece_ts(rows, item_ids):
    """Cross-fitted ECE after temperature scaling (2 folds by a hash of the item id)."""
    folds = [[r for r, i in zip(rows, item_ids) if zlib.crc32(i.encode()) % 2 == h] for h in (0, 1)]
    out = []
    for fit, test in ((folds[0], folds[1]), (folds[1], folds[0])):
        if not fit or not test:
            continue
        t = min(TEMPS, key=lambda x: nll(fit, x))
        cal = [{"dist": scaled(r["dist"], t), "gold": r["gold"]} for r in test]
        c, k = confidences(cal)
        out.append(ece(c, k))
    return float(np.mean(out)) if out else float("nan")


def main():
    rng = np.random.default_rng(0)
    rows = []
    for size, dec, dec_tags, base, base_tags, base_key in PAIRS:
        a = load_system(dec, dec_tags)
        b = load_system(base, base_tags)
        if not b:
            print(f"no data for {base}")
            continue
        per_task = {}
        for task in TASKS:
            keys = [k for k in a.keys() & b.keys() if k[0] == task and "#" not in k[1]
                    and a[k]["dist"] is not None and b[k]["dist"] is not None and a[k]["gold"] is not None]
            if not keys:
                continue
            ra, rb = [a[k] for k in keys], [b[k] for k in keys]
            ca, ka = confidences(ra)
            cb, kb = confidences(rb)
            per_task[task] = {
                "n": len(keys), "items": len({k[1] for k in keys}),
                "acc_decision": ka.mean(), "acc_backbone": kb.mean(),
                "ece_decision": ece(ca, ka), "ece_backbone": ece(cb, kb),
                "ece_ts_decision": ece_ts(ra, [k[1] for k in keys]), "ece_ts_backbone": ece_ts(rb, [k[1] for k in keys]),
                "by_item": defaultdict(lambda: [[], []]),
            }
            for k, x, y in zip(keys, ka, kb):
                per_task[task]["by_item"][k[1]][0].append(x)
                per_task[task]["by_item"][k[1]][1].append(y)
        if not per_task:
            continue
        # bootstrap of the macro gain: resample the items within each task
        stats = []
        item_lists = {t: list(v["by_item"].values()) for t, v in per_task.items()}
        for _ in range(1000):
            gains = []
            for t, items in item_lists.items():
                pick = rng.integers(0, len(items), len(items))
                da = [x for i in pick for x in items[i][0]]
                db = [x for i in pick for x in items[i][1]]
                gains.append(np.mean(da) - np.mean(db))
            stats.append(np.mean(gains))
        for task, m in list(per_task.items()) + [("all", None)]:
            if m is None:
                vals = per_task.values()
                m = {k: float(np.mean([v[k] for v in vals])) for k in
                     ("acc_decision", "acc_backbone", "ece_decision", "ece_backbone", "ece_ts_decision", "ece_ts_backbone")}
                m.update(n=sum(v["n"] for v in vals), items=sum(v["items"] for v in vals),
                         gain_lo=float(np.percentile(stats, 2.5)), gain_hi=float(np.percentile(stats, 97.5)),
                         tasks=len(per_task))
            row = {"size": size, "decision": dec, "backbone": base_key, "task": task}
            row.update({k: v for k, v in m.items() if k != "by_item"})
            row["gain"] = row["acc_decision"] - row["acc_backbone"]
            rows.append(row)
    keys = ["size", "decision", "backbone", "task", "tasks", "n", "items", "acc_decision", "acc_backbone", "gain",
            "gain_lo", "gain_hi", "ece_decision", "ece_backbone", "ece_ts_decision", "ece_ts_backbone"]
    with open(ROOT / "results" / "e5_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("| Size | Decision model | Backbone | Task | n | Acc dec | Acc base | Gain [95% CI] | ECE dec | ECE base | ECE-TS dec | ECE-TS base |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ci = f" [{100 * r['gain_lo']:+.1f}, {100 * r['gain_hi']:+.1f}]" if "gain_lo" in r else ""
        print(f"| {r['size']} | {SYSTEM_LABEL.get(r['decision'])} | {r['backbone']} | {r['task']} | {r['n']} | "
              f"{100 * r['acc_decision']:.1f} | {100 * r['acc_backbone']:.1f} | {100 * r['gain']:+.1f}{ci} | "
              f"{100 * r['ece_decision']:.1f} | {100 * r['ece_backbone']:.1f} | {100 * r['ece_ts_decision']:.1f} | {100 * r['ece_ts_backbone']:.1f} |")


if __name__ == "__main__":
    main()
