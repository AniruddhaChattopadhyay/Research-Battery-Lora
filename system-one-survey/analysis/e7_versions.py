"""E7: version changes against the noise floor (H7). No new calls.

  uv run python analysis/e7_versions.py

All comparisons use the same requests (the first 250 items of BoolQ, BANKING77, HelpSteer2 and
Civil Comments for the pilot rows; 200 or 500 items of all tasks for the E2 rows):
  repeats     Jev 1.14 in E2 against the 4 repeats of E3 (the same version, minutes apart)
  pilot-e2    the same version in the E7 pilot (runs/e7-2026-10-15) and in E2
  pilot-later the same version in the E7 pilot and in a later run (runs/e7-later-2026-10-16)
  version     Jev 1.13 against Jev 1.14, in E2, in the pilot and in the later run
The time gap of each comparison is the median gap between the two calls of an item.
Intervals: 1,000 bootstrap resamples of the items (all questions of an item stay together).
Writes results/e7_versions.csv.
"""
import csv
import datetime
import json
from collections import defaultdict

import numpy as np

from common import ROOT, TASK_ORDER, answers, file_name, top

E2 = ROOT / "runs" / "e2"
PILOT = ROOT / "runs" / "e7-2026-10-15"
LATER = sorted((ROOT / "runs").glob("e7-later*-2026-10-16"))  # later runs of the pilot requests
E3 = ROOT / "runs" / "e3"


def times(path):
    """(task, item_id) -> the time of the status-200 call."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] == 200:
            out[(r["task"], r["item_id"])] = datetime.datetime.fromisoformat(r["time_utc"])
    return out


def pairs(a, b, key_map=lambda k: k):
    """Matched answers: a list of (item key, flip, distance, right_a, right_b)."""
    out = []
    for k, x in a.items():
        y = b.get(key_map(k))
        if x["dist"] is None or y is None or y["dist"] is None or x["gold"] is None:
            continue
        if x["type"] == "noul":
            d = abs(x["dist"]["true"] - y["dist"]["true"])
        else:
            d = 0.5 * sum(abs(x["dist"].get(o, 0) - y["dist"].get(o, 0)) for o in x["dist"].keys() | y["dist"].keys())
        out.append(((k[0], k[1].split("#")[0]), k[0], top(x["dist"]) != top(y["dist"]), d,
                    top(x["dist"]) == x["gold"], top(y["dist"]) == x["gold"]))
    return out


def summarize(name, gap_min, rows, out, rng):
    for task in TASK_ORDER + ["all"]:
        sel = [r for r in rows if task == "all" or r[1] == task]
        if not sel:
            continue
        by_item = defaultdict(list)
        for r in sel:
            by_item[r[0]].append(r[2])
        items = list(by_item.values())
        stats = []
        for _ in range(1000):
            pick = rng.integers(0, len(items), len(items))
            vals = [v for i in pick for v in items[i]]
            stats.append(np.mean(vals))
        out.append({"comparison": name, "gap_min": gap_min, "task": task, "n": len(sel),
                    "flip": np.mean([r[2] for r in sel]), "flip_lo": np.percentile(stats, 2.5),
                    "flip_hi": np.percentile(stats, 97.5), "dist": np.mean([r[3] for r in sel]),
                    "acc_a": np.mean([r[4] for r in sel]), "acc_b": np.mean([r[5] for r in sel])})


def median_gap(ta, tb, key_map=lambda k: k):
    gaps = [abs((tb[key_map(k)] - t).total_seconds()) / 60 for k, t in ta.items() if key_map(k) in tb]
    return float(np.median(gaps)) if gaps else float("nan")


def main():
    rng = np.random.default_rng(0)
    rows = []
    j13 = answers(E2 / file_name("typesafe/jev-1.13"))
    j14 = answers(E2 / file_name("typesafe/jev-1.14"))
    t14 = times(E2 / file_name("typesafe/jev-1.14"))

    # same version, the E2 call against the 4 repeats of E3
    rep = answers(E3 / file_name("typesafe/jev-1.14"))
    trep = times(E3 / file_name("typesafe/jev-1.14"))
    rep_rows, gaps = [], []
    for i in range(1, 5):
        sub = {(t, iid.split("#")[0], q): a for (t, iid, q), a in rep.items() if iid.endswith(f"#repeat{i}")}
        rep_rows += pairs({k: v for k, v in j14.items() if k in sub}, sub)
        gaps.append(median_gap({k: v for k, v in t14.items()},
                               {(t, iid.split("#")[0]): v for (t, iid), v in trep.items() if iid.endswith(f"#repeat{i}")}))
    summarize("1.14 repeats", float(np.median(gaps)), rep_rows, rows, rng)

    # same version: pilot against E2, and pilot against the later run
    for model in ("typesafe/jev-1.14", "typesafe/jev-1.13"):
        v = model.split("-")[-1]
        pilot = answers(PILOT / file_name(model))
        tp = times(PILOT / file_name(model))
        e2 = j14 if v == "1.14" else j13
        summarize(f"{v} pilot vs E2", median_gap(tp, times(E2 / file_name(model))), pairs(pilot, e2), rows, rng)
        for k, folder in enumerate(LATER, 1):
            if (folder / file_name(model)).exists():
                later = answers(folder / file_name(model))
                summarize(f"{v} pilot vs later{k}", median_gap(tp, times(folder / file_name(model))), pairs(pilot, later),
                          rows, rng)

    # version change, at three times
    summarize("version (E2)", 0.0, pairs(j13, j14), rows, rng)
    summarize("version (pilot)", 0.0, pairs(answers(PILOT / file_name("typesafe/jev-1.13")),
                                             answers(PILOT / file_name("typesafe/jev-1.14"))), rows, rng)
    for k, folder in enumerate(LATER, 1):
        if (folder / file_name("typesafe/jev-1.13")).exists() and (folder / file_name("typesafe/jev-1.14")).exists():
            summarize(f"version (later{k})", 0.0, pairs(answers(folder / file_name("typesafe/jev-1.13")),
                                                       answers(folder / file_name("typesafe/jev-1.14"))), rows, rng)

    # interval of the ratio: resample the items of the two comparisons independently
    def item_lists(prs):
        by = defaultdict(list)
        for r in prs:
            by[r[0]].append(r[2])
        return list(by.values())
    ver, noi = item_lists(pairs(j13, j14)), item_lists(rep_rows)
    ratios = []
    for _ in range(1000):
        pv = rng.integers(0, len(ver), len(ver))
        pn = rng.integers(0, len(noi), len(noi))
        fv = np.mean([x for i in pv for x in ver[i]])
        fn = np.mean([x for i in pn for x in noi[i]])
        ratios.append(fv / fn)
    with open(ROOT / "results" / "e7_ratio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["ratio", "lo", "hi"])
        w.writeheader()
        flip_v = np.mean([x for v in ver for x in v])
        flip_n = np.mean([x for v in noi for x in v])
        w.writerow({"ratio": flip_v / flip_n, "lo": np.percentile(ratios, 2.5), "hi": np.percentile(ratios, 97.5)})
    with open(ROOT / "results" / "e7_versions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| Comparison | Gap (min) | Task | Questions | Top change [95% CI] | Mean dist | Acc A | Acc B |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['comparison']} | {r['gap_min']:.0f} | {r['task']} | {r['n']} | {100 * r['flip']:.1f}% "
              f"[{100 * r['flip_lo']:.1f}, {100 * r['flip_hi']:.1f}] | {r['dist']:.3f} | {100 * r['acc_a']:.1f} | {100 * r['acc_b']:.1f} |")


if __name__ == "__main__":
    main()
