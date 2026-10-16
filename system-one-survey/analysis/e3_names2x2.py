"""E3, exploratory (added on 2026-10-16): the name test with both positions of the true rubric.

  uv run python analysis/e3_names2x2.py

Each predicate becomes a choice between two options. The rubric texts are "This is true for
the input." (true rubric) and "This is not true for the input." The four variants:

  namealign         yes: true rubric (first)    no: not true (second)
  nameswap          no: true rubric (first)     yes: not true (second)
  namealign-second  no: not true (first)        yes: true rubric (second)
  nameswap-second   yes: not true (first)       no: true rubric (second)

A flip is a change of the chosen rubric between two variants. Four comparisons:
  names, rubric first    namealign against nameswap (the "Names" column of Table 5)
  names, rubric second   namealign-second against nameswap-second
  position, names match  namealign against namealign-second
  position, names swap   nameswap against nameswap-second
Values are macro means over BoolQ, Civil Comments and Aegis 2.0, with 95% bootstrap
intervals (items resampled within each task, all questions of an item together).
For the two position comparisons, a flip means that the system chose the option in the
same position both times. "first" and "second" count these flips (pooled over all answers),
and an exact two-sided sign test compares them.
Writes results/e3_names2x2.csv.
"""
import csv
import math
from collections import defaultdict

import numpy as np

from common import ROOT, SYSTEM_LABEL, answers, file_name, top

VARIANTS = ("namealign", "nameswap", "namealign-second", "nameswap-second")
PAIRS = {
    "names_first": ("namealign", "nameswap"),
    "names_second": ("namealign-second", "nameswap-second"),
    "position_match": ("namealign", "namealign-second"),
    "position_swap": ("nameswap", "nameswap-second"),
}


FIRST_NAME = {"namealign": "yes", "nameswap": "no", "namealign-second": "no", "nameswap-second": "yes"}
FIRST_RUBRIC = {"namealign": "true", "nameswap": "true", "namealign-second": "false", "nameswap-second": "false"}


def chosen(variant, dist):
    """The chosen rubric: "true" or "false"."""
    p_true = dist.get("yes", 0.0) if variant.startswith("namealign") else dist.get("no", 0.0)
    p_false = dist.get("no", 0.0) if variant.startswith("namealign") else dist.get("yes", 0.0)
    return top({"true": p_true, "false": p_false})


def sign_test(a, b):
    """Exact two-sided p-value of a against b under p = 0.5."""
    n, k = a + b, min(a, b)
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def load(model):
    """(task, base_id, qname) -> variant -> chosen rubric."""
    out = defaultdict(dict)
    for tag in ("e3-names", "e3-names-second"):
        for (task, item_id, qname), a in answers(ROOT / "runs" / tag / file_name(model)).items():
            meta = a["meta"] or {}
            if meta.get("variant") in VARIANTS and a["dist"]:
                out[(task, meta["base_id"], qname)][meta["variant"]] = chosen(meta["variant"], a["dist"])
    return out


def macro_flip(by_task, n_boot=500, seed=0):
    """Macro mean over tasks and a 95% interval. by_task: task -> base_id -> list of flips."""
    if not by_task:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([np.mean([x for v in items.values() for x in v]) for items in by_task.values()]))
    rng = np.random.default_rng(seed)
    groups = [list(items.values()) for items in by_task.values()]
    stats = []
    for _ in range(n_boot):
        means = []
        for g in groups:
            pick = rng.integers(0, len(g), len(g))
            means.append(np.mean([x for i in pick for x in g[i]]))
        stats.append(np.mean(means))
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    models = [line.strip() for line in open(ROOT / "runs" / "e3-names-second" / "models.txt") if line.strip()]
    floors = {(r["system"], r["family"]): r for r in csv.DictReader(open(ROOT / "results" / "e3_summary.csv"))}
    rows = []
    for model in models:
        data = load(model)
        for pair, (a, b) in PAIRS.items():
            by_task = defaultdict(lambda: defaultdict(list))
            for (task, base_id, qname), v in data.items():
                if a in v and b in v:
                    by_task[task][base_id].append(v[a] != v[b])
            point, lo, hi = macro_flip(by_task)
            floor = floors.get((model, "namealign"), {}).get("floor", "")
            # position flips: the chosen option was first both times, or second both times
            first = second = 0
            if pair.startswith("position"):
                for v in data.values():
                    if a in v and b in v and v[a] != v[b]:
                        if v[a] == FIRST_RUBRIC[a]:
                            first += 1
                        else:
                            second += 1
            rows.append({"system": model, "label": SYSTEM_LABEL.get(model, model), "pair": pair,
                         "n": sum(len(x) for items in by_task.values() for x in items.values()),
                         "tasks": len(by_task), "flip": point, "lo": lo, "hi": hi, "floor": floor,
                         "first": first, "second": second,
                         "p_sign": sign_test(first, second) if pair.startswith("position") else ""})
    with open(ROOT / "results" / "e3_names2x2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {(r["system"], r["pair"]): r for r in rows}
    print("Flip rate (%) of the chosen rubric. Macro mean over BoolQ, Civil Comments and Aegis 2.0.")
    print("| System | Names, true rubric first | Names, true rubric second | Position, names match | "
          "Position, names swapped | Floor |")
    print("|---|---:|---:|---:|---:|---:|")
    for model in models:
        cells = []
        for pair in PAIRS:
            r = by.get((model, pair))
            cells.append(f"{100 * r['flip']:.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]" if r and r["n"] else "-")
        floor = by[(model, "names_first")]["floor"]
        print(f"| {SYSTEM_LABEL.get(model, model)} | " + " | ".join(cells)
              + f" | {100 * float(floor):.1f} |" if floor != "" else " | - |")


if __name__ == "__main__":
    main()
