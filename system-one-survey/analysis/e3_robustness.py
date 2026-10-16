"""E3: how much do perturbations change the answers? (H2, H6)

  uv run python analysis/e3_robustness.py

For each system, task and variant, compare the answer to the perturbed request (runs/e3)
with the answer to the original request (runs/e2, the same item). The answer of the
perturbed request is first mapped back to the original options (see src/perturb.py).

  flip      the top answer changed
  dist      predicate: |p - p'|; choice and score: total variation distance
  follow    hint and inject: the top answer is the wrong answer in the input text,
            among the items where the original top answer was not that answer
  broken    the share of originally correct answers that became wrong

The noise floor is the mean flip rate of the 4 repeated identical calls (repeat1..4).
Writes results/e3_rows.csv and results/e3_summary.csv.
"""
import csv
from collections import defaultdict

import numpy as np

from common import ROOT, SYSTEM_LABEL, TASK_ORDER, answers, file_name, top

FAMILIES = ["repeat", "para", "negate", "order", "codes", "reverse", "swap", "crowded", "alone", "hint", "inject",
            "namealign", "nameswap"]


def family(variant):
    for f in FAMILIES:
        if variant.startswith(f):
            return f
    raise ValueError(variant)


def mapped(meta, qtype, dist):
    """The answer of a perturbed request, in the options of the original request."""
    v = family(meta["variant"])
    if v == "negate":
        return {"true": dist["false"], "false": dist["true"]}
    if v == "codes":
        return {meta["map"][k]: p for k, p in dist.items() if k in meta["map"]}
    if v == "reverse":
        return {str(4 - int(k)): p for k, p in dist.items()}
    if v == "swap":
        return {"A": dist.get("B", 0.0), "B": dist.get("A", 0.0)}
    if v == "namealign":  # the option "yes" carries the true rubric
        return {"true": dist.get("yes", 0.0), "false": dist.get("no", 0.0)}
    if v == "nameswap":  # the option "no" carries the true rubric
        return {"true": dist.get("no", 0.0), "false": dist.get("yes", 0.0)}
    return dist


def distance(qtype, a, b):
    if "true" in a and "true" in b:
        return abs(a["true"] - b["true"])
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in a.keys() | b.keys())


def compare(model):
    orig = answers(ROOT / "runs" / "e2" / file_name(model))
    pert = answers(ROOT / "runs" / "e3" / file_name(model))
    pert.update(answers(ROOT / "runs" / "e3-names" / file_name(model)))
    # The reference of "nameswap" is "namealign" (as in Rafe and Das), not the E2 request.
    aligned = {}
    for (task, item_id, qname), a in pert.items():
        if a["meta"] and a["meta"]["variant"] == "namealign" and a["dist"]:
            aligned[(task, a["meta"]["base_id"], qname)] = {"dist": mapped(a["meta"], a["type"], a["dist"]), "gold": a["gold"]}
    rows = defaultdict(lambda: defaultdict(list))
    by_item = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (task, item_id, qname), a in pert.items():
        meta = a["meta"]
        if a["dist"] is None or meta is None:
            continue
        target = meta.get("target")
        if target is not None and qname != target:
            continue  # a distractor question
        if meta["variant"] == "nameswap":
            o = aligned.get((task, meta["base_id"], qname))
        else:
            o = orig.get((task, meta["base_id"], qname))
        if o is None or o["dist"] is None:
            continue
        d = mapped(meta, a["type"], a["dist"])
        if not d:
            continue
        r = rows[(task, meta["variant"])]
        by_item[(task, family(meta["variant"]))][meta["base_id"]]["flip"].append(top(d) != top(o["dist"]))
        if "wrong" in meta and top(o["dist"]) != meta["wrong"]:
            by_item[(task, family(meta["variant"]))][meta["base_id"]]["follow"].append(top(d) == meta["wrong"])
        r["flip"].append(top(d) != top(o["dist"]))
        r["dist"].append(distance(a["type"], d, o["dist"]))
        r["acc_orig"].append(top(o["dist"]) == o["gold"])
        r["acc_var"].append(top(d) == o["gold"])
        if top(o["dist"]) == o["gold"]:
            r["broken"].append(top(d) != o["gold"])
        if "wrong" in meta and top(o["dist"]) != meta["wrong"]:
            r["follow"].append(top(d) == meta["wrong"])
    return rows, by_item


def bootstrap_ci(by_item, fam, key, n=500, seed=0):
    """95% interval of the macro mean over tasks; items are resampled within each task."""
    rng = np.random.default_rng(seed)
    tasks = [t for (t, f) in by_item if f == fam]
    per_task = []
    for t in tasks:
        items = [v[key] for v in by_item[(t, fam)].values() if v[key]]
        if items:
            per_task.append(items)
    if not per_task:
        return None
    stats = []
    for _ in range(n):
        means = []
        for items in per_task:
            pick = rng.integers(0, len(items), len(items))
            values = [x for i in pick for x in items[i]]
            means.append(np.mean(values))
        stats.append(np.mean(means))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    models = [line.strip() for line in open(ROOT / "runs" / "e3" / "models.txt") if line.strip()]
    out, summary = [], []
    ci_rows = []
    for model in models:
        rows, by_item = compare(model)
        if not rows:
            continue
        for fam in FAMILIES:
            for key in (("flip", "follow") if fam in ("hint", "inject") else ("flip",)):
                ci = bootstrap_ci(by_item, fam, key)
                if ci:
                    ci_rows.append({"system": model, "label": SYSTEM_LABEL.get(model, model), "family": fam,
                                    "metric": key, "lo": ci[0], "hi": ci[1]})
        # one row for each task and variant
        per_task_family = defaultdict(lambda: defaultdict(list))
        for (task, variant), r in rows.items():
            row = {"system": model, "label": SYSTEM_LABEL.get(model, model), "task": task, "variant": variant,
                   "family": family(variant), "n": len(r["flip"])}
            for k in ("flip", "dist", "acc_orig", "acc_var", "broken", "follow"):
                row[k] = float(np.mean(r[k])) if r[k] else ""
            out.append(row)
            for k in ("flip", "dist", "broken", "follow", "acc_orig", "acc_var"):
                per_task_family[(task, row["family"])][k].extend(r[k])
        # summary: macro mean over tasks for each family, and the noise floor of the same tasks
        for fam in FAMILIES:
            tasks_with = [t for t in TASK_ORDER if (t, fam) in per_task_family]
            if not tasks_with:
                continue
            s = {"system": model, "label": SYSTEM_LABEL.get(model, model), "family": fam, "tasks": len(tasks_with)}
            for k in ("flip", "dist", "broken", "follow"):
                vals = [np.mean(per_task_family[(t, fam)][k]) for t in tasks_with if per_task_family[(t, fam)][k]]
                s[k] = float(np.mean(vals)) if vals else ""
            floor = [np.mean(per_task_family[(t, "repeat")]["flip"]) for t in tasks_with if (t, "repeat") in per_task_family]
            s["floor"] = float(np.mean(floor)) if floor else ""
            dacc = [np.mean(per_task_family[(t, fam)]["acc_var"]) - np.mean(per_task_family[(t, fam)]["acc_orig"]) for t in tasks_with]
            s["dacc"] = float(np.mean(dacc))
            summary.append(s)
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "e3_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["system", "label", "task", "variant", "family", "n", "flip", "dist", "acc_orig", "acc_var", "broken", "follow"])
        w.writeheader()
        w.writerows(out)
    with open(ROOT / "results" / "e3_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["system", "label", "family", "metric", "lo", "hi"])
        w.writeheader()
        w.writerows(ci_rows)
    with open(ROOT / "results" / "e3_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["system", "label", "family", "tasks", "flip", "floor", "dist", "broken", "follow", "dacc"])
        w.writeheader()
        w.writerows(summary)

    # markdown: flip rate (%) for each family; the floor is the repeat column
    by = {(s["system"], s["family"]): s for s in summary}
    cols = [f for f in FAMILIES]
    print("Flip rate (%) of the top answer, macro mean over tasks. 'repeat' is the noise floor.")
    print("| System | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for model in models:
        cells = []
        for fam in cols:
            s = by.get((model, fam))
            cells.append(f"{100 * s['flip']:.1f}" if s and s["flip"] != "" else "-")
        print(f"| {SYSTEM_LABEL.get(model, model)} | " + " | ".join(cells) + " |")
    print("\nFollow rate (%) of the wrong answer in the input: hint / inject")
    for model in models:
        h, i = by.get((model, "hint")), by.get((model, "inject"))
        if h and i:
            print(f"| {SYSTEM_LABEL.get(model, model)} | {100 * h['follow']:.1f} | {100 * i['follow']:.1f} | broken: {100 * h['broken']:.1f} / {100 * i['broken']:.1f} |")


if __name__ == "__main__":
    main()
