"""E2: accuracy and calibration of all systems on all tasks (H1).

  uv run python analysis/e2_calibration.py

Writes results/e2_metrics.csv (one row for each system and task) and
results/e2_groups.csv (signed calibration error for each task group, with a
bootstrap interval). Prints markdown tables.
"""
import csv
from collections import defaultdict

import numpy as np

from common import (
    ROOT, SYSTEM_LABEL, SYSTEM_TWO, TASK_GROUP, TASK_ORDER, auroc, aurc, bootstrap, brier,
    classwise_ece, confidences, ece, load_system, top,
)

E2_MODELS = [line.strip() for line in open(ROOT / "runs" / "e2" / "models.txt") if line.strip()]
B2_MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.8-27b"]
# (key, model, run tags, label). The key is unique: one model can run with two methods.
SYSTEMS = (
    [(m, m, ("e2",), SYSTEM_LABEL[m]) for m in E2_MODELS]
    + [(m, m, ("b1",), SYSTEM_LABEL[m]) for m in SYSTEM_TWO]
    + [("openai/gpt-6-luna@none", "openai/gpt-6-luna", ("b3-none",), "GPT-6 Luna (chat, no reasoning)")]
    + [("local/qwen3.5-4b", "local/qwen3.5-4b", ("b2-local",), SYSTEM_LABEL["local/qwen3.5-4b"])]
    + [(m, m, ("b2",), SYSTEM_LABEL[m]) for m in B2_MODELS]
)


def task_metrics(rows):
    conf, correct = confidences(rows)
    out = {
        "n": len(rows),
        "acc": correct.mean(),
        "conf": conf.mean(),
        "sce": conf.mean() - correct.mean(),
        "ece": ece(conf, correct),
        "ece_mass": ece(conf, correct, mass=True),
        "cw_ece": classwise_ece(rows),
        "brier": brier(rows),
        "aurc": aurc(conf, correct),
    }
    if rows[0]["type"] == "noul":
        p = np.array([r["dist"]["true"] for r in rows])
        y = np.array([r["gold"] == "true" for r in rows])
        out["auroc"] = auroc(p, y)
        out["pos_rate"] = y.mean()
        out["mean_p_true"] = p.mean()
        soft = np.array([r["soft"] for r in rows])
        out["soft_brier"] = float(np.mean((p - soft) ** 2))
    if rows[0]["type"] == "score":
        exp = np.array([sum(int(k) * v for k, v in r["dist"].items()) for r in rows])
        gold = np.array([int(r["gold"]) for r in rows])
        out["mae"] = float(np.abs(exp - gold).mean())
        out["within1"] = float(np.mean([abs(int(top(r["dist"])) - int(r["gold"])) <= 1 for r in rows]))
    return out


def usable(ans):
    return {k: a for k, a in ans.items() if a["dist"] is not None and a["gold"] is not None and "#" not in k[1]}


def main():
    out_rows, group_rows = [], []
    for key, model, tags, label in SYSTEMS:
        ans = usable(load_system(model, tags))
        if not ans:
            continue
        by_task = defaultdict(list)
        for (task, item_id, qname), a in ans.items():
            by_task[task].append(a)
        for task in TASK_ORDER:
            rows = by_task.get(task)
            if not rows:
                continue
            m = task_metrics(rows)
            m.update(system=key, label=label, task=task, group=TASK_GROUP[task],
                     resolved=next((a["resolved"] for a in rows if a["resolved"]), ""))
            out_rows.append(m)
        # Signed calibration error for each task group: the mean over the tasks of the group
        # (each task counts once). The interval resamples the items within each task.
        rng = np.random.default_rng(0)
        items_by_task = defaultdict(lambda: defaultdict(list))
        for (task, item_id, qname), a in ans.items():
            items_by_task[task][item_id].append(a)

        def sce(rows):
            c, k = confidences(rows)
            return c.mean() - k.mean()

        for group in ["knowledge", "classification", "agent", "judging", "all"]:
            tasks_in = [t for t in TASK_ORDER if t in items_by_task and (group == "all" or TASK_GROUP[t] == group)]
            if not tasks_in:
                continue
            per_task = {t: list(items_by_task[t].values()) for t in tasks_in}
            point = np.mean([sce([r for v in per_task[t] for r in v]) for t in tasks_in])
            stats = []
            for _ in range(500):
                vals = []
                for t in tasks_in:
                    items = per_task[t]
                    pick = rng.integers(0, len(items), len(items))
                    vals.append(sce([r for i in pick for r in items[i]]))
                stats.append(np.mean(vals))
            pooled = sce([r for t in tasks_in for v in per_task[t] for r in v])
            group_rows.append({"system": key, "label": label, "group": group, "tasks": len(tasks_in),
                               "n": sum(len(r) for t in tasks_in for r in per_task[t]), "sce": point,
                               "lo": float(np.percentile(stats, 2.5)), "hi": float(np.percentile(stats, 97.5)),
                               "sce_pooled": pooled})
        print(f"done {key}")

    (ROOT / "results").mkdir(exist_ok=True)
    keys = ["system", "label", "task", "group", "resolved", "n", "acc", "conf", "sce", "ece", "ece_mass", "cw_ece",
            "brier", "aurc", "auroc", "pos_rate", "mean_p_true", "soft_brier", "mae", "within1"]
    with open(ROOT / "results" / "e2_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    with open(ROOT / "results" / "e2_groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["system", "label", "group", "tasks", "n", "sce", "lo", "hi", "sce_pooled"])
        w.writeheader()
        w.writerows(group_rows)

    # Majority baselines (B7): for each question, the most frequent gold answer of the 500 items.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import tasks as task_defs
    base_rows = []
    for t in TASK_ORDER:
        items = task_defs.load(t, 500)
        right = total = 0
        for qname, q in items[0]["questions"].items():
            golds = []
            for it in items:
                g = it["gold"][qname]
                golds.append(("true" if g >= 0.5 else "false") if isinstance(g, float) else
                             ("true" if g is True else "false") if isinstance(g, bool) else str(g))
            top_gold = max(set(golds), key=golds.count)
            right += golds.count(top_gold)
            total += len(golds)
            if t == "civil":
                base_rows.append({"task": f"civil:{qname}", "majority_acc": golds.count(top_gold) / len(golds),
                                  "positive_rate": golds.count("true") / len(golds)})
        base_rows.append({"task": t, "majority_acc": right / total, "positive_rate": ""})
    with open(ROOT / "results" / "e2_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task", "majority_acc", "positive_rate"])
        w.writeheader()
        w.writerows(base_rows)

    # Sensitivity: Civil Comments without severe_toxicity (no positive item at the 0.5 rule)
    sens = []
    for key, model, tags, label in SYSTEMS:
        ans = usable(load_system(model, tags))
        rows_all = [a for (t, i, q), a in ans.items() if t == "civil"]
        rows_wo = [a for (t, i, q), a in ans.items() if t == "civil" and q != "severe_toxicity"]
        if rows_all:
            m_all, m_wo = task_metrics(rows_all), task_metrics(rows_wo)
            sens.append({"system": key, "label": label, "acc": m_all["acc"], "acc_wo": m_wo["acc"],
                         "sce": m_all["sce"], "sce_wo": m_wo["sce"], "ece": m_all["ece"], "ece_wo": m_wo["ece"],
                         "cw_ece": m_all["cw_ece"], "cw_ece_wo": m_wo["cw_ece"]})
    with open(ROOT / "results" / "e2_civil_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(sens[0]))
        w.writeheader()
        w.writerows(sens)

    # markdown: accuracy for each task
    acc = {(r["system"], r["task"]): r for r in out_rows}
    print("\n| System | " + " | ".join(TASK_ORDER) + " | mean |")
    print("|---|" + "---:|" * (len(TASK_ORDER) + 1))
    for key, _, _, label in SYSTEMS:
        vals = [acc.get((key, t)) for t in TASK_ORDER]
        cells = [f"{100 * v['acc']:.1f}" if v else "-" for v in vals]
        mean = np.mean([v["acc"] for v in vals if v]) if all(vals) else float("nan")
        print(f"| {label} | " + " | ".join(cells) + f" | {100 * mean:.1f} |")
    print("\nSigned calibration error (confidence - accuracy) by group, 95% bootstrap interval")
    groups = ["knowledge", "classification", "judging", "agent", "all"]
    print("| System | " + " | ".join(groups) + " |")
    print("|---|" + "---:|" * len(groups))
    g = {(r["system"], r["group"]): r for r in group_rows}
    for key, _, _, label in SYSTEMS:
        cells = []
        for grp in groups:
            r = g.get((key, grp))
            cells.append(f"{r['sce']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" if r else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
