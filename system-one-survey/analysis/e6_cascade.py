"""E6: cascades from a System One model to a System Two model (H5). No new calls.

  uv run python analysis/e6_cascade.py

Rule: for each item, if the lowest top probability of the System One answers is at or
above a threshold t, keep the System One answers. If not, send the item to the System
Two model and use its answers. The cost is the System One cost for every item plus the
System Two cost for the items that we send.

Threshold choice (2-fold cross-fitting): split the items into two halves by a hash of
the item id. On one half, choose the smallest share of escalated items that reaches the
accuracy of the System Two model on that half. Apply that threshold to the other half.
Then swap the halves. Report the mean of the two test halves.

Writes results/e6_cascade.csv and results/e6_curves.csv.
"""
import csv
import zlib
from collections import defaultdict

import numpy as np

from common import ROOT, SYSTEM_LABEL, TASK_ORDER, auroc, load_system, top

FIRST = [
    "typesafe/jev-1.14",
    "openai/gpt-6-luna-decisions",
    "microsoft/microsoft-decision-1",
    "perplexity/pplx-decider-v1.1-27b",
    "cloudflare/clef",
    "liquid/d1",
    "inception/mercury-decide",
]
SECOND = ["openai/gpt-6-sol", "anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash"]
GRID = np.round(np.concatenate([np.arange(0.0, 0.9, 0.05), np.arange(0.9, 1.0001, 0.01)]), 2)


SKIP = set()  # (task, question) pairs to leave out; used for the Civil Comments check


def items(model, tag):
    """item key -> {"conf": lowest top probability, "correct": {q: ..}, "qconf": {q: ..}, "cost": .., "latency": ..}."""
    out = defaultdict(lambda: {"conf": 1.0, "correct": {}, "qconf": {}, "cost": 0.0, "latency": 0.0})
    for (task, item_id, qname), a in load_system(model, (tag,)).items():
        if a["dist"] is None or a["gold"] is None or "#" in item_id or (task, qname) in SKIP:
            continue
        it = out[(task, item_id)]
        it["conf"] = min(it["conf"], max(a["dist"].values()))
        it["qconf"][qname] = max(a["dist"].values())
        it["correct"][qname] = top(a["dist"]) == a["gold"]
        it["cost"] = a["cost"] or 0.0  # the same call for all questions of the item
        it["latency"] = a["latency_ms"] or 0.0
    return out


def evaluate(keys, s1, s2, t, question_level=False):
    """Accuracy (over questions), cost per 1,000 items, latency and share escalated.

    Item level (the main rule): if the lowest top probability of the item is below t, the
    System Two answers replace all System One answers of the item. Question level (a check
    for multi-question items): only the questions below t take the System Two answer, but the
    System Two call (and its cost) is the same, because one call answers the full card.
    """
    right = total = 0
    cost = lat = esc = 0.0
    for k in keys:
        a, b = s1[k], s2[k]
        send = a["conf"] < t
        for q, ok in a["correct"].items():
            use_b = send and (not question_level or a["qconf"][q] < t) and q in b["correct"]
            right += b["correct"][q] if use_b else ok
            total += 1
        cost += a["cost"] + (b["cost"] if send else 0.0)
        lat += a["latency"] + (b["latency"] if send else 0.0)
        esc += send
    n = len(keys)
    return {"acc": right / total, "cost": 1000 * cost / n, "latency": lat / n, "esc": esc / n}


def alone(keys, s):
    right = sum(sum(s[k]["correct"].values()) for k in keys)
    total = sum(len(s[k]["correct"]) for k in keys)
    return {"acc": right / total, "cost": 1000 * sum(s[k]["cost"] for k in keys) / len(keys),
            "latency": sum(s[k]["latency"] for k in keys) / len(keys)}


def conf_auroc(keys, s):
    """AUROC of the top probability for right against wrong answers (question level)."""
    conf = [s[k]["qconf"][q] for k in keys for q in s[k]["correct"]]
    right = [s[k]["correct"][q] for k in keys for q in s[k]["correct"]]
    return auroc(conf, right)


def main():
    second = {m: items(m, "b1") for m in SECOND}
    rows, curves = [], []
    for m1 in FIRST:
        s1 = items(m1, "e2")
        for m2 in SECOND:
            s2 = second[m2]
            for task in TASK_ORDER:
                keys = sorted(k for k in s1.keys() & s2.keys() if k[0] == task)
                if len(keys) < 100:
                    continue
                halves = [[k for k in keys if zlib.crc32(k[1].encode()) % 2 == h] for h in (0, 1)]
                results = {}
                for mode in (False, True):
                    tests = []
                    for dev, test in ((halves[0], halves[1]), (halves[1], halves[0])):
                        target = alone(dev, s2)["acc"]
                        ok = [t for t in GRID if evaluate(dev, s1, s2, t, mode)["acc"] >= target]
                        t = min(ok) if ok else 1.01  # 1.01: send every item
                        tests.append(evaluate(test, s1, s2, t, mode) | {"t": t})
                    results[mode] = {k: float(np.mean([x[k] for x in tests])) for k in ("acc", "cost", "latency", "esc")}
                c, cq = results[False], results[True]
                a1, a2 = alone(keys, s1), alone(keys, s2)
                rows.append({
                    "first": m1, "second": m2, "task": task, "items": len(keys),
                    "acc_first": a1["acc"], "acc_second": a2["acc"], "acc_cascade": c["acc"],
                    "cost_first": a1["cost"], "cost_second": a2["cost"], "cost_cascade": c["cost"],
                    "lat_first": a1["latency"], "lat_second": a2["latency"], "lat_cascade": c["latency"],
                    "escalated": c["esc"], "cost_ratio": c["cost"] / a2["cost"] if a2["cost"] else float("nan"),
                    "acc_cascade_q": cq["acc"], "escalated_q": cq["esc"],
                    "cost_ratio_q": cq["cost"] / a2["cost"] if a2["cost"] else float("nan"),
                    "auroc_first": conf_auroc(keys, s1),
                })
                for t in GRID:
                    e = evaluate(keys, s1, s2, t)
                    curves.append({"first": m1, "second": m2, "task": task, "t": t, **e})
        print(f"done {m1}")
    # check: Civil Comments without severe_toxicity (no positive item at the 0.5 rule)
    SKIP.add(("civil", "severe_toxicity"))
    second_wo = {m: items(m, "b1") for m in SECOND}
    civ = []
    for m1 in FIRST:
        s1 = items(m1, "e2")
        for m2 in SECOND:
            s2 = second_wo[m2]
            keys = sorted(k for k in s1.keys() & s2.keys() if k[0] == "civil")
            halves = [[k for k in keys if zlib.crc32(k[1].encode()) % 2 == h] for h in (0, 1)]
            tests = []
            for dev, test in ((halves[0], halves[1]), (halves[1], halves[0])):
                target = alone(dev, s2)["acc"]
                ok = [t for t in GRID if evaluate(dev, s1, s2, t)["acc"] >= target]
                t = min(ok) if ok else 1.01
                tests.append(evaluate(test, s1, s2, t))
            c = {k: float(np.mean([x[k] for x in tests])) for k in ("acc", "cost", "esc")}
            a1, a2 = alone(keys, s1), alone(keys, s2)
            civ.append({"first": m1, "second": m2, "acc_first": a1["acc"], "acc_second": a2["acc"],
                        "acc_cascade": c["acc"], "escalated": c["esc"],
                        "cost_ratio": c["cost"] / a2["cost"] if a2["cost"] else float("nan")})
    SKIP.clear()
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "e6_civil_wo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(civ[0]))
        w.writeheader()
        w.writerows(civ)
    with open(ROOT / "results" / "e6_cascade.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "results" / "e6_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(curves[0]))
        w.writeheader()
        w.writerows(curves)

    print("\n| First | Second | Task | Acc 1 | Acc 2 | Acc cascade | Escalated | Cost ratio (cascade / 2) |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["first"] == "typesafe/jev-1.14":
            print(f"| {SYSTEM_LABEL[r['first']]} | {SYSTEM_LABEL[r['second']]} | {r['task']} | {100 * r['acc_first']:.1f} | "
                  f"{100 * r['acc_second']:.1f} | {100 * r['acc_cascade']:.1f} | {100 * r['escalated']:.0f}% | {r['cost_ratio']:.2f} |")
    # H5 summary: the share of (first, second, task) cases where the cascade keeps the accuracy
    # of the System Two model (within 1 point) at less than half of its cost
    tested = [r for r in rows if r["acc_second"] > r["acc_first"]]
    kept = [r for r in tested if r["acc_cascade"] >= r["acc_second"] - 0.01 and r["cost_ratio"] < 0.5]
    print(f"\nCases where System Two is more accurate than System One: {len(tested)} of {len(rows)}")
    print(f"Of these, the cascade keeps the accuracy (within 1 point) at < 50% of the cost: {len(kept)}")


if __name__ == "__main__":
    main()
