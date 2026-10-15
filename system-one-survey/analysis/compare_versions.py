"""Compare two runs on the same items (experiment E7): how much did the answers change?

Usage:
  uv run python analysis/compare_versions.py runs/e7-2026-10-15/typesafe__jev-1.13.jsonl runs/e7-2026-10-15/typesafe__jev-1.14.jsonl

One row for each task, and one row for all tasks:
  questions   questions with a usable answer in both runs
  top change  share of questions where the top answer changed (predicate: the side of 0.5;
              choice: the chosen option; score: the most probable level)
  mean dist   predicate: mean |p_A - p_B|. Choice and score: mean total variation distance
              between the two probability distributions.
  acc A/B     accuracy against the gold answers (soft predicate labels are cut at 0.5;
              score: the most probable level must be the gold level)
  McNemar p   exact two-sided test on the questions that only one run answers correctly
"""
import json
import math
import sys
from collections import defaultdict


def distribution(qtype, ans):
    """The answer as a dict of probabilities, or None if it is not usable."""
    if not isinstance(ans, dict):
        return None
    if qtype == "noul":
        p = ans.get("noul")
        return {"true": p, "false": 1 - p} if isinstance(p, (int, float)) else None
    probs = ans.get("probabilities")
    return {str(k): v for k, v in probs.items()} if isinstance(probs, dict) and probs else None


def top(dist):
    return max(dist, key=dist.get)


def gold_label(qtype, gold):
    if qtype == "noul":
        return "true" if (gold >= 0.5 if isinstance(gold, float) else gold) else "false"
    return str(gold)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] != 200:
            continue
        answers = r["response"].get("answers", {})
        for qname, q in r["request"]["questions"].items():
            d = distribution(q["type"], answers.get(qname))
            if d is not None:
                out[(r["task"], r["item_id"], qname)] = (q["type"], d, gold_label(q["type"], r["gold"][qname]))
    model = r["response"].get("model", path)
    return model, out


def main():
    model_a, a = load(sys.argv[1])
    model_b, b = load(sys.argv[2])
    rows = defaultdict(lambda: {"n": 0, "changed": 0, "dist": 0.0, "right_a": 0, "right_b": 0, "only_a": 0, "only_b": 0})
    for key in a.keys() & b.keys():
        qtype, da, gold = a[key]
        _, db, _ = b[key]
        if qtype == "noul":
            dist = abs(da["true"] - db["true"])
        else:
            dist = 0.5 * sum(abs(da.get(k, 0) - db.get(k, 0)) for k in da.keys() | db.keys())
        ra, rb = top(da) == gold, top(db) == gold
        for task in (key[0], "all"):
            s = rows[task]
            s["n"] += 1
            s["changed"] += top(da) != top(db)
            s["dist"] += dist
            s["right_a"] += ra
            s["right_b"] += rb
            s["only_a"] += ra and not rb
            s["only_b"] += rb and not ra
    print(f"A = {model_a}\nB = {model_b}\n")
    print("| Task | Questions | Top change | Mean dist | Acc A | Acc B | McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for task in sorted(rows, key=lambda t: (t == "all", t)):
        s = rows[task]
        n = s["n"]
        print(
            f"| {task} | {n} | {100 * s['changed'] / n:.1f}% | {s['dist'] / n:.3f} | "
            f"{100 * s['right_a'] / n:.1f}% | {100 * s['right_b'] / n:.1f}% | {mcnemar(s['only_a'], s['only_b']):.3g} |"
        )


if __name__ == "__main__":
    main()
