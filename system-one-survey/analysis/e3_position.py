"""E3, exploratory (added on 2026-10-16): where does the answer go when the option order changes?

  uv run python analysis/e3_position.py

Each item of BANKING77, When2Call and arXiv has two random option orders (order1, order2).
When the top answer changes between them, there are two candidate answers, a and b.
  earlier  in each order, the system picked the one of a and b that came earlier in that list
  later    in each order, it picked the one that came later
  mixed    the other flips
An exact two-sided sign test compares "earlier" with "later". We also give the share of
wrong answers in the first position. The orders are random, so without a position bias
this share is 1/K (K options).
Uses only saved calls (runs/e3). Writes results/e3_position.csv.
"""
import csv
import math
from collections import defaultdict

from common import ROOT, SYSTEM_LABEL, file_name, records

TASKS = ("when2call", "banking77", "arxiv")


def sign_test(a, b):
    """Exact two-sided p-value of a against b under p = 0.5."""
    n, k = a + b, min(a, b)
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def rows_for(model):
    pairs = defaultdict(dict)  # (task, base_id) -> variant -> (option names in order, choice, gold)
    for (task, item_id), r in records(ROOT / "runs" / "e3" / file_name(model)).items():
        variant = (r.get("meta") or {}).get("variant")
        if task not in TASKS or variant not in ("order1", "order2"):
            continue
        (qname, q), = r["request"]["questions"].items()
        answer = (r["response"].get("answers") or {}).get(qname) or {}
        if "choice" in answer:
            pairs[(task, r["meta"]["base_id"])][variant] = (list(q["criteria"]), answer["choice"], r["gold"][qname])
    out = []
    for task in TASKS:
        c = defaultdict(int)
        k = None
        for (t, _), d in pairs.items():
            if t != task:
                continue
            for names, choice, gold in d.values():
                k = len(names)
                if choice != gold:
                    c["wrong"] += 1
                    c["first_wrong"] += names.index(choice) == 0
            if len(d) < 2:
                continue
            c["items"] += 1
            (n1, a, _), (n2, b, _) = d["order1"], d["order2"]
            if a == b:
                continue
            c["flips"] += 1
            if n1.index(a) < n1.index(b) and n2.index(b) < n2.index(a):
                c["earlier"] += 1
            elif n1.index(a) > n1.index(b) and n2.index(b) > n2.index(a):
                c["later"] += 1
            else:
                c["mixed"] += 1
        if k is not None:
            out.append({"task": task, "k": k, **{key: c[key] for key in
                        ("items", "flips", "earlier", "later", "mixed", "wrong", "first_wrong")}})
    return out


def main():
    models = [line.strip() for line in open(ROOT / "runs" / "e3" / "models.txt") if line.strip()]
    table = []
    for model in models:
        rows = rows_for(model)
        if not rows:
            continue
        total = {key: sum(r[key] for r in rows) for key in ("items", "flips", "earlier", "later", "mixed")}
        for r in rows + [{"task": "all", "k": "", **total, "wrong": "", "first_wrong": ""}]:
            r = {"system": model, "label": SYSTEM_LABEL.get(model, model), **r}
            r["p_sign"] = sign_test(r["earlier"], r["later"])
            if r["task"] != "all":
                r["share_first_wrong"] = r["first_wrong"] / r["wrong"] if r["wrong"] else float("nan")
                r["share_first_chance"] = 1 / r["k"]
            table.append(r)
    keys = ["system", "label", "task", "k", "items", "flips", "earlier", "later", "mixed", "p_sign", "wrong",
            "first_wrong", "share_first_wrong", "share_first_chance"]
    with open(ROOT / "results" / "e3_position.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(table)
    print("Flips between the two random option orders (BANKING77, When2Call, arXiv pooled)")
    print("| System | Items | Flips | Earlier | Later | Mixed | Sign test p | Wrong answers first, W2C (chance 25%) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    by = {(r["system"], r["task"]): r for r in table}
    for model in models:
        r = by.get((model, "all"))
        if r is None:
            continue
        w2c = by.get((model, "when2call"))
        first = f"{100 * w2c['share_first_wrong']:.1f}" if w2c else "-"
        print(f"| {r['label']} | {r['items']} | {r['flips']} | {r['earlier']} | {r['later']} | {r['mixed']} | "
              f"{r['p_sign']:.2g} | {first} |")


if __name__ == "__main__":
    main()
