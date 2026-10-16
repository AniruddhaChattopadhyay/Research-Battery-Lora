"""Details of the System Two runs (B1, B3): reasoning tokens, failed calls and their effect.

  uv run python analysis/s2_details.py

For each model: the share of calls with reasoning tokens, the mean reasoning tokens, the
number of items without a usable answer (status not 200), and the mean accuracy over tasks
when we count these items as wrong. For JudgeBench-Pro: the accuracy of the calls with and
without reasoning tokens (the model chooses when to reason, so this split is not random).
Writes results/s2_details.csv.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from common import ROOT, SYSTEM_LABEL, SYSTEM_TWO, TASK_ORDER, distribution, file_name, gold_label, top

RUNS = [(m, "b1", m) for m in SYSTEM_TWO] + [("openai/gpt-6-luna", "b3-none", "openai/gpt-6-luna@none")]


def main():
    out = []
    for model, tag, key in RUNS:
        last = {}
        for line in open(ROOT / "runs" / tag / file_name(model), encoding="utf-8"):
            r = json.loads(line)
            k = (r["task"], r["item_id"])
            if k not in last or r["status"] == 200:
                last[k] = r
        reasoning, has_reasoning = [], []
        acc_ok, acc_fail = defaultdict(list), defaultdict(list)
        jb = {True: [], False: []}
        failed = 0
        for (task, item_id), r in last.items():
            if r["status"] != 200:
                failed += 1
                for q in r["request"]["questions"]:
                    acc_fail[task].append(False)
                continue
            usage = r["response"].get("usage") or {}
            rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            reasoning.append(rt)
            has_reasoning.append(rt > 0)
            dists = {qname: distribution(q["type"], (r["response"].get("answers") or {}).get(qname))
                     for qname, q in r["request"]["questions"].items()}
            if all(d is None for d in dists.values()):
                failed += 1  # no usable answer: the E2 analysis drops the item
                for q in dists:
                    acc_fail[task].append(False)
                continue
            for qname, q in r["request"]["questions"].items():
                d = dists[qname]
                if d is None:
                    acc_fail[task].append(False)
                    continue
                right = top(d) == gold_label(q["type"], r["gold"][qname])
                acc_ok[task].append(right)
                acc_fail[task].append(right)
                if task == "judgebench":
                    jb[rt > 0].append(right)
        out.append({
            "system": key, "label": SYSTEM_LABEL.get(model, model) if "@" not in key else "GPT-6 Luna (no reasoning)",
            "calls": len(last), "failed": failed,
            "share_with_reasoning": float(np.mean(has_reasoning)), "mean_reasoning_tokens": float(np.mean(reasoning)),
            "acc_mean_ok": float(np.mean([np.mean(v) for t, v in acc_ok.items()])),
            "acc_mean_fail_wrong": float(np.mean([np.mean(v) for t, v in acc_fail.items()])),
            "jb_share_reasoning": len(jb[True]) / max(1, len(jb[True]) + len(jb[False])),
            "jb_acc_reasoning": float(np.mean(jb[True])) if jb[True] else "",
            "jb_acc_no_reasoning": float(np.mean(jb[False])) if jb[False] else "",
            "jb_n_no_reasoning": len(jb[False]),
        })
    with open(ROOT / "results" / "s2_details.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
