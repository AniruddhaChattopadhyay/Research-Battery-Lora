"""MuSiQue diagnostic for path-planned systems: does the chain find the gold bridge entities?

MuSiQue gives the gold decomposition with the answer of each hop. A bridge is the answer of a hop
that is not the last hop. A bridge is "found" when one hop answer of the chain matches it
(token F1 >= 0.5 after normalization, or one contains the other).

Usage: uv run python analysis/bridges.py runs/musique_n100 dense_plan graph_qbranch_plan ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate  # noqa: E402


def match(gold, pred):
    g, p = evaluate._norm(gold), evaluate._norm(pred)
    return bool(g) and bool(p) and (g in p or p in g or evaluate._f1(pred, gold) >= 0.5)


def main(run_dir, systems):
    raw = {q["id"]: q for q in json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "hipporag2",
                                                           "musique.json")))}
    out = {}
    for s in systems:
        rows = [json.loads(line) for line in open(os.path.join(run_dir, "answers", s + ".jsonl"))]
        found, all_found, steps_ok, acc_all, acc_miss = [], [], [], [], []
        for r in rows:
            dec = raw[r["qid"]]["question_decomposition"]
            bridges = [d["answer"] for d in dec[:-1]]
            answers = [t["answer"] for t in r["trace"] or []]
            hit = [any(match(b, a) for a in answers) for b in bridges]
            found += hit
            ok = all(hit)
            all_found.append(ok)
            steps_ok.append(len(answers) == len(dec))
            (acc_all if ok else acc_miss).append(bool(r["judge"]))
        out[s] = {"n": len(rows), "bridge_recall": float(np.mean(found)), "all_bridges_found": float(np.mean(all_found)),
                  "plan_length_equals_gold": float(np.mean(steps_ok)),
                  "judge_if_all_bridges_found": float(np.mean(acc_all)) if acc_all else None,
                  "judge_if_a_bridge_missed": float(np.mean(acc_miss)) if acc_miss else None}
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2:]), indent=1))
