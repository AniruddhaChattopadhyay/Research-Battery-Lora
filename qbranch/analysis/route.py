"""Label-free routing between two plan systems by the length of the system's own plan.

The planner writes the plan before retrieval, so the plan length is a signal that needs no labels.
Rule: use system A when the plan has >= K steps, else system B. K is chosen on the development
questions (the 100 questions of the pilot slice), then tested on the other questions (held out).
Both systems get the same plan (the plan call is cached), so the length is the same for both.

Usage: uv run python analysis/route.py runs/musique_nall graph_qbranch_plan dense_plan runs/musique_n100
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import boot, load  # noqa: E402

run_dir, sys_a, sys_b, dev_dir = sys.argv[1:5]
A, B = load(run_dir, sys_a), load(run_dir, sys_b)
dev = {json.loads(line)["qid"] for line in open(os.path.join(dev_dir, "answers", "dense.jsonl"))}
qids = sorted(set(A) & set(B))
steps = {q: len(A[q]["trace"] or []) for q in qids}
assert all(steps[q] == len(B[q]["trace"] or []) for q in qids), "the plans differ"


def acc(qs, k):
    return float(np.mean([(A if steps[q] >= k else B)[q]["judge"] for q in qs]))


dev_q = [q for q in qids if q in dev]
test_q = [q for q in qids if q not in dev]
cands = sorted({1, 2, 3, 4, 5})
dev_acc = {k: acc(dev_q, k) for k in cands}
best = max(cands, key=lambda k: (dev_acc[k], k))  # ties: the larger K (use A less often)
out = {"dev_n": len(dev_q), "test_n": len(test_q), "dev_accuracy_by_K": dev_acc, "K": best,
       "share_routed_to_A_test": float(np.mean([steps[q] >= best for q in test_q])),
       "test": {"routed": acc(test_q, best), sys_a: acc(test_q, 1), sys_b: acc(test_q, 99)}}
d = np.array([float((A if steps[q] >= best else B)[q]["judge"]) - float(B[q]["judge"]) for q in test_q])
out["routed_minus_%s" % sys_b] = boot(d)
d = np.array([float((A if steps[q] >= best else B)[q]["judge"]) - float(A[q]["judge"]) for q in test_q])
out["routed_minus_%s" % sys_a] = boot(d)
out["test_accuracy_by_K"] = {k: acc(test_q, k) for k in cands}
print(json.dumps(out, indent=1))
