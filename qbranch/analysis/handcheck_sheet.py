"""Make a sheet of 50 C2 checker decisions for a person to check by hand (CSV, opens in Google Sheets).

The rows come from c2_<variant>.json (C2 version 2): 20 predicted questions that the checker found answered,
10 that it did not, 10 strict-control questions (random_fit) found answered, and 10 not answered.
For each row, the sheet gives the 10 later facts nearest to the question (the checker saw 30) and the fact
that the checker chose. Fill in the column "your_verdict" with correct or wrong.

Usage: uv run python analysis/handcheck_sheet.py runs/multihoprag_nall [--variant qbranch]
Output: <run_dir>/handcheck_c2_<variant>.csv
"""
import collections
import csv
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import answer  # noqa: E402
import build  # noqa: E402
import json  # noqa: E402
from graph import Graph, norm_entity  # noqa: E402

run_dir = sys.argv[1]
VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "qbranch"
d = json.load(open(os.path.join(run_dir, "c2_%s.json" % VARIANT)))
g = Graph(build.load_records(os.path.join(run_dir, "build_" + VARIANT)), VARIANT)
F = answer.DenseIndex([g.fact_text(f) for f in g.facts])
about = collections.defaultdict(list)
for f in g.facts:
    about[f["s_norm"]].append(f)
    if f["o_norm"] != f["s_norm"]:
        about[f["o_norm"]].append(f)
rng = random.Random(0)


def ans(r, key):
    return bool(r.get(key) and r[key]["answered"])


groups = [("pred", True, 20), ("pred", False, 10), ("random_fit", True, 10), ("random_fit", False, 10)]
picked = []
for key, val, n in groups:
    pool = [r for r in d["rows"] if (key == "pred" or r.get("q_fit")) and ans(r, key) == val]
    picked += [(key, r) for r in rng.sample(pool, min(n, len(pool)))]
rng.shuffle(picked)  # the person does not see the group order

with open(os.path.join(run_dir, "handcheck_c2_%s.csv" % VARIANT), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "question_kind", "entity", "raised_in", "question", "checker_says", "checker_answer",
                "checker_fact", "checker_fact_source", "nearest_later_facts", "your_verdict", "note"])
    for i, (key, r) in enumerate(picked, 1):
        q = r["q"] if key == "pred" else r["q_fit"]
        s_norm = norm_entity(r["slot"][0])
        later = [f for f in about.get(s_norm, []) if f["batch"] > r["batch"]]
        if later:
            sc = F.M[[f["fid"] for f in later]] @ answer.embed_queries([q])[0]
            later = [later[j] for j in np.argsort(-sc)[:10]]
        c = r[key] or {}
        w.writerow([i, "predicted" if key == "pred" else "control", r["slot"][0], r["title"], q,
                    "answered" if c.get("answered") else "not answered", c.get("answer") or "",
                    " | ".join(c["fact"]) if c.get("fact") else "", c.get("fact_title") or "",
                    "\n".join("%s | %s | %s  [%s]" % (f["s"], f["r"], f["o"], g.pid_title[f["pid"]]) for f in later),
                    "", ""])
print("wrote", os.path.join(run_dir, "handcheck_c2_%s.csv" % VARIANT), len(picked), "rows")
