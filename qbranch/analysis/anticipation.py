"""C2: do the open slots predict the facts that later documents bring?

For each open slot (entity s, relation r, ?) that a passage in batch t raised, look for a fact
(s, r', o) from a passage in a later batch, where r' matches r. A match is either the same
normalized name, or relation-name embeddings with cosine >= TAU.
Baselines:
  - any relation: a later batch gives any new fact about s (upper bound of what an entity predicts).
  - random relation: replace r by a relation drawn from the global distribution of fact relations
    (20 draws per slot). The gap between "predicted r" and "random r" is the value of the prediction.
Also: the slot was "already known" if an earlier or the same batch had the fact (from another passage).

Usage: uv run python analysis/anticipation.py runs/hotpotqa_n100
"""
import collections
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import answer  # noqa: E402
import build  # noqa: E402
import llm  # noqa: E402
from graph import Graph  # noqa: E402

TAU = 0.85
run_dir = sys.argv[1]
use_norm = "--norm" in sys.argv
rel_map = json.load(open(os.path.join(run_dir, "relmap_qbranch.json"))) if use_norm else None
g = Graph(build.load_records(os.path.join(run_dir, "build_qbranch")), "qbranch", rel_map=rel_map)
rels = sorted({f["r"] for f in g.facts} | {s["r"] for s in g.open_slots})
R = answer._unit(llm.embed(answer.EMBED_MODEL, [r.replace("_", " ") for r in rels]))
ri = {r: i for i, r in enumerate(rels)}
SIM = R @ R.T

by_ent = collections.defaultdict(list)  # facts by subject
by_obj = collections.defaultdict(list)  # facts by object (for inverse slots)
for f in g.facts:
    by_ent[f["s_norm"]].append(f)
    by_obj[f["o_norm"]].append(f)


def match(r1, r2, soft):
    return r1 == r2 or (soft and SIM[ri[r1], ri[r2]] >= TAU)


def hit(slot, r, soft, when):
    pool = by_obj if slot.get("inverse") else by_ent
    for f in pool.get(slot["s_norm"], []):
        if f["pid"] == slot["pid"]:
            continue
        if when == "later" and f["batch"] <= slot["batch"]:
            continue
        if when == "known" and f["batch"] > slot["batch"]:
            continue
        if r is None or match(f["r"], r, soft):
            return f
    return None


rng = random.Random(0)
fact_rels = [f["r"] for f in g.facts]
last = max(s["batch"] for s in g.open_slots)
slots = [s for s in g.open_slots if s["batch"] < last]  # the last batch has no later batch
out = {"open_slots_total": len(g.open_slots), "open_slots_with_a_later_batch": len(slots)}
for soft in (False, True):
    tag = "soft" if soft else "exact"
    pred = [hit(s, s["r"], soft, "later") is not None for s in slots]
    known = [hit(s, s["r"], soft, "known") is not None for s in slots]
    rand = [np.mean([hit(s, rng.choice(fact_rels), soft, "later") is not None for _ in range(20)]) for s in slots]
    out[tag] = {"bound_later": float(np.mean(pred)), "already_known": float(np.mean(known)),
                "random_relation_bound_later": float(np.mean(rand))}
out["any_relation_later"] = float(np.mean([hit(s, None, False, "later") is not None for s in slots]))
examples = []
for s in slots:
    f = hit(s, s["r"], True, "later")
    if f and len(examples) < 12:
        examples.append({"slot": [s["s"], s["r"], "?"], "batch": s["batch"], "bound_by": [f["s"], f["r"], f["o"]],
                         "bound_batch": f["batch"], "bound_in": g.pid_title[f["pid"]]})
out["examples"] = examples
out["top_open_relations"] = collections.Counter(s["r"] for s in g.open_slots).most_common(15)
out["relations_in_graph"] = len({f["r"] for f in g.facts})
json.dump(out, open(os.path.join(run_dir, "anticipation%s.json" % ("_norm" if use_norm else "")), "w"), indent=1)
print(json.dumps(out, indent=1))
