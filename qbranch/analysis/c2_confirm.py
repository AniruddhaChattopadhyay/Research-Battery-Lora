"""Confirmatory C2 numbers with intervals that resample entities (notes/2026-11-19-confirmatory-plan.md).

Many open slots share an entity and its later facts, so the rows are not independent. Each bootstrap draw
resamples the entities (5,000 draws, 95%). A difference between two builds of the same stream resamples the
entities of both builds together.

Usage:
  uv run python analysis/c2_confirm.py gap  <c2 file> [<c2 file> ...]   predicted - strict control per file, and
                                                                        the mean of the files (test T1)
  uv run python analysis/c2_confirm.py diff <c2 file base> <c2 file new>  new - base (tests T2 and T3)
Output: printed JSON. Add --out <file> to save it.
Options (notes/2026-11-24-history-context-plan.md):
  --min_batch <b>   keep only the slots of batch b or later (history builds: batch 0 has no memory)
  --raw             do not apply the rule against name-only answers
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import build  # noqa: E402
from graph import Graph, norm_entity  # noqa: E402

DRAWS = 5000


def hit(r, key):
    return float(bool(r.get(key) and r[key]["answered"]))


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MIN_BATCH = int(opt("--min_batch", 0))
RAW = "--raw" in sys.argv
N_SELF = collections.Counter()


def drop_self(r, path):
    """Rule of 2026-11-24: a fact with the same subject and object, or an answer that is only the name of the
    entity of the slot, is not an answer."""
    ent = norm_entity(r["slot"][0])
    for key in ("pred", "random_fit", "random_fit_shared"):
        c = r.get(key)
        if c and c["answered"]:
            f = c.get("fact")
            if norm_entity(c.get("answer")) == ent or (f and norm_entity(f[0]) == norm_entity(f[2])):
                c["answered"] = False
                N_SELF[(os.path.basename(path), key)] += 1


def load(path):
    d = json.load(open(path))
    rows = [r for r in d["rows"] if r.get("q_fit") and r["batch"] >= MIN_BATCH]
    if not RAW:
        for r in rows:
            drop_self(r, path)
    g = Graph(build.load_records(os.path.join(d["run_dir"], d.get("build_dir") or "build_" + d["variant"])),
              d["variant"])
    first = {}  # first batch in which memory links two entities or values
    for f in g.facts:
        k = frozenset((f["s_norm"], f["o_norm"]))
        first[k] = min(first.get(k, 10 ** 6), f["batch"])

    def new(r, key):  # answered, and memory did not link the entity to this value at the batch of the slot
        c = r.get(key)
        if not (c and c["answered"] and c["fact"]):
            return 0.0
        return float(first.get(frozenset((norm_entity(c["fact"][0]), norm_entity(c["fact"][2]))), 10 ** 6) > r["batch"])

    for r in rows:
        r["ent"] = r.get("s_norm") or norm_entity(r["slot"][0])
        ctrl_s = "random_fit_shared" if r.get("q_fit_shared") else ("random_fit" if not d.get("shared_control") else None)
        r["v"] = {"pred": hit(r, "pred"), "ctrl": hit(r, "random_fit"),
                  "gap": hit(r, "pred") - hit(r, "random_fit"),
                  "gap_new": new(r, "pred") - new(r, "random_fit"),
                  "pred_new": new(r, "pred"), "pred_nopool": None if r.get("pool_entity") else hit(r, "pred"),
                  "gap_shared": (hit(r, "pred") - hit(r, ctrl_s)) if ctrl_s else None,
                  "gap_nopool": None if r.get("pool_entity") else hit(r, "pred") - hit(r, "random_fit")}
    return d, rows


def sums(rows, key, ents):
    """Per entity: the sum of the value and the count of rows that have it."""
    idx = {e: i for i, e in enumerate(ents)}
    s, n = np.zeros(len(ents)), np.zeros(len(ents))
    for r in rows:
        v = r["v"][key]
        if v is not None:
            s[idx[r["ent"]]] += v
            n[idx[r["ent"]]] += 1
    return s, n


def weights(n_ent, seed):
    return np.random.default_rng(seed).multinomial(n_ent, np.ones(n_ent) / n_ent, size=DRAWS).astype(float)


def ci(point, boots):
    return [round(float(point), 4), round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def one_file(rows, keys=("pred", "ctrl", "gap", "gap_new", "gap_shared", "gap_nopool", "pred_new", "pred_nopool"),
             seed=0):
    ents = sorted({r["ent"] for r in rows})
    w = weights(len(ents), seed)
    out, boots = {"rows": len(rows), "entities": len(ents)}, {}
    for k in keys:
        s, n = sums(rows, k, ents)
        if n.sum() == 0:
            continue
        b = (w @ s) / np.maximum(w @ n, 1)
        out[k], boots[k] = ci(s.sum() / n.sum(), b), b
        if k.startswith("gap"):
            out[k + "_n"] = int(n.sum())
    return out, boots


mode, files = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
for o in ("--out", "--min_batch"):
    if o in sys.argv:
        files.remove(sys.argv[sys.argv.index(o) + 1])
res = {"mode": mode, "files": files, "min_batch": MIN_BATCH, "name_only_rule": not RAW}
if mode == "gap":
    per, gb = [], []
    for i, f in enumerate(files):
        d, rows = load(f)
        o, b = one_file(rows, seed=i)
        per.append({"file": f, **o})
        gb.append({k: (o[k][0], b[k]) for k in ("gap", "gap_new")})
    res["per_file"] = per
    if len(files) > 1:  # the streams are independent, so the mean of the draws is a draw of the mean
        for k in ("gap", "gap_new"):
            res["mean_" + k] = ci(np.mean([x[k][0] for x in gb]), np.mean([x[k][1] for x in gb], axis=0))
else:
    (da, ra), (db, rb) = load(files[0]), load(files[1])
    ents = sorted({r["ent"] for r in ra} | {r["ent"] for r in rb})
    w = weights(len(ents), 0)
    res["base"], res["new"] = one_file(ra)[0], one_file(rb)[0]
    res["shared_entities"] = len({r["ent"] for r in ra} & {r["ent"] for r in rb})
    for k in ("gap", "pred", "gap_shared", "gap_new", "gap_nopool", "pred_new", "pred_nopool"):
        (sa, na), (sb, nb) = sums(ra, k, ents), sums(rb, k, ents)
        if na.sum() == 0 or nb.sum() == 0:
            continue
        b = (w @ sb) / np.maximum(w @ nb, 1) - (w @ sa) / np.maximum(w @ na, 1)
        res["new_minus_base_" + k] = ci(sb.sum() / nb.sum() - sa.sum() / na.sum(), b)
res["name_only_answers_removed"] = {"%s %s" % k: v for k, v in sorted(N_SELF.items())}
print(json.dumps(res, indent=1))
if "--out" in sys.argv:
    json.dump(res, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
