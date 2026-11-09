"""Paired comparison of answer systems (bootstrap, 5,000 draws, 95% interval).

Usage:
  uv run python analysis/compare.py runs/musique_nall graph_qbranch_plan:dense_plan graph_qbranch:dense
  add --exclude runs/musique_n100 to drop the questions of the development slice (held-out view).
  add --type 4hop to keep only one question type.
Output: a table per comparison, the scores per system, and judge accuracy by question type (MuSiQue:
number of hops).
"""
import json
import os
import sys

import numpy as np


def load(run_dir, system):
    p = os.path.join(run_dir, "answers", system + ".jsonl")
    return {r["qid"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else None


def qtype(qid):
    q = str(qid)
    return q.split("hop")[0] + "hop" if "hop" in q[:5] else q.rsplit("_", 1)[0]


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(argv):
    run_dir = argv[0]
    exclude = set()
    if "--exclude" in argv:
        ex_dir = argv[argv.index("--exclude") + 1]
        exclude = {json.loads(line)["qid"] for line in open(os.path.join(ex_dir, "answers", "dense.jsonl"))}
        argv = [a for a in argv if a not in ("--exclude", ex_dir)]
    only = None
    if "--type" in argv:
        only = argv[argv.index("--type") + 1]
        argv = [a for a in argv if a not in ("--type", only)]
    pairs = [a.split(":") for a in argv[1:] if ":" in a]
    systems = sorted({s for p in pairs for s in p} | set(a for a in argv[1:] if ":" not in a))
    data = {s: load(run_dir, s) for s in systems}
    data = {s: d for s, d in data.items() if d}
    common = set.intersection(*(set(d) for d in data.values())) - exclude
    if only:
        common = {q for q in common if qtype(q) == only}
    out = {"n": len(common), "excluded": len(exclude), "systems": {}, "pairs": {}, "by_type": {}}
    for s, d in data.items():
        rows = [d[q] for q in common]
        out["systems"][s] = {k: float(np.mean([float(r[k] or 0) for r in rows])) for k in ("em", "f1", "judge")}
        rec = [r["recall"] for r in rows if r["recall"] is not None]
        out["systems"][s]["recall"] = float(np.mean(rec)) if rec else None
        out["systems"][s]["passages"] = float(np.mean([len(r["pids"]) for r in rows]))
        out["systems"][s]["reader_prompt_tokens"] = float(np.mean([r["reader_usage"].get("prompt", 0) for r in rows]))
    for a, b in pairs:
        if a not in data or b not in data:
            continue
        qs = sorted(common)
        res = {}
        for k in ("judge", "f1", "recall"):
            d = np.array([float(data[a][q][k] or 0) - float(data[b][q][k] or 0) for q in qs])
            res[k] = boot(d)
        res["judge_wins_losses"] = [int(sum(1 for q in qs if data[a][q]["judge"] and not data[b][q]["judge"])),
                                    int(sum(1 for q in qs if data[b][q]["judge"] and not data[a][q]["judge"]))]
        out["pairs"]["%s - %s" % (a, b)] = res
    types = sorted({qtype(q) for q in common})
    if len(types) > 20:  # the qid has no type (HotpotQA, 2Wiki): take the type from the data
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        import data as _data
        name = os.path.basename(os.path.normpath(run_dir)).rsplit("_n", 1)[0]
        tmap = {q["qid"]: q["type"] for q in _data.load(name, None)[0]}
        types = sorted({tmap.get(q, "?") for q in common})
        qtype_ = lambda q: tmap.get(q, "?")  # noqa: E731
    else:
        qtype_ = qtype
    for t in types:
        qs = [q for q in common if qtype_(q) == t]
        out["by_type"][t] = {"n": len(qs), **{s: float(np.mean([d[q]["judge"] for q in qs])) for s, d in data.items()}}
    return out


if __name__ == "__main__":
    o = main(sys.argv[1:])
    print("n = %d questions (excluded %d)" % (o["n"], o["excluded"]))
    print("%-26s %6s %6s %6s %7s %6s %7s" % ("system", "EM", "F1", "judge", "recall", "pass.", "tokens"))
    for s, v in sorted(o["systems"].items(), key=lambda x: x[1]["judge"]):
        print("%-26s %6.3f %6.3f %6.3f %7s %6.1f %7.0f" % (s, v["em"], v["f1"], v["judge"],
              "%.3f" % v["recall"] if v["recall"] is not None else "-", v["passages"], v["reader_prompt_tokens"]))
    for p, r in o["pairs"].items():
        print("%s: judge %+.3f [%+.3f, %+.3f]  F1 %+.3f [%+.3f, %+.3f]  recall %+.3f [%+.3f, %+.3f]  wins/losses %s" % (
            p, *r["judge"], *r["f1"], *r["recall"], r["judge_wins_losses"]))
    print("judge by type:")
    for t, v in o["by_type"].items():
        print("  %-18s n=%-4d " % (t, v["n"]) + "  ".join("%s %.2f" % (s, x) for s, x in v.items() if s != "n"))
