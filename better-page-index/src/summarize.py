"""Score result files in CLEVRER terms (per type: accuracy, per-option, per-question)."""
import collections
import json
import sys

TYPES = ("descriptive", "explanatory", "predictive", "counterfactual")


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    by = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        c = by[r["type"]]
        c["hits"] += r["hits"]
        c["total"] += r["total"]
        c["correct"] += bool(r["correct"])
        c["n"] += 1
        c["seconds"] += r.get("seconds", 0)
        c["errors"] += bool(r.get("error"))
        c["unparsed"] += r.get("parsed") is None
    out = {}
    for t in TYPES:
        c = by[t]
        if c["n"]:
            out[t] = {"n": c["n"], "per_option": 100 * c["hits"] / c["total"], "per_question": 100 * c["correct"] / c["n"],
                      "avg_seconds": c["seconds"] / c["n"], "errors": c["errors"], "unparsed": c["unparsed"]}
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        for t, v in summarize(p).items():
            print("  %-15s n=%3d  per-option %5.1f  per-question %5.1f  avg %.1fs  errors %d  unparsed %d" % (
                t, v["n"], v["per_option"], v["per_question"], v["avg_seconds"], v["errors"], v["unparsed"]))
