"""Show one pilot question as the three systems saw it. Reads the saved runs; makes no API call.

Usage:
  uv run python analysis/show_example.py --list          # questions that PageIndex missed and both others got
  uv run python analysis/show_example.py 10572 1         # one question: run id, question id
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import qa  # noqa: E402
from pageindex_traces import PAGES, trace  # noqa: E402

PILOT = os.path.join(ROOT, "runs", "pilot")


def load(name):
    return {(r["sid"], r["qid"]): r for r in map(json.loads, open(os.path.join(PILOT, name)))}


def mark(rec):
    return "RIGHT" if rec["correct"] else "WRONG"


def gold(item):
    q = item["q"]
    if q["question_type"] == "descriptive":
        return q["answer"]
    return ", ".join("%d=%s" % (c["choice_id"], "yes" if c["answer"] == "correct" else "no") for c in q["choices"])


def given(rec):
    p = rec.get("parsed") or {}
    if "answer" in p:
        return str(p["answer"])
    ch = p.get("choices") or {}
    return ", ".join("%s=%s" % (k, "yes" if v is True else "no") for k, v in sorted(ch.items()))


def show(key, items, pi, fc, ga):
    item = items[key]
    first, last = PAGES[str(key[0])]
    print("=" * 78)
    print(qa.question_text(item))
    print("Correct answer: %s" % gold(item))
    print("-" * 78)
    r = pi[key]
    t = trace(r)
    calls = [e for e in r["events"] if e.get("type") == "tool_call"]
    print("PageIndex      %s  answer: %s" % (mark(r), given(r)))
    print("  run %d is on pages %d-%d (%d pages)" % (key[0], first, last, last - first + 1))
    for c in calls:
        args = c.get("arguments") or {}
        what = "tree part %s" % args.get("part") if c["name"] == "get_document_structure" else "pages %s" % args.get("pages")
        print("  %-24s %s" % (c["name"], what))
    print("  pages read: %d of %d; %.0f s" % (t["right_run_pages"], last - first + 1, r["seconds"]))
    r = fc[key]
    print("Full context   %s  answer: %s  (all %d pages of the run; %.0f s)" % (mark(r), given(r), last - first + 1, r["seconds"]))
    r = ga[key]
    print("Graph agent    %s  answer: %s  (%.0f s)" % (mark(r), given(r), r["seconds"]))
    for c in r["tool_calls"]:
        print("  %-24s %s" % (c["name"], json.dumps({k: v for k, v in c["args"].items() if k != "run_id"})))


if __name__ == "__main__":
    items = {(i["sid"], i["qid"]): i for i in qa.pilot_questions(qa.pilot_runs())}
    pi, fc, ga = load("pageindex.jsonl"), load("fullcontext.jsonl"), load("graph_agent.jsonl")
    if sys.argv[1:] == ["--list"]:
        for k in sorted(items):
            if not pi[k]["correct"] and fc[k]["correct"] and ga[k]["correct"]:
                print("%d %2d  %-15s %s" % (k[0], k[1], items[k]["type"], items[k]["q"]["question"]))
    else:
        show((int(sys.argv[1]), int(sys.argv[2])), items, pi, fc, ga)
