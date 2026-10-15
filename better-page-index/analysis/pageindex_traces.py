"""Analyze the PageIndex agent traces: what did the agent read before it answered?

For each question: tool calls, tree parts read, pages read, whether the pages belong to the
named run, and whether the agent answered from the tree summaries alone (no page read).
Usage: uv run python analysis/pageindex_traces.py runs/pilot/pageindex.jsonl
"""
import collections
import json
import os
import re
import sys

PAGES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "pilot",
                                    "plant_log_30_runs_pages.json")))


def pages_of(spec):
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        m = re.match(r"^(\d+)(?:-(\d+))?$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            out.update(range(a, b + 1))
    return out


def trace(rec):
    calls = [e for e in rec.get("events", []) if e.get("type") == "tool_call"]
    struct = [c for c in calls if c.get("name") == "get_document_structure"]
    reads = [c for c in calls if c.get("name") == "get_page_content"]
    pages = set()
    for c in reads:
        pages |= pages_of((c.get("arguments") or {}).get("pages", ""))
    first, last = PAGES[str(rec["sid"])]
    run_pages = set(range(first, last + 1))
    return {"n_calls": len(calls), "n_struct": len(struct), "n_reads": len(reads), "pages": len(pages),
            "right_run_pages": len(pages & run_pages), "other_run_pages": len(pages - run_pages),
            "summary_only": not reads}


if __name__ == "__main__":
    recs = [json.loads(line) for line in open(sys.argv[1])]
    rows = [(r, trace(r)) for r in recs]
    groups = collections.defaultdict(lambda: collections.Counter())
    for r, t in rows:
        g = groups[(r["type"], "summary only" if t["summary_only"] else "read pages")]
        g["n"] += 1
        g["correct"] += bool(r["correct"])
        g["pages"] += t["pages"]
        g["other_run_pages"] += t["other_run_pages"] > 0
        g["struct"] += t["n_struct"]
    print("%-15s %-13s %4s %8s %10s %12s %14s" % ("type", "evidence", "n", "acc %", "pages/q", "tree parts/q", "read wrong run"))
    for (t, ev), g in sorted(groups.items()):
        print("%-15s %-13s %4d %8.1f %10.1f %12.1f %14d" % (t, ev, g["n"], 100 * g["correct"] / g["n"], g["pages"] / g["n"],
                                                           g["struct"] / g["n"], g["other_run_pages"]))
    tot = collections.Counter()
    for r, t in rows:
        tot["n"] += 1
        tot["summary_only"] += t["summary_only"]
        tot["pages"] += t["pages"]
        tot["calls"] += t["n_calls"]
        tot["hit_cap"] += "MaxTurns" in str(r.get("error", ""))
    print("questions %d, answered from summaries only %d (%.0f%%), avg pages read %.1f, avg tool calls %.1f, hit turn cap %d" % (
        tot["n"], tot["summary_only"], 100 * tot["summary_only"] / tot["n"], tot["pages"] / tot["n"], tot["calls"] / tot["n"], tot["hit_cap"]))
