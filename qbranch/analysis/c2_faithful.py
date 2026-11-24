"""Does the source passage state the fact that the checker used as the answer? (history builds, 2026-11-24)

A history build shows its open questions to later passages, and the prompt asks the model to bind them. This
check reads the passage of each answering fact and asks the checker model if the passage states the fact.
It samples the answered predicted rows of one C2 file (batch 1 or later).

Usage: uv run python analysis/c2_faithful.py <c2 file> [--n 200]
"""
import collections
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import build  # noqa: E402
import data  # noqa: E402
import llm  # noqa: E402
from build import _norm_rel  # noqa: E402

MODEL = "anthropic/claude-haiku-5.5"
PROMPT = """Passage ({title}):
{text}

Fact: {s} | {r} | {o}

Does the passage state this fact, or clearly imply it? Reply with one word: yes or no."""

d = json.load(open(sys.argv[1]))
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
v = d.get("facts_from") or d["variant"]
recs = build.load_records(os.path.join(d["run_dir"], "build_" + v))
by_pid = {p["pid"]: p for p in data.load("multihoprag", None)[1]}
idx = collections.defaultdict(list)
for rec in recs:
    for tr in build.triples_of(rec, v):
        if tr.get("o") is not None:
            idx[(str(tr.get("s")), _norm_rel(tr.get("r")), str(tr.get("o")), rec["batch"])].append(rec)
rows = []
for r in d["rows"]:
    c = r.get("pred")
    if r["batch"] >= 1 and r.get("q_fit") and c and c["answered"] and c["fact"]:
        rs = idx.get((c["fact"][0], c["fact"][1], c["fact"][2], c["fact_batch"]), [])
        if rs:
            shown = any(r["q"].strip() in (x.get("mem") or ["", ""])[1] for x in rs)
            rows.append((r, rs[0], shown))
rows = random.Random(0).sample(rows, min(n, len(rows)))


def one(item):
    r, rec, shown = item
    p, f = by_pid[rec["pid"]], r["pred"]["fact"]
    t = llm.chat(MODEL, PROMPT.format(title=p["title"], text=p["text"], s=f[0], r=f[1], o=f[2]), temperature=None,
                 max_tokens=10)["text"]
    return {"q": r["q"], "fact": f, "shown": shown, "stated": t.strip().lower().startswith("yes")}


with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(one, rows))
for key in (True, False):
    part = [o for o in out if o["shown"] == key]
    if part:
        print("question shown to the passage: %s  rows %d  fact stated in the passage %.3f"
              % (key, len(part), sum(o["stated"] for o in part) / len(part)))
json.dump({"file": sys.argv[1], "rows": out, "cost_usd": llm.usage_report()["total_cost_usd"]},
          open(sys.argv[1].replace(".json", "_faithful.json"), "w"), indent=1)
