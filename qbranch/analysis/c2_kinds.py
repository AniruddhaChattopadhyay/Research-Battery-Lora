"""Label the sampled open slots of c2_semantic.json as a background fact or an open event, and give the
C2 rates for each kind.

  background: a stable fact that is already true when the passage is written (a founding year, a team).
  event: an outcome or a state that the passage leaves open and that can change or happen later
         (a verdict, a signing, a score, a decision, a release date that is not fixed yet).

Usage: uv run python analysis/c2_kinds.py runs/multihoprag_nall
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import llm  # noqa: E402

MODEL = "anthropic/claude-haiku-5.5"
PROMPT = """A news passage raised this open question. It did not give the answer.

Passage title: {title}
Open question: {q}

Is the answer a background fact (stable and already true when the passage was written, for example a founding year, a birthplace, the team of a player) or an open event (an outcome or a state that the passage leaves open and that can happen or change later, for example a verdict, a signing, a match result, a decision)?
Reply with one word: background or event."""

run_dir = sys.argv[1]
d = json.load(open(os.path.join(run_dir, "c2_semantic.json")))
rows = d["rows"]


def label(r):
    t = llm.chat(MODEL, PROMPT.format(title=r["title"], q=r["q"]), temperature=None, max_tokens=20)["text"]
    return "event" if "event" in t.lower() else "background"


with ThreadPoolExecutor(8) as ex:
    kinds = list(ex.map(label, rows))
out = {}
for k in ("background", "event"):
    rs = [r for r, kk in zip(rows, kinds) if kk == k and r["q_random"]]
    p = [bool(r["pred"] and r["pred"]["answered"]) for r in rs]
    q = [bool(r["random"] and r["random"]["answered"]) for r in rs]
    out[k] = {"n": len(rs), "share": len(rs) / len(rows), "pred": float(np.mean(p)) if rs else None,
              "random": float(np.mean(q)) if rs else None}
out["cost_usd"] = llm.usage_report()["total_cost_usd"]
out["event_examples_bound"] = [[r["q"], r["title"], r["pred"]["fact"], r["pred"]["fact_title"]]
                               for r, k in zip(rows, kinds) if k == "event" and r["pred"] and r["pred"]["answered"]][:15]
json.dump(out, open(os.path.join(run_dir, "c2_kinds.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
