"""Test-retest of the C2 checker: ask the checker again (new call, no cache) for the predicted questions of a
c2 file, and compare with the stored decisions. This shows the noise of the checker and any drift in time.

Usage: uv run python analysis/c2_recheck.py runs/multihoprag_nall/c2_qbranch.json [--n 600]
Output: printed JSON, and <c2 file without .json>_recheck.json
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import answer  # noqa: E402
import build  # noqa: E402
import llm  # noqa: E402
from graph import Graph, norm_entity  # noqa: E402

CHECK_MODEL, TOP = "anthropic/claude-haiku-5.5", 30
CHECK = """Facts from a knowledge graph:
{facts}

Question: {question}

Does one of the facts give the answer to the question? The fact must be about the same entity and must give the specific value that the question asks for. A related fact that does not give that value is not an answer.
Return only a JSON object: {{"answered": true or false, "fact": <the number of the fact, or null>, "answer": "<short answer or empty>"}}"""

path = sys.argv[1]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 600
d = json.load(open(path))
g = Graph(build.load_records(os.path.join(d["run_dir"], d.get("build_dir") or "build_" + d["variant"])), d["variant"])
F = answer.DenseIndex([g.fact_text(f) for f in g.facts])
about = collections.defaultdict(list)
for f in g.facts:
    about[f["s_norm"]].append(f)
    if f["o_norm"] != f["s_norm"]:
        about[f["o_norm"]].append(f)
rows = d["rows"][:N]


def one(r):
    later = [f for f in about.get(r.get("s_norm") or norm_entity(r["slot"][0]), []) if f["batch"] > r["batch"]]
    sc = F.M[[f["fid"] for f in later]] @ answer.embed_queries([r["q"]])[0]
    facts = [later[i] for i in np.argsort(-sc)[:TOP]]
    lines = "\n".join("%d. %s | %s | %s" % (i + 1, f["s"], f["r"], f["o"]) for i, f in enumerate(facts))
    msg = CHECK.format(facts=lines, question=r["q"])
    same = llm.chat(CHECK_MODEL, msg, temperature=None, max_tokens=300)  # the stored call, if the prompt is the same
    again = llm.chat(CHECK_MODEL, msg, temperature=None, max_tokens=300, seed_tag="recheck")
    return {"stored": bool(r["pred"] and r["pred"]["answered"]), "prompt_in_cache": same["cached"],
            "again": bool((llm.parse_json(again["text"]) or {}).get("answered"))}


with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(one, rows))
a, b = np.array([o["stored"] for o in out]), np.array([o["again"] for o in out])
res = {"file": path, "rows": len(out), "prompt_in_cache": float(np.mean([o["prompt_in_cache"] for o in out])),
       "stored_rate": float(a.mean()), "again_rate": float(b.mean()), "agree": float((a == b).mean()),
       "yes_to_no": int((a & ~b).sum()), "no_to_yes": int((~a & b).sum()), "cost_usd": llm.usage_report()["total_cost_usd"]}
print(json.dumps(res, indent=1))
json.dump({**res, "rows_detail": out}, open(path[:-5] + "_recheck.json", "w"), indent=1)
