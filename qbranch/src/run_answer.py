"""Answer the benchmark questions with each system and score them.

Example: uv run python src/run_answer.py --dataset hotpotqa --n 100
Output: runs/<dataset>_n<n>/answers/<system>.jsonl and summary.json
Systems:
  closed_book     no evidence
  dense           top-5 passages by embedding similarity
  oracle          gold supporting passages
  graph_qbranch   QBranch graph + inferred-question index (full method, panel c)
  graph_qbranch_noq  QBranch graph, no inferred-question index (ablation)
  graph_openie    graph from plain triple extraction, no questions (ablation)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer  # noqa: E402
import build  # noqa: E402
import data  # noqa: E402
import evaluate  # noqa: E402
import llm  # noqa: E402
from graph import Graph  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="hotpotqa")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--systems", default="closed_book,dense,oracle,graph_qbranch,graph_qbranch_noq,graph_openie")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

qs, ps = data.load(a.dataset, a.n or None)
run_dir = os.path.join(llm.ROOT, "runs", "%s_n%s" % (a.dataset, a.n or "all"))
ans_dir = os.path.join(run_dir, "answers")
os.makedirs(ans_dir, exist_ok=True)
by_pid = {p["pid"]: p for p in ps}
qvecs = answer.embed_queries([q["question"] for q in qs])

dense_idx = graphs = None
retrievers = {}


def get_retriever(system):
    if system not in retrievers:
        variant = "openie" if system == "graph_openie" else "qbranch"
        g = Graph(build.load_records(os.path.join(run_dir, "build_" + variant)), variant)
        print(system, json.dumps({k: v for k, v in g.stats().items() if k != "top_relations"}), flush=True)
        retrievers[system] = answer.GraphRetriever(g, ps, use_questions=(system == "graph_qbranch"), n_passages=a.k)
    return retrievers[system]


def evidence_for(system, i):
    if system == "closed_book":
        return None, []
    if system == "oracle":
        gp = [by_pid[pid] for pid in qs[i]["gold_pids"]]
        return answer.fmt_passages(gp), gp
    if system == "dense":
        global dense_idx
        if dense_idx is None:
            dense_idx = answer.DenseIndex([answer.passage_text(p) for p in ps])
        top = np.argsort(-dense_idx.scores(qvecs[i]))[:a.k]
        sel = [ps[j] for j in top]
        return answer.fmt_passages(sel), sel
    r = get_retriever(system)
    facts, sel = r.retrieve(qvecs[i])
    return r.evidence(facts, sel), sel


summary = {}
for system in a.systems.split(","):
    if system not in ("closed_book", "dense", "oracle"):
        get_retriever(system)
    elif system == "dense":
        evidence_for("dense", 0)

    def one(i):
        q = qs[i]
        ev, sel = evidence_for(system, i)
        pred, r = answer.read(q["question"], ev)
        em, f1 = evaluate.em_f1(pred, q["aliases"])
        ok = evaluate.judge(q["question"], q["answer"], pred)
        return {"qid": q["qid"], "question": q["question"], "gold": q["answer"], "pred": pred, "em": em, "f1": f1,
                "judge": ok, "recall": evaluate.recall(q["gold_pids"], [p["pid"] for p in sel]),
                "pids": [p["pid"] for p in sel], "titles": [p["title"] for p in sel],
                "reader_usage": r["usage"], "evidence_chars": len(ev or "")}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, range(len(qs))))
    with open(os.path.join(ans_dir, system + ".jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    summary[system] = {"n": len(rows), "em": np.mean([r["em"] for r in rows]), "f1": np.mean([r["f1"] for r in rows]),
                       "judge": np.mean([r["judge"] for r in rows]), "recall@%d" % a.k: np.mean(rec) if rec else None,
                       "reader_prompt_tokens": np.mean([r["reader_usage"].get("prompt", 0) for r in rows])}
    print(system, json.dumps({k: (round(float(v), 3) if v is not None else None) for k, v in summary[system].items()}),
          flush=True)

json.dump({"summary": summary, "usage": llm.usage_report()}, open(os.path.join(run_dir, "summary.json"), "w"),
          indent=1, default=float)
print(json.dumps(llm.usage_report()["total_cost_usd"]))
