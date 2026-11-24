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
  graph_qbranch_noopen  QBranch build without open questions + inferred-question index (ablation)
  graph_qbranch_qa  QBranch build with held-out dataset Q&A pairs in the prompt + inferred-question index
  graph_qbranch_events  QBranch build with open events in the prompt (news) + inferred-question index
  graph_qbranch_hist    QBranch build with history context + inferred-question index
  graph_qbranch_hist_qa QBranch build with history context and Q&A pairs + inferred-question index
  <system>_plan   path-planned answering (src/plan.py) on top of dense or a graph system,
                  for example dense_plan, graph_qbranch_plan. Recall is over the union of the hop passages.
  <graph>_plan_noanchor  the plan without anchored facts (ablation of the symbolic path step)
  <graph>_plan_factanchor  the anchored facts go to the reader, but they do not choose the passages (the fix)
  dense_k<k>      dense with k passages (k per hop for dense_k<k>_plan), a control for the evidence budget
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
import plan  # noqa: E402
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


def variant_of(system):
    return {"graph_openie": "openie", "graph_qbranch_noopen": "qbranch_noopen",
            "graph_qbranch_qa": "qbranch_qa", "graph_qbranch_events": "qbranch_events",
            "graph_qbranch_hist": "qbranch_hist", "graph_qbranch_hist_qa": "qbranch_hist_qa"}.get(system.split("_k")[0], "qbranch")


def get_retriever(system):
    """A suffix _k<k> (for example graph_qbranch_k10) sets the number of passages."""
    if system not in retrievers:
        variant = variant_of(system)
        same = next((r for s2, r in retrievers.items() if variant_of(s2) == variant), None)
        bdir = os.path.join(run_dir, "build_" + variant)
        assert os.path.isdir(bdir), "no build for %s: %s" % (system, bdir)
        g = same.g if same else Graph(build.load_records(bdir), variant)
        print(system, json.dumps({k: v for k, v in g.stats().items() if k != "top_relations"}), flush=True)
        retrievers[system] = answer.GraphRetriever(
            g, ps, use_questions=system.split("_k")[0] in ("graph_qbranch", "graph_qbranch_noopen", "graph_qbranch_qa",
                                                         "graph_qbranch_events", "graph_qbranch_hist",
                                                         "graph_qbranch_hist_qa"),
            n_passages=dense_k(system), share=same)
    return retrievers[system]


def evidence_for(system, i):
    if system == "closed_book":
        return None, []
    if system == "oracle":
        gp = [by_pid[pid] for pid in qs[i]["gold_pids"]]
        return answer.fmt_passages(gp), gp
    if system.startswith("dense"):
        global dense_idx
        if dense_idx is None:
            dense_idx = answer.DenseIndex([answer.passage_text(p) for p in ps])
        top = np.argsort(-dense_idx.scores(qvecs[i]))[:dense_k(system)]
        sel = [ps[j] for j in top]
        return answer.fmt_passages(sel), sel
    r = get_retriever(system)
    facts, sel = r.retrieve(qvecs[i])
    return r.evidence(facts, sel), sel


def dense_k(system):
    """dense_k8 means dense retrieval with 8 passages (a control at a larger evidence budget)."""
    return int(system.split("_k")[1]) if "_k" in system else a.k


def plan_one(base, q, use_anchors=True, anchor_mode="rank"):
    if base.startswith("dense"):
        def hop(subq, refs):
            top = np.argsort(-dense_idx.scores(answer.embed_queries([subq])[0]))[:dense_k(base)]
            return [], [ps[j] for j in top]
        fmt = lambda facts, sel: answer.fmt_passages(sel)  # noqa: E731
    else:
        r = get_retriever(base)

        def hop(subq, refs):
            anchors = [n for n in (r.resolve(x) for x in refs) if n] if use_anchors else []
            return r.retrieve(answer.embed_queries([subq])[0], anchors=anchors, anchor_mode=anchor_mode)
        fmt = r.evidence
    return plan.run(q["question"], hop, fmt)


sum_path = os.path.join(run_dir, "summary.json")
summary = json.load(open(sum_path))["summary"] if os.path.exists(sum_path) else {}
for system in a.systems.split(","):
    is_plan = "_plan" in system  # <base>_plan, <base>_plan_noanchor or <base>_plan_factanchor
    base = system.split("_plan")[0]
    if base not in ("closed_book", "oracle") and not base.startswith("dense"):
        get_retriever(base)
        if is_plan:
            get_retriever(base).resolve("x")  # build the entity index before the threads start
    elif base.startswith("dense"):
        evidence_for("dense", 0)

    def one(i):
        q = qs[i]
        trace = None
        if is_plan:
            pred, trace, sel, usages, ev = plan_one(base, q, use_anchors=not system.endswith("_noanchor"),
                                                    anchor_mode="facts" if system.endswith("_factanchor") else "rank")
            usage = {"prompt": sum(u.get("prompt", 0) for u in usages), "calls": len(usages)}
        else:
            ev, sel = evidence_for(system, i)
            pred, r = answer.read(q["question"], ev)
            usage = r["usage"]
        em, f1 = evaluate.em_f1(pred, q["aliases"])
        ok = evaluate.judge(q["question"], q["answer"], pred)
        return {"qid": q["qid"], "question": q["question"], "gold": q["answer"], "pred": pred, "em": em, "f1": f1,
                "judge": ok, "recall": evaluate.recall(q["gold_pids"], [p["pid"] for p in sel]),
                "pids": [p["pid"] for p in sel], "titles": [p["title"] for p in sel],
                "reader_usage": usage, "evidence_chars": len(ev or ""), "trace": trace}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, range(len(qs))))
    with open(os.path.join(ans_dir, system + ".jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    summary[system] = {"n": len(rows), "em": np.mean([r["em"] for r in rows]), "f1": np.mean([r["f1"] for r in rows]),
                       "judge": np.mean([r["judge"] for r in rows]), "recall@%d" % a.k: np.mean(rec) if rec else None,
                       "reader_prompt_tokens": np.mean([r["reader_usage"].get("prompt", 0) for r in rows]),
                       "passages": np.mean([len(r["pids"]) for r in rows]),
                       "reader_calls": np.mean([r["reader_usage"].get("calls", 1) for r in rows])}
    print(system, json.dumps({k: (round(float(v), 3) if v is not None else None) for k, v in summary[system].items()}),
          flush=True)

mine = {s: v for s, v in summary.items() if s in a.systems.split(",")}
summary = json.load(open(sum_path))["summary"] if os.path.exists(sum_path) else {}  # other runs can write too
summary.update(mine)
json.dump({"summary": summary, "usage": llm.usage_report()}, open(sum_path, "w"), indent=1, default=float)
print(json.dumps(llm.usage_report()["total_cost_usd"]))
