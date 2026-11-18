"""Goal 2 of the open slots (Prof. Roy, 2026-11-13): do the dataset questions need the facts that the open
slots ask for?

Goal 1 (C2) asks if later documents fill the open slots. Goal 2 asks if the questions of the users need
those facts. This script measures goal 2 with the gold hops of the benchmark questions:
  MuSiQue: the sub-questions of question_decomposition (#k is replaced by the answer of step k), each with
           its answer and its support passage.
  2Wiki:   the evidence triples (s, r, o), as "What is the r of s?" with the answer o. The support passage
           is the passage with the title s (about 90% of the triples).

For each gold hop, the 10 open slots with the nearest question text (embedding similarity) are candidates.
Slots that come from the support passage of the hop do not count. A checker (Claude Haiku 5.5) decides which
candidates ask for the same fact about the same entity.
  asked        some open slot asks for the fact of the hop
  asked_before some open slot asks for it, and the slot opened in a batch before the support passage arrived
               (the slot was open, and the support passage then filled it: goal 1 and goal 2 together)
Question level: a question counts when at least one of its hops counts.
Slot level: the share of all open slots that at least one hop matched (do the slots ask for needed facts?).
Only the evaluation questions count (not the example pool of the variant qbranch_qa).

Usage: uv run python analysis/goal2.py runs/musique_nall --variants qbranch,qbranch_qa
Output: <run_dir>/goal2.json
"""
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import answer  # noqa: E402
import build  # noqa: E402
import data  # noqa: E402
import llm  # noqa: E402
from graph import Graph  # noqa: E402

CHECK_MODEL = "anthropic/claude-haiku-5.5"
TOP = 10
CHECK = """A user asked this question. The answer is given.
Question: {question}
Answer: {answer}

A knowledge graph keeps these open questions. It does not know their answers yet.
{slots}

Which open questions ask for the same fact as the user question, about the same entity, so that the answer above also answers them? Small differences in wording do not matter.
Return only a JSON object: {{"same": [<numbers of the open questions>]}}"""


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


run_dir = os.path.normpath(sys.argv[1])
name = os.path.basename(run_dir).rsplit("_n", 1)[0]
variants = arg("--variants", "qbranch,qbranch_qa").split(",")
qs, ps = data.load(name, None)
pool_path = os.path.join(run_dir, "faq_pool.json")
eval_ids = set(json.load(open(pool_path))["eval_qids"]) if os.path.exists(pool_path) else {q["qid"] for q in qs[100:]}
raw = {(q.get("id") or q.get("_id")): q for q in json.load(open(os.path.join(data.DATA, name + ".json")))}
by_text = {data._key(p["text"]): p["pid"] for p in ps}
by_title = {p["title"]: p["pid"] for p in ps}


def hops_of(q):
    r = raw[q["qid"]]
    out = []
    if "question_decomposition" in r:
        answers = []
        for h in r["question_decomposition"]:
            t = re.sub(r"#(\d+)", lambda m: answers[int(m.group(1)) - 1] if int(m.group(1)) <= len(answers)
                       else m.group(0), h["question"])
            if ">>" in t:  # "Ireland >> country" means "What is the country of Ireland?"
                a, b = [x.strip() for x in t.split(">>", 1)]
                t = "What is the %s of %s?" % (b, a)
            sp = r["paragraphs"][h["paragraph_support_idx"]] if h.get("paragraph_support_idx") is not None else None
            out.append({"text": t, "answer": h["answer"],
                        "support": by_text.get(data._key(sp["paragraph_text"])) if sp else None})
            answers.append(h["answer"])
    else:
        for s, rel, o in r["evidences"]:
            out.append({"text": "What is the %s of %s?" % (rel, s), "answer": o, "support": by_title.get(s)})
    return out


evq = [q for q in qs if q["qid"] in eval_ids]
hops = [(q["qid"], h) for q in evq for h in hops_of(q)]
hvec = answer.embed_queries([h["text"] for _, h in hops])
print("%s: %d evaluation questions, %d hops, %d with a known support passage" % (
    name, len(evq), len(hops), sum(h["support"] is not None for _, h in hops)), flush=True)

result = {"run_dir": run_dir, "questions": len(evq), "hops": len(hops), "variants": {}}
per_q = {}
for v in variants:
    recs = build.load_records(os.path.join(run_dir, "build_" + v))
    batch_of = {r["pid"]: r["batch"] for r in recs}
    g = Graph(recs, v)
    slots = [s for s in g.open_slots if s["iqid"] is not None]
    for s in slots:
        s["q"] = g.agenda[s["iqid"]]["q"]
    S = answer.DenseIndex([s["q"] for s in slots])

    def one(i):
        qid, h = hops[i]
        sc = S.scores(hvec[i])
        cand = [slots[j] for j in np.argsort(-sc)[:TOP + 5] if slots[j]["pid"] != h["support"]][:TOP]
        lines = "\n".join("%d. %s" % (k + 1, c["q"]) for k, c in enumerate(cand))
        r = llm.chat(CHECK_MODEL, CHECK.format(question=h["text"], answer=h["answer"], slots=lines),
                     temperature=None, max_tokens=200)
        j = llm.parse_json(r["text"]) or {}
        same = [cand[k - 1] for k in (j.get("same") or []) if isinstance(k, int) and 0 < k <= len(cand)]
        sb = batch_of.get(h["support"])
        before = [c for c in same if sb is not None and c["batch"] < sb]
        return {"qid": qid, "hop": h["text"], "answer": h["answer"], "support_batch": sb,
                "asked": bool(same), "asked_before": bool(before),
                "slots": [[c["q"], c["batch"], g.pid_title[c["pid"]]] for c in same],
                "slot_ids": [id(c) for c in same]}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, range(len(hops))))
    matched = {sid for r in rows for sid in r.pop("slot_ids")}
    qa = collections.defaultdict(lambda: [False, False])
    for r in rows:
        qa[r["qid"]][0] |= r["asked"]
        qa[r["qid"]][1] |= r["asked_before"]
    per_q[v] = qa
    known = [r for r in rows if r["support_batch"] is not None]
    result["variants"][v] = {
        "open_slots": len(g.open_slots), "passages": len(recs),
        "open_slots_per_passage": len(g.open_slots) / max(1, len(recs)),
        "hops_asked": float(np.mean([r["asked"] for r in rows])),
        "hops_asked_before": float(np.mean([r["asked_before"] for r in known])) if known else None,
        "questions_asked": float(np.mean([qa[q["qid"]][0] for q in evq])),
        "questions_asked_before": float(np.mean([qa[q["qid"]][1] for q in evq])),
        "slots_needed_by_a_hop": len(matched) / max(1, len(slots)),
        "examples": [r for r in rows if r["asked_before"]][:12]}
    json.dump({"rows_" + v: rows}, open(os.path.join(run_dir, "goal2_rows_%s.json" % v), "w"), indent=1)
    print(v, json.dumps({k: x for k, x in result["variants"][v].items() if k != "examples"}), flush=True)

if len(variants) == 2:  # paired bootstrap over the questions: variant 2 - variant 1
    a, b = variants
    bs = np.random.default_rng(0)
    for k, name_ in ((0, "questions_asked"), (1, "questions_asked_before")):
        d = np.array([float(per_q[b][q["qid"]][k]) - float(per_q[a][q["qid"]][k]) for q in evq])
        boots = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        result["%s - %s, %s" % (b, a, name_)] = [float(d.mean()), float(np.percentile(boots, 2.5)),
                                                  float(np.percentile(boots, 97.5))]
result["cost_usd"] = llm.usage_report()["total_cost_usd"]
json.dump(result, open(os.path.join(run_dir, "goal2.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in result.items() if k != "variants"}, indent=1))
