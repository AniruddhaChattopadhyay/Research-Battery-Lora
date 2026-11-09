"""C2 with a semantic check: do later documents answer the open questions that a passage raised?

The exact test (anticipation.py) needs the same relation name, so it misses bindings with a different
name (born_in vs birthplace). Here an LLM from another model family checks the binding instead.

For each sampled open slot (entity e, open question q, raised in batch t):
  later   = facts about e (as subject or object) from other passages in batches > t
  known   = facts about e from other passages in batches <= t
  pred    = does a fact in `later` answer q?
  random  = does a fact in `later` answer q_rand? q_rand is the open question of a random slot whose
            entity has the same type, with the name of e put in. Same entity, same later facts, but a
            question that the passage did not predict. pred - random is the value of the prediction.
  ceiling = `later` is not empty (some later fact about e exists)
  memory  = does a fact in `known` answer q? (the agent should check memory before it opens a slot)
The check sees the 30 facts nearest to the question (embedding similarity), for pred and for random.
The ceiling and the memory ceiling use all slots (no LLM). The LLM checks use a sample of the slots that
have later facts (pred, random) or earlier facts (memory). Rate over all slots = ceiling x sample rate.

Usage: uv run python analysis/c2_semantic.py runs/musique_n100 [--n 400]
Output: <run_dir>/c2_semantic.json
"""
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import answer  # noqa: E402
import build  # noqa: E402
import llm  # noqa: E402
from graph import Graph  # noqa: E402

CHECK_MODEL = "anthropic/claude-haiku-5.5"
TOP = 30
CHECK = """Facts from a knowledge graph:
{facts}

Question: {question}

Does one of the facts give the answer to the question? The fact must be about the same entity and must give the specific value that the question asks for. A related fact that does not give that value is not an answer.
Return only a JSON object: {{"answered": true or false, "fact": <the number of the fact, or null>, "answer": "<short answer or empty>"}}"""

run_dir = sys.argv[1]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 600
g = Graph(build.load_records(os.path.join(run_dir, "build_qbranch")), "qbranch")
F = answer.DenseIndex([g.fact_text(f) for f in g.facts])
about = collections.defaultdict(list)
for f in g.facts:
    about[f["s_norm"]].append(f)
    if f["o_norm"] != f["s_norm"]:
        about[f["o_norm"]].append(f)

last = max(s["batch"] for s in g.open_slots)
slots = [s for s in g.open_slots if s["batch"] < last and s["iqid"] is not None and s["s_norm"]]
for s in slots:
    s["q"] = g.agenda[s["iqid"]]["q"]
rng = random.Random(0)

# Pool for the random question: slots whose question names their entity, by entity type.
by_type = collections.defaultdict(list)
for s in slots:
    if s["s"] and s["s"].lower() in s["q"].lower():
        by_type[str(s.get("s_type") or "").lower()].append(s)
all_named = [s for v in by_type.values() for s in v]


def random_question(slot):
    pool = by_type.get(str(slot.get("s_type") or "").lower()) or all_named
    for _ in range(50):
        o = rng.choice(pool)
        if o["r"] != slot["r"] and o["s_norm"] != slot["s_norm"]:
            return re.sub(re.escape(o["s"]), slot["s"].replace("\\", ""), o["q"], flags=re.I)
    return None


def nearest(facts, question):
    if not facts:
        return []
    qv = answer.embed_queries([question])[0]
    sc = F.M[[f["fid"] for f in facts]] @ qv
    return [facts[i] for i in np.argsort(-sc)[:TOP]]


def check(facts, question):
    if not facts or not question:
        return None
    lines = "\n".join("%d. %s | %s | %s" % (i + 1, f["s"], f["r"], f["o"]) for i, f in enumerate(facts))
    r = llm.chat(CHECK_MODEL, CHECK.format(facts=lines, question=question), temperature=None, max_tokens=300)
    j = llm.parse_json(r["text"]) or {}
    ok = bool(j.get("answered"))
    k = j.get("fact")
    f = facts[k - 1] if ok and isinstance(k, int) and 0 < k <= len(facts) else None
    return {"answered": ok, "answer": j.get("answer"), "fact": [f["s"], f["r"], f["o"]] if f else None,
            "fact_batch": f["batch"] if f else None, "fact_title": g.pid_title[f["pid"]] if f else None}


for s in slots:
    ent = about.get(s["s_norm"], [])
    s["later"] = [f for f in ent if f["batch"] > s["batch"] and f["pid"] != s["pid"]]
    s["known"] = [f for f in ent if f["batch"] <= s["batch"] and f["pid"] != s["pid"]]
ceiling = float(np.mean([bool(s["later"]) for s in slots]))
known_ceiling = float(np.mean([bool(s["known"]) for s in slots]))
sample_l = rng.sample([s for s in slots if s["later"]], min(N, sum(1 for s in slots if s["later"])))
sample_k = rng.sample([s for s in slots if s["known"]], min(N // 2, sum(1 for s in slots if s["known"])))


def one_later(slot):
    qr = random_question(slot)
    return {"slot": [slot["s"], slot["r"], "?"], "q": slot["q"], "q_random": qr, "batch": slot["batch"],
            "title": g.pid_title[slot["pid"]], "n_later": len(slot["later"]),
            "pred": check(nearest(slot["later"], slot["q"]), slot["q"]),
            "random": check(nearest(slot["later"], qr), qr) if qr else None}


def one_known(slot):
    return {"slot": [slot["s"], slot["r"], "?"], "q": slot["q"], "batch": slot["batch"],
            "memory": check(nearest(slot["known"], slot["q"]), slot["q"])}


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(one_later, sample_l))
    mem_rows = list(ex.map(one_known, sample_k))


def rate(key, rows_):
    return float(np.mean([bool(r[key] and r[key]["answered"]) for r in rows_])) if rows_ else None


paired = [r for r in rows if r["q_random"]]
gaps = [r["pred"]["fact_batch"] - r["batch"] for r in rows if r["pred"] and r["pred"]["answered"]
        and r["pred"]["fact_batch"] is not None]
p_rate, r_rate, m_rate = rate("pred", paired), rate("random", paired), rate("memory", mem_rows)
out = {"run_dir": run_dir, "open_slots": len(g.open_slots), "slots_with_a_later_batch": len(slots),
       "batches": last + 1, "ceiling_some_later_fact": ceiling, "memory_ceiling_some_earlier_fact": known_ceiling,
       "sample_with_later_facts": len(rows), "n_paired": len(paired),
       "given_later_facts": {"pred": p_rate, "random": r_rate},
       "all_slots": {"pred": ceiling * p_rate if p_rate is not None else None,
                     "random": ceiling * r_rate if r_rate is not None else None,
                     "already_in_memory": known_ceiling * m_rate if m_rate is not None else None},
       "sample_with_earlier_facts": len(mem_rows), "memory_given_earlier_facts": m_rate,
       "batch_gap_mean": float(np.mean(gaps)) if gaps else None,
       "top_bound_relations": collections.Counter(r["slot"][1] for r in rows if r["pred"] and r["pred"]["answered"])
       .most_common(10),
       "cost_usd": llm.usage_report()["total_cost_usd"]}
# Paired bootstrap for pred - random on the paired slots.
d = np.array([float(bool(r["pred"] and r["pred"]["answered"])) - float(bool(r["random"] and r["random"]["answered"]))
              for r in paired])
bs = np.random.default_rng(0)
boots = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(5000)] if len(d) else [0]
out["pred_minus_random"] = [float(d.mean()) if len(d) else None, float(np.percentile(boots, 2.5)),
                            float(np.percentile(boots, 97.5))]
json.dump({**out, "rows": rows, "memory_rows": mem_rows}, open(os.path.join(run_dir, "c2_semantic.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
