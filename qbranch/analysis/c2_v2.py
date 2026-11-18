"""C2, version 2: do later documents answer the open questions that a passage raised? (three controls)

Same test as c2_semantic.py (keep that file: the 2026-11-10 numbers come from it), with these changes:
1. Version 1 had a bug: the open slots did not keep the subject type, so its "random question of the same
   entity type" was a random open question of any entity. It often did not fit the entity ("When did
   Municipal Stadium die?"), so the control was too easy. Version 2 reports three controls:
     random_any   a random open question of any other entity (the version 1 control)
     random_type  a random open question of another entity of the same type
     random_fit   as random_type, but the checker first confirms that the question fits the entity
                  (a true answer can exist). This is the strict control: only "predicted or not" changes.
   Each control puts the name of e into the question and sees the same later facts as the prediction.
2. The random draws happen before the threads start, so a rerun gives the same draws.
3. Leak check for the variant qbranch_qa: a row is "pool" when the entity of the slot is named in an example
   question or answer of faq_pool.json. The output gives the rates with and without these rows.
4. The output goes to <run_dir>/c2_<variant>.json.

Usage: uv run python analysis/c2_v2.py runs/musique_nall [--variant qbranch_qa] [--n 600]
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
N_CAND = 8
CHECK = """Facts from a knowledge graph:
{facts}

Question: {question}

Does one of the facts give the answer to the question? The fact must be about the same entity and must give the specific value that the question asks for. A related fact that does not give that value is not an answer.
Return only a JSON object: {{"answered": true or false, "fact": <the number of the fact, or null>, "answer": "<short answer or empty>"}}"""
FIT = """Entity: {entity}
Question: {question}

Can a person ask this question about this entity, so that a true answer can exist? For example, a city has no birth date, a film has no spouse, and a person has no population. You do not need to know the answer.
Reply with one word: yes or no."""


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


run_dir = sys.argv[1]
VARIANT = arg("--variant", "qbranch")
N = int(arg("--n", 600))
g = Graph(build.load_records(os.path.join(run_dir, "build_" + VARIANT)), VARIANT)
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


def tkey(s):
    return str(s.get("s_type") or "").strip().lower()


by_type = collections.defaultdict(list)  # slots whose question names their entity, by entity type
for s in slots:
    if s["s"] and s["s"].lower() in s["q"].lower():
        by_type[tkey(s)].append(s)
all_named = [s for v in by_type.values() for s in v]


def put_name(o, slot):
    return re.sub(re.escape(o["s"]), slot["s"].replace("\\", ""), o["q"], flags=re.I)


def candidates(slot, pool, n):
    out = []
    for _ in range(50 * n):
        if len(out) == n:
            break
        o = rng.choice(pool)
        if o["r"] != slot["r"] and o["s_norm"] != slot["s_norm"]:
            out.append(put_name(o, slot))
    return out


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


def fits(entity, question):
    t = llm.chat(CHECK_MODEL, FIT.format(entity=entity, question=question), temperature=None, max_tokens=10)["text"]
    return t.strip().lower().startswith("yes")


for s in slots:
    ent = about.get(s["s_norm"], [])
    s["later"] = [f for f in ent if f["batch"] > s["batch"] and f["pid"] != s["pid"]]
    s["known"] = [f for f in ent if f["batch"] <= s["batch"] and f["pid"] != s["pid"]]
ceiling = float(np.mean([bool(s["later"]) for s in slots]))
known_ceiling = float(np.mean([bool(s["known"]) for s in slots]))
sample_l = rng.sample([s for s in slots if s["later"]], min(N, sum(1 for s in slots if s["later"])))
sample_k = rng.sample([s for s in slots if s["known"]], min(N // 2, sum(1 for s in slots if s["known"])))
for s in sample_l:  # all random draws here, in one thread
    s["c_any"] = candidates(s, all_named, 1)
    s["c_type"] = candidates(s, by_type.get(tkey(s)) or all_named, N_CAND)

pool_path = os.path.join(run_dir, "faq_pool.json")
pool_text = ""
if os.path.exists(pool_path):
    pool_text = " ".join("%s %s" % (q["question"], q["answer"]) for q in json.load(open(pool_path))["pool"]).lower()


def in_pool(slot):
    return bool(pool_text) and re.search(r"\b%s\b" % re.escape(slot["s_norm"]), pool_text) is not None


def one_later(slot):
    q_any = slot["c_any"][0] if slot["c_any"] else None
    q_type = slot["c_type"][0] if slot["c_type"] else None
    q_fit = next((c for c in slot["c_type"] if fits(slot["s"], c)), None)
    return {"slot": [slot["s"], slot["r"], "?"], "s_type": slot.get("s_type"), "q": slot["q"],
            "batch": slot["batch"], "title": g.pid_title[slot["pid"]], "n_later": len(slot["later"]),
            "pool_entity": in_pool(slot), "q_any": q_any, "q_type": q_type, "q_fit": q_fit,
            "pred": check(nearest(slot["later"], slot["q"]), slot["q"]),
            "random_any": check(nearest(slot["later"], q_any), q_any) if q_any else None,
            "random_type": check(nearest(slot["later"], q_type), q_type) if q_type else None,
            "random_fit": check(nearest(slot["later"], q_fit), q_fit) if q_fit else None}


def one_known(slot):
    return {"slot": [slot["s"], slot["r"], "?"], "q": slot["q"], "batch": slot["batch"],
            "memory": check(nearest(slot["known"], slot["q"]), slot["q"])}


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(one_later, sample_l))
    mem_rows = list(ex.map(one_known, sample_k))


def hit(r, key):
    return float(bool(r.get(key) and r[key]["answered"]))


def compare(rows_, ctrl):
    """pred - control on the rows that have a control question; paired bootstrap, 5,000 draws."""
    rs = [r for r in rows_ if r["q_" + ctrl.split("_")[1]]]
    if not rs:
        return None
    d = np.array([hit(r, "pred") - hit(r, ctrl) for r in rs])
    bs = np.random.default_rng(0)
    boots = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return {"n": len(rs), "pred": float(np.mean([hit(r, "pred") for r in rs])),
            "control": float(np.mean([hit(r, ctrl) for r in rs])),
            "diff": [float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def block(rows_):
    return {c: compare(rows_, c) for c in ("random_any", "random_type", "random_fit")}


m_rate = float(np.mean([hit(r, "memory") for r in mem_rows])) if mem_rows else None
gaps = [r["pred"]["fact_batch"] - r["batch"] for r in rows if r["pred"] and r["pred"]["answered"]
        and r["pred"]["fact_batch"] is not None]
out = {"run_dir": run_dir, "variant": VARIANT, "open_slots": len(g.open_slots), "passages": len(g.pid_title),
       "open_slots_per_passage": len(g.open_slots) / max(1, len(g.pid_title)),
       "slots_with_a_later_batch": len(slots), "batches": last + 1,
       "ceiling_some_later_fact": ceiling, "memory_ceiling_some_earlier_fact": known_ceiling,
       "sample_with_later_facts": len(rows),
       "fit_found": float(np.mean([bool(r["q_fit"]) for r in rows])) if rows else None,
       "all_rows": block(rows),
       "pool_rows": sum(r["pool_entity"] for r in rows),
       "rows_without_pool_entity": block([r for r in rows if not r["pool_entity"]]) if pool_text else None,
       "pred_over_all_slots": ceiling * float(np.mean([hit(r, "pred") for r in rows])) if rows else None,
       "already_in_memory_over_all_slots": known_ceiling * m_rate if m_rate is not None else None,
       "memory_given_earlier_facts": m_rate, "batch_gap_mean": float(np.mean(gaps)) if gaps else None,
       "cost_usd": llm.usage_report()["total_cost_usd"]}
json.dump({**out, "rows": rows, "memory_rows": mem_rows}, open(os.path.join(run_dir, "c2_%s.json" % VARIANT), "w"),
          indent=1)
print(json.dumps(out, indent=1))
