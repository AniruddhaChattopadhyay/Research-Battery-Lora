"""Path-planned answering (figure panel c, step 2).

1. A planner splits the question into a chain of sub-questions ("Who directed X?", "Where was #1 born?").
2. Each hop puts the earlier answers into its sub-question, retrieves evidence and reads a short answer.
   On a graph, the entity that an earlier hop found is an anchor: the hop always keeps its best facts.
   This is the symbolic path step. Dense retrieval gets only the rewritten sub-question.
3. A final reader answers the original question with the chain and all the evidence of the hops.

The plan depends only on the question, so all systems get the same plan (the call is cached).
"""
import re

import answer
import llm
import prompts

MAX_STEPS = 4


def make_plan(question):
    r = llm.chat(answer.READER_MODEL, prompts.PLAN.format(question=question), json_mode=True,
                 **answer.READER_KW)
    j = llm.parse_json(r["text"])
    steps = [s.strip() for s in (j or {}).get("steps", []) if isinstance(s, str) and s.strip()][:MAX_STEPS] \
        if isinstance(j, dict) else []
    return steps or [question], r["usage"]


def _refs(step, answers):
    return [answers[int(x) - 1] for x in re.findall(r"#(\d+)", step) if 0 < int(x) <= len(answers)]


def _fill(step, answers):
    return re.sub(r"#(\d+)", lambda m: answers[int(m.group(1)) - 1]
                  if 0 < int(m.group(1)) <= len(answers) else m.group(0), step)


def run(question, hop_fn, format_fn):
    """hop_fn(sub_question, earlier_answers) -> (facts, passages).
    format_fn(facts, passages) -> evidence text.
    Return (prediction, trace, passages, usages)."""
    steps, u0 = make_plan(question)
    usages, answers, trace = [u0], [], []
    all_facts, all_ps = {}, {}
    for step in steps:
        subq = _fill(step, answers)
        facts, ps = hop_fn(subq, _refs(step, answers))
        a, r = answer.read(subq, format_fn(facts, ps))
        usages.append(r["usage"])
        answers.append(a)
        trace.append({"step": step, "subq": subq, "answer": a, "pids": [p["pid"] for p in ps]})
        for f in facts:
            all_facts.setdefault(f["fid"], f)
        for p in ps:
            all_ps.setdefault(p["pid"], p)
    chain = "\n".join("%d. %s -> %s" % (i + 1, t["subq"], t["answer"]) for i, t in enumerate(trace))
    ev = ("Sub-questions and the answers found for them (they can be wrong):\n" + chain + "\n\n" +
          format_fn(list(all_facts.values()), list(all_ps.values())))
    pred, r = answer.read(question, ev)
    usages.append(r["usage"])
    return pred, trace, list(all_ps.values()), usages, ev
