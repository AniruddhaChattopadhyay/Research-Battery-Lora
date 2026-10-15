"""Pilot question set, shared task instructions, answer parsing and CLEVRER scoring.

Every system gets the same instructions and the same question text. Only the way that the
system finds the evidence is different.
"""
import json
import random
import re

import clevrer

TASK_INSTRUCTIONS = """You answer questions about recorded runs of a simple physics scene. Objects slide on a floor, enter and exit the camera view, and collide. The report of each run gives the objects, an event log (entrances, exits and collisions) and the state of every object in every frame (position, velocity, speed, moving or stationary, in view or not).

Rules of this dataset:
- A run has frames 0 to 127. "When the video begins" means frame 0. "When the video ends" means frame 127.
- Only objects in view count. An object is "moving" when its speed is 0.02 or more, else it is "stationary".
- An object "enters" when it first comes into view. It "exits" when it leaves the view.
- "Responsible for" an event E means "in the causal past of E": the objects of E, their earlier collisions, the earlier collisions of those partner objects (before their collision with them), and so on, and the entrances of all these objects. "The presence of X" is responsible if X is one of these objects.
- Predictive questions ask about the events after frame 127. The run continues for about 2 seconds after the video ends. The report does not contain these events. Predictive questions have exactly one correct option.
- Counterfactual questions ask what happens in the whole run (the video and the 2 seconds after it) if one object is removed from the scene at the start.
- For a multiple-choice question, judge each option on its own. Any number of options can be correct, also none (except for predictive questions).

Answer format: end your reply with one JSON object on the last line, and write nothing after it.
- Open questions: {"answer": "<one word>"}. Use a digit for a count, "yes" or "no", or one color, material or shape word.
- Multiple-choice questions: {"choices": {"0": true, "1": false}} with one entry for each option. true means that the option is correct."""


def pilot_runs(n_runs=30, seed=7):
    sids = sorted(clevrer.all_questions())
    return sorted(random.Random(seed).sample(sids, n_runs))


def pilot_questions(sids, n_desc=4, seed=7):
    """All multiple-choice questions of each run and n_desc random descriptive questions."""
    rng = random.Random(seed)
    out = []
    for sid in sids:
        qs = clevrer.all_questions()[sid]
        desc = [q for q in qs if q["question_type"] == "descriptive"]
        chosen = rng.sample(desc, min(n_desc, len(desc))) + [q for q in qs if q["question_type"] != "descriptive"]
        for q in chosen:
            out.append({"sid": sid, "qid": q["question_id"], "type": q["question_type"], "q": q})
    return out


def question_text(item, with_run=True):
    q = item["q"]
    head = "Question about run %d: " % item["sid"] if with_run else "Question: "
    text = head + q["question"]
    if q["question_type"] != "descriptive":
        text += "\nOptions:\n" + "\n".join("%d) %s" % (c["choice_id"], c["choice"]) for c in q["choices"])
    return text


def parse_answer(text):
    """Return the last JSON object in the reply, or None."""
    if not text:
        return None
    for m in reversed(list(re.finditer(r"\{.*?\}\s*\}|\{[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def score(item, parsed):
    """Return (option_hits, option_total, question_correct) in CLEVRER terms."""
    q = item["q"]
    if q["question_type"] == "descriptive":
        ans = str((parsed or {}).get("answer", "")).strip().lower().rstrip(".")
        ok = ans == q["answer"].lower()
        return int(ok), 1, ok
    pred = (parsed or {}).get("choices", {}) or {}
    pred = {str(k): v for k, v in pred.items()}
    hits = 0
    for c in q["choices"]:
        p = pred.get(str(c["choice_id"]))
        if isinstance(p, str):
            p = p.strip().lower() in ("true", "correct", "yes")
        hits += (p is True) == (c["answer"] == "correct") if p is not None else 0
    return hits, len(q["choices"]), hits == len(q["choices"])
