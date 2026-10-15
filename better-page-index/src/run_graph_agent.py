"""Dual-view graph agent: the same chat model, with graph tools instead of a document tree.

This is a small proof of concept of the online phase in Prof. Roy's design:
  - data-graph lookups (objects, states at a frame),
  - process-graph traversal (event log, causal past of an event),
  - the two proposed operators (world simulation of the future; intervention = remove an object).
The graph comes from the same facts as the scene report. The constants come from
analysis/calibrate_moving.py and analysis/eval_operators.py.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import env  # noqa: F401  (loads the API keys from .env)
import litellm

import clevrer
import qa
import world

CHAT_MODEL = "openrouter/openai/gpt-5.6-sol"
REASONING = "medium"  # the same reasoning effort for every system
OPERATORS = json.load(open(os.path.join(clevrer.ROOT, "runs", "operators_eval.json")))
MAX_TURNS = 10


@lru_cache(maxsize=None)
def graph(run_id):
    return world.SceneGraph(clevrer.Run(int(run_id)), contact=OPERATORS["contact"], horizon=OPERATORS["horizon"])


def _names(g, ids):
    return [g.run.name(i) for i in sorted(ids)]


def t_list_objects(run_id):
    g = graph(run_id)
    return [{"object_id": i, "name": g.run.name(i)} for i in sorted(g.run.objects)]


def t_event_log(run_id):
    g = graph(run_id)
    out = [{"frame": 0, "event": "in view at start", "objects": _names(g, g.run.in_view_at_start)}]
    out += [{"frame": e["frame"], "event": e["kind"], "objects": _names(g, e["objects"])} for e in g.run.events()]
    return out


def t_states_at(run_id, frame):
    g = graph(run_id)
    f = max(0, min(int(frame), clevrer.LAST_FRAME))
    return {"frame": f, "objects": [
        {"name": g.run.name(i), "x": round(s["x"], 2), "y": round(s["y"], 2), "speed": round(s["speed"], 2),
         "state": "moving" if s["moving"] else "stationary", "in_view": s["in_view"]}
        for i, s in sorted(g.run.states[f].items())]}


def t_causal_past(run_id, event_kind, object_ids):
    g = graph(run_id)
    ids = [int(i) for i in object_ids]
    if event_kind == "collision":
        pair = frozenset(ids)
        if pair not in g.observed:
            return {"error": "no observed collision between these objects"}
        frame = g.observed[pair]
    else:
        if ids[0] not in g.run.exits:
            return {"error": "this object does not exit in the video"}
        frame = g.run.exits[ids[0]]
    objs, events = g.causal_past(ids, frame)
    return {"event_frame": frame, "objects_in_causal_past": _names(g, objs),
            "events_in_causal_past": [
                {"event": k, "objects": _names(g, [v] if k == "enter" else v)} for k, v in sorted(events, key=str)]}


def t_predict_future(run_id):
    g = graph(run_id)
    return {"note": "world simulation: collisions that happen after the video ends and before the run ends, earliest first",
            "collisions": [{"frame": f, "objects": _names(g, p)} for f, p in g.predict_future()]}


def t_remove_object(run_id, object_id):
    g = graph(run_id)
    x = int(object_id)
    happen = g.counterfactual_events(x)
    factual = set(g.observed) | {p for _, p in g.predict_future()}
    return {"removed": g.run.name(x),
            "collisions_that_happen_without_it": [_names(g, p) for p in sorted(happen, key=sorted)],
            "factual_collisions_that_no_longer_happen": [_names(g, p) for p in sorted(factual - happen, key=sorted)
                                                         if x not in p]}


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


RUN = {"run_id": {"type": "integer", "description": "The run number, for example 10307."}}
TOOLS = [
    _fn("list_objects", "Data graph: the objects of a run (id and name: color, material, shape).", RUN, ["run_id"]),
    _fn("event_log", "Process graph: the observed events of a run in time order (frames 0-127): objects in view at the start, entrances, exits and collisions.", RUN, ["run_id"]),
    _fn("states_at", "Data graph: the state of every object at one frame: position, speed, moving or stationary, in view or not.",
        {**RUN, "frame": {"type": "integer"}}, ["run_id", "frame"]),
    _fn("causal_past", "Process graph: the objects and events in the causal past of an observed event (a collision between two objects, or the exit of one object). Use it for 'responsible for' questions.",
        {**RUN, "event_kind": {"type": "string", "enum": ["collision", "exit"]},
         "object_ids": {"type": "array", "items": {"type": "integer"}, "description": "Two ids for a collision, one id for an exit."}},
        ["run_id", "event_kind", "object_ids"]),
    _fn("predict_future", "Operator, world simulation: the collisions after the video ends, earliest first.", RUN, ["run_id"]),
    _fn("remove_object", "Operator, intervention plus world simulation: all collisions of the whole run (the video and the 2 seconds after it) in a world without the given object.",
        {**RUN, "object_id": {"type": "integer"}}, ["run_id", "object_id"]),
]
IMPL = {"list_objects": t_list_objects, "event_log": t_event_log, "states_at": t_states_at,
        "causal_past": t_causal_past, "predict_future": t_predict_future, "remove_object": t_remove_object}
SYSTEM = qa.TASK_INSTRUCTIONS + "\n\nYou have tools that query the scene graph of each run. The question names the run. Use the tools to get the facts that you need, then answer. The two operators (predict_future and remove_object) simulate the events that the report does not contain: use them as the evidence for predictive and counterfactual questions."


def ask_one(item, model=CHAT_MODEL):
    t0 = time.time()
    rec = {"sid": item["sid"], "qid": item["qid"], "type": item["type"], "tool_calls": [],
           "usage": {"prompt": 0, "completion": 0}}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": qa.question_text(item)}]
    answer = ""
    try:
        for _ in range(MAX_TURNS):
            r = litellm.completion(model=model, messages=msgs, tools=TOOLS, reasoning_effort=REASONING,
                                   timeout=600, num_retries=2)
            rec["usage"]["prompt"] += r.usage.prompt_tokens
            rec["usage"]["completion"] += r.usage.completion_tokens
            m = r.choices[0].message
            msgs.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                answer = m.content or ""
                break
            for tc in m.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                try:
                    out = IMPL[tc.function.name](**args)
                except Exception as ex:
                    out = {"error": "%s: %s" % (type(ex).__name__, ex)}
                rec["tool_calls"].append({"name": tc.function.name, "args": args})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
        rec["answer"] = answer
    except Exception as ex:
        rec.update(error="%s: %s" % (type(ex).__name__, ex), answer="")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["parsed"] = qa.parse_answer(rec["answer"])
    rec["hits"], rec["total"], rec["correct"] = qa.score(item, rec["parsed"])
    return rec


def main(out, limit=None, workers=8):
    items = qa.pilot_questions(qa.pilot_runs())[: limit or None]
    for sid in {i["sid"] for i in items}:
        graph(sid)
    done = set()
    if os.path.exists(out):
        done = {(r["sid"], r["qid"]) for r in map(json.loads, open(out))}
    todo = [i for i in items if (i["sid"], i["qid"]) not in done]
    print("questions to ask:", len(todo))
    with open(out, "a") as f, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(ask_one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["sid"], rec["qid"], rec["type"], "correct" if rec["correct"] else "wrong", rec["seconds"], "s",
                  [t["name"] for t in rec["tool_calls"]], rec.get("error", ""))


if __name__ == "__main__":
    main(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
