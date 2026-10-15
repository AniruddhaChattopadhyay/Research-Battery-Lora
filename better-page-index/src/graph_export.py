"""Export the dual-view graph of each pilot run, and the exact context of each graph-agent question.

The graph is made explicit here (nodes and edges) for viewing. The agent itself queries the
same data through the six tools in run_graph_agent.py.

  Data view:    object nodes; state segments (moving or stationary, in view or not) over time.
  Process view: event nodes (entrances, exits, collisions); a causal edge goes from the previous
                event of an object to its next event. The ancestors of an event under these
                edges are exactly what the causal_past tool returns.
  Operators:    the simulated future, and the world without each object.

Usage: PYTHONPATH=src uv run python src/graph_export.py results/graph_data.json
"""
import json
import os
import sys

import clevrer
import qa
import run_graph_agent as agent


def segments(run, oid):
    out = []
    for f in range(clevrer.LAST_FRAME + 1):
        s = run.states[f][oid]
        key = ("moving" if s["moving"] else "stationary", s["in_view"])
        if out and (out[-1]["state"], out[-1]["in_view"]) == key:
            out[-1]["end"] = f
        else:
            out.append({"obj": oid, "start": f, "end": f, "state": key[0], "in_view": key[1]})
    return out


def process_view(run):
    """Event nodes and causal edges (previous event of each object -> next event)."""
    events = []
    for k, e in enumerate(run.events()):
        events.append({"id": "e%d" % k, "kind": e["kind"], "frame": e["frame"], "objects": e["objects"]})
    edges = []
    for e in events:
        if e["kind"] == "enter":
            continue
        for o in e["objects"]:
            prev = [p for p in events if o in p["objects"] and p["frame"] < e["frame"] and p["kind"] != "exit"]
            if prev:
                edges.append({"from": prev[-1]["id"], "to": e["id"], "obj": o})
    return events, edges


def ancestors(event_id, edges):
    parents = {}
    for ed in edges:
        parents.setdefault(ed["to"], []).append(ed["from"])
    seen, stack = set(), [event_id]
    while stack:
        for p in parents.get(stack.pop(), []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def check_against_tool(run, graph, events, edges):
    """The edge closure must give the same events as the causal_past tool."""
    by_id = {e["id"]: e for e in events}
    for e in events:
        if e["kind"] == "enter":
            continue
        _, tool_events = graph.causal_past(e["objects"], e["frame"])
        mine = set()
        for a in ancestors(e["id"], edges):
            x = by_id[a]
            mine.add(("enter", x["objects"][0]) if x["kind"] == "enter" else ("collision", frozenset(x["objects"])))
        assert mine == tool_events, (run.sid, e)


def export_run(sid, items, recs):
    graph = agent.graph(sid)
    run = graph.run
    events, edges = process_view(run)
    check_against_tool(run, graph, events, edges)
    factual = {p: f for p, f in graph.observed.items()}
    future = graph.predict_future()
    for f, p in future:
        factual.setdefault(p, f)
    counterfactual = {}
    for x in run.objects:
        cf = graph.counterfactual_frames(x)
        happen = []
        for pair, f in sorted(cf.items(), key=lambda kv: kv[1]):
            if pair in graph.observed and graph.observed[pair] == f:
                status = "kept"
            elif pair in factual:
                status = "kept (future)"
            else:
                status = "new"
            happen.append({"objects": sorted(pair), "frame": f, "status": status})
        prevented = [{"objects": sorted(p), "frame": f} for p, f in factual.items() if p not in cf]
        counterfactual[str(x)] = {"happen": happen, "prevented": prevented}
    questions = []
    for it in [i for i in items if i["sid"] == sid]:
        key = (sid, it["qid"])
        rec = recs["graph"][key]
        q = it["q"]
        transcript = [{"name": c["name"], "args": c["args"], "result": agent.IMPL[c["name"]](**c["args"])}
                      for c in rec["tool_calls"]]
        gold = q["answer"] if q["question_type"] == "descriptive" else \
            {str(c["choice_id"]): c["answer"] == "correct" for c in q["choices"]}
        questions.append({
            "qid": it["qid"], "type": it["type"], "text": qa.question_text(it), "gold": gold,
            "answer": rec["answer"], "correct": {k: bool(recs[k][key]["correct"]) for k in recs},
            "tokens": rec["usage"], "transcript": transcript})
    return {
        "sid": sid, "end": clevrer.LAST_FRAME, "horizon": graph.horizon,
        "objects": [{"id": i, "name": run.name(i), **{k: o[k] for k in ("color", "material", "shape")}}
                    for i, o in sorted(run.objects.items())],
        "segments": [s for i in sorted(run.objects) for s in segments(run, i)],
        "states": {str(i): [[round(run.states[f][i]["x"], 2), round(run.states[f][i]["y"], 2),
                             round(run.states[f][i]["speed"], 2), int(run.states[f][i]["moving"]),
                             int(run.states[f][i]["in_view"])] for f in range(clevrer.LAST_FRAME + 1)]
                   for i in sorted(run.objects)},
        "in_view_at_start": sorted(run.in_view_at_start),
        "events": events, "causal_edges": edges,
        "future": [{"frame": f, "objects": sorted(p)} for f, p in future],
        "counterfactual": counterfactual, "questions": questions,
    }


def main(out):
    items = qa.pilot_questions(qa.pilot_runs())
    pilot = os.path.join(clevrer.ROOT, "runs", "pilot")
    recs = {name: {(r["sid"], r["qid"]): r for r in map(json.loads, open(os.path.join(pilot, f)))}
            for name, f in (("graph", "graph_agent.jsonl"), ("pageindex", "pageindex.jsonl"),
                            ("fullcontext", "fullcontext.jsonl"))}
    data = {"system_prompt": agent.SYSTEM, "tools": agent.TOOLS,
            "runs": [export_run(sid, items, recs) for sid in qa.pilot_runs()]}
    json.dump(data, open(out, "w"), separators=(",", ":"))
    print("wrote %s: %d runs, %d questions, %.0f KB" % (
        out, len(data["runs"]), sum(len(r["questions"]) for r in data["runs"]), os.path.getsize(out) / 1024))


if __name__ == "__main__":
    main(sys.argv[1])
