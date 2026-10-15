"""Score the proposed operators on all CLEVRER validation predictive and counterfactual questions.

No LLM. Each question is answered by graph operations only:
  - "observed graph": causal links over the observed events (the 4.1 ceiling in the note).
  - "+ intervention + world simulation": src/world.py.
Constants are tuned on videos 10000-11999 and reported on the held-out videos 12000-14999.

Usage: uv run python analysis/eval_operators.py
"""
import collections
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import clevrer  # noqa: E402
import world  # noqa: E402
from clevrer_observed_graph import choice_kind, obj_specs, resolve  # noqa: E402

TUNE = range(10000, 12000)
TEST = range(12000, 15000)


def answer(graph, run, q, use_sim):
    """Predicted 'correct' flag for each choice of one question."""
    objs = list(run.objects.values())
    prog = q["program"]
    negate = prog[-1] == "negate"
    kinds = [choice_kind(c["program"], objs) for c in q["choices"]]
    pairs = [k[1] if k and k[0] == "coll" else None for k in kinds]
    if q["question_type"] == "predictive":
        if use_sim:
            order = {pair: i for i, (_, pair) in enumerate(graph.predict_future())}
            ranked = [order.get(p, 10 ** 6) for p in pairs]
            if min(ranked) < 10 ** 6:
                best = ranked.index(min(ranked))
                return [i == best for i in range(len(pairs))]
        return [p is not None and p not in graph.observed for p in pairs]
    x = resolve(obj_specs(prog)[0], objs)
    if use_sim:
        happens = graph.counterfactual_events(x)
        flags = [p in happens for p in pairs]
    else:
        flags = []
        for p in pairs:
            if p in graph.observed:
                flags.append(x not in graph.causal_past(p, graph.observed[p])[0])
            else:
                flags.append(False)
    return [not f for f in flags] if negate else flags


def evaluate(sids, contact, horizon, use_sim):
    score = collections.Counter()
    qs = clevrer.all_questions()
    for sid in sids:
        run = clevrer.Run(sid)
        graph = world.SceneGraph(run, contact=contact, horizon=horizon)
        for q in qs[sid]:
            t = q["question_type"]
            if t not in ("predictive", "counterfactual"):
                continue
            pred = answer(graph, run, q, use_sim)
            gold = [c["answer"] == "correct" for c in q["choices"]]
            hits = sum(p == g for p, g in zip(pred, gold))
            score[(t, "opt")] += hits
            score[(t, "opt_n")] += len(gold)
            score[(t, "q")] += hits == len(gold)
            score[(t, "q_n")] += 1
    return {t: (100 * score[(t, "opt")] / score[(t, "opt_n")], 100 * score[(t, "q")] / score[(t, "q_n")], score[(t, "q_n")])
            for t in ("predictive", "counterfactual")}


if __name__ == "__main__":
    tune_ids = [s for s in TUNE if s % 4 == 0]  # every 4th tuning video keeps the grid search fast
    best = None
    for contact, horizon in itertools.product((0.42, 0.45, 0.5), (40, 51, 64, 80)):
        r = evaluate(tune_ids, contact, horizon, True)
        key = r["predictive"][1] + r["counterfactual"][1]
        print("tune contact=%.2f horizon=%d -> predictive %.1f/%.1f counterfactual %.1f/%.1f" % (
            contact, horizon, *r["predictive"][:2], *r["counterfactual"][:2]))
        if best is None or key > best[0]:
            best = (key, contact, horizon)
    _, contact, horizon = best
    print("chosen: contact=%.2f horizon=%d" % (contact, horizon))
    base = evaluate(TEST, contact, horizon, False)
    sim = evaluate(TEST, contact, horizon, True)
    result = {"contact": contact, "horizon": horizon, "test_videos": [TEST.start, TEST.stop - 1],
              "observed_graph": base, "with_operators": sim}
    for name, r in (("observed graph", base), ("+ intervention + world simulation", sim)):
        print("%-36s predictive %.1f / %.1f (n=%d)   counterfactual %.1f / %.1f (n=%d)" % (
            name, *r["predictive"], *r["counterfactual"]))
    out = os.path.join(HERE, "..", "runs", "operators_eval.json")
    json.dump(result, open(out, "w"), indent=1)
    print("wrote", out)
