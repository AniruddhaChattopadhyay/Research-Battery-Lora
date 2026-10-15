"""What can a system that only sees the observed events of a CLEVRER video answer?

The script uses only the ground-truth annotations (no LLM). It reports two things for the
multiple-choice questions (explanatory, predictive, counterfactual):

1. Unseen events: how many events that happen in the asked world (the future, or the world
   without object X) are NOT in the observed video. No retrieval method can find these events
   in a document, because no document contains them.
2. Observed-graph ceiling: the score of a perfect causal graph over the observed events
   (entrances and collisions, with time-ordered causal links), with no physics simulation.
   - Explanatory: a choice is responsible if it is a causal ancestor of the target event.
   - Predictive: the next event is an unobserved collision (an observed one already happened).
   - Counterfactual: an observed collision survives the removal of X if X is not a causal
     ancestor of it; an unobserved collision never happens.

Usage:
    python3 -I clevrer_observed_graph.py <annotation_validation_dir> <validation.json>
"""
import collections
import json
import os
import sys

FILTERS = {"filter_color": "color", "filter_material": "material", "filter_shape": "shape"}
MC_TYPES = ("explanatory", "predictive", "counterfactual")


def obj_specs(prog):
    """Attribute filters of each 'objects ... unique' span in a CLEVRER program."""
    specs, i = [], 0
    while i < len(prog):
        if prog[i] == "objects":
            spec, j = {}, i + 1
            while j < len(prog) and prog[j] != "unique":
                if prog[j] in FILTERS:
                    spec[FILTERS[prog[j]]] = prog[j - 1]
                j += 1
            specs.append(spec)
            i = j
        i += 1
    return specs


def resolve(spec, objs):
    hits = [o["object_id"] for o in objs if all(o[k] == v for k, v in spec.items())]
    return hits[0] if len(hits) == 1 else None


def scene(ann):
    """Entrance frames, exit frames and collisions of the observed video."""
    entrance, exits = {}, {}
    for fr in ann["motion_trajectory"]:
        for o in fr["objects"]:
            oid = o["object_id"]
            if o["inside_camera_view"] and oid not in entrance:
                entrance[oid] = fr["frame_id"]
            if not o["inside_camera_view"] and oid in entrance and oid not in exits:
                exits[oid] = fr["frame_id"]
    entrance = {k: v for k, v in entrance.items() if v > 0}
    colls = [(frozenset(c["object_ids"]), c["frame_id"]) for c in ann["collision"]]
    return entrance, exits, colls


def ancestors(target_objs, t, entrance, colls):
    """Objects and events in the causal past of an event at frame t that involves target_objs."""
    anc_objs, anc_events, best = set(), set(), {}
    stack = [(o, t) for o in target_objs]
    while stack:
        o, tl = stack.pop()
        if best.get(o, -1) >= tl:
            continue
        best[o] = tl
        anc_objs.add(o)
        if o in entrance and entrance[o] < tl:
            anc_events.add(("in", o))
        for pair, ft in colls:
            if o in pair and ft < tl:
                anc_events.add(("coll", pair))
                for p in pair:
                    if p != o:
                        stack.append((p, ft))
    return anc_objs, anc_events


def choice_kind(prog, objs):
    ids = [resolve(s, objs) for s in obj_specs(prog)]
    if None in ids:
        return None
    if "filter_collision" in prog and len(ids) == 2:
        return ("coll", frozenset(ids))
    if "filter_in" in prog and len(ids) == 1:
        return ("in", ids[0])
    if prog[0] == "objects" and len(ids) == 1:
        return ("obj", ids[0])
    return None


def main(ann_dir, q_path):
    score = collections.Counter()
    for v in json.load(open(q_path)):
        sid = v["scene_index"]
        lo = (sid // 1000) * 1000
        ann = json.load(open(os.path.join(
            ann_dir, "annotation_%05d-%05d" % (lo, lo + 1000), "annotation_%05d.json" % sid)))
        objs = ann["object_property"]
        entrance, exits, colls = scene(ann)
        observed = dict(colls)
        for q in v["questions"]:
            t = q["question_type"]
            if t not in MC_TYPES:
                continue
            prog = q["program"]
            negate = prog[-1] == "negate"
            specs = obj_specs(prog)
            target = removed = None
            if t == "explanatory":
                if "filter_out" in prog:
                    x = resolve(specs[0], objs)
                    target = ancestors({x}, exits[x], entrance, colls)
                else:
                    pair = frozenset([resolve(specs[0], objs), resolve(specs[1], objs)])
                    target = ancestors(pair, observed[pair], entrance, colls)
            elif t == "counterfactual":
                removed = resolve(specs[0], objs)
            q_ok, q_has_unseen = True, False
            for c in q["choices"]:
                kind = choice_kind(c["program"], objs)
                correct = c["answer"] == "correct"
                score[(t, "label_correct")] += correct
                if kind is None:
                    score[(t, "unparsed_choice")] += 1
                    q_ok = False
                    continue
                if t != "explanatory":
                    happens = (not correct) if negate else correct
                    unseen = happens and kind[1] not in observed
                    score[(t, "events_happen")] += happens
                    score[(t, "events_happen_unseen")] += unseen
                    q_has_unseen |= unseen
                if t == "explanatory":
                    anc_objs, anc_events = target
                    yes = kind[1] in anc_objs if kind[0] == "obj" else kind in anc_events
                elif t == "predictive":
                    yes = kind[1] not in observed
                elif kind[1] in observed:
                    yes = removed not in ancestors(kind[1], observed[kind[1]], entrance, colls)[0]
                else:
                    yes = False
                ok = ((not yes) if negate else yes) == correct
                score[(t, "opt_total")] += 1
                score[(t, "opt_ok")] += ok
                q_ok &= ok
            score[(t, "q_total")] += 1
            score[(t, "q_ok")] += q_ok
            score[(t, "q_with_unseen")] += q_has_unseen

    for t in MC_TYPES:
        line = "%-14s n=%5d | observed-graph ceiling: per-option %5.1f%%, per-question %5.1f%% | label 'correct' share %5.1f%%" % (
            t, score[(t, "q_total")],
            100 * score[(t, "opt_ok")] / score[(t, "opt_total")],
            100 * score[(t, "q_ok")] / score[(t, "q_total")],
            100 * score[(t, "label_correct")] / score[(t, "opt_total")])
        if t != "explanatory":
            line += " | unseen events: %5.1f%% of events that happen, %5.1f%% of questions" % (
                100 * score[(t, "events_happen_unseen")] / score[(t, "events_happen")],
                100 * score[(t, "q_with_unseen")] / score[(t, "q_total")])
        print(line)
    if any(k[1] == "unparsed_choice" for k in score):
        print("unparsed choices:", {k: n for k, n in score.items() if k[1] == "unparsed_choice"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
