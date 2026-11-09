"""Schema normalization (figure panel a, tool "Schema Normalization").

The build step makes many relation names for one relation (born_on, born_on_date, birth_date) and
inverse names (directed vs directed_by). An LLM maps each name to a canonical relation and marks
inverse names, so the subject and the object change places. The names go to the LLM in the order of
the stream (first batch first), and the canonical list grows from chunk to chunk. So the same code
can run online, one batch at a time.
"""
import collections
import json
import os

import llm
from build import _norm_rel, triples_of

NORM_MODEL = "openai/gpt-6-luna"
NORM_KW = {"json_mode": True, "max_tokens": 16000, "reasoning": {"effort": "medium"}}

PROMPT = """You normalize the relation names of a knowledge graph. Many names mean the same relation, for example born_on, born_on_date, birth_date and born_in_year (all give the time of birth). Some names are inverse forms, for example directed_by (film -> person) and directed (person -> film).

Each canonical relation has one direction, written as "subject type -> object type". Map each new relation name to one canonical relation:
1. Use a canonical relation from the list when it has the same meaning. Compare the example triple with the direction of the canonical relation. Set "inverse": true when the subject and the object of the new name must change places to fit that direction.
2. Else make a new canonical relation: a short snake_case name, and its direction from the example triple.
Do not merge relations with different meanings (born_in, a place, and born_on, a time, are different; born_in and died_in are different).

Canonical relations so far (name: direction):
{canon}

New relation names, with the count and one example triple (subject type, object type):
{names}

Return only a JSON object: {{"map": {{"<new name>": {{"canonical": "<name>", "direction": "<subject type> -> <object type>", "inverse": false}}}}}}"""


def relation_examples(records, variant):
    first, count, example = {}, collections.Counter(), {}
    for rec in sorted(records, key=lambda r: r["batch"]):
        for tr in triples_of(rec, variant):
            r = _norm_rel(tr.get("r"))
            if not r:
                continue
            count[r] += 1
            first.setdefault(r, rec["batch"])
            if r not in example and tr.get("o"):
                example[r] = "(%s | %s | %s) (%s, %s)" % (tr.get("s"), r, tr.get("o"), tr.get("s_type"),
                                                          tr.get("o_type"))
    return first, count, example


def normalize(records, variant, out_path, chunk=120):
    if os.path.exists(out_path):
        return json.load(open(out_path))
    first, count, example = relation_examples(records, variant)
    names = sorted(count, key=lambda r: (first[r], -count[r], r))
    mapping, canon, direction = {}, collections.Counter(), {}
    for i in range(0, len(names), chunk):
        part = names[i:i + chunk]
        lines = "\n".join("%s (%d) %s" % (r, count[r], example.get(r, "")) for r in part)
        canon_txt = "\n".join("%s: %s" % (c, direction.get(c, "?")) for c, _ in canon.most_common(500))
        msg = PROMPT.format(canon=canon_txt or "(none yet)", names=lines)
        m = {}
        for tag in ("", "retry1", "retry2"):
            j = llm.parse_json(llm.chat(NORM_MODEL, msg, seed_tag=tag, **NORM_KW)["text"])
            if isinstance(j, dict) and isinstance(j.get("map"), dict):
                m = j["map"]
                break
        for r in part:
            v = m.get(r)
            if isinstance(v, dict) and v.get("canonical"):
                mapping[r] = {"canonical": _norm_rel(v["canonical"]), "inverse": bool(v.get("inverse"))}
                direction.setdefault(mapping[r]["canonical"], str(v.get("direction") or "?"))
            else:
                mapping[r] = {"canonical": r, "inverse": False}
            canon[mapping[r]["canonical"]] += count[r]
        print("normalized %d/%d names -> %d canonical" % (min(i + chunk, len(names)), len(names), len(canon)),
              flush=True)
    json.dump(mapping, open(out_path, "w"), indent=1)
    return mapping
