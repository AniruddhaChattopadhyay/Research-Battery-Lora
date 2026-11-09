"""Online build: read the passage stream batch by batch and write one JSON record per passage.

Variant "qbranch": infer questions, name the schema, bind facts and keep open slots (figure panel b).
Variant "qbranch_noopen": the same, without the open questions (ablation of the open slots).
Variant "openie": extract triples only (the ablation without questions).
Between batches, the most frequent relation names go into the prompt, so the schema can grow
and be reused (a light form of schema normalization).
"""
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

import llm
import prompts

BUILD_MODEL = "deepseek/deepseek-v4.1-flash"
BUILD_KW = {"json_mode": True, "max_tokens": 4000, "reasoning": {"enabled": False}}


def _norm_rel(r):
    return "_".join(str(r or "").strip().lower().replace("-", " ").split())


def _call(variant, passage, vocab, model):
    tmpl = {"qbranch": prompts.BUILD_QBRANCH, "qbranch_noopen": prompts.BUILD_QBRANCH_NOOPEN,
            "openie": prompts.BUILD_OPENIE}[variant]
    rule = prompts.TITLE_RULE_NEWS if "time" in passage else prompts.TITLE_RULE_WIKI
    msg = tmpl.format(relations=", ".join(vocab) if vocab else "(none yet)", title_rule=rule,
                      title=passage["title"], text=passage["text"])
    for tag in ("", "retry1", "retry2"):
        r = llm.chat(model, msg, seed_tag=tag, **BUILD_KW)
        j = llm.parse_json(r["text"])
        if isinstance(j, dict) and isinstance(j.get("triples" if variant == "openie" else "questions"), list):
            return j, r["usage"]
    return None, r["usage"]


def build(passages_by_batch, variant, out_dir, model=BUILD_MODEL, workers=12, vocab_size=80):
    os.makedirs(out_dir, exist_ok=True)
    rel_count = collections.Counter()
    for t, batch in enumerate(passages_by_batch):
        path = os.path.join(out_dir, "batch_%02d.jsonl" % t)
        if os.path.exists(path):
            recs = [json.loads(line) for line in open(path)]
        else:
            vocab = [r for r, _ in rel_count.most_common(vocab_size)]
            with ThreadPoolExecutor(workers) as ex:
                outs = list(ex.map(lambda p: _call(variant, p, vocab, model), batch))
            recs = [{"pid": p["pid"], "title": p["title"], "batch": t, "out": j, "usage": u}
                    for p, (j, u) in zip(batch, outs)]
            with open(path, "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
        for rec in recs:
            for tr in triples_of(rec, variant):
                rel_count[_norm_rel(tr.get("r"))] += 1
        bad = sum(1 for rec in recs if rec["out"] is None)
        print("batch %d: %d passages, %d failed, %d relation names" % (t, len(recs), bad, len(rel_count)),
              flush=True)


def triples_of(rec, variant):
    """Yield (triple dict, question dict or None) items as plain triple dicts with a 'q' key."""
    out = rec.get("out") or {}
    if variant != "openie":
        for qi, q in enumerate(out.get("questions") or []):
            if not isinstance(q, dict):
                continue
            for tr in q.get("triples") or []:
                if isinstance(tr, dict):
                    yield {**tr, "q": q.get("q"), "qi": qi, "answered": bool(q.get("answered"))}
    else:
        for tr in out.get("triples") or []:
            if isinstance(tr, dict):
                yield {**tr, "q": None, "qi": None, "answered": True}


def load_records(out_dir):
    recs = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("batch_") and name.endswith(".jsonl"):
            recs += [json.loads(line) for line in open(os.path.join(out_dir, name))]
    return recs
