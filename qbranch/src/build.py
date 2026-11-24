"""Online build: read the passage stream batch by batch and write one JSON record per passage.

Variant "qbranch": infer questions, name the schema, bind facts and keep open slots (figure panel b).
Variant "qbranch_noopen": the same, without the open questions (ablation of the open slots).
Variant "openie": extract triples only (the ablation without questions).
Variant "qbranch_qa": "qbranch" plus held-out question-answer pairs of the dataset in the prompt (an FAQ).
Variant "qbranch_events": "qbranch", and the open questions can also be open outcomes of events (news).
Variant "qbranch_hist": "qbranch" plus history context: the related facts and open questions of earlier batches.
Variant "qbranch_hist_qa": "qbranch_hist" plus the question-answer pairs of "qbranch_qa" (budgeted oracle).
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


TEMPLATES = {"qbranch": prompts.BUILD_QBRANCH, "qbranch_noopen": prompts.BUILD_QBRANCH_NOOPEN,
             "openie": prompts.BUILD_OPENIE, "qbranch_qa": prompts.BUILD_QBRANCH_QA,
             "qbranch_events": prompts.BUILD_QBRANCH_EVENTS, "qbranch_hist": prompts.BUILD_QBRANCH_HIST,
             "qbranch_hist_qa": prompts.BUILD_QBRANCH_HIST_QA}
HIST = ("qbranch_hist", "qbranch_hist_qa")
QA = ("qbranch_qa", "qbranch_hist_qa")
MEM_FACTS, MEM_OPEN = 20, 8


class Memory:
    """Facts and open questions of the earlier batches, with dense retrieval for one passage."""

    def __init__(self):
        self.facts, self.opens, self.F, self.O = [], [], None, None
        self._seen = set()

    def add(self, recs, variant):
        import answer
        import numpy as np
        nf, no = [], []
        for rec in recs:
            for tr in triples_of(rec, variant):
                if not tr.get("s") or not tr.get("r"):
                    continue
                o = tr.get("o")
                if o is None or str(o).strip().lower() in ("", "null", "none", "unknown"):
                    t = ("o", str(tr.get("q") or "").strip())
                else:
                    t = ("f", "%s | %s | %s" % (tr["s"], _norm_rel(tr["r"]), o))
                if t[1] and t not in self._seen:
                    self._seen.add(t)
                    (nf if t[0] == "f" else no).append(t[1])
        if nf:
            m = answer.DenseIndex(nf).M
            self.F = m if self.F is None else np.vstack([self.F, m])
            self.facts += nf
        if no:
            m = answer.DenseIndex(no).M
            self.O = m if self.O is None else np.vstack([self.O, m])
            self.opens += no

    def lookup(self, batch):
        """passage pid -> (text of the related facts, text of the related open questions)."""
        import answer
        import numpy as np
        qv = answer.embed_queries([answer.passage_text(p) for p in batch])
        out = {}
        for p, v in zip(batch, qv):
            parts = []
            for items, M, k in ((self.facts, self.F, MEM_FACTS), (self.opens, self.O, MEM_OPEN)):
                if M is None:
                    parts.append("(none yet)")
                else:
                    top = np.argsort(-(M @ v))[:k]
                    parts.append("\n".join("- " + items[i] for i in top))
            out[p["pid"]] = tuple(parts)
        return out


def _call(variant, passage, vocab, model, examples="", mem=("", "")):
    tmpl = TEMPLATES[variant]
    rule = prompts.TITLE_RULE_NEWS if "time" in passage else prompts.TITLE_RULE_WIKI
    msg = tmpl.format(relations=", ".join(vocab) if vocab else "(none yet)", title_rule=rule,
                      title=passage["title"], text=passage["text"], examples=examples,
                      mem_facts=mem[0], mem_open=mem[1])
    for tag in ("", "retry1", "retry2"):
        r = llm.chat(model, msg, seed_tag=tag, **BUILD_KW)
        j = llm.parse_json(r["text"])
        if isinstance(j, dict) and isinstance(j.get("triples" if variant == "openie" else "questions"), list):
            return j, r["usage"]
    return None, r["usage"]


def build(passages_by_batch, variant, out_dir, model=BUILD_MODEL, workers=12, vocab_size=80, examples_fn=None):
    """examples_fn(pid) -> the example text for the prompt of the variant qbranch_qa."""
    if variant in QA and examples_fn is None:
        raise ValueError("%s needs examples_fn" % variant)
    memory = Memory() if variant in HIST else None
    os.makedirs(out_dir, exist_ok=True)
    rel_count = collections.Counter()
    for t, batch in enumerate(passages_by_batch):
        path = os.path.join(out_dir, "batch_%02d.jsonl" % t)
        if os.path.exists(path):
            recs = [json.loads(line) for line in open(path)]
        else:
            vocab = [r for r, _ in rel_count.most_common(vocab_size)]
            mem = memory.lookup(batch) if memory else {}
            with ThreadPoolExecutor(workers) as ex:
                outs = list(ex.map(lambda p: _call(variant, p, vocab, model,
                                                   examples_fn(p["pid"]) if examples_fn else "",
                                                   mem.get(p["pid"], ("", ""))), batch))
            recs = [{"pid": p["pid"], "title": p["title"], "batch": t, "out": j, "usage": u,
                     **({"mem": list(mem[p["pid"]])} if memory else {})}
                    for p, (j, u) in zip(batch, outs)]
            with open(path, "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
        if memory:
            memory.add(recs, variant)
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
