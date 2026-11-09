"""Load the HippoRAG 2 multi-hop QA sets (HotpotQA, MuSiQue, 2Wiki) and make pilot slices.

The HippoRAG 2 release pools the paragraphs of 1,000 questions into one corpus per dataset.
A pilot slice keeps the first n questions and the corpus passages that their contexts name, so
the slice is a smaller version of the same pooled setting.
"""
import json
import os
import random

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hipporag2")


def _gold_titles(q):
    if "supporting_facts" in q:  # HotpotQA, 2Wiki
        return sorted({t for t, _ in q["supporting_facts"]})
    if "paragraphs" in q:  # MuSiQue
        return sorted({p["title"] for p in q["paragraphs"] if p.get("is_supporting")})
    return []


def _context_titles(q):
    if "context" in q:
        return [c[0] for c in q["context"]]
    if "paragraphs" in q:
        return [p["title"] for p in q["paragraphs"]]
    return []


def _key(text):
    return " ".join(str(text).split())[:300]


def load(name="hotpotqa", n_questions=None):
    """Return (questions, passages). Each passage: {"pid", "title", "text"}.

    Each question has gold_pids and context_pids. HotpotQA and 2Wiki titles are unique, so the
    match is by title. MuSiQue repeats titles, so the match is by paragraph text.
    """
    qs = json.load(open(os.path.join(DATA, name + ".json")))
    corpus = json.load(open(os.path.join(DATA, name + "_corpus.json")))
    passages = [{"pid": i, "title": p["title"], "text": p["text"]} for i, p in enumerate(corpus)]
    by_title = {p["title"]: p["pid"] for p in passages}
    by_text = {_key(p["text"]): p["pid"] for p in passages}
    items = []
    for q in qs[:n_questions] if n_questions else qs:
        ans = q.get("answer") if "answer" in q else q.get("possible_answers", [""])[0]
        aliases = [ans] + list(q.get("answer_aliases", []) or [])
        if "paragraphs" in q:
            gold = [by_text.get(_key(p["paragraph_text"])) for p in q["paragraphs"] if p.get("is_supporting")]
            ctx = [by_text.get(_key(p["paragraph_text"])) for p in q["paragraphs"]]
        else:
            gold = [by_title.get(t) for t in _gold_titles(q)]
            ctx = [by_title.get(t) for t in _context_titles(q)]
        items.append({"qid": q.get("_id") or q.get("id"), "question": q["question"], "answer": ans,
                      "aliases": aliases, "gold_titles": _gold_titles(q), "gold_pids": sorted(set(gold) - {None}),
                      "context_pids": sorted(set(ctx) - {None}), "n_gold_missing": gold.count(None),
                      "type": q.get("type", "") or str(q.get("id", "")).split("__")[0]})
    if n_questions:
        keep = {pid for it in items for pid in it["context_pids"]}
        passages = [p for p in passages if p["pid"] in keep]
    return items, passages


def stream_batches(passages, n_batches, seed=0):
    """Shuffle the passages once and cut them into n_batches equal parts (the online stream)."""
    order = list(passages)
    random.Random(seed).shuffle(order)
    size = -(-len(order) // n_batches)
    return [order[i:i + size] for i in range(0, len(order), size)]
