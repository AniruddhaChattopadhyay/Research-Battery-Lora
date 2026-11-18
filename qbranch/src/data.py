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
    if name == "multihoprag":
        return load_multihoprag(n_questions)
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


def _chunks(body, min_words=200):
    """Cut an article into passages of about min_words or more words, at paragraph ends."""
    out, cur = [], []
    for para in [p.strip() for p in body.split("\n") if p.strip()]:
        cur.append(para)
        if sum(len(p.split()) for p in cur) >= min_words:
            out.append("\n".join(cur))
            cur = []
    if cur:
        if out and sum(len(p.split()) for p in cur) < min_words // 2:
            out[-1] += "\n" + "\n".join(cur)
        else:
            out.append("\n".join(cur))
    return out


def load_multihoprag(n_questions=None, per_type=100, seed=0):
    """MultiHop-RAG (609 dated news articles, Sep-Dec 2023). The stream is in publication order.

    Passages: article chunks of about 200 words, with "time" (ISO date) and "source".
    Questions: a fixed sample of per_type inference, comparison and temporal queries (seed 0).
    The null queries need abstention, so they are not in this sample.
    """
    d = os.path.join(DATA, "..", "multihoprag")
    corpus = json.load(open(os.path.join(d, "corpus.json")))
    passages, by_url = [], {}
    for art in sorted(corpus, key=lambda a: a["published_at"]):
        for ch in _chunks(art["body"]):
            p = {"pid": len(passages), "title": "%s (%s, %s)" % (art["title"], art["source"], art["published_at"][:10]),
                 "text": ch, "time": art["published_at"][:10],
                 "source": art["source"], "url": art["url"]}
            passages.append(p)
            by_url.setdefault(art["url"], []).append(p)
    qs = json.load(open(os.path.join(d, "MultiHopRAG.json")))
    rng = random.Random(seed)
    items = []
    for t in ("inference_query", "comparison_query", "temporal_query"):
        pool = [q for q in qs if q["question_type"] == t]
        for q in rng.sample(pool, per_type):
            gold = set()
            for e in q["evidence_list"]:
                for p in by_url.get(e["url"], []):
                    if e["fact"][:60] in p["text"]:
                        gold.add(p["pid"])
            items.append({"qid": "%s_%d" % (t, qs.index(q)), "question": q["query"], "answer": q["answer"],
                          "aliases": [q["answer"]], "gold_titles": sorted({e["title"] for e in q["evidence_list"]}),
                          "gold_pids": sorted(gold), "context_pids": sorted(gold), "n_gold_missing": 0, "type": t})
    return (items[:n_questions] if n_questions else items), passages


def stream_batches(passages, n_batches, seed=0):
    """Cut the passages into n_batches equal parts (the online stream). Passages with a "time" key
    keep their time order. Else shuffle them once with the seed."""
    order = list(passages)
    if order and "time" in order[0]:
        order.sort(key=lambda p: (p["time"], p["pid"]))
    else:
        random.Random(seed).shuffle(order)
    size = -(-len(order) // n_batches)
    return [order[i:i + size] for i in range(0, len(order), size)]


def faq_pool(name, qs, size=100, seed=1):
    """Held-out question-answer pairs for the build prompt of the variant qbranch_qa (an FAQ of the collection).

    HippoRAG 2 sets: the candidates are the first `size` questions (the development slice). MultiHop-RAG: a random
    sample of the inference, comparison and temporal queries that are not in the evaluation sample.
    MuSiQue keeps only 23 of 100, because most MuSiQue questions share a sub-question with another question.
    A candidate is dropped when it shares a gold passage, a MuSiQue sub-question id or a 2Wiki evidence triple with
    an evaluation question, so the examples do not give the facts of a test question.
    Return (pool items, ids of the evaluation questions). Evaluate the variant only on those ids.
    """
    if name == "multihoprag":
        d = os.path.join(DATA, "..", "multihoprag")
        raw = json.load(open(os.path.join(d, "MultiHopRAG.json")))
        used = {q["qid"] for q in qs}
        cands = [{"qid": "%s_%d" % (q["question_type"], i), "question": q["query"], "answer": q["answer"],
                  "urls": {e["url"] for e in q["evidence_list"]}}
                 for i, q in enumerate(raw) if q["question_type"] != "null_query"]
        cands = [c for c in cands if c["qid"] not in used]
        random.Random(seed).shuffle(cands)
        by_q = {q["query"]: q for q in raw}
        eval_urls = {e["url"] for q in qs for e in by_q[q["question"]]["evidence_list"]}
        pool = [c for c in cands if not (c["urls"] & eval_urls)][:size]
        return pool, [q["qid"] for q in qs]
    raw = json.load(open(os.path.join(DATA, name + ".json")))
    hop_keys = []
    for q in raw:
        if "question_decomposition" in q:
            hop_keys.append({h["id"] for h in q["question_decomposition"]})
        elif "evidences" in q:
            hop_keys.append({tuple(e) for e in q["evidences"]})
        else:
            hop_keys.append(set())
    eval_ids = [q["qid"] for q in qs[size:]]
    eval_pids = {pid for q in qs[size:] for pid in q["gold_pids"]}
    eval_hops = set().union(*hop_keys[size:])
    pool = [q for q, h in zip(qs[:size], hop_keys[:size]) if not (set(q["gold_pids"]) & eval_pids) and not (h & eval_hops)]
    return pool, eval_ids


def faq_examples(pool, pid, k=8):
    """k pool pairs for one passage. The sample depends only on the passage id, so a rerun gives the same prompt."""
    pick = random.Random(int(pid)).sample(pool, min(k, len(pool)))
    return "\n".join("- Q: %s A: %s" % (q["question"], q["answer"]) for q in pick)
