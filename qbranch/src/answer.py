"""Answer systems (figure panel c) and the baselines. All systems use the same reader prompt.

- closed_book: no evidence.
- dense: top-k passages by embedding similarity.
- oracle: the gold supporting passages.
- graph: seed facts (and inferred questions, for QBranch) by embedding similarity, one hop of
  expansion from their entities, a path score, then the top facts and the passages they come from.
"""
import re

import numpy as np

import llm
import prompts

EMBED_MODEL = "google/gemini-embedding-2"
READER_MODEL = "openai/gpt-6-luna"
READER_KW = {"max_tokens": 3000, "reasoning": {"effort": "low"}}


def _unit(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def passage_text(p):
    return p["title"] + "\n" + p["text"]


class DenseIndex:
    def __init__(self, texts):
        self.M = _unit(llm.embed(EMBED_MODEL, texts))

    def scores(self, qvec):
        return self.M @ qvec


def embed_queries(texts):
    return _unit(llm.embed(EMBED_MODEL, texts))


def read(question, evidence):
    msg = prompts.READER.format(evidence=evidence, question=question) if evidence is not None \
        else prompts.CLOSED_BOOK.format(question=question)
    r = llm.chat(READER_MODEL, msg, **READER_KW)
    m = re.findall(r"Answer:\s*(.+)", r["text"])
    return (m[-1].strip() if m else r["text"].strip().splitlines()[-1] if r["text"].strip() else ""), r


def fmt_passages(ps):
    return "Passages:\n" + "\n\n".join("[%d] %s\n%s" % (i + 1, p["title"], p["text"]) for i, p in enumerate(ps))


class GraphRetriever:
    def __init__(self, graph, passages, use_questions, n_seed=15, n_facts=25, n_passages=5,
                 hub_degree=60, alpha=0.5):
        self.g, self.use_q = graph, use_questions
        self.by_pid = {p["pid"]: p for p in passages}
        self.n_seed, self.n_facts, self.n_passages = n_seed, n_facts, n_passages
        self.hub_degree, self.alpha = hub_degree, alpha
        self.F = DenseIndex([graph.fact_text(f) for f in graph.facts])
        self.answered = [a for a in graph.agenda if a["answered"] and a["fids"]]
        self.Q = DenseIndex([a["q"] for a in self.answered]) if use_questions and self.answered else None

    def retrieve(self, qvec):
        fs = self.F.scores(qvec)
        score = {}
        for fid in np.argsort(-fs)[:self.n_seed]:
            score[int(fid)] = float(fs[fid])
        if self.Q is not None:  # match the user question against the inferred questions
            qs = self.Q.scores(qvec)
            for i in np.argsort(-qs)[:self.n_seed]:
                for fid in self.answered[i]["fids"]:
                    score[fid] = max(score.get(fid, -1), float(qs[i]))
        seeds = sorted(score.items(), key=lambda x: -x[1])
        for fid, sc in seeds:  # one hop from the entities of each seed fact
            f = self.g.facts[fid]
            for node in {f["s_norm"], f["o_norm"]}:
                nbrs = self.g.adj.get(node, [])
                if len(nbrs) > self.hub_degree:
                    continue
                for nf in nbrs:
                    s2 = self.alpha * sc + (1 - self.alpha) * float(fs[nf])
                    if s2 > score.get(nf, -1):
                        score[nf] = s2
        top = sorted(score.items(), key=lambda x: -x[1])[:self.n_facts]
        pscore = {}
        for fid, sc in top:
            pid = self.g.facts[fid]["pid"]
            pscore[pid] = pscore.get(pid, 0) + sc
        pids = [pid for pid, _ in sorted(pscore.items(), key=lambda x: -x[1])[:self.n_passages]]
        return [self.g.facts[fid] for fid, _ in top], [self.by_pid[pid] for pid in pids]

    def evidence(self, facts, ps):
        lines = ["(%s | %s | %s) [%s]" % (f["s"], f["r"], f["o"], self.g.pid_title[f["pid"]]) for f in facts]
        return "Facts from the knowledge graph:\n" + "\n".join(lines) + "\n\n" + fmt_passages(ps)
