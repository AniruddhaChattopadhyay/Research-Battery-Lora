"""Scores: exact match and F1 (HotpotQA normalization, best over answer aliases), an independent
LLM judge (a different model family from the reader), and passage recall of the gold titles."""
import collections
import re
import string

import llm
import prompts

JUDGE_MODEL = "anthropic/claude-haiku-5.5"


def _norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_f1(pred, aliases):
    return (max(float(_norm(pred) == _norm(a)) for a in aliases),
            max(_f1(pred, a) for a in aliases))


def judge(question, gold, pred):
    r = llm.chat(JUDGE_MODEL, prompts.JUDGE.format(question=question, gold=gold, pred=pred),
                 temperature=None, max_tokens=2000)
    return r["text"].strip().lower().startswith("correct")


def recall(gold_titles, retrieved_titles):
    if not gold_titles:
        return None
    return len(set(gold_titles) & set(retrieved_titles)) / len(gold_titles)
