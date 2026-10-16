"""Shared code for the analysis: load runs, make probability distributions, metrics.

An "answer" is one question of one item. Its distribution is a dict from option name to
probability (predicate: "true" and "false"; score: "0" to "4").
"""
import glob
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

TASK_ORDER = ["boolq", "aegis", "civil", "banking77", "arxiv", "when2call", "judgebench", "helpsteer2"]
TASK_LABEL = {
    "boolq": "BoolQ",
    "aegis": "Aegis 2.0",
    "civil": "Civil Comments",
    "banking77": "BANKING77",
    "arxiv": "arXiv (fresh)",
    "when2call": "When2Call",
    "judgebench": "JudgeBench-Pro",
    "helpsteer2": "HelpSteer2",
}
# Task groups for H1 (experiment design, E2).
TASK_GROUP = {
    "boolq": "knowledge",
    "aegis": "classification",
    "civil": "classification",
    "banking77": "classification",
    "arxiv": "classification",
    "when2call": "agent",
    "judgebench": "judging",
    "helpsteer2": "judging",
}

# Short names for tables and figures.
SYSTEM_LABEL = {
    "typesafe/jev-1.13": "Jev 1.13",
    "typesafe/jev-1.14": "Jev 1.14",
    "openai/gpt-6-luna-decisions": "OpenAI Decisions (Luna)",
    "microsoft/microsoft-decision-1": "Microsoft-Decision-1",
    "perplexity/pplx-decider-v1.1-27b": "pplx-decider v1.1",
    "cloudflare/clef": "Clef",
    "cloudflare/clef-flash": "Clef-flash",
    "cloudflare/clef-omni": "Clef-omni",
    "inception/mercury-decide": "Mercury Decide",
    "jaredpalmer/kev-4b": "Kev-4B",
    "liquid/d1": "Liquid d1",
    "nace-ai/drex-v1.5": "Drex v1.5",
    "nace-ai/drex-v2.0": "Drex v2.0",
    "respan/span-01": "Span-01",
    "respan/span-01-lite": "Span-01-lite",
    "scaledown/classify-1": "Classify-1",
    "togethercomputer/tev1-4b-experimental": "Tev1-4B",
    "upstage/solar-decide": "Solar Decide",
    "upstage/solar-decide-flash": "Solar Decide Flash",
    "openai/gpt-6-sol": "GPT-6 Sol",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "openai/gpt-6-luna": "GPT-6 Luna (chat)",
    "qwen/qwen3.5-9b": "Qwen3.5-9B readout",
    "qwen/qwen3.8-27b": "Qwen3.8-27B readout",
    "local/qwen3.5-4b": "Qwen3.5-4B readout",
}
TIER1 = [
    "typesafe/jev-1.14",
    "openai/gpt-6-luna-decisions",
    "microsoft/microsoft-decision-1",
    "perplexity/pplx-decider-v1.1-27b",
    "cloudflare/clef",
    "cloudflare/clef-flash",
]
SYSTEM_TWO = [
    "openai/gpt-6-sol",
    "anthropic/claude-sonnet-5.5",
    "google/gemini-3.8-flash",
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
]


def file_name(model):
    return model.replace("~", "alias-").replace("/", "__").replace(":", "_") + ".jsonl"


def distribution(qtype, ans):
    """The answer as a dict of probabilities that sum to 1, or None if it is not usable."""
    if not isinstance(ans, dict):
        return None
    if qtype == "noul":
        p = ans.get("noul")
        if not isinstance(p, (int, float)) or math.isnan(p):
            return None
        p = min(1.0, max(0.0, float(p)))
        return {"true": p, "false": 1 - p}
    probs = ans.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    probs = {str(k): float(v) for k, v in probs.items() if isinstance(v, (int, float))}
    total = sum(probs.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in probs.items()}


def gold_label(qtype, gold):
    if qtype == "noul":
        return "true" if (gold >= 0.5 if isinstance(gold, float) else gold) else "false"
    return str(gold)


def top(dist):
    return max(dist, key=dist.get)


def records(path):
    """The last status-200 record for each (task, item_id) in one run file."""
    out = {}
    if not pathlib.Path(path).exists():
        return out
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["status"] == 200:
            out[(r["task"], r["item_id"])] = r
    return out


def answers(path):
    """(task, item_id, qname) -> a dict with the answer, the gold and the record data."""
    out = {}
    for (task, item_id), r in records(path).items():
        resp = r["response"]
        given = resp.get("answers") or {}
        usage = resp.get("usage") or {}
        for qname, q in r["request"]["questions"].items():
            dist = distribution(q["type"], given.get(qname))
            gold = r["gold"].get(qname)
            out[(task, item_id, qname)] = {
                "type": q["type"],
                "dist": dist,
                "gold": gold_label(q["type"], gold) if gold is not None else None,
                "soft": float(gold) if q["type"] == "noul" and gold is not None else None,
                "latency_ms": r.get("latency_ms"),
                "cost": usage.get("cost"),
                "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                "resolved": resp.get("model"),
                "provider": resp.get("provider"),
                "meta": r.get("meta"),
                "n_questions": len(r["request"]["questions"]),
            }
    return out


def load_system(model, tags=("e2",)):
    """All answers of one system in E2 (or in the baseline runs b1 and b2)."""
    out = {}
    for tag in tags:
        out.update(answers(RUNS / tag / file_name(model)))
    return out


# ---------- metrics ----------

def confidences(rows):
    """Arrays of the top probability and of 1/0 for a correct top answer."""
    conf = np.array([max(r["dist"].values()) for r in rows])
    correct = np.array([top(r["dist"]) == r["gold"] for r in rows], dtype=float)
    return conf, correct


def ece(conf, correct, bins=15, mass=False):
    """Expected calibration error of the top answer (equal-width or equal-mass bins)."""
    if len(conf) == 0:
        return float("nan")
    if mass:
        order = np.argsort(conf, kind="stable")
        groups = np.array_split(order, bins)
    else:
        idx = np.minimum((conf * bins).astype(int), bins - 1)
        groups = [np.where(idx == b)[0] for b in range(bins)]
    total = 0.0
    for g in groups:
        if len(g):
            total += len(g) * abs(conf[g].mean() - correct[g].mean())
    return total / len(conf)


def classwise_ece(rows, bins=15):
    """Class-wise ECE: the mean over options of the calibration of p(option)."""
    options = sorted({k for r in rows for k in r["dist"]})
    values = []
    for k in options:
        p = np.array([r["dist"].get(k, 0.0) for r in rows])
        y = np.array([r["gold"] == k for r in rows], dtype=float)
        if y.sum() == 0 and p.sum() < 1e-9:
            continue
        values.append(ece(p, y, bins))
    return float(np.mean(values)) if values else float("nan")


def brier(rows):
    """Multi-class Brier score: the sum over options of (p - y)^2, mean over answers."""
    out = []
    for r in rows:
        keys = set(r["dist"]) | {r["gold"]}
        out.append(sum((r["dist"].get(k, 0.0) - (k == r["gold"])) ** 2 for k in keys))
    return float(np.mean(out)) if out else float("nan")


def auroc(scores, labels):
    """AUROC with ties counted as one half (the Mann-Whitney statistic)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rpos = ranks[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(order) + 1)
    return float(risk.mean()) if len(risk) else float("nan")


def bootstrap(values_by_item, fn, n=1000, seed=0):
    """A 95% interval of fn over items, resampled with replacement.

    values_by_item is a list of lists (the rows of one item stay together).
    """
    rng = np.random.default_rng(seed)
    k = len(values_by_item)
    stats = []
    for _ in range(n):
        pick = rng.integers(0, k, k)
        rows = [row for i in pick for row in values_by_item[i]]
        stats.append(fn(rows))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar(b, c):
    """Exact two-sided McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h
