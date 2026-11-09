"""Run the three probe systems on runs/probe/items.jsonl. All use the same chat model and instructions.

  index      build one PageIndex store for each PDF (Flash index; gpt-5.6-luna writes the summaries)
  pageindex  the PageIndex agent answers each item; save the answer, the events and the tokens
  full       the chat model reads the text of the whole PDF in one prompt
  keyword    the chat model with two tools over the same PDF: BM25 page search and page reading
  judge      LongMemEval only: the official judge prompt for each type, with gpt-4o-2024-08-06

The OpenRouter key is shared with other projects, so the cost comes from the token counts and the prices.
The run stops when the estimated probe spend passes BUDGET, or when less than MIN_CREDIT dollars of credit is left.
Usage: PYTHONPATH=src uv run python src/probe_run.py <cmd> [dataset] [questions per type]
"""
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import env  # noqa: F401  (loads the API keys from .env)
import litellm
from pypdf import PdfReader

import probe_data
import usage_tracker

ROOT = probe_data.ROOT
PROBE = os.path.join(ROOT, "runs", "probe")
ITEMS = os.path.join(PROBE, "items.jsonl")
DOCS = os.path.join(PROBE, "pi_docs.json")
INDEX_MODEL = "openrouter/openai/gpt-5.6-luna"
# PROBE_CHAT_MODEL selects a cheaper chat model for all systems; its outputs get a suffix (e.g. _luna).
CHAT_MODEL = os.environ.get("PROBE_CHAT_MODEL", "openrouter/openai/gpt-5.6-sol")
SUFFIX = "" if CHAT_MODEL.endswith("gpt-5.6-sol") else "_" + CHAT_MODEL.rsplit("-", 1)[-1]
WORKERS = int(os.environ.get("PROBE_WORKERS", "6"))
JUDGE_MODEL = "openrouter/openai/gpt-4o-2024-08-06"
REASONING = "medium"
MAX_TURNS = 16
MIN_CREDIT = 15.0  # the key is shared with other projects: leave them this much credit
BUDGET = 33.0  # the estimated spend of the whole probe, from token counts (raised from 30 on 2026-10-17)
SPEND = os.path.join(PROBE, "spend.json")
# Dollars per million tokens: input, cached input, output (OpenRouter prices on 2026-10-16).
PRICES = {"gpt-5.6-sol": (4.0, 0.4, 20.0), "gpt-5.6-luna": (0.2, 0.02, 1.2), "gpt-4o-2024-08-06": (2.5, 1.25, 10.0)}
litellm.drop_params = True


def est_cost(tokens, model):
    pin, pcache, pout = PRICES[model.rsplit("/", 1)[-1]]
    cached = tokens.get("cached", 0)
    return ((tokens.get("prompt", 0) - cached) * pin + cached * pcache + tokens.get("completion", 0) * pout) / 1e6


def add_spend(system, dollars):
    spend = json.load(open(SPEND)) if os.path.exists(SPEND) else {}
    spend[system] = round(spend.get(system, 0) + dollars, 4)
    json.dump(spend, open(SPEND, "w"), indent=1)
    return sum(spend.values())


def credit_left():
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    return d["total_credits"] - d["total_usage"]


def load_items(dataset=None, per_type=None):
    """All items of a dataset; with per_type, only the first per_type items of each question type."""
    items = [json.loads(line) for line in open(ITEMS)]
    items = [i for i in items if dataset in (None, "all", i["dataset"])]
    if per_type:
        seen = Counter()
        items = [i for i in items if seen.update([i["type"]]) or seen[i["type"]] <= per_type]
    return items


def done_ids(out):
    return {json.loads(line)["id"] for line in open(out)} if os.path.exists(out) else set()


def run_batch(name, fn, items, out, workers):
    """Run fn on each item in worker processes; append each record to out; stop at the budget or credit guard."""
    todo = [i for i in items if i["id"] not in done_ids(out)]
    left = credit_left()
    print("%s: %d items to run; credit left $%.2f" % (name, len(todo), left))
    if left < MIN_CREDIT:
        sys.exit("credit guard: less than $%.0f left" % MIN_CREDIT)
    batch = 0.0
    with open(out, "a") as f, ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, it) for it in todo]
        for k, fut in enumerate(futures):
            rec = fut.result()
            rec["cost"] = round(est_cost(rec.get("tokens", {}), CHAT_MODEL), 4)
            batch += rec["cost"]
            total = add_spend(name, rec["cost"])
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print("%-24s %-20s %6.1fs $%.3f %s %s" % (rec["id"], rec["type"], rec["seconds"], rec["cost"],
                                                     json.dumps(rec.get("score", "")), rec.get("error", "")[:120]))
            if total > BUDGET or (k % 10 == 9 and credit_left() < MIN_CREDIT):
                pool.shutdown(cancel_futures=True)
                sys.exit("stop: probe spend $%.2f (budget $%.0f) or credit guard" % (total, BUDGET))
    with open(os.path.join(PROBE, "cost_log.jsonl"), "a") as f:
        f.write(json.dumps({"system": name, "out": os.path.basename(out), "items": len(todo),
                            "dollars_est": round(batch, 4), "time": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    print("%s: estimated $%.4f for %d items" % (name, batch, len(todo)))


def score(item, answer):
    return probe_data.score_designqa(item, answer) if item["dataset"] == "designqa" else None


# ---------------------------------------------------------------- PageIndex

_clients = {}


def _client(store):
    from pageindex import PageIndexClient
    if store not in _clients:
        _clients[store] = PageIndexClient(index_model=INDEX_MODEL, chat_model=CHAT_MODEL, storage_path=store)
    return _clients[store]


def _store(pdf):
    return os.path.join(PROBE, "pi", os.path.splitext(os.path.basename(pdf))[0])


def _index_one(pdf):
    if not getattr(litellm, "_probe_tracker", False):
        usage_tracker.install()
        litellm._probe_tracker = True
    before, t0 = _totals(), time.time()
    res = _client(_store(pdf)).submit_document(pdf)
    time.sleep(1.5)
    tokens = _diff(before, _totals())
    return pdf, {"store": _store(pdf), "doc_id": res["doc_id"], "index_seconds": round(time.time() - t0, 1),
                 "tokens": tokens, "cost": round(est_cost(tokens, INDEX_MODEL), 4)}


def build_index(items, workers=4):
    docs = json.load(open(DOCS)) if os.path.exists(DOCS) else {}
    pdfs = sorted({i["pdf"] for i in items} - set(docs))
    print("PDFs to index:", len(pdfs), "credit left $%.2f" % credit_left())
    with ProcessPoolExecutor(workers) as pool:
        for pdf, info in pool.map(_index_one, pdfs):
            docs[pdf] = info
            json.dump(docs, open(DOCS, "w"), indent=1)
            add_spend("pageindex-index", info["cost"])
            print(os.path.basename(pdf), info)


def _event_dict(e):
    return e if isinstance(e, dict) else getattr(e, "__dict__", {"repr": repr(e)})


def _totals():
    return {k: dict(v) for k, v in usage_tracker.TOTALS.items()}


def _diff(a, b):
    out = Counter()
    for model, t in b.items():
        for k in ("calls", "prompt", "completion", "cached"):
            out[k] += t.get(k, 0) - a.get(model, {}).get(k, 0)
    return dict(out)


def ask_pageindex(item):
    if not getattr(litellm, "_probe_tracker", False):
        usage_tracker.install()
        litellm._probe_tracker = True
    doc = json.load(open(DOCS))[item["pdf"]]
    rec = {"id": item["id"], "type": item["type"], "system": "pageindex" + SUFFIX}
    before, t0 = _totals(), time.time()
    try:
        stream = _client(doc["store"]).chat(item["question"], doc_id=doc["doc_id"], stream=True,
                                            instructions=probe_data.INSTRUCTIONS[item["dataset"]],
                                            max_turns=MAX_TURNS, reasoning_effort=REASONING)
        events = [_event_dict(e) for e in stream.events]
        answer = "".join(e.get("delta", "") or e.get("text", "") for e in events
                         if e.get("type") in ("answer", "answer_delta", "text", "text_delta"))
        rec.update(events=events, answer=answer)
    except Exception as ex:  # a failure counts as a wrong answer
        rec.update(error="%s: %s" % (type(ex).__name__, ex), answer="")
    rec["seconds"] = round(time.time() - t0, 1)
    time.sleep(1.5)  # LiteLLM logs the usage in a background thread
    rec["tokens"] = _diff(before, _totals())
    rec["score"] = score(item, rec["answer"])
    return rec


# ---------------------------------------------------------------- Full context

@lru_cache(maxsize=4)
def pdf_pages(pdf):
    return [p.extract_text() or "" for p in PdfReader(pdf).pages]


def pdf_text(pdf):
    return "\n\n".join("[page %d]\n%s" % (k + 1, t) for k, t in enumerate(pdf_pages(pdf)))


def _usage(r):
    u = r.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {"calls": 1, "prompt": u.prompt_tokens, "completion": u.completion_tokens, "cached": cached}


def ask_full(item):
    rec = {"id": item["id"], "type": item["type"], "system": "full" + SUFFIX}
    msgs = [{"role": "system", "content": probe_data.INSTRUCTIONS[item["dataset"]]},
            {"role": "user", "content": "Document:\n\n%s\n\n%s" % (pdf_text(item["pdf"]), item["question"])}]
    t0 = time.time()
    try:
        r = litellm.completion(model=CHAT_MODEL, messages=msgs, reasoning_effort=REASONING, timeout=900, num_retries=2)
        rec["answer"] = r.choices[0].message.content or ""
        rec["tokens"] = _usage(r)
    except Exception as ex:
        rec.update(error="%s: %s" % (type(ex).__name__, ex), answer="")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["score"] = score(item, rec["answer"])
    return rec


# ---------------------------------------------------------------- Keyword agent (BM25 over pages)

TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def _tokens(text):
    out = []
    for t in TOKEN.findall(text.lower()):
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") else t)
        if "." in t:
            out.extend(t.split("."))
    return out


@lru_cache(maxsize=4)
def bm25(pdf):
    docs = [Counter(_tokens(t)) for t in pdf_pages(pdf)]
    n, avg = len(docs), sum(sum(d.values()) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    return docs, avg, idf


def search(pdf, query, k=5):
    docs, avg, idf = bm25(pdf)
    q = set(_tokens(query))
    scores = []
    for p, d in enumerate(docs):
        length = sum(d.values())
        s = sum(idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * length / avg)) for t in q if t in d)
        if s > 0:
            scores.append((s, p))
    out = []
    for s, p in sorted(scores, reverse=True)[: max(1, min(int(k or 5), 10))]:
        lines = [ln for ln in pdf_pages(pdf)[p].splitlines() if ln.strip()]
        best = sorted(lines, key=lambda ln: -len(q & set(_tokens(ln))))[:3]
        out.append({"page": p + 1, "score": round(s, 2), "snippet": " ... ".join(ln.strip()[:160] for ln in best)})
    return out


def read_pages(pdf, spec):
    pages = sorted(p for p in _pages_of(spec) if 1 <= p <= len(pdf_pages(pdf)))[:8]
    return "\n\n".join("[page %d]\n%s" % (p, pdf_pages(pdf)[p - 1]) for p in pages) or "No such pages."


def _pages_of(spec):
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        m = re.match(r"^(\d+)(?:-(\d+))?$", part)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
    return out


KEYWORD_TOOLS = [
    {"type": "function", "function": {
        "name": "search", "description": "BM25 keyword search over the pages of the document. "
                                         "Returns the best pages, each with a short snippet.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
            "k": {"type": "integer", "description": "Number of pages to return, 1 to 10 (default 5)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_pages", "description": "Read the full text of pages, for example '12' or '12-14,20'. "
                                             "At most 8 pages for each call.",
        "parameters": {"type": "object", "properties": {"pages": {"type": "string"}}, "required": ["pages"]}}},
]


def ask_keyword(item):
    pdf = item["pdf"]
    system = (probe_data.INSTRUCTIONS[item["dataset"]] + "\n\nThe document has %d pages. Use the tools to search it "
              "and to read the pages that you need. When you have the answer, reply with the answer only."
              % len(pdf_pages(pdf)))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": item["question"]}]
    rec = {"id": item["id"], "type": item["type"], "system": "keyword" + SUFFIX, "calls": [], "answer": ""}
    tokens, t0 = Counter(), time.time()
    try:
        for _ in range(MAX_TURNS):
            r = litellm.completion(model=CHAT_MODEL, messages=msgs, tools=KEYWORD_TOOLS, reasoning_effort=REASONING,
                                   timeout=600, num_retries=2)
            tokens.update(_usage(r))
            m = r.choices[0].message
            if not m.tool_calls:
                rec["answer"] = m.content or ""
                break
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in m.tool_calls]})
            for c in m.tool_calls:
                args = json.loads(c.function.arguments or "{}")
                if c.function.name == "search":
                    result = json.dumps(search(pdf, args.get("query", ""), args.get("k", 5)))
                else:
                    result = read_pages(pdf, args.get("pages", ""))
                rec["calls"].append({"name": c.function.name, "args": args})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
    except Exception as ex:
        rec["error"] = "%s: %s" % (type(ex).__name__, ex)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["tokens"] = dict(tokens)
    rec["score"] = score(item, rec["answer"])
    return rec


# ---------------------------------------------------------------- LongMemEval judge

# The official prompts, copied from LongMemEval src/evaluation/evaluate_qa.py (get_anscheck_prompt).
_JUDGE_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
               "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
               "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
               "the response only contains a subset of the information required by the answer, answer no. ")
_JUDGE = {
    "default": _JUDGE_BASE + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
                             "correct? Answer yes or no only.",
    "temporal-reasoning": _JUDGE_BASE + "In addition, do not penalize off-by-one errors for the number of days. If the "
        "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect "
        "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
        "the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the updated "
        "answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
        "response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer "
        "yes if the model correctly identifies the question as unanswerable. The model could say that the information "
        "is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
        "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
        "Answer yes or no only.",
}


def _judge_prompt(item, answer):
    key = "abstention" if item["abstention"] else item["lme_type"]
    template = _JUDGE.get(key, _JUDGE["default"])
    return template.format(item["raw_question"], item["gold"], answer)


def judge_one(args):
    item, rec = args
    prompt = _judge_prompt(item, rec["answer"])
    r = litellm.completion(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                           max_tokens=10, timeout=120, num_retries=3)
    verdict = (r.choices[0].message.content or "").strip().lower()
    return rec["id"], rec["system"], verdict.startswith("yes"), est_cost(_usage(r), JUDGE_MODEL)


def judge(systems=("pageindex", "full", "keyword", "pageindex_luna", "full_luna", "keyword_luna")):
    items = {i["id"]: i for i in load_items("longmemeval")}
    out = os.path.join(PROBE, "judge.jsonl")
    seen = {(j["id"], j["system"]) for j in map(json.loads, open(out))} if os.path.exists(out) else set()
    todo = []
    for s in systems:
        path = os.path.join(PROBE, "%s.jsonl" % s)
        if os.path.exists(path):
            todo += [(items[r["id"]], r) for r in map(json.loads, open(path))
                     if r["id"] in items and (r["id"], s) not in seen]
    with open(out, "a") as f, ProcessPoolExecutor(8) as pool:
        for qid, system, ok, cost in pool.map(judge_one, todo):
            f.write(json.dumps({"id": qid, "system": system, "correct": ok}) + "\n")
            add_spend("judge", cost)
            print(qid, system, ok)


if __name__ == "__main__":
    cmd = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else None
    per_type = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(PROBE, exist_ok=True)
    if cmd == "index":
        build_index(load_items(dataset, per_type))
    elif cmd == "judge":
        judge()
    else:
        fn = {"pageindex": ask_pageindex, "full": ask_full, "keyword": ask_keyword}[cmd]
        run_batch(cmd + SUFFIX, fn, load_items(dataset, per_type), os.path.join(PROBE, "%s%s.jsonl" % (cmd, SUFFIX)),
                  WORKERS)
