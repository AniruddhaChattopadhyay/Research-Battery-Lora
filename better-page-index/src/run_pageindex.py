"""Run the real PageIndex SDK (local mode) on the plant-log PDF.

  index: build the PageIndex tree of the corpus PDF (Flash mode, the SDK default).
  ask:   ask each pilot question with the PageIndex agent, and save the answer and the full
         agent trace (tool calls and results) for each question.

Models (the README recommendation): gpt-5.6-luna builds the index, gpt-5.6-sol answers.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import env  # noqa: F401  (loads the API keys from .env)
from pageindex import PageIndexClient

import qa
import usage_tracker

INDEX_MODEL = "openrouter/openai/gpt-5.6-luna"
CHAT_MODEL = "openrouter/openai/gpt-5.6-sol"
REASONING = "medium"  # the same reasoning effort for every system


def client(store):
    return PageIndexClient(index_model=INDEX_MODEL, chat_model=CHAT_MODEL, storage_path=store)


def index(pdf, store, out):
    c = client(store)
    t0 = time.time()
    res = c.submit_document(pdf)
    res["index_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)
    print(res)


def _event_dict(e):
    if isinstance(e, dict):
        return e
    return getattr(e, "__dict__", {"repr": repr(e)})


def ask_one(c, doc_id, item):
    t0 = time.time()
    rec = {"sid": item["sid"], "qid": item["qid"], "type": item["type"]}
    try:
        stream = c.chat(qa.question_text(item), doc_id=doc_id, stream=True,
                        instructions=qa.TASK_INSTRUCTIONS, max_turns=16, reasoning_effort=REASONING)
        events = [_event_dict(e) for e in stream.events]
        answer = "".join(e.get("delta", "") or e.get("text", "") for e in events
                         if e.get("type") in ("answer", "answer_delta", "text", "text_delta"))
        rec.update(events=events, answer=answer)
    except Exception as ex:  # record the failure; it counts as a wrong answer
        rec.update(error="%s: %s" % (type(ex).__name__, ex), answer="")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["parsed"] = qa.parse_answer(rec["answer"])
    rec["hits"], rec["total"], rec["correct"] = qa.score(item, rec["parsed"])
    return rec


def ask(store, doc_id, out, limit=None, workers=8):
    usage_tracker.install()
    c = client(store)
    items = qa.pilot_questions(qa.pilot_runs())[: limit or None]
    done = set()
    if os.path.exists(out):
        done = {(r["sid"], r["qid"]) for r in map(json.loads, open(out))}
    todo = [i for i in items if (i["sid"], i["qid"]) not in done]
    print("questions to ask:", len(todo))
    with open(out, "a") as f, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(lambda it: ask_one(c, doc_id, it), todo):
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(rec["sid"], rec["qid"], rec["type"], "correct" if rec["correct"] else "wrong", rec["seconds"], "s",
                  rec.get("error", ""))
    json.dump(usage_tracker.TOTALS, open(out + ".usage.json", "w"), indent=1)
    print("usage:", usage_tracker.TOTALS)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "index":
        index(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "ask":
        ask(sys.argv[2], sys.argv[3], sys.argv[4], limit=int(sys.argv[5]) if len(sys.argv) > 5 else None)
