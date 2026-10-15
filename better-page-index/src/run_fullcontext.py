"""Full-context baseline: the same chat model reads the complete report of the named run.

This is "perfect retrieval": the model gets exactly the pages of the right run, as text
extracted from the same PDF that PageIndex indexes. Any error here is a reasoning error.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import env  # noqa: F401  (loads the API keys from .env)
import litellm
from pypdf import PdfReader

import qa

CHAT_MODEL = "openrouter/openai/gpt-5.6-sol"
REASONING = "medium"  # the same reasoning effort for every system
PDF = "runs/pilot/plant_log_30_runs.pdf"
PAGES = json.load(open("runs/pilot/plant_log_30_runs_pages.json"))
_reader = PdfReader(PDF)


@lru_cache(maxsize=None)
def run_text(sid):
    first, last = PAGES[str(sid)]
    return "\n\n".join("[page %d]\n%s" % (p, _reader.pages[p - 1].extract_text()) for p in range(first, last + 1))


def ask_one(item, model=CHAT_MODEL):
    t0 = time.time()
    rec = {"sid": item["sid"], "qid": item["qid"], "type": item["type"]}
    msgs = [{"role": "system", "content": qa.TASK_INSTRUCTIONS},
            {"role": "user", "content": "Report of run %d:\n\n%s\n\n%s" % (item["sid"], run_text(item["sid"]), qa.question_text(item))}]
    try:
        r = litellm.completion(model=model, messages=msgs, reasoning_effort=REASONING, timeout=600, num_retries=2)
        rec["answer"] = r.choices[0].message.content or ""
        rec["usage"] = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}
    except Exception as ex:
        rec.update(error="%s: %s" % (type(ex).__name__, ex), answer="")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["parsed"] = qa.parse_answer(rec["answer"])
    rec["hits"], rec["total"], rec["correct"] = qa.score(item, rec["parsed"])
    return rec


def main(out, limit=None, workers=8):
    items = qa.pilot_questions(qa.pilot_runs())[: limit or None]
    for sid in {i["sid"] for i in items}:
        run_text(sid)  # extract once, before the threads start
    done = set()
    if os.path.exists(out):
        done = {(r["sid"], r["qid"]) for r in map(json.loads, open(out))}
    todo = [i for i in items if (i["sid"], i["qid"]) not in done]
    print("questions to ask:", len(todo))
    with open(out, "a") as f, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(ask_one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["sid"], rec["qid"], rec["type"], "correct" if rec["correct"] else "wrong", rec["seconds"], "s",
                  rec.get("error", ""))


if __name__ == "__main__":
    main(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
