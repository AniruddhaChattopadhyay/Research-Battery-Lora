"""B2 on this Mac: the letter readout of Qwen3.5-4B with MLX (the backbone of Tev1-4B, E5).

OpenRouter does not serve Qwen3.5-4B. This script uses the same prompt and the same rule
as llm.readout: one forward pass for each question, the 20 most probable next tokens, and
a softmax over the option letters among them. Precision: bf16.

  PYTHONPATH=src uv run --no-project --with mlx-lm --with datasets --with httpx \
      --with python-dotenv python src/local_readout.py --n 500

Output: runs/b2-local/local__qwen3.5-4b.jsonl, in the same format as src/run.py.
"""
import argparse
import datetime
import json
import math
import pathlib
import string
import time

import mlx.core as mx
from mlx_lm import load

import tasks
from llm import READOUT_TEMPLATE, as_answer, options

ROOT = pathlib.Path(__file__).resolve().parent.parent
HF_ID = "Qwen/Qwen3.5-4B"
MODEL = "local/qwen3.5-4b"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--tasks", default=",".join(t for t in tasks.TASKS if t != "banking77"))
    a = p.parse_args()

    path = ROOT / "runs" / "b2-local" / "local__qwen3.5-4b.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(r["task"], r["item_id"]) for r in map(json.loads, path.open(encoding="utf-8"))}
    model, tok = load(HF_ID)
    with path.open("a", encoding="utf-8") as f:
        for task in a.tasks.split(","):
            for item in tasks.load(task, a.n):
                if (task, item["id"]) in done:
                    continue
                start = time.perf_counter()
                answers, tokens = {}, 0
                for qname, q in item["questions"].items():
                    opts = options(q)
                    letters = string.ascii_uppercase[: len(opts)]
                    text = READOUT_TEMPLATE.format(
                        state=item["state"],
                        question=q["instructions"],
                        options="\n".join(f"{letter}. {t}" for letter, (_, t) in zip(letters, opts)),
                    )
                    prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                     enable_thinking=False, tokenize=False)
                    ids = tok.encode(prompt)
                    tokens += len(ids)
                    logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
                    logprobs = logits - mx.logsumexp(logits)
                    best = mx.argsort(-logprobs)[:20].tolist()
                    values = logprobs[mx.array(best)].tolist()
                    probs = {name: 0.0 for name, _ in opts}
                    for token_id, lp in zip(best, values):
                        token = tok.decode([token_id]).strip()
                        if token in letters:
                            probs[opts[letters.index(token)][0]] += math.exp(lp)
                    answers[qname] = as_answer(q, probs)
                rec = {
                    "tag": "b2-local", "model": MODEL, "task": task, "item_id": item["id"],
                    "time_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                    "status": 200, "latency_ms": round((time.perf_counter() - start) * 1000, 1), "attempts": 1,
                    "response": {"answers": answers, "model": HF_ID, "provider": "local-mlx-bf16",
                                 "usage": {"prompt_tokens": tokens, "cost": 0.0}},
                    "request": {"model": MODEL, "method": "readout", "state": item["state"], "questions": item["questions"]},
                    "gold": item["gold"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            print(f"{task}: done")


if __name__ == "__main__":
    main()
