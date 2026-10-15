"""Run System One models on tasks through OpenRouter. Write one JSON line for each call.

Example (experiment E0, 5 items from each task):
  PYTHONPATH=src uv run python src/run.py --tag e0 --n 5 --models typesafe/jev-1.13,microsoft/microsoft-decision-1

Output: runs/<tag>/<model>.jsonl. Each line has the full request, the full response,
the latency, the gold answers and the time. A second run with the same tag skips the
items that already returned status 200.
"""
import argparse
import datetime
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import httpx

import tasks
from openrouter import PROVIDER_PIN, decide

ROOT = pathlib.Path(__file__).resolve().parent.parent


def file_name(model):
    return model.replace("~", "alias-").replace("/", "__").replace(":", "_") + ".jsonl"


def run_model(client, tag, model, jobs, workers):
    path = ROOT / "runs" / tag / file_name(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["status"] == 200:
                done.add((rec["task"], rec["item_id"]))
    todo = [(task, item) for task, item in jobs if (task, item["id"]) not in done]

    def call(job):
        task, item = job
        request = {"model": model, "state": item["state"], "questions": item["questions"]}
        if model in PROVIDER_PIN:
            request["provider_pin"] = PROVIDER_PIN[model]
        result = decide(client, model, item["state"], item["questions"])
        return {
            "tag": tag,
            "model": model,
            "task": task,
            "item_id": item["id"],
            "time_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            **result,
            "request": request,
            "gold": item["gold"],
        }

    ok = 0
    with ThreadPoolExecutor(workers) as pool, path.open("a", encoding="utf-8") as f:
        for rec in pool.map(call, todo):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += rec["status"] == 200
    print(f"{model}: {ok} of {len(todo)} new calls returned 200 ({len(done)} done before) -> {path.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="the run name, for example e0")
    p.add_argument("--models", required=True, help="OpenRouter model ids, separated by commas")
    p.add_argument("--tasks", default=",".join(tasks.TASKS), help="task names, separated by commas")
    p.add_argument("--n", type=int, required=True, help="the number of items from each task")
    p.add_argument("--workers", type=int, default=8, help="parallel requests for each model")
    a = p.parse_args()

    jobs = [(task, item) for task in a.tasks.split(",") for item in tasks.load(task, a.n)]
    with httpx.Client() as client:
        for model in a.models.split(","):
            run_model(client, a.tag, model.strip(), jobs, a.workers)


if __name__ == "__main__":
    main()
