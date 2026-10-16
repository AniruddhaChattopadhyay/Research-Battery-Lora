"""Experiment E3: send the perturbed requests (src/perturb.py) for the first n items of each task.

  PYTHONPATH=src uv run python src/run_e3.py --tag e3 --n 200 --models typesafe/jev-1.14

Compare the answers with the original requests of E2 (the same items) in the analysis.
"""
import argparse

import httpx

import perturb
import tasks
from run import run_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--tasks", default=",".join(tasks.TASKS))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", default="", help="variant prefixes, separated by commas (default: all)")
    a = p.parse_args()

    only = tuple(v for v in a.only.split(",") if v)
    jobs = [
        (task, v)
        for task in a.tasks.split(",")
        for item in tasks.load(task, a.n)
        for v in perturb.variants(task, item)
        if not only or v["meta"]["variant"].startswith(only)
    ]
    with httpx.Client() as client:
        for model in a.models.split(","):
            run_model(client, a.tag, model.strip(), jobs, a.workers)


if __name__ == "__main__":
    main()
