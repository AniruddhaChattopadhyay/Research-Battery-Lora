"""Progress of the runs: calls and status codes for each model file.

  uv run python analysis/status.py runs/e2 runs/b1
"""
import collections
import glob
import json
import sys

for folder in sys.argv[1:]:
    print(f"== {folder}")
    for f in sorted(glob.glob(f"{folder}/*.jsonl")):
        status, cost = collections.Counter(), 0.0
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            status[r["status"]] += 1
            usage = r["response"].get("usage") or {}
            cost += usage.get("cost", 0) or 0
        print(f"{f.rsplit('/', 1)[-1]:50s} {sum(status.values()):6d} ${cost:7.3f} {dict(status)}")
