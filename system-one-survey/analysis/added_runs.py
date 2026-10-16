"""The time window and the cost of the runs that we added on 2026-10-16 (exploratory).

  uv run python analysis/added_runs.py

Reads all records (also the failed calls) of the added runs. Writes results/added_runs.csv:
one row for each run, with the first and the last call (UTC), the number of calls and the cost.
"""
import csv
import datetime
import glob
import json

from common import ROOT

TAGS = ["e3-names-second", "probe-length", "image-pilot", "e3-image", "e3-image-s2"]


def main():
    rows = []
    for tag in TAGS:
        times, calls, cost = [], 0, 0.0
        for path in glob.glob(str(ROOT / "runs" / tag / "*.jsonl")):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                calls += 1
                times.append(datetime.datetime.fromisoformat(r["time_utc"]))
                usage = (r.get("response") or {}).get("usage") or {}
                cost += usage.get("cost") or 0.0
        if times:
            rows.append({"tag": tag, "first_utc": min(times).isoformat(), "last_utc": max(times).isoformat(),
                         "calls": calls, "cost_usd": round(cost, 4)})
    with open(ROOT / "results" / "added_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["tag", "first_utc", "last_utc", "calls", "cost_usd"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("total cost:", round(sum(r["cost_usd"] for r in rows), 2), "calls:", sum(r["calls"] for r in rows))


if __name__ == "__main__":
    main()
