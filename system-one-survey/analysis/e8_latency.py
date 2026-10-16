"""E8 (light): latency and cost from the logs of E2, B1 and E3. No new calls.

  uv run python analysis/e8_latency.py

The latency is the client time of one HTTP call from Bangalore through OpenRouter: it
includes the network and the router. It is not the latency of the model alone.
  p50, p95        over all E2 calls of a system (all tasks)
  ms / 1k tok     the slope of latency against input tokens (least squares)
  7 vs 1 q        Civil Comments: median latency of the call with 7 questions (E2)
                  divided by the median latency of a call with 1 question (E3 "alone")
Writes results/e8_latency.csv.
"""
import csv
import json

import numpy as np

from common import ROOT, SYSTEM_LABEL, SYSTEM_TWO, file_name


def calls(path, keep=lambda r: True):
    out = []
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] == 200 and r.get("latency_ms") and keep(r):
            u = r["response"].get("usage") or {}
            out.append((r["latency_ms"], u.get("input_tokens", u.get("prompt_tokens")) or 0, u.get("cost") or 0.0, r["task"]))
    return out


def main():
    e2_models = [line.strip() for line in open(ROOT / "runs" / "e2" / "models.txt") if line.strip()]
    rows = []
    for model, tag in [(m, "e2") for m in e2_models] + [(m, "b1") for m in SYSTEM_TWO]:
        c = calls(ROOT / "runs" / tag / file_name(model))
        if not c:
            continue
        lat = np.array([x[0] for x in c])
        tok = np.array([x[1] for x in c], dtype=float)
        slope = np.polyfit(tok / 1000, lat, 1)[0] if tok.std() > 0 else float("nan")
        row = {"system": model, "label": SYSTEM_LABEL.get(model, model), "calls": len(c),
               "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
               "ms_per_1k_tokens": float(slope), "median_input_tokens": float(np.median(tok)),
               "usd_per_1k_calls": 1000 * float(np.mean([x[2] for x in c]))}
        multi = [x[0] for x in c if x[3] == "civil"]
        single = [x[0] for x in calls(ROOT / "runs" / "e3" / file_name(model),
                                      lambda r: r["task"] == "civil" and "#alone-" in r["item_id"])]
        row["civil_7q_over_1q"] = float(np.median(multi) / np.median(single)) if multi and single else ""
        rows.append(row)
    with open(ROOT / "results" / "e8_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| System | Calls | p50 ms | p95 ms | ms per 1k input tokens | Median input tokens | $ per 1k calls | 7 q / 1 q |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ratio = f"{r['civil_7q_over_1q']:.2f}" if r["civil_7q_over_1q"] != "" else "-"
        print(f"| {r['label']} | {r['calls']} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['ms_per_1k_tokens']:.0f} | "
              f"{r['median_input_tokens']:.0f} | {r['usd_per_1k_calls']:.3f} | {ratio} |")


if __name__ == "__main__":
    main()
