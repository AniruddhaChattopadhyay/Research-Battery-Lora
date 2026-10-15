"""Summarize one run (for example E0) as a markdown table: one row for each model.

Usage:
  uv run python analysis/audit.py runs/e0

Columns:
  resolved    the model id and version that OpenRouter reports in the response
  ok          calls with status 200, of all calls
  errors      the other status codes, with counts
  p50, p95    latency in ms of the calls with status 200 (network included)
  in tok      mean input tokens for each call
  $/1k calls  mean cost for each 1,000 calls, from usage.cost
  usable      answers that have the fields for our metrics (a noul value, or
              probabilities that sum to 1 for choice and score)
  sanity acc  accuracy on the few items of the run. It only shows that the
              output is not broken. It is not a result.
"""
import json
import pathlib
import statistics
import sys
from collections import Counter


def usable(qtype, ans):
    if not isinstance(ans, dict):
        return False
    if qtype == "noul":
        value = ans.get("noul")
        return isinstance(value, (int, float)) and 0 <= value <= 1
    probs = ans.get("probabilities")
    return isinstance(probs, dict) and len(probs) > 0 and abs(sum(probs.values()) - 1) < 0.05


def correct(qtype, ans, gold):
    if qtype == "noul":
        target = gold >= 0.5 if isinstance(gold, float) else bool(gold)
        return (ans["noul"] >= 0.5) == target
    if qtype == "choice":
        return ans.get("choice") == gold
    return round(ans.get("score", -9)) == gold


def summarize(path):
    recs = [json.loads(line) for line in path.open(encoding="utf-8")]
    ok = [r for r in recs if r["status"] == 200]
    errors = Counter(str(r["status"]) for r in recs if r["status"] != 200)
    lat = sorted(r["latency_ms"] for r in ok)
    n_ans = n_usable = n_right = 0
    for r in ok:
        answers = r["response"].get("answers", {})
        for qname, q in r["request"]["questions"].items():
            n_ans += 1
            ans = answers.get(qname)
            if usable(q["type"], ans):
                n_usable += 1
                n_right += correct(q["type"], ans, r["gold"][qname])
    usage = [r["response"].get("usage", {}) for r in ok]
    return {
        "model": recs[0]["model"],
        "resolved": ", ".join(sorted({r["response"].get("model", "?") for r in ok})) or "-",
        "provider": ", ".join(sorted({str(r["response"].get("provider", "?")) for r in ok})) or "-",
        "ok": f"{len(ok)}/{len(recs)}",
        "errors": ", ".join(f"{k}×{v}" for k, v in errors.items()) or "-",
        "p50": f"{statistics.median(lat):.0f}" if lat else "-",
        "p95": f"{lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.0f}" if lat else "-",
        "in_tok": f"{statistics.mean(u.get('input_tokens', 0) for u in usage):.0f}" if usage else "-",
        "cost_1k": f"{1000 * statistics.mean(u.get('cost', 0) or 0 for u in usage):.3f}" if usage else "-",
        "usable": f"{100 * n_usable / n_ans:.0f}%" if n_ans else "-",
        "acc": f"{100 * n_right / n_usable:.0f}%" if n_usable else "-",
    }


def main():
    run_dir = pathlib.Path(sys.argv[1])
    rows = [summarize(p) for p in sorted(run_dir.glob("*.jsonl"))]
    cols = ["model", "resolved", "provider", "ok", "errors", "p50", "p95", "in_tok", "cost_1k", "usable", "acc"]
    head = ["Model", "Resolved", "Provider", "OK", "Errors", "p50 ms", "p95 ms", "In tok", "$/1k calls", "Usable", "Sanity acc"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(f"`{row[c]}`" if c == "model" else row[c] for c in cols) + " |")


if __name__ == "__main__":
    main()
