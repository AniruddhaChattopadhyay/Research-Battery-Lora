"""E2: Jev 1.14 under the calibration metrics of three published studies (no new calls).

  uv run python analysis/e2_metric_replication.py

For each task:
  sce        our signed calibration error (mean confidence minus accuracy)
  ece15      top-label ECE, 15 equal-width bins (our metric)
  ece_mass10 top-label ECE, 10 equal-mass bins (Rafe and Das)
  ece_pyes   ECE of P(yes) for predicates, 15 equal-width bins (Deusser et al.); for choices,
             the top-option ECE, as in their paper; no value for scores
  temp       a temperature refit by NLL on one half of the items, tested on the other half
             (TypedBench): T > 1 means overconfident, T < 1 underconfident
Writes results/e2_metric_replication.csv.
"""
import csv
import zlib

import numpy as np

from common import ROOT, TASK_ORDER, confidences, ece, load_system

TEMPS = np.exp(np.linspace(np.log(0.1), np.log(20), 120))


def scaled(dist, t):
    p = {k: max(v, 1e-4) ** (1 / t) for k, v in dist.items()}
    z = sum(p.values())
    return {k: v / z for k, v in p.items()}


def fit_t(rows):
    def nll(t):
        return -np.mean([np.log(scaled(r["dist"], t).get(r["gold"], 1e-4)) for r in rows])
    return float(min(TEMPS, key=nll))


def main():
    ans = load_system("typesafe/jev-1.14", ("e2",))
    out = []
    for task in TASK_ORDER:
        keys = [k for k, a in ans.items() if k[0] == task and a["dist"] is not None and a["gold"] is not None]
        rows = [ans[k] for k in keys]
        conf, correct = confidences(rows)
        row = {"task": task, "n": len(rows), "acc": correct.mean(), "sce": conf.mean() - correct.mean(),
               "ece15": ece(conf, correct), "ece_mass10": ece(conf, correct, bins=10, mass=True)}
        qtype = rows[0]["type"]
        if qtype == "noul":
            p = np.array([r["dist"]["true"] for r in rows])
            y = np.array([r["gold"] == "true" for r in rows], dtype=float)
            row["ece_pyes"] = ece(p, y)
        elif qtype == "choice":
            row["ece_pyes"] = row["ece15"]
        else:
            row["ece_pyes"] = ""
        # TypedBench: fit on one half, report the two fitted temperatures (2 folds)
        halves = [[r for r, k in zip(rows, keys) if zlib.crc32(k[1].encode()) % 2 == h] for h in (0, 1)]
        temps = [fit_t(h) for h in halves if h]
        row["temp"] = float(np.exp(np.mean(np.log(temps))))
        out.append(row)
    with open(ROOT / "results" / "e2_metric_replication.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
