"""Probe: does a system read the full input? (src/probe_length.py; added on 2026-10-16, exploratory)

  uv run python analysis/probe_length.py

For each system, set and position of the fact: the accuracy (4 options, chance 25%), the
mean probability of the right code, and the position of the fact in tokens. The token
position is an estimate: the input tokens of the item, minus the input tokens of the short
item "length-calib" (the question card), times the share of characters before the fact.
Sets: base (plain text, about 3,050 tokens), long (plain text, about 6,900 tokens) and
array (the base inputs as a one-element array, the form of the OpenRouter image guide).
Writes results/probe_length.csv.
"""
import csv
from collections import defaultdict

import numpy as np

from common import ROOT, SYSTEM_LABEL, answers, file_name, top

MODELS = ["cloudflare/clef", "cloudflare/clef-flash", "cloudflare/clef-omni",
          "typesafe/jev-1.14", "perplexity/pplx-decider-v1.1-27b", "microsoft/microsoft-decision-1"]


def main():
    rows = []
    for model in MODELS:
        a = answers(ROOT / "runs" / "probe-length" / file_name(model))
        calib = a.get(("length", "length-calib", "code"))
        if calib is None:
            continue
        card = calib["input_tokens"] or 0
        by = defaultdict(list)
        for (task, item_id, qname), x in a.items():
            m = x["meta"] or {}
            if m.get("calib") or x["dist"] is None:
                continue
            kind = m.get("set", "base")
            state_tokens = (x["input_tokens"] or 0) - card
            pos = m["pos_chars"] / m["total_chars"] * state_tokens
            key = m.get("target_tokens", m.get("share"))
            by[(kind, key)].append((top(x["dist"]) == x["gold"], x["dist"].get(x["gold"], 0.0), pos, state_tokens))
        for (kind, key), v in sorted(by.items()):
            rows.append({"system": model, "label": SYSTEM_LABEL.get(model, model), "set": kind, "position": key,
                         "n": len(v), "acc": 100 * np.mean([c for c, _, _, _ in v]),
                         "p_gold": float(np.mean([p for _, p, _, _ in v])),
                         "pos_tokens": float(np.mean([q for _, _, q, _ in v])),
                         "input_tokens": float(np.mean([s for _, _, _, s in v])), "card_tokens": card})
    for kind in ("base", "long", "array"):
        part = [r for r in rows if r["set"] == kind]
        if not part:
            continue
        keys = sorted({r["position"] for r in part})
        first = [r for r in part if r["system"] == part[0]["system"]]
        print(f"\nSet '{kind}': accuracy (%) by the position of the fact (chance 25%). "
              f"Input about {first[0]['input_tokens']:.0f} tokens.")
        pos = {r["position"]: r["pos_tokens"] for r in first}
        print("| System | " + " | ".join(f"{pos.get(k, float('nan')):.0f}" for k in keys) + " |")
        print("|---|" + "---:|" * len(keys))
        for model in MODELS:
            cells = {r["position"]: r for r in part if r["system"] == model}
            if cells:
                print(f"| {SYSTEM_LABEL.get(model, model)} | "
                      + " | ".join(f"{cells[k]['acc']:.0f}" if k in cells else "-" for k in keys) + " |")
    with open(ROOT / "results" / "probe_length.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
