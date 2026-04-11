"""Post-hoc analysis script for tool-drift experiments.

Reads result JSONs from outputs/ and produces:
1. Confidence intervals for all key metrics
2. Repair trigger rate and token overhead
3. "Repair > Original" investigation
4. Drift-type breakdown table
5. Component ablation table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from eval.metrics import accuracy, bootstrap_ci, recovery_rate


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ci_str(values: list[float]) -> str:
    mean_val = sum(values) / len(values) if values else 0.0
    lo, hi = bootstrap_ci(values)
    return f"{mean_val:.3f} [{lo:.3f}, {hi:.3f}]"


def analyze_main_results(data: dict[str, Any]) -> None:
    results = data["results"]
    summary = data["summary"]
    n = len(results)

    original_values = [float(r["original_match"]["matched"]) for r in results]
    drifted_values = [float(r["drifted_match"]["matched"]) for r in results]
    repaired_values = [float(r["repaired_match"]["matched"]) for r in results]

    print(f"\n{'='*60}")
    print(f"Run: {summary.get('run_id', 'unknown')} | n={n}")
    print(f"{'='*60}")
    print(f"  Original:  {ci_str(original_values)}")
    print(f"  Drifted:   {ci_str(drifted_values)}")
    print(f"  Repaired:  {ci_str(repaired_values)}")

    if "naive_retry_score" in summary:
        naive_values = [float(r["naive_retry_match"]["matched"]) for r in results if "naive_retry_match" in r]
        print(f"  Naive Rtr: {ci_str(naive_values)}")

    originally_correct = [r for r in results if r["original_match"]["matched"]]
    oc_n = len(originally_correct)
    drift_misses = sum(not r["drifted_match"]["matched"] for r in originally_correct)
    recoveries = sum(
        (not r["drifted_match"]["matched"]) and r["repaired_match"]["matched"]
        for r in originally_correct
    )
    harms = sum(
        r["drifted_match"]["matched"] and (not r["repaired_match"]["matched"])
        for r in originally_correct
    )

    print(f"\n  Originally correct: {oc_n}/{n}")
    print(f"  Drift misses on clean slice: {drift_misses}")
    print(f"  Repair recoveries: {recoveries}")
    print(f"  Repair harms: {harms}")


def analyze_repair_overhead(data: dict[str, Any]) -> None:
    results = data["results"]
    n = len(results)

    repair_used = sum(1 for r in results if r.get("repair_used", False))
    trigger_rate = repair_used / n if n else 0.0

    token_records = [r.get("token_usage", {}) for r in results]
    total_orig = sum(r.get("original_tokens", 0) for r in token_records)
    total_drift = sum(r.get("drifted_tokens", 0) for r in token_records)
    total_repair = sum(r.get("repair_tokens", 0) for r in token_records)
    total_naive = sum(r.get("naive_retry_tokens", 0) for r in token_records)

    print(f"\n--- Repair Overhead ---")
    print(f"  Repair trigger rate: {repair_used}/{n} = {trigger_rate:.2%}")
    print(f"  Total original tokens: {total_orig:,}")
    print(f"  Total drifted tokens:  {total_drift:,}")
    print(f"  Total repair tokens:   {total_repair:,}")
    if total_naive > 0:
        print(f"  Total naive retry tokens: {total_naive:,}")
    if total_drift > 0:
        overhead = total_repair / total_drift
        print(f"  Repair token overhead: {overhead:.1%} of drifted tokens")


def analyze_repair_beyond_drift(data: dict[str, Any]) -> None:
    results = data["results"]
    bonus = []
    for r in results:
        if not r["original_match"]["matched"] and r["repaired_match"]["matched"]:
            bonus.append(r)

    print(f"\n--- Repair > Original Investigation ---")
    print(f"  Examples where repair fixed original failures: {len(bonus)}")

    if not bonus:
        return

    for r in bonus:
        om = r["original_match"]
        issues = []
        if not om["name_match"]:
            issues.append("wrong_tool")
        if om["missing_fields"]:
            issues.append(f"missing:{','.join(om['missing_fields'])}")
        if om["extra_fields"]:
            issues.append(f"extra:{','.join(om['extra_fields'])}")
        if om["mismatched_fields"]:
            issues.append(f"mismatch:{','.join(om['mismatched_fields'])}")
        print(f"    {r['id']}: {' | '.join(issues) or 'unknown'}")


def analyze_drift_ablation(paths: list[Path]) -> None:
    print(f"\n{'='*60}")
    print("Drift-Type Ablation")
    print(f"{'='*60}")
    print(f"  {'Drift Type':<25} {'Orig':>8} {'Drift':>8} {'Repair':>8} {'Delta':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for path in paths:
        data = load_results(path)
        s = data["summary"]
        label = s.get("output_dir", path.stem).split("/")[-1] if "/" in s.get("output_dir", "") else path.stem
        label = label.replace("dice_ablation_", "").replace("dice_results", path.parent.name)
        orig = s["original_score"]
        drift = s["drifted_score"]
        repair = s["repaired_score"]
        delta = drift - orig
        print(f"  {label:<25} {orig:>8.3f} {drift:>8.3f} {repair:>8.3f} {delta:>+8.3f}")


def analyze_component_ablation(paths: list[Path]) -> None:
    print(f"\n{'='*60}")
    print("Component Ablation")
    print(f"{'='*60}")
    print(f"  {'Component':<25} {'Drift':>8} {'Repair':>8} {'Recov':>8} {'Harms':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for path in paths:
        data = load_results(path)
        s = data["summary"]
        label = path.stem.replace("dice_ablation_", "").replace("dice_results", path.parent.name)
        drift = s["drifted_score"]
        repair = s["repaired_score"]
        recov = s.get("repair_recoveries_on_originally_correct", "?")
        harms = s.get("repair_harms_on_originally_correct", "?")
        print(f"  {label:<25} {drift:>8.3f} {repair:>8.3f} {recov:>8} {harms:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tool-drift experiment results")
    parser.add_argument("--results", nargs="+", required=True, help="Result JSON files to analyze")
    parser.add_argument("--drift-ablation", nargs="*", default=None, help="Drift ablation result files")
    parser.add_argument("--component-ablation", nargs="*", default=None, help="Component ablation result files")
    args = parser.parse_args()

    for path_str in args.results:
        path = Path(path_str)
        data = load_results(path)
        analyze_main_results(data)
        analyze_repair_overhead(data)
        analyze_repair_beyond_drift(data)

    if args.drift_ablation:
        analyze_drift_ablation([Path(p) for p in args.drift_ablation])

    if args.component_ablation:
        analyze_component_ablation([Path(p) for p in args.component_ablation])


if __name__ == "__main__":
    main()
