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
from collections import Counter
from math import comb
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


def discordant_counts(left: list[bool], right: list[bool]) -> tuple[int, int]:
    left_only = sum(bool(l) and (not bool(r)) for l, r in zip(left, right, strict=True))
    right_only = sum((not bool(l)) and bool(r) for l, r in zip(left, right, strict=True))
    return left_only, right_only


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


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
    print(f"  Repair target mode: {summary.get('repair_target_mode', 'oracle_target')}")

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

    if "originally_incorrect_count" in summary:
        print(f"\n  Originally incorrect: {summary['originally_incorrect_count']}/{n}")
        print(
            "  Repair improvements over drifted: "
            f"{summary.get('repair_improvements_total', 0)} "
            f"(drift recovery={summary.get('repair_improvements_from_drift_recovery', 0)}, "
            f"baseline patching={summary.get('repair_improvements_from_baseline_patching', 0)})"
        )
    strategy_counts = Counter(r.get("repair_strategy", "unknown") for r in results)
    print(f"\n  Repair strategies: {dict(strategy_counts)}")


def analyze_repair_overhead(data: dict[str, Any]) -> None:
    results = data["results"]
    summary = data["summary"]
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
    if "avg_original_latency_ms" in summary:
        print(f"  Avg original latency: {summary['avg_original_latency_ms']:.1f} ms")
        print(f"  Avg drifted latency:  {summary['avg_drifted_latency_ms']:.1f} ms")
        print(f"  Avg repair latency:   {summary['avg_repair_latency_ms']:.1f} ms")
    unresolved = summary.get("repair_target_unresolved_count")
    if unresolved is not None:
        print(f"  Unresolved repair targets: {unresolved}")
    if total_naive > 0:
        print(f"  Total naive retry tokens: {total_naive:,}")
        if "avg_naive_retry_latency_ms" in summary:
            print(f"  Avg naive retry latency: {summary['avg_naive_retry_latency_ms']:.1f} ms")
    if total_drift > 0:
        overhead = total_repair / total_drift
        print(f"  Repair token overhead: {overhead:.1%} of drifted tokens")
    if "avg_repair_latency_ms" in summary and "repair_trigger_rate" in summary:
        expected_added_latency = summary["avg_repair_latency_ms"] * summary["repair_trigger_rate"]
        print(f"  Expected added latency per example: {expected_added_latency:.1f} ms")


def analyze_pairwise_significance(data: dict[str, Any]) -> None:
    results = data["results"]
    drifted = [bool(r["drifted_match"]["matched"]) for r in results]
    repaired = [bool(r["repaired_match"]["matched"]) for r in results]
    drift_only, repair_only = discordant_counts(drifted, repaired)
    print(f"\n--- Paired Significance ---")
    print(
        "  Drifted vs repaired: "
        f"repair_only={repair_only}, drift_only={drift_only}, "
        f"exact McNemar p={exact_mcnemar_pvalue(drift_only, repair_only):.3g}"
    )

    if all("naive_retry_match" in r for r in results):
        naive = [bool(r["naive_retry_match"]["matched"]) for r in results]
        naive_only, repair_only_vs_naive = discordant_counts(naive, repaired)
        print(
            "  Naive retry vs repaired: "
            f"repair_only={repair_only_vs_naive}, naive_only={naive_only}, "
            f"exact McNemar p={exact_mcnemar_pvalue(naive_only, repair_only_vs_naive):.3g}"
        )


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
        analyze_pairwise_significance(data)
        analyze_repair_beyond_drift(data)

    if args.drift_ablation:
        analyze_drift_ablation([Path(p) for p in args.drift_ablation])

    if args.component_ablation:
        analyze_component_ablation([Path(p) for p in args.component_ablation])


if __name__ == "__main__":
    main()
