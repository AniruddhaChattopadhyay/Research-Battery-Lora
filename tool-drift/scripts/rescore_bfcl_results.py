from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from benchmarks.bfcl_adapter import load_bfcl_tasks
from eval.metrics import compare_tool_calls
from scripts.common import dump_json, load_json
from scripts.run_pilot_bfcl import build_summary, prepare_drifted_tools


def _resolve_original_tool(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    saved = record.get("original_tool_schema")
    if isinstance(saved, dict) and saved:
        return saved
    return dict(task["tool"])


def _resolve_drifted_tool(task: dict[str, Any], record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    saved = record.get("drifted_tool_schema")
    if isinstance(saved, dict) and saved:
        return saved
    drifted_tool, _ = prepare_drifted_tools(task, config)
    return drifted_tool


def rescore_payload(payload: dict[str, Any], results_path: Path) -> dict[str, Any]:
    config = dict(payload.get("config", {}))
    tasks = {task["id"]: task for task in load_bfcl_tasks(config)}
    results = list(payload.get("results", []))

    for record in results:
        task = tasks[str(record["id"])]
        original_tool = _resolve_original_tool(task, record)
        drifted_tool = _resolve_drifted_tool(task, record, config)
        record["original_tool_schema"] = original_tool
        record["drifted_tool_schema"] = drifted_tool
        record["original_match"] = compare_tool_calls(record["original_call"], record["gold_call"], original_tool)
        record["drifted_match"] = compare_tool_calls(record["pred_call"], record["drifted_gold_call"], drifted_tool)
        record["repaired_match"] = compare_tool_calls(record["repaired_call"], record["drifted_gold_call"], drifted_tool)

    prior_summary = dict(payload.get("summary", {}))
    output_dir = Path(prior_summary.get("output_dir", results_path.parent))
    run_id = str(prior_summary.get("run_id", results_path.parent.name))
    demo_mode = bool(prior_summary.get("demo_mode", False))
    payload["summary"] = build_summary(results, demo=demo_mode, output_dir=output_dir, run_id=run_id)
    payload["results"] = results
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    payload = load_json(results_path)
    rescored = rescore_payload(payload, results_path)

    if args.in_place:
        dump_json(results_path, rescored)
    elif args.output:
        dump_json(args.output, rescored)

    print(json.dumps(rescored.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
