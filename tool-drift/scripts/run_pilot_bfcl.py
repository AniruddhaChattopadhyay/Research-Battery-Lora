from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from benchmarks.bfcl_adapter import load_bfcl_tasks
from defense.repair_prompt import build_repair_prompt
from defense.validator import validate_tool_call
from drift.candidate_drift import apply_candidate_drift
from drift.description_drift import apply_description_drift
from drift.schema_drift import apply_schema_drift
from eval.error_taxonomy import summarize_errors
from eval.metrics import accuracy, compare_tool_calls, recovery_rate, summarize_series
from inference.openrouter_client import request_json_tool_call, request_tool_call
from scripts.common import adapt_gold_call_to_tool, build_example_call, dump_json, load_yaml, prepare_output_dir, synthetic_tools


def synthetic_tasks(kind: str, sample_count: int) -> list[dict[str, Any]]:
    tools = synthetic_tools()
    if kind == "bfcl":
        prompts = [
            {"prompt": "Book a flight from Paris to Tokyo for next Monday in economy class.", "tool": tools[0]},
            {"prompt": "What is the weather in Trento this weekend?", "tool": tools[1]},
            {"prompt": "Schedule a team sync with Alice and Bob at 3 PM for 30 minutes.", "tool": tools[2]},
        ]
    else:
        prompts = [{"prompt": f"Demo task {i}", "tool": deepcopy(tools[i % len(tools)])} for i in range(sample_count)]
    result = []
    for idx in range(sample_count):
        item = deepcopy(prompts[idx % len(prompts)])
        item["id"] = f"{kind}_{idx}"
        item["gold_call"] = build_example_call(item["tool"])
        result.append(item)
    return result


def predict_call(
    *,
    prompt: str,
    tools: list[dict[str, Any]],
    config: dict[str, Any],
    demo: bool,
) -> dict[str, Any]:
    if demo:
        return build_example_call(tools[0])

    model_cfg = config.get("model", {})
    provider = model_cfg.get("provider", "openrouter")
    if provider != "openrouter":
        raise NotImplementedError(f"Unsupported non-demo provider: {provider}")

    return request_tool_call(
        model=str(model_cfg.get("name", "")),
        prompt=prompt,
        tools=tools,
        base_url=model_cfg.get("endpoint"),
    )


def prepare_original_tools(task: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_tools = task.get("candidate_tools")
    if isinstance(candidate_tools, list) and candidate_tools:
        return deepcopy(candidate_tools)
    return [deepcopy(task["tool"])]


def prepare_drifted_tools(task: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool = deepcopy(task["tool"])
    drift_cfg = config.get("drift", {})
    description_mode = list(drift_cfg.get("description_modes", ["verbose"]))[0]
    schema_mode = list(drift_cfg.get("schema_modes", ["rename_parameters"]))[0]
    candidate_mode = list(drift_cfg.get("candidate_modes", ["distractors"]))[0]

    drifted = apply_description_drift(tool, description_mode)
    drifted = apply_schema_drift(drifted, schema_mode)

    candidate_tools = task.get("candidate_tools")
    if isinstance(candidate_tools, list) and candidate_tools:
        pool = []
        gold_name = str(task.get("gold_call", {}).get("name", tool.get("name", "")))
        replaced = False
        for candidate in deepcopy(candidate_tools):
            if str(candidate.get("name", "")) == gold_name and not replaced:
                pool.append(deepcopy(drifted))
                replaced = True
            else:
                pool.append(candidate)
        if not replaced:
            pool.insert(0, deepcopy(drifted))
        return drifted, apply_candidate_drift(pool, candidate_mode)

    return drifted, apply_candidate_drift([drifted], candidate_mode)


def repair_call(
    *,
    task: dict[str, Any],
    tool: dict[str, Any],
    invalid_call: dict[str, Any],
    validation: Any,
    config: dict[str, Any],
    demo: bool,
) -> tuple[dict[str, Any], str, str]:
    repair_prompt = build_repair_prompt(task, tool, invalid_call, validation.to_dict())
    if demo:
        return build_example_call(tool), repair_prompt, "demo"

    model_cfg = config.get("model", {})
    repaired = request_tool_call(
        model=str(model_cfg.get("name", "")),
        prompt=repair_prompt,
        tools=[tool],
        base_url=model_cfg.get("endpoint"),
    )
    first_pass = validate_tool_call(tool, repaired)
    if first_pass.valid:
        return repaired, repair_prompt, "tool_call"

    repaired = request_json_tool_call(
        model=str(model_cfg.get("name", "")),
        prompt=repair_prompt,
        base_url=model_cfg.get("endpoint"),
    )
    return repaired, repair_prompt, "json_fallback"


def run(config: dict[str, Any], demo: bool = False) -> dict[str, Any]:
    sample_count = int(config.get("evaluation", {}).get("sample_count", 8))
    output_dir, run_id = prepare_output_dir(config, demo)
    if demo:
        tasks = synthetic_tasks("bfcl", sample_count)
    else:
        tasks = load_bfcl_tasks(config)
    results = []

    for task in tasks:
        original_tools = prepare_original_tools(task)
        original_call = predict_call(
            prompt=task["prompt"],
            tools=original_tools,
            config=config,
            demo=demo,
        )

        drifted_tool, tools = prepare_drifted_tools(task, config)
        drifted_gold_call = adapt_gold_call_to_tool(task["gold_call"], drifted_tool)
        drifted_eval_tool = tools[0]
        if demo:
            call = deepcopy(original_call)
        else:
            call = predict_call(
                prompt=task["prompt"],
                tools=tools,
                config=config,
                demo=demo,
            )
        validation = validate_tool_call(drifted_eval_tool, call)
        if validation.valid:
            repaired = call
            repair_prompt = None
            repair_strategy = "not_needed"
        else:
            repaired, repair_prompt, repair_strategy = repair_call(
                task=task,
                tool=drifted_eval_tool,
                invalid_call=call,
                validation=validation,
                config=config,
                demo=demo,
            )
        repaired_validation = validate_tool_call(drifted_eval_tool, repaired)
        original_match = compare_tool_calls(original_call, task["gold_call"], task["tool"])
        drifted_match = compare_tool_calls(call, drifted_gold_call, drifted_eval_tool)
        repaired_match = compare_tool_calls(repaired, drifted_gold_call, drifted_eval_tool)
        results.append(
            {
                "id": task["id"],
                "prompt": task["prompt"],
                "original_tool": task["tool"]["name"],
                "drifted_tool": drifted_eval_tool["name"],
                "original_call": original_call,
                "validation": validation.to_dict(),
                "repaired_validation": repaired_validation.to_dict(),
                "repaired_valid": repaired_validation.valid,
                "repair_used": not validation.valid,
                "repair_strategy": repair_strategy,
                "repair_prompt": repair_prompt,
                "gold_call": task["gold_call"],
                "drifted_gold_call": drifted_gold_call,
                "pred_call": call,
                "repaired_call": repaired,
                "original_match": original_match,
                "drifted_match": drifted_match,
                "repaired_match": repaired_match,
                "error_type": "clean" if validation.valid else "schema_or_description_drift",
            }
        )

    original_score = accuracy(record["original_match"]["matched"] for record in results)
    drifted_score = accuracy(record["drifted_match"]["matched"] for record in results)
    repaired_score = accuracy(record["repaired_match"]["matched"] for record in results)
    summary = {
        "benchmark": "bfcl",
        "demo_mode": demo,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "sample_count": len(results),
        "scoring_policy": "normalized_semantic_match_v1",
        "original_score": original_score,
        "drifted_score": drifted_score,
        "repaired_score": repaired_score,
        "recovery_rate": recovery_rate(original_score, drifted_score, repaired_score),
        "validation_failures": sum(not record["validation"]["valid"] for record in results),
        "repaired_failures": sum(not record["repaired_validation"]["valid"] for record in results),
        "error_breakdown": summarize_errors(results),
        "drifted_stats": summarize_series([drifted_score]),
        "repaired_stats": summarize_series([repaired_score]),
    }
    dump_json(output_dir / "bfcl_results.json", {"summary": summary, "config": config, "results": results})
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    run(config, demo=args.demo or bool(config.get("pilot", {}).get("demo_mode", False)))


if __name__ == "__main__":
    main()
