from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from benchmarks.dice_adapter import load_dice_tasks
from defense.repair_prompt import build_repair_prompt
from defense.validator import validate_tool_call
from drift.candidate_drift import apply_candidate_drift
from drift.description_drift import apply_description_drift
from drift.schema_drift import apply_schema_drift
from eval.error_taxonomy import summarize_errors
from eval.metrics import exact_match_rate, recovery_rate, summarize_series
from inference.openrouter_client import request_tool_call
from scripts.common import adapt_gold_call_to_tool, build_example_call, default_output_dir, dump_json, ensure_dir, load_yaml, synthetic_tools


def synthetic_dice_tasks(sample_count: int) -> list[dict[str, Any]]:
    tools = synthetic_tools()
    prompts = [
        {"prompt": "We need to book a flight and then check the weather at the destination.", "tool": tools[0]},
        {"prompt": "Set up a meeting after confirming city weather.", "tool": tools[2]},
        {"prompt": "Get a forecast, then plan the trip.", "tool": tools[1]},
    ]
    result = []
    for idx in range(sample_count):
        item = deepcopy(prompts[idx % len(prompts)])
        item["id"] = f"dice_{idx}"
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
    description_mode = list(drift_cfg.get("description_modes", ["marketing"]))[0]
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
) -> tuple[dict[str, Any], str]:
    repair_prompt = build_repair_prompt(task, tool, invalid_call, validation.to_dict())
    if demo:
        return build_example_call(tool), repair_prompt

    model_cfg = config.get("model", {})
    repaired = request_tool_call(
        model=str(model_cfg.get("name", "")),
        prompt=repair_prompt,
        tools=[tool],
        base_url=model_cfg.get("endpoint"),
    )
    return repaired, repair_prompt


def run(config: dict[str, Any], demo: bool = False) -> dict[str, Any]:
    sample_count = int(config.get("evaluation", {}).get("sample_count", 8))
    output_dir = ensure_dir(default_output_dir(config))
    if demo:
        tasks = synthetic_dice_tasks(sample_count)
    else:
        tasks = load_dice_tasks(config)
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
        if demo:
            call = deepcopy(original_call)
        else:
            call = predict_call(
                prompt=task["prompt"],
                tools=tools,
                config=config,
                demo=demo,
            )
        validation = validate_tool_call(tools[0], call)
        if validation.valid:
            repaired = call
            repair_prompt = None
        else:
            repaired, repair_prompt = repair_call(
                task=task,
                tool=tools[0],
                invalid_call=call,
                validation=validation,
                config=config,
                demo=demo,
            )
        repaired_validation = validate_tool_call(tools[0], repaired)
        results.append(
            {
                "id": task["id"],
                "prompt": task["prompt"],
                "original_tool": task["tool"]["name"],
                "drifted_tool": tools[0]["name"],
                "original_call": original_call,
                "validation": validation.to_dict(),
                "repaired_validation": repaired_validation.to_dict(),
                "repaired_valid": repaired_validation.valid,
                "repair_used": not validation.valid,
                "repair_prompt": repair_prompt,
                "gold_call": task["gold_call"],
                "drifted_gold_call": drifted_gold_call,
                "pred_call": call,
                "repaired_call": repaired,
                "error_type": "clean" if validation.valid else "multi_turn_drift",
            }
        )

    original_score = exact_match_rate(
        (record["original_call"] for record in results),
        (record["gold_call"] for record in results),
    )
    drifted_score = exact_match_rate(
        (record["pred_call"] for record in results),
        (record["drifted_gold_call"] for record in results),
    )
    repaired_score = exact_match_rate(
        (record["repaired_call"] for record in results),
        (record["drifted_gold_call"] for record in results),
    )
    summary = {
        "benchmark": "dice-bench",
        "demo_mode": demo,
        "sample_count": len(results),
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
    dump_json(output_dir / "dice_results.json", {"summary": summary, "results": results})
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
