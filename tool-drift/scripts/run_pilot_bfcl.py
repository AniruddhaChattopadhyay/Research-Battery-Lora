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
from drift.pipeline import apply_drift_pipeline, build_drifted_toolset, candidate_count_for_severity, resolve_mode_sequence
from eval.error_taxonomy import summarize_errors
from eval.metrics import accuracy, compare_tool_calls, recovery_rate, summarize_series
from inference.openrouter_client import (
    request_json_tool_call_with_payload,
    request_tool_call,
    request_tool_call_with_payload,
)
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
    evaluation_cfg = config.get("evaluation", {})
    provider = model_cfg.get("provider", "openrouter")
    if provider != "openrouter":
        raise NotImplementedError(f"Unsupported non-demo provider: {provider}")

    seed = evaluation_cfg.get("seed") if bool(model_cfg.get("use_seed", False)) else None
    return request_tool_call(
        model=str(model_cfg.get("name", "")),
        prompt=prompt,
        tools=tools,
        base_url=model_cfg.get("endpoint"),
        temperature=float(model_cfg.get("temperature", 0.0)),
        seed=seed,
        provider_preferences=model_cfg.get("provider_preferences"),
    )


def prepare_original_tools(task: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_tools = task.get("candidate_tools")
    if isinstance(candidate_tools, list) and candidate_tools:
        return deepcopy(candidate_tools)
    return [deepcopy(task["tool"])]


def prepare_drifted_tools(task: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool = deepcopy(task["tool"])
    drift_cfg = config.get("drift", {})
    severity = str(drift_cfg.get("severity", "mild"))
    description_modes = resolve_mode_sequence(
        drift_cfg.get("description_modes"),
        default="verbose",
        severity=severity,
    )
    schema_modes = resolve_mode_sequence(
        drift_cfg.get("schema_modes"),
        default="rename_parameters",
        severity=severity,
    )
    candidate_modes = resolve_mode_sequence(
        drift_cfg.get("candidate_modes"),
        default="distractors",
        severity=severity,
    )
    candidate_mode = candidate_modes[0]
    extra_candidates = candidate_count_for_severity(drift_cfg)

    drifted = apply_drift_pipeline(
        tool,
        description_modes=description_modes,
        schema_modes=schema_modes,
    )
    toolset = build_drifted_toolset(
        task=task,
        drifted_tool=drifted,
        candidate_mode=candidate_mode,
        extra_candidates=extra_candidates,
    )
    return drifted, toolset


def force_tool_name(tool: dict[str, Any], tool_call: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(tool.get("name", "")),
        "arguments": dict(tool_call.get("arguments", {})),
    }


def repair_call(
    *,
    task: dict[str, Any],
    tool: dict[str, Any],
    invalid_call: dict[str, Any],
    validation: Any,
    config: dict[str, Any],
    demo: bool,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    repair_prompt = build_repair_prompt(task, tool, invalid_call, validation.to_dict())
    target_tool_name = str(tool.get("name", ""))
    repair_debug: dict[str, Any] = {
        "target_tool_name": target_tool_name,
        "tool_choice_fallback_error": None,
        "tool_call_extracted": None,
        "tool_call_raw_output": None,
        "json_fallback_extracted": None,
        "json_fallback_raw_output": None,
    }
    if demo:
        return build_example_call(tool), repair_prompt, "demo", repair_debug

    model_cfg = config.get("model", {})
    evaluation_cfg = config.get("evaluation", {})
    seed = evaluation_cfg.get("seed") if bool(model_cfg.get("use_seed", False)) else None
    try:
        repaired, tool_call_payload = request_tool_call_with_payload(
            model=str(model_cfg.get("name", "")),
            prompt=repair_prompt,
            tools=[tool],
            base_url=model_cfg.get("endpoint"),
            force_tool_name=target_tool_name,
            temperature=float(model_cfg.get("temperature", 0.0)),
            seed=seed,
            provider_preferences=model_cfg.get("provider_preferences"),
        )
    except RuntimeError as exc:
        if "tool_choice" not in str(exc):
            raise
        repair_debug["tool_choice_fallback_error"] = str(exc)
        repaired, tool_call_payload = request_tool_call_with_payload(
            model=str(model_cfg.get("name", "")),
            prompt=repair_prompt,
            tools=[tool],
            base_url=model_cfg.get("endpoint"),
            temperature=float(model_cfg.get("temperature", 0.0)),
            seed=seed,
            provider_preferences=model_cfg.get("provider_preferences"),
        )
    repair_debug["tool_call_extracted"] = repaired
    repair_debug["tool_call_raw_output"] = tool_call_payload
    repaired = force_tool_name(tool, repaired)
    first_pass = validate_tool_call(tool, repaired)
    if first_pass.valid:
        return repaired, repair_prompt, "forced_tool_call", repair_debug

    repaired, json_fallback_payload = request_json_tool_call_with_payload(
        model=str(model_cfg.get("name", "")),
        prompt=repair_prompt,
        base_url=model_cfg.get("endpoint"),
        temperature=float(model_cfg.get("temperature", 0.0)),
        seed=seed,
        provider_preferences=model_cfg.get("provider_preferences"),
    )
    repair_debug["json_fallback_extracted"] = repaired
    repair_debug["json_fallback_raw_output"] = json_fallback_payload
    repaired = force_tool_name(tool, repaired)
    return repaired, repair_prompt, "json_fallback_forced_tool", repair_debug


def build_summary(results: list[dict[str, Any]], *, demo: bool, output_dir: Path, run_id: str) -> dict[str, Any]:
    original_score = accuracy(record["original_match"]["matched"] for record in results)
    drifted_score = accuracy(record["drifted_match"]["matched"] for record in results)
    repaired_score = accuracy(record["repaired_match"]["matched"] for record in results)

    originally_correct = [record for record in results if record["original_match"]["matched"]]
    drifted_score_on_originally_correct = accuracy(
        record["drifted_match"]["matched"] for record in originally_correct
    )
    repaired_score_on_originally_correct = accuracy(
        record["repaired_match"]["matched"] for record in originally_correct
    )
    drift_misses_on_originally_correct = sum(
        not record["drifted_match"]["matched"] for record in originally_correct
    )
    repair_recoveries_on_originally_correct = sum(
        (not record["drifted_match"]["matched"]) and record["repaired_match"]["matched"]
        for record in originally_correct
    )
    repair_harms_on_originally_correct = sum(
        record["drifted_match"]["matched"] and (not record["repaired_match"]["matched"])
        for record in originally_correct
    )

    return {
        "benchmark": "bfcl",
        "demo_mode": demo,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "sample_count": len(results),
        "scoring_policy": "normalized_semantic_match_v2",
        "original_score": original_score,
        "drifted_score": drifted_score,
        "repaired_score": repaired_score,
        "recovery_rate": recovery_rate(original_score, drifted_score, repaired_score),
        "validation_failures": sum(not record["validation"]["valid"] for record in results),
        "repaired_failures": sum(not record["repaired_validation"]["valid"] for record in results),
        "error_breakdown": summarize_errors(results),
        "drifted_stats": summarize_series([drifted_score]),
        "repaired_stats": summarize_series([repaired_score]),
        "originally_correct_count": len(originally_correct),
        "drifted_score_on_originally_correct": drifted_score_on_originally_correct,
        "repaired_score_on_originally_correct": repaired_score_on_originally_correct,
        "recovery_rate_on_originally_correct": recovery_rate(
            1.0,
            drifted_score_on_originally_correct,
            repaired_score_on_originally_correct,
        ),
        "drift_misses_on_originally_correct": drift_misses_on_originally_correct,
        "repair_recoveries_on_originally_correct": repair_recoveries_on_originally_correct,
        "repair_harms_on_originally_correct": repair_harms_on_originally_correct,
    }


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
        drifted_eval_tool = drifted_tool
        candidate_tool_names = [str(candidate.get("name", "")) for candidate in tools]
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
            repair_debug = None
        else:
            repaired, repair_prompt, repair_strategy, repair_debug = repair_call(
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
                "original_tool_schema": deepcopy(task["tool"]),
                "drifted_tool": drifted_eval_tool["name"],
                "drifted_tool_schema": deepcopy(drifted_eval_tool),
                "candidate_tool_names": candidate_tool_names,
                "original_call": original_call,
                "validation": validation.to_dict(),
                "repaired_validation": repaired_validation.to_dict(),
                "repaired_valid": repaired_validation.valid,
                "repair_used": not validation.valid,
                "repair_strategy": repair_strategy,
                "repair_prompt": repair_prompt,
                "repair_debug": repair_debug,
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

    summary = build_summary(results, demo=demo, output_dir=output_dir, run_id=run_id)
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
