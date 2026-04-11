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
from drift.pipeline import apply_drift_pipeline, build_drifted_toolset, candidate_count_for_severity, resolve_mode_sequence
from eval.error_taxonomy import summarize_errors
from eval.metrics import accuracy, compare_tool_calls, recovery_rate, summarize_series
from defense.canonicalizer import render_canonical_tool_card
from inference.openrouter_client import (
    extract_usage,
    request_json_tool_call_with_payload,
    request_tool_call,
    request_tool_call_with_payload,
)
from scripts.common import adapt_gold_call_to_tool, build_example_call, dump_json, load_yaml, prepare_output_dir, synthetic_tools


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
) -> tuple[dict[str, Any], dict[str, int]]:
    if demo:
        return build_example_call(tools[0]), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    model_cfg = config.get("model", {})
    evaluation_cfg = config.get("evaluation", {})
    provider = model_cfg.get("provider", "openrouter")
    if provider != "openrouter":
        raise NotImplementedError(f"Unsupported non-demo provider: {provider}")

    seed = evaluation_cfg.get("seed") if bool(model_cfg.get("use_seed", False)) else None
    call, payload = request_tool_call_with_payload(
        model=str(model_cfg.get("name", "")),
        prompt=prompt,
        tools=tools,
        base_url=model_cfg.get("endpoint"),
        temperature=float(model_cfg.get("temperature", 0.0)),
        seed=seed,
        provider_preferences=model_cfg.get("provider_preferences"),
    )
    return call, extract_usage(payload)


def prepare_original_tools(task: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_tools = task.get("candidate_tools")
    if isinstance(candidate_tools, list) and candidate_tools:
        return deepcopy(candidate_tools)
    return [deepcopy(task["tool"])]


def prepare_drifted_tools(task: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool = deepcopy(task["tool"])
    drift_cfg = config.get("drift", {})
    severity = str(drift_cfg.get("severity", "mild"))

    raw_desc = drift_cfg.get("description_modes")
    if isinstance(raw_desc, list) and len(raw_desc) == 0:
        description_modes: list[str] = []
    else:
        description_modes = resolve_mode_sequence(raw_desc, default="marketing", severity=severity)

    raw_schema = drift_cfg.get("schema_modes")
    if isinstance(raw_schema, list) and len(raw_schema) == 0:
        schema_modes: list[str] = []
    else:
        schema_modes = resolve_mode_sequence(raw_schema, default="rename_parameters", severity=severity)

    raw_candidates = drift_cfg.get("candidate_modes")
    skip_candidates = isinstance(raw_candidates, list) and len(raw_candidates) == 0

    drifted = apply_drift_pipeline(
        tool,
        description_modes=description_modes,
        schema_modes=schema_modes,
    )

    if skip_candidates:
        candidate_tools = task.get("candidate_tools")
        if isinstance(candidate_tools, list) and candidate_tools:
            pool = []
            gold_name = str(task.get("gold_call", {}).get("name", drifted.get("name", "")))
            replaced = False
            for candidate in deepcopy(candidate_tools):
                if str(candidate.get("name", "")) == gold_name and not replaced:
                    pool.append(deepcopy(drifted))
                    replaced = True
                else:
                    pool.append(candidate)
            if not replaced:
                pool.insert(0, deepcopy(drifted))
            toolset = pool
        else:
            toolset = [deepcopy(drifted)]
    else:
        candidate_modes = resolve_mode_sequence(raw_candidates, default="distractors", severity=severity)
        candidate_mode = candidate_modes[0]
        extra_candidates = candidate_count_for_severity(drift_cfg)
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
    ablation_mode: str = "full",
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    from defense.repair_prompt import build_repair_prompt_no_card
    if ablation_mode == "validation_retry":
        repair_prompt = build_repair_prompt_no_card(task, tool, invalid_call, validation.to_dict())
    else:
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
    seed = config.get("evaluation", {}).get("seed") if bool(model_cfg.get("use_seed", False)) else None
    repair_max_tokens = int(model_cfg.get("repair_max_tokens", 512))
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
            max_tokens=repair_max_tokens,
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
            max_tokens=repair_max_tokens,
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
        max_tokens=repair_max_tokens,
    )
    repair_debug["json_fallback_extracted"] = repaired
    repair_debug["json_fallback_raw_output"] = json_fallback_payload
    repaired = force_tool_name(tool, repaired)
    return repaired, repair_prompt, "json_fallback_forced_tool", repair_debug


def build_summary(results: list[dict[str, Any]], *, demo: bool, output_dir: Path, run_id: str) -> dict[str, Any]:
    from eval.metrics import bootstrap_ci

    original_score = accuracy(record["original_match"]["matched"] for record in results)
    drifted_score = accuracy(record["drifted_match"]["matched"] for record in results)
    repaired_score = accuracy(record["repaired_match"]["matched"] for record in results)

    original_values = [float(record["original_match"]["matched"]) for record in results]
    drifted_values = [float(record["drifted_match"]["matched"]) for record in results]
    repaired_values = [float(record["repaired_match"]["matched"]) for record in results]

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

    repair_triggered = sum(1 for record in results if record.get("repair_used", False))
    repair_trigger_rate = repair_triggered / len(results) if results else 0.0

    token_usage_records = [record.get("token_usage", {}) for record in results]
    total_original_tokens = sum(r.get("original_tokens", 0) for r in token_usage_records)
    total_drifted_tokens = sum(r.get("drifted_tokens", 0) for r in token_usage_records)
    total_repair_tokens = sum(r.get("repair_tokens", 0) for r in token_usage_records)

    summary: dict[str, Any] = {
        "benchmark": "dice-bench",
        "demo_mode": demo,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "sample_count": len(results),
        "scoring_policy": "normalized_semantic_match_v1",
        "original_score": original_score,
        "drifted_score": drifted_score,
        "repaired_score": repaired_score,
        "original_ci_95": list(bootstrap_ci(original_values)),
        "drifted_ci_95": list(bootstrap_ci(drifted_values)),
        "repaired_ci_95": list(bootstrap_ci(repaired_values)),
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
        "repair_trigger_rate": repair_trigger_rate,
        "repair_triggered_count": repair_triggered,
        "total_original_tokens": total_original_tokens,
        "total_drifted_tokens": total_drifted_tokens,
        "total_repair_tokens": total_repair_tokens,
    }

    has_naive = any("naive_retry_match" in record for record in results)
    if has_naive:
        naive_score = accuracy(record["naive_retry_match"]["matched"] for record in results if "naive_retry_match" in record)
        naive_on_correct = accuracy(
            record["naive_retry_match"]["matched"] for record in originally_correct if "naive_retry_match" in record
        )
        summary["naive_retry_score"] = naive_score
        summary["naive_retry_score_on_originally_correct"] = naive_on_correct

    return summary


def _maybe_prepend_card(prompt: str, tool: dict[str, Any], ablation_mode: str) -> str:
    if ablation_mode == "card_only":
        card_text = render_canonical_tool_card(tool)
        return f"{card_text}\n\n{prompt}"
    return prompt


def run(config: dict[str, Any], demo: bool = False) -> dict[str, Any]:
    sample_count = int(config.get("evaluation", {}).get("sample_count", 8))
    eval_cfg = config.get("evaluation", {})
    run_naive_retry = bool(eval_cfg.get("run_naive_retry", False))
    ablation_mode = str(eval_cfg.get("ablation_mode", "full"))
    use_repair = bool(eval_cfg.get("use_repair", True)) and ablation_mode != "card_only"

    output_dir, run_id = prepare_output_dir(config, demo)
    if demo:
        tasks = synthetic_dice_tasks(sample_count)
    else:
        tasks = load_dice_tasks(config)
    results = []

    for task in tasks:
        original_tools = prepare_original_tools(task)
        original_call, original_usage = predict_call(
            prompt=task["prompt"],
            tools=original_tools,
            config=config,
            demo=demo,
        )

        drifted_tool, tools = prepare_drifted_tools(task, config)
        drifted_gold_call = adapt_gold_call_to_tool(task["gold_call"], drifted_tool)
        drifted_eval_tool = drifted_tool

        drifted_prompt = _maybe_prepend_card(task["prompt"], drifted_eval_tool, ablation_mode)

        if demo:
            call = deepcopy(original_call)
            drifted_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            call, drifted_usage = predict_call(
                prompt=drifted_prompt,
                tools=tools,
                config=config,
                demo=demo,
            )

        validation = validate_tool_call(drifted_eval_tool, call)
        repair_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not use_repair or validation.valid:
            repaired = call
            repair_prompt_text = None
            repair_strategy = "not_needed"
            repair_debug = None
        else:
            repaired, repair_prompt_text, repair_strategy, repair_debug = repair_call(
                task=task,
                tool=drifted_eval_tool,
                invalid_call=call,
                validation=validation,
                config=config,
                demo=demo,
                ablation_mode=ablation_mode,
            )

        repaired_validation = validate_tool_call(drifted_eval_tool, repaired)
        original_match = compare_tool_calls(original_call, task["gold_call"], task["tool"])
        drifted_match = compare_tool_calls(call, drifted_gold_call, drifted_eval_tool)
        repaired_match = compare_tool_calls(repaired, drifted_gold_call, drifted_eval_tool)

        record: dict[str, Any] = {
            "id": task["id"],
            "prompt": task["prompt"],
            "original_tool": task["tool"]["name"],
            "original_tool_schema": deepcopy(task["tool"]),
            "drifted_tool": drifted_eval_tool["name"],
            "drifted_tool_schema": deepcopy(drifted_eval_tool),
            "original_call": original_call,
            "validation": validation.to_dict(),
            "repaired_validation": repaired_validation.to_dict(),
            "repaired_valid": repaired_validation.valid,
            "repair_used": use_repair and not validation.valid,
            "repair_strategy": repair_strategy,
            "repair_prompt": repair_prompt_text,
            "repair_debug": repair_debug,
            "gold_call": task["gold_call"],
            "drifted_gold_call": drifted_gold_call,
            "pred_call": call,
            "repaired_call": repaired,
            "original_match": original_match,
            "drifted_match": drifted_match,
            "repaired_match": repaired_match,
            "error_type": "clean" if validation.valid else "multi_turn_drift",
            "token_usage": {
                "original_tokens": original_usage.get("total_tokens", 0),
                "drifted_tokens": drifted_usage.get("total_tokens", 0),
                "repair_tokens": repair_usage.get("total_tokens", 0),
            },
        }

        if run_naive_retry and not demo:
            naive_call, naive_usage = predict_call(
                prompt=task["prompt"],
                tools=tools,
                config=config,
                demo=demo,
            )
            record["naive_retry_call"] = naive_call
            record["naive_retry_match"] = compare_tool_calls(naive_call, drifted_gold_call, drifted_eval_tool)
            record["token_usage"]["naive_retry_tokens"] = naive_usage.get("total_tokens", 0)

        results.append(record)

    summary = build_summary(results, demo=demo, output_dir=output_dir, run_id=run_id)
    dump_json(output_dir / "dice_results.json", {"summary": summary, "config": config, "results": results})
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
