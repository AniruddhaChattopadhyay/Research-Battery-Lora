from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from scripts.common import dump_json, repo_root


DEFAULT_CATEGORIES = [
    "BFCL_v4_simple_python",
    "BFCL_v4_multiple",
    "BFCL_v4_live_simple",
    "BFCL_v4_live_multiple",
]


TYPE_MAP = {
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
    "bool": "boolean",
    "boolean": "boolean",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "str": "string",
    "string": "string",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_type(value: Any) -> str:
    return TYPE_MAP.get(str(value).strip().lower(), "string")


def normalize_tool(raw_tool: dict[str, Any]) -> dict[str, Any]:
    tool = deepcopy(raw_tool)
    schema = dict(tool.get("parameters", {}))
    properties = dict(schema.get("properties", {}))
    normalized_properties: dict[str, Any] = {}

    for field, raw_spec in properties.items():
        spec = dict(raw_spec)
        spec["type"] = normalize_type(spec.get("type", "string"))
        normalized_properties[str(field)] = spec

    schema["type"] = "object"
    schema["properties"] = normalized_properties
    schema["required"] = [str(item) for item in schema.get("required", [])]
    tool["name"] = str(tool.get("name", "tool"))
    tool["description"] = str(tool.get("description", ""))
    tool["parameters"] = schema
    return tool


def render_prompt(question: list[Any]) -> str:
    lines: list[str] = []
    for turn in question:
        if not isinstance(turn, list):
            continue
        for message in turn:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).strip()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
    return "\n".join(lines)


def parse_ground_truth(entry: dict[str, Any]) -> dict[str, Any] | None:
    ground_truth = list(entry.get("ground_truth", []))
    if len(ground_truth) != 1:
        return None

    first_call = ground_truth[0]
    if not isinstance(first_call, dict) or len(first_call) != 1:
        return None

    name, raw_arguments = next(iter(first_call.items()))
    if not isinstance(raw_arguments, dict):
        return None

    arguments: dict[str, Any] = {}
    for field, values in raw_arguments.items():
        if not isinstance(values, list) or len(values) != 1:
            return None
        arguments[str(field)] = values[0]

    return {"name": str(name), "arguments": arguments}


def iter_clean_tasks(category: str) -> Iterable[dict[str, Any]]:
    base_dir = repo_root() / "external" / "gorilla-repo" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    data_path = base_dir / f"{category}.json"
    answer_path = base_dir / "possible_answer" / f"{category}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"BFCL data file not found: {data_path}")
    if not answer_path.exists():
        raise FileNotFoundError(f"BFCL answer file not found: {answer_path}")

    data_rows = read_jsonl(data_path)
    answer_rows = read_jsonl(answer_path)
    answers_by_id = {str(row["id"]): row for row in answer_rows}

    for row in data_rows:
        task_id = str(row.get("id", ""))
        answer_row = answers_by_id.get(task_id)
        if not answer_row:
            continue

        gold_call = parse_ground_truth(answer_row)
        if gold_call is None:
            continue

        candidate_tools = [normalize_tool(tool) for tool in row.get("function", [])]
        if not candidate_tools:
            continue

        tool = next((candidate for candidate in candidate_tools if candidate["name"] == gold_call["name"]), None)
        if tool is None:
            continue

        yield {
            "id": task_id,
            "prompt": render_prompt(list(row.get("question", []))),
            "tool": tool,
            "candidate_tools": candidate_tools,
            "gold_call": gold_call,
            "source_benchmark": "bfcl",
            "source_category": category,
        }


def build_subset(categories: list[str], per_category: int) -> list[dict[str, Any]]:
    subset: list[dict[str, Any]] = []
    for category in categories:
        count = 0
        for task in iter_clean_tasks(category):
            subset.append(task)
            count += 1
            if count >= per_category:
                break
    return subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="BFCL categories to export.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=5,
        help="Number of clean single-call examples to export per category.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root() / "data" / "bfcl_pilot_subset.json"),
        help="Output path for the exported subset JSON.",
    )
    args = parser.parse_args()

    subset = build_subset(list(args.categories), int(args.per_category))
    output_path = Path(args.output)
    dump_json(output_path, subset)
    print(f"Wrote {len(subset)} BFCL tasks to {output_path}")


if __name__ == "__main__":
    main()
