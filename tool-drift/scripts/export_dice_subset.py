from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from scripts.common import dump_json, repo_root


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


def normalize_type(value: Any) -> str:
    return TYPE_MAP.get(str(value).strip().lower(), "string")


def load_tool_docs() -> dict[str, dict[str, Any]]:
    tool_docs_path = repo_root() / "external" / "dice-bench-repo" / "src" / "graph" / "tool_docs.json"
    payload = json.loads(tool_docs_path.read_text(encoding="utf-8"))
    functions = list(payload.get("functions", []))
    tools: dict[str, dict[str, Any]] = {}
    for item in functions:
        name = str(item.get("function", "tool"))
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in item.get("parameters", []):
            field = str(param.get("name", "arg"))
            properties[field] = {
                "type": normalize_type(param.get("type", "string")),
                "description": str(param.get("desc", "")),
            }
            required.append(field)
        tools[name] = {
            "name": name,
            "description": str(item.get("desc", "")),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    return tools


def render_prompt(conversation: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in conversation:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not content or role == "assistant":
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def iter_round_tasks(round_file: Path, max_count: int, *, allow_duplicates: bool = False) -> list[dict[str, Any]]:
    tools_by_name = load_tool_docs()
    selected: list[dict[str, Any]] = []
    seen_functions: set[str] = set()

    with round_file.open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            metadata = dict(row.get("metadata", {}))
            params_ret_val = list(metadata.get("params_ret_val", []))
            functions = list(metadata.get("functions", []))
            if len(functions) != 1 or len(params_ret_val) != 1:
                continue

            call = dict(params_ret_val[0])
            function_name = str(call.get("function", ""))
            tool = tools_by_name.get(function_name)
            if tool is None:
                continue
            if not allow_duplicates and function_name in seen_functions:
                continue

            selected.append(
                {
                    "id": f"dice_round_{metadata.get('round_num', 0)}_{row.get('diag_id', '')}",
                    "prompt": render_prompt(list(row.get("conversation", []))),
                    "tool": deepcopy(tool),
                    "candidate_tools": [deepcopy(tool)],
                    "gold_call": {
                        "name": function_name,
                        "arguments": dict(call.get("parameters", {})),
                    },
                    "source_benchmark": "dice-bench",
                    "source_round": metadata.get("round_num"),
                    "source_diag_id": row.get("diag_id"),
                    "source_category": metadata.get("category"),
                }
            )
            seen_functions.add(function_name)
            if len(selected) >= max_count:
                break

    return selected


def iter_multi_round_tasks(round_files: list[Path], max_count: int, *, allow_duplicates: bool = False) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for round_file in round_files:
        if not round_file.exists():
            continue
        remaining = max_count - len(selected)
        if remaining <= 0:
            break
        selected.extend(iter_round_tasks(round_file, remaining, allow_duplicates=allow_duplicates))
    return selected[:max_count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round-file",
        default=str(repo_root() / "external" / "dice-bench" / "data" / "round_1.jsonl"),
        help="Path to the DICE round file to export from.",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        default=None,
        help="Multiple round files to draw from (overrides --round-file).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of diverse single-function examples to export.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow multiple examples per function name (needed for >122 examples).",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root() / "data" / "dice_pilot_subset.json"),
        help="Output path for the exported subset JSON.",
    )
    args = parser.parse_args()

    if args.rounds:
        round_files = [Path(f) for f in args.rounds]
        subset = iter_multi_round_tasks(round_files, int(args.count), allow_duplicates=args.allow_duplicates)
    else:
        subset = iter_round_tasks(Path(args.round_file), int(args.count), allow_duplicates=args.allow_duplicates)
    output_path = Path(args.output)
    dump_json(output_path, subset)
    print(f"Wrote {len(subset)} DICE tasks to {output_path}")


if __name__ == "__main__":
    main()
