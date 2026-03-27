from __future__ import annotations

import json
from typing import Any, Mapping

from .canonicalizer import render_canonical_tool_card


def build_repair_prompt(
    task: Mapping[str, Any],
    tool: Mapping[str, Any],
    invalid_call: Mapping[str, Any],
    validation_result: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "You are repairing a tool call.",
            "Return only corrected JSON with keys: name, arguments.",
            "Output schema: {\"name\": \"<tool_name>\", \"arguments\": {\"field\": \"value\"}}",
            "",
            f"Task: {task.get('prompt', '')}",
            "",
            render_canonical_tool_card(tool),
            "",
            "Invalid tool call:",
            json.dumps(invalid_call, indent=2, ensure_ascii=True),
            "",
            "Validation errors:",
            json.dumps(validation_result, indent=2, ensure_ascii=True),
            "",
            "Constraints:",
            "- Use the exact tool name from the tool card.",
            "- If fields were renamed, use the renamed field names from the tool card.",
            "- Fix fields, types, and enum values.",
            "- Do not add explanations.",
        ]
    )
