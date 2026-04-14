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
    tool_name = str(tool.get("name", "tool"))
    return "\n".join(
        [
            "You are repairing a tool call.",
            "Return only corrected JSON with keys: name, arguments.",
            "Output schema: {\"name\": \"<tool_name>\", \"arguments\": {\"field\": \"value\"}}",
            f"You must return the exact tool name: {tool_name}",
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
            f"- The name must be exactly {tool_name}.",
            "- Do not switch to any other tool.",
            "- If fields were renamed, use the renamed field names from the tool card.",
            "- Infer missing required fields from the task, even if the invalid call is empty.",
            "- Every required field in the tool card must appear in arguments.",
            "- Fix fields, types, and enum values.",
            "- Do not add explanations.",
        ]
    )


def build_repair_prompt_no_card(
    task: Mapping[str, Any],
    tool: Mapping[str, Any],
    invalid_call: Mapping[str, Any],
    validation_result: Mapping[str, Any],
) -> str:
    tool_name = str(tool.get("name", "tool"))
    return "\n".join(
        [
            "You are repairing a tool call.",
            "Return only corrected JSON with keys: name, arguments.",
            "Output schema: {\"name\": \"<tool_name>\", \"arguments\": {\"field\": \"value\"}}",
            f"You must return the exact tool name: {tool_name}",
            "",
            f"Task: {task.get('prompt', '')}",
            "",
            "Invalid tool call:",
            json.dumps(invalid_call, indent=2, ensure_ascii=True),
            "",
            "Validation errors:",
            json.dumps(validation_result, indent=2, ensure_ascii=True),
            "",
            "Constraints:",
            f"- The name must be exactly {tool_name}.",
            "- Do not switch to any other tool.",
            "- Infer missing required fields from the task, even if the invalid call is empty.",
            "- Fix fields, types, and enum values.",
            "- Do not add explanations.",
        ]
    )


def build_tool_list_reprompt(
    task: Mapping[str, Any],
    tools: list[Mapping[str, Any]],
) -> str:
    """Minimal re-prompt baseline: tells the model the call was invalid and lists
    available tool names, but provides no canonical card, no validation errors,
    and no view of the invalid call.  Used to isolate whether gains come from
    structured feedback vs. simply flagging the error and listing tools."""
    tool_names = [str(tool.get("name", "tool")) for tool in tools]
    return "\n".join(
        [
            "Your previous tool call was invalid. Please try again.",
            "",
            f"Task: {task.get('prompt', '')}",
            "",
            "Available tools:",
            *[f"- {name}" for name in tool_names],
            "",
            "Select the correct tool and provide the required arguments.",
        ]
    )


def build_candidate_repair_prompt(
    task: Mapping[str, Any],
    tools: list[Mapping[str, Any]],
    invalid_call: Mapping[str, Any],
    validation_result: Mapping[str, Any],
) -> str:
    available_tools = [
        f"- {tool.get('name', 'tool')}: {tool.get('description', '')}".strip()
        for tool in tools
    ]
    return "\n".join(
        [
            "You are repairing a tool call.",
            "The previous attempt omitted or failed to resolve the tool name.",
            "Return only corrected JSON with keys: name, arguments.",
            "Output schema: {\"name\": \"<tool_name>\", \"arguments\": {\"field\": \"value\"}}",
            "",
            f"Task: {task.get('prompt', '')}",
            "",
            "Available tools:",
            *available_tools,
            "",
            "Invalid tool call:",
            json.dumps(invalid_call, indent=2, ensure_ascii=True),
            "",
            "Validation errors:",
            json.dumps(validation_result, indent=2, ensure_ascii=True),
            "",
            "Constraints:",
            "- Choose exactly one tool from the available tools.",
            "- Infer missing required fields from the task, even if the invalid call is empty.",
            "- Use the field names required by the selected tool schema.",
            "- Fix fields, types, and enum values.",
            "- Do not add explanations.",
        ]
    )
