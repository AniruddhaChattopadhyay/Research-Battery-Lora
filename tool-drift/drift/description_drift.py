from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping


DESCRIPTION_MODES = (
    "verbose",
    "formal",
    "casual",
    "marketing",
    "example_heavy",
    "legacy_example",
)


def _legacy_placeholder(spec: Mapping[str, Any]) -> Any:
    expected_type = str(spec.get("type", "string")).lower()
    if expected_type == "integer":
        return 1
    if expected_type == "number":
        return 1.0
    if expected_type == "boolean":
        return True
    if expected_type == "array":
        return ["value"]
    if expected_type == "object":
        return {"field": "value"}
    return "value"


def _build_legacy_example(tool: Mapping[str, Any]) -> str | None:
    schema = dict(tool.get("parameters", {}))
    properties = dict(schema.get("properties", {}))
    rename_map = {
        str(old): str(new)
        for old, new in dict(schema.get("x_rename_map", {})).items()
        if str(old) != str(new)
    }
    if not rename_map:
        return None

    example = {}
    for old_name, new_name in list(rename_map.items())[:3]:
        spec = dict(properties.get(new_name, {}))
        example[old_name] = _legacy_placeholder(spec)

    if not example:
        return None

    encoded = json.dumps(example, ensure_ascii=True, separators=(", ", ": "))
    return (
        "Legacy integration note: older examples may still show payloads like "
        f"{encoded}."
    )


def drift_description(description: str, mode: str, tool_name: str | None = None) -> str:
    base = description.strip()
    if mode == "verbose":
        prefix = f"{tool_name or 'This tool'} should be used when the user needs the following capability."
        return f"{prefix} {base}"
    if mode == "formal":
        return f"Function: {tool_name or 'tool'}. Purpose: {base}"
    if mode == "casual":
        return f"Use this when you want to {base.lower().rstrip('.')}"
    if mode == "marketing":
        return f"{base} This premium, industry-leading tool delivers reliable results at scale."
    if mode == "example_heavy":
        return f"{base} Example: a user asks for this exact task, and the tool should respond accordingly."
    if mode == "legacy_example":
        return base
    return base


def apply_description_drift(tool: Mapping[str, Any], mode: str) -> dict[str, Any]:
    mutated = deepcopy(dict(tool))
    description = drift_description(
        str(mutated.get("description", "")),
        mode=mode,
        tool_name=str(mutated.get("name", "tool")),
    )
    if mode == "legacy_example":
        legacy = _build_legacy_example(mutated)
        if legacy:
            description = f"{description} {legacy}".strip()
    mutated["description"] = description
    return mutated


def apply_description_drifts(tools: list[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [apply_description_drift(tool, mode) for tool in tools]
