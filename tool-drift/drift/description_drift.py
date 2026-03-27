from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DESCRIPTION_MODES = ("verbose", "formal", "casual", "marketing", "example_heavy")


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
    return base


def apply_description_drift(tool: Mapping[str, Any], mode: str) -> dict[str, Any]:
    mutated = deepcopy(dict(tool))
    mutated["description"] = drift_description(
        str(mutated.get("description", "")),
        mode=mode,
        tool_name=str(mutated.get("name", "tool")),
    )
    return mutated


def apply_description_drifts(tools: list[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [apply_description_drift(tool, mode) for tool in tools]

