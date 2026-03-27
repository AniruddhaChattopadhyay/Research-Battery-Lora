from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def distractor_tool(tool: Mapping[str, Any], index: int) -> dict[str, Any]:
    mutated = deepcopy(dict(tool))
    base_name = str(mutated.get("name", f"tool_{index}"))
    mutated["name"] = f"{base_name}_alt_{index}"
    mutated["description"] = (
        f"Alternative utility related to {base_name}. "
        f"{mutated.get('description', '')}"
    ).strip()
    return mutated


def apply_candidate_drift(
    tools: list[Mapping[str, Any]],
    mode: str,
    extra_candidates: int = 2,
) -> list[dict[str, Any]]:
    base = [deepcopy(dict(tool)) for tool in tools]
    if mode != "distractors":
        return base
    distractors = [distractor_tool(tool, idx) for idx, tool in enumerate(base[:extra_candidates], start=1)]
    return base + distractors

