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
    if not base or extra_candidates <= 0:
        return base

    distractors = [
        distractor_tool(base[idx % len(base)], idx + 1)
        for idx in range(extra_candidates)
    ]
    return base + distractors
