from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from drift.candidate_drift import apply_candidate_drift
from drift.description_drift import apply_description_drift
from drift.schema_drift import apply_schema_drift_sequence


DEFAULT_CANDIDATE_COUNTS = {
    "mild": 1,
    "medium": 3,
    "strong": 5,
}


def resolve_mode_sequence(configured: Iterable[str] | None, *, default: str, severity: str) -> list[str]:
    modes = [str(mode) for mode in (configured or [default]) if str(mode).strip()]
    if not modes:
        modes = [default]

    lowered = severity.casefold()
    if lowered == "mild":
        return modes[:1]
    if lowered == "medium":
        return modes[: min(2, len(modes))]
    return modes


def candidate_count_for_severity(drift_cfg: Mapping[str, Any]) -> int:
    explicit = drift_cfg.get("candidate_extra_candidates")
    if explicit is not None:
        return max(0, int(explicit))
    severity = str(drift_cfg.get("severity", "mild")).casefold()
    return DEFAULT_CANDIDATE_COUNTS.get(severity, DEFAULT_CANDIDATE_COUNTS["mild"])


def apply_drift_pipeline(tool: Mapping[str, Any], *, description_modes: list[str], schema_modes: list[str]) -> dict[str, Any]:
    drifted = deepcopy(dict(tool))
    drifted = apply_schema_drift_sequence(drifted, schema_modes)
    for mode in description_modes:
        drifted = apply_description_drift(drifted, mode)
    return drifted


def build_drifted_toolset(
    *,
    task: Mapping[str, Any],
    drifted_tool: Mapping[str, Any],
    candidate_mode: str,
    extra_candidates: int,
) -> list[dict[str, Any]]:
    candidate_tools = task.get("candidate_tools")
    if isinstance(candidate_tools, list) and candidate_tools:
        pool = []
        gold_name = str(task.get("gold_call", {}).get("name", drifted_tool.get("name", "")))
        replaced = False
        for candidate in deepcopy(candidate_tools):
            if str(candidate.get("name", "")) == gold_name and not replaced:
                pool.append(deepcopy(drifted_tool))
                replaced = True
            else:
                pool.append(candidate)
        if not replaced:
            pool.insert(0, deepcopy(drifted_tool))
        return apply_candidate_drift(pool, candidate_mode, extra_candidates=extra_candidates)

    return apply_candidate_drift([deepcopy(drifted_tool)], candidate_mode, extra_candidates=extra_candidates)
