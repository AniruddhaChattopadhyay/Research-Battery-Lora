from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .validator import ValidationIssue, ValidationResult, validate_tool_call


def _aliases(spec: Mapping[str, Any]) -> set[str]:
    values = set()
    for key in ("aliases", "x_aliases"):
        raw = spec.get(key, [])
        if isinstance(raw, list):
            values.update(str(item) for item in raw)
    return values


def _call_name(tool_call: Mapping[str, Any]) -> str:
    return str(tool_call.get("name", tool_call.get("function", ""))).strip()


def _find_tool_by_call_name(
    tools: Sequence[Mapping[str, Any]],
    tool_call: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    call_name = _call_name(tool_call)
    if not call_name:
        return None, "missing_tool_name"

    for tool in tools:
        tool_name = str(tool.get("name", ""))
        if call_name == tool_name or call_name in _aliases(tool):
            return dict(tool), "predicted_tool_name"

    return None, "unresolved_tool_name"


def _unresolved_validation(tool_call: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]) -> ValidationResult:
    call_name = _call_name(tool_call)
    available = [str(tool.get("name", "")) for tool in tools]
    if call_name:
        message = f"Predicted tool {call_name} is not in the available tool set: {available}"
        code = "unresolved_tool"
    else:
        message = f"Predicted tool name is missing and no repair target can be resolved from: {available}"
        code = "missing_tool_name"
    return ValidationResult(
        valid=False,
        issues=[ValidationIssue(code=code, field="name", message=message)],
    )


@dataclass(frozen=True)
class RepairTarget:
    mode: str
    source: str
    tool: dict[str, Any] | None
    validation: ValidationResult

    @property
    def resolved(self) -> bool:
        return self.tool is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": self.source,
            "resolved": self.resolved,
            "tool_name": None if self.tool is None else str(self.tool.get("name", "")),
        }


def resolve_repair_target(
    *,
    oracle_tool: Mapping[str, Any],
    candidate_tools: Sequence[Mapping[str, Any]],
    predicted_call: Mapping[str, Any],
    mode: str,
) -> RepairTarget:
    if mode == "oracle_target":
        tool = dict(oracle_tool)
        return RepairTarget(
            mode=mode,
            source="oracle_target",
            tool=tool,
            validation=validate_tool_call(tool, predicted_call),
        )

    if mode == "predicted_tool":
        selected_tool, source = _find_tool_by_call_name(candidate_tools, predicted_call)
        if selected_tool is None and len(candidate_tools) == 1:
            selected_tool = dict(candidate_tools[0])
            source = "single_candidate_fallback"
        if selected_tool is None:
            return RepairTarget(
                mode=mode,
                source=source,
                tool=None,
                validation=_unresolved_validation(predicted_call, candidate_tools),
            )
        return RepairTarget(
            mode=mode,
            source=source,
            tool=selected_tool,
            validation=validate_tool_call(selected_tool, predicted_call),
        )

    raise ValueError(f"Unsupported repair target mode: {mode}")
