from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str | None
    message: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: list[ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _aliases(spec: Mapping[str, Any]) -> set[str]:
    values = set()
    for key in ("aliases", "x_aliases"):
        raw = spec.get(key, [])
        if isinstance(raw, list):
            values.update(str(item) for item in raw)
    return values


def validate_tool_call(tool: Mapping[str, Any], tool_call: Mapping[str, Any]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    tool_name = str(tool.get("name", "tool"))
    call_name = str(tool_call.get("name", tool_call.get("function", "")))
    if call_name != tool_name and call_name not in _aliases(tool):
        issues.append(
            ValidationIssue(
                code="wrong_tool",
                field="name",
                message=f"Expected {tool_name}, got {call_name or '<missing>'}",
            )
        )

    schema = dict(tool.get("parameters", {}))
    properties = dict(schema.get("properties", {}))
    required = set(schema.get("required", []))
    arguments = dict(tool_call.get("arguments", {}))
    known_fields = set(properties)

    for field in required:
        if field not in arguments:
            issues.append(
                ValidationIssue(
                    code="missing_field",
                    field=field,
                    message=f"Missing required field: {field}",
                )
            )

    for field, value in arguments.items():
        if field not in known_fields:
            alias_hit = next((name for name, spec in properties.items() if field in _aliases(spec)), None)
            if alias_hit is None:
                issues.append(
                    ValidationIssue(
                        code="unknown_field",
                        field=field,
                        message=f"Unknown field: {field}",
                    )
                )
                continue
            field = alias_hit

        spec = properties[field]
        expected_type = spec.get("type", "string")
        if expected_type == "integer" and not isinstance(value, int):
            issues.append(
                ValidationIssue(
                    code="type_mismatch",
                    field=field,
                    message=f"Field {field} expects integer",
                )
            )
        elif expected_type == "array" and not isinstance(value, list):
            issues.append(
                ValidationIssue(
                    code="type_mismatch",
                    field=field,
                    message=f"Field {field} expects array",
                )
            )
        elif expected_type == "string" and not isinstance(value, str):
            issues.append(
                ValidationIssue(
                    code="type_mismatch",
                    field=field,
                    message=f"Field {field} expects string",
                )
            )

        enum_values = list(spec.get("enum", []))
        if enum_values and str(value) not in enum_values:
            issues.append(
                ValidationIssue(
                    code="invalid_enum",
                    field=field,
                    message=f"Field {field} must be one of {enum_values}",
                )
            )

    return ValidationResult(valid=not issues, issues=issues)

