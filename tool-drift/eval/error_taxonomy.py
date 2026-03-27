from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def classify_error(record: dict[str, Any]) -> str:
    if record.get("parse_error"):
        return "malformed_json"
    validation = record.get("validation", {})
    issues = validation.get("issues", [])
    if not issues:
        return "unknown"
    codes = [issue.get("code", "unknown") for issue in issues]
    if "wrong_tool" in codes:
        return "wrong_tool"
    if "missing_field" in codes:
        return "missing_field"
    if "unknown_field" in codes:
        return "unknown_field"
    if "invalid_enum" in codes:
        return "invalid_enum"
    if "type_mismatch" in codes:
        return "type_mismatch"
    return codes[0]


def summarize_errors(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(classify_error(record) for record in records)
    return dict(counts)

