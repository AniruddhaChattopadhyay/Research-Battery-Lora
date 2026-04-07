from __future__ import annotations

import re
import random
from datetime import datetime
from statistics import mean
from typing import Any, Iterable, Mapping


def accuracy(items: Iterable[bool]) -> float:
    values = list(items)
    return mean(values) if values else 0.0


def exact_match_rate(predictions: Iterable[Any], references: Iterable[Any], comparator: Any | None = None) -> float:
    preds = list(predictions)
    refs = list(references)
    if not preds:
        return 0.0
    if comparator is None:
        return sum(pred == ref for pred, ref in zip(preds, refs)) / len(preds)
    return sum(bool(comparator(pred, ref)) for pred, ref in zip(preds, refs)) / len(preds)


def recovery_rate(original: float, drifted: float, repaired: float) -> float:
    gap = original - drifted
    if gap <= 0:
        return 0.0
    return max(0.0, min(1.0, (repaired - drifted) / gap))


def bootstrap_ci(values: list[float], samples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    draws = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        draws.append(mean(draw))
    draws.sort()
    lower = draws[int((alpha / 2) * len(draws))]
    upper = draws[int((1 - alpha / 2) * len(draws)) - 1]
    return lower, upper


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def _compact_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_time_string(value: str) -> str | None:
    text = _compact_spaces(value).lower().replace(".", "")
    patterns = [
        "%I %p",
        "%I:%M %p",
        "%I%p",
        "%H:%M",
    ]
    for pattern in patterns:
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.strftime("%H:%M")
        except ValueError:
            continue
    return None


def _normalize_date_string(value: str) -> str | None:
    text = _compact_spaces(value)
    patterns = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for pattern in patterns:
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_text(value: Any) -> str:
    text = _compact_spaces(str(value)).casefold()
    normalized_time = _normalize_time_string(text)
    if normalized_time is not None:
        return normalized_time
    normalized_date = _normalize_date_string(text)
    if normalized_date is not None:
        return normalized_date
    return text


def _normalize_boolean(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().casefold()
        if text == "true":
            return True
        if text == "false":
            return False
    return value


def _normalize_integer(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
        return int(value.strip())
    return value


def _normalize_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_coordinate_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) == 2:
        first = _coerce_float(value[0])
        second = _coerce_float(value[1])
        if first is not None and second is not None:
            return (first, second)

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip() for part in text.split(",")]
        if len(parts) == 2:
            first = _coerce_float(parts[0])
            second = _coerce_float(parts[1])
            if first is not None and second is not None:
                return (first, second)

    return None


def _looks_like_coordinate_field(field: str) -> bool:
    lowered = field.casefold()
    return "coord" in lowered or "coordinates" in lowered


def _normalize_value(field: str, value: Any, spec: Mapping[str, Any]) -> Any:
    expected_type = str(spec.get("type", "")).lower()
    if expected_type == "integer":
        return _normalize_integer(value)
    if expected_type == "number":
        return _normalize_number(value)
    if expected_type == "boolean":
        return _normalize_boolean(value)
    if expected_type == "array":
        if not isinstance(value, list):
            return _freeze(value)
        normalized_items = [_normalize_value(field, item, {}) for item in value]
        if field in {"attendees", "participants", "invitees"}:
            return tuple(sorted(_freeze(item) for item in normalized_items))
        return tuple(_freeze(item) for item in normalized_items)
    if expected_type == "object":
        if not isinstance(value, dict):
            return _freeze(value)
        normalized = {
            str(key): _normalize_value(str(key), item, {})
            for key, item in value.items()
        }
        return _freeze(normalized)

    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return tuple(_freeze(_normalize_value(field, item, {})) for item in value)
    if isinstance(value, dict):
        normalized = {
            str(key): _normalize_value(str(key), item, {})
            for key, item in value.items()
        }
        return _freeze(normalized)
    return value


def _values_match(field: str, prediction: Any, reference: Any, spec: Mapping[str, Any]) -> bool:
    expected_type = str(spec.get("type", "")).lower()

    if _looks_like_coordinate_field(field):
        pred_coords = _normalize_coordinate_pair(prediction)
        ref_coords = _normalize_coordinate_pair(reference)
        if pred_coords is not None and ref_coords is not None:
            return pred_coords == ref_coords

    if isinstance(reference, list) and expected_type != "array":
        return any(_values_match(field, prediction, candidate, spec) for candidate in reference)

    if isinstance(prediction, list) and expected_type != "array" and len(prediction) == 1:
        return _values_match(field, prediction[0], reference, spec)

    if expected_type == "object" or isinstance(reference, dict) or isinstance(prediction, dict):
        if not isinstance(reference, dict) or not isinstance(prediction, dict):
            return False
        if set(reference) != set(prediction):
            return False
        nested_properties = dict(spec.get("properties", {}))
        for key in reference:
            nested_spec = dict(nested_properties.get(key, {}))
            if not _values_match(str(key), prediction[key], reference[key], nested_spec):
                return False
        return True

    if expected_type == "array":
        if not isinstance(reference, list) or not isinstance(prediction, list):
            return False
        ref_values = [_normalize_value(field, item, {}) for item in reference]
        pred_values = [_normalize_value(field, item, {}) for item in prediction]
        if field in {"attendees", "participants", "invitees"}:
            return sorted(_freeze(item) for item in pred_values) == sorted(_freeze(item) for item in ref_values)
        return tuple(_freeze(item) for item in pred_values) == tuple(_freeze(item) for item in ref_values)

    pred_value = _normalize_value(field, prediction, spec)
    ref_value = _normalize_value(field, reference, spec)
    return pred_value == ref_value


def compare_tool_calls(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
    tool: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pred_name = _compact_spaces(str(prediction.get("name", "")))
    ref_name = _compact_spaces(str(reference.get("name", "")))
    name_match = pred_name == ref_name

    pred_args = dict(prediction.get("arguments", {}))
    ref_args = dict(reference.get("arguments", {}))
    pred_keys = set(pred_args)
    ref_keys = set(ref_args)
    missing_fields = sorted(ref_keys - pred_keys)
    extra_fields = sorted(pred_keys - ref_keys)

    properties = {}
    if tool is not None:
        schema = dict(tool.get("parameters", {}))
        properties = dict(schema.get("properties", {}))

    mismatched_fields: list[str] = []
    for field in sorted(pred_keys & ref_keys):
        spec = dict(properties.get(field, {}))
        if not _values_match(field, pred_args[field], ref_args[field], spec):
            mismatched_fields.append(field)

    matched = name_match and not missing_fields and not extra_fields and not mismatched_fields
    return {
        "matched": matched,
        "name_match": name_match,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "mismatched_fields": mismatched_fields,
    }
