from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        del exc
        return _fallback_yaml_load(Path(path).read_text(encoding="utf-8"))
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _parse_scalar(text: str) -> Any:
    if text in {"null", "Null", "NULL", "~", ""}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def _fallback_yaml_load(text: str) -> Dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, stripped.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        mapping: Dict[str, Any] = {}
        sequence: list[Any] | None = None
        while index < len(lines):
            cur_indent, content = lines[index]
            if cur_indent < indent:
                break
            if content.startswith("- "):
                if sequence is None:
                    sequence = []
                value_text = content[2:].strip()
                index += 1
                if value_text:
                    sequence.append(_parse_scalar(value_text))
                else:
                    value, index = parse_block(index, cur_indent + 2)
                    sequence.append(value)
                continue

            key, sep, rest = content.partition(":")
            if not sep:
                raise ValueError(f"Invalid YAML line: {content}")
            key = key.strip()
            rest = rest.strip()
            index += 1
            if rest:
                mapping[key] = _parse_scalar(rest)
            else:
                value, index = parse_block(index, cur_indent + 2)
                mapping[key] = value

        return (sequence if sequence is not None else mapping), index

    loaded, _ = parse_block(0, 0)
    if not isinstance(loaded, dict):
        raise ValueError("Top-level YAML structure must be a mapping")
    return loaded


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    env_path = Path(path) if path else repo_root() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        loaded[key] = value
    return loaded


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def dump_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(data):
        data = asdict(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir(config: Dict[str, Any]) -> Path:
    output_dir = config.get("project", {}).get("output_dir", "outputs")
    return repo_root() / output_dir


def synthetic_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "book_flight",
            "description": "Book a flight for a traveler between two cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                    "class": {"type": "string", "enum": ["economy", "business"]},
                },
                "required": ["origin", "destination", "date"],
            },
        },
        {
            "name": "lookup_weather",
            "description": "Look up a weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["city"],
            },
        },
        {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with attendees and a time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "attendees": {"type": "array"},
                    "start_time": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                },
                "required": ["title", "attendees", "start_time"],
            },
        },
    ]


def example_value(field: str, spec: Dict[str, Any]) -> Any:
    expected_type = spec.get("type", "string")
    if expected_type == "integer":
        return 1
    if expected_type == "array":
        return [f"{field}_item"]
    if expected_type == "boolean":
        return True
    if expected_type == "number":
        return 1.0
    if expected_type == "object":
        return {}
    return f"{field}_value"


def build_example_call(tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = dict(tool.get("parameters", {}))
    properties = dict(schema.get("properties", {}))
    required = list(schema.get("required", []))
    return {
        "name": str(tool.get("name", "tool")),
        "arguments": {
            field: example_value(field, dict(properties.get(field, {})))
            for field in required
        },
    }


def adapt_gold_call_to_tool(gold_call: Dict[str, Any], tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = dict(tool.get("parameters", {}))
    properties = dict(schema.get("properties", {}))
    rename_map = dict(schema.get("x_rename_map", {}))

    original_args = dict(gold_call.get("arguments", {}))
    adapted_args: Dict[str, Any] = {}

    for old_name, value in original_args.items():
        new_name = rename_map.get(old_name, old_name)
        if new_name in properties:
            adapted_args[new_name] = value

    return {
        "name": str(tool.get("name", gold_call.get("name", ""))),
        "arguments": adapted_args,
    }
