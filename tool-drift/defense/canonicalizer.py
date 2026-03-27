from __future__ import annotations

from typing import Any, Mapping


def canonical_tool_card(tool: Mapping[str, Any]) -> dict[str, Any]:
    schema = dict(tool.get("parameters", {}))
    properties = schema.get("properties", {})
    return {
        "name": str(tool.get("name", "tool")),
        "description": " ".join(str(tool.get("description", "")).split()),
        "required": list(schema.get("required", [])),
        "properties": {
            key: {
                "type": value.get("type", "string"),
                "enum": list(value.get("enum", [])),
            }
            for key, value in properties.items()
        },
    }


def render_canonical_tool_card(tool: Mapping[str, Any]) -> str:
    card = canonical_tool_card(tool)
    lines = [
        f"Tool: {card['name']}",
        f"Description: {card['description']}",
        f"Required: {', '.join(card['required']) or 'none'}",
        "Fields:",
    ]
    for name, spec in card["properties"].items():
        enum_text = f" enum={spec['enum']}" if spec["enum"] else ""
        lines.append(f"- {name}: type={spec['type']}{enum_text}")
    return "\n".join(lines)

