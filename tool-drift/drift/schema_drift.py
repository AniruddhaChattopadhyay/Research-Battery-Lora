from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


SCHEMA_MODES = (
    "rename_parameters",
    "rename_all_parameters",
    "alias_enums",
    "reorder_parameters",
    "camel_case",
)


def _properties(schema: Mapping[str, Any]) -> dict[str, Any]:
    return dict(schema.get("properties", {}))


def _compose_rename_map(
    existing_map: Mapping[str, str] | None,
    step_map: Mapping[str, str],
) -> dict[str, str]:
    if not existing_map:
        return dict(step_map)

    composed = {original: step_map.get(current, current) for original, current in existing_map.items()}
    for current, renamed in step_map.items():
        composed.setdefault(current, renamed)
    return composed


def apply_schema_drift(tool: Mapping[str, Any], mode: str) -> dict[str, Any]:
    mutated = deepcopy(dict(tool))
    schema = dict(mutated.get("parameters", {}))
    props = _properties(schema)
    required = list(schema.get("required", []))
    existing_rename_map = dict(schema.get("x_rename_map", {}))

    if mode == "rename_parameters" and props:
        renamed: dict[str, Any] = {}
        rename_map: dict[str, str] = {}
        for idx, (name, spec) in enumerate(props.items()):
            new_name = name if idx == 0 else f"{name}_v2"
            renamed[new_name] = spec
            rename_map[name] = new_name
        schema["properties"] = renamed
        schema["required"] = [rename_map.get(name, name) for name in required]
        schema["x_rename_map"] = _compose_rename_map(existing_rename_map, rename_map)
    elif mode == "rename_all_parameters" and props:
        renamed = {}
        rename_map = {}
        for name, spec in props.items():
            new_name = f"{name}_v2"
            renamed[new_name] = spec
            rename_map[name] = new_name
        schema["properties"] = renamed
        schema["required"] = [rename_map.get(name, name) for name in required]
        schema["x_rename_map"] = _compose_rename_map(existing_rename_map, rename_map)
    elif mode == "alias_enums":
        for name, spec in props.items():
            if "enum" in spec:
                spec = dict(spec)
                spec["enum"] = list(spec["enum"]) + [f"{value}_alias" for value in spec["enum"]]
                props[name] = spec
        schema["properties"] = props
    elif mode == "reorder_parameters":
        keys = list(props.keys())
        keys.reverse()
        schema["properties"] = {key: props[key] for key in keys}
    elif mode == "camel_case":
        renamed = {}
        rename_map = {}
        for name, spec in props.items():
            parts = name.split("_")
            new_name = parts[0] + "".join(piece.capitalize() for piece in parts[1:])
            renamed[new_name] = spec
            rename_map[name] = new_name
        schema["properties"] = renamed
        schema["required"] = [rename_map.get(name, name) for name in required]
        schema["x_rename_map"] = _compose_rename_map(existing_rename_map, rename_map)
    mutated["parameters"] = schema
    return mutated


def apply_schema_drifts(tools: list[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [apply_schema_drift(tool, mode) for tool in tools]


def apply_schema_drift_sequence(tool: Mapping[str, Any], modes: list[str]) -> dict[str, Any]:
    drifted = deepcopy(dict(tool))
    for mode in modes:
        drifted = apply_schema_drift(drifted, mode)
    return drifted
