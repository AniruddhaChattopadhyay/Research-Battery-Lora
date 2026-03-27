from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from scripts.common import load_dotenv, require_env


def _tool_to_openai_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name", "tool")),
            "description": str(tool.get("description", "")),
            "parameters": dict(tool.get("parameters", {})),
        },
    }


def _read_message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_json_dict(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_tool_call(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = list(payload.get("choices", []))
    if not choices:
        return {"name": "", "arguments": {}}

    message = dict(choices[0].get("message", {}))
    tool_calls = list(message.get("tool_calls", []))
    if tool_calls:
        function = dict(tool_calls[0].get("function", {}))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return {
            "name": str(function.get("name", "")),
            "arguments": arguments if isinstance(arguments, dict) else {},
        }

    # Fallback: some providers may return JSON in content instead of tool_calls.
    parsed = _extract_json_dict(_read_message_content(message))
    if parsed:
        return {
            "name": str(parsed.get("name", parsed.get("tool", ""))),
            "arguments": dict(parsed.get("arguments", {})),
        }

    return {"name": "", "arguments": {}}


def _request_payload(body: Mapping[str, Any], base_url: str | None = None) -> dict[str, Any]:
    load_dotenv()
    api_key = require_env("OPENROUTER_API_KEY")
    endpoint = base_url or require_env("OPENROUTER_BASE_URL")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    return payload


def request_tool_call(
    *,
    model: str,
    prompt: str,
    tools: list[Mapping[str, Any]],
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a function-calling model. "
                    "Return exactly one tool call when the request matches a tool."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "tools": [_tool_to_openai_schema(tool) for tool in tools],
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload = _request_payload(body, base_url=base_url)
    return _extract_tool_call(payload)


def request_json_tool_call(
    *,
    model: str,
    prompt: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You repair function calls. "
                    "Return only a single JSON object with keys name and arguments."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload = _request_payload(body, base_url=base_url)
    choices = list(payload.get("choices", []))
    if not choices:
        return {"name": "", "arguments": {}}

    message = dict(choices[0].get("message", {}))
    parsed = _extract_json_dict(_read_message_content(message))
    if not parsed:
        return {"name": "", "arguments": {}}
    return {
        "name": str(parsed.get("name", parsed.get("tool", ""))),
        "arguments": dict(parsed.get("arguments", {})),
    }
