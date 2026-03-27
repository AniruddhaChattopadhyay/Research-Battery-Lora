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
    content = message.get("content", "")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return {
                    "name": str(parsed.get("name", parsed.get("tool", ""))),
                    "arguments": dict(parsed.get("arguments", {})),
                }
        except json.JSONDecodeError:
            pass

    return {"name": "", "arguments": {}}


def request_tool_call(
    *,
    model: str,
    prompt: str,
    tools: list[Mapping[str, Any]],
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict[str, Any]:
    load_dotenv()
    api_key = require_env("OPENROUTER_API_KEY")
    endpoint = base_url or require_env("OPENROUTER_BASE_URL")

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

    return _extract_tool_call(payload)
