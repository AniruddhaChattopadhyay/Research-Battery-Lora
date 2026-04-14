from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from scripts.common import load_dotenv, require_env


_SAFE_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _normalize_tool_name(name: str) -> str:
    normalized = _SAFE_TOOL_NAME_RE.sub("_", name).strip("_")
    return normalized or "tool"


def _alias_tools(
    tools: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    aliased_tools: list[dict[str, Any]] = []
    original_to_alias: dict[str, str] = {}
    alias_to_original: dict[str, str] = {}
    used_aliases: set[str] = set()

    for index, tool in enumerate(tools, start=1):
        original_name = str(tool.get("name", "tool"))
        base_alias = _normalize_tool_name(original_name)
        alias = base_alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base_alias}_{suffix}"
            suffix += 1
        used_aliases.add(alias)
        original_to_alias[original_name] = alias
        alias_to_original[alias] = original_name
        aliased_tool = dict(tool)
        aliased_tool["name"] = alias
        aliased_tools.append(aliased_tool)

    return aliased_tools, original_to_alias, alias_to_original


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


def extract_usage(payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def _restore_tool_name(name: str, alias_to_original: Mapping[str, str]) -> str:
    return str(alias_to_original.get(name, name))


def _extract_tool_call(
    payload: Mapping[str, Any],
    *,
    alias_to_original: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    alias_lookup = alias_to_original or {}
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
            "name": _restore_tool_name(str(function.get("name", "")), alias_lookup),
            "arguments": arguments if isinstance(arguments, dict) else {},
        }

    # Fallback: some providers may return JSON in content instead of tool_calls.
    parsed = _extract_json_dict(_read_message_content(message))
    if parsed:
        return {
            "name": _restore_tool_name(str(parsed.get("name", parsed.get("tool", ""))), alias_lookup),
            "arguments": dict(parsed.get("arguments", {})),
        }

    return {"name": "", "arguments": {}}


def _retry_delay_seconds(
    attempt: int,
    *,
    detail: str = "",
    retry_after: str | None = None,
) -> float:
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

    parsed = _extract_json_dict(detail)
    error = parsed.get("error", {})
    if isinstance(error, dict):
        metadata = error.get("metadata", {})
        if isinstance(metadata, dict):
            retry_after_seconds = metadata.get("retry_after_seconds")
            if isinstance(retry_after_seconds, (int, float)):
                return max(float(retry_after_seconds), 0.0)

    return min(2**attempt, 60.0)


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

    max_attempts = 12
    retryable_status_codes = {400, 408, 429, 500, 502, 503, 504}
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in retryable_status_codes and attempt + 1 < max_attempts:
                time.sleep(
                    _retry_delay_seconds(
                        attempt,
                        detail=detail,
                        retry_after=exc.headers.get("Retry-After"),
                    )
                )
                continue
            raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 < max_attempts:
                time.sleep(_retry_delay_seconds(attempt))
                continue
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt + 1 < max_attempts:
                time.sleep(_retry_delay_seconds(attempt))
                continue
            raise RuntimeError(f"OpenRouter request timed out: {exc}") from exc

    raise RuntimeError("OpenRouter request failed after exhausting retries")


def _request_tool_call_payload(
    *,
    model: str,
    prompt: str,
    tools: list[Mapping[str, Any]],
    base_url: str | None = None,
    force_tool_name: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    provider_preferences: Mapping[str, Any] | None = None,
    max_tokens: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliased_tools, original_to_alias, alias_to_original = _alias_tools(tools)
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
        "tools": [_tool_to_openai_schema(tool) for tool in aliased_tools],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = int(seed)
    if provider_preferences:
        body["provider"] = dict(provider_preferences)
    if force_tool_name:
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": original_to_alias.get(force_tool_name, force_tool_name)},
        }
    else:
        body["tool_choice"] = "auto"
    payload = _request_payload(body, base_url=base_url)
    return _extract_tool_call(payload, alias_to_original=alias_to_original), payload


def request_tool_call(
    *,
    model: str,
    prompt: str,
    tools: list[Mapping[str, Any]],
    base_url: str | None = None,
    force_tool_name: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    provider_preferences: Mapping[str, Any] | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    tool_call, _ = _request_tool_call_payload(
        model=model,
        prompt=prompt,
        tools=tools,
        base_url=base_url,
        force_tool_name=force_tool_name,
        temperature=temperature,
        seed=seed,
        provider_preferences=provider_preferences,
        max_tokens=max_tokens,
    )
    return tool_call


def request_tool_call_with_payload(
    *,
    model: str,
    prompt: str,
    tools: list[Mapping[str, Any]],
    base_url: str | None = None,
    force_tool_name: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    provider_preferences: Mapping[str, Any] | None = None,
    max_tokens: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _request_tool_call_payload(
        model=model,
        prompt=prompt,
        tools=tools,
        base_url=base_url,
        force_tool_name=force_tool_name,
        temperature=temperature,
        seed=seed,
        provider_preferences=provider_preferences,
        max_tokens=max_tokens,
    )


def _request_json_tool_call_payload(
    *,
    model: str,
    prompt: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    provider_preferences: Mapping[str, Any] | None = None,
    max_tokens: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if seed is not None:
        body["seed"] = int(seed)
    if provider_preferences:
        body["provider"] = dict(provider_preferences)
    payload = _request_payload(body, base_url=base_url)
    choices = list(payload.get("choices", []))
    if not choices:
        return {"name": "", "arguments": {}}, payload

    message = dict(choices[0].get("message", {}))
    parsed = _extract_json_dict(_read_message_content(message))
    if not parsed:
        return {"name": "", "arguments": {}}, payload
    return {
        "name": str(parsed.get("name", parsed.get("tool", ""))),
        "arguments": dict(parsed.get("arguments", {})),
    }, payload


def request_json_tool_call(
    *,
    model: str,
    prompt: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    provider_preferences: Mapping[str, Any] | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    tool_call, _ = _request_json_tool_call_payload(
        model=model,
        prompt=prompt,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        provider_preferences=provider_preferences,
        max_tokens=max_tokens,
    )
    return tool_call


def request_json_tool_call_with_payload(
    *,
    model: str,
    prompt: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    provider_preferences: Mapping[str, Any] | None = None,
    max_tokens: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _request_json_tool_call_payload(
        model=model,
        prompt=prompt,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        provider_preferences=provider_preferences,
        max_tokens=max_tokens,
    )
