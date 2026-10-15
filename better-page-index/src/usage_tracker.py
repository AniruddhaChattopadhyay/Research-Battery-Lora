"""Count the tokens of every LiteLLM call in this process (PageIndex calls LiteLLM inside its agent)."""
import threading

import litellm
from litellm.integrations.custom_logger import CustomLogger

_lock = threading.Lock()
TOTALS = {}


def _record(kwargs, response):
    u = getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None)
    if not u:
        return
    get = (lambda k: u.get(k, 0)) if isinstance(u, dict) else (lambda k: getattr(u, k, 0))
    with _lock:
        t = TOTALS.setdefault(kwargs.get("model", "?"), {"calls": 0, "prompt": 0, "completion": 0})
        t["calls"] += 1
        t["prompt"] += get("prompt_tokens") or 0
        t["completion"] += get("completion_tokens") or 0


class _Tracker(CustomLogger):
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _record(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _record(kwargs, response_obj)


def install():
    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [_Tracker()]
