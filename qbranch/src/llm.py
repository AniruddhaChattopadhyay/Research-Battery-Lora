"""OpenRouter client for chat and embeddings, with a disk cache and a cost counter.

Every call goes through `chat` or `embed`. The cache key is the full request, so a rerun of
the same request costs nothing. The counter keeps the tokens and the dollar cost that
OpenRouter reports for each model, so each run can report its own cost (the account is shared).
"""
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sitecustomize  # noqa: F401,E402  (IPv4 first: IPv6 times out on this network)

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(os.path.join(ROOT, ".env"), override=True)

_client = OpenAI(base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                 api_key=os.environ["OPENROUTER_API_KEY"], timeout=180, max_retries=0)
_lock = threading.Lock()
_local = threading.local()
CACHE_PATH = os.path.join(ROOT, "runs", "llm_cache.sqlite")
USAGE = {}  # model -> {"calls", "cached_calls", "prompt", "completion", "reasoning", "cost"}


def _db():
    if not hasattr(_local, "db"):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        _local.db = sqlite3.connect(CACHE_PATH, timeout=60)
        _local.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    return _local.db


def _get(key):
    row = _db().execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _put(key, value):
    with _lock:
        _db().execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
        _db().commit()


def _count(model, usage, cached):
    with _lock:
        u = USAGE.setdefault(model, {"calls": 0, "cached_calls": 0, "prompt": 0, "completion": 0,
                                     "reasoning": 0, "cost": 0.0})
        u["cached_calls" if cached else "calls"] += 1
        if not cached:
            u["prompt"] += usage.get("prompt", 0)
            u["completion"] += usage.get("completion", 0)
            u["reasoning"] += usage.get("reasoning", 0)
            u["cost"] += usage.get("cost", 0.0)


def _usage(resp):
    u = resp.usage
    if u is None:
        return {}
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    det = d.get("completion_tokens_details") or {}
    return {"prompt": d.get("prompt_tokens") or 0, "completion": d.get("completion_tokens") or 0,
            "reasoning": det.get("reasoning_tokens") or 0, "cost": float(d.get("cost") or 0.0)}


def chat(model, messages, temperature=0.0, max_tokens=4096, json_mode=False, reasoning=None,
         seed_tag="", retries=5):
    """Return {"text", "usage", "cached"}. `seed_tag` makes a separate cache entry (for repeats)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    extra = {"usage": {"include": True}}
    if reasoning:
        extra["reasoning"] = reasoning
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    key = hashlib.sha256(json.dumps([body, extra, seed_tag], sort_keys=True).encode()).hexdigest()
    hit = _get(key)
    if hit is not None:
        _count(model, hit["usage"], cached=True)
        return {**hit, "cached": True}
    err = None
    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(**body, extra_body=extra)
            text = resp.choices[0].message.content or ""
            out = {"text": text, "usage": _usage(resp), "provider": getattr(resp, "provider", None)}
            _count(model, out["usage"], cached=False)
            if text.strip():
                _put(key, out)
            return {**out, "cached": False}
        except Exception as e:  # network errors, 429, 5xx: wait and retry
            err = e
            time.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError("chat failed after %d tries: %s" % (retries, err))


def _emb_table():
    db = _db()
    db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
    return db


def _put_emb(pairs):
    with _lock:
        db = _emb_table()
        db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                       [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in pairs])
        db.commit()


def embed(model, texts, batch=64):
    """Return an (n, d) float32 array. Each text is cached on its own (binary float32 in table emb;
    vectors in the old JSON cache are read once and copied to emb)."""
    keys = [hashlib.sha256(json.dumps([model, t]).encode()).hexdigest() for t in texts]
    out = [None] * len(texts)
    db = _emb_table()
    for s in range(0, len(keys), 500):
        part = keys[s:s + 500]
        got = dict(db.execute("SELECT k, v FROM emb WHERE k IN (%s)" % ",".join("?" * len(part)), part))
        for j, k in enumerate(part):
            if k in got:
                out[s + j] = np.frombuffer(got[k], dtype=np.float32)
    old = []
    for i, v in enumerate(out):
        if v is None:
            hit = _get(keys[i])
            if hit is not None:
                out[i] = np.asarray(hit, dtype=np.float32)
                old.append((keys[i], out[i]))
    if old:
        _put_emb(old)
    todo = [i for i, v in enumerate(out) if v is None]

    def one_batch(idx):
        for attempt in range(6):
            try:
                resp = _client.embeddings.create(model=model, input=[texts[i] for i in idx],
                                                 extra_body={"usage": {"include": True}})
                if len(resp.data) == len(idx):
                    break
            except Exception:
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt * 2)
        _count(model, _usage(resp), cached=False)
        new = []
        for i, d in zip(idx, sorted(resp.data, key=lambda d: d.index)):
            out[i] = np.asarray(d.embedding, dtype=np.float32)
            new.append((keys[i], out[i]))
        _put_emb(new)

    parts = [todo[s:s + batch] for s in range(0, len(todo), batch)]
    if len(parts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one_batch, parts))
    elif parts:
        one_batch(parts[0])
    return np.stack(out) if out else np.zeros((0, 0), dtype=np.float32)


def parse_json(text):
    """Return the first JSON object in `text` (models sometimes wrap it in a code fence)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None


def usage_report():
    tot = sum(u["cost"] for u in USAGE.values())
    return {"by_model": USAGE, "total_cost_usd": round(tot, 4)}
