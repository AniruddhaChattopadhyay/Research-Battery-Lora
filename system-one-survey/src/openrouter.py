"""Call System One models through the OpenRouter decisions API.

Endpoint: POST https://openrouter.ai/api/alpha/decisions
Request:  model, state and questions (the Jev shape: noul, choice, score, box).
Response: answers, the resolved model (with its version), provider and usage (with cost).
The API has no temperature or seed setting.
"""
import os
import random
import time

import httpx

import env  # noqa: F401  (loads the API key from .env)

URL = "https://openrouter.ai/api/alpha/decisions"
RETRY_STATUS = {429, 500, 502, 503, 524, 529}

# OpenRouter sends some models to more than one provider (E0: Clef went to PrimeIntellect,
# DeepInfra and Cloudflare in one run). Each provider can serve the model differently, so
# we pin one provider with no fallback. Use the vendor when it serves its own model.
PROVIDER_PIN = {
    "cloudflare/clef": "Cloudflare",
    "cloudflare/clef-flash": "Cloudflare",
    "cloudflare/clef-omni": "Cloudflare",
    "jaredpalmer/kev-4b": "Phala",
}


def decide(client, model, state, questions, max_tries=5):
    """Send one decisions request. Return the status, latency, attempts and the JSON body.

    The latency is the time of the last attempt, from send to full response, and it
    includes the network. It is not the protocol of experiment E8.
    """
    body = {"model": model, "state": state, "questions": questions}
    if model in PROVIDER_PIN:
        body["provider"] = {"only": [PROVIDER_PIN[model]], "allow_fallbacks": False}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    for attempt in range(1, max_tries + 1):
        start = time.perf_counter()
        try:
            r = client.post(URL, json=body, headers=headers, timeout=120)
        except httpx.HTTPError as e:
            if attempt == max_tries:
                return {"status": None, "latency_ms": None, "attempts": attempt, "response": {"error": repr(e)}}
            time.sleep(2**attempt + random.random())
            continue
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if r.status_code in RETRY_STATUS and attempt < max_tries:
            time.sleep(2**attempt + random.random())
            continue
        try:
            data = r.json()
        except ValueError:
            data = {"raw_text": r.text[:2000]}
        return {"status": r.status_code, "latency_ms": latency_ms, "attempts": attempt, "response": data}
