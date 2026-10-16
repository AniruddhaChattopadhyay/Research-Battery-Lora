"""Baselines through the OpenRouter chat API. Both methods return answers in the shape of
the decisions API, so that one analysis reads all systems.

structured  B1 and B3: a System Two model writes a probability for each option in JSON
            (structured outputs). We normalize the values of each question to a sum of 1.
readout     B2: the label readout of an open backbone. The prompt lists the options with
            letters. The model writes one token, and we take the probabilities of the
            letters from the top logprobs. One call for each question.
"""
import json
import math
import os
import random
import string
import time

import httpx

import env  # noqa: F401  (loads the API key from .env)

URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 524, 529}

# One provider for each backbone, with logprobs and the highest precision on OpenRouter
# (checked on 2026-10-16 with /api/v1/models/<id>/endpoints).
READOUT_PIN = {
    "qwen/qwen3.5-9b": "Parasail",  # bf16
    "qwen/qwen3.8-27b": "Cerebras",  # fp16
    "google/gemma-4-31b-it": "Novita",  # bf16
}

MAX_STRICT_OPTIONS = 26

STRUCTURED_SYSTEM = (
    "You answer questions about an input. For each question, give the probability that each "
    "option is the correct answer. The probabilities for one question must add up to 1. "
    "Return only the JSON object."
)


def options(q):
    """The options of a question card as (name, text) pairs."""
    if q["type"] == "noul":
        return [("true", q["criteria"]["true"]), ("false", q["criteria"]["false"])]
    if q["type"] == "score":
        return [(str(i), text) for i, text in enumerate(q["criteria"])]
    return list(q["criteria"].items())


def as_answer(q, probs):
    """Probabilities for each option -> an answer in the shape of the decisions API."""
    total = sum(probs.values())
    if total <= 0:
        return None
    probs = {k: v / total for k, v in probs.items()}
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}
    top = max(probs, key=probs.get)
    if q["type"] == "score":
        return {"type": "score", "score": sum(int(k) * v for k, v in probs.items()), "probabilities": probs}
    return {"type": "choice", "choice": top, "probabilities": probs}


def post(client, body, max_tries=6):
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    for attempt in range(1, max_tries + 1):
        start = time.perf_counter()
        try:
            r = client.post(URL, json=body, headers=headers, timeout=300)
        except httpx.HTTPError as e:
            if attempt == max_tries:
                return None, None, attempt, {"error": repr(e)}
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
        return r.status_code, latency_ms, attempt, data


def structured(client, model, state, questions, effort="low"):
    """B1 and B3: one call for the full question card. effort is the reasoning effort.

    For the image test, the state is a list of parts: plain strings and image_url parts
    (the shape of the decisions API). Then the input goes to the model as these parts.
    """
    head = None
    if isinstance(state, list):
        head = [{"type": "text", "text": "Input: the image below."}]
        head += [{"type": "text", "text": p} if isinstance(p, str) else p for p in state]
        lines = ["Questions:"]
    else:
        lines = [f"Input:\n<<<\n{state}\n>>>", "", "Questions:"]
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    for qname, q in questions.items():
        lines.append(f"- {qname}: {q['instructions']}")
        for name, text in options(q):
            lines.append(f"    - {name}: {text}")
        names = [name for name, _ in options(q)]
        if len(names) > MAX_STRICT_OPTIONS:
            # Claude rejects a strict schema with 77 properties ("The compiled grammar is too
            # large"). For such questions, the model lists only the likely options.
            lines.append(
                f"    (For {qname}, give a list of the likely options and their probabilities. "
                "Use the exact option names. Leave out the options with a probability below 0.01.)"
            )
            schema["properties"][qname] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"option": {"type": "string"}, "probability": {"type": "number"}},
                    "required": ["option", "probability"],
                    "additionalProperties": False,
                },
            }
        else:
            schema["properties"][qname] = {
                "type": "object",
                "properties": {name: {"type": "number"} for name in names},
                "required": names,
                "additionalProperties": False,
            }
        schema["required"].append(qname)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": STRUCTURED_SYSTEM},
            {"role": "user", "content": "\n".join(lines) if head is None
             else head + [{"type": "text", "text": "\n".join(lines)}]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answers", "strict": True, "schema": schema}},
        "reasoning": {"effort": effort},
        "max_tokens": 16000,
        "usage": {"include": True},
    }
    status, latency_ms, attempts, data = post(client, body)
    response = {"raw": data}
    if status == 200:
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content[content.index("{"): content.rindex("}") + 1])
            answers = {}
            for qname, q in questions.items():
                given = parsed.get(qname, {})
                if isinstance(given, list):  # the list form for many options
                    given = {g.get("option"): g.get("probability") for g in given if isinstance(g, dict)}
                probs = {name: max(0.0, float(given.get(name, 0) or 0)) for name, _ in options(q)}
                answers[qname] = as_answer(q, probs)
            response.update(answers=answers, model=data.get("model"), provider=data.get("provider"), usage=data.get("usage"))
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as e:
            # AttributeError: the content is None, for example after a refusal
            status = "parse_error"
            response["error"] = repr(e)
    return {"status": status, "latency_ms": latency_ms, "attempts": attempts, "response": response}


READOUT_TEMPLATE = """Read the input and answer the question.

Input:
<<<
{state}
>>>

Question: {question}

Options:
{options}

Answer with the letter of one option only."""


def readout(client, model, state, questions):
    """B2: one call for each question. The answer is a softmax over the option letters."""
    answers, raws, usage, latency, attempts_all, status = {}, {}, {}, 0.0, 0, 200
    for qname, q in questions.items():
        opts = options(q)
        letters = string.ascii_uppercase[: len(opts)]
        text = READOUT_TEMPLATE.format(
            state=state,
            question=q["instructions"],
            options="\n".join(f"{letter}. {t}" for letter, (_, t) in zip(letters, opts)),
        )
        body = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "reasoning": {"enabled": False},
            "provider": {"only": [READOUT_PIN[model]], "allow_fallbacks": False, "require_parameters": True},
            "usage": {"include": True},
        }
        s, latency_ms, attempts, data = post(client, body)
        raws[qname] = data
        attempts_all += attempts
        latency += latency_ms or 0
        if s != 200:
            status = s
            continue
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            status = "no_logprobs"
            continue
        probs = {name: 0.0 for name, _ in opts}
        for t in top:
            token = t["token"].strip()
            if token in letters:
                probs[opts[letters.index(token)][0]] += math.exp(t["logprob"])
        answers[qname] = as_answer(q, probs)
        for k, v in (data.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        provider = data.get("provider")
    response = {"answers": answers, "usage": usage, "raw": raws}
    if answers:
        response.update(model=model, provider=provider)
    return {"status": status, "latency_ms": round(latency, 1), "attempts": attempts_all, "response": response}
