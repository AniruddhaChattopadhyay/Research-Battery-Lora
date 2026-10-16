"""The image test (added on 2026-10-16, exploratory): the same input as text and as an image.

For each item, two requests go to the same system in the same run:
  <id>#text   the input text, as in E2
  <id>#image  only an image of the same text (src/render.py), with the same question card
  <id>#list   the same text as a one-element list: a control for the list form of the image
              state (--variants text,image,list; added after the first image run)

Items that render() cannot show (an emoji, or too long) are left out, and the next items
take their place. Thus, every system gets the same items.

  PYTHONPATH=src uv run python src/run_image.py --tag image-pilot --tasks civil,banking77 --n 50 --models cloudflare/clef
  PYTHONPATH=src uv run python src/run_image.py --tag image-pilot --check --models cloudflare/clef
  PYTHONPATH=src uv run python src/run_image.py --tag e3-image-s2 --method structured --models openai/gpt-6-luna

--check sends one short test request to each system and prints the answers.
The images go to runs/<tag>/images/; the run records keep only the file name and the hash.
"""
import argparse
import json
import pathlib

import httpx

import tasks
from openrouter import decide
from render import image_part, render, sha256
from run import METHODS, ROOT, run_model

CHECK_TEXT = "The cat in this story is black. It sleeps on the red sofa."
CHECK_QUESTIONS = {
    "cat_is_black": {"type": "noul", "instructions": "Does the input say that the cat is black?",
                     "criteria": tasks.NOUL_CRITERIA},
    "sofa_color": {"type": "choice", "instructions": "What color is the sofa, according to the input?",
                   "criteria": {"red": "The sofa is red.", "green": "The sofa is green.", "blue": "The sofa is blue."}},
}


def jobs(tag, task_names, n, variants=("text", "image")):
    image_dir = ROOT / "runs" / tag / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out, skipped = [], {}
    for task in task_names:
        kept = 0
        skipped[task] = 0
        for item in tasks.TASKS[task]():
            if kept == n:
                break
            made = render(item["state"]) if isinstance(item["state"], str) else None
            if made is None:
                skipped[task] += 1
                continue
            data, settings = made
            path = image_dir / f"{item['id']}.png"
            if not path.exists():
                path.write_bytes(data)
            record = [{"type": "image_url", "image_file": str(path.relative_to(ROOT)), "sha256": sha256(data)}]
            base = {"questions": item["questions"], "gold": item["gold"]}
            if "text" in variants:
                out.append((task, {**base, "id": f"{item['id']}#text", "state": item["state"],
                                   "meta": {"variant": "text", "base_id": item["id"]}}))
            if "image" in variants:
                out.append((task, {**base, "id": f"{item['id']}#image", "state": [image_part(data)],
                                   "state_record": record, "meta": {"variant": "image", "base_id": item["id"], **settings}}))
            if "list" in variants:
                out.append((task, {**base, "id": f"{item['id']}#list", "state": [item["state"]],
                                   "meta": {"variant": "list", "base_id": item["id"]}}))
            kept += 1
    return out, skipped


def adapt(model, job):
    """ScaleDown Classify-1 rejects image_url parts. It takes an image as {"document": <base64 PNG>}
    and reads it with OCR (its error messages say "OCR processing failed" for other forms)."""
    task, item = job
    if model.startswith("scaledown/") and item["meta"]["variant"] == "image":
        b64 = item["state"][0]["image_url"]["url"].split(",", 1)[1]
        item = {**item, "state": {"document": b64}, "meta": {**item["meta"], "form": "document"}}
    if model.startswith("scaledown/") and item["meta"]["variant"] == "list":
        # ScaleDown rejects a list state; its object form {"text": ...} is the control
        item = {**item, "state": {"text": item["state"][0]}, "meta": {**item["meta"], "form": "text object"}}
    return task, item


def check(client, models):
    data, settings = render(CHECK_TEXT)
    print("test image:", settings)
    for model in models:
        _, item = adapt(model, ("check", {"state": [image_part(data)], "meta": {"variant": "image"}}))
        for name, state in (("image", item["state"]), ("text", CHECK_TEXT)):
            r = decide(client, model, state, CHECK_QUESTIONS)
            body = r["response"]
            answers = body.get("answers") if isinstance(body, dict) else None
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            short = {k: (v.get("noul") if v.get("type") == "noul" else v.get("choice")) for k, v in (answers or {}).items()}
            error = None if answers else json.dumps(body)[:300]
            print(f"{model:36s} {name:5s} status={r['status']} answers={short} "
                  f"tokens={usage.get('prompt_tokens', usage.get('input_tokens'))} error={error}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--tasks", default="civil,banking77")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--check", action="store_true")
    p.add_argument("--variants", default="text,image", help="text, image, list (separated by commas)")
    p.add_argument("--method", default="decisions", choices=METHODS,
                   help="decisions (System One) or structured (System Two, B1 settings)")
    a = p.parse_args()
    models = [m.strip() for m in a.models.split(",")]
    with httpx.Client() as client:
        if a.check:
            check(client, models)
            return
        todo, skipped = jobs(a.tag, a.tasks.split(","), a.n, tuple(a.variants.split(",")))
        print("left out (cannot render):", skipped)
        for model in models:
            run_model(client, a.tag, model, [adapt(model, job) for job in todo], a.workers, a.method)


if __name__ == "__main__":
    main()
