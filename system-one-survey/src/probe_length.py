"""Probe: does a system read the full input? (added on 2026-10-16, exploratory)

The OpenRouter image guide says that Clef and Clef-flash read only about the first 2,000
text tokens and drop the rest without an error. This probe puts one fact ("the access
code of this document is blue") at different positions in a long input of arXiv abstracts
(about 3,800 tokens). One choice question asks for the code (4 options). If a system does
not read a part of the input, its accuracy falls to chance (25%) when the fact is in that part.

  PYTHONPATH=src uv run python src/probe_length.py --models cloudflare/clef,typesafe/jev-1.14

Output: runs/probe-length/<model>.jsonl. The item "length-calib" has a short input. Its
input tokens give the size of the question card, so that the analysis can convert the
character position of the fact to a token position.
"""
import argparse
import json
import pathlib

import httpx

from run import run_model

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORDS = ["red", "blue", "green", "yellow"]
TOTAL_CHARS = 16000
# Target positions in tokens; about 4.2 characters for each token in arXiv abstracts.
TARGET_TOKENS = [50, 500, 1000, 1500, 1800, 2000, 2200, 2500, 3000, 3600]
CHARS_PER_TOKEN = 4.2
FILLER_STARTS = [0, 100, 200]  # three different fillers: the first abstract of each

QUESTION = {
    "code": {
        "type": "choice",
        "instructions": "What is the access code of this document, as the note in the input says?",
        "criteria": {w: f"The access code is {w}." for w in WORDS},
    }
}


def filler(start, total=TOTAL_CHARS):
    lines = (ROOT / "fixtures" / "arxiv-fresh-2026-10.jsonl").read_text(encoding="utf-8").splitlines()
    parts, size = [], 0
    for line in lines[start:]:
        abstract = " ".join(json.loads(line)["abstract"].split())
        parts.append(abstract)
        size += len(abstract) + 2
        if size >= total:
            break
    return "\n\n".join(parts)[:total]


def insert(text, pos, sentence):
    """Put the sentence after the last sentence end before position pos."""
    cut = max(text.rfind(". ", 0, pos), text.rfind(".\n", 0, pos))
    cut = 0 if cut < 0 else cut + 1
    return text[:cut] + " " + sentence + text[cut:], cut


# Two more sets, added after the first result (all answers correct up to about 2,850 tokens):
#   long   a plain text input of 36,000 characters (about 6,900 tokens)
#   array  the first input as a one-element array (the form of the image guide)
LONG_CHARS = 36000
LONG_SHARES = [0.05, 0.25, 0.45, 0.6, 0.75, 0.9, 0.98]


def items(sets=("base",)):
    out = [{"id": "length-calib", "state": "Hello.", "questions": QUESTION, "gold": {"code": "red"},
            "meta": {"calib": True}}]
    for f, start in enumerate(FILLER_STARTS):
        for word in WORDS:
            sentence = f"Note for the reader: the access code of this document is {word}."
            if "base" in sets or "array" in sets:
                text = filler(start)
                for target in TARGET_TOKENS:
                    state, cut = insert(text, int(target * CHARS_PER_TOKEN), sentence)
                    meta = {"filler": f, "target_tokens": target, "pos_chars": cut, "total_chars": len(state)}
                    if "base" in sets:
                        out.append({"id": f"length-f{f}-t{target}-{word}", "state": state, "questions": QUESTION,
                                    "gold": {"code": word}, "meta": {**meta, "set": "base"}})
                    if "array" in sets:
                        out.append({"id": f"length-array-f{f}-t{target}-{word}", "state": [state], "questions": QUESTION,
                                    "gold": {"code": word}, "meta": {**meta, "set": "array"}})
            if "long" in sets:
                text = filler(start, LONG_CHARS)
                for share in LONG_SHARES:
                    state, cut = insert(text, int(share * len(text)), sentence)
                    out.append({"id": f"length-long-f{f}-s{share}-{word}", "state": state, "questions": QUESTION,
                                "gold": {"code": word},
                                "meta": {"filler": f, "share": share, "pos_chars": cut, "total_chars": len(state), "set": "long"}})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sets", default="base", help="base, long, array (separated by commas)")
    a = p.parse_args()
    jobs = [("length", item) for item in items(tuple(a.sets.split(",")))]
    with httpx.Client() as client:
        for model in a.models.split(","):
            run_model(client, "probe-length", model.strip(), jobs, a.workers)


if __name__ == "__main__":
    main()
