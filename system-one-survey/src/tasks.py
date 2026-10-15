"""Tasks for the experiments: datasets, question cards and fixed samples of items.

Each item has:
  id         a stable id: "<task>-<row index in the split>"
  state      the input that the system judges (a string or an object)
  questions  the question card, in the OpenRouter decisions format (the Jev shape)
  gold       the reference answer for each question

The same seed and the same filter always give the same items in the same order.
Thus, "--n 5" gives the first 5 items of "--n 250", and a later run can send the
exact requests of an earlier run (experiment E7). Do not change a question card
after a run that you want to repeat: make a new task name instead.
"""
import random
from functools import cache

from datasets import load_dataset

SEED = 20261015
MAX_CHARS = 6000  # far below 8,000 tokens, the training limit of pplx-decider

NOUL_CRITERIA = {
    "true": "Yes, this is true for the input.",
    "false": "No, this is not true for the input.",
}


def _pick(rows, n):
    """A fixed random sample of n rows."""
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    return rows[:n]


@cache
def boolq():
    """BoolQ (CC BY-SA 3.0): yes/no questions about a passage. One predicate."""
    ds = load_dataset("google/boolq", split="validation")
    rows = [(i, r) for i, r in enumerate(ds) if len(r["passage"]) <= MAX_CHARS]
    items = []
    for i, r in _pick(rows, len(rows)):
        q = r["question"].strip().rstrip("?")
        q = q[0].upper() + q[1:] + "?"
        items.append({
            "id": f"boolq-{i}",
            "state": r["passage"],
            "questions": {
                "answer_is_yes": {
                    "type": "noul",
                    "instructions": f"Use only the passage. Is the answer to this question yes? Question: {q}",
                    "criteria": NOUL_CRITERIA,
                }
            },
            "gold": {"answer_is_yes": bool(r["answer"])},
        })
    return items


@cache
def banking77():
    """BANKING77 (CC BY 4.0): 77 intents of bank customers. One choice with 77 options."""
    # PolyAI/banking77 has only a loading script, which datasets v4 and later do not run.
    ds = load_dataset("legacy-datasets/banking77", split="test")
    names = ds.features["label"].names
    criteria = {name: name.replace("_", " ").lower() for name in names}
    question = {
        "intent": {
            "type": "choice",
            "instructions": "Which intent does this message from a bank customer express?",
            "criteria": criteria,
        }
    }
    rows = [(i, r) for i, r in enumerate(ds)]
    return [
        {"id": f"banking77-{i}", "state": r["text"], "questions": question, "gold": {"intent": names[r["label"]]}}
        for i, r in _pick(rows, len(rows))
    ]


# The attribute definitions follow the HelpSteer2 dataset card.
HELPSTEER_ATTRIBUTES = {
    "helpfulness": "Overall helpfulness of the response to the prompt.",
    "correctness": "Inclusion of all pertinent facts without errors.",
    "coherence": "Consistency and clarity of expression.",
    "complexity": "Intellectual depth that is necessary to write the response: basic language competency (low) or deep domain expertise (high).",
    "verbosity": "Amount of detail in the response, relative to what the prompt asks for.",
}
HELPSTEER_LEVELS = ["0: very low", "1: low", "2: medium", "3: high", "4: very high"]


@cache
def helpsteer2():
    """HelpSteer2 (CC BY 4.0): human ratings of responses, 0 to 4. Five scores."""
    ds = load_dataset("nvidia/HelpSteer2", split="validation")
    question = {
        name: {
            "type": "score",
            "instructions": f"Rate the response to the prompt for {name}. {definition}",
            "criteria": HELPSTEER_LEVELS,
        }
        for name, definition in HELPSTEER_ATTRIBUTES.items()
    }
    rows = [(i, r) for i, r in enumerate(ds) if len(r["prompt"]) + len(r["response"]) <= MAX_CHARS]
    # The state is one string: ScaleDown and Respan reject a state object with other keys (E0).
    return [
        {
            "id": f"helpsteer2-{i}",
            "state": f"Prompt:\n{r['prompt']}\n\nResponse:\n{r['response']}",
            "questions": question,
            "gold": {name: int(r[name]) for name in HELPSTEER_ATTRIBUTES},
        }
        for i, r in _pick(rows, len(rows))
    ]


CIVIL_ATTRIBUTES = {
    "toxicity": "Is the comment toxic, that is, rude, disrespectful or unreasonable?",
    "severe_toxicity": "Is the comment very hateful, aggressive or disrespectful?",
    "obscene": "Does the comment contain obscene or vulgar language?",
    "threat": "Does the comment contain a threat?",
    "insult": "Does the comment contain an insult?",
    "identity_attack": "Does the comment attack a person or a group because of their identity?",
    "sexual_explicit": "Does the comment contain sexually explicit content?",
}


@cache
def civil():
    """Civil Comments (labels CC0, text CC BY-SA 3.0): seven predicates on one comment.

    The gold values are the fractions of raters who said yes (soft labels). The order
    alternates between comments with toxicity >= 0.5 and < 0.5, so that any prefix of
    the list has both classes in equal numbers.
    """
    ds = load_dataset("google/civil_comments", split="test")
    question = {
        name: {"type": "noul", "instructions": text, "criteria": NOUL_CRITERIA}
        for name, text in CIVIL_ATTRIBUTES.items()
    }
    rows = [(i, r) for i, r in enumerate(ds) if 20 <= len(r["text"]) <= MAX_CHARS]
    toxic = _pick([x for x in rows if x[1]["toxicity"] >= 0.5], len(rows))
    clean = _pick([x for x in rows if x[1]["toxicity"] < 0.5], len(rows))
    mixed = [x for pair in zip(toxic, clean) for x in pair]
    return [
        {
            "id": f"civil-{i}",
            "state": r["text"],
            "questions": question,
            "gold": {name: float(r[name]) for name in CIVIL_ATTRIBUTES},
        }
        for i, r in mixed
    ]


TASKS = {"boolq": boolq, "banking77": banking77, "helpsteer2": helpsteer2, "civil": civil}


def load(task, n):
    """The first n items of a task, always in the same order."""
    return TASKS[task]()[:n]
