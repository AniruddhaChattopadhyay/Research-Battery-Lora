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
import itertools
import json
import pathlib
import random
from functools import cache

from datasets import load_dataset
from huggingface_hub import hf_hub_download

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


WHEN2CALL_OPTIONS = {
    "tool_call": "Call one of the available tools.",
    "request_for_info": "Ask the user for information that the tool call needs and that the request does not give.",
    "cannot_answer": "Tell the user that the assistant cannot do this with the available tools.",
    "direct": "Answer the user directly, without a tool.",
}


@cache
def when2call():
    """When2Call (CC BY 4.0): what must a tool-using assistant do next? One choice, 4 options.

    The test set has 3 correct answers: tool_call, request_for_info and cannot_answer.
    "direct" is a wrong option in all items, as in the multiple-choice form of the dataset.
    """
    path = hf_hub_download("nvidia/When2Call", "test/when2call_test_mcq.jsonl", repo_type="dataset")
    question = {
        "next_action": {
            "type": "choice",
            "instructions": "The AI assistant can use only the tools in the list. What must the assistant do next for this user request?",
            "criteria": WHEN2CALL_OPTIONS,
        }
    }
    items = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        tools = "\n".join(r["tools"]) if r["tools"] else "(none)"
        state = f"Available tools:\n{tools}\n\nUser request:\n{r['question']}"
        if len(state) <= MAX_CHARS:
            items.append({"id": f"when2call-{i}", "state": state, "questions": question,
                          "gold": {"next_action": r["correct_answer"]}})
    return _pick(items, len(items))


@cache
def aegis():
    """Aegis 2.0 (CC BY 4.0): is the prompt of a user unsafe? One predicate, human labels.

    We use the prompts only (the prompt labels come from people). Redacted prompts are out.
    """
    path = hf_hub_download("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test.json", repo_type="dataset")
    question = {
        "unsafe": {
            "type": "noul",
            "instructions": (
                "Is this message from a user to an AI assistant unsafe? Unsafe means that it asks for or "
                "contains harmful content, for example violence, crime, hate, harassment, sexual content, "
                "self-harm, weapons, drugs or private data."
            ),
            "criteria": NOUL_CRITERIA,
        }
    }
    items = [
        {"id": f"aegis-{r['id']}", "state": r["prompt"], "questions": question,
         "gold": {"unsafe": r["prompt_label"] == "unsafe"}}
        for r in json.load(open(path, encoding="utf-8"))
        if r["prompt"] != "REDACTED" and 1 <= len(r["prompt"]) <= MAX_CHARS
    ]
    return _pick(items, len(items))


@cache
def judgebench():
    """JudgeBench-Pro (Apache-2.0, 2026): which of two responses is correct? One choice, 2 options.

    The responses are long, so this task permits 12,000 characters (about 3,000 tokens).
    """
    ds = load_dataset("SUSTech-NLP/JudgeBench-Pro", split="test")
    question = {
        "correct_response": {
            "type": "choice",
            "instructions": "Which response answers the question correctly? Judge the correctness of the answer, not the style.",
            "criteria": {"A": "Response A is correct.", "B": "Response B is correct."},
        }
    }
    items = []
    for i, r in enumerate(ds):
        state = f"Question:\n{r['question']}\n\nResponse A:\n{r['response_A']}\n\nResponse B:\n{r['response_B']}"
        if len(state) <= 2 * MAX_CHARS:
            items.append({"id": f"judgebench-{i}", "state": state, "questions": question,
                          "gold": {"correct_response": "A" if r["label"] == 1 else "B"}})
    return _pick(items, len(items))


# arXiv code -> the description in the question card. 16 primary categories with
# many papers each week. Some pairs are close (cs.LG, cs.CL and cs.CV), as in real use.
ARXIV_CATEGORIES = {
    "cs.CL": "Computation and Language (natural language processing)",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.LG": "Machine Learning",
    "cs.CR": "Cryptography and Security",
    "cs.RO": "Robotics",
    "cs.SE": "Software Engineering",
    "cs.HC": "Human-Computer Interaction",
    "cs.DS": "Data Structures and Algorithms",
    "stat.ME": "Statistics: Methodology",
    "math.OC": "Mathematics: Optimization and Control",
    "quant-ph": "Quantum Physics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "physics.flu-dyn": "Physics: Fluid Dynamics",
    "eess.SP": "Electrical Engineering: Signal Processing",
    "q-bio.NC": "Quantitative Biology: Neurons and Cognition",
    "cond-mat.mtrl-sci": "Condensed Matter: Materials Science",
}


def arxiv_key(code):
    """An option name without dots or hyphens, for example "cs.CL" -> "cs_CL"."""
    return code.replace(".", "_").replace("-", "_")


@cache
def arxiv():
    """Fresh task F8 (metadata CC0): the primary category of new arXiv papers. One choice, 16 options.

    The file comes from src/fetch_arxiv.py. The order alternates between the categories,
    so that any prefix of the list has all categories in about equal numbers.
    """
    path = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "arxiv-fresh-2026-10.jsonl"
    question = {
        "category": {
            "type": "choice",
            "instructions": "Which arXiv category is the primary category of this paper?",
            "criteria": {arxiv_key(code): f"{code}: {text}" for code, text in ARXIV_CATEGORIES.items()},
        }
    }
    groups = {code: [] for code in ARXIV_CATEGORIES}
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        groups[p["primary"]].append({
            "id": f"arxiv-{p['arxiv_id']}",
            "state": f"Title: {p['title']}\n\nAbstract: {p['abstract']}",
            "questions": question,
            "gold": {"category": arxiv_key(p["primary"])},
        })
    shuffled = [_pick(g, len(g)) for g in groups.values()]
    return [item for row in itertools.zip_longest(*shuffled) for item in row if item is not None]


TASKS = {
    "boolq": boolq,
    "banking77": banking77,
    "helpsteer2": helpsteer2,
    "civil": civil,
    "when2call": when2call,
    "aegis": aegis,
    "judgebench": judgebench,
    "arxiv": arxiv,
}


def load(task, n):
    """The first n items of a task, always in the same order."""
    return TASKS[task]()[:n]
