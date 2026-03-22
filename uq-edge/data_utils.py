"""Benchmark loading and answer checking."""

import re
import random
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from config import BenchmarkSpec


@dataclass
class Sample:
    question: str
    reference_answer: str
    choices: Optional[list[str]] = None
    choice_labels: Optional[list[str]] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders — each returns list[Sample]
# ---------------------------------------------------------------------------

def _load_triviaqa(spec: BenchmarkSpec, seed: int) -> list[Sample]:
    ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
    ds = ds.shuffle(seed=seed).select(range(min(spec.max_samples, len(ds))))
    samples = []
    for row in ds:
        answer_aliases = row["answer"]["aliases"] + [row["answer"]["value"]]
        prompt = f"Answer the following question in a few words.\n\nQuestion: {row['question']}\nAnswer:"
        samples.append(Sample(
            question=prompt,
            reference_answer=row["answer"]["value"],
            metadata={"aliases": answer_aliases},
        ))
    return samples


def _load_mmlu(spec: BenchmarkSpec, seed: int) -> list[Sample]:
    ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
    ds = ds.shuffle(seed=seed).select(range(min(spec.max_samples, len(ds))))
    labels = ["A", "B", "C", "D"]
    samples = []
    for row in ds:
        choices_text = "\n".join(
            f"{labels[i]}. {row['choices'][i]}" for i in range(len(row["choices"]))
        )
        prompt = (
            f"Answer the following multiple choice question. "
            f"Reply with just the letter (A, B, C, or D).\n\n"
            f"Question: {row['question']}\n{choices_text}\nAnswer:"
        )
        samples.append(Sample(
            question=prompt,
            reference_answer=labels[row["answer"]],
            choices=row["choices"],
            choice_labels=labels,
            metadata={"subject": row.get("subject", "")},
        ))
    return samples


def _load_gsm8k(spec: BenchmarkSpec, seed: int) -> list[Sample]:
    ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
    ds = ds.shuffle(seed=seed).select(range(min(spec.max_samples, len(ds))))
    samples = []
    for row in ds:
        # Extract numeric answer after ####
        answer_text = row["answer"]
        numeric = answer_text.split("####")[-1].strip().replace(",", "")
        prompt = (
            f"Solve the following math problem. "
            f"Give your final numeric answer after 'The answer is'.\n\n"
            f"Problem: {row['question']}\nSolution:"
        )
        samples.append(Sample(
            question=prompt,
            reference_answer=numeric,
        ))
    return samples


def _load_truthfulqa(spec: BenchmarkSpec, seed: int) -> list[Sample]:
    ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
    ds = ds.shuffle(seed=seed).select(range(min(spec.max_samples, len(ds))))
    samples = []
    for row in ds:
        # multiple_choice config: mc1_targets has choices + labels
        choices = row["mc1_targets"]["choices"]
        correct_idx = row["mc1_targets"]["labels"].index(1)
        labels = [chr(65 + i) for i in range(len(choices))]
        choices_text = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
        prompt = (
            f"Answer the following question. Reply with just the letter.\n\n"
            f"Question: {row['question']}\n{choices_text}\nAnswer:"
        )
        samples.append(Sample(
            question=prompt,
            reference_answer=labels[correct_idx],
            choices=choices,
            choice_labels=labels,
        ))
    return samples


def _load_csqa(spec: BenchmarkSpec, seed: int) -> list[Sample]:
    ds = load_dataset(spec.hf_dataset, split=spec.split)
    ds = ds.shuffle(seed=seed).select(range(min(spec.max_samples, len(ds))))
    samples = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices_text = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(labels)))
        prompt = (
            f"Answer the following question. Reply with just the letter.\n\n"
            f"Question: {row['question']}\n{choices_text}\nAnswer:"
        )
        samples.append(Sample(
            question=prompt,
            reference_answer=row["answerKey"],
            choices=texts,
            choice_labels=labels,
        ))
    return samples


_LOADERS = {
    "triviaqa": _load_triviaqa,
    "mmlu": _load_mmlu,
    "gsm8k": _load_gsm8k,
    "truthfulqa": _load_truthfulqa,
    "csqa": _load_csqa,
}


def load_benchmark(spec: BenchmarkSpec, seed: int = 42) -> list[Sample]:
    loader = _LOADERS.get(spec.name)
    if loader is None:
        raise ValueError(f"Unknown benchmark: {spec.name}. Available: {list(_LOADERS)}")
    samples = loader(spec, seed)
    print(f"  Loaded {len(samples)} samples from {spec.name}")
    return samples


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip, remove articles and punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def check_answer(sample: Sample, model_answer: str, benchmark_name: str) -> bool:
    model_answer = model_answer.strip()

    if benchmark_name in ("mmlu", "truthfulqa", "csqa"):
        # Multiple choice: extract the letter
        letter = _extract_letter(model_answer)
        return letter == sample.reference_answer.upper()

    if benchmark_name == "gsm8k":
        extracted = _extract_number(model_answer)
        try:
            return float(extracted) == float(sample.reference_answer)
        except (ValueError, TypeError):
            return False

    if benchmark_name == "triviaqa":
        # Fuzzy match against all aliases
        norm_answer = _normalize(model_answer)
        aliases = sample.metadata.get("aliases", [sample.reference_answer])
        for alias in aliases:
            if _normalize(alias) in norm_answer or norm_answer in _normalize(alias):
                return True
        return False

    # Fallback: exact normalized match
    return _normalize(model_answer) == _normalize(sample.reference_answer)


def _extract_letter(text: str) -> str:
    """Extract a single letter answer (A-E) from model output."""
    text = text.strip()
    if text and text[0].upper() in "ABCDE":
        return text[0].upper()
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else ""


def _extract_number(text: str) -> str:
    """Extract the last number from model output."""
    # Look for 'the answer is X' pattern first
    match = re.search(r"the answer is[:\s]*([+-]?[\d,]+\.?\d*)", text.lower())
    if match:
        return match.group(1).replace(",", "")
    # Fall back to last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""
