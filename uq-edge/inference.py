"""Phase 1: Run models on benchmarks, capture logits, save raw outputs."""

import json
import os
import time

import numpy as np
import torch

from config import ExperimentConfig, ModelSpec, QuantSpec, BenchmarkSpec, output_path
from data_utils import Sample, check_answer


def run_inference(
    model,
    tokenizer,
    samples: list[Sample],
    benchmark_name: str,
    max_new_tokens: int = 128,
) -> dict:
    """Run greedy inference on all samples, capturing logits.

    Returns a dict ready to be saved as .npz.
    """
    device = next(model.parameters()).device

    answers = []
    is_correct = []
    mean_log_probs = []
    mean_entropies = []
    max_probs = []  # MSP: mean of max softmax prob per token

    for i, sample in enumerate(samples):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    [{i + 1}/{len(samples)}]")

        inputs = tokenizer(
            sample.question, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode generated tokens only
        gen_ids = outputs.sequences[0, input_len:]
        answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        answers.append(answer_text)
        is_correct.append(check_answer(sample, answer_text, benchmark_name))

        # Extract per-token logit statistics from scores
        if outputs.scores:
            token_log_probs = []
            token_entropies = []
            token_max_probs = []
            for score in outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                log_p = torch.log_softmax(score[0], dim=-1)

                # Max prob (for MSP)
                token_max_probs.append(probs.max().item())

                # Log-prob of the generated token
                # (the token that was actually selected)
                # We use the max prob token since greedy
                token_log_probs.append(log_p.max().item())

                # Entropy: -sum(p * log(p))
                entropy = -(probs * log_p).sum().item()
                token_entropies.append(entropy)

            mean_log_probs.append(float(np.mean(token_log_probs)))
            mean_entropies.append(float(np.mean(token_entropies)))
            max_probs.append(float(np.mean(token_max_probs)))
        else:
            mean_log_probs.append(0.0)
            mean_entropies.append(0.0)
            max_probs.append(0.0)

    return {
        "answers": np.array(answers, dtype=object),
        "is_correct": np.array(is_correct, dtype=bool),
        "mean_log_probs": np.array(mean_log_probs, dtype=np.float32),
        "mean_entropies": np.array(mean_entropies, dtype=np.float32),
        "max_probs": np.array(max_probs, dtype=np.float32),  # MSP values
    }


def run_verbalized_confidence(
    model,
    tokenizer,
    samples: list[Sample],
    answers: list[str],
) -> np.ndarray:
    """Ask the model to self-report confidence for each answer.

    Returns array of confidence values in [0, 1]. NaN where parsing fails.
    """
    import re
    device = next(model.parameters()).device
    confidences = []

    for i, (sample, answer) in enumerate(zip(samples, answers)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Verbal conf [{i + 1}/{len(samples)}]")

        prompt = (
            f"{sample.question} {answer}\n\n"
            f"On a scale of 0 to 100, how confident are you that the above answer "
            f"is correct? Respond with just a number."
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        gen_text = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Parse number
        match = re.search(r"(\d+\.?\d*)", gen_text)
        if match:
            val = float(match.group(1))
            confidences.append(min(val / 100.0, 1.0))
        else:
            confidences.append(float("nan"))

    return np.array(confidences, dtype=np.float32)


def run_self_consistency(
    model,
    tokenizer,
    samples: list[Sample],
    benchmark_name: str,
    k: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
) -> np.ndarray:
    """Generate k sampled responses per question, return agreement fraction.

    Returns array of confidence values in [0, 1].
    """
    device = next(model.parameters()).device
    consistencies = []

    for i, sample in enumerate(samples):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    Self-consistency [{i + 1}/{len(samples)}]")

        inputs = tokenizer(
            sample.question, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        responses = []
        for _ in range(k):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
            text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
            responses.append(text.strip())

        # Count most common answer
        from collections import Counter
        if benchmark_name in ("mmlu", "truthfulqa", "csqa"):
            from data_utils import _extract_letter
            normalized = [_extract_letter(r) for r in responses]
        elif benchmark_name == "gsm8k":
            from data_utils import _extract_number
            normalized = [_extract_number(r) for r in responses]
        else:
            from data_utils import _normalize
            normalized = [_normalize(r)[:50] for r in responses]

        counts = Counter(normalized)
        most_common_count = counts.most_common(1)[0][1]
        consistencies.append(most_common_count / k)

    return np.array(consistencies, dtype=np.float32)


def save_raw_output(data: dict, path: str, metadata: dict):
    """Save inference results to .npz with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **data)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {path} ({size_mb:.1f} MB)")


def load_raw_output(path: str) -> dict:
    """Load .npz raw output file."""
    data = dict(np.load(path, allow_pickle=True))
    if "metadata" in data:
        data["metadata"] = json.loads(str(data["metadata"]))
    return data
