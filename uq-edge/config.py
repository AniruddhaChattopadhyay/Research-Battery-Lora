"""Central configuration for UQ-Edge experiments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    hf_id: str
    short_name: str
    family: str

    def __str__(self):
        return self.short_name


@dataclass
class QuantSpec:
    method: str  # "fp16", "int8", "nf4", "gptq", "awq"
    bits: int
    quantized_model_id: Optional[str] = None  # override HF ID for pre-quantized models

    def __str__(self):
        return self.method


@dataclass
class BenchmarkSpec:
    name: str
    hf_dataset: str
    hf_config: Optional[str] = None
    split: str = "test"
    max_samples: int = 1000
    task_type: str = "open_gen"  # "open_gen" or "multiple_choice"

    def __str__(self):
        return self.name


@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature_sampling: float = 0.7
    self_consistency_k: int = 5
    calibration_split: float = 0.2
    batch_size: int = 1  # keep 1 for logit capture simplicity
    seed: int = 42


@dataclass
class ExperimentConfig:
    models: list[ModelSpec] = field(default_factory=list)
    quant_levels: list[QuantSpec] = field(default_factory=list)
    benchmarks: list[BenchmarkSpec] = field(default_factory=list)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: str = "raw_outputs"
    results_dir: str = "results"
    plots_dir: str = "plots"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ALL_MODELS = [
    ModelSpec("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M", "smollm2"),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2-1.7B", "smollm2"),
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B", "qwen2.5"),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "qwen2.5"),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "qwen2.5"),
    ModelSpec("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B", "tinyllama"),
    ModelSpec("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B", "llama3.2"),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B", "llama3.2"),
]

ALL_QUANT_LEVELS = [
    QuantSpec("fp16", 16),
    QuantSpec("int8", 8),
    QuantSpec("nf4", 4),
    # GPTQ/AWQ added when pre-quantized models are confirmed
]

ALL_BENCHMARKS = [
    BenchmarkSpec(
        "triviaqa", "trivia_qa", hf_config="rc.nocontext",
        split="validation", max_samples=1000, task_type="open_gen",
    ),
    BenchmarkSpec(
        "mmlu", "cais/mmlu", hf_config="all",
        split="test", max_samples=1000, task_type="multiple_choice",
    ),
    BenchmarkSpec(
        "gsm8k", "openai/gsm8k", hf_config="main",
        split="test", max_samples=1000, task_type="open_gen",
    ),
    BenchmarkSpec(
        "truthfulqa", "truthful_qa", hf_config="multiple_choice",
        split="validation", max_samples=817, task_type="multiple_choice",
    ),
    BenchmarkSpec(
        "csqa", "tau/commonsense_qa",
        split="validation", max_samples=1000, task_type="multiple_choice",
    ),
]


def _find(registry, name):
    for item in registry:
        if str(item) == name or getattr(item, "short_name", None) == name or getattr(item, "name", None) == name:
            return item
    names = [str(x) for x in registry]
    raise ValueError(f"{name!r} not found. Available: {names}")


def find_model(name: str) -> ModelSpec:
    return _find(ALL_MODELS, name)


def find_quant(name: str) -> QuantSpec:
    return _find(ALL_QUANT_LEVELS, name)


def find_benchmark(name: str) -> BenchmarkSpec:
    return _find(ALL_BENCHMARKS, name)


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

def get_quick_test_config() -> ExperimentConfig:
    """1 model, FP16 only, 1 benchmark, 10 samples. For smoke testing."""
    return ExperimentConfig(
        models=[find_model("SmolLM2-135M")],
        quant_levels=[find_quant("fp16")],
        benchmarks=[BenchmarkSpec(
            "triviaqa", "trivia_qa", hf_config="rc.nocontext",
            split="validation", max_samples=10, task_type="open_gen",
        )],
    )


def get_local_dev_config() -> ExperimentConfig:
    """Small model, FP16 only (MPS-safe), 2 benchmarks, 50 samples."""
    return ExperimentConfig(
        models=[find_model("SmolLM2-135M")],
        quant_levels=[find_quant("fp16")],
        benchmarks=[
            BenchmarkSpec(
                "triviaqa", "trivia_qa", hf_config="rc.nocontext",
                split="validation", max_samples=50, task_type="open_gen",
            ),
            BenchmarkSpec(
                "mmlu", "cais/mmlu", hf_config="all",
                split="test", max_samples=50, task_type="multiple_choice",
            ),
        ],
    )


def get_full_config() -> ExperimentConfig:
    """All models, all quant levels, all benchmarks, full sample sizes."""
    return ExperimentConfig(
        models=ALL_MODELS,
        quant_levels=ALL_QUANT_LEVELS,
        benchmarks=ALL_BENCHMARKS,
    )


def output_path(cfg: ExperimentConfig, model: ModelSpec, quant: QuantSpec, bench: BenchmarkSpec) -> str:
    """Return the .npz path for a given (model, quant, benchmark) triple."""
    import os
    fname = f"{model.short_name}_{quant.method}_{bench.name}.npz"
    return os.path.join(cfg.output_dir, fname)


def metrics_path(cfg: ExperimentConfig, model: ModelSpec, quant: QuantSpec, bench: BenchmarkSpec) -> str:
    """Return the metrics JSON path."""
    import os
    fname = f"{model.short_name}_{quant.method}_{bench.name}_metrics.json"
    return os.path.join(cfg.results_dir, fname)
