"""Model loading at various quantization levels."""

import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ModelSpec, QuantSpec


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def can_quantize(quant: QuantSpec) -> bool:
    """Check whether this quant method works on the current hardware."""
    if quant.method == "fp16":
        return True
    # bitsandbytes, GPTQ, AWQ all require CUDA
    return torch.cuda.is_available()


def load_model(model_spec: ModelSpec, quant_spec: QuantSpec):
    """Load model + tokenizer at the specified quantization level.

    Returns (model, tokenizer).
    """
    device = get_device()
    model_id = quant_spec.quantized_model_id or model_spec.hf_id

    common_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )

    if quant_spec.method == "fp16":
        if device == "mps":
            # MPS: use float32 for stability
            common_kwargs["torch_dtype"] = torch.float32
        else:
            common_kwargs["torch_dtype"] = torch.float16

    elif quant_spec.method == "int8":
        common_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    elif quant_spec.method == "nf4":
        common_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    elif quant_spec.method == "gptq":
        from transformers import GPTQConfig
        common_kwargs["quantization_config"] = GPTQConfig(bits=4, use_exllama=False)

    elif quant_spec.method == "awq":
        # AWQ models are loaded from pre-quantized checkpoints
        pass

    else:
        raise ValueError(f"Unknown quant method: {quant_spec.method}")

    print(f"  Loading {model_id} ({quant_spec.method})...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_spec.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def unload_model(model):
    """Free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_mb(model) -> float:
    """Estimate model memory footprint in MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)


def measure_latency_ms(model, tokenizer, prompt: str = "Hello", n_runs: int = 5) -> float:
    """Measure average inference latency in milliseconds."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # warm up
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return sum(times) / len(times)
