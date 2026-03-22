"""Smoke test — verify every component works end-to-end."""

import sys
import os
import time

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    print("1. Testing imports...")
    import torch
    import transformers
    import numpy
    import scipy
    import sklearn
    import matplotlib
    import datasets
    print(f"   torch={torch.__version__}, transformers={transformers.__version__}")
    print(f"   numpy={numpy.__version__}, scipy={scipy.__version__}")
    print("   OK")


def test_device():
    print("\n2. Testing device...")
    from model_utils import get_device
    device = get_device()
    print(f"   Device: {device}")
    if device == "cuda":
        import torch
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("   OK")


def test_config():
    print("\n3. Testing config...")
    from config import get_quick_test_config
    cfg = get_quick_test_config()
    print(f"   Models: {[str(m) for m in cfg.models]}")
    print(f"   Quants: {[str(q) for q in cfg.quant_levels]}")
    print(f"   Benchmarks: {[str(b) for b in cfg.benchmarks]}")
    print("   OK")


def test_data_loading():
    print("\n4. Testing data loading...")
    from config import get_quick_test_config
    from data_utils import load_benchmark
    cfg = get_quick_test_config()
    samples = load_benchmark(cfg.benchmarks[0], seed=42)
    print(f"   Sample question: {samples[0].question[:80]}...")
    print(f"   Reference answer: {samples[0].reference_answer}")
    print("   OK")


def test_model_loading():
    print("\n5. Testing model loading...")
    from config import get_quick_test_config
    from model_utils import load_model, unload_model, get_memory_mb
    cfg = get_quick_test_config()
    model, tokenizer = load_model(cfg.models[0], cfg.quant_levels[0])
    mem = get_memory_mb(model)
    print(f"   Model loaded. Memory: {mem:.0f} MB")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print("   OK")
    return model, tokenizer


def test_inference(model, tokenizer):
    print("\n6. Testing inference...")
    from config import get_quick_test_config
    from data_utils import load_benchmark
    from inference import run_inference

    cfg = get_quick_test_config()
    samples = load_benchmark(cfg.benchmarks[0], seed=42)

    t0 = time.time()
    data = run_inference(model, tokenizer, samples, cfg.benchmarks[0].name, max_new_tokens=64)
    elapsed = time.time() - t0

    print(f"   Ran {len(samples)} samples in {elapsed:.1f}s")
    print(f"   Accuracy: {data['is_correct'].mean():.0%}")
    print(f"   Mean MSP: {data['max_probs'].mean():.3f}")
    print(f"   Mean entropy: {data['mean_entropies'].mean():.3f}")
    print(f"   Sample answer: {data['answers'][0][:60]}...")
    print("   OK")
    return data


def test_uq_methods(data):
    print("\n7. Testing UQ methods...")
    from uq_methods import compute_all_scores
    scores = compute_all_scores(data)
    for name, vals in scores.items():
        if name.startswith("_"):
            continue
        print(f"   {name}: mean={vals.mean():.3f}, std={vals.std():.3f}")
    print("   OK")
    return scores


def test_metrics(scores, data):
    print("\n8. Testing metrics...")
    from metrics import compute_all_metrics
    results = compute_all_metrics(scores, data["is_correct"])
    for method, vals in results.items():
        if method.startswith("_"):
            continue
        print(f"   {method}: ECE={vals['ece']:.4f}, AUROC={vals['auroc']:.4f}, Brier={vals['brier']:.4f}")
    print("   OK")


def test_plotting():
    print("\n9. Testing plotting...")
    import matplotlib
    matplotlib.use("Agg")
    from metrics import compute_reliability_diagram
    import numpy as np
    # Dummy data
    conf = np.random.rand(100)
    correct = (np.random.rand(100) < conf).astype(bool)
    rel = compute_reliability_diagram(conf, correct)
    print(f"   Reliability bins: {len(rel['bin_centers'])} bins")
    print("   OK")


def main():
    print("=" * 50)
    print("UQ-Edge Quick Test")
    print("=" * 50)

    test_imports()
    test_device()
    test_config()
    test_data_loading()
    model, tokenizer = test_model_loading()
    data = test_inference(model, tokenizer)
    scores = test_uq_methods(data)
    test_metrics(scores, data)
    test_plotting()

    from model_utils import unload_model
    unload_model(model)

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
