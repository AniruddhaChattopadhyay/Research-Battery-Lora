"""
Quick Test Script
=================
Run this first to verify everything is installed correctly.
Tests each component independently before running the full experiment.
"""

import sys


def test_imports():
    """Test that all required packages are installed."""
    print("1. Testing imports...")
    errors = []

    try:
        import torch
        print(f"   torch {torch.__version__} - OK")
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("   MPS (Apple Silicon) available")
        else:
            print("   WARNING: No GPU detected. Training will be slow.")
    except ImportError:
        errors.append("torch")

    for pkg in [
        "transformers", "peft", "trl", "bitsandbytes",
        "flwr", "flwr_datasets", "datasets", "numpy", "matplotlib",
    ]:
        try:
            __import__(pkg)
            print(f"   {pkg} - OK")
        except ImportError:
            errors.append(pkg)
            print(f"   {pkg} - MISSING")

    if errors:
        print(f"\n   FAILED: Missing packages: {errors}")
        print(f"   Run: pip install -r requirements.txt")
        return False
    print("   All imports OK!\n")
    return True


def test_battery_simulator():
    """Test the battery simulator."""
    print("2. Testing battery simulator...")
    from config import BatteryConfig, DeviceTierConfig
    from battery_simulator import BatterySimulator

    sim = BatterySimulator(
        num_clients=10,
        battery_cfg=BatteryConfig(),
        tier_cfg=DeviceTierConfig(),
        seed=42,
    )

    # Check initialization
    assert len(sim.devices) == 10
    active = sim.get_active_clients()
    assert len(active) == 10

    # Simulate a few rounds
    for _ in range(5):
        for cid in active:
            if sim.can_participate(cid):
                sim.update_after_training(cid, rank_used=8)
            else:
                sim.update_idle_round(cid)

    stats = sim.get_summary_stats()
    print(f"   Active: {stats['active_clients']}/10")
    print(f"   Avg battery: {stats['avg_battery']:.1f}%")
    print(f"   Total energy: {stats['total_energy_wh']:.2f} Wh")
    print(f"   Jain fairness: {stats['jain_fairness_index']:.4f}")
    print("   Battery simulator OK!\n")
    return True


def test_rank_policies():
    """Test all rank policies."""
    print("3. Testing rank policies...")
    from config import RankPolicyConfig, DeviceTierConfig, BatteryConfig
    from battery_simulator import DeviceState
    from rank_policy import create_rank_policy

    tier_cfg = DeviceTierConfig()

    # Test threshold policy
    policy = create_rank_policy(
        RankPolicyConfig(policy_type="threshold"),
        tier_cfg,
    )

    # High tier, high battery → should get rank 32
    dev = DeviceState(client_id=0, tier="high", battery_percent=95.0, is_charging=False)
    rank = policy.get_rank(dev)
    assert rank == 32, f"Expected 32, got {rank}"

    # High tier, low battery → should get rank 4
    dev.battery_percent = 25.0
    rank = policy.get_rank(dev)
    assert rank == 4, f"Expected 4, got {rank}"

    # Low tier, charging → should get max for tier (8)
    dev = DeviceState(client_id=1, tier="low", battery_percent=20.0, is_charging=True)
    rank = policy.get_rank(dev)
    assert rank == 8, f"Expected 8, got {rank}"

    # Test all policy types
    for policy_type in ["threshold", "continuous", "binary", "fixed", "static_tier", "random"]:
        p = create_rank_policy(
            RankPolicyConfig(policy_type=policy_type),
            tier_cfg,
        )
        dev = DeviceState(client_id=0, tier="mid", battery_percent=50.0, is_charging=False)
        r = p.get_rank(dev)
        assert r in [2, 4, 8, 16, 32], f"Policy {policy_type} returned invalid rank: {r}"
        print(f"   {policy_type}: rank={r} for mid-tier 50% battery - OK")

    print("   All rank policies OK!\n")
    return True


def test_flora_aggregation():
    """Test FLoRA stacking aggregation."""
    print("4. Testing FLoRA aggregation...")
    import numpy as np
    from flora_aggregation import aggregate_flora, extract_sub_adapter

    # Create a fake global state (rank 32)
    global_state = {
        "layer.lora_A.weight": np.random.randn(32, 2048).astype(np.float32),
        "layer.lora_B.weight": np.random.randn(2048, 32).astype(np.float32),
    }

    # Create client updates at different ranks
    client_updates = [
        # Client 1: rank 32
        (
            {
                "layer.lora_A.weight": np.random.randn(32, 2048).astype(np.float32),
                "layer.lora_B.weight": np.random.randn(2048, 32).astype(np.float32),
            },
            100,  # num_samples
            32,   # rank
        ),
        # Client 2: rank 8
        (
            {
                "layer.lora_A.weight": np.random.randn(8, 2048).astype(np.float32),
                "layer.lora_B.weight": np.random.randn(2048, 8).astype(np.float32),
            },
            80,
            8,
        ),
        # Client 3: rank 2
        (
            {
                "layer.lora_A.weight": np.random.randn(2, 2048).astype(np.float32),
                "layer.lora_B.weight": np.random.randn(2048, 2).astype(np.float32),
            },
            50,
            2,
        ),
    ]

    # Aggregate
    result = aggregate_flora(global_state, client_updates, max_rank=32)

    assert result["layer.lora_A.weight"].shape == (32, 2048)
    assert result["layer.lora_B.weight"].shape == (2048, 32)
    print(f"   Aggregated shapes: A={result['layer.lora_A.weight'].shape}, B={result['layer.lora_B.weight'].shape}")

    # Test sub-adapter extraction
    sub = extract_sub_adapter(result, target_rank=8)
    assert sub["layer.lora_A.weight"].shape == (8, 2048)
    assert sub["layer.lora_B.weight"].shape == (2048, 8)
    print(f"   Sub-adapter shapes: A={sub['layer.lora_A.weight'].shape}, B={sub['layer.lora_B.weight'].shape}")

    print("   FLoRA aggregation OK!\n")
    return True


def test_model_loading():
    """Test model loading (downloads the model — may take a few minutes first time)."""
    print("5. Testing model loading...")
    print("   This will download TinyLlama (~1.1GB) on first run...")

    from config import ModelConfig, LoRAConfig
    from model_utils import load_base_model, apply_lora, count_lora_parameters, print_rank_comparison

    model_cfg = ModelConfig()
    lora_cfg = LoRAConfig()

    # Print rank comparison first (no download needed)
    print_rank_comparison(lora_cfg, model_cfg)

    # Load model
    model, tokenizer = load_base_model(model_cfg)
    print(f"   Model loaded: {model_cfg.name}")
    print(f"   Tokenizer vocab size: {len(tokenizer)}")

    # Apply LoRA
    peft_model = apply_lora(model, lora_cfg, rank=8)

    # Quick generation test
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}
    import torch
    with torch.no_grad():
        out = peft_model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"   Generation test: '{text[:80]}...'")

    print("   Model loading OK!\n")
    return True


def main():
    print("=" * 50)
    print("BatteryLoRA Quick Test")
    print("=" * 50 + "\n")

    results = {}

    # Run tests in order
    results["imports"] = test_imports()
    if not results["imports"]:
        print("Fix missing imports before continuing.")
        sys.exit(1)

    results["battery"] = test_battery_simulator()
    results["policies"] = test_rank_policies()
    results["flora"] = test_flora_aggregation()

    # Model loading is optional (requires download)
    try:
        results["model"] = test_model_loading()
    except Exception as e:
        print(f"   Model loading failed: {e}")
        print("   This is OK for now — model will download when you run experiments.\n")
        results["model"] = False

    # Summary
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:.<30} {status}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed — see above.'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
