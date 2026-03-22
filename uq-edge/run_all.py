"""CLI experiment runner for UQ-Edge.

Usage:
    python run_all.py --quick                           # Smoke test
    python run_all.py --phase1                          # All inference (GPU)
    python run_all.py --phase1 --model SmolLM2-135M     # Single model
    python run_all.py --phase1 --model SmolLM2-135M --quant fp16
    python run_all.py --phase2                          # Compute metrics (CPU)
    python run_all.py --phase3                          # Self-consistency (GPU)
    python run_all.py --plots                           # Generate figures (CPU)
    python run_all.py --summary                         # Print results table
"""

import argparse
import json
import os
import time

import numpy as np


def run_phase1(cfg, model_filter=None, quant_filter=None):
    """Phase 1: Run inference on all (model, quant, benchmark) triples."""
    from config import output_path
    from model_utils import load_model, unload_model, can_quantize, get_memory_mb
    from data_utils import load_benchmark
    from inference import run_inference, run_verbalized_confidence, save_raw_output

    total = len(cfg.models) * len(cfg.quant_levels) * len(cfg.benchmarks)
    done = 0
    skipped = 0
    t_start = time.time()

    for model_spec in cfg.models:
        if model_filter and model_spec.short_name != model_filter:
            continue

        for quant_spec in cfg.quant_levels:
            if quant_filter and str(quant_spec) != quant_filter:
                continue

            if not can_quantize(quant_spec):
                print(f"SKIP {model_spec} @ {quant_spec} — requires CUDA")
                skipped += len(cfg.benchmarks)
                continue

            # Load model once for all benchmarks at this quant level
            print(f"\n{'='*60}")
            print(f"MODEL: {model_spec} @ {quant_spec}")
            print(f"{'='*60}")

            try:
                model, tokenizer = load_model(model_spec, quant_spec)
                mem = get_memory_mb(model)
                print(f"  Memory: {mem:.0f} MB")
            except Exception as e:
                print(f"  FAILED to load: {e}")
                skipped += len(cfg.benchmarks)
                continue

            for bench_spec in cfg.benchmarks:
                out_path = output_path(cfg, model_spec, quant_spec, bench_spec)

                if os.path.exists(out_path):
                    print(f"  SKIP {bench_spec} — already exists: {out_path}")
                    skipped += 1
                    done += 1
                    continue

                print(f"\n  Benchmark: {bench_spec}")
                t0 = time.time()

                try:
                    samples = load_benchmark(bench_spec, seed=cfg.inference.seed)

                    # Main inference
                    data = run_inference(
                        model, tokenizer, samples, bench_spec.name,
                        max_new_tokens=cfg.inference.max_new_tokens,
                    )

                    # Verbalized confidence
                    try:
                        verbal_conf = run_verbalized_confidence(
                            model, tokenizer, samples, list(data["answers"]),
                        )
                        data["verbalized_conf"] = verbal_conf
                        parse_rate = np.isfinite(verbal_conf).mean()
                        print(f"  Verbal confidence parse rate: {parse_rate:.0%}")
                    except Exception as e:
                        print(f"  Verbal confidence failed: {e}")

                    # Compute accuracy
                    acc = data["is_correct"].mean()
                    elapsed = time.time() - t0
                    print(f"  Accuracy: {acc:.1%} | Time: {elapsed:.0f}s")

                    # Save
                    metadata = {
                        "model": model_spec.short_name,
                        "model_hf_id": model_spec.hf_id,
                        "quant": str(quant_spec),
                        "benchmark": bench_spec.name,
                        "n_samples": len(samples),
                        "accuracy": float(acc),
                        "memory_mb": mem,
                    }
                    save_raw_output(data, out_path, metadata)

                except Exception as e:
                    print(f"  ERROR on {bench_spec}: {e}")
                    import traceback
                    traceback.print_exc()

                done += 1
                elapsed_total = time.time() - t_start
                remaining = total - done - skipped
                if done > skipped:
                    rate = elapsed_total / (done - skipped + 1)
                    print(f"  Progress: {done}/{total} | ~{remaining * rate / 60:.0f} min remaining")

            # Unload model before loading next
            unload_model(model)
            del model, tokenizer

    print(f"\nPhase 1 complete. {done} done, {skipped} skipped.")


def run_phase2(cfg):
    """Phase 2: Compute UQ scores and metrics from saved raw outputs. CPU only."""
    from config import output_path, metrics_path
    from inference import load_raw_output
    from uq_methods import compute_all_scores
    from metrics import compute_all_metrics

    os.makedirs(cfg.results_dir, exist_ok=True)
    computed = 0

    for model_spec in cfg.models:
        for quant_spec in cfg.quant_levels:
            for bench_spec in cfg.benchmarks:
                raw_path = output_path(cfg, model_spec, quant_spec, bench_spec)
                met_path = metrics_path(cfg, model_spec, quant_spec, bench_spec)

                if not os.path.exists(raw_path):
                    continue

                print(f"  {model_spec} @ {quant_spec} — {bench_spec}")

                data = load_raw_output(raw_path)
                scores = compute_all_scores(data)
                all_metrics = compute_all_metrics(scores, data["is_correct"])

                # Add accuracy to metrics
                all_metrics["_accuracy"] = float(data["is_correct"].mean())
                all_metrics["_n_samples"] = int(len(data["is_correct"]))

                with open(met_path, "w") as f:
                    json.dump(all_metrics, f, indent=2, default=_json_default)
                computed += 1

    print(f"Phase 2 complete. Computed metrics for {computed} configurations.")


def run_phase3(cfg):
    """Phase 3: Self-consistency sampling for selected configs. GPU required."""
    from config import output_path
    from model_utils import load_model, unload_model, can_quantize
    from data_utils import load_benchmark
    from inference import run_self_consistency, load_raw_output, save_raw_output

    for model_spec in cfg.models:
        for quant_spec in cfg.quant_levels:
            if not can_quantize(quant_spec):
                continue

            model, tokenizer = load_model(model_spec, quant_spec)

            for bench_spec in cfg.benchmarks:
                raw_path = output_path(cfg, model_spec, quant_spec, bench_spec)
                if not os.path.exists(raw_path):
                    continue

                # Check if SC already done
                data = load_raw_output(raw_path)
                if "self_consistency" in data:
                    print(f"  SKIP SC for {model_spec} @ {quant_spec} — {bench_spec}")
                    continue

                print(f"  Self-consistency: {model_spec} @ {quant_spec} — {bench_spec}")
                samples = load_benchmark(bench_spec, seed=cfg.inference.seed)

                sc_scores = run_self_consistency(
                    model, tokenizer, samples, bench_spec.name,
                    k=cfg.inference.self_consistency_k,
                    temperature=cfg.inference.temperature_sampling,
                )

                # Re-save with SC data added
                data["self_consistency"] = sc_scores
                metadata = data.get("metadata", {})
                save_raw_output(dict(data), raw_path, metadata)

            unload_model(model)

    print("Phase 3 complete.")


def run_summary(cfg):
    """Print a summary table of all results."""
    from plotting import load_all_metrics

    all_data = load_all_metrics(cfg.results_dir)
    if not all_data:
        print("No results found.")
        return

    print(f"\n{'Model':<20} {'Quant':<8} {'Acc':>6} {'ECE(MSP)':>10} {'ECE(Ent)':>10} "
          f"{'ECE(TS)':>10} {'AUROC':>8}")
    print("-" * 80)

    for model in sorted(all_data):
        for quant in sorted(all_data[model]):
            accs, eces_msp, eces_ent, eces_ts, aurocs = [], [], [], [], []
            for bench_data in all_data[model][quant].values():
                if "_accuracy" in bench_data:
                    accs.append(bench_data["_accuracy"])
                for method, key, lst in [
                    ("msp", "ece", eces_msp),
                    ("entropy", "ece", eces_ent),
                    ("temp_scaled_msp", "ece", eces_ts),
                    ("msp", "auroc", aurocs),
                ]:
                    try:
                        v = bench_data[method][key]
                        if v is not None and not np.isnan(v):
                            lst.append(v)
                    except (KeyError, TypeError):
                        pass

            acc = np.mean(accs) if accs else float("nan")
            ece_msp = np.mean(eces_msp) if eces_msp else float("nan")
            ece_ent = np.mean(eces_ent) if eces_ent else float("nan")
            ece_ts = np.mean(eces_ts) if eces_ts else float("nan")
            auroc = np.mean(aurocs) if aurocs else float("nan")

            print(f"{model:<20} {quant:<8} {acc:>5.1%} {ece_msp:>10.4f} {ece_ent:>10.4f} "
                  f"{ece_ts:>10.4f} {auroc:>8.4f}")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="UQ-Edge experiment runner")
    parser.add_argument("--quick", action="store_true", help="Smoke test (10 samples)")
    parser.add_argument("--local", action="store_true", help="Local dev config (50 samples, FP16 only)")
    parser.add_argument("--phase1", action="store_true", help="Run inference (GPU)")
    parser.add_argument("--phase2", action="store_true", help="Compute metrics (CPU)")
    parser.add_argument("--phase3", action="store_true", help="Self-consistency (GPU)")
    parser.add_argument("--plots", action="store_true", help="Generate figures")
    parser.add_argument("--summary", action="store_true", help="Print results table")
    parser.add_argument("--model", type=str, default=None, help="Filter to single model")
    parser.add_argument("--quant", type=str, default=None, help="Filter to single quant method")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    from config import get_quick_test_config, get_local_dev_config, get_full_config

    if args.quick:
        cfg = get_quick_test_config()
    elif args.local:
        cfg = get_local_dev_config()
    else:
        cfg = get_full_config()

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)

    run_all = args.quick or args.local

    if run_all or args.phase1:
        run_phase1(cfg, model_filter=args.model, quant_filter=args.quant)

    if run_all or args.phase2:
        run_phase2(cfg)

    if args.phase3:
        run_phase3(cfg)

    if run_all or args.plots:
        from plotting import generate_all_plots
        generate_all_plots(cfg.results_dir, cfg.plots_dir)

    if run_all or args.summary:
        run_summary(cfg)


if __name__ == "__main__":
    main()
