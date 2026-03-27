# Tool Drift

Scaffold for the NGEN-AI 2026 short paper on robust tool calling under interface drift.

This repository is intentionally small:

- `drift/` contains perturbation generators.
- `defense/` contains canonicalization, validation, and repair prompt logic.
- `eval/` contains metrics and error categorization.
- `inference/` contains remote model backends.
- `scripts/` contains runnable pilot entry points.
- `notebooks/` contains a Colab orchestration notebook.

## Quick Start

```bash
cd /Users/aniruddha/Documents/research/tool-drift
pip install -e .
python scripts/run_pilot_bfcl.py --config configs/pilot_bfcl.yaml --demo
python scripts/run_pilot_dice.py --config configs/pilot_dice.yaml --demo
python scripts/summarize_results.py --results-dir outputs
```

## Local OpenRouter Flow

You can run the orchestration locally and use OpenRouter only for model inference.

1. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.
2. Keep `pilot.demo_mode: true` while validating the scaffold.
3. Export small BFCL and DICE subsets to JSON.
4. Set `data.bfcl_subset_path` and `data.dice_subset_path` in the configs.
5. Set `pilot.demo_mode: false`.
6. Run:

```bash
python scripts/run_pilot_bfcl.py --config configs/pilot_bfcl.yaml
python scripts/run_pilot_dice.py --config configs/pilot_dice.yaml
python scripts/summarize_results.py --results-dir outputs
```

Notes:

- the default non-demo backend is OpenRouter,
- the configured endpoint is OpenAI-compatible chat completions,
- if your OpenRouter account uses a different model slug, update `model.name` in the YAML,
- each run now writes to its own timestamped subdirectory under the configured `output_dir`,
- the pilot score uses normalized semantic matching, not raw string equality,
- repair now tries tool-calling first and falls back to JSON-only extraction if the first repair remains invalid.

## Colab Flow

1. Open `notebooks/tool_drift_pilot_colab.ipynb`.
2. Clone or copy this folder into `/content/tool-drift`.
3. Install dependencies with `pip install -e .`.
4. Run the BFCL and DICE pilot scripts in demo mode first.
5. Export a small BFCL subset and a DICE subset to JSON.
6. Set `data.bfcl_subset_path` and `data.dice_subset_path` in the YAML configs.
7. Turn off demo mode and rerun the notebook.

## Design Notes

The scaffold is built to let you validate the research idea quickly:

- synthetic/demo mode is fully runnable without external benchmark assets,
- the real benchmark path is explicit via adapter stubs and JSON subset paths,
- the repair loop is one-shot and inference-only,
- the notebook is meant to orchestrate Colab execution, not hide the logic,
- result files now include per-example match diagnostics for `original`, `drifted`, and `repaired`.
