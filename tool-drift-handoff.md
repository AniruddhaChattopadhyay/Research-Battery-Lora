# Tool Drift Handoff

**Date:** March 27, 2026  
**Project root:** [tool-drift](/Users/aniruddha/Documents/research/tool-drift)  
**Related planning docs:**

- [NGEN-AI-2026-research-discovery.md](/Users/aniruddha/Documents/research/NGEN-AI-2026-research-discovery.md)
- [NGEN-AI-2026-tool-drift-research-plan.md](/Users/aniruddha/Documents/research/NGEN-AI-2026-tool-drift-research-plan.md)

This document is the handoff context for the next agent. It explains:

- what was already built,
- what was verified,
- what is still incomplete or wrong,
- what the next agent should do first.

---

## 1. Objective

The target paper direction is:

> **Robust tool calling under interface drift**

The scaffold is for a pilot project called **SchemaShield-Lite**:

- perturb tool interfaces,
- run a model,
- validate tool calls,
- do one-shot repair,
- measure `original -> drifted -> repaired`.

The current code is a **pilot scaffold**, not a final benchmark system.

---

## 2. Current Code Layout

Project root:

- [tool-drift](/Users/aniruddha/Documents/research/tool-drift)

Important files:

- [README.md](/Users/aniruddha/Documents/research/tool-drift/README.md)
- [pyproject.toml](/Users/aniruddha/Documents/research/tool-drift/pyproject.toml)
- [configs/pilot_bfcl.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/pilot_bfcl.yaml)
- [configs/pilot_dice.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/pilot_dice.yaml)
- [scripts/run_pilot_bfcl.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_bfcl.py)
- [scripts/run_pilot_dice.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_dice.py)
- [scripts/summarize_results.py](/Users/aniruddha/Documents/research/tool-drift/scripts/summarize_results.py)
- [scripts/common.py](/Users/aniruddha/Documents/research/tool-drift/scripts/common.py)
- [inference/openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py)
- [defense/validator.py](/Users/aniruddha/Documents/research/tool-drift/defense/validator.py)
- [defense/canonicalizer.py](/Users/aniruddha/Documents/research/tool-drift/defense/canonicalizer.py)
- [defense/repair_prompt.py](/Users/aniruddha/Documents/research/tool-drift/defense/repair_prompt.py)
- [drift/description_drift.py](/Users/aniruddha/Documents/research/tool-drift/drift/description_drift.py)
- [drift/schema_drift.py](/Users/aniruddha/Documents/research/tool-drift/drift/schema_drift.py)
- [drift/candidate_drift.py](/Users/aniruddha/Documents/research/tool-drift/drift/candidate_drift.py)
- [benchmarks/bfcl_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/bfcl_adapter.py)
- [benchmarks/dice_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/dice_adapter.py)
- [notebooks/tool_drift_pilot_colab.ipynb](/Users/aniruddha/Documents/research/tool-drift/notebooks/tool_drift_pilot_colab.ipynb)

Smoke test data:

- [data/bfcl_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/bfcl_smoke.json)
- [data/dice_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/dice_smoke.json)

---

## 3. Environment State

### Virtual environment

A local `uv` environment was created successfully:

- [tool-drift/.venv](/Users/aniruddha/Documents/research/tool-drift/.venv)

Setup used:

```bash
cd /Users/aniruddha/Documents/research/tool-drift
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

### Packaging issue that was fixed

`uv pip install -e .` originally failed because setuptools auto-discovered multiple top-level folders in a flat layout.  
This was fixed by constraining package discovery in [pyproject.toml](/Users/aniruddha/Documents/research/tool-drift/pyproject.toml).

### Env files

- local `.env` exists under `tool-drift`
- `.env` and `*.env` were added to [.gitignore](/Users/aniruddha/Documents/research/.gitignore)

Important note:

The OpenRouter key was pasted in chat and written into the local `.env`. It should be rotated after testing.

---

## 4. What Was Verified

## 4.1 Demo path works

The synthetic/demo mode runs successfully through the `uv` interpreter:

```bash
cd /Users/aniruddha/Documents/research/tool-drift
./.venv/bin/python scripts/run_pilot_bfcl.py --config configs/pilot_bfcl.yaml --demo
./.venv/bin/python scripts/run_pilot_dice.py --config configs/pilot_dice.yaml --demo
./.venv/bin/python scripts/summarize_results.py --results-dir outputs
```

The intended synthetic pattern was previously observed:

- drift lowers score,
- repair restores score.

However, note that the current `outputs/` directory may contain overwritten results from later non-demo smoke runs. Do not trust `outputs/` as the historical source of truth without rerunning.

## 4.2 OpenRouter backend works at the HTTP/request level

[openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py) is implemented and importable.

It uses:

- `.env` loading from [scripts/common.py](/Users/aniruddha/Documents/research/tool-drift/scripts/common.py)
- OpenRouter OpenAI-compatible chat completions
- tool calling via `tools`

Verified behavior:

- requests return successfully,
- model responds with usable tool-call structures in at least some smoke cases.

## 4.3 BFCL non-demo smoke run proved the local remote-inference loop works

Using [data/bfcl_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/bfcl_smoke.json), the non-demo runner completed successfully and produced:

- `original_score = 1.0`
- `drifted_score = 0.0`
- `repaired_score = 0.0`

This is a valid proof that:

- orchestration can run locally,
- OpenRouter can be called,
- the benchmark runner can pass through `original -> drifted -> repaired`.

That BFCL smoke case currently fails to recover. That is useful signal, not an infrastructure bug.

## 4.4 DICE non-demo smoke run is not yet a reliable research result

The DICE smoke path runs, but the current setup is not yet a reliable pilot measure because:

- exact-match scoring is too brittle,
- prompt/answer normalization is weak,
- the single example is still acting more like a plumbing test than a benchmark slice.

So: DICE local remote execution is partially tested, but not yet research-ready.

---

## 5. What Was Changed During This Session

## 5.1 Added OpenRouter local backend

Implemented in:

- [inference/openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py)

## 5.2 Added env loading helpers

Implemented in:

- [scripts/common.py](/Users/aniruddha/Documents/research/tool-drift/scripts/common.py)

Functions added:

- `load_dotenv`
- `require_env`

## 5.3 Switched default configs to OpenRouter

Updated:

- [configs/pilot_bfcl.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/pilot_bfcl.yaml)
- [configs/pilot_dice.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/pilot_dice.yaml)

Current default model config in YAML:

- provider: `openrouter`
- endpoint: `https://openrouter.ai/api/v1/chat/completions`
- model: `qwen/qwen3.5-9b`

## 5.4 Added benchmark adapter stubs

Files:

- [benchmarks/bfcl_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/bfcl_adapter.py)
- [benchmarks/dice_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/dice_adapter.py)

These currently expect pre-exported JSON subset files.

## 5.5 Added smoke-test datasets

Files:

- [data/bfcl_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/bfcl_smoke.json)
- [data/dice_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/dice_smoke.json)

These are only for small end-to-end local tests.

## 5.6 Fixed gold-call adaptation under schema drift

Added:

- `adapt_gold_call_to_tool` in [scripts/common.py](/Users/aniruddha/Documents/research/tool-drift/scripts/common.py)

Also updated:

- [drift/schema_drift.py](/Users/aniruddha/Documents/research/tool-drift/drift/schema_drift.py)

Reason:

The original scaffold used fake placeholder gold labels like `origin_value`, which made real non-demo scoring meaningless. The scoring path now attempts to preserve real gold values across renamed fields.

---

## 6. Known Issues

These are the important ones. The next agent should start here instead of broad refactoring.

## 6.1 Output directories are reused by demo and non-demo runs

Current issue:

- demo and non-demo runs write to the same `outputs/bfcl` and `outputs/dice` directories
- this causes summaries to be overwritten and become misleading

Fix needed:

- separate output paths by mode, timestamp, or run id
- or pass an output override through config/CLI

## 6.2 Exact-match scoring is too brittle for real runs

Current issue:

- string-level exact match is too harsh for realistic remote-model behavior
- examples:
  - `3 PM` vs `3:00 PM`
  - title capitalization
  - harmless formatting differences

Fix needed:

- add normalized argument comparison
- possibly type-aware comparison
- possibly executable or schema-aware matching for some fields

This is likely the most important methodological next step after real subset wiring.

## 6.3 Repair is not effective yet on the BFCL smoke failure

Observed in the BFCL smoke case:

- original call was valid,
- drifted call failed,
- repair also failed.

Likely reasons:

- repair prompt is too weak,
- OpenRouter/model may not reliably emit tool calls on the repair prompt as currently framed,
- wrong-tool/missing-field failures need stronger constrained repair logic.

Fix needed:

- inspect the returned repaired payload carefully,
- decide whether repair should use:
  - tool calling again,
  - strict JSON extraction mode,
  - or a hybrid.

## 6.4 DICE smoke case is still just a plumbing check

The DICE smoke test is not yet a meaningful research signal.

Fix needed:

- use a better small DICE subset with multiple examples,
- pick examples where drift materially affects required arguments,
- then re-evaluate with normalized scoring.

## 6.5 Real BFCL/DICE subset export is not done

The adapters exist, but the real subset-export path is not implemented.

Current assumption:

- subsets will be provided as JSON in the format:
  - `id`
  - `prompt`
  - `tool`
  - `gold_call`

The next agent needs to either:

1. build exporter scripts from BFCL and DICE repos, or
2. document a manual export path and standardize the subset schema.

---

## 7. Current Result Snapshot

From the latest non-demo smoke outputs:

### BFCL smoke

Summary currently written in:

- [outputs/bfcl/bfcl_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/bfcl/bfcl_results.json)

Observed:

- `original_score = 1.0`
- `drifted_score = 0.0`
- `repaired_score = 0.0`
- `validation_failures = 1`
- `repaired_failures = 1`
- `error_breakdown = {"wrong_tool": 1}`

Interpretation:

- endpoint path works,
- drift can break the tool call,
- repair path is currently not good enough.

### DICE smoke

Summary currently written in:

- [outputs/dice/dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice_results.json)

Observed:

- `original_score = 0.0`
- `drifted_score = 0.0`
- `repaired_score = 0.0`

Interpretation:

- this is not a reliable evaluation result yet,
- mostly a local remote-execution plumbing test,
- scoring/normalization/examples need work.

---

## 8. Recommended Next Actions for the Next Agent

The next agent should do these in order:

### Step 1: Fix run isolation

Create per-run output directories so demo and non-demo results do not overwrite each other.

### Step 2: Improve evaluation

Implement a normalized scorer for tool-call arguments.

Minimum scope:

- case-insensitive string normalization,
- whitespace normalization,
- simple time normalization,
- list normalization for unordered attendee lists if appropriate.

### Step 3: Implement real subset exporters

Highest-value work item.

Build:

- a small BFCL subset exporter,
- a small DICE subset exporter,

that produce the JSON format expected by:

- [benchmarks/bfcl_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/bfcl_adapter.py)
- [benchmarks/dice_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/dice_adapter.py)

### Step 4: Strengthen repair

Investigate why the BFCL repair call failed.

Likely path:

- log raw repaired response,
- compare tool-calling vs JSON-mode repair,
- maybe add a stricter repair system prompt.

### Step 5: Run a true pilot slice

After exporters and scorer are fixed:

- run 10-20 BFCL examples,
- run 10-20 DICE examples,
- produce real `original -> drifted -> repaired` tables.

Only after that should the team decide whether to scale further.

---

## 9. Suggested Commands

### Recreate/install env

```bash
cd /Users/aniruddha/Documents/research/tool-drift
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

### Demo mode

```bash
cd /Users/aniruddha/Documents/research/tool-drift
./.venv/bin/python scripts/run_pilot_bfcl.py --config configs/pilot_bfcl.yaml --demo
./.venv/bin/python scripts/run_pilot_dice.py --config configs/pilot_dice.yaml --demo
./.venv/bin/python scripts/summarize_results.py --results-dir outputs
```

### BFCL smoke non-demo

```bash
cd /Users/aniruddha/Documents/research/tool-drift
./.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.common import load_yaml
from scripts.run_pilot_bfcl import run

cfg = load_yaml('configs/pilot_bfcl.yaml')
cfg['pilot']['demo_mode'] = False
cfg['data']['bfcl_subset_path'] = str(Path('data/bfcl_smoke.json').resolve())
res = run(cfg, demo=False)
print(res['summary'])
PY
```

### DICE smoke non-demo

```bash
cd /Users/aniruddha/Documents/research/tool-drift
./.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.common import load_yaml
from scripts.run_pilot_dice import run

cfg = load_yaml('configs/pilot_dice.yaml')
cfg['pilot']['demo_mode'] = False
cfg['data']['dice_subset_path'] = str(Path('data/dice_smoke.json').resolve())
res = run(cfg, demo=False)
print(res['summary'])
PY
```

---

## 10. Final Notes

- Colab is no longer required for orchestration.
- OpenRouter-backed local execution is working.
- The codebase is at the "first real pilot plumbing" stage, not the "benchmark-ready experimentation" stage.
- The next agent should focus on **evaluation correctness and real subset wiring**, not broad architecture changes.
