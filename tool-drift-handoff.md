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

Current behavior:

- each run writes to its own timestamped subdirectory,
- summaries include `run_id`, `demo_mode`, and `scoring_policy`,
- per-example records now include `original_match`, `drifted_match`, and `repaired_match`.

Note:

- old legacy flat files like `outputs/bfcl/bfcl_results.json` still exist from the earlier scaffold,
- new runs do **not** overwrite each other anymore,
- but mixed old/new files under `outputs/` can still make summaries look noisy.

## 4.2 OpenRouter backend works at the HTTP/request level

[openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py) is implemented and importable.

It uses:

- `.env` loading from [scripts/common.py](/Users/aniruddha/Documents/research/tool-drift/scripts/common.py)
- OpenRouter OpenAI-compatible chat completions
- tool calling via `tools`
- JSON extraction fallback for repair mode

Verified behavior:

- requests return successfully,
- model responds with usable tool-call structures in smoke cases,
- the live local orchestration path works from the local machine when network access is allowed.

## 4.3 BFCL live smoke still fails under drift and repair

Using [data/bfcl_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/bfcl_smoke.json), the latest live run produced:

- `original_score = 1.0`
- `drifted_score = 0.0`
- `repaired_score = 0.0`
- `validation_failures = 1`
- `repaired_failures = 1`
- `error_breakdown = {"wrong_tool": 1}`

This proves:

- orchestration can run locally,
- OpenRouter can be called,
- the benchmark runner can pass through `original -> drifted -> repaired`,
- BFCL remains the stronger failure case and is still the right place to improve repair.

Latest result file:

- [outputs/bfcl/bfcl-live-20260327-062422-930874/bfcl_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/bfcl/bfcl-live-20260327-062422-930874/bfcl_results.json)

## 4.4 DICE live smoke now validates the normalized scorer

Using [data/dice_smoke.json](/Users/aniruddha/Documents/research/tool-drift/data/dice_smoke.json), the latest live run produced:

- `original_score = 1.0`
- `drifted_score = 1.0`
- `repaired_score = 1.0`
- `validation_failures = 0`
- `repaired_failures = 0`
- `error_breakdown = {"clean": 1}`

Interpretation:

- the earlier DICE false negative caused by brittle exact matching is fixed,
- the current DICE smoke example is now too easy and mostly confirms the scoring path,
- DICE still needs a better subset before it becomes a useful pilot benchmark.

Latest result file:

- [outputs/dice/dice-bench-live-20260327-062613-009920/dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260327-062613-009920/dice_results.json)

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

## 5.7 Added run isolation and richer result metadata

Implemented in:

- [scripts/common.py](/Users/aniruddha/Documents/research/tool-drift/scripts/common.py)
- [scripts/run_pilot_bfcl.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_bfcl.py)
- [scripts/run_pilot_dice.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_dice.py)
- [scripts/summarize_results.py](/Users/aniruddha/Documents/research/tool-drift/scripts/summarize_results.py)

Added:

- timestamped per-run output directories,
- `run_id` and `output_dir` in summaries,
- `config` snapshots in result payloads,
- per-example `repair_strategy`.

## 5.8 Replaced brittle exact match with normalized semantic matching

Implemented in:

- [eval/metrics.py](/Users/aniruddha/Documents/research/tool-drift/eval/metrics.py)

Added:

- case-insensitive string normalization,
- whitespace normalization,
- simple time normalization,
- simple date normalization,
- integer/number coercion for semantic comparison,
- unordered normalization for attendee-like lists,
- per-example field mismatch diagnostics.

The current score label is:

- `normalized_semantic_match_v1`

## 5.9 Added JSON repair fallback and improved repair prompt

Implemented in:

- [inference/openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py)
- [defense/repair_prompt.py](/Users/aniruddha/Documents/research/tool-drift/defense/repair_prompt.py)
- [scripts/run_pilot_bfcl.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_bfcl.py)
- [scripts/run_pilot_dice.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_dice.py)

Behavior:

- repair first tries tool-calling,
- if that remains invalid, repair falls back to a JSON-only response path,
- repair prompt now includes an explicit output schema and renamed-field reminder.

## 5.10 Fixed clean-case error labeling

Updated:

- [eval/error_taxonomy.py](/Users/aniruddha/Documents/research/tool-drift/eval/error_taxonomy.py)

`error_breakdown` now uses `clean` instead of `unknown` when validation has no issues.

---

## 6. Known Issues

These are the important ones. The next agent should start here instead of broad refactoring.

## 6.1 Legacy flat output files still exist and can confuse summaries

Status:

- fixed for new runs,
- not fixed for old files already written at:
  - `outputs/bfcl/bfcl_results.json`
  - `outputs/dice/dice_results.json`

Fix needed:

- optionally archive or delete legacy flat outputs,
- or teach the summarizer to ignore results without `run_id`.

## 6.2 Real subset export is still not done

The adapters still expect pre-exported JSON subsets.

Current expected schema:

- `id`
- `prompt`
- `tool`
- `gold_call`
- optional `candidate_tools`

The next agent should either:

1. build exporter scripts from BFCL and DICE repos, or
2. document a strict manual export path and commit small real subsets.

## 6.3 BFCL repair is still not effective

Observed in the BFCL smoke case:

- original call was valid,
- drifted call failed,
- repair still failed even after adding JSON fallback.

Likely reasons:

- the BFCL failure starts from an empty or missing tool call,
- the repair prompt still may not provide enough structure for hard rename cases,
- raw repair responses are not yet stored, so detailed diagnosis is still manual.

Fix needed:

- persist raw repair responses or raw provider payloads for failed cases,
- test a stricter repair format with field-by-field extraction,
- consider a deterministic canonicalizer/field-mapper before calling repair.

## 6.4 DICE smoke case is now too easy

The current DICE smoke example no longer stress-tests the system because the model succeeds under the current drift setting.

Fix needed:

- use multiple DICE examples,
- ensure at least some cases actually break under rename/reorder/distractor drift,
- keep normalized scoring but avoid “plumbing only” tasks.

---

## 7. Current Result Snapshot

From the latest non-demo smoke outputs:

### BFCL smoke

Summary currently written in:

- [outputs/bfcl/bfcl-live-20260327-062422-930874/bfcl_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/bfcl/bfcl-live-20260327-062422-930874/bfcl_results.json)

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
- normalized scoring is not hiding the failure,
- repair path is still not good enough for BFCL.

### DICE smoke

Summary currently written in:

- [outputs/dice/dice-bench-live-20260327-062613-009920/dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260327-062613-009920/dice_results.json)

Observed:

- `original_score = 1.0`
- `drifted_score = 1.0`
- `repaired_score = 1.0`
- `validation_failures = 0`
- `repaired_failures = 0`
- `error_breakdown = {"clean": 1}`

Interpretation:

- normalized scoring fixed the earlier false negative,
- this smoke example is now mostly a scorer/inference sanity check,
- the benchmark slice still needs harder examples.

---

## 8. Recommended Next Actions for the Next Agent

The next agent should do these in order:

### Step 1: Implement real subset exporters

Highest-value work item.

Build:

- a small BFCL subset exporter,
- a small DICE subset exporter,

that produce the JSON format expected by:

- [benchmarks/bfcl_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/bfcl_adapter.py)
- [benchmarks/dice_adapter.py](/Users/aniruddha/Documents/research/tool-drift/benchmarks/dice_adapter.py)

### Step 2: Strengthen BFCL repair diagnosis

Investigate why the BFCL repair call still fails after JSON fallback.

Likely path:

- log raw repaired response,
- store the raw provider message or parsed JSON fallback attempt,
- compare tool-calling vs JSON-only repair on the same failure,
- maybe add a stricter field-by-field repair prompt.

### Step 3: Replace smoke-only DICE with a harder subset

- keep the normalized scorer,
- pick 10-20 examples where rename/reorder/distractor drift actually matters,
- avoid examples the model solves unchanged under drift.

### Step 4: Run a true pilot slice

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
- Run isolation and normalized scoring are in place.
- The codebase is still at the "pilot infrastructure plus smoke validation" stage, not the "benchmark-ready experimentation" stage.
- The next agent should focus on **real subset wiring and BFCL repair diagnosis**, not broad architecture changes.
