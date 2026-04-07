# Tool Drift Stage 2 Results

**Date:** April 5, 2026  
**Project root:** [tool-drift](/Users/aniruddha/Documents/research/tool-drift)

## Main Result

The current strongest result is the 50-example DICE stage-2 run:

- [dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260405-112603-236556/dice_results.json)

Configuration:

- [dice_stage2_50.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/dice_stage2_50.yaml)
- subset: [dice_stage2_subset_50.json](/Users/aniruddha/Documents/research/tool-drift/data/dice_stage2_subset_50.json)
- model: `qwen/qwen3.5-9b`
- provider route: `OpenRouter -> Venice`
- drift: stale-doc description drift (`legacy_example`) + schema rename + distractors
- repair: forced-tool repair with `repair_max_tokens = 512`

Summary:

- `original_score = 0.80`
- `drifted_score = 0.72`
- `repaired_score = 0.88`
- `recovery_rate = 1.0`
- `validation_failures = 10`
- `repaired_failures = 0`
- `error_breakdown = {"clean": 40, "wrong_tool": 7, "missing_field": 3}`

Originally-correct slice:

- `originally_correct_count = 40`
- `drifted_score_on_originally_correct = 0.85`
- `repaired_score_on_originally_correct = 1.0`
- `drift_misses_on_originally_correct = 6`
- `repair_recoveries_on_originally_correct = 6`
- `repair_harms_on_originally_correct = 0`

Interpretation:

- The drift now causes a real drop on a larger DICE slice.
- The repair stage fully recovers all true drift misses on the originally-correct slice.
- There are no repair harms on the originally-correct slice in this run.

## Two-Model Comparison

Second model run:

- [dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260405-202744-978292/dice_results.json)
- config: [dice_stage2_50_qwen35_35b.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/dice_stage2_50_qwen35_35b.yaml)
- model: `qwen/qwen3.5-35b-a3b`

Comparison summary:

1. `qwen/qwen3.5-9b`
   - global: `0.80 -> 0.72 -> 0.88`
   - originally-correct slice: `0.85 -> 1.0`
   - recoveries: `6/6`
   - harms: `0`

2. `qwen/qwen3.5-35b-a3b`
   - global: `0.66 -> 0.68 -> 0.86`
   - originally-correct slice: `0.7879 -> 1.0`
   - recoveries: `7/7`
   - harms: `0`

Interpretation:

- The `35b-a3b` run has a weaker baseline on this slice, so its global `original -> drifted` numbers are less clean.
- But on the fairer `originally_correct` slice, it still shows real drift harm and full recovery.
- The strongest paper table should therefore report both the global numbers and the originally-correct slice, not only global accuracy.

## Comparison To Earlier Runs

20-example DICE run after forced-tool repair:

- [dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260405-111148-163567/dice_results.json)
- `0.75 -> 0.70 -> 0.85`
- originally-correct slice: `0.9333 -> 1.0`
- recoveries: `1`
- harms: `0`

50-example DICE run before the higher repair token budget:

- [dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260405-111920-485376/dice_results.json)
- `0.82 -> 0.74 -> 0.86`
- originally-correct slice: `0.8780 -> 0.9756`
- recoveries: `4`
- harms: `0`

50-example DICE run after increasing `repair_max_tokens` to `512`:

- [dice_results.json](/Users/aniruddha/Documents/research/tool-drift/outputs/dice/dice-bench-live-20260405-112603-236556/dice_results.json)
- `0.80 -> 0.72 -> 0.88`
- originally-correct slice: `0.85 -> 1.0`
- recoveries: `6`
- harms: `0`

## What Changed To Get The Improvement

Code changes:

- [run_pilot_dice.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_dice.py)
- [repair_prompt.py](/Users/aniruddha/Documents/research/tool-drift/defense/repair_prompt.py)
- [openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py)
- [dice_stage2_50.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/dice_stage2_50.yaml)

Key changes:

- DICE repair now uses forced-tool tool-calling before JSON fallback.
- Raw repair payloads are stored in `repair_debug`.
- The repair prompt now explicitly tells the model to infer all required fields from the conversation.
- Repair token budget was raised from the default path to `512`, which fixed truncated repair outputs.
- OpenRouter timeout handling now retries `TimeoutError` and `socket.timeout`.

## Remaining Weaknesses

These examples are still baseline misses or semantic-quality misses rather than unrecovered drift misses:

- `dice_round_1_3`
- `dice_round_1_17`
- `dice_round_1_18`

These do not weaken the main repair claim as much as true unrecovered drift misses, but they do cap the overall `original_score`.

## Recommended Paper Framing

For the current writeup, the strongest claim is:

> Under stale-doc plus schema drift on a 50-example DICE slice, forced-tool repair improves tool-call accuracy from `0.72` to `0.88`, and recovers all `6/6` drift-induced failures on examples the model originally solved correctly, with `0` repair harms.

With the second model included, a broader claim is:

> Across two Qwen-family open models on the same 50-example DICE slice, forced-tool repair achieves `0` harms on the originally-correct slice and recovers all observed drift-induced failures (`6/6` for `Qwen3.5-9B`, `7/7` for `Qwen3.5-35B-A3B`).

## Suggested Next Experiment

If there is time for one more experiment, the best next move is:

- run the same 50-example DICE stage-2 config on one additional open model

That would convert the current result from a strong single-model result into a small comparative table.
