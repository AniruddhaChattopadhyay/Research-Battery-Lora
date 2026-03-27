# Research Plan: Robust Tool Calling Under Interface Drift

**Project codename:** SchemaShield-Lite  
**Conference target:** NGEN-AI 2026  
**Submission type:** Short paper (8 pages)  
**Verified deadline:** May 25, 2026  
**Plan date:** March 27, 2026  
**Companion doc:** [NGEN-AI-2026-research-discovery.md](/Users/aniruddha/Documents/research/NGEN-AI-2026-research-discovery.md)

---

## 1. Executive Summary

### Working thesis

Open tool-calling LLMs are brittle under **post-deployment interface drift** even when the underlying tool semantics remain unchanged. This brittleness comes from changes in:

- tool descriptions,
- parameter names and enum values,
- candidate tool set composition,
- error message format and execution feedback.

We will build a short paper around the claim that:

> **A lightweight inference-time layer combining canonical schema normalization, alias-aware validation, and one-shot repair can recover a meaningful portion of tool-calling performance lost under realistic interface drift, without retraining the base model.**

### Paper shape

This should be positioned as a **robustness + methods** paper, not just a benchmark paper:

1. define a realistic interface-drift taxonomy,
2. create perturbation layers on top of existing tool benchmarks,
3. show that recent open models degrade under those perturbations,
4. propose a lightweight repair pipeline,
5. run ablations and error analysis.

### Why this topic is the right bet

- strong 2025-2026 literature exists,
- the gap is real and current,
- pilot experiments can be run quickly,
- no large-scale fine-tuning is required,
- the contribution is credible for an 8-page short paper.

---

## 2. Scope and Boundaries

### In scope

- inference-time robustness under benign but realistic interface drift,
- open models only,
- benchmarked tool-calling and multi-turn tool use,
- lightweight defenses that are practical in deployment,
- exact-call, execution-success, and recovery-style evaluation.

### Out of scope

- training a new large tool-calling model from scratch,
- RL-heavy post-training such as GRPO/Fission-GRPO,
- large agent frameworks with many moving parts,
- safety/red-team claims beyond interface robustness,
- multilingual tool use.

### Practical boundary

If the project starts to look like "build a new agent platform", scope has drifted too far. The paper must stay centered on:

- drift,
- failure modes,
- repair,
- measurable recovery.

---

## 3. Main Research Questions

### RQ1

How much do recent open tool-calling models degrade when tool interfaces drift but semantics do not change?

### RQ2

Which drift types are most damaging?

- description drift,
- schema drift,
- candidate-set drift,
- feedback drift.

### RQ3

Can a lightweight, training-free mitigation recover reliability?

### RQ4

Does the mitigation help mostly with:

- tool selection,
- argument generation,
- execution recovery,
- or all three?

### RQ5

What is the accuracy-latency tradeoff of the mitigation?

---

## 4. Hypotheses

### H1: Drift hurts a lot

Semantics-preserving interface drift will cause large drops in exact-call or execution-success metrics on open models.

### H2: Not all drift is equal

Description drift and candidate-set drift will mostly hurt **tool selection**, while schema drift will mostly hurt **argument correctness** and **executability**.

### H3: Lightweight repair is enough to matter

A simple inference-time stack can recover a substantial fraction of lost performance without model retraining.

### H4: Smaller models suffer more but also benefit more

The relative gain from repair will likely be larger on smaller or less tool-specialized open models.

---

## 5. Literature Positioning and Novelty

## 5.1 What recent work already established

Recent papers already support the importance of the problem:

- **Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use** (2026) argues that tool interfaces are a bottleneck and that tool descriptions and schemas matter, especially for large tool sets.
- **ASA: Training-Free Representation Engineering for Tool-Calling Agents** (2026) explicitly frames tool calling as brittle under evolving interfaces and schema shift.
- **Robust Tool Use via Fission-GRPO** (2026) shows that error recovery is a major weakness in multi-turn tool use, especially for smaller models.
- **Gaming Tool Preferences in Agentic LLMs** (2025) shows that changing tool descriptions alone can alter usage by more than 10x.
- **DICE-BENCH** (2025) shows that realistic multi-round, multi-party tool use remains difficult for current models.
- **ComplexFuncBench** (2025) shows that complex function calling with constraints, reasoning, and long context is still hard even for strong models.
- **Hammer** (2024) shows that function and parameter naming conventions can mislead models and destabilize cross-benchmark performance.

## 5.2 The gap we will target

The recent literature is still fragmented:

- some papers study **description optimization**,
- some study **schema shift**,
- some study **error recovery**,
- some study **benchmark realism**,
- some study **naming sensitivity**.

What is still missing is a practical paper centered on:

> **post-deployment interface drift as a unified robustness problem, with a compact mitigation that can be layered on top of existing models at inference time.**

## 5.3 What makes our contribution distinct

We are not trying to beat all tool-learning methods. We are making a narrower and more defensible claim:

1. existing open models are brittle under realistic interface drift,
2. this brittleness can be measured systematically across drift types,
3. a small inference-time pipeline can recover a meaningful share of performance.

That positioning is different from:

- training better models,
- rewriting descriptions with large teacher models,
- RL-based recovery training,
- purely adversarial security work.

---

## 6. Proposed Method: SchemaShield-Lite

### 6.1 Method summary

SchemaShield-Lite is a training-free inference-time layer with three parts:

1. **Canonical Tool Card**
2. **Alias-Aware Validator**
3. **One-Shot Repair**

The goal is to reduce dependence on fragile surface forms while keeping the method simple enough to implement and explain.

### 6.2 Component A: Canonical Tool Card

For each tool, transform the raw interface into a compact, standardized representation:

- action-oriented tool summary,
- canonical function name,
- required fields first,
- optional fields after,
- field descriptions normalized into short imperative text,
- enum values listed explicitly,
- aliases table when available or derived.

This is deterministic preprocessing, not model training.

Purpose:

- reduce verbosity noise,
- reduce stylistic variation,
- emphasize semantics over naming quirks,
- make prompt format consistent across tools.

### 6.3 Component B: Alias-Aware Validator

After the model outputs a tool call:

- validate function name,
- validate argument names,
- validate required fields,
- validate types,
- validate enum values.

If validation fails, generate structured feedback such as:

- unknown argument `departDate`; closest valid field: `departure_date`
- missing required field `city`
- enum value `firstclass`; allowed values: `economy`, `business`, `first_class`

Alias matching will start lightweight:

- normalization rules,
- string similarity,
- description overlap,
- optional small-embedding similarity if needed later.

### 6.4 Component C: One-Shot Repair

If the initial call is invalid, run one repair step with a constrained prompt that includes:

- original task context,
- candidate tool call,
- canonical tool card,
- validator feedback,
- instruction to return only corrected structured output.

Important:

- one repair turn only in the main method,
- no long reasoning chains,
- no multiple self-reflection rounds during the pilot.

### 6.5 Minimal version for the pilot

The pilot version should be intentionally narrow:

- canonicalized tool card,
- validator with name/type/required-field checks,
- one-shot repair prompt.

Do **not** start with:

- learned rerankers,
- agent memory,
- multiple repair loops,
- tool retrieval models,
- fine-tuning.

### 6.6 Ablation systems

We should compare:

- `B0` Original benchmark setting
- `B1` Drifted interface, no defense
- `B2` Drifted interface + naive retry on failure
- `B3` Drifted interface + canonical tool card only
- `B4` Drifted interface + validator repair only
- `B5` Drifted interface + full SchemaShield-Lite

If time allows, add:

- `B6` Drifted interface + LLM-generated concise rewrite baseline

But `B6` is optional and should not block the main project.

---

## 7. Interface Drift Taxonomy

This taxonomy is central to the paper. It should be explicit and reproducible.

### D1. Description Drift

The schema is the same, but the natural-language tool description changes.

Examples:

- verbose marketing text,
- overly casual or overly formal tone,
- added examples,
- endorsement lines,
- misleading emphasis,
- company/product name injection.

Motivation from literature:

- `Gaming Tool Preferences in Agentic LLMs` shows description edits can strongly bias usage.
- `Learning to Rewrite Tool Descriptions...` shows interface wording materially affects agent performance.

### D2. Schema Drift

The interface changes at the field level while preserving semantics.

Examples:

- renamed parameter,
- alias introduced,
- enum value renamed,
- snake_case to camelCase,
- requirement moved from description to structured schema,
- field order changes.

Motivation from literature:

- `ASA` explicitly discusses schema shift and evolving interfaces.
- `Hammer` shows naming conventions are a major source of instability.

### D3. Candidate-Set Drift

The correct tool remains available, but the surrounding tool pool changes.

Examples:

- extra distractor tools,
- near-duplicate tools,
- semantically adjacent tools,
- larger tool set with overlapping descriptions.

Motivation from literature:

- `Trace-Free+` studies scaling to over 100 candidate tools.
- `Hammer` highlights irrelevance detection.
- BFCL includes irrelevance-focused categories.

### D4. Feedback Drift

The execution environment returns different error messages or provider-specific feedback.

Examples:

- concise validation error,
- stack-trace style message,
- vendor-specific error wording,
- retryable vs non-retryable wording.

Motivation from literature:

- `Fission-GRPO` shows recovery from execution errors is a major bottleneck.

### Pilot drift set

Use only:

- D1 description drift,
- D2 schema drift,
- D3 candidate-set drift.

Add D4 only after the pilot is successful.

---

## 8. Benchmarks and Evaluation Assets

## 8.1 Primary benchmark for the paper: DICE-Bench

Why use it:

- recent and open,
- realistic multi-round, multi-party tool use,
- 1,607 dialogues,
- 124 tools,
- built for dispersed tool clues across turns.

Role in the paper:

- main benchmark for realistic tool calling under dialogue dispersion.

### Pilot use

Start with the provided sample data and then a 100-200 example subset from the full dataset.

## 8.2 Secondary benchmark: BFCL selected categories

Use BFCL because it gives executable evaluation and ready-made categories aligned with our problem:

- `irrelevance`
- `multi_turn_base`
- `multi_turn_miss_param`
- `multi_turn_long_context`
- `format_sensitivity` (if compatible with chosen prompting mode)

Role in the paper:

- robustness sanity check across a widely used function-calling evaluation harness,
- especially useful for candidate-set drift and prompt/format sensitivity.

### Pilot use

Use small subsets through `--run-ids` rather than full categories.

## 8.3 Optional tertiary benchmark: ComplexFuncBench

Why it is useful:

- complex constraints,
- multi-step function calling,
- parameter reasoning,
- long parameter values,
- long-context behavior.

Why it is optional:

- setup is heavier,
- response-based evaluation may require extra API dependencies,
- it should not block the main study.

Use this only after the main pipeline works on DICE-Bench and BFCL.

---

## 9. Model Selection

**Updated decision:** use the **Qwen3.5** family as the primary model line and run it locally on **Colab**.  
We are no longer assuming Groq/Grok-hosted inference, and we are de-emphasizing older Llama checkpoints.

## 9.1 Pilot model

Use one model for the pilot:

- **Qwen3.5-9B**

Why:

- current Qwen family choice,
- large enough to be meaningful without turning the pilot into a serving problem,
- practical to host on Colab A100/H100 with vLLM or SGLang,
- easy to rerun many times with cached prompts and outputs.

## 9.2 Full paper model set

Recommended:

- **Qwen3.5-4B**
- **Qwen3.5-9B**

Rationale:

- current within-family scale comparison,
- avoids spending early project time on cross-family serving and prompt-format differences,
- both are realistic for Colab-class GPUs.

Optional fourth model if setup is easy:

- one larger Qwen3.5 checkpoint or one non-Qwen comparison model

But this is optional and should not delay the project.

## 9.3 Deployment strategy

Default execution path:

- self-host on **Colab**
- use **vLLM** first; switch to **SGLang** only if it gives a clear advantage
- keep the pilot on one stable serving stack

Practical guidance:

- prefer **bf16** or the cleanest supported local format for the pilot,
- only use quantization if memory pressure makes it necessary,
- cache raw generations, parsed tool calls, validator traces, and repair outputs,
- avoid mixing local and hosted inference in the pilot because it complicates latency and reproducibility analysis.

## 9.4 Model diversity rule

For the pilot, **one model is enough**.

For the paper, **two Qwen3.5 sizes are enough** unless the results are so strong that adding a third model is trivial.

The paper does not need to become an ecosystem-wide leaderboard.

---

## 10. Metrics

We need metrics that separate selection failures from execution failures.

### Primary metrics

- **Tool selection accuracy**
- **Argument exact match**
- **Execution success rate**
- **Overall task success**

### Robustness metrics

- **Performance drop under drift**: `score(original) - score(drifted)`
- **Recovery rate**: `(score(ours) - score(drifted)) / (score(original) - score(drifted))`
- **Repair success rate** among initially invalid calls

### Efficiency metrics

- average latency per example,
- average extra tokens due to repair,
- percentage of examples triggering repair.

### Analysis metrics

- error type distribution:
  - wrong tool,
  - missing field,
  - unknown field,
  - invalid enum,
  - malformed JSON,
  - unhelpful retry.

### Minimum stats for the paper

For each benchmark/model:

- mean score,
- absolute drop,
- absolute recovery,
- at least one significance or bootstrap confidence interval if possible.

---

## 11. Pilot Plan

## 11.1 Pilot objective

Determine within 7-10 days whether the topic is worth carrying through to submission.

The pilot is successful if it establishes both:

1. **there is a real robustness problem**, and
2. **our lightweight defense recovers enough performance to be interesting**.

## 11.2 Pilot scope

### Benchmarks

- BFCL small subset:
  - `irrelevance`
  - `multi_turn_miss_param`
  - `multi_turn_base`
- DICE-Bench:
  - sample data first,
  - then ~100 example subset

### Model

- Qwen3.5-9B only

### Drift types

- D1 description drift:
  - verbose,
  - example-heavy,
  - promotional
- D2 schema drift:
  - parameter rename,
  - enum alias,
  - field order variation
- D3 candidate-set drift:
  - 3-5 distractor tools with overlapping descriptions

### Systems to run

- Original benchmark baseline
- Drifted baseline
- Drifted + naive retry
- Drifted + SchemaShield-Lite

## 11.3 Pilot deliverables

At the end of the pilot we must have:

- a working inference harness,
- a perturbation script,
- a validator,
- one-shot repair prompt,
- a result table,
- 15-20 manually reviewed failure examples,
- a go/no-go decision.

## 11.4 Pilot go/no-go criteria

Proceed with the paper only if **all** of the following hold:

1. Drift causes at least **8-10 absolute points** drop on at least one benchmark slice.
2. Full SchemaShield-Lite recovers at least **25% of lost performance** or at least **4-5 absolute points**.
3. The recovered examples are interpretable and cluster into recognizable failure modes.
4. Latency overhead remains reasonable:
   - repair triggered on a minority of cases, or
   - average runtime increase is acceptable for an academic robustness paper.

## 11.5 Pilot pivot criteria

Pivot to the backup RAG topic if:

- drift effects are too small,
- results are inconsistent across reruns,
- repair does not recover enough performance,
- or benchmark setup overhead dominates the research work.

---

## 12. Pilot Implementation Plan

## 12.1 Suggested repository structure

Create a new project area, for example:

```text
tool-drift/
  README.md
  pyproject.toml
  configs/
    pilot_bfcl.yaml
    pilot_dice.yaml
    full_main.yaml
  data/
    bfcl_run_ids.json
    dice_subsets/
  benchmarks/
    bfcl/
    dice/
  drift/
    description_drift.py
    schema_drift.py
    candidate_drift.py
  defense/
    canonicalizer.py
    validator.py
    alias_map.py
    repair_prompt.py
    repair_runner.py
  eval/
    metrics.py
    aggregate.py
    error_taxonomy.py
  scripts/
    run_pilot_bfcl.py
    run_pilot_dice.py
    run_full.py
    summarize_results.py
```

## 12.2 Pilot coding order

1. Benchmark runner for one benchmark
2. Model inference wrapper
3. Drift injection module
4. Validator
5. Repair prompt
6. Metrics aggregation
7. Error review notebook or script

### Important sequencing rule

Do not build the full framework first. Build the smallest loop that can answer:

> original vs drifted vs repaired

on a tiny benchmark slice.

## 12.3 Repair prompt design

The repair prompt should be short and strict. It should include:

- task context,
- current invalid call,
- canonical schema,
- explicit validation error,
- output contract: only corrected tool call JSON.

Do not ask for explanations in the repair step during the pilot.

## 12.4 Validator design

Start with deterministic checks:

- JSON parse success,
- valid tool name,
- valid argument names,
- required-field coverage,
- type checks,
- enum checks.

Only add fuzzy alias mapping after the deterministic pipeline works.

---

## 13. Full Timeline to Submission

## Phase 0: Setup and pilot harness
**March 27 - March 30**

- finalize topic and plan,
- clone/install DICE-Bench and BFCL,
- verify local Colab inference for Qwen3.5-9B,
- prepare small benchmark subsets,
- define the initial drift taxonomy.

**Exit criterion:** one end-to-end baseline run completes.

## Week 1: Pilot baseline and drift injection
**March 31 - April 5**

- run original baseline on BFCL subset and DICE sample,
- implement D1/D2/D3 perturbations,
- measure raw drift-induced drop,
- inspect examples manually.

**Gate 1 decision:** continue only if the robustness drop is clearly measurable.

## Week 2: Minimal defense
**April 6 - April 12**

- implement canonical tool card,
- implement validator,
- implement one-shot repair,
- compare drifted baseline vs repaired system.

**Gate 2 decision:** continue only if the repair meaningfully recovers performance.

## Week 3: Expand benchmarks and ablations
**April 13 - April 19**

- expand BFCL subset,
- move from DICE sample to 100-200 examples,
- run ablations:
  - canonicalization only,
  - repair only,
  - full method,
- begin formal error taxonomy.

**Deliverable:** first credible result table.

## Week 4: Strengthen the method
**April 20 - April 26**

- improve alias matching,
- add better distractor-tool generation,
- refine drift taxonomy wording,
- add D4 feedback drift if time permits.

**Gate 3 decision:** freeze the method after this week unless results force a change.

## Week 5: Main evaluation
**April 27 - May 3**

- run the full main experiment matrix on 2-3 models,
- cache outputs and validator traces,
- collect latency and repair-rate stats,
- start figure drafting.

## Week 6: Analysis and optional benchmark extension
**May 4 - May 10**

- error breakdown tables,
- performance-by-drift-type analysis,
- optional ComplexFuncBench subset if stable,
- select best qualitative examples.

## Week 7: Paper drafting
**May 11 - May 17**

- write full first draft,
- build tables and figures,
- finalize related work and bibliography,
- check if any missing experiment is essential.

## Week 8: Final experiments and submission
**May 18 - May 25**

- run only the experiments needed to close gaps,
- polish wording and figures,
- finalize abstract, title, and contributions,
- prepare camera-ready-quality plots even before acceptance,
- submit by May 25.

---

## 14. Experiment Matrix

We need a matrix that is ambitious enough for a paper but not too large.

### Pilot matrix

| Benchmark | Model | Drift types | Systems |
|-----------|-------|-------------|---------|
| BFCL subset | Qwen3.5-9B | D1, D2, D3 | Original, Drifted, Naive Retry, Ours |
| DICE sample/subset | Qwen3.5-9B | D1, D2, D3 | Original, Drifted, Naive Retry, Ours |

### Main matrix

| Benchmark | Models | Drift types | Systems |
|-----------|--------|-------------|---------|
| DICE-Bench | 2-3 models | D1, D2, D3 | B0-B5 |
| BFCL selected categories | 2-3 models | D1, D2, D3, optional D4 | B0-B5 |
| ComplexFuncBench optional | 1-2 models | D2, D3 | B0, B1, B5 |

### Severity levels

Use at least two severity levels for each drift type:

- `mild`
- `strong`

This helps avoid a paper that depends on one arbitrary perturbation strength.

---

## 15. Planned Figures and Tables

### Likely figures

- **Figure 1:** Interface drift taxonomy with examples
- **Figure 2:** SchemaShield-Lite pipeline
- **Figure 3:** Performance under increasing drift severity
- **Figure 4:** Error type breakdown before and after repair
- **Figure 5:** Recovery vs latency tradeoff

### Likely tables

- **Table 1:** Related benchmark coverage
- **Table 2:** Main benchmark results
- **Table 3:** Ablation study
- **Table 4:** Per-drift-type recovery rates
- **Table 5:** Qualitative case studies

---

## 16. Risks and Mitigations

## Risk 1: Benchmark setup slows everything down

Mitigation:

- use small subsets first,
- start with BFCL `--run-ids`,
- use DICE sample data before full dataset.

## Risk 2: Drift perturbations look too synthetic

Mitigation:

- anchor D1 to the modes used in `Gaming Tool Preferences`,
- anchor D2 to schema-shift scenarios discussed in `ASA`,
- anchor D3 to irrelevance and candidate growth in BFCL/Hammer/Trace-Free+.

## Risk 3: Method gains are too small

Mitigation:

- keep the problem framing narrow,
- add alias-aware matching,
- emphasize recovery of invalid calls, not only top-line exact match,
- if needed, reframe as a strong empirical robustness study plus lightweight mitigation.

## Risk 4: Too many models, too little time

Mitigation:

- freeze pilot to one model,
- cap full paper to 2-3 models max.

## Risk 5: ComplexFuncBench becomes a sink

Mitigation:

- treat it as optional,
- do not let it block the paper.

---

## 17. Decision Gates

### Gate 1: End of Week 1

Question:

> Is interface drift measurably hurting performance?

If no, pivot to the RAG backup.

### Gate 2: End of Week 2

Question:

> Does the minimal defense recover enough performance to justify a paper?

If no, either:

- pivot to the backup topic, or
- narrow the paper into a diagnostic benchmark study only if the failure analysis is very strong.

### Gate 3: End of Week 4

Question:

> Is the result set strong enough to freeze the method and spend the remaining time on scale, analysis, and writing?

If no, cut optional work immediately.

---

## 18. Writing Plan

The paper should follow this structure:

1. **Introduction**
2. **Problem Setup and Drift Taxonomy**
3. **Method: SchemaShield-Lite**
4. **Experimental Setup**
5. **Main Results**
6. **Ablations and Error Analysis**
7. **Related Work**
8. **Limitations and Future Work**

### Claims we should be able to make

- current open models are brittle under realistic interface drift,
- drift is not monolithic; different drift types break different parts of the tool-calling pipeline,
- a lightweight training-free layer can recover meaningful performance,
- recovery comes at moderate cost.

### Claims we should avoid unless strongly supported

- safety guarantees,
- universal robustness,
- superiority over all fine-tuned tool-learning methods,
- real-world deployment readiness.

---

## 19. Citation Map: Which Papers Support Which Claims

This section is meant to help with writing later.

### Tool interfaces matter

- **Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use** (2026-02-23, arXiv:2602.20426)  
  Use for: tool descriptions and schemas are a bottleneck; interfaces are often human-oriented; scaling to large tool sets matters.

### Evolving interfaces and schema shift are real

- **ASA: Training-Free Representation Engineering for Tool-Calling Agents** (2026-02-04, arXiv:2602.04935)  
  Use for: tool calling is brittle under evolving interfaces; prompt/schema engineering is fragile under distribution shift; schema shift is a practical deployment problem.

### Error recovery is a major weakness

- **Robust Tool Use via Fission-GRPO** (2026-01-22, arXiv:2601.15625)  
  Use for: multi-turn execution errors are inevitable; smaller models often fail to recover; recovery rate is an important metric.

### Single-turn benchmarks are not enough

- **DICE-BENCH** (2025-06-28, arXiv:2506.22853)  
  Use for: realistic function calling requires multi-round, multi-party dialogue and dispersed clues.

### Complex function calling is still hard

- **ComplexFuncBench** (2025-01-17, arXiv:2501.10132)  
  Use for: multi-step, constrained, reasoning-heavy, long-context function calling remains challenging.

### Tool description wording can radically bias usage

- **Gaming Tool Preferences in Agentic LLMs** (2025-05-23, arXiv:2505.18135)  
  Use for: tool descriptions can strongly alter selection behavior; surface-form sensitivity is a real phenomenon.

### Naming conventions mislead models

- **Hammer: Robust Function-Calling for On-Device Language Models via Function Masking** (2024-10-06, arXiv:2410.04587)  
  Use for: function and parameter names can mislead models; robustness benefits when the model is pushed to focus on semantics rather than names.

### Stable benchmark infrastructure matters

- **StableToolBench** (2024-03-12, arXiv:2403.07714)  
  Use for: tool-learning evaluation needs stable APIs and stable evaluation conditions.

### Clarification and correction matter for incomplete tool calls

- **AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification** (2025-03-03, arXiv:2503.01940)  
  Use for: real-world tool use often involves missing or ambiguous information and error correction.

### Strong tool-use training baselines exist, so our goal is not to out-train them

- **ToolACE** (2024-09-02, arXiv:2409.00920)  
  Use for: synthetic tool-learning data can produce strong function-calling models.
- **xLAM** (2024-09-05, arXiv:2409.03215)  
  Use for: specialized action models are powerful baselines in tool use.

### Optional security framing

- **Les Dissonances** (2025-04-04, arXiv:2504.03111)  
  Use only lightly if needed: multi-tool environments have fragile control flows and real robustness consequences.

---

## 20. Core References

- NGEN-AI 2026 official site: <https://ngen-ai.org/>
- Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use: <https://arxiv.org/abs/2602.20426>
- ASA: Training-Free Representation Engineering for Tool-Calling Agents: <https://arxiv.org/abs/2602.04935>
- Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors: <https://arxiv.org/abs/2601.15625>
- DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues: <https://arxiv.org/abs/2506.22853>
- ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario: <https://arxiv.org/abs/2501.10132>
- Gaming Tool Preferences in Agentic LLMs: <https://arxiv.org/abs/2505.18135>
- Hammer: Robust Function-Calling for On-Device Language Models via Function Masking: <https://arxiv.org/abs/2410.04587>
- StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models: <https://arxiv.org/abs/2403.07714>
- AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification: <https://arxiv.org/abs/2503.01940>
- ToolACE: Winning the Points of LLM Function Calling: <https://arxiv.org/abs/2409.00920>
- xLAM: A Family of Large Action Models to Empower AI Agent Systems: <https://arxiv.org/abs/2409.03215>

---

## 21. Immediate Next Actions

This is the exact sequence to start now:

1. Set up BFCL and DICE-Bench locally.
2. Run one clean baseline with `Qwen3.5-9B`.
3. Create tiny benchmark subsets for fast iteration.
4. Implement D1/D2/D3 perturbations.
5. Measure original vs drifted performance.
6. Implement validator and one-shot repair.
7. Re-run the same tiny subsets.
8. Make the go/no-go call by the end of Week 2 at the latest.

If this sequence works, the project is viable.
