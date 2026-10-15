# NGEN-AI 2026 Conference Q&A Guide

**Paper 15:** Robust Function Calling for LLM Agents Under Interface Drift  
**Presentation:** Current 17-slide PPTX  
**Talk format:** 16-minute presentation plus approximately 4 minutes of Q&A  
**Main slides:** 1-13 | **Backup slides:** 14-17

## How to answer live

Use a three-part structure:

1. Give the direct answer in one sentence.
2. Support it with one result or design detail.
3. End with the relevant limitation.

Do not try to recite the entire paper. Most answers below are intentionally short enough to deliver in 20-40 seconds.

## Thirty-second overview

We study interface drift: descriptions, parameter schemas, or visible tool catalogs changing after deployment while the underlying tool semantics remain fixed. SchemaShield-Lite validates a model's call against the live interface and, only when the call is invalid or incomplete, gives the model structured errors and one constrained repair opportunity. On Qwen3.5-9B, this raised DICE-300 accuracy from 71.7% under drift to 81.3% after repair, and BFCL-200 from 58.0% to 80.5%. The method works best when the failure is validator-visible and a usable tool name can be resolved; it does not solve schema-valid but semantically wrong calls.

## Five points to remember

| Topic | Key message |
|---|---|
| Contribution | A controlled study of semantics-preserving interface drift and a training-free, non-oracle recovery path. |
| Main result | Repair recovered 19/22 DICE and 33/45 BFCL drift-induced failures on the originally-correct slices. |
| Mechanism | Structured validation feedback matters; an identical retry or tool-name list alone does not explain the gain. |
| Main limitation | Valid-but-wrong calls bypass structural validation, and portability depends on tool-name resolution. |
| Cost | Repair-stage calls fired on 21.0% of DICE and 36.5% of BFCL examples, adding 32.6%-47.1% more tokens. |

# Questions to rehearse first

## What is the central contribution of this paper?

The paper defines interface drift as semantics-preserving changes to tool descriptions, schemas, or candidate catalogs after deployment. It then evaluates a lightweight, training-free recovery layer consisting of live validation, tool-selection retry when the name is missing, and one constrained repair. The contribution is not schema validation by itself; it is the problem formulation, controlled evaluation, and deployment-facing non-oracle protocol.

## What exactly makes a change semantics-preserving?

The underlying tool capability and the information required from the user remain unchanged. Only the surface contract seen by the model changes, such as renamed parameters, a stale example in the description, or a similar-looking tool entering the catalog. The work does not cover tool outages, changed business behavior, or genuinely new tasks.

If someone argues that a parameter rename is a breaking change, distinguish syntax from semantics: it breaks the old call syntax, but the capability and intended result remain the same. The evaluation deterministically re-expresses the gold call in the renamed schema before scoring.

## Why is adding a distractor tool considered interface drift?

The paper represents the interface as the description, schema, and visible candidate set. Adding a tool changes the decision surface presented to the model even when the original target tool and its behavior remain unchanged. Candidate-set drift therefore captures catalog evolution rather than a change in the target capability.

## How does SchemaShield-Lite work?

The base model first produces its ordinary tool call. A validator checks the tool name, required fields, and argument structure against the live catalog and schema. Valid calls pass through unchanged. Invalid calls receive the user context, failed call, current schema, and structured validation errors for one constrained repair. If the model supplied no usable tool name, the system first retries selection over the runtime-visible tools.

## Why do you call the primary protocol non-oracle?

The repair system never receives the benchmark's gold tool. It sees only deployment-available information: user context, the model output, and runtime-visible tools. If the model names a visible tool, repair is constrained to that tool. If it supplies no usable name, the system retries selection over the visible catalog.

The oracle-constrained runs are mechanism diagnostics and upper-bound context, not the primary deployment evidence.

## Isn't this just giving the model a second attempt?

The retry baselines test that explanation directly. On DICE, naive retry reached 72.7% and tool-list retry reached 75.3%, versus 81.3% for SchemaShield-Lite. On BFCL, they reached 61.0% and 62.5%, versus 79.5% for SchemaShield-Lite in that separate baseline run. Repetition and merely listing available names therefore do not explain the full gain; the structured failure context is important.

The careful version of the claim is that the structured feedback bundle drives the observed advantage. The experiments do not separately isolate every prompt field or token-budget difference.

## What is the main empirical result?

On Qwen3.5-9B, DICE accuracy moved from 75.7% on the original interface to 71.7% after drift and 81.3% after repair. BFCL moved from 79.0% to 58.0% and then 80.5%. On the originally-correct slice, repair recovered 19 of 22 DICE drift failures and 33 of 45 BFCL failures, with zero observed repair harms.

The paired repaired-versus-drifted tests give p = 3.7 x 10^-9 on DICE and p = 5.7 x 10^-14 on BFCL.

## How can repaired accuracy exceed original accuracy?

Repair sometimes fixes pre-existing base-model errors as well as drift-induced errors. Of 29 DICE improvements over the drifted call, 19 are genuine drift recovery and 10 are baseline patching. On BFCL, the split is 33 drift recoveries and 12 baseline patches. The central causal claim is about the drift-recovery portion, not that repair universally creates a better system than the original model.

## Which kind of drift caused the most damage?

In the Qwen3.5-9B DICE ablation, candidate-set drift was dominant. Description-only drift cost 1.3 percentage points, schema-only drift 0.7, candidate-set drift 4.7, and all three combined 5.0. A single overlapping distractor nearly reproduced the full combined drop.

This ordering is not claimed to be universal. It comes from one model, one benchmark slice, and one distractor. The transferable lesson is that catalog growth should be tested separately from documentation and schema changes.

## How realistic are the synthetic drift operators?

They are author-designed and deployment-motivated, not mined from a real-world change distribution. As a plausibility check, the paper audits 11 Notion and Stripe changelog items: six resemble description or schema changes, three have candidate-set analogues, and two are larger structural changes outside the paper's semantics-preserving scope.

That audit demonstrates resemblance, not prevalence. A larger longitudinal dataset mined from real interface histories is still needed.

## What does repair cost?

Repair-stage inference fires on 21.0% of DICE examples and 36.5% of BFCL examples. Relative token overhead is 32.6% and 47.1%, respectively. In one instrumented BFCL rerun, the repair path added approximately 4.1 seconds per example averaged over the whole slice, or 13 seconds among invoked cases.

Those are run-specific measurements, not a systematic comparison across providers, models, or pricing schemes.

## Does this generalize across models?

The strongest evidence remains within the Qwen family. Qwen3.5-35B-A3B strongly replicated the BFCL effect, moving from 63.5% to 84.5%. Llama-4-Scout improved only from 63.5% to 66.5% and left 43 tool names unresolved. GPT-4o-mini moved from 84.0% to 86.0%, but that gain was not statistically significant.

The correct conclusion is that the approach is promising but model-dependent. Tool-name resolution is the main observed portability bottleneck.

## What failures can the validator not detect?

Schema-valid but semantically wrong calls are the central blind spot. On the originally-correct slice, two of 22 DICE drift failures were valid-but-wrong and bypassed validation; BFCL had none in this run. A wrong but visible tool can also pass validation or cause repair to be constrained to the wrong target.

Addressing that requires semantic, execution-based, or task-level validation rather than structural schema checking alone.

# Harder reviewer questions

## Does zero repair harm mean the method is safe?

No. It means zero harms were observed in these particular evaluation slices. Passing valid calls through unchanged reduces the opportunity for harm, but it is not a formal guarantee, and these sample sizes cannot rule out rare failures.

Say **"zero observed harms in these slices,"** never **"repair cannot cause harm."**

## Why does the non-oracle BFCL result exceed the supposed oracle upper bound?

Oracle describes an information advantage, not a guarantee that a separate finite, nondeterministic run will produce a higher numerical score. The rows came from separate hosted runs. On shared examples, the repaired-score difference between the oracle and non-oracle protocols was not significant: p = 0.125 on DICE and p = 0.754 on BFCL.

## Why is BFCL repair 80.5% on one slide but 79.5% on the retry-baseline slide?

They come from separate runs. The 80.5% result is the final primary candidate-retry run. The 79.5% result was produced alongside the retry baselines. The benchmark slice and drift selection were fixed, but hosted inference could still be nondeterministic at temperature zero.

## Is one-shot repair really only one additional model call?

Not always. There is at most one constrained repair call, but when the original output contains no usable tool name, the system first performs a separate candidate-selection retry. Such cases can therefore incur two post-validation model calls. The reported invocation rate counts any post-validation call, including unsuccessful name-resolution attempts.

## Why constrain repair to the model-selected tool?

This preserves the non-oracle property and prevents the evaluator from leaking the benchmark target. It also narrows the repair problem. The tradeoff is that a confidently named but wrong visible tool may not be corrected, which is an explicit limitation of the current design.

## What does the canonical tool card contribute?

At repair time, much less than validation feedback. In the oracle-constrained ablation, validation plus retry without the card reached 83.0% on DICE and 82.0% on BFCL, compared with 82.7% and 79.5% for the full variant.

The card was more useful as prevention before the initial prediction: the BFCL card-only condition lifted the drifted score from 64.0% to 81.0%. The card can help prevent drift, while structured errors drive recovery after a failure is detected.

## Do overlapping confidence intervals undermine the significance claim?

No. The outcomes are paired on the same examples, so the paired exact McNemar test is more informative than visually comparing marginal bootstrap intervals. The intervals and paired tests support the large repaired-versus-drifted effects, but they do not justify claims about small differences between nearby runs.

Only the slice and drift selection used seed 42. Provider-side model nondeterminism was not quantified through a multi-seed inference study.

## Why use only 300 DICE and 200 BFCL examples?

These are fixed controlled slices, not full-benchmark leaderboard claims. DICE uses 300 single-function, round-one dialogues and allows repeated functions. BFCL uses 50 examples from each of four categories. The effects are large on these slices, but full-benchmark and production-scale replication remains necessary.

A particularly careful statistical caveat is that the reported uncertainty treats examples as the unit. Because DICE permits repeated functions, a function-clustered analysis would be a useful stronger check.

## Does exact matching underestimate performance?

In some cases. Among 13 manually inspected unrecovered cases from non-primary oracle-constrained runs, 11 were formatting or paraphrase differences, such as "regular_season" versus "regular season." This suggests that the metric is conservative for those inspected cases, but the small audit should not be converted into a revised headline accuracy.

The scoring is normalized but remains strict about some value representations.

## Are the retry baselines completely fair or token-matched?

They address two specific alternatives: an identical second attempt and a retry told only that the call is invalid plus the available tool names. They do not fully isolate the failed call, every validation field, and the larger repair token budget in a factorial or token-matched design. A fully context- and budget-matched ablation would be a useful follow-up.

## Why use different drift severities on DICE and BFCL?

DICE uses mild drift with one distractor, while BFCL uses medium drift with three distractors. They were intended as complementary stress settings, not directly comparable measurements of benchmark sensitivity. Recovery should therefore be compared within each benchmark; BFCL's larger raw drop does not prove that BFCL is intrinsically less robust.

## Did you evaluate full agents or actual tool execution?

No. The evaluation scores one function call per example. DICE supplies multi-turn conversational context, but the chosen slice is single-function and round one. BFCL includes multiple-function candidate settings, but the study does not execute sequential tool plans. Compounding errors, tool responses, side effects, and end-to-end task completion remain untested.

## Why not use constrained decoding instead?

Constrained decoding can guarantee syntactic or schema validity, but it cannot by itself guarantee the correct tool or semantically correct values, especially under catalog ambiguity. External validation and repair are complementary to constrained decoding.

If a deterministic schema migration map is available, that should be used first. The learned repair path is most useful when exact deterministic adaptation is unavailable.

## Could the method defend against malicious tool injection?

This was not tested. Automatic repair could even mask the presence of a malicious distractor and make a compromised catalog appear operational. Production systems should authenticate and provenance-check newly added tools independently of the repair layer.

## Why was Qwen3.5-9B selected as the primary model?

The paper designates it as the primary model but does not present a formal model-selection or preregistration study. It should not be presented as representative of every tool-calling model. The larger Qwen replication strengthens the within-family result, while the Llama and GPT experiments demonstrate portability limits.

## What happens if the repaired call remains invalid?

The evaluation allows one repair output and does not retry until success. Persistent missing-field errors are among the BFCL residuals. In production, the repaired output should be validated again and the system should fail closed or escalate rather than execute an invalid call; that final operational policy was not separately evaluated here.

## Is the schema validator sufficient for production safety?

No. It is useful for structural failures such as tool identity, missing required fields, and argument structure. It cannot establish business correctness, authorization, side-effect safety, or factual consistency. It should be one layer in a broader execution-safety system.

## What is the most important next experiment?

Build a larger longitudinal dataset of real interface changes and evaluate it across more model families and production-scale catalogs. Important extensions include sequential tool execution, semantic validators for valid-but-wrong calls, stronger tool-name resolution, multi-run latency and cost analysis, and function-clustered statistical checks.

# Numbers to memorize

| Metric | DICE-300 | BFCL-200 |
|---|---:|---:|
| Original accuracy | 75.7% | 79.0% |
| Drifted accuracy | 71.7% | 58.0% |
| Repaired accuracy | 81.3% | 80.5% |
| Drift failures recovered | 19/22 | 33/45 |
| Validator-visible drift misses | 20/22 | 45/45 |
| Repair-stage invocation | 21.0% | 36.5% |
| Relative token overhead | 32.6% | 47.1% |
| Unresolved tool names | 15 | 9 |
| Observed repair harms | 0 | 0 |

## Portability numbers

| BFCL-200 model | Drifted | Repaired | Interpretation |
|---|---:|---:|---|
| Qwen3.5-9B | 58.0% | 80.5% | Primary non-oracle result |
| Qwen3.5-35B-A3B | 63.5% | 84.5% | Strong same-family replication |
| Llama-4-Scout | 63.5% | 66.5% | 43 unresolved tool names |
| GPT-4o-mini | 84.0% | 86.0% | Small, non-significant gain |

# Phrasing guardrails

| Say this | Avoid this |
|---|---|
| "Zero observed repair harms in these slices." | "Repair cannot cause harm." |
| "Candidate-set drift dominates this Qwen3.5-9B DICE ablation." | "Candidate-set drift is always the most damaging." |
| "The changelog audit is a plausibility check." | "The operators represent the real-world drift distribution." |
| "The strongest replication remains within the Qwen family." | "The method is model-independent." |
| "One constrained repair, with an optional name-selection retry." | "The method always requires exactly one extra call." |
| "Structured feedback does more than blind retry in these runs." | "Every individual prompt component has been causally isolated." |
| "The repair gain is significant relative to the drifted call." | "Repair is significantly better than the original interface." |

# Strong closing answer

The method works when the failure is validator-visible and the intended tool name can be resolved. It does not solve schema-valid but semantically wrong calls, and the strongest evidence remains within the Qwen family. The practical result is that validation against the live interface plus one targeted repair is substantially more effective than retrying blindly.

# Reproducibility

The repository releases the fixed benchmark slices, drift configurations, prompts, scoring logic, normalized matching policy, example-level outputs, validation traces, repair targets, token usage, and raw artifacts. The paper pins repository commit `fbac156` for the reported results.

## Source materials

- `ngenai/main.tex` - paper manuscript.
- `ngenai/presentation-talk-track.md` - current 17-slide speaking guide.
- `ngenai/NGEN-AI-2026_Session7_Paper15_Chattopadhyay_slides.pptx` - presentation used at the conference.
