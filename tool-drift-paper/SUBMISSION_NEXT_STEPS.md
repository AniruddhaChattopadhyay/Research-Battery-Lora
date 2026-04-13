# Submission Next Steps

Working draft combining two independent reviews into an action plan for making
`tool-drift-paper` submission-ready.

## Status Update

As of Apr 13, the paper has moved past the main validity blocker. The manuscript
now uses the non-oracle candidate-retry protocol as its primary story, and the
oracle-constrained runs have been demoted to upper-bound / ablation context.

The current state is:

- `main.tex` has been rewritten around the non-oracle protocol
- confidence intervals, figures, and expanded related work are in the paper
- paired significance for the main non-oracle contrasts is now in the paper
- broader oracle-vs-non-oracle repaired comparisons are now in the paper
- a refreshed BFCL live rerun now provides wall-clock latency context
- drift recovery vs baseline patching is explicitly separated in the discussion
- a secondary BFCL non-oracle replication on Llama-4-Scout is now in the paper
- a BFCL closed-model sanity check on GPT-4o-mini is now in the paper
- the drift-realism discussion now cites public GitHub/Stripe API-evolution docs
- the paper compiles cleanly to `main.pdf`

The main remaining risks are now evidence gaps rather than protocol mismatch:

- the clean primary non-oracle story is still mostly anchored on Qwen3.5-9B,
  even though there is now a weaker BFCL replication on Llama-4-Scout
- wall-clock latency is only measured on a refreshed BFCL rerun, not yet as a
  broader sweep
- the synthetic nature of the drift operators remains a real limitation
- the closed-model evidence is still only one BFCL sanity slice

## Current Active Experiment

No critical live experiment is blocking the draft now.

Most recent completed evidence upgrade:

- run: `bfcl-live-20260413-203305-358013`
- model: `GPT-4o-mini`
- protocol: BFCL-200, non-oracle candidate-retry
- result: `0.865 -> 0.840 -> 0.860`
- note: the initial OpenRouter attempt failed because the OpenAI provider
  rejected BFCL tool names that did not match `^[a-zA-Z0-9_-]+$`; the rerun
  succeeded after adding provider-safe tool-name aliasing in
  `tool-drift/inference/openrouter_client.py`

## Historical Review Snapshot

The paper has a real problem setting, strong internal structure, useful controls,
and several genuinely interesting empirical findings:

- naive retry does not explain the gains
- candidate-set drift appears much more damaging than description/schema drift
- validation feedback seems to matter more than the canonical card
- repair harms are zero in the reported runs
- the manual error analysis is better than average for a draft at this stage

This section reflects the pre-fix review snapshot that originally drove the work.
Several items below have since been resolved in the manuscript and codebase.

At the time of those reviews, the largest issues were:

- the current evaluation may rely on oracle knowledge of the correct tool during repair
- the method contribution is thin if reduced to validation plus retry
- the empirical scale is still modest for the level of generality claimed
- the drift-recovery story is confounded by the fact that repair also fixes baseline errors
- the related work and evaluation framing need expansion

## Historical Recommendation

At that stage, the correct recommendation was not to submit the draft as-is.

The fastest path to a defensible submission is:

1. Resolve the oracle/non-oracle repair issue.
2. Separate drift recovery from baseline error correction.
3. Reposition the paper around the strongest empirical findings, not around the method name alone.
4. Add a small amount of targeted evidence to shore up generalizability.

## Initial Non-Oracle Rerun

First realistic reruns completed on Apr 12, 2026 using `repair_target_mode:
predicted_tool` for Qwen3.5-9B.

Observed results:

- DICE-300:
  - oracle-style paper run: `0.750 -> 0.713 -> 0.827`
  - non-oracle rerun: `0.747 -> 0.703 -> 0.780`
- BFCL-200:
  - oracle-style paper run: `0.790 -> 0.640 -> 0.795`
  - non-oracle rerun: `0.790 -> 0.580 -> 0.725`

Interpretation:

- Non-oracle repair still improves over drifted on both benchmarks.
- The gains are materially smaller than in the oracle-style setup:
  - DICE repaired score drops by about `4.7` points relative to the paper run.
  - BFCL repaired score drops by about `7.0` points.
- Zero repair harms still hold on the originally-correct slice in these reruns.
- A meaningful fraction of missed recoveries are due to unresolved repair targets
  caused by missing tool names:
  - DICE unresolved targets: `35`
  - BFCL unresolved targets: `34`

Implication for the paper:

- The broad original deployment story is too strong as currently written.
- The paper can still claim that inference-time validation and repair help under
  a non-oracle setting, but it must clearly distinguish oracle-constrained and
  non-oracle protocols.
- The next obvious improvement is a non-oracle fallback for missing-tool-name
  cases, since those currently block repair entirely.

## Missing-Tool Fallback Rerun

Follow-up reruns completed on Apr 12-13, 2026 after adding a non-oracle
candidate-set retry for missing-tool-name cases (`unresolved_repair_mode:
candidate_retry`).

Observed results:

- DICE-300:
  - earlier non-oracle rerun: `0.747 -> 0.703 -> 0.780`
  - candidate-fallback non-oracle rerun: `0.757 -> 0.717 -> 0.813`
- BFCL-200:
  - earlier non-oracle rerun: `0.790 -> 0.580 -> 0.725`
  - candidate-fallback non-oracle rerun: `0.790 -> 0.580 -> 0.805`

Key changes:

- DICE unresolved repair targets dropped from `35` to `15`
- BFCL unresolved repair targets dropped from `34` to `9`
- DICE recoveries on the originally-correct slice improved from `16` to `19`
- BFCL recoveries on the originally-correct slice improved from `26` to `33`
- Zero repair harms still held

Implication for the paper:

- The paper can now be rewritten around a fully non-oracle repair protocol.
- The non-oracle story is no longer merely a weaker appendix result.
- On BFCL, the candidate-fallback non-oracle run is now at least competitive
  with the earlier oracle-style run, which strongly suggests the original
  evaluation story should be replaced rather than defended.

## Priority 0: Must Resolve Before Submission

### P0.1 Clarify and fix the repair setting

Core question:

- Does repair know the correct target tool during evaluation?

Why this matters:

- If repair is constrained to the gold or benchmark-target tool, the current setup is much closer to argument repair than to full robust tool calling.
- Reviewers will treat this as the main threat to validity.

Required actions:

- Audit the exact repair protocol and write down, in one paragraph, what information is available at repair time.
- If the current evaluation uses gold-tool knowledge, run a non-oracle version.
- Add a comparison between:
  - full current setup
  - non-oracle validation + retry
  - oracle-constrained repair

Deliverable:

- A clear table or subsection titled `Oracle vs Non-Oracle Repair`.

Decision gate:

- If non-oracle repair still works well, keep the current core story.
- If non-oracle repair drops sharply, reframe the paper as a narrower study of post-selection argument repair under interface drift.

### P0.2 Separate drift recovery from baseline repair

Why this matters:

- `repair > original` is interesting, but it muddies the causal story.
- Right now the paper mixes two effects:
  - recovering performance lost due to drift
  - improving examples the model already got wrong before drift

Required actions:

- Report three disjoint slices:
  - originally correct, then broken by drift
  - originally wrong, still wrong under drift, then fixed by repair
  - originally wrong, changed in some other way
- Add one compact decomposition table for `drift recovery` vs `baseline patching`.

Deliverable:

- A new results table or short subsection that makes the two effects impossible to confuse.

### P0.3 Reposition the paper around the strongest contribution

Why this matters:

- Reviewers are likely to say the method itself is an engineering recipe, not a novel algorithmic contribution.
- The strongest parts of the draft are empirical:
  - naive retry underperforms
  - candidate-set drift dominates
  - canonical card is not the load-bearing component

Required actions:

- Rewrite the paper so the headline contribution is primarily empirical/analytical.
- Keep `SchemaShield-Lite` as the vehicle if useful, but stop overselling it as the central novelty.

Possible framing:

- `An empirical study of interface drift in tool calling`
- `Inference-time validation and repair for tool-call robustness under interface drift`
- `What actually breaks under tool interface drift, and what repair helps`

Deliverable:

- Revised title, abstract, introduction, and conclusion aligned to the same story.

## Priority 1: High-Value Evidence Upgrades

### P1.1 Add at least one more BFCL model

Why:

- BFCL at two models is thin.
- The abstract and conclusion currently imply broader generality than the evidence supports.

Minimum acceptable improvement:

- Add one more open model on BFCL.

Preferred improvement:

- Add both remaining models, if cost allows.

### P1.2 Add one frontier-model sanity check

Why:

- A reviewer will ask whether this matters for models used in real production.

Minimum acceptable improvement:

- Run a small slice on one closed or frontier model.

Possible scope:

- 50-100 examples on BFCL or DICE
- original / drifted / repaired only

Note:

- This does not need to be the centerpiece. It is a credibility check.

### P1.3 Strengthen the drift realism argument

Why:

- The drift operators are currently author-designed and synthetic.

Required actions:

- Add 3-5 real examples of interface drift drawn from public API changelogs, SDK migrations, or tool updates.
- Show how each maps onto the paper's categories:
  - description drift
  - schema drift
  - candidate-set drift

Optional stronger version:

- Use one or two real drift patterns directly in evaluation.

### P1.4 Expand the drift-type ablation beyond one model

Why:

- The claim `candidate-set drift dominates` is currently based on a single model.

Minimum acceptable improvement:

- Run the drift-type ablation on one additional model.

Preferred improvement:

- Run it on one BFCL model too.

### P1.5 Add significance language, not just CIs

Why:

- Bootstrap CIs are already a good step, but overlapping intervals weaken several narrative claims.

Required actions:

- State explicitly which comparisons are clearly separated and which are not.
- Use paired tests where appropriate, since the same examples are reused across conditions.

Deliverable:

- One sentence in Results noting which deltas are robust and which should be interpreted cautiously.

### P1.6 Add latency, not only token overhead

Why:

- A deployment-oriented paper should discuss wall-clock cost.

Required actions:

- Report average added latency when repair triggers.
- Report end-to-end average overhead per example.

## Priority 2: Evaluation Quality Fixes

### P2.1 Add a second evaluation view beyond strict matching

Why:

- The paper already admits that strict scoring undercounts recoveries.
- Once scorer noise is this visible, reviewers will want confirmation.

Required actions:

- Add one of:
  - relaxed semantic scoring
  - manual annotation on a sampled subset
  - canonicalized argument comparison for formatting-only differences

Deliverable:

- A short appendix or subsection confirming that the main conclusions survive.

### P2.2 Make the repair prompt details concrete

Why:

- Section 4.2 is too thin given that these decisions materially affect outcomes.

Required actions:

- Include the repair prompt template in the appendix or main text.
- State the repair token budget and why it changed.
- Clarify what the validator reports back to the model.

### P2.3 Tighten claims about modern-model robustness

Why:

- The current wording is broader than the evidence.

Required actions:

- Replace general claims like `modern models are robust to description/schema drift`
  with benchmark- and model-scoped wording unless additional runs support the broader statement.

## Priority 3: Writing and Positioning Revisions

### P3.1 Expand related work

Current state:

- The reference list is too short for submission.

Add coverage for:

- tool-use and function-calling benchmarks
- tool robustness and tool-selection literature
- schema validation / contract checking / interface robustness from software engineering
- LLM self-repair, retry, and reflection-style recovery methods
- broader function-calling evaluation ecosystems

Goal:

- Make it clear what is actually new here:
  - the interface-drift framing
  - the empirical decomposition
  - the analysis of repair mechanisms

### P3.2 Rewrite the abstract and intro to match the evidence

Current issue:

- The abstract reads like a method paper with broad deployment conclusions.

Rewrite target:

- emphasize the empirical findings first
- present the repair stack as a simple, practical intervention
- avoid implying stronger novelty or generality than the results support

### P3.3 Add a short `Threats to Validity` style paragraph

Include:

- synthetic drift design
- hosted inference / provider effects
- scorer imperfections
- limited model coverage
- possible oracle bias if retained in any part of the evaluation

### P3.4 Consider renaming or reframing the method

Why:

- The current ablation weakens the case for the name as the main story.

Options:

- keep `SchemaShield-Lite` as the historical full stack name
- introduce a minimal variant like `validation + repair`
- describe the method as a practical pipeline rather than a novel architecture

## Priority 4: Presentation Polish

### P4.1 Improve figure quality

Required actions:

- clean x-axis labels
- improve spacing and typography
- ensure the figures are publication-quality rather than internal-result-quality

### P4.2 Add one summary figure for the main narrative

Best candidate:

- a simple conceptual figure showing:
  - where drift enters
  - what types of failure it causes
  - what part of the pipeline addresses which failure

### P4.3 Add appendix materials

Recommended appendix items:

- repair prompt template
- examples of each drift type
- example unrecovered cases
- oracle vs non-oracle protocol details
- scoring details and relaxed-eval notes

## Recommended Paper Story

Unless the non-oracle rerun is unexpectedly strong, the safest framing is:

- primary contribution: empirical analysis of interface drift and repair in tool calling
- secondary contribution: a simple validation-and-repair inference recipe

In that version, the three most defensible headline findings are:

1. naive retry is not enough
2. candidate-set drift is more damaging than lightweight description/schema drift in the tested setup
3. structured validation feedback is the key component of recovery

## Execution Plan

### Phase 1: Resolve validity blockers

- [ ] Audit the repair protocol and document exactly what is oracle vs non-oracle.
- [ ] Run a non-oracle repair comparison.
- [ ] Add drift-recovery vs baseline-patching decomposition.

### Phase 2: Add minimal evidence for generality

- [ ] Run at least one more BFCL model.
- [ ] Run one frontier-model sanity check on a small slice.
- [ ] Run one extra model for the drift-type ablation.

### Phase 3: Strengthen realism and scoring

- [ ] Add real-world drift examples from public API/tool evolution.
- [ ] Add a second evaluation view beyond strict exact matching.
- [ ] Add latency reporting.

### Phase 4: Rewrite for submission

- [ ] Rewrite title, abstract, intro, discussion, and conclusion around the final evidence.
- [ ] Expand related work substantially.
- [ ] Add appendix materials and protocol details.
- [ ] Polish figures and tables.

## Minimum Viable Submission Version

If time is limited, the minimum credible upgrade path is:

- resolve the oracle/non-oracle repair issue
- separate drift recovery from baseline repair
- add one more BFCL model
- add one frontier sanity check
- expand related work
- soften all broad claims to match the actual evidence

If those are done well, the paper can become submission-ready.
If they are not, the likely reviewer outcome is that the paper is interesting but under-validated.
