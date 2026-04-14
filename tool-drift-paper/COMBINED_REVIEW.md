# Combined Submission Review

Working synthesis of two independent reviews of `tool-drift-paper/main.tex`, combining overlapping concerns into one prioritized document we can use as the paper-fix backlog.

Review sources merged here:
- Codex review on April 13, 2026
- Claude review shared by the author on April 13, 2026

## Bottom Line

The paper is materially closer to submission-ready than it was before the latest pass.

The core idea is real, the non-oracle protocol is materially better than the earlier oracle-style story, and the paper is unusually honest about its own limitations. The main remaining risks are structural, not cosmetic:
- the strongest successful non-oracle evidence is still concentrated in one model family
- the drift realism story is still synthetic
- some of the paper's explanatory ablations do not match the paper's primary non-oracle protocol
- the current headline numbers still mix drift recovery with generic baseline patching, even though the framing is now clearer

## Current Status

Updated after the current manuscript-edit pass and the completed
`Qwen3.5-35B-A3B` BFCL non-oracle replication.

### Done

- `P0.1` Single-model dependence materially reduced: a BFCL-200 non-oracle replication on `Qwen3.5-35B-A3B` completed cleanly and supports the main claim, though the strongest successful evidence is still same-family rather than cross-family.
- `P0.2` Synthetic-drift validity strengthened with a compact official-changelog audit from Notion and Stripe; the manuscript now grounds the synthetic operators in real vendor change patterns while keeping the narrower claim language.
- `P0.3` Oracle/non-oracle ablation mismatch closed for the DICE drift-type section: singleton-candidate rows are now justified as non-oracle-equivalent, the missing distractor-bearing candidate-set row was rerun under the primary protocol, and the all-combined non-oracle row is taken from the existing primary DICE main run.
- `P0.4` Headline gains now read more clearly as drift recovery plus baseline patching, not pure drift recovery.
- `P0.5` Repairability gap addressed with a new manuscript subsection using existing-output analysis.
- `P1.1` Primary-run unrecovered-case audit added in compact form from existing JSON artifacts.
- `P1.3` Narrative structure cleaned up: primary non-oracle evidence is foregrounded, oracle context is labeled as upper-bound context, and Llama/GPT replications were moved into Results.
- `P1.4` Reproducibility and release statement added.
- `P1.5` Determinism/harms wording softened.
- `P1.2` Related work / bibliography expanded with targeted additions on constrained decoding, agentic function-calling robustness, and broader API evolution.
- `P2.1` DICE table inconsistency now explained in the main-results table caption.
- `P2.2` Abstract tightened substantially.
- `P2.3` "Two details" mismatch fixed.
- `P2.4` Hardcoded section reference replaced with `\ref`.

### In Progress

### Pending

- `P2.5` Figure polish.
- `P3.1` NGEN-AI packaging: de-anonymize, final author block, final page-budget check.
- `P3.2` IntelliSys packaging: keep anonymization and add any required GenAI declaration if used.

## Submission Recommendation

As of April 13, 2026:
- `NGEN-AI 2026` looks like the better fit for topic alignment
- `IntelliSys 2026` is still possible, but it is riskier and more time-constrained

Current venue implications:
- `NGEN-AI` submission deadline: May 25, 2026
- `NGEN-AI` review model: single-blind
- `NGEN-AI` full-paper limit: 16 LNCS pages
- `IntelliSys` late-breaking round deadline: May 1, 2026
- `IntelliSys` review model: double-blind
- `IntelliSys` formatting guidance allows 18 pages for main text, excluding references and appendices
- `IntelliSys` late-breaking papers are published in a separate post-conference proceedings volume

Practical recommendation:
- If we want time to improve the evidence, optimize for `NGEN-AI`
- If we keep `IntelliSys` alive, we should treat it as the stretch option
- Do not submit to both simultaneously; `NGEN-AI` explicitly requires the work not to be under review elsewhere

## Priority 0: Main Acceptance Risks

### P0.1 Single-model dependence [Done]

This is the clearest consensus blocker.

Why it mattered:
- The clean deployment-facing story initially rested almost entirely on one non-oracle Qwen3.5-9B result
- The BFCL Llama-4-Scout replication was directionally positive but weak
- The GPT-4o-mini sanity check was useful credibility context, but the repair gain was non-significant

What changed:
- A BFCL-200 non-oracle replication on `Qwen3.5-35B-A3B` now shows a second strong positive result under the same protocol: `0.635 -> 0.845`, `33/38` recoveries on the originally-correct slice, `0` observed harms, and only `1` unresolved repair target
- This materially reduces the literal single-model criticism and is strong enough to cite in the abstract, results, and conclusion
- The remaining caveat is narrower: the strongest successful non-oracle evidence is still concentrated in the `Qwen3.5` family rather than spread across clearly different families

Current evidence pressure points:
- Section 6.1
- Section 7.1
- Section 7.2

Residual risk:
- A reviewer can still ask for broader cross-family evidence, but this is no longer the same clean "one-model paper" objection

### P0.2 Synthetic drift still weakens ecological validity [Done]

This is the second major structural risk.

Why it matters:
- The drift operators are explicitly author-designed rather than mined from real changelog distributions
- The GitHub and Stripe examples help, but they do not yet validate that the synthetic perturbation mix reflects real deployment drift

Reviewer risk:
- Reviewers can accept synthetic perturbations if they are defended well, but not if broad deployment claims depend on them without validation

Current evidence pressure points:
- Section 5.3
- Section 8

What changed:
- The manuscript now includes a compact official-source audit across Notion and Stripe changelogs, mapping real vendor updates onto the paper's schema, example/documentation, and candidate-set change families
- The updated text also makes the remaining boundary explicit: broader structural refactors still fall outside the paper's semantics-preserving drift scope

Residual risk:
- This is still a hand-audited sanity check rather than a mined empirical distribution over change events, so the ecological-validity argument is now defensible but not exhaustive

### P0.3 Key explanatory ablations do not use the primary protocol [Done]

This is a real logic gap in the current narrative.

Why it matters:
- The paper's central claim is about non-oracle repair
- The drift-type ablation and component ablation are oracle-constrained upper bounds
- Reviewers may reject the implicit claim that those conclusions automatically transfer to the non-oracle setting

Current evidence pressure points:
- Table 4
- Table 5
- Sections 6.4 and 6.5

What changed:
- We verified that every baseline DICE task in the 300-example slice has exactly one candidate tool, and the repair targeter uses single-candidate fallback under predicted-tool repair
- That means the description-only and schema-only rows are already functionally identical under oracle-target and non-oracle targeting
- We then ran the missing distractor-bearing candidate-set-only row under the primary non-oracle protocol and used the already-existing primary DICE main result as the matching all-combined non-oracle row
- The repaired distractor rows drop slightly relative to the old oracle upper bounds, but the substantive conclusion is unchanged: candidate-set drift remains dominant and the non-oracle repaired scores stay close to the upper-bound values

Residual risk:
- The DICE drift-type section is now aligned with the primary protocol logic, but the component ablation in Section 6.5 remains an upper-bound mechanism analysis rather than a full non-oracle rerun

### P0.4 Headline gains still mix two effects [Done]

This issue is already acknowledged in the paper, but it still affects how the headline numbers read.

Why it matters:
- A nontrivial fraction of repaired gains are baseline patching rather than true drift recovery
- That means the headline repaired improvements are not pure evidence about interface drift robustness

Current evidence pressure points:
- Table 3
- Abstract
- Introduction contributions list
- Conclusion

Recommended action:
- Promote drift recovery to the foreground in the abstract and introduction
- Consider explicitly reporting a drift-recovery-focused metric or sentence before the full repaired score

### P0.5 Repairability gap under candidate-set confusion [Done]

This issue appeared in the Codex review and should be added to the working backlog.

Why it matters:
- The paper says candidate-set drift is the dominant failure mode
- But the validator triggers only on invalid or incomplete calls
- A valid call to the wrong distractor tool may not be repairable by the current pipeline at all

Reviewer risk:
- A reviewer may ask whether the proposed repair layer actually addresses the dominant failure mode, or only the subset of failures that manifest as invalid structure

Current evidence pressure points:
- Section 4.1
- Section 6.4

Recommended action:
- Add an error breakdown for candidate-set misses, separating invalid/incomplete calls from valid-but-wrong-tool selections that bypass validation

## Priority 1: High-Value Strengthening

### P1.1 Add a primary-run scoring audit [Done]

Why it matters:
- The current error analysis shows strict-match artifacts on non-primary models
- But the primary Qwen runs, which carry the paper, do not yet get the same audit depth

Recommended action:
- Manually inspect a sample or full set of primary unrecovered cases
- If feasible, add a lenient or normalized secondary metric

### P1.2 Expand related work [Done]

Current bibliography size is still a bit thin for the paper's scope.

Most obvious additions:
- constrained decoding / structured outputs relevant to forced-tool repair
- additional API/schema evolution literature beyond AutoGuard
- recent tool-calling robustness and evaluation work beyond the benchmark citations already present
- tool use in agentic or multi-agent settings if directly relevant

### P1.3 Clean up narrative structure [Done]

Why it matters:
- The paper repeatedly switches between non-oracle and oracle discussion
- The Llama non-oracle replication is currently placed in Discussion even though it is empirical evidence

Recommended action:
- Keep Section 6 fully results-oriented
- Move the Llama replication into Results
- Consolidate oracle-constrained context into one clearly marked upper-bound subsection

### P1.4 Reproducibility and artifact statement [Done]

Why it matters:
- The paper needs a clearer statement about code, configs, outputs, prompts, subsets, and release intent
- Both venues will expect this level of clarity even if artifact review is not formalized

Recommended action:
- Add a short reproducibility statement near the end of Experimental Setup or in Limitations
- Include release status such as "code/configs will be released upon acceptance" if immediate release is not possible

### P1.5 Temper the wording around determinism and harms [Done]

Recommended action:
- Replace "deterministic-style evaluation" with a more careful description of temperature-0 hosted inference
- Replace categorical "zero harms" phrasing with "observed zero harms in the reported slices" unless a confidence statement is added

## Priority 2: Important but Mostly Presentational

### P2.1 Explain score inconsistencies across protocol rows [Done]

Issue:
- Table 1 shows slightly different DICE Original scores for non-oracle and oracle rows

Why it matters:
- Even small unexplained differences create reproducibility doubt

Recommended action:
- Add one explicit sentence explaining that these come from separate historical runs or different shared-example subsets
- If possible, normalize the presentation so the difference is less visually jarring

### P2.2 Tighten the abstract [Done]

Issue:
- The abstract is dense with numbers and secondary findings

Recommended action:
- Reduce to the problem, method, two headline gains, and one key takeaway
- Push ablation nuance into the body

### P2.3 Fix the "two details" / three-bullet mismatch [Done]

Issue:
- Section 4.2 says "Two implementation details" and then lists three

Recommended action:
- Change the sentence to "Three implementation details"

### P2.4 Replace hardcoded section references [Done]

Issue:
- The paper still contains at least one hardcoded section number

Recommended action:
- Replace hardcoded section references with `\ref`

### P2.5 Improve figure polish [Pending]

Issue:
- The figures are serviceable but still look like draft plots

Recommended action:
- Increase label size
- use a cleaner publication palette
- ensure the bars and legends remain readable after LNCS scaling

## Priority 3: Venue-Specific Packaging

### P3.1 NGEN-AI packaging [Pending]

Required changes if we submit there:
- de-anonymize the manuscript
- add final author names and affiliations
- stay within the 16-page LNCS cap

Implication:
- We do not have much spare page budget for new experiments unless we also tighten the paper

### P3.2 IntelliSys packaging [Pending]

Required changes if we submit there:
- keep anonymization
- follow the double-blind instructions closely
- include any required declaration on generative AI use in the manuscript if applicable

Implication:
- The deadline is much tighter, so only modest paper changes are realistic before that venue

## Consolidated Action Order

If the goal is to maximize acceptance probability rather than freeze immediately, the strongest order is:

1. Decide whether to rerun the drift-type ablation under non-oracle conditions.
2. Add a compact scoring audit or lenient-evaluation check for the primary Qwen results.
3. Rewrite abstract, introduction, and conclusion to foreground drift recovery and narrow claims.
4. Restructure the results/discussion boundary and move the Llama replication into Results.
5. Add a reproducibility statement and expand related work.
6. Fix the mechanical issues: "Two details" wording, hardcoded section reference, and table/figure polish.
7. Apply venue-specific author/anonymization packaging.

## Freeze Threshold

The paper becomes much more defensible once the following are true:
- at least one additional non-oracle replication materially supports the main claim
- the paper either validates synthetic drift better or narrows its claim language
- the oracle-ablation mismatch is either fixed experimentally or clearly caveated
- the abstract and conclusion no longer let readers overinterpret repaired score gains as pure drift recovery

Status against that threshold:
- The first condition is met via the completed `Qwen3.5-35B-A3B` BFCL non-oracle replication.
- The second condition is met via the official-changelog audit plus narrower claim language.
- The third condition is met for the DICE drift-type section through singleton-candidate equivalence plus the missing non-oracle distractor-row rerun.
- The fourth condition is met through abstract/conclusion framing edits.

At this point the draft is in plausible submission shape for `NGEN-AI`, with the main remaining weaknesses shifted from `P0` structural blockers to more ordinary scope/polish concerns.
