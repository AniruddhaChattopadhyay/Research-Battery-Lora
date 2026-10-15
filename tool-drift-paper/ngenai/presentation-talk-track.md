# NGEN-AI 2026 presentation talk track

**Slot:** Session 7, Thursday 3 September, 11:30-13:00. Each paper gets 20 minutes.

Plan a 16-minute talk plus about 4 minutes of live Q&A. The main talk is pages 1-13.
Pages 14-17 are backup slides for questions.

This is a speaking guide, not a script to memorize. The quoted lines use natural
spoken English on purpose, because you say them out loud.

**Where the notes live.** A condensed version of each page below is in the notes pane of
`NGEN-AI-2026_Session7_Paper15_Chattopadhyay_slides.pptx`, so PowerPoint Presenter View
shows the timing and the key beats. The `\note{}` blocks in `presentation.tex` hold the
same text, but they are *not* compiled into the PDF (that would need
`\setbeameroption{show notes}`), so the PDF has no notes pages. Present from the .pptx if
you want the notes on screen.

---

## Page 1 - Title

**Target time:** 0:00-0:40

**Say something like:**

> Hi everyone, I'm Aniruddha Chattopadhyay. This is joint work with Kanishka Halder and Partha Sarathi Roy.
>
> The observation behind this work is simple. A tool can keep doing exactly the same job, but the interface an agent reads can change underneath it. A description gets edited, a parameter gets renamed, or another similar-looking tool gets added. That alone can break a function call.
>
> So we ask whether we can catch and repair that kind of failure at runtime, without retraining the model.

**Move on with:** "Let me make precise what we mean by interface drift."

## Page 2 - Tool semantics stay fixed; interfaces move

**Target time:** 0:40-1:55

**Point to:** The stable intent on the left, then the three drift types on the right.

**Say something like:**

> Suppose the user's intent is: find the same customer orders as before. The API behaves exactly as it did, and the information the agent has to supply hasn't changed either.
>
> What changed is the surface the model reads. The description may still carry an old example. The schema may rename or reorder parameters. Or the catalog may now contain another tool that looks almost right.
>
> We call those description drift, schema drift, and candidate-set drift. The key property is on the bottom line: every edit is semantics-preserving. This is not a new task, not a tool outage, and not a change in what the tool does. The behavior is stable; the contract moved.
>
> That makes it a deployment problem. A model that worked yesterday can fail today because the catalog around it changed.

**Move on with:** "It helps to see exactly what that breaks."

## Page 3 - What interface drift breaks

**Target time:** 1:55-3:20

**Point to:** The Before/After signature first, then the two panels below it.

**Say something like:**

> Here it is concretely. The honest change is a rename: `customer_id` becomes `client_ref`, `start_date` becomes `from_date`. The tool still returns the same orders.
>
> Two things happen alongside it. The description still carries the old example with the old field names. And a near-duplicate tool, `search_order_archive`, has joined the catalog.
>
> On the left is what the model emits. It picks the wrong tool, reuses the field name it saw in the stale example, and never supplies the required `client_ref`.
>
> That call is invalid against the live schema, so a validator can see it. It reports the unknown tool, the unknown field, and the missing required field. With that feedback plus the current schema, one repair attempt produces the call on the right: right tool, renamed fields, same intent.
>
> To be clear, this example is constructed to show the mechanism. It is not a verbatim benchmark trace.

**Move on with:** "So how do we measure that across a whole benchmark?"

## Page 4 - Evaluation protocol

**Target time:** 3:20-4:30

**Point to:** The three-stage ladder on the left, then the non-oracle rules.

**Say something like:**

> Every example is scored three times. First on the original interface. Then we inject drift once and score it again. Then we let validation and repair act on that same drifted call.
>
> Scoring all three is what lets us separate two different quantities: how much drift breaks, and how much repair gets back.
>
> The primary protocol is deliberately non-oracle. The repair system never sees the benchmark's gold tool. It only gets what a deployed system would have: the user context, the model's own output, and the tools visible at runtime.
>
> And we do not retry until something works. We intervene only when validation flags an invalid or incomplete call.
>
> There are oracle-constrained runs later, but those are mechanism diagnostics and upper bounds, not deployment evidence.

**Move on with:** "Here is the recovery path itself."

## Page 5 - SchemaShield-Lite: validate, then repair once

**Target time:** 4:30-6:05

**Point to:** Walk left to right. Pause on each of the two notes at the bottom.

**Say something like:**

> The base model sees the user context and the drifted catalog and produces a tool call as usual.
>
> We validate that call against the live catalog and the current schema. If it is valid, it passes through unchanged. That matters for cost: we are not rewriting every output or adding a second model call everywhere.
>
> If it is invalid, the model gets structured feedback - what failed, the invalid call it produced, and the current schema - and one constrained repair attempt.
>
> One case needs an extra step. If the call omits the tool name, we cannot constrain a repair to a schema yet, because we don't know which tool is meant. So we first retry selection over the visible tools, and only then repair.
>
> And it really is one attempt: a 512-token repair budget, no open-ended retry loop, and no training anywhere in this.

**Move on with:** "We ran that same loop on two quite different function-calling settings."

## Page 6 - Benchmarks and model

**Target time:** 6:05-7:00

**Point to:** DICE first, then BFCL. Do not read every bullet.

**Say something like:**

> DICE-300 gives us multi-turn conversational examples, where the information the call needs is spread across the dialogue. We apply mild drift there.
>
> BFCL-200 is single-turn and spans a broader range of APIs, with a stronger candidate-set change - three distractor tools.
>
> So they stress different things: DICE stresses dispersed conversational context, BFCL stresses breadth and catalog ambiguity.
>
> The primary model is Qwen3.5-9B, at temperature zero. Hosted inference can still be a little nondeterministic even at zero, which is why the slices and the drift selection are fixed by seed.

**Move on with:** "With that setup, here is the main result."

## Page 7 - Repair recovers what drift breaks

**Target time:** 7:00-9:00

**Point to:** The brick bars, then the navy bars, then the three figures on the right.

**Say something like:**

> Gray is the original interface, brick red is after drift, navy is after repair.
>
> On DICE, accuracy goes 75.7, down to 71.7 under drift, then up to 81.3 with repair. On BFCL the drift damage is far larger: 79.0 drops to 58.0, and repair brings it back to 80.5.
>
> The numbers on the right are the ones I care about most. They count only examples that were correct before drift and then failed because of drift. We recover 19 of 22 on DICE and 33 of 45 on BFCL.
>
> And we observed zero repair harms on originally-correct calls in these slices. I want to phrase that carefully: that is an empirical result for this evaluation, not a guarantee that repair can never break a working call.
>
> One thing that can look odd: repaired accuracy can exceed the original. That happens because the repair path also fixes a few calls the base model already got wrong before any drift.

**Move on with:** "The obvious objection is that any second attempt would do this."

## Page 8 - A second attempt does not explain the gain

**Target time:** 9:00-10:20

**Point to:** The three gray bars, then the navy bar.

**Say something like:**

> We tested that directly with two baselines. Naive retry just repeats the identical drifted prompt. Tool-list retry goes further: it tells the model the call was invalid and lists the available tool names - but it withholds the failed call and the validation errors.
>
> On DICE those reach 72.7 and 75.3, against 81.3 for the full stack. On BFCL, 61.0 and 62.5, against 79.5.
>
> Note the axis starts at 50, and these are separate baseline runs, which is why the drifted anchor differs slightly from the previous slide.
>
> The point is that the gain is not just paying for another inference call. It is the structured feedback. Knowing which tools exist is not enough; the model needs to know what specifically failed.

**Move on with:** "Next: which of the three drift types is doing the damage?"

## Page 9 - Candidate-set drift causes most of the damage

**Target time:** 10:20-11:35

**Point to:** The two small gray bars, then the two long bars.

**Say something like:**

> In this DICE ablation, rewording the description costs 1.3 points. Renaming schema fields costs 0.7. Both are almost noise.
>
> Adding a single overlapping distractor tool costs 4.7 points - which is nearly the entire 5.0-point drop we get from all three combined.
>
> I want to be careful here. This is one model and one slice, and I would not claim this ordering holds everywhere. But the transferable lesson is real: catalog growth deserves to be tested separately from description and schema revisions. A team can keep every individual schema valid and still make tool selection much harder just by adding similar-looking tools.

**Move on with:** "Are these failures even visible to a validator?"

## Page 10 - Failures are mostly visible, but repair is not free

**Target time:** 11:35-13:00

**Point to:** The two counts, then move to the cost table.

**Say something like:**

> Mostly yes. On the originally-correct slice, validation sees 20 of 22 DICE drift misses and all 45 on BFCL. So the repair layer is aimed at the dominant failure mode, not an edge case.
>
> The two exceptions matter though. They are schema-valid but semantically wrong - the model produced a well-formed call to the wrong effect. A schema validator structurally cannot catch that. That is the blind spot of this whole approach.
>
> And the reliability is not free. The repair stage fires on 21 percent of DICE examples and 36.5 percent of BFCL examples, and adds 32.6 to 47.1 percent more tokens in these runs. Fifteen tool names on DICE and nine on BFCL never resolve at all.
>
> So the deployment question isn't just "does repair improve accuracy". It's whether that gain is worth this trigger rate for your application.

**Move on with:** "And that trade-off shifts a lot when you change the model."

## Page 11 - Portability tracks tool-name resolution

**Target time:** 13:00-14:20

**Point to:** The two Qwen rows, then the two below the gap.

**Say something like:**

> The larger Qwen model replicates the effect strongly: 63.5 to 84.5, a 21-point gain.
>
> Llama-4-Scout gains only 3 points, and the reason is in the last column. It leaves 43 tool names unresolved. If the model never names a visible tool, the schema-constrained repair has nothing to attach to - which is exactly the bottleneck the bottom line names.
>
> GPT-4o-mini starts from much less drift damage, at 84 percent, and its 2-point gain is not statistically significant. We treat that run as a sanity check, not as confirmation.
>
> So this is not a model-independent layer. The strongest non-oracle evidence stays inside the Qwen family, and I'd rather say that plainly than oversell it.

**Move on with:** "Let me be explicit about the rest of the limits."

## Page 12 - Limitations

**Target time:** 14:20-15:15

**Point to:** Left column, then right column. Keep this brisk.

**Say something like:**

> Two honest limits. First, the evidence is narrow: the strongest non-oracle results sit in one model family, Llama is bottlenecked on names, and the GPT-4o-mini gain isn't significant.
>
> Second, the setting is controlled. We designed the drift operators ourselves. They are deployment-motivated, and we did audit 11 real Notion and Stripe changelog items against them - six resemble our description and schema operators, three look like candidate-set changes - but we did not derive the operators from real changelogs, and that small audit does not tell you how often real API changes look like ours.
>
> The drift-type and component ablations are oracle-constrained, so treat them as upper bounds on the mechanism. And we score one call per example rather than executing a full sequential tool plan.
>
> One thing cuts in our favor: strict exact matching makes the reported recovery rates conservative.

**Move on with:** "So, the practical version of all this."

## Page 13 - Summary

**Target time:** 15:15-16:00

**Point to:** The headline, then the three numbered actions.

**Say something like:**

> The takeaway is that tool interfaces should be treated as evolving contracts, not as static prompt text.
>
> The recipe is: validate calls against the live catalog and schema; if the tool name is missing, retry selection over the tools that are actually visible; then allow one constrained repair using structured feedback.
>
> Alongside that, monitor three things: catalog growth, token overhead, and whether your model tends to omit tool names. It works when the failure is visible and the name resolves. It does not help valid-but-wrong calls.
>
> The repository has the code, prompts, fixed slices, and example-level traces. Thank you - happy to take questions.

**Stop here for the main talk.** Pages 14-17 are for Q&A.

---

# Backup-slide commentary

## Page 14 - Backup: statistical detail

**Use when asked:** "How confident are you the recovery is real?" or "Are the differences significant?"

**Say something like:**

> Same primary non-oracle results, with bootstrap intervals and paired McNemar tests. The repaired-versus-drifted change is large and the paired tests give very small p-values on both benchmarks.
>
> The caveat is in the sentence below: these intervals support the large recovery effect, but they do not license strong claims about small gaps between nearby runs. I would not use this table to over-read a small oracle-versus-non-oracle difference.

## Page 15 - Backup: validation drives the recovery

**Use when asked:** "Which component actually does the work?" or "Is the canonical schema card enough?"

**Say something like:**

> This is an oracle-constrained mechanism check, so read it as an upper-bound diagnostic rather than a deployment result.
>
> Giving the model the canonical card by itself recovers nothing - drifted and repaired are identical. The jump appears when validation feedback and retry are active, even with the card removed from the repair prompt.
>
> So the active ingredient is not showing the schema again. It is identifying what failed and giving one targeted chance to fix it. The card still earns its keep, but earlier - as drift prevention at the initial prediction step, where on BFCL it lifts the drifted score from 0.640 to 0.810.

## Page 16 - Backup: residual failure modes

**Use when asked:** "What still fails?" or "Does exact match underestimate performance?"

**Say something like:**

> The non-oracle residuals are missing fields that stay missing after repair, tool names that never resolve, and the two DICE calls that are schema-valid but wrong.
>
> Separately, we manually inspected 13 unrecovered cases on the non-primary models, and 11 were formatting or paraphrase mismatches rather than genuinely wrong behavior - things like `regular_season` versus `regular season`. So exact-match recovery is conservative for those cases. I'd keep that narrow though: it is a small inspected set, so it is evidence of an evaluation artifact, not a revised accuracy number.

## Page 17 - Backup: reproducibility

**Use when asked:** "Can we reproduce this?" or "What exactly is released?"

**Say something like:**

> The repository pins the benchmark slices and drift configurations, plus the prompts, scoring logic, normalized semantic matching, per-example outputs, validation traces, repair targets, token usage, and raw artifacts.
>
> The paper pins commit `fbac156`, so that is the snapshot to use when reproducing the reported numbers.

---

## Questions to expect

- **"Isn't this just retry with extra steps?"** Page 8. Both retry baselines stay far
  below the full stack, so a second attempt alone does not explain it.
- **"How do you know the drift is realistic?"** Page 12. Say plainly that the operators
  are author-designed and the 11-item changelog audit is a plausibility check, not a
  prevalence estimate.
- **"Why does repaired beat original?"** Baseline patching. Of 29 DICE gains, 19 are
  drift recovery and 10 patch pre-existing errors. On BFCL it is 33 and 12.
- **"What does it cost?"** Page 10. A 21 to 36.5 percent trigger rate and 32.6 to 47.1
  percent more tokens. Add that the single-rerun latency figure in the paper is not a
  systematic latency benchmark.
- **"Does this help against adversarial tool injection?"** Not tested. Say that the
  repair stack could even mask a malicious distractor tool, so deployments should
  provenance-check newly added tools.
- **"Why not just fix the descriptions?"** Page 9. Description drift was the *least*
  damaging operator here. The expensive one was catalog growth, which no amount of
  description hygiene addresses.

## Quick delivery reminders

- Do not read the slides line by line. Use them as the evidence while you explain.
- Slow down on pages 3, 5 and 7. They carry the concrete failure, the method, and the
  main result.
- On page 7, say "zero observed harms in these slices," never "repair cannot cause harm."
- On page 9, keep saying "in this model and this slice." Its bar labels are shortened:
  they mean description-only, schema-only, candidate-set-only, and all three combined.
- On pages 11 and 12, state the limits plainly. It makes the main result more credible.
- The charts on page 8 shorten the method name to "SchemaShield" to fit. Say the full
  name out loud.
- Page 13 shortens items 2 and 3. Say "then constrain one repair" and "model-specific
  name omission" out loud.
- If time runs short, compress pages 6 and 9. Do not rush page 7, the cost caveats on
  page 10, or the limits on page 12.
- Leave the last 4 minutes for live Q&A. The organizers ask for it explicitly.
