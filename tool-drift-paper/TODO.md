# Paper TODOs — Submission Push

As of Apr 13, the main protocol-validity issue is closed. The paper now uses the
non-oracle candidate-retry protocol as its primary story, includes confidence
intervals and figures, expands related work, and separates drift recovery from
baseline patching in the results and discussion.

The remaining work is now about strengthening evidence and polishing the final
submission package, not about fixing a broken core claim.

---

## Completed

### C.1 Oracle vs non-oracle repair evaluation

Completed and incorporated into `main.tex`.

Key outcome:
- DICE-300 non-oracle + candidate retry: `0.757 -> 0.717 -> 0.813`
- BFCL-200 non-oracle + candidate retry: `0.790 -> 0.580 -> 0.805`
- Zero repair harms still hold on the originally-correct slice
- Missing-tool-name retry reduced unresolved targets from `35 -> 15` on DICE and
  `34 -> 9` on BFCL

Paper impact:
- The non-oracle protocol is now the primary deployment-facing evidence.
- Oracle-constrained runs remain only as upper bounds and ablation context.

### C.2 Manuscript repositioning

Completed.

Included changes:
- title reframed around a non-oracle study of validation and repair
- abstract/introduction rewritten around the non-oracle protocol
- results split into primary non-oracle evidence vs oracle upper bounds
- limitations updated to explicitly call out synthetic drift, still-limited
  latency coverage, and the gap between the clean Qwen primary story and the
  weaker Llama non-oracle replication
- a compact decomposition table now separates drift recovery from baseline
  patching
- the discussion now includes adversarial-drift context, a formal drift-operator
  view, and real API-evolution examples from public GitHub/Stripe docs

### C.3 Presentation upgrades

Completed.

Included changes:
- confidence intervals added to the main results table
- figures included in the paper
- related work expanded beyond the original sparse bibliography
- the paper builds successfully via `latexmk`

### C.4 Paired significance on the main contrasts

Completed and incorporated into `main.tex`.

Current paired exact McNemar results on the final completed non-oracle runs:
- DICE drifted vs repaired: `p = 3.7e-9`
- BFCL drifted vs repaired: `p = 5.7e-14`
- DICE naive retry vs repaired: `p = 4.1e-9`
- BFCL naive retry vs repaired: `p = 1.1e-13`

### C.5 Broader protocol comparisons

Completed and incorporated into `main.tex`.

On shared examples, repaired oracle-vs-non-oracle differences are not
significant on the primary Qwen runs:
- DICE repaired oracle vs repaired non-oracle: `0.827` vs `0.813`, `p = 0.125`
- BFCL repaired oracle vs repaired non-oracle: `0.795` vs `0.805`, `p = 0.754`

### C.6 Second non-oracle replication

Completed and incorporated into `main.tex` as secondary evidence.

Current supporting run:
- `bfcl-live-20260413-202008-710245` (`Llama-4-Scout`, BFCL-200, non-oracle)

Key outcome:
- `0.520 -> 0.635 -> 0.665`
- exact McNemar drifted vs repaired: `p = 0.031`
- zero repair harms on the originally-correct slice
- but `43` unresolved repair targets and only `1/11` recoveries on the
  originally-correct slice

Paper impact:
- This reduces the literal “single non-oracle model” limitation.
- It does not replace Qwen3.5-9B as the clean primary evidence, because the run
  is dominated by missing-tool-name failures.

### C.7 Closed-model sanity check

Completed and incorporated into `main.tex` as a secondary credibility check.

Current supporting run:
- `bfcl-live-20260413-203305-358013` (`GPT-4o-mini`, BFCL-200, non-oracle)

Key outcome:
- `0.865 -> 0.840 -> 0.860`
- exact McNemar drifted vs repaired: `p = 0.125`
- zero repair harms and zero unresolved repair targets
- repair trigger rate only `2.0%`

Paper impact:
- This answers the reviewer-style question about whether the problem matters on a
  stronger closed model.
- The answer is nuanced: drift damage is much smaller, repair is directionally
  helpful but not significant on this slice, and the repair layer still appears
  safe.

---

## Priority 1 — Best Remaining Evidence Upgrades

### 1.1 Report wall-clock latency

Why:
- The paper now reports one real BFCL live latency measurement, but not yet a
  full sweep across both benchmarks/providers.

How:
1. Keep the current BFCL latency sentence in the paper.
2. If time permits, add a matching DICE latency rerun or a second-provider check.

Current status:
- Completed on a refreshed BFCL-200 live rerun with latency instrumentation.
- Current paper-ready numbers: original `8.8s`, drifted `9.3s`, repair `13.0s`,
  about `3.5s` expected added latency per example at a `27.0%` trigger rate.

### 1.2 BFCL component ablation

Completed in the paper already.

Current supporting runs:
- `bfcl-live-20260412-001432-682965` (`card_only`)
- `bfcl-live-20260412-002045-495346` (`validation_retry`)

Paper impact:
- BFCL rows are already present in the upper-bound component ablation table.
- This item no longer needs new runs unless we want a second replication.

---

## Priority 2 — Nice Strengthening If Time Allows

### 2.1 Error-analysis appendix

Why:
- The current error analysis is solid. A dedicated appendix would still be nice,
  but the main paper now includes a compact unrecovered-case summary table.

### 2.2 Second-provider or second-benchmark latency sweep

Why:
- The paper has one real BFCL latency measurement, but still not a broader
  deployment-facing latency sweep.

---

## Priority 3 — Optional Polish

### 3.1 Reconsider the method name

Only worth doing if we want the paper to foreground the minimal
validation-and-repair stack instead of `SchemaShield-Lite`.

---

## Recommended Order

If only a few more hours are available:
1. Decide whether a second latency sweep is worth the time.
2. Otherwise freeze the paper and move to submission packaging.
3. Only add more runs if they materially change the reviewer story.

At this point, additional work should be judged by how much it changes reviewer
confidence, not by whether it adds another experiment for its own sake.
