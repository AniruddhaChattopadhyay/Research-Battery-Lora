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
- limitations updated to explicitly call out synthetic drift, single-model
  non-oracle evidence, and missing latency

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

---

## Priority 1 — Best Remaining Evidence Upgrades

### 1.1 Add one more non-oracle replication

Why:
- The main remaining substantive limitation is that the primary non-oracle story
  is still centered on Qwen3.5-9B.

Best next move:
- Add one more non-oracle candidate-retry run on an existing benchmark.
- Prefer BFCL, since that benchmark is thinner in the paper.

Current status:
- `tool-drift/configs/bfcl_stage3_200_llama4scout_non_oracle.yaml` has been added.
- Check whether the live run finished before deciding on manuscript updates.

### 1.2 Report wall-clock latency

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

### 1.3 BFCL component ablation

Why:
- The “validation matters more than the canonical card” result is stronger now
  that BFCL rows exist in the oracle upper-bound table, but a dedicated BFCL
  ablation remains useful support.

How:
1. Run `bfcl_ablation_card_only.yaml`
2. Run `bfcl_ablation_validation_retry.yaml`
3. Fold the result into the ablation section if it materially helps

---

## Priority 2 — Nice Strengthening If Time Allows

### 2.1 Broader paired significance testing

Why:
- The paper now reports paired significance for the main contrasts, but not for
  broader protocol-to-protocol comparisons.

### 2.2 Error-analysis appendix

Why:
- The current error analysis is solid, but an appendix table of unrecovered-case
  categories would make reviewer pushback easier to answer.

### 2.3 Frontier-model sanity check

Why:
- Even a small closed-model slice would help answer “does this matter on models
  people deploy?”

---

## Priority 3 — Optional Polish

### 3.1 Adversarial-drift discussion paragraph

Tie the paper more explicitly to description-surface manipulation / adversarial
tool routing work.

### 3.2 Formalize the drift operators in Section 3

Nice for reproducibility, but not necessary for a short submission.

### 3.3 Reconsider the method name

Only worth doing if we want the paper to foreground the minimal
validation-and-repair stack instead of `SchemaShield-Lite`.

---

## Recommended Order

If only a few more hours are available:
1. Check the Llama-4-Scout BFCL non-oracle run.
2. Re-run at least one primary non-oracle config for wall-clock latency.
3. Decide whether BFCL component ablation is still needed after those results.

At this point, additional work should be judged by how much it changes reviewer
confidence, not by whether it adds another experiment for its own sake.
