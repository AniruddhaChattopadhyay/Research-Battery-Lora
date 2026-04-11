# Paper TODOs — Post-Rewrite

After the Apr 11 rewrite, the paper reflects all 12 experiment results. These are
the remaining improvements, in priority order.

---

## Priority 1 — Must-Do Before Submission

### 1.1 Validate component ablation on BFCL

**Why:** The "canonical card doesn't matter" finding (Table 4) is currently
established on a single benchmark (DICE) and one model (Qwen 9B). A reviewer will
push on this. Running the same component ablation on BFCL-200 + Qwen 9B lets us
say "we verify this on both benchmarks."

**How:**
1. Copy `configs/dice_ablation_card_only.yaml` → `configs/bfcl_ablation_card_only.yaml`, point `data.bfcl_subset_path` at `data/bfcl_stage3_subset_200.json`, update drift config to medium severity (match BFCL main).
2. Same for `validation_retry` variant.
3. Run both via `scripts/run_pilot_bfcl.py`.
4. Add a row/column to Table 4 or add a new Table showing BFCL component ablation.

**Cost:** ~2 runs × 35 min = ~70 min wall clock, ~$1 API.

---

### 1.2 Generate figures

**Why:** The paper has zero figures. We already built
`scripts/generate_figures.py` but never ran it. Figures will make the key
findings instantly readable.

**How:** Three figures to add:

1. **Figure 1 (§4):** Pipeline diagram. Already have TikZ source in
   `generate_figures.py --tikz`. Just insert into §4.
2. **Figure 2 (§6.1):** Grouped bar chart of Orig/Drift/Naive/Repaired across
   models. Replaces visual content of Table 1.
   ```bash
   python scripts/generate_figures.py \
     --results "Qwen-9B:outputs/dice/dice-bench-live-20260409-200527-873158/dice_results.json" \
               "Qwen-35B:outputs/dice/dice-bench-live-20260411-004643-014074/dice_results.json" \
               "Llama-4:outputs/dice/dice-bench-live-20260411-014645-393246/dice_results.json" \
               "Mistral:outputs/dice/dice-bench-live-20260410-233844-888583/dice_results.json"
   ```
3. **Figure 3 (§6.3):** Drift-type ablation bar chart. Makes the candidate-set
   finding instantly visible.
   ```bash
   python scripts/generate_figures.py --drift-ablation \
     "Desc:outputs/dice/dice-bench-live-20260410-195355-530634/dice_results.json" \
     "Schema:outputs/dice/dice-bench-live-20260410-204147-107156/dice_results.json" \
     "Candidates:outputs/dice/dice-bench-live-20260410-211037-802999/dice_results.json" \
     "Combined:outputs/dice/dice-bench-live-20260410-214209-680674/dice_results.json"
   ```

**Cost:** ~30 min (matplotlib install + figure tweaking).

---

### 1.3 Add 95% confidence intervals to Table 1

**Why:** Makes the result look more rigorous. The summary JSONs already contain
`original_ci_95`, `drifted_ci_95`, `repaired_ci_95` — we just don't display them.

**How:** Edit `main.tex` Table 1 to show `0.750 [0.70, 0.80]` format for each
score. Bootstrap CI is already computed (seed=42) in `eval/metrics.py` and
stored in every summary.

**Cost:** ~20 min of table reformatting.

---

## Priority 2 — Strongly Recommended

### 2.1 Fix repair token tracking

**Why:** All summary JSONs currently report `total_repair_tokens: 0` because I
wired token tracking through the base/drift calls but forgot the repair call. §8
Limitations currently glosses over real repair overhead.

**How:** In `scripts/run_pilot_dice.py` (and mirror in `run_pilot_bfcl.py`),
modify `repair_call()` to extract `extract_usage(tool_call_payload)` and
`extract_usage(json_fallback_payload)` and return alongside the existing tuple.
Then accumulate into `repair_usage` in the `run()` loop. ~15-line change.

After fix, re-run just `dice_stage3_300_qwen9b.yaml` to get real repair token
numbers for the paper (~50 min).

**Cost:** ~15 min code + ~50 min re-run.

---

### 2.2 Investigate Mistral's "drifted > original" anomaly

**Why:** §6.5 currently speculates that Mistral's drifted score (0.777) beating
its original (0.740) is because description drift "resembles training
distribution." This is hand-wavy. Reading 5-10 concrete examples would either
confirm or reveal something more interesting.

**How:** Load the Mistral results JSON, find examples where
`original_match.matched = False` but `drifted_match.matched = True`, inspect the
original error categories (wrong tool? missing field? wrong value?), and see if
there's a pattern.

```python
import json
d = json.load(open("outputs/dice/dice-bench-live-20260410-233844-888583/dice_results.json"))
for r in d["results"]:
    if not r["original_match"]["matched"] and r["drifted_match"]["matched"]:
        print(r["id"], r["original_match"], r["prompt"][:80])
```

If the pattern is clean ("description drift adds explicit field hints"), that's
worth a sentence. If it's noisy, §6.5 needs softening.

**Cost:** ~30 min inspection + maybe a paragraph rewrite.

---

### 2.3 Add error analysis appendix for unrecovered cases

**Why:** §6.5 has a two-sentence treatment of the unrecovered drift misses on
non-9B models (20/25 on Qwen 35B, 12/16 on Llama-4, 8/12 on Mistral). A
one-paragraph appendix characterizing what the unrecovered cases have in common
would strengthen the paper.

**How:** For each of the three models, filter results to
`original=True, drift=False, repair=False`. Read the prompts, drifted tool
schemas, and repair outputs. Categorize into buckets (inference failure under
empty input / strict string-match artifact / truly wrong tool / etc.) and report
counts.

**Cost:** ~45 min.

---

## Priority 3 — Consider But Not Urgent

### 3.1 Reconsider the method name

**Why:** Given the component ablation finding, "SchemaShield-Lite" is arguably
misleading — the canonical card (the "schema shield") isn't doing the repair
work. The load-bearing component is validation-plus-retry.

**Options:**
- **Keep as-is.** Method-name inertia. Cheap.
- **Rename to "Validate-and-Repair" (VR).** Describes what actually works.
  Requires search-replace in the paper.
- **Reframe:** Define SchemaShield-Lite as the full stack, introduce "VR-Lite"
  as the minimal effective variant, recommend VR-Lite in Discussion.

**Recommendation:** Option 3 gives the reader both the history and the
simplification, and doesn't lose the brand equity of SchemaShield-Lite.

**Cost:** ~1 hour if we go with option 3 — mostly §4 rewriting.

---

### 3.2 Add an "adversarial drift" discussion paragraph

**Why:** The Faghih et al. paper (cited in §2) shows small description edits can
dramatically shift tool preference — i.e., tool descriptions are adversarial
surfaces. Our repair pipeline is relevant to that setting too, but we don't
mention it. A short paragraph in §7 (or a new §7.1) discussing adversarial
implications would broaden the paper's relevance.

**Cost:** ~15 min of writing.

---

### 3.3 Formalize drift operators in §3

**Why:** §3 currently lists drift types informally. A precise grammar would
make the work more reproducible and extensible, and gives the paper a bit more
math.

**How:** Define $\Delta$ as a composition of three operators $\delta_d,
\delta_s, \delta_c$ acting on $(d, s, C)$, and write each out precisely.

**Cost:** ~30 min. Low priority — paper reads fine without this.

---

## Priority 4 — Nice-to-Have

### 4.1 Self-hosted inference cross-check
Limitation §8 mentions that hosted inference may include provider-specific
behavior. A quick sanity check on one model via vLLM would address this.
**Cost:** Several hours of setup. Probably not worth it for a short paper.

### 4.2 More drift modes
Type narrowing, enum replacement, multi-tool refactors. §8 mentions these as
untested. Could add if time allows.
**Cost:** Several hours of drift operator implementation + new runs.

---

## Recommended Order

If time is tight, do **1.1, 1.2, 1.3, 2.1** in order. That gives the paper:
- Cross-benchmark validation of the component ablation
- Three figures
- Statistical rigor via CIs
- Real token overhead numbers

Total wall-clock: ~4-5 hours. Total API cost: ~$2.

Everything else is polish. The Priority 1 items alone make the paper materially
stronger for review.
