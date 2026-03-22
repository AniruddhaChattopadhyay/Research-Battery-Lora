# UQ-Edge: Does Quantization Break Model Confidence?
# A Systematic Study of Calibration and Uncertainty in Quantized Small Language Models

## Research Plan for NGEN-AI 2026

**Project codename**: UQ-Edge
**Conference**: NGEN-AI 2026 (Trento, Italy, Sep 1-4)
**Deadline**: May 25, 2026
**Target**: Short paper (8 pages, Springer LNCS)
**Conference track**: Trustworthy AI + Edge AI + LLMs

---

## 1. The Problem (In Plain English)

Small language models (1-3B parameters) are being deployed on phones, IoT devices, and edge hardware. To save memory and speed up inference, these models are **quantized** — their weights are compressed from 16-bit floating point down to 8-bit or 4-bit integers. This makes them smaller and faster, but what happens to their **confidence estimates**?

When a model says "I'm 90% sure this answer is correct," is that still reliable after quantization? If quantization makes models **overconfident** (saying 90% when they're only right 60% of the time) or **underconfident** (saying 30% when they're right 80% of the time), then any downstream system that relies on confidence — safety filters, routing decisions, selective prediction, human-AI collaboration — breaks silently.

**Nobody has systematically measured this for small models.** That's our contribution.

---

## 2. Literature Review & Gap Analysis

### 2.1 UQ Methods for LLMs — What Exists

UQ methods fall into four categories, ranging from cheap to expensive:

**A. Single-Pass Methods (cheap, ~0% overhead)**
- **Predictive Entropy**: Average entropy of token probability distributions. One forward pass.
- **Log-Likelihood**: Average token log-probabilities. One forward pass.
- **Mean Softmax Probability (MSP)**: Max probability of the predicted token, averaged.
- **Self-Certainty**: KL divergence between predicted distribution and uniform.

**B. Probe/Head Methods (~0% inference overhead, requires training)**
- **Semantic Entropy Probes (SEPs)** (2024): Linear probes on hidden states approximate semantic entropy from a single generation. Near-zero overhead.
- **SelectLLM** (NeurIPS 2025): Jointly trains a selection head that outputs confidence scores. Tested on 7B+ only.
- **CoCA** (March 2026): Uses GRPO RL to jointly optimize confidence calibration + accuracy. Tested down to Qwen2.5-1.5B. Finding: "confidence generation is less reliable on 1.5B model."
- **RAUQ** (May 2025): Uncertainty via recurrent attention weight aggregation. <1% overhead. Tested on 7B+ only.

**C. Sampling Methods (expensive, 200-800% overhead)**
- **Semantic Entropy** (Nature, June 2024): Generate multiple responses, cluster by meaning, compute entropy. 5-10x cost.
- **Self-Consistency**: Generate N responses, measure agreement. 3-5x cost.
- **CISC / RASC** (2025): Reduce sampling cost by ~40-70% while maintaining quality.
- **TokUR** (May 2025): Low-rank weight perturbation for epistemic uncertainty. 5x cost. Tested on Llama-3.2-1B (one of few papers testing small models).

**D. Ensemble/Bayesian Methods (expensive)**
- **LoRA Ensembles** (ICLR 2025 Workshop): M=5 LoRA adapters, 3.4M params each. Only on Mistral-7B.
- **PoLAR-VBLL** (June 2025): Bayesian low-rank adapters. Tested 350M to 27B. One of few testing small models.
- **DDD** (Feb 2026): Uses 1B draft models to audit 8B teachers. 42% FLOPs reduction vs TokUR.

**Key insight from Apple/MIT (NeurIPS 2024 Workshop)**: "High computational cost does not translate into significant performance gains" — simple single-pass features can match or outperform expensive sampling methods.

### 2.2 Calibration Studies — What We Know

| Study | Key Finding |
|-------|------------|
| **"Dunning-Kruger Effect in LLMs"** (Mar 2026) | Frontier models (Claude Opus 4.5, GPT-5.2, DeepSeek-V3.2) are systematically overconfident. Best ECE = 0.120 |
| **Biomedical study** (2025) | 84.3% of LLM responses are overconfident |
| **Medical QA** (2025) | Even top models show minimal confidence variation between right and wrong answers |
| **AbstentionBench** (Meta, 2025) | Abstention is "unsolved" — model scale has almost no effect. Reasoning fine-tuning actually HURTS abstention by 24% |
| **"Are LLM Decisions Faithful to Verbal Confidence?"** (Jan 2026) | Models verbalize uncertainty well but fail to act on it — fundamental disconnect |

**Post-hoc calibration methods**:
- **Temperature Scaling**: Single learned temperature parameter. Simple, widely used.
- **Adaptive Temperature Scaling (ATS)** (ICLR 2025): Per-token temperature. 10-50% better calibration.
- **THERMOMETER** (ICML 2024, MIT/IBM): Auxiliary calibration model trained across 57 tasks.
- **QA-Calibration** (ICLR 2025): Group-conditional calibration. Reduces ECE from 0.532 to 0.160 on MMLU.

**Critical finding**: RLHF **systematically causes overconfidence**. This is relevant because most deployed models are RLHF-tuned.

### 2.3 Quantization + Calibration — The Sparse Existing Work

This is where the gap becomes clear. **Only 3 papers directly study quantization's effect on calibration:**

| Paper | Venue | What It Found | Models Tested | Gap |
|-------|-------|--------------|---------------|-----|
| **Proskurina et al.** | NAACL 2024 | GPTQ 4-bit **decreases confidence on true labels** and **increases calibration error**, disproportionately affecting low-confidence predictions | OPT, BLOOM, Falcon (6.7B-40B) | Large models only, single quant method (GPTQ) |
| **"Quantized Can Still Be Calibrated"** (Zhong et al.) | ACL 2025 | Proposes soft-prompt tuning to recover calibration post-quantization. Derives analytic upper bound on calibration error (UBCE) | Large models | No small models, no systematic comparison across quant methods |
| **"Reliability Scaling Laws for Quantized LLMs"** | OpenReview, Oct 2025 | **Reliability peaks at 4-bit** — nonlinear behavior. Performance scales monotonically with bits, but reliability does NOT | GPT-2, OPT, BLOOM, Llama 2/3 (varies) | Tested large model families, not sub-3B SLMs |

**The "Reliability Scaling Laws" finding is crucial**: they discovered that going from FP16 → INT8 → INT4 does NOT monotonically degrade reliability. INT4 sometimes has BETTER reliability than INT8. This nonlinear behavior has never been verified on small models, where the effect could be amplified (fewer parameters = each bit matters more).

### 2.4 Quantization Benchmarks for SLMs — What They Missed

| Benchmark | What It Measures | What It Doesn't Measure |
|-----------|-----------------|------------------------|
| **SLMQuant** (ACM 2025) | Accuracy, memory, throughput for quantized SmolLM2-135M and Qwen2.5-0.5B | **No calibration, no UQ, no confidence metrics** |
| **LM-Polygraph** (TACL 2025) | 15+ UQ methods compared across multiple benchmarks | **Only 7B+ models, no quantized models** |
| **HELM** (Stanford) | Broad LLM evaluation including calibration | **No systematic quantization study** |

### 2.5 UQ Surveys — Identified Open Challenges

The **ACM Computing Surveys 2025** UQ survey (Shorinwa et al.) and the **KDD 2025** UQ survey (Liu et al.) both explicitly identify as open challenges:

1. **UQ under quantization** — "virtually unexplored"
2. **Edge/on-device UQ** — no solutions proposed
3. **UQ for small models** — most work targets 7B+
4. **Decoding strategy effects** on uncertainty
5. **Benchmark scarcity** for UQ evaluation

---

## 3. The Gap (Precisely)

No existing work provides a systematic study across all three dimensions:

```
1. MULTIPLE SMALL MODELS (sub-3B parameters)
   SmolLM2, Qwen2.5, Phi-4 mini, Gemma, TinyLlama, Llama-3.2

2. MULTIPLE QUANTIZATION LEVELS
   FP16 (baseline) → INT8 → INT4 (and potentially NF4, GPTQ, AWQ variants)

3. MULTIPLE UQ/CALIBRATION METRICS
   ECE, Brier Score, AUROC for selective prediction, reliability diagrams
```

| Paper | Small models? | Multiple quant levels? | Calibration metrics? | All three? |
|-------|:------------:|:---------------------:|:-------------------:|:----------:|
| SLMQuant | Yes (135M-0.5B) | Yes (W4A8, W8A8) | **No** | No |
| LM-Polygraph | No (7B+) | No | Yes (15+ methods) | No |
| Proskurina (NAACL 2024) | No (6.7B+) | Single (GPTQ 4-bit) | Yes (ECE) | No |
| "Quantized Can Still Be Calibrated" | No (large) | Yes | Yes | No |
| "Reliability Scaling Laws" | Partial (GPT-2) | Yes (2/3/4/8-bit) | Yes | **Close, but not SLMs** |
| CoCA | Partial (1.5B-7B) | No | Yes | No |
| **UQ-Edge (ours)** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## 4. Our Contribution: UQ-Edge

### 4.1 Research Questions

**RQ1**: Does quantization degrade calibration in small language models, and how does this vary across model architectures?

**RQ2**: Is the "reliability peaks at 4-bit" phenomenon (found for large models) also present in sub-3B models?

**RQ3**: Which UQ methods remain reliable after quantization? Do cheap single-pass methods (entropy, MSP) degrade faster or slower than expensive methods (sampling, ensembles)?

**RQ4**: Can simple post-hoc calibration (temperature scaling) recover calibration lost during quantization, or is a more sophisticated fix needed?

**RQ5**: What is the practical overhead of UQ methods on resource-constrained hardware?

### 4.2 Experimental Design

**Models** (selected for architectural diversity and edge relevance):

| Model | Params | Architecture Notes | Why Include |
|-------|--------|-------------------|-------------|
| SmolLM2-135M | 135M | NoPE (no positional encoding), YaRN, GQA | Smallest edge LLM, unique architecture |
| SmolLM2-1.7B | 1.7B | Same architecture family, larger | Scale comparison within family |
| Qwen2.5-0.5B | 0.5B | RoPE, GQA, standard transformer | Different architecture from SmolLM2 |
| Qwen2.5-1.5B | 1.5B | Same family, larger | Scale comparison within family |
| Qwen2.5-3B | 3B | Same family, upper bound | Largest "small" model |
| TinyLlama-1.1B | 1.1B | Llama architecture | We already have experience with this |
| Llama-3.2-1B | 1B | Latest Llama small | Industry standard small model |
| Llama-3.2-3B | 3B | Latest Llama small | Industry standard, upper bound |

**Quantization Levels**:

| Level | Method | Bits | Notes |
|-------|--------|------|-------|
| FP16 (baseline) | None | 16 | Full precision reference |
| INT8 | bitsandbytes LLM.int8() | 8 | Standard 8-bit |
| NF4 | bitsandbytes NF4 (QLoRA-style) | 4 | Most common 4-bit for edge |
| GPTQ-4bit | GPTQ | 4 | Post-training quantization |
| AWQ-4bit | AWQ | 4 | Activation-aware quantization |

This gives us **8 models × 5 quant levels = 40 configurations** to evaluate.

**UQ Methods** (selected for cost diversity):

| Method | Category | Cost | Needs Logits? |
|--------|----------|------|---------------|
| Mean Softmax Probability (MSP) | Single-pass | 1x | Yes |
| Predictive Entropy | Single-pass | 1x | Yes |
| Token Log-Likelihood | Single-pass | 1x | Yes |
| Verbalized Confidence | Prompting | ~1x | No |
| Temperature Scaling | Post-hoc | 1x (+ calibration set) | Yes |
| Self-Consistency (k=5) | Sampling | 5x | No |

We use 6 methods: 3 cheap single-pass, 1 prompting-based, 1 post-hoc fix, 1 expensive sampling baseline.

**Benchmarks**:

| Benchmark | Domain | Size | Why This One |
|-----------|--------|------|-------------|
| **TriviaQA** | Factual QA | 1,000 (subset) | Standard UQ benchmark, factual recall |
| **MMLU** | Multi-domain knowledge | 1,000 (subset) | Broad coverage, mix of easy/hard |
| **GSM8K** | Math reasoning | 1,319 | Tests reasoning under quantization |
| **TruthfulQA** | Truthfulness | 817 | Tests whether overconfidence increases |
| **CommonsenseQA** | Common sense | 1,221 | Tests everyday reasoning calibration |

### 4.3 Metrics

**Primary metrics**:

| Metric | What It Measures | How to Interpret |
|--------|-----------------|------------------|
| **ECE (Expected Calibration Error)** | Mismatch between predicted confidence and actual accuracy | Lower = better calibrated. 0 = perfectly calibrated |
| **AUROC (for selective prediction)** | Can the model's uncertainty correctly rank correct vs incorrect predictions? | Higher = better. 1.0 = perfect uncertainty ranking |
| **Brier Score** | Combined measure of calibration + discrimination | Lower = better. Decomposes into calibration + refinement |
| **Reliability Diagram** | Visual: predicted confidence (x) vs actual accuracy (y) | Diagonal = perfect calibration. Above diagonal = overconfident |

**Secondary metrics**:
- **Accuracy** (to track task performance degradation from quantization)
- **ECE delta** (change in ECE from FP16 baseline — isolates quantization effect)
- **AUROC delta** (same, for selective prediction)
- **Inference latency** (ms per sample, to show practical cost of UQ methods)
- **Memory footprint** (GB, per quantization level)

### 4.4 Experimental Protocol

**Phase 1 — Baseline Collection** (~4-6 hours on A100):
For each of the 40 model×quant configurations:
1. Load model at specified quantization level
2. Run on all 5 benchmarks
3. For each question, record:
   - Model answer (greedy decoding)
   - Whether answer is correct
   - Full logit distribution (for MSP, entropy, log-likelihood)
   - Verbalized confidence (prompt: "How confident are you? Give a percentage.")
4. Save all raw data

**Phase 2 — UQ Evaluation** (fast, CPU-only post-processing):
1. Compute all 6 UQ method scores from saved data
2. Compute ECE, AUROC, Brier Score per (model, quant, UQ method, benchmark)
3. Fit temperature scaling on calibration split (20%), evaluate on test split (80%)
4. Generate reliability diagrams
5. Compute all deltas from FP16 baseline

**Phase 3 — Self-Consistency** (~6-8 hours on A100):
For a subset of configurations (best and worst from Phase 2):
1. Generate 5 responses per question with temperature sampling
2. Compute self-consistency agreement
3. Compare against single-pass methods

**Phase 4 — Analysis & Visualization** (CPU-only):
1. Heatmaps: ECE across (models × quant levels)
2. Reliability diagrams: before vs after quantization
3. "Does temperature scaling fix it?" comparison
4. AUROC curves for selective prediction
5. Latency/memory breakdown

---

## 5. Expected Results & Claims

### 5.1 Primary Claim

**Quantization degrades calibration in small language models, and the degradation is non-monotonic — 4-bit quantization can be MORE or LESS reliable than 8-bit depending on the model architecture, mirroring the "reliability scaling law" finding but with potentially amplified effects in small models.**

### 5.2 Expected Findings (Hypotheses)

| Hypothesis | Based On | Testable? |
|-----------|----------|-----------|
| H1: INT4 increases ECE relative to FP16 by 5-20% for most SLMs | Proskurina (NAACL 2024) found this for large models | Direct measurement |
| H2: The reliability-peaks-at-4-bit phenomenon appears in some but not all SLM architectures | "Reliability Scaling Laws" paper | Direct measurement |
| H3: Simple single-pass methods (entropy, MSP) degrade proportionally with quantization — they don't break catastrophically | Logit distributions shift smoothly under quantization | Direct measurement |
| H4: Temperature scaling partially recovers calibration (50-80% of the ECE gap) | "Quantized Can Still Be Calibrated" (ACL 2025) showed this for large models | Direct measurement |
| H5: Verbalized confidence is MORE robust to quantization than logit-based confidence | Verbalized confidence uses generated text, not raw logits | Direct comparison |
| H6: Smaller models (135M-500M) are more affected by quantization than 1-3B models | Fewer parameters = each quantized weight matters more | Scale analysis |

### 5.3 Example Results Table (Expected Shape)

| Model | Quant | Accuracy | ECE (MSP) | ECE (Entropy) | ECE (Verbal) | ECE (TempScale) | AUROC |
|-------|-------|----------|-----------|---------------|-------------|-----------------|-------|
| Qwen2.5-1.5B | FP16 | 58.2% | 0.142 | 0.138 | 0.165 | 0.091 | 0.721 |
| Qwen2.5-1.5B | INT8 | 57.8% | 0.158 | 0.151 | 0.168 | 0.098 | 0.715 |
| Qwen2.5-1.5B | NF4 | 55.1% | 0.203 | 0.189 | 0.172 | 0.125 | 0.688 |
| Qwen2.5-1.5B | GPTQ-4 | 54.7% | 0.215 | 0.198 | 0.170 | 0.131 | 0.681 |

(Numbers are illustrative — actual values from experiments)

---

## 6. Paper Outline

### Title Options
1. "UQ-Edge: How Quantization Affects Uncertainty and Calibration in Small Language Models"
2. "Does Quantization Break Confidence? A Systematic Study of Calibration in Edge-Scale Language Models"
3. "Trust but Verify: Calibration Degradation Under Quantization for On-Device Language Models"

### Structure (8 pages, Springer LNCS)

**1. Introduction** (1 page)
- Edge LLMs are everywhere (Gartner: 3x more SLMs than LLMs by 2027)
- Quantization is mandatory for edge deployment
- Confidence estimates drive safety-critical decisions
- Nobody measured whether quantization breaks these estimates for small models
- Our contribution: first systematic study across 8 models × 5 quant levels × 6 UQ methods

**2. Related Work** (1 page)
- UQ for LLMs (surveys, methods taxonomy)
- Quantization benchmarks (SLMQuant — accuracy only)
- Quantization + calibration (3 papers, all large models)
- Position our work at the intersection

**3. Experimental Setup** (1.5 pages)
- Models, quantization methods, UQ methods, benchmarks, metrics
- Clear tables for reproducibility

**4. Results** (2.5 pages)
- RQ1: Calibration degradation heatmaps across models × quant levels
- RQ2: Nonlinear reliability behavior in small models
- RQ3: Which UQ methods survive quantization?
- RQ4: Temperature scaling as a fix — how far does it go?
- RQ5: Practical overhead measurements

**5. Discussion & Recommendations** (1.5 pages)
- Practical guidance: "If you're deploying model X at INT4, use UQ method Y and apply temperature scaling"
- Limitations (simulated edge, not real device measurements)
- Future work (learned calibration heads, on-device validation)

**6. Conclusion** (0.5 pages)

---

## 7. What We Reuse from BatteryLoRA

| Component | How We Reuse It |
|-----------|----------------|
| **Model loading + quantization** (model_utils.py) | BitsAndBytes quantization setup transfers directly |
| **Colab A100 workflow** (run_all.py) | Batch experiment runner, Drive checkpointing |
| **Plotting utilities** (plot_results.py) | Heatmaps, comparison charts |
| **Config system** (config.py) | Dataclass-based experiment configs |
| **uv package management** | Same tooling |
| **TinyLlama experience** | Already tested this model extensively |

---

## 8. Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| **1** | Mar 22 - Mar 28 | Set up project structure. Implement model loading for all 8 models at all 5 quant levels. Verify each config runs on Colab A100. Implement UQ metric computation (ECE, AUROC, Brier, reliability diagrams). |
| **2** | Mar 29 - Apr 4 | Run Phase 1: all 40 configurations × 5 benchmarks. Save all logits and predictions. This is the most GPU-intensive phase (~4-6 hours). Implement verbalized confidence prompting. |
| **3** | Apr 5 - Apr 11 | Run Phase 2: compute all UQ scores and metrics from saved data (CPU-only, fast). Generate initial heatmaps and reliability diagrams. **GO/NO-GO checkpoint**: do we see meaningful ECE differences across quant levels? |
| **4** | Apr 12 - Apr 18 | Run Phase 3: self-consistency experiments on select configs. Implement and evaluate temperature scaling fix. Run ablations (different quant methods for same bit-width). |
| **5** | Apr 19 - Apr 25 | Complete analysis. Generate all figures and tables. Start writing: Introduction, Related Work, Experimental Setup. |
| **6** | Apr 26 - May 2 | Write: Results, Discussion. |
| **7** | May 3 - May 9 | Complete first draft. Internal review. |
| **8** | May 10 - May 16 | Run any gap-filling experiments. Revise based on internal review. |
| **9** | May 17 - May 25 | Final polish, Springer LNCS formatting, references. Submit by May 25. |

### Key Milestones
- **Apr 4**: All raw data collected (Phase 1 complete)
- **Apr 11**: **GO/NO-GO checkpoint** — we see meaningful calibration differences (ECE delta > 0.03 between FP16 and INT4)
- **Apr 25**: All experiments complete, writing begins
- **May 9**: First complete draft
- **May 25**: Submit

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Quantization barely affects calibration** — ECE delta is tiny (<0.02) | Low-medium | High (no finding) | Still publishable as a "good news" result: "you can safely quantize." Shift framing to practical guidelines. Also test extreme quant (INT3/INT2) where effects should be larger. |
| **Some models don't load on Colab A100** — OOM at FP16 | Low | Low | 3B models at FP16 use ~6GB, well within 40GB. If issues, drop FP16 for 3B and use INT8 as baseline. |
| **SLMQuant team publishes a calibration extension** | Low | Medium | Our study is broader (more models, more UQ methods, more quant methods). Differentiate on coverage. |
| **Colab session timeouts** | Medium | Low | Phase 1 is the bottleneck. Break into per-model runs. Save intermediate results to Drive. Phase 2-4 need no GPU. |
| **Verbalized confidence is incoherent for 135M model** | High | Low | Drop verbalized confidence for smallest models. Note as a finding: "models below X params cannot meaningfully self-report confidence." |

### GO/NO-GO Checkpoint: April 11

After Phase 2 metrics are computed:
- **GO**: ECE delta (FP16 → INT4) > 0.03 on at least 3 models across at least 2 benchmarks. Or: AUROC drops by > 0.02. Or: nonlinear reliability behavior observed.
- **SOFT GO**: Effects are small but consistent across models → reframe as "quantization is surprisingly safe for calibration" (still publishable as a practical finding).
- **NO-GO**: Results are completely random with no pattern → pivot to a different research question (e.g., federated LoRA privacy study using existing infrastructure).

---

## 10. Required Resources

| Resource | Cost | Notes |
|----------|------|-------|
| Colab Pro (A100) | Already have | ~12-16 hours total GPU time |
| Model weights | Free | All from HuggingFace |
| Benchmark datasets | Free | TriviaQA, MMLU, GSM8K, TruthfulQA, CommonsenseQA — all public |
| Springer LNCS template | Free | Overleaf |

---

## 11. Why This Works for NGEN-AI 2026

**Conference fit**: Hits 3+ topic areas directly:
- **Trustworthy AI**: "Uncertainty estimation, calibration, and communicating confidence to users"
- **LLMs**: "Model compression, distillation, quantization, and sparsity for efficient deployment"
- **AI Systems/Edge**: "Edge AI and embedded AI for IoT, CPS, and real-time applications"
- **XAI**: "Evaluation and benchmarking of explanations (faithfulness, robustness, usefulness)"

**Contribution type**: Empirical benchmark/evaluation study — the most achievable paper type for a first submission. The contribution is the systematic measurement itself, not a new algorithm.

**Practical value**: Produces a concrete lookup table — "if you deploy model X at INT4, your calibration degrades by Y%, but temperature scaling recovers Z% of it." Practitioners can use this directly.

---

## 12. Key References

### Quantization + Calibration (Must-Cite)
1. Proskurina et al. "When Quantization Affects Confidence" — NAACL 2024 Findings
2. Zhong et al. "Quantized Can Still Be Calibrated" — ACL 2025
3. "Reliability Scaling Laws for Quantized LLMs" — OpenReview, Oct 2025

### UQ Methods
4. Semantic Entropy — Farquhar, Kossen, Kuhn, Gal — Nature, June 2024
5. Semantic Entropy Probes — 2024
6. TokUR — May 2025 (tested on Llama-3.2-1B)
7. DDD (Data-Diverse Drafts) — Feb 2026
8. LoRA Ensembles — ICLR 2025 Workshop
9. PoLAR-VBLL — June 2025

### Calibration Methods
10. Adaptive Temperature Scaling (ATS) — ICLR 2025
11. THERMOMETER — ICML 2024
12. QA-Calibration — ICLR 2025
13. CoCA — March 2026

### UQ Surveys
14. Shorinwa et al. UQ Survey — ACM Computing Surveys, Vol. 58, Sep 2025
15. Liu et al. UQ and Confidence Survey — KDD 2025
16. "Reasoning on a Budget" Survey — Jul 2025

### Calibration Studies
17. "Dunning-Kruger Effect in LLMs" — March 2026
18. AbstentionBench — Meta, 2025
19. "Know Your Limits" Abstention Survey — TACL 2025
20. "Are LLM Decisions Faithful to Verbal Confidence?" — Jan 2026

### SLM Quantization
21. SLMQuant — ACM 2025
22. On-Device LLMs State of the Union — 2026

### Benchmarking
23. LM-Polygraph — TACL 2025
24. HELM — Stanford

---

## 13. Glossary (For Team Reference)

| Term | Meaning |
|------|---------|
| **Calibration** | How well a model's stated confidence matches its actual accuracy. A well-calibrated model that says "80% confident" should be correct ~80% of the time. |
| **ECE (Expected Calibration Error)** | Main calibration metric. Groups predictions by confidence level, measures gap between average confidence and average accuracy per group. 0 = perfect. |
| **AUROC** | Area Under the ROC curve. Measures how well uncertainty scores separate correct from incorrect predictions. 1.0 = perfect separation. |
| **Brier Score** | Mean squared error between predicted probability and actual outcome (0 or 1). Lower = better. Captures both calibration and discrimination. |
| **Reliability Diagram** | Plot with predicted confidence on x-axis, actual accuracy on y-axis. Perfect calibration = diagonal line. Above diagonal = overconfident. |
| **Quantization** | Reducing the numerical precision of model weights (e.g., from 16-bit float to 4-bit integer) to reduce memory and speed up inference. |
| **GPTQ** | Post-Training Quantization method using approximate second-order information. Popular for 4-bit quantization. |
| **AWQ** | Activation-Aware Weight Quantization — protects salient weights based on activation magnitude. |
| **NF4** | 4-bit NormalFloat quantization from bitsandbytes. Used in QLoRA. Information-theoretically optimal for normally distributed weights. |
| **Temperature Scaling** | Post-hoc calibration method. Divides logits by a learned scalar T before softmax. T > 1 reduces confidence (fixes overconfidence). T < 1 increases confidence. |
| **Selective Prediction** | The model can "abstain" (refuse to answer) on uncertain inputs. Quality measured by accuracy-coverage curves. |
| **MSP (Mean Softmax Probability)** | Simplest uncertainty metric: the model's confidence in its own prediction. High MSP = confident. |
| **Predictive Entropy** | Entropy of the probability distribution over next tokens. High entropy = uncertain (probability spread across many tokens). |
