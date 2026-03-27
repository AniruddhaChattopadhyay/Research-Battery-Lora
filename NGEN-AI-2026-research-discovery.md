# Research Discovery: Feasible NGEN-AI 2026 Paper Directions

**Conference:** International Conference on Next Generation AI Systems (NGEN-AI 2026)  
**Venue:** Trento, Italy  
**Conference dates:** September 1-4, 2026  
**Verified submission deadline:** May 25, 2026  
**Target submission type:** Short paper (8 pages)  
**Prepared on:** March 27, 2026

---

## 1. Purpose of This Document

This is a **research discovery document**, not yet a full research plan. Its job is to narrow the space to directions that are:

- aligned with the NGEN-AI 2026 CFP,
- recent enough to matter in the 2025-2026 AI literature,
- feasible to prototype quickly on Colab Pro with A100/H100 access,
- differentiated enough to support a credible short paper.

The main output of this document is a recommended topic to convert into a full execution plan.

---

## 2. Verified CFP Snapshot

The current NGEN-AI website lists:

- **Submission deadline:** May 25, 2026
- **Notification:** July 15, 2026
- **Camera ready:** August 10, 2026
- **Conference:** September 1-4, 2026

This means the working timeline from March 27, 2026 is about **8 weeks**, not 1 month.

Primary tracks relevant to this search:

- Agentic AI
- Small & Large Language Models and Generative AI
- Explainable AI (XAI) & Transparency
- Trustworthy, Responsible & Sustainable AI
- MLOps, AI Engineering & Lifecycle Management
- Federated Learning

Source: <https://ngen-ai.org/>

---

## 3. Working Constraints

Assumptions used in topic selection:

- team is small and not deeply specialized in one narrow subfield,
- fastest path is a paper with strong engineering and evaluation rather than a large new model,
- experiments should run on **open models** with manageable GPU budgets,
- early signal should be available in **48-72 hours**,
- the safest target is an **8-page short paper** with clear empirical claims.

Practical implication:

- prefer **benchmark + stress setting + lightweight method** over large-scale pretraining,
- prefer **inference-time methods** or small fine-tunes over expensive RL,
- prefer settings with public code and public baselines.

---

## 4. Selection Criteria

Each candidate direction was judged on:

1. **Recency of predecessor work**  
   There should be direct anchors in 2025 or 2026, not only older 2023-2024 papers.

2. **Novelty window**  
   The paper cannot just reproduce an existing benchmark on a different model.

3. **Execution speed**  
   A basic baseline should be runnable quickly.

4. **Evaluation clarity**  
   The paper needs measurable claims, not only qualitative discussion.

5. **Conference fit**  
   The direction should map cleanly to NGEN-AI tracks.

---

## 5. Discovery Summary

Three directions emerged as the most credible:

1. **Robust tool calling under interface drift**
2. **Budget-aware online hallucination repair for RAG**
3. **Safety- and OOD-aware routing for multi-LLM systems**

A fourth direction, **personalized federated LoRA**, is publishable but slower and riskier for the available timeline.

The strongest overall recommendation is:

> **Primary recommendation:** Robust tool calling under interface drift  
> **Backup recommendation:** Budget-aware online hallucination repair for RAG

---

## 6. Shortlisted Direction A: Robust Tool Calling Under Interface Drift

### A.1 Core Problem

Tool-using LLMs are usually evaluated on fixed APIs, fixed tool descriptions, and mostly clean schemas. Real systems drift:

- parameter names change,
- descriptions become verbose or marketing-heavy,
- optional arguments become required,
- aliases appear,
- tool sets get larger and noisier,
- error messages vary by provider.

The result is a reliability gap between benchmarked tool use and deployed tool use.

### A.2 Why This Topic Is Timely

Very recent papers show that the field is active, but still fragmented:

- **Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use** (Feb 23, 2026) focuses on improving tool interfaces, especially for generalization to unseen tools.
- **ASA: Training-Free Representation Engineering for Tool-Calling Agents** (Feb 4, 2026) improves tool calling with inference-time activation steering.
- **Robust Tool Use via Fission-GRPO** (Jan 22, 2026) studies recovery from execution errors in multi-turn tool use.
- **DICE-BENCH** (Jun 28, 2025) shows single-turn function-calling benchmarks are unrealistic and introduces harder multi-round settings.
- **ComplexFuncBench** (Jan 17, 2025) studies multi-step and constrained function calling under long context.
- **Gaming Tool Preferences in Agentic LLMs** (May 23, 2025) shows tool descriptions can strongly bias model choice.

What is still missing is a compact, deployable answer to:

> How robust are open tool-calling models when interfaces drift after deployment, and can we recover reliability without retraining?

### A.3 Proposed Paper Angle

**Working title:**  
**SchemaShield: Robust Tool Calling Under Interface Drift via Canonicalization and Validator-Guided Repair**

Possible contribution:

- define a taxonomy of **interface drift** for tool-calling systems,
- build a stress-test layer on top of an existing benchmark,
- introduce a lightweight defense:
  - canonical schema normalization,
  - constrained argument formatting,
  - automatic validator feedback,
  - one-shot repair before tool execution.

This is stronger than just a benchmark paper because it adds a method and ablations.

### A.4 Novelty Window

This direction is likely novel enough if the paper does **all three**:

- introduces realistic drift perturbations,
- evaluates open models under those perturbations,
- proposes a simple defense that materially recovers performance.

It is **not** novel enough if it only says:

- "tool descriptions matter,"
- "benchmark performance drops under perturbations,"
- "larger models do better than smaller models."

### A.5 Minimum Publishable Experiment Package

Benchmarks:

- `DICE-BENCH`
- `ComplexFuncBench`
- optionally a smaller custom API set for controlled perturbations

Models:

- Qwen2.5-7B-Instruct or similar open tool-capable baseline
- Llama-family 8B baseline if available
- one compact model if you want a size comparison

Drift types:

- argument rename
- argument reorder
- enum alias change
- extra distractor tools
- longer/noisier descriptions
- stale descriptions
- changed optional/required fields
- realistic provider-specific error messages

Metrics:

- exact tool selection accuracy
- exact argument match
- execution success rate
- repair success rate
- latency and token overhead

### A.6 Feasibility

**Feasibility:** High  
**Why:** most work is benchmark/evaluation/inference-time logic, not expensive training.

This direction gives fast feedback:

- Day 1-2: reproduce baseline benchmark scores on a subset
- Day 3: add perturbation layer
- Day 4-5: measure drop
- Day 6-7: test canonicalization/repair

### A.7 Risks

- risk that some benchmark code is messy or hard to adapt quickly,
- risk that a very simple repair heuristic already solves most cases, making the contribution too narrow,
- risk that the perturbations look synthetic unless carefully designed.

### A.8 Conference Fit

Strong fit for:

- Agentic AI
- Small & Large Language Models
- Trustworthy AI
- AI Engineering & Lifecycle Management

### A.9 Overall Verdict

**Best overall topic for a short paper on this timeline.**

---

## 7. Shortlisted Direction B: Budget-Aware Online Hallucination Repair for RAG

### B.1 Core Problem

RAG improves factuality, but hallucinations still happen even when retrieval is available. Recent work covers:

- offline hallucination detection,
- adaptive retrieval before answering,
- uncertainty benchmarks,
- faithfulness monitoring.

The underexploited angle is **online intervention**:

> Can the system detect elevated risk during generation and choose a cheap repair action before the answer is finalized?

### B.2 Why This Topic Is Timely

Recent anchors are strong:

- **URAG** (Mar 2, 2026) benchmarks uncertainty quantification in RAG.
- **LettuceDetect** (Feb 24, 2025) provides efficient hallucination detection for RAG.
- **Osiris** (May 7, 2025) offers lightweight hallucination detection with open models.
- **To Retrieve or Not to Retrieve?** (Jan 16, 2025) studies uncertainty for dynamic retrieval.
- **Adaptive Retrieval Without Self-Knowledge?** (Jan 22, 2025) compares adaptive retrieval and uncertainty methods.
- **Synchronous Faithfulness Monitoring for Trustworthy RAG** (Jun 19, 2024) is older but still relevant for online monitoring.

### B.3 Proposed Paper Angle

**Working title:**  
**GuardRAG-Lite: Budget-Aware Online Hallucination Repair for Retrieval-Augmented Generation**

Possible method:

- compute cheap risk features during decoding or sentence generation,
- decide among repair actions:
  - continue,
  - retrieve more,
  - rerank context,
  - switch to extractive mode,
  - abstain / say insufficient evidence.

Key contribution:

- not only detecting hallucination risk,
- but selecting the **lowest-cost corrective action**.

### B.4 Novelty Window

This direction is publishable if the paper is framed around **decision policy and intervention**, not only detection.

Good novelty position:

- combine uncertainty and faithfulness cues,
- optimize a reliability-cost tradeoff,
- show that cheap interventions recover most faithfulness gains of heavier pipelines.

Weak novelty position:

- train another detector on RAGTruth and report F1.

### B.5 Minimum Publishable Experiment Package

Datasets:

- `RAGTruth`
- one QA dataset where retrieval is easy to control

System:

- a simple BM25 or FAISS retriever,
- one 7B instruct model,
- optional comparison against a smaller model

Risk signals:

- token entropy,
- context overlap/support score,
- answer-context contradiction heuristics,
- detector score from a lightweight classifier

Actions:

- no intervention
- retrieve-more
- answer-short-with-citations
- abstain

Metrics:

- faithfulness / hallucination rate
- task accuracy
- calibration or uncertainty alignment
- latency
- retrieval calls per query
- token cost

### B.6 Feasibility

**Feasibility:** High to medium-high  
**Why:** plenty of public baselines exist, but RAG pipelines can become sprawling if not scoped tightly.

### B.7 Risks

- crowded area,
- easy to drift into a large systems paper,
- novelty can collapse if the intervention policy is too obvious.

### B.8 Conference Fit

Strong fit for:

- Generative AI
- Trustworthy AI
- Explainable AI
- Small & Large Language Models

### B.9 Overall Verdict

**Best backup topic.** Strong if the team prefers RAG over agent/tool evaluation.

---

## 8. Shortlisted Direction C: Safety- and OOD-Aware Routing for Multi-LLM Systems

### C.1 Core Problem

Routers are often optimized for cost and quality, but newer work shows routing decisions can be fragile across query type, safety risk, and output-length budget.

### C.2 Recent Anchors

- **R2-Router** (Feb 2, 2026) jointly routes by model choice and output-length budget.
- **How Robust Are Router-LLMs?** (Mar 20, 2025) shows routers can make fragile category-specific decisions and create safety risks.
- **RouterBench** (Mar 18, 2024) provides a benchmark foundation.

### C.3 Possible Angle

Route not only by estimated quality and cost, but also:

- uncertainty,
- out-of-domain detection,
- safety/jailbreak likelihood,
- abstention or escalation threshold.

### C.4 Verdict

Interesting, feasible, and cleanly scoped, but weaker than Directions A and B unless the routing method is especially sharp.

---

## 9. Direction D: Personalized Federated LoRA

### D.1 Why It Was Considered

This matches the CFP well and has strong recent papers:

- **FlowerTune** (Jun 3, 2025)
- **Fed-SB** (Feb 21, 2025)
- **Selective Aggregation for LoRA in FL** (Oct 2, 2024)
- **FedEx-LoRA** (Oct 12, 2024)

### D.2 Why It Is Not the First Choice

- heavier engineering,
- slower iteration,
- harder to get a crisp result quickly,
- larger risk of spending time on infrastructure instead of publishable insight.

### D.3 Verdict

Viable only if the team already has FL code and wants a systems-heavy project.

---

## 10. Comparative Assessment

| Direction | Novelty Window | Feasibility | Evaluation Clarity | Time-to-First-Signal | Overall |
|-----------|----------------|-------------|--------------------|----------------------|---------|
| Tool calling under interface drift | High | High | High | Very fast | **Top pick** |
| Online hallucination repair for RAG | Medium-high | High | High | Fast | **Best backup** |
| Safety/OOD-aware routing | Medium | High | Medium-high | Fast | Good backup |
| Personalized federated LoRA | Medium-high | Medium | Medium-high | Slower | Stretch |

---

## 11. Recommendation

### Recommended Topic to Pursue Now

**Robust tool calling under interface drift**

Recommended framing:

> Open tool-calling LLMs are brittle under realistic interface drift.  
> We introduce a drift taxonomy, a stress-test benchmark layer, and a lightweight canonicalization-plus-repair pipeline that improves execution reliability without retraining.

Why this is the right choice:

- very recent literature exists, so the topic is current,
- there is still clear room for a practical contribution,
- experiments are not prohibitively expensive,
- failure modes are easy to visualize and explain,
- the story is strong for an 8-page short paper.

### Recommended Backup

**Budget-aware online hallucination repair for RAG**

Use this if the team is more comfortable with QA/RAG pipelines than tool-calling infrastructure.

---

## 12. Immediate Go/No-Go Experiments

Before committing to a full plan, run one of these fast pilots.

### Pilot A: Tool-Calling Drift Stress Test

1. Reproduce a baseline on a subset of `ComplexFuncBench` or `DICE-BENCH`.
2. Apply 4-5 drift perturbations.
3. Measure exact-call and execution-success drop.
4. Add a simple repair layer.
5. Continue only if:
   - baseline drop is clearly measurable, and
   - repair recovers a meaningful fraction.

### Pilot B: Online RAG Repair

1. Build a minimal RAG pipeline.
2. Evaluate on a small `RAGTruth` subset.
3. Compute cheap risk signals.
4. Trigger one repair action when risk is high.
5. Continue only if:
   - risk correlates with hallucination,
   - intervention improves faithfulness without exploding latency.

---

## 13. Decision Rule

Lock the topic after the first pilot week using this rule:

- choose the direction with the clearest failure mode,
- the smallest implementation burden,
- and the strongest measurable recovery from a simple method.

If both directions work:

- choose **tool-calling drift** for stronger novelty positioning,
- choose **RAG repair** for easier demoability.

---

## 14. Seed Bibliography

### Conference

- NGEN-AI 2026 official site: <https://ngen-ai.org/>

### Tool Calling / Agent Reliability

- Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use (2026-02-23): <https://arxiv.org/abs/2602.20426>
- ASA: Training-Free Representation Engineering for Tool-Calling Agents (2026-02-04): <https://arxiv.org/abs/2602.04935>
- Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors (2026-01-22): <https://arxiv.org/abs/2601.15625>
- DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues (2025-06-28): <https://arxiv.org/abs/2506.22853>
- ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario (2025-01-17): <https://arxiv.org/abs/2501.10132>
- Gaming Tool Preferences in Agentic LLMs (2025-05-23): <https://arxiv.org/abs/2505.18135>

### RAG Reliability / Uncertainty

- URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models (2026-03-02): <https://arxiv.org/abs/2603.19281>
- LettuceDetect: A Hallucination Detection Framework for RAG Applications (2025-02-24): <https://arxiv.org/abs/2502.17125>
- Osiris: A Lightweight Open-Source Hallucination Detection System (2025-05-07): <https://arxiv.org/abs/2505.04844>
- To Retrieve or Not to Retrieve? Uncertainty Detection for Dynamic Retrieval Augmented Generation (2025-01-16): <https://arxiv.org/abs/2501.09292>
- Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home (2025-01-22): <https://arxiv.org/abs/2501.12835>
- Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation (2024-06-19): <https://arxiv.org/abs/2406.13692>
- RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models (2023-12-31): <https://arxiv.org/abs/2401.00396>

### Routing

- R2-Router: A New Paradigm for LLM Routing with Reasoning (2026-02-02): <https://arxiv.org/abs/2602.02823>
- How Robust Are Router-LLMs? Analysis of the Fragility of LLM Routing Capabilities (2025-03-20): <https://arxiv.org/abs/2504.07113>
- RouterBench: A Benchmark for Multi-LLM Routing System (2024-03-18): <https://arxiv.org/abs/2403.12031>

### Federated LLM Fine-Tuning

- FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models (2025-06-03): <https://arxiv.org/abs/2506.02961>
- Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning (2025-02-21): <https://arxiv.org/abs/2502.15436>
- Selective Aggregation for Low-Rank Adaptation in Federated Learning (2024-10-02): <https://arxiv.org/abs/2410.01463>
- FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models (2024-10-12): <https://arxiv.org/abs/2410.09432>

---

## 15. Next Document

The next step after this discovery document should be a **full research execution plan** for one chosen topic, including:

- exact benchmark and model choices,
- baselines,
- perturbation or intervention design,
- metrics,
- week-by-week timeline,
- compute budget,
- paper outline.
