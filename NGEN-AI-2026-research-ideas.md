# NGEN-AI 2026 -- Research Ideas: Explainable & Trustworthy AI
**Conference:** International Conference on Next Generation AI Systems (NGEN-AI 2026)
**Venue:** Trento, Italy, 1-4 September 2026
**Submission deadline:** May 25, 2026
**Available time:** ~30-45 days

---

## Summary of Landscape Findings

The research landscape (2024-2026) reveals several underexplored intersections:

- **XAI for LLMs** remains dominated by post-hoc analysis; temporal/causal explainability and explainability under compression are barely addressed.
- **Federated learning + explainability** is a recognized open problem (debugging, interpretability across heterogeneous clients) with almost no concrete solutions published.
- **On-device LLMs** are maturing fast (ExecuTorch 1.0, sub-billion models), but uncertainty quantification and explainability for these models is essentially absent.
- **Multi-agent explainability** is identified as critical (agent-to-agent "black box" effect) but has no standard frameworks.
- **Carbon-aware inference** is an active area, but integration with explainability/trustworthiness metrics is unexplored.

---

## IDEA 1: ExplainFL -- Federated Explainability Aggregation for Heterogeneous Edge Clients

### Core Concept
Design a federated learning protocol where clients not only share model updates but also share lightweight local explanation summaries (e.g., compressed SHAP value distributions or feature attribution sketches). The server aggregates these into a global "explanation model" alongside the global model, enabling system-wide interpretability without exposing raw data.

### Novelty Assessment: HIGH
- Current FL research focuses on model accuracy and privacy; explainability aggregation across heterogeneous clients is an open problem explicitly identified in recent surveys but with no published solution.
- No existing work federates explanations themselves -- only model weights.

### Feasibility Assessment: HIGH (30-35 days)
- Use Flower (FL framework) + SHAP/LIME on standard benchmarks (CIFAR-10, sentiment analysis).
- Prototype: 5-10 simulated clients with non-IID data splits.
- Measure: explanation fidelity (agreement between local and global explanations), communication overhead, and privacy leakage of explanation summaries.

### Evaluation Methodology
1. Explanation fidelity: cosine similarity between local and aggregated global explanations vs. centralized-training explanations.
2. Communication overhead: bytes transmitted for explanation summaries vs. model updates.
3. Privacy: membership inference attack success rate on explanation summaries vs. raw gradients.
4. Model accuracy: verify no degradation from explanation-aware aggregation.

### Conference Fit
Intersects: Federated Learning + Explainability + Privacy-Preserving AI + Edge Systems

---

## IDEA 2: UQ-Edge -- Lightweight Uncertainty Quantification for On-Device Small Language Models

### Core Concept
Develop a computationally cheap uncertainty quantification method tailored for quantized small language models (e.g., SmolLM2-360M, Gemma-3-270M) running on mobile/edge devices. Use a combination of (a) token-level entropy from the softmax distribution and (b) a tiny learned "uncertainty head" (a small MLP added during fine-tuning) that predicts calibrated confidence scores, requiring <1% additional parameters.

### Novelty Assessment: HIGH
- Existing UQ methods for LLMs (Monte Carlo dropout, deep ensembles) are too expensive for edge. The ACM Computing Surveys 2025 survey on LLM UQ identifies edge-adapted UQ as an open challenge.
- No published work specifically addresses UQ for sub-1B parameter quantized models on mobile.

### Feasibility Assessment: HIGH (25-30 days)
- Fine-tune SmolLM2-360M or Phi-4-mini with an uncertainty head using LoRA on a QA dataset (TriviaQA, NaturalQuestions).
- Deploy on a Raspberry Pi 5 or Android phone via ExecuTorch/llama.cpp.
- Compare against baselines: raw softmax entropy, temperature scaling, MC dropout (if feasible).

### Evaluation Methodology
1. Calibration: Expected Calibration Error (ECE) and reliability diagrams.
2. Selective prediction: accuracy-coverage curves (reject uncertain predictions).
3. Out-of-distribution detection: AUROC on OOD inputs.
4. Resource cost: latency overhead (ms), memory overhead (MB), energy overhead (mJ per inference).

### Conference Fit
Intersects: Trustworthy AI + Edge/On-Device AI + Small Language Models + Uncertainty

---

## IDEA 3: XAI-Distill -- Explainability-Preserving Knowledge Distillation for Mobile Deployment

### Core Concept
Extend XAI-driven knowledge distillation (building on DiXtill, 2024) by adding an explicit "explanation alignment" loss during distillation: the student model's feature attributions must match the teacher's, not just the output logits. Evaluate whether this produces small models that are both accurate AND provide faithful explanations on mobile devices.

### Novelty Assessment: MEDIUM-HIGH
- DiXtill (2024) showed XAI can guide distillation, but did not evaluate explanation faithfulness of the compressed model post-deployment, nor target mobile/edge constraints.
- The gap between "using XAI to distill" and "distilling to preserve XAI" is unexplored.

### Feasibility Assessment: HIGH (30-35 days)
- Teacher: Llama-3.2-3B or Phi-4-mini. Student: SmolLM2-360M or custom 200M model.
- Task: text classification (SST-2, AG News) or extractive QA.
- Distillation with standard KD loss + explanation alignment loss (L2 distance between teacher/student attention-based attributions).
- Deploy student on mobile, measure explanation quality.

### Evaluation Methodology
1. Task performance: accuracy, F1 on held-out test sets.
2. Explanation faithfulness: attribution agreement (Kendall tau, top-k overlap) between teacher and student explanations.
3. Explanation plausibility: human evaluation on a small set (50-100 examples).
4. Compression metrics: model size, latency, memory footprint on target device.

### Conference Fit
Intersects: XAI + Knowledge Distillation + Small Language Models + Edge AI

---

## IDEA 4: CarbonXAI -- Carbon-Aware Inference with Explainable Scheduling Decisions

### Core Concept
Build an inference scheduler for edge-cloud LLM systems that (a) routes queries to edge or cloud based on carbon intensity of the local grid, query complexity, and latency requirements, and (b) provides human-readable explanations for each routing decision (why a query was sent to cloud vs. processed locally). The system uses a lightweight decision tree or rule-based model as the scheduler, making decisions inherently interpretable.

### Novelty Assessment: HIGH
- Carbon-aware scheduling exists (CarbConscious 2025, EcoServe 2025), but none provide explainability for scheduling decisions.
- The combination of sustainability + XAI is a completely novel intersection.
- Addresses the EU AI Act's dual requirements for transparency AND sustainability reporting.

### Feasibility Assessment: MEDIUM-HIGH (35-40 days)
- Simulate an edge-cloud setup: edge = Raspberry Pi 5 with SmolLM2; cloud = GPU server with Llama-3.2-3B.
- Use real carbon intensity data from electricityMap API.
- Scheduler: trained decision tree or gradient-boosted model on features (query length, carbon intensity, time-of-day, edge load).
- Generate natural language explanations from decision tree paths.

### Evaluation Methodology
1. Carbon savings: total CO2e reduction vs. cloud-only and edge-only baselines.
2. Quality of service: latency distribution, answer quality (BLEU/ROUGE for generation tasks).
3. Explanation quality: user study (N=20-30) rating explanation usefulness and trust.
4. Scheduling accuracy: precision/recall of routing decisions vs. oracle optimal.

### Conference Fit
Intersects: Green/Sustainable AI + XAI + Edge-Cloud Systems + Trustworthy AI

---

## IDEA 5: XMAS -- Explainable Multi-Agent System Interaction Traces

### Core Concept
Design a lightweight logging and explanation framework for multi-agent LLM systems (e.g., CrewAI, AutoGen) that captures inter-agent communication, tracks how each agent's output influenced the final result, and generates attribution scores for each agent's contribution. Use provenance-based explanation (tracking the causal chain of agent interactions) rather than post-hoc feature attribution.

### Novelty Assessment: HIGH
- The "black box effect" in agent-to-agent interactions is explicitly identified as an unsolved problem in recent multi-agent research (DIS 2025, EMAS 2025).
- No existing framework provides systematic, quantitative explainability for multi-agent LLM orchestration.

### Feasibility Assessment: MEDIUM (35-45 days)
- Instrument an existing multi-agent framework (CrewAI or AutoGen) with logging hooks.
- Implement: (a) communication graph construction, (b) influence propagation scoring (based on textual similarity between agent outputs and final answer), (c) counterfactual analysis (re-run with agent removed, measure output change).
- Test on 2-3 multi-agent tasks: research summarization, code generation, debate.

### Evaluation Methodology
1. Attribution accuracy: correlation between influence scores and counterfactual agent removal impact.
2. Bug localization: can the framework identify which agent introduced an error? Measure on injected-fault scenarios.
3. Overhead: latency and token cost of explanation generation vs. base multi-agent execution.
4. User study (N=15-20): do explanations help developers debug multi-agent pipelines faster?

### Conference Fit
Intersects: Multi-Agent Systems + XAI + Agentic AI + Trustworthy AI

---

## IDEA 6: GreenTrust -- A Unified Benchmark for Trustworthy and Sustainable Edge AI

### Core Concept
Create a benchmark suite that jointly evaluates edge AI models across BOTH trustworthiness dimensions (accuracy, fairness, robustness, uncertainty calibration, explainability) AND sustainability dimensions (energy consumption, carbon footprint, memory usage, latency). Provide a Pareto-frontier analysis showing trade-offs. No existing benchmark evaluates both axes together.

### Novelty Assessment: MEDIUM-HIGH
- Sustainability benchmarks (LLM energy footprint, 2025) and trustworthiness benchmarks exist separately. Their intersection is unexplored.
- ISO TR 20226:2025 calls for lifecycle sustainability metrics; this benchmark operationalizes that call for edge AI.

### Feasibility Assessment: HIGH (25-30 days)
- Select 8-10 models spanning sizes (135M to 3B): SmolLM2 family, Gemma-3, Phi-4-mini, Qwen2.5.
- Tasks: text classification, QA, summarization.
- Trustworthiness metrics: accuracy, ECE (calibration), fairness (demographic parity), robustness (adversarial accuracy), explanation faithfulness.
- Sustainability metrics: energy (measured via CodeCarbon/Zeus), peak memory, latency, estimated CO2e.
- Hardware: Raspberry Pi 5, Jetson Orin Nano, smartphone (via ExecuTorch).

### Evaluation Methodology
1. Pareto frontier analysis: which models dominate on trust-sustainability trade-off?
2. Correlation analysis: does higher trustworthiness cost more energy? Quantify.
3. Recommendations: model selection guidelines for different edge deployment scenarios.
4. Reproducibility: open-source benchmark code and results.

### Conference Fit
Intersects: Trustworthy AI + Sustainable AI + Edge AI + Benchmarking + Small Language Models

---

## Comparative Assessment Matrix

| Idea | Novelty | Feasibility (30-45 days) | Evaluation Clarity | Multi-Track Intersection | Recommended Priority |
|------|---------|--------------------------|-------------------|--------------------------|---------------------|
| 1. ExplainFL | High | High | Strong | FL + XAI + Privacy + Edge | **TOP PICK** |
| 2. UQ-Edge | High | High | Strong | Trust + Edge + SLMs | **TOP PICK** |
| 3. XAI-Distill | Medium-High | High | Strong | XAI + Distillation + Edge | Strong backup |
| 4. CarbonXAI | High | Medium-High | Strong | Green AI + XAI + Edge-Cloud | **TOP PICK** |
| 5. XMAS | High | Medium | Good | Multi-Agent + XAI + Trust | High risk, high reward |
| 6. GreenTrust | Medium-High | High | Strong | Trust + Sustainability + Edge | Safe bet, high impact |

---

## Recommended Strategy

### If submitting ONE paper:
**Go with Idea 2 (UQ-Edge) or Idea 1 (ExplainFL)**. Both have high novelty, clear evaluation, and are achievable in 30 days. UQ-Edge has a slight edge because on-device LLMs are the hottest topic right now and the gap in UQ is very clear.

### If submitting TWO papers:
- **Primary:** Idea 1 (ExplainFL) or Idea 2 (UQ-Edge) -- novel method paper
- **Secondary:** Idea 6 (GreenTrust) -- benchmark/resource paper (different contribution type, lower rejection risk)

### If the team wants maximum novelty:
**Combine Ideas 2 + 4**: Build an on-device LLM system with uncertainty quantification that uses uncertainty scores to decide edge-vs-cloud routing, with carbon-aware scheduling and explainable decisions. This would be a systems paper spanning UQ + XAI + Green AI + Edge.

---

## Key Tools and Frameworks for Prototyping

| Component | Tool |
|-----------|------|
| Federated Learning | Flower (flwr.dev) |
| On-device LLM deployment | ExecuTorch, llama.cpp, MLC-LLM |
| Small language models | SmolLM2, Gemma-3, Phi-4-mini, Qwen2.5 |
| XAI methods | SHAP, LIME, Captum (PyTorch) |
| Carbon/energy measurement | CodeCarbon, Zeus, electricityMap API |
| Multi-agent frameworks | CrewAI, AutoGen, LangGraph |
| Fine-tuning | LoRA via PEFT/Hugging Face |
| Hardware | Raspberry Pi 5, Jetson Orin Nano, Android phone |

---

## References and Key Sources

1. XAI 2.0 Manifesto: Arrieta et al., "Explainable Artificial Intelligence (XAI) 2.0: A manifesto of open challenges" (2024)
2. DiXtill: "XAI-driven knowledge distillation of LLMs for efficient deployment on low-resource devices" (2024)
3. LLM UQ Survey: ACM Computing Surveys (2025), "A Survey on Uncertainty Quantification of LLMs"
4. On-Device LLMs 2026: ExecuTorch 1.0, sub-billion parameter convergence across labs
5. CarbConscious: "Carbon-Aware Scheduling for Sustainable LLM Inference" (2025)
6. EcoServe: "Designing Carbon-Aware AI Inference Systems" (2025)
7. Federated Carbon Intelligence: MRS Energy & Sustainability (2025)
8. EMAS 2025 Roadmap: "Engineering the Next Generation of Multi-agent Systems" (2025)
9. ISO TR 20226:2025: Sustainable AI evaluation indicators
10. LLM Energy Benchmark: "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference" (2025)

---
---

# PART 2: AGENTIC AI RESEARCH IDEAS

*Added: March 13, 2026 -- focused on agentic AI, multi-agent coordination, edge deployment, privacy, and safety*

---

## Current Landscape (2024-2026): Key Findings from Web Research

### What's Hot
1. **MCP + A2A protocols** are becoming the HTTP of agentic AI. Anthropic's MCP (model-to-tools) and Google's A2A (agent-to-agent, launched April 2025, now Linux Foundation governed) have 50+ enterprise partners.
2. **SLMs replacing LLMs for agentic tasks**: NVIDIA position paper (arxiv 2506.02153) shows 7B SLMs are 10-30x cheaper and can handle 40-70% of agentic queries. Phi-2 (2.7B) matches 30B models on reasoning.
3. **Edge-native agents** moving from concept to product: NXP eIQ Agentic AI Framework (CES 2026), Qualcomm Snapdragon on-device agents, targeting robotics/industrial/IoT.
4. **Agent safety** becoming a first-class concern: AgentSpec (ICSE 2026), Pro2Guard, VeriGuard, multiple AAAI-26 workshops.
5. **1,445% surge** in multi-agent system interest (Gartner, Q1 2024 to Q2 2025). 40% of enterprise apps expected to embed agents by end of 2026.
6. **Federated MAS** is a recognized new paradigm (arxiv 2503.08175, ICML 2025 CFAgentic workshop) distinct from federated learning.

### What's Underexplored (Gaps = Opportunities)
1. **Privacy-preserving inter-agent communication on edge** -- EPEAgent is cloud-only.
2. **Safety enforcement under resource constraints** -- all guardrail work (AgentSpec, VeriGuard) assumes cloud compute.
3. **Federated alignment for agents** -- completely unexplored territory.
4. **Energy measurement and optimization for agentic workflows** -- lots of talk, no rigorous empirical studies.
5. **A2A/MCP for edge/IoT** -- protocols are cloud-first, no edge adaptation exists.
6. **Multi-device collaborative agent inference** -- speculative decoding work is device-to-server, not device-to-device swarms.

---

## IDEA 7: Federated Agentic Swarm on Edge -- Privacy-Preserving Multi-Agent Collaboration Across Personal Devices

### Core Concept
Design a lightweight multi-agent framework where SLM-powered agents on personal edge devices (phones, tablets, laptops) collaborate on shared tasks without centralizing data. Each device hosts a specialized agent; agents coordinate via a privacy-preserving protocol that combines A2A-style communication with differential privacy guarantees on inter-agent messages.

### Novelty Assessment: HIGH
- The intersection of federated MAS + edge SLMs + privacy-preserving agent communication is largely unexplored. EPEAgent (arxiv 2503.08175) addresses privacy in cloud-based MAS but not edge-distributed scenarios.
- NVIDIA's SLM paper argues SLMs are the future of agentic AI but does not address federated coordination.
- NXP's eIQ framework targets single-device edge agents, not cross-device collaboration.
- No existing work combines all three: (a) SLM agents on heterogeneous edge devices, (b) A2A-style inter-agent communication, (c) differential privacy on shared task state.

### Feasibility Assessment: HIGH (30-35 days)
- **Week 1-2**: Build prototype using 2-3 edge devices (phones/Raspberry Pi) running quantized SLMs (Phi-3-mini, Gemma-2B, or Qwen-2.5-3B). Use CrewAI or OpenAI Agents SDK for agent orchestration, MQTT for device communication.
- **Week 2-3**: Implement DP-noise injection on inter-agent messages. Define task decomposition protocol (e.g., collaborative document summarization where each device holds private documents).
- **Week 3-4**: Evaluate on benchmark tasks: (a) collaborative QA across distributed private knowledge bases, (b) multi-device sensor fusion for IoT scenario. Measure latency, privacy leakage (membership inference attacks), task accuracy.
- **Tools needed**: 2-3 edge devices, quantized SLM weights (freely available), Python, MQTT broker.

### Key Contribution
First framework demonstrating feasible privacy-preserving multi-agent collaboration using SLMs on real edge hardware, with formal DP guarantees on inter-agent communication.

### Conference Track Fit
Agentic systems + Federated/privacy-preserving AI + Edge-cloud infrastructure (triple intersection).

### Key References
- [Privacy-Enhancing Paradigms within Federated MAS](https://arxiv.org/abs/2503.08175)
- [Small Language Models are the Future of Agentic AI (NVIDIA)](https://arxiv.org/html/2506.02153)
- [Secure Multi-LLM Agentic AI for Edge General Intelligence](https://arxiv.org/html/2508.19870v1)
- [ICML 2025 Workshop on Collaborative and Federated Agentic Workflows](https://icml.cc/virtual/2025/workshop/39961)

---

## IDEA 8: AgentSplit -- Distributed Speculative Decoding for Collaborative Multi-Agent Inference on Heterogeneous Edge Devices

### Core Concept
Multiple edge devices collaboratively run a single large agentic workflow by splitting speculative decoding across devices. Smaller devices run draft SLM agents for speculation; a more capable device (e.g., laptop) verifies and corrects. The system adaptively routes agent sub-tasks to devices based on their computational capacity, creating a "device swarm" that collectively achieves LLM-quality agentic reasoning using only SLMs.

### Novelty Assessment: HIGH
- DSSD (ICML 2025) addresses split speculative decoding between one device and one edge server. This extends to N heterogeneous devices in a multi-agent setting where each device is also an autonomous agent.
- Existing multi-agent frameworks (CrewAI, LangChain) assume homogeneous cloud compute. No framework addresses heterogeneous edge device pools.
- The combination of speculative decoding + multi-agent task decomposition + heterogeneous device orchestration is novel.

### Feasibility Assessment: MEDIUM-HIGH (35-40 days)
- **Week 1**: Implement baseline speculative decoding with draft model on one device, verify on another. Use llama.cpp or MLC-LLM for on-device inference.
- **Week 2-3**: Extend to 3+ devices with adaptive task routing. Implement agent-level task decomposition (e.g., Agent A drafts code, Agent B on stronger device verifies/refines).
- **Week 3-4**: Benchmark against: (a) single-device SLM agent, (b) cloud LLM agent. Metrics: tokens/sec, task completion accuracy, energy consumption, latency.
- **Risk**: Inter-device communication latency may negate speculative decoding gains on LAN. Mitigate by focusing on coarse-grained task-level speculation rather than token-level.

### Key Contribution
Novel architecture for pooling heterogeneous edge devices into a collaborative agentic inference system, achieving near-LLM quality without cloud dependency.

### Conference Track Fit
Edge-cloud infrastructure + Agentic systems + Deep learning (model serving).

### Key References
- [DSSD: Distributed Split Speculative Decoding (ICML 2025)](https://openreview.net/forum?id=5vkXfhUmnn)
- [Collaborative Multi-Device Edge Inference with Speculative Decoding (IEEE 2025)](https://ieeexplore.ieee.org/document/11275395/)
- [Fast Collaborative Inference via Distributed Speculative Decoding](https://arxiv.org/abs/2512.16273)

---

## IDEA 9: SafeEdgeAgents -- Runtime Verification and Safety Guardrails for Autonomous Edge Agents

### Core Concept
Adapt AgentSpec-style runtime safety enforcement for resource-constrained edge agents. Cloud-based guardrails (LLM self-examination, external validator calls) are too expensive for edge. Propose a lightweight safety specification language and enforcement engine using pre-compiled rule trees and a tiny classifier model (<100M params) that runs alongside the agent SLM, providing sub-millisecond safety checks without cloud dependency.

### Novelty Assessment: HIGH
- AgentSpec (ICSE 2026) and VeriGuard target cloud/server agents with access to validator LLMs. No work addresses safety enforcement under edge resource constraints.
- The combination of formal safety specification + lightweight neural verification + edge deployment is unexplored.
- Addresses a critical gap: as agents move to edge (NXP eIQ, Qualcomm Snapdragon), safety mechanisms must follow.

### Feasibility Assessment: HIGH (25-30 days)
- **Week 1**: Define a subset of AgentSpec's DSL suitable for edge (remove LLM-self-examine, add compiled predicate trees). Implement rule compiler converting safety specs to efficient decision trees.
- **Week 2**: Train a tiny safety classifier (distilled from larger model) on AgentHarm or similar benchmarks. Target: <50M parameters, <5ms inference.
- **Week 3**: Integrate with an edge agent running on Raspberry Pi or phone. Test on: unsafe tool calls, prompt injection attempts, out-of-bounds actions.
- **Week 4**: Compare against: (a) no guardrails, (b) full AgentSpec with LLM validator, (c) our lightweight approach. Metrics: safety violation rate, latency overhead, memory footprint.

### Key Contribution
First safety enforcement framework specifically designed for resource-constrained edge agents, maintaining >90% safety coverage with <5ms overhead and no cloud dependency.

### Conference Track Fit
Trustworthy/responsible AI + Agentic systems + Edge-cloud infrastructure.

### Key References
- [AgentSpec: Customizable Runtime Enforcement (ICSE 2026)](https://arxiv.org/abs/2503.18666)
- [Pro2Guard: Proactive Runtime Enforcement via Probabilistic Model Checking](https://arxiv.org/abs/2508.00500)
- [Systems-Level Attack Surface of Edge Agent Deployments on IoT](https://arxiv.org/html/2602.22525)

---

## IDEA 10: GreenAgents -- Energy-Aware Multi-Agent Orchestration with Adaptive Model Selection

### Core Concept
A multi-agent orchestrator that dynamically selects between SLMs of different sizes (1B, 3B, 7B) for each agent sub-task based on task complexity and an energy budget. Uses a lightweight complexity estimator to route simple sub-tasks to smaller models and complex sub-tasks to larger ones, minimizing total energy while maintaining task quality. Includes carbon-aware scheduling for non-urgent tasks.

### Novelty Assessment: MEDIUM-HIGH
- NVIDIA's SLM paper identifies heterogeneous model selection as important but proposes no dynamic orchestration mechanism.
- LLM routing exists in cloud settings but not in energy-constrained multi-agent edge scenarios.
- Carbon-aware scheduling is established in cloud computing but novel when applied to agentic task orchestration.

### Feasibility Assessment: HIGH (25-30 days)
- **Week 1**: Implement task complexity classifier (fine-tune small model on AgentBench data). Set up 3 SLMs of different sizes.
- **Week 2**: Build orchestrator that routes agent sub-tasks to appropriate model. Implement energy measurement (CodeCarbon, PowerJoular).
- **Week 3**: Evaluate on multi-step agentic benchmarks (GAIA subset, WebArena). Measure energy per task, accuracy, latency.
- **Week 4**: Add carbon-intensity API integration, write results.

### Key Contribution
First energy-aware multi-agent orchestration framework with empirical measurements of energy savings (target: 40-60% reduction) at <5% accuracy loss.

### Conference Track Fit
Responsible AI + Agentic systems + Edge-cloud infrastructure.

### Key References
- [Small Language Models are the Future of Agentic AI (NVIDIA)](https://arxiv.org/html/2506.02153)
- [Edge-First Language Model Inference: Models, Metrics, and Tradeoffs (IEEE ICDCS 2025)](https://arxiv.org/html/2505.16508v1)

---

## IDEA 11: A2A-Edge -- Extending Agent-to-Agent Protocol for Resource-Constrained Decentralized Agent Networks

### Core Concept
Google's A2A protocol assumes enterprise cloud environments with stable connectivity. Propose A2A-Edge: a lightweight adaptation for decentralized agent networks on edge devices, addressing: (a) intermittent connectivity via store-and-forward messaging, (b) resource discovery in local networks without centralized registries (mDNS-based), (c) compressed agent capability cards for low-bandwidth, (d) lightweight mutual authentication for constrained devices.

### Novelty Assessment: HIGH
- A2A (v0.3) and MCP are cloud-first. No work adapts these protocols for edge/IoT.
- The protocol interoperability survey (arxiv 2505.02279) identifies this as an open challenge but does not address edge constraints.
- Novel protocol design + reference implementation + evaluation is a strong systems contribution.

### Feasibility Assessment: MEDIUM (35-40 days)
- **Week 1-2**: Analyze A2A protocol spec, identify components needing adaptation. Design A2A-Edge extensions.
- **Week 2-3**: Implement reference Python library. Deploy on 3-5 devices (Raspberry Pi, phones).
- **Week 3-4**: Evaluate: message delivery rate under intermittent connectivity, discovery latency, bandwidth savings, security overhead. Compare with baseline A2A over HTTP.
- **Risk**: Protocol design papers require careful specification. Mitigate by focusing on 2-3 key extensions rather than complete redesign.

### Key Contribution
First adaptation of standardized agent interoperability protocols (A2A/MCP) for edge-constrained, decentralized agent networks.

### Conference Track Fit
Agentic systems + Edge-cloud infrastructure + Federated/privacy-preserving AI.

### Key References
- [A2A Protocol (Google, v0.3)](https://a2a-protocol.org/latest/)
- [Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1)
- [LLM Multi-Agent System for 6G Edge-Terminal Collaboration](https://arxiv.org/abs/2509.04993)

---

## IDEA 12: FedAgent-Align -- Federated Safety Alignment for Distributed Edge Agents Without Sharing Private Interaction Data

### Core Concept
When edge agents interact with users in private settings (healthcare, finance), their safety alignment must improve over time without centralizing sensitive interaction logs. Propose federated alignment: each edge agent maintains a local safety reward model that learns from user interactions; periodically, agents share only differentially-private gradient updates to a global safety model, improving collective alignment without exposing private data.

### Novelty Assessment: VERY HIGH
- Federated learning for LLM fine-tuning exists, but federated *safety alignment* for autonomous agents is completely unexplored.
- Combines three cutting-edge areas: RLHF/safety alignment + federated learning + edge agents.
- Directly addresses a real-world gap: improving agent safety in privacy-sensitive deployments.

### Feasibility Assessment: MEDIUM (40-45 days)
- **Week 1**: Set up simulated environment with 5-10 agent instances, each with a local safety reward model (small classifier). Use AgentHarm or SafeAgentBench dataset partitioned across agents.
- **Week 2-3**: Implement federated averaging of safety reward model updates with DP noise. Compare: (a) no federation (local only), (b) centralized training, (c) federated with DP.
- **Week 3-4**: Measure safety improvement over rounds, privacy guarantee (epsilon values), convergence speed.
- **Risk**: May need simulation rather than real edge deployment. Acceptable for a first paper.

### Key Contribution
First framework for federated safety alignment of autonomous agents, enabling collective safety improvement without compromising user privacy.

### Conference Track Fit
Trustworthy/responsible AI + Federated/privacy-preserving AI + Agentic systems (triple intersection).

### Key References
- [Agent Safety Alignment via Reinforcement Learning](https://arxiv.org/pdf/2507.08270)
- [Privacy-Enhancing Paradigms within Federated MAS](https://arxiv.org/abs/2503.08175)
- [TRiSM for Agentic AI](https://arxiv.org/html/2506.04133v2)

---

## Comparative Assessment Matrix (Agentic AI Ideas)

| # | Idea | Novelty | Feasibility | Days | Track Intersections | Risk |
|---|------|---------|-------------|------|---------------------|------|
| 7 | Federated Agentic Swarm on Edge | HIGH | HIGH | 30-35 | Agentic + Federated + Edge | Low |
| 8 | AgentSplit (Distributed Spec Decoding) | HIGH | MED-HIGH | 35-40 | Edge + Agentic + DL | Medium |
| 9 | SafeEdgeAgents (Runtime Safety) | HIGH | HIGH | 25-30 | Safety + Agentic + Edge | Low |
| 10 | GreenAgents (Energy-Aware) | MED-HIGH | HIGH | 25-30 | Responsible AI + Agentic + Edge | Low |
| 11 | A2A-Edge Protocol | HIGH | MEDIUM | 35-40 | Agentic + Edge + Federated | Medium |
| 12 | FedAgent-Align (Federated Safety) | VERY HIGH | MEDIUM | 40-45 | Safety + Federated + Agentic | Med-High |

---

## Recommended Strategy for Agentic AI Submission

### Top Pick: IDEA 7 (Federated Agentic Swarm on Edge)
**Rationale**: Best balance of novelty, feasibility, and conference fit. Uses readily available tools (quantized SLMs, MQTT, CrewAI). Clear evaluation metrics. Hits three NGEN-AI tracks. Lowest execution risk.

### Strong Alternative: IDEA 9 (SafeEdgeAgents)
**Rationale**: Fastest to implement (25-30 days). Addresses timely and critical gap. Builds on well-established AgentSpec with a clear novel contribution (edge adaptation).

### Ambitious Choice: IDEA 12 (FedAgent-Align)
**Rationale**: Highest novelty score. If the team has strong ML/federated learning expertise, this could yield a high-impact paper. Tight timeline but simulation-based approach is viable.

### Combination Strategy (Highest Impact)
Combine Ideas 7 + 9: a federated multi-agent edge system WITH lightweight safety guardrails. This comprehensive systems paper covering privacy, safety, and edge deployment is highly aligned with NGEN-AI's vision and difficult for reviewers to find close prior work.

---

## Cross-Cutting Opportunities: Combining Part 1 and Part 2 Ideas

| Combination | Description | Novelty |
|-------------|-------------|---------|
| Idea 2 (UQ-Edge) + Idea 9 (SafeEdgeAgents) | Uncertainty-aware safety guardrails: agent refuses to act when uncertainty is high AND safety classifier flags risk | Very High |
| Idea 1 (ExplainFL) + Idea 7 (Federated Swarm) | Federated agents that share explanations of their decisions alongside DP-protected task outputs | Very High |
| Idea 5 (XMAS) + Idea 10 (GreenAgents) | Explainable multi-agent orchestration with energy-aware model selection and transparent routing decisions | High |
| Idea 4 (CarbonXAI) + Idea 10 (GreenAgents) | Carbon-aware agentic scheduler with explainable routing across edge-cloud-SLM tiers | High |

---

## Additional References (Agentic AI Section)

11. [7 Agentic AI Trends to Watch in 2026](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/)
12. [NXP eIQ Agentic AI Framework](https://www.embedded.com/nxp-develops-eiq-agentic-ai-framework/)
13. [Agentic AI Framework for Edge Applications (EEJournal)](https://www.eejournal.com/article/agentic-ai-framework-for-edge-applications/)
14. [AI Trends 2026: Agentic AI, Multi-Agent Systems & Edge AI Roadmap](https://imigo.ai/en/media/ai-trends-2026)
15. [EDGE AI San Diego 2026 Conference](https://www.edgeaifoundation.org/events/edge-ai-san-diego-2026)
16. [Agentic Edge AI: Autonomous Intelligence on the Edge (Trend Micro)](https://www.trendmicro.com/vinfo/us/security/news/cybercrime-and-digital-threats/agentic-edge-ai-autonomous-intelligence-on-the-edge)
17. [A2A Protocol v0.3](https://a2a-protocol.org/latest/)
18. [Google A2A Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
19. [Agentic AI Driving Workloads to Edge (HPCwire, March 2026)](https://www.hpcwire.com/2026/03/05/agentic-ai-is-driving-workloads-and-infra-on-prem-and-to-the-edge/)
20. [Qualcomm: Agentic AI at the Edge with Snapdragon](https://www.qualcomm.com/news/onq/2025/09/edge-essential-role-future-of-ai-snapdragon-summit-2025)
21. [Lightweight Multi-Agent Edge Framework for Cybersecurity (CMC Journal)](https://www.techscience.com/cmc/v86n1/64446)
22. [Federated Multi-Agent Reasoning (Medium)](https://medium.com/@raktims2210/federated-multi-agent-reasoning-how-ai-systems-collaborate-across-organizations-without-sharing-586b8e099e09)
23. [Security of LLM-based Agents Survey (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1566253525010036)
24. [AI Security Trends 2026](https://www.practical-devsecops.com/ai-security-trends-2026/)
25. [Multi-Agent Collaboration Mechanisms Survey (arxiv 2501.06322)](https://arxiv.org/html/2501.06322v1)

---
---

# PART 3: MLOps, AI Systems, and Efficient AI Infrastructure Ideas

*Added: March 13, 2026 -- focused on efficient LLM serving, cost/carbon-aware routing, heterogeneous scheduling, edge-cloud orchestration*

---

## Landscape Summary (2024-2026)

### Efficient LLM Serving
- Inference-time scaling is the dominant 2026 trend: performance gains come from tooling and inference-time compute, not bigger models.
- Smaller, domain-focused models are increasingly used for operational tasks (TinyGPT, Phi-family, Gemma).
- Microservices architectures with MCP/A2A protocols are emerging for GenAI-native systems.

### Heterogeneous Device Scheduling
- Agent.xpu and Puzzle address scheduling LLM ops across CPU/GPU/NPU on mobile SoCs.
- NPUs match or exceed GPU throughput in inference while consuming 35-70% less power.
- The scheduling problem across heterogeneous compute is identified as a "trillion-dollar challenge."

### Carbon-Aware AI Inference
- GAR (NeurIPS 2025) introduced carbon-aware LLM routing with grid intensity data.
- CCWise (NeurIPS 2025) studied carbon-cost regional orchestration.
- Sprout achieved 37.67% reduction in yearly embodied carbon for LLM inference.
- Key gaps: inconsistent energy-metric reporting, no standardized benchmarks, accuracy focus over efficiency.

### LLM Routing and Model Selection
- BEST-Route (ICML 2025): selects model AND number of samples based on difficulty, 60% cost reduction.
- CSCR: contrastive routing embedding cost-awareness into training objective.
- SCORE: constrained optimization under incomplete information for cost+latency.
- Route-to-Reason (RTR): routes across both LLMs and reasoning strategies.
- Key finding: LLMs frequently "overthink" simple queries and "underthink" complex ones.

### Edge AI and On-Device Inference
- KV cache management is critical: for long context, KV cache can exceed model weights in memory.
- Multi-model fallback systems use confidence-based switching but are reactive, not predictive.
- Speculative decoding delivers 2-3x speedups; SLED adapts it for edge.
- Adaptive quantization + scheduling + caching reduces latency by up to 70%.

### Hybrid Edge-Cloud
- Gartner 2025 elevates hybrid computing to top strategic priority.
- Agentic AI in hybrid frameworks: agents use edge for quick decisions, cloud for deep learning.
- Hybrid edge-cloud for agentic AI: up to 75% energy savings, 80% cost reduction vs. pure cloud.

---

## IDEA 13: Carbon-Cost-Aware LLM Router with Real-Time Grid Intensity

### Core Concept
Build a lightweight LLM routing system that jointly optimizes for response quality, monetary cost, AND carbon emissions by incorporating real-time electricity grid carbon intensity data into routing decisions. The router selects among a pool of models (e.g., GPT-4o, Claude Sonnet, Llama-70B, Llama-8B) based on query complexity, cost per token, and the current carbon intensity of the datacenter region serving each model.

### What Exists
- GAR (NeurIPS 2025): carbon-aware routing but single-region, no cost optimization.
- CCWise (NeurIPS 2025): carbon-cost regional orchestration but no query-difficulty awareness.
- CSCR, BEST-Route, SCORE: optimize cost and quality but ignore carbon entirely.
- Sprout: focuses on embodied carbon, not operational carbon from grid mix.

### Novel Angle
No existing work jointly optimizes the three-way tradeoff of quality, cost, and carbon with real-time grid signals AND query-complexity-aware routing. Key insight: a "cheap" model in a high-carbon region may have worse carbon footprint than a "larger" model in a clean-grid region.

### Approach
1. Use Electricity Maps API or WattTime for real-time grid carbon intensity.
2. Build a lightweight BERT-based query difficulty classifier (train on existing routing benchmarks like RouterBench).
3. Formulate as constrained optimization: minimize carbon subject to quality SLO and cost budget.
4. Evaluate on MMLU, MT-Bench, and custom routing benchmarks.

### Metrics
- Carbon emissions (gCO2eq per query), monetary cost, response quality (accuracy/win-rate), latency.
- Pareto frontier analysis across all three objectives.

### Novelty: HIGH | Feasibility: HIGH | Time: 3-4 weeks | Hardware: None (API-based)

---

## IDEA 14: Adaptive Speculative Decoding Across Heterogeneous Edge SoCs

### Core Concept
Design an adaptive speculative decoding system that dynamically selects draft-target model pairs and token speculation lengths based on runtime thermal state, battery level, and processor availability on heterogeneous mobile SoCs (CPU/GPU/NPU).

### What Exists
- SLED: speculative decoding for edge but static strategies, no device-state adaptation.
- Compiler-assisted speculative sampling: addresses heterogeneous devices but static.
- MoE-SpAc: MoE-specific, not general-purpose.

### Novel Angle
No existing work dynamically adapts speculative decoding parameters (speculation length, draft model choice, processor assignment) based on real-time device thermal/power state. On mobile SoCs, thermal throttling degrades NPU performance by 40-60%, making static strategies suboptimal.

### Approach
1. Profile 2-3 edge devices (Snapdragon 8 Gen 3, Apple M-series, Raspberry Pi 5).
2. Implement speculative decoding with llama.cpp or MLC-LLM.
3. Build a lightweight controller monitoring thermal/battery state, adjusting speculation length and processor assignment.
4. Compare against fixed speculation strategies.

### Metrics
- Tokens/second, tokens/joule, time-to-first-token, thermal stability over sustained workloads.

### Novelty: HIGH | Feasibility: MEDIUM | Time: 4-5 weeks | Hardware: 2-3 edge devices

---

## IDEA 15: Predictive Multi-Model Edge Cache Manager

### Core Concept
Build an intelligent model cache manager for resource-constrained edge devices that predicts which models will be needed next and preloads/evicts them based on usage patterns, time-of-day, and task context. Addresses the key bottleneck: loading/unloading models from flash to RAM takes seconds, destroying user experience.

### What Exists
- EdgeAIGC: model caching for generative AI at edge, but reactive.
- Multi-model fallback systems: reactive switching based on confidence.
- KV-cache compression: well-studied, but model-level caching strategies are not.

### Novel Angle
Treat model management as a predictive caching problem (analogous to CPU cache prefetching or CDN caching). Use lightweight temporal/contextual models to predict upcoming model needs and preload during idle periods.

### Approach
1. Simulate a multi-model edge environment with 4-6 quantized models of varying sizes.
2. Collect or synthesize usage traces (time-of-day patterns, task sequences).
3. Implement caching policies: LRU baseline, frequency-based, and a lightweight predictor (small LSTM or rule-based).
4. Measure cold-start latency reduction and memory efficiency.

### Metrics
- Cache hit rate, cold-start latency, average response time, memory utilization, prediction accuracy.

### Novelty: MEDIUM-HIGH | Feasibility: HIGH | Time: 3 weeks | Hardware: Optional (can simulate)

---

## IDEA 16: Difficulty-Aware Hybrid Edge-Cloud Inference Orchestrator

### Core Concept
Build an orchestrator that estimates query difficulty at the edge and decides whether to serve locally (fast, private, cheap) or offload to the cloud (accurate, expensive). Uses a learned difficulty estimator trained to predict when the edge model's answer quality will drop below an acceptable threshold, combining model uncertainty signals with input features.

### What Exists
- Speculative edge-cloud decoding with early exits: token-level decisions.
- Cloud offloading research: fixed confidence thresholds.
- ACAR: 55.6% difficulty estimation accuracy -- significant room for improvement.

### Novel Angle
Combine a calibrated uncertainty estimator on the edge model with a learned difficulty classifier. Key insight: the edge model's own uncertainty signal (token-level entropy or MC dropout) combined with input features predicts offloading benefit more accurately than either alone.

### Approach
1. Deploy small model (Phi-3-mini or Llama-3.2-3B quantized) as edge, larger model (Llama-70B or GPT-4o) as cloud.
2. Train lightweight offloading classifier on: (a) input features from small encoder, (b) edge model uncertainty signals.
3. Evaluate on QA, reasoning, and coding benchmarks.
4. Measure quality-cost-latency Pareto frontier.

### Metrics
- Accuracy vs. offload rate, end-to-end latency, cost per query, bandwidth usage.

### Novelty: MEDIUM-HIGH | Feasibility: HIGH | Time: 3-4 weeks | Hardware: 1 edge device or laptop

---

## IDEA 17: EnergyBench -- Standardized Energy-per-Token Benchmarking for Edge LLMs

### Core Concept
Create an open-source benchmarking framework that measures energy-per-token, tokens-per-joule, and carbon-per-query across different edge devices, models, quantization levels, and runtime backends. Addresses the critical gap of inconsistent energy-metric reporting.

### What Exists
- MLPerf Tiny: benchmarks TinyML, doesn't cover LLMs on edge.
- SHEAB: automates edge AI benchmarking, focuses on latency/throughput.
- ELIB: evaluates LLM inference on edge, limited energy measurement.
- EuroMLSys 2025 paper: advocates energy-per-token as standard metric but provides no framework.

### Novel Angle
First comprehensive, reproducible framework specifically for energy-per-token measurement of LLMs on edge devices, covering the full matrix of: {models} x {quantization levels} x {devices} x {backends} x {workload types}, with a carbon estimation module using regional grid data.

### Approach
1. Instrument power measurement (Intel RAPL, ARM Energy Probe, nvidia-smi) and optionally USB power meter.
2. Test matrix: 3-4 models (Phi-3, Llama-3.2, Gemma-2, Qwen-2.5) x 3 quant levels (FP16, INT8, INT4) x 2-3 devices x 2 backends (llama.cpp, MLC-LLM).
3. Standard workload suite: short QA, long-context summarization, multi-turn chat, code generation.
4. Open-source the framework with reproducible scripts.

### Metrics
- Energy per token (J/token), tokens per joule, peak power, thermal throttling onset time, accuracy at each quant level.

### Novelty: MEDIUM | Feasibility: HIGH | Time: 3-4 weeks | Hardware: 2-3 devices + power meter

---

## IDEA 18: Budget-Aware Reasoning Router -- When to Think Slow vs. Think Fast

### Core Concept
Build a router that decides not just WHICH model to use, but WHETHER to invoke reasoning (chain-of-thought, multi-step) for a given query. Exploits the finding that LLMs frequently "overthink" simple queries and "underthink" complex ones. The router has a token budget and must allocate reasoning compute optimally across a stream of queries.

### What Exists
- Route-to-Reason (RTR): routes across LLMs and reasoning strategies but doesn't handle budget constraints across query streams.
- BEST-Route: selects model and number of samples but doesn't control reasoning mode.
- ACAR: difficulty estimation for ensembling, not reasoning allocation.

### Novel Angle
Frame reasoning allocation as an online knapsack/budgeting problem: given a token budget for a session/time-window, decide per-query whether to use fast (direct answer) or slow (CoT/reasoning) mode, and which model to use. This captures the real-world constraint where inference compute is budgeted, not unlimited.

### Approach
1. Build on an open routing benchmark (RouterBench or LMSYS arena data).
2. Implement a simple online policy: (a) estimate difficulty, (b) estimate reasoning benefit, (c) allocate from remaining budget.
3. Compare against: always-reason, never-reason, random, and oracle baselines.
4. Test with multiple budget levels (50%, 25%, 10% of "always-reason" cost).

### Metrics
- Quality vs. total token spend, budget utilization efficiency, per-query accuracy, reasoning overhead ratio.

### Novelty: HIGH | Feasibility: HIGH | Time: 3 weeks | Hardware: None (API-based)

---

## PART 3 Comparative Assessment

| # | Idea | Novelty | Feasibility | Hardware Needed | Time Est. |
|---|------|---------|-------------|-----------------|-----------|
| 13 | Carbon-Cost-Aware LLM Router | HIGH | HIGH | None (API-based) | 3-4 weeks |
| 14 | Adaptive Speculative Decoding on Edge | HIGH | MEDIUM | 2-3 edge devices | 4-5 weeks |
| 15 | Predictive Model Cache Manager | MED-HIGH | HIGH | Optional (simulate) | 3 weeks |
| 16 | Difficulty-Aware Edge-Cloud Orchestrator | MED-HIGH | HIGH | 1 device or laptop | 3-4 weeks |
| 17 | EnergyBench for Edge LLMs | MEDIUM | HIGH | 2-3 devices + meter | 3-4 weeks |
| 18 | Budget-Aware Reasoning Router | HIGH | HIGH | None (API-based) | 3 weeks |

---

## PART 3 Recommended Strategy

### Best Single Paper: Idea 13 (Carbon-Cost-Aware Router) or Idea 18 (Budget-Aware Reasoning Router)
- Both are highly novel, require no special hardware, and can be completed in 3-4 weeks.
- Idea 13 is more timely (sustainability + EU AI Act alignment).
- Idea 18 is more technically elegant (online optimization + "overthinking" problem).

### Best Combination (two submissions):
- Idea 18 (Budget-Aware Reasoning Router) + Idea 17 (EnergyBench)
- Complementary: method paper + benchmark/resource paper.

### Most Ambitious (if team has edge devices):
- Idea 14 (Adaptive Speculative Decoding) -- highest novelty but needs more time and hardware.

### Safest Bet:
- Idea 16 (Edge-Cloud Orchestrator) -- well-scoped, clear evaluation, proven concept with novel twist.

---

## Cross-Cutting Opportunities Across All Parts

1. **Idea 2 (UQ-Edge) + Idea 16 (Edge-Cloud Orchestrator)**: Use uncertainty quantification from the edge model to drive offloading decisions. Unified system paper spanning trustworthy AI + efficient systems.

2. **Idea 4 (CarbonXAI) + Idea 13 (Carbon-Cost-Aware Router)**: An explainable carbon-cost-quality router with human-readable justifications. Unique intersection of sustainability + XAI + efficient systems.

3. **Idea 7 (AgentSafe-Edge) + Idea 14 (Adaptive Speculative Decoding)**: Safe on-device agent inference with thermal-aware speculation. Combines safety + systems efficiency.

4. **Idea 18 (Budget-Aware Reasoning Router) + Idea 8 (FedSafe-MAS)**: Budget-constrained reasoning in a federated multi-agent system where each agent manages its own compute budget while coordinating safely.

---

## Key References for Part 3 (2025-2026)

1. [GAR: Carbon-Aware Routing for LLM Inference (NeurIPS 2025)](https://openreview.net/forum?id=wVd99lgt4j)
2. [CCWise: Carbon-Cost Aware Regional LLM Orchestration (NeurIPS 2025)](https://neurips.cc/virtual/2025/122414)
3. [BEST-Route: Adaptive LLM Routing (ICML 2025)](https://arxiv.org/abs/2506.22716)
4. [CSCR: Cost-Aware Contrastive Routing (2025)](https://arxiv.org/html/2508.12491v1)
5. [SCORE: Cost and Latency-Constrained Routing (Harvard, 2025)](http://minlanyu.seas.harvard.edu/writeup/sllm25-score.pdf)
6. [Route-to-Reason (2025)](https://arxiv.org/abs/2505.19435)
7. [ACAR: Adaptive Complexity Routing (2026)](https://arxiv.org/html/2602.21231)
8. [SLED: Speculative Decoding for Edge (2025)](https://arxiv.org/abs/2506.09397)
9. [Compiler-Assisted Speculative Sampling (2026)](https://arxiv.org/html/2602.08060)
10. [MoE-SpAc (2026)](https://arxiv.org/html/2603.09983)
11. [Speculative Edge-Cloud Decoding (2025)](https://arxiv.org/abs/2505.21594)
12. [Sprout: Green Generative AI (2025)](https://www.researchgate.net/publication/386183847)
13. [EdgeAIGC: Model Caching for Edge (2025)](https://www.sciencedirect.com/science/article/pii/S2352864825001142)
14. [SHEAB: Edge AI Benchmarking (2025)](https://www.mdpi.com/2227-7080/13/11/515)
15. [Agent.xpu: Scheduling on Heterogeneous SoC (2025)](https://arxiv.org/html/2506.24045v1/)
16. [Puzzle: Multi-DL Scheduling on Mobile (2025)](https://arxiv.org/html/2508.17764v1)
17. [Dynamic Model Routing Survey (2026)](https://arxiv.org/html/2603.04445)
18. [Cross-Attention Routing (2025)](https://arxiv.org/html/2509.09782v1)
19. [Energy-per-Token Advocacy (EuroMLSys 2025)](https://euromlsys.eu/pdf/euromlsys25-27.pdf)
20. [Efficient GenLLM Serving Survey (ACM 2025)](https://dl.acm.org/doi/10.1145/3754448)
21. [Sustainable Carbon-Aware LLM Scheduling (GLSVLSI 2025)](https://dl.acm.org/doi/10.1145/3716368.3735301)
22. [Green AI: Systematic Review (2025)](https://arxiv.org/html/2511.07090v1)
23. [Toward Sustainable Generative AI: Scoping Review (2025)](https://arxiv.org/pdf/2511.17179)
24. [Edge-Cloud Collaborative Computing Survey (2025)](https://arxiv.org/html/2505.01821v2)
25. [On-Device LLMs: State of the Union 2026](https://v-chandra.github.io/on-device-llms/)
26. [NGEN-AI 2026 Conference](https://www.ngen-ai.org/)
