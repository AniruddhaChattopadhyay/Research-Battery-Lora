# BatteryLoRA — Research Directions & Options Going Forward

**Status as of 2026-03-15**: Paper-scale experiments complete. Results are mixed — see honest assessment below.

---

## Current Results Summary

| Method | Final Loss | Energy (Wh) | Min Battery |
|--------|-----------|-------------|-------------|
| **BatteryLoRA (threshold)** | 0.973 | 33.7 | 33.7% |
| **BatteryLoRA (continuous)** | 0.409 | 46.4 | 23.4% |
| HomLoRA r=8 | 0.582 | 30.0 | 27.6% |
| HomLoRA r=32 | 0.513 | 90.0 | 11.2% |
| HetLoRA | 0.262 | 51.9 | 17.9% |

## The Problem

Our main threshold policy (0.973 loss) is **worse than every baseline** on loss. HomLoRA r=8 achieves better loss (0.582) AND lower energy (30.0 Wh). We don't win on either axis.

The continuous policy (0.409) is better but still loses to HetLoRA (0.262) which doesn't consider battery at all.

**Root cause**: Convergence instability from heterogeneous rank mixing. BatteryLoRA reached 0.42 loss at round 20 — then collapsed back to 0.97 by round 30. The rank changes across rounds (a client going from rank 4 one round to rank 16 the next) disrupt the FLoRA aggregation. The global adapter's subspaces get partially overwritten each round.

The continuous policy's relative success (0.409) confirms the general approach *can* work — abrupt rank changes in the threshold policy are the specific problem.

---

## Option A: Fix BatteryLoRA's Convergence and Re-run

**Effort**: ~1-2 weeks of code changes + another Colab A100 session (~18 hours)
**Risk**: Medium — fixes are theoretically sound but not guaranteed to close the gap

### Potential Fixes

**A1. Rank Smoothing**
Don't let a client's rank change by more than one step per round. If the policy says jump from rank 4 to rank 16, instead go 4→8 this round, 8→16 next round. This prevents the large subspace disruptions that cause the aggregation instability.

```
Current:  round 5: rank 4 → round 6: rank 16 (abrupt, disrupts subspaces)
Smoothed: round 5: rank 4 → round 6: rank 8 → round 7: rank 16 (gradual)
```

**A2. Momentum/EMA in Aggregation**
Instead of directly replacing the global adapter with the aggregated result, use exponential moving average:

```
new_global = α × old_global + (1-α) × aggregated_result    (α = 0.5-0.8)
```

This prevents any single round's aggregation from drastically changing the global adapter. Well-established technique in federated learning (used in FedAvgM).

**A3. Subspace Freezing**
Once higher dimensions (rows 8-32 of the A matrix) have been trained by high-rank clients, lock them so low-rank client updates (which only touch rows 0-7) can't indirectly interfere. Only unlock when a high-rank client participates again.

**A4. Rank-Weighted Aggregation**
Give higher-rank clients more influence in the aggregation. A client using rank 32 has seen more of the subspace than a rank 2 client, so its contribution should carry more weight.

**A5. Warm-up Period**
Start all clients at their maximum tier rank for the first N rounds (like HetLoRA), then gradually introduce battery-aware rank reduction. This lets the global adapter stabilize before ranks start changing.

### What to Re-run

If fixes are implemented:
1. Re-run BatteryLoRA (threshold + smoothing) — seed 42
2. Re-run BatteryLoRA (threshold + EMA aggregation) — seed 42
3. Keep existing baseline results (no need to re-run those)
4. If results improve, run with seeds 123, 456 for statistical significance

---

## Option B: Pivot the Paper Angle

**Effort**: ~1 week (reframing + minor additional analysis)
**Risk**: Low execution risk, but the contribution may be perceived as weak

### B1. Present Continuous Policy as Main Method
- Lead with continuous policy (0.409 loss) instead of threshold
- Claim: "comparable quality to HomLoRA r=8 (0.582 vs 0.409) with dynamic battery-aware rank adaptation"
- Weakness: HetLoRA still beats us on loss (0.262), and we don't clearly win on energy either

### B2. Frame as a Negative/Analytical Result
- Title: "On the Challenges of Dynamic Rank Adaptation in Federated LoRA: A Battery-Aware Case Study"
- Present the convergence instability as the finding itself
- Analyze WHY rank-changing disrupts aggregation (with the round-by-round evidence)
- Propose fixes (A1-A5 above) as future work
- Weakness: Negative results are harder to publish, especially at a conference

### B3. Reframe as Energy-Efficiency Study
- Focus claim on battery preservation: "BatteryLoRA keeps all devices above 33% battery vs HomLoRA-r32 nearly killing a client at 11%"
- Show the rank adaptation trajectories as the contribution (no prior work shows this behavior)
- Acknowledge loss gap and propose fixes
- Weakness: Battery preservation alone may not be a strong enough contribution

---

## Option C: Pick a Different Research Idea

**Effort**: 4-6 weeks depending on the idea
**Risk**: Starting fresh, but plenty of time before May 25 deadline (~10 weeks)

We have 17 other research ideas documented in `NGEN-AI-2026-research-ideas.md`. The strongest candidates that reuse our existing infrastructure (Flower, PEFT, edge AI expertise):

### C1. UQ-Edge — Uncertainty Quantification for On-Device SLMs
- **Novelty**: HIGH — no published work on UQ for sub-1B quantized models
- **Feasibility**: HIGH — 25-30 days, uses similar tooling
- **Reuse from BatteryLoRA**: Model loading pipeline, quantization setup, evaluation framework
- **Why it might work better**: Single clear metric (calibration error), no aggregation instability issues

### C2. SafeEdgeAgents — Runtime Safety for Edge Agents
- **Novelty**: HIGH — no safety enforcement for resource-constrained edge agents
- **Feasibility**: HIGH — 25-30 days, fastest to implement
- **Reuse from BatteryLoRA**: Edge AI experience, model deployment knowledge
- **Why it might work better**: Well-scoped problem with clear baselines (AgentSpec)

### C3. ExplainFL — Federated Explainability Aggregation
- **Novelty**: HIGH — nobody federates explanations, only model weights
- **Feasibility**: HIGH — 30-35 days, uses Flower framework
- **Reuse from BatteryLoRA**: Flower setup, federated training pipeline, aggregation code
- **Why it might work better**: Directly extends our FL expertise, novel contribution is cleaner

### C4. GreenAgents — Energy-Aware Multi-Agent Orchestration
- **Novelty**: MEDIUM-HIGH — dynamic SLM selection for agents
- **Feasibility**: HIGH — 25-30 days
- **Reuse from BatteryLoRA**: Energy measurement approach, battery/energy simulation
- **Why it might work better**: Hot topic (agentic AI), clear energy savings metric

### C5. Budget-Aware Reasoning Router
- **Novelty**: HIGH — no existing work on reasoning budget allocation
- **Feasibility**: HIGH — 3 weeks, API-based (no hardware needed)
- **Reuse from BatteryLoRA**: Minimal, but the optimization mindset transfers
- **Why it might work better**: Clean problem formulation, clear baselines, no hardware dependencies

---

## Option D: Combine — Fix + Pivot

**Effort**: 2-3 weeks
**Risk**: Low — hedges between improving BatteryLoRA and having a backup

### Strategy
1. **Week 1**: Implement fixes A1 (rank smoothing) + A2 (EMA aggregation) — these are the simplest and most likely to help
2. **Week 1-2**: Re-run fixed BatteryLoRA on Colab
3. If results improve (loss < 0.5, competitive with HomLoRA r=8):
   - Continue with BatteryLoRA paper, present fixed version
4. If results don't improve:
   - Pivot immediately to Option C1 (UQ-Edge) or C5 (Budget-Aware Router)
   - Still have 7-8 weeks before deadline

---

## Recommendation

**Go with Option D (Combine).**

Rationale:
- The BatteryLoRA fix (especially rank smoothing + EMA) is quick to implement and test
- The continuous policy's success (0.409) proves the approach can work — we just need smoother rank transitions
- If the fix works, we have a novel contribution with strong experimental support
- If it doesn't, we pivot early enough to complete a different paper
- The May 25 deadline gives us enough runway for one pivot

### Decision Timeline
- **By March 22**: Implement fixes A1+A2, re-run on Colab
- **By March 25**: Evaluate new results. Go/no-go decision on BatteryLoRA
- **If go**: Write paper through April, polish in May
- **If no-go**: Start UQ-Edge or Budget-Aware Router by March 26
