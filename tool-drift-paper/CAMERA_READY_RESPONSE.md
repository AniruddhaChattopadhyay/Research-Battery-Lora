# NGEN-AI 2026 Camera-Ready Review Traceability

## Review 1

### Candidate-set retry was difficult to understand

**Response:** Rewrote the method and protocol descriptions in chronological
order. The manuscript now distinguishes an initial call with a visible tool name
from one with a missing name, explains the candidate-list retry, and states when
validation and constrained repair occur.

### “Load-bearing” was difficult to understand

**Response:** Removed “load-bearing” and “heavy lifting.” The results now say
directly that candidate-set retry is essential to the evaluated non-oracle
protocol and that structured validation supplies the principal repair benefit.

### The oracle protocol needed explanation

**Response:** Defined non-oracle and oracle-constrained protocols in the problem
formulation. Oracle-constrained runs receive the benchmark gold target tool and
are labeled as upper bounds rather than deployment-facing evidence.

### Latency is substantial and stronger models benefit less

**Response:** Retained token overhead and latency measurements, labeled the
latency evidence as one instrumented rerun, and stated that systematic
cross-provider/model cost remains unmeasured. The GPT-4o-mini result is described
as a small, non-significant sanity-check gain rather than confirmatory evidence.

## Review 2

### Broader model evaluation

**Response:** Made the existing coverage explicit in Experimental Setup:
Qwen3.5-9B, Qwen3.5-35B-A3B, Llama-4-Scout,
Mistral-Small-3.2-24B, and GPT-4o-mini. The manuscript separates full
non-oracle evidence from oracle-constrained sweeps and retains the limitation
that the strongest non-oracle results remain in the Qwen family.

### Real API histories

**Response:** Retained the 11-item Notion/Stripe changelog audit but narrowed its
interpretation. It is now a plausibility check, not a sampled distribution of
production API evolution, and the drift operators remain explicitly
author-designed.

### Multi-step agents and larger tool ecosystems

**Response:** Added an explicit limitation: DICE provides multi-round context and
BFCL includes multiple-function categories, but the evaluation scores each
example rather than executing sequential tool plans. Compounding recovery and
production-scale tool catalogs remain future work.

## Review 3

### Model-family portability

**Response:** Preserved the positive Qwen3.5-35B-A3B replication and the weaker
Llama-4-Scout result. The manuscript now states more directly that missing tool
names are a model-dependent portability bottleneck and that same-family size
replication is not broad portability evidence.

### Synthetic ecological validity

**Response:** Narrowed the changelog-audit claim and retained the limitation that
the perturbations were not mined from an empirical change distribution.

### Token, latency, and API cost

**Response:** Audited the released artifacts and corrected the repair-stage
accounting. The primary non-oracle invocation rates are 63/300 (21.0%) on DICE
and 73/200 (36.5%) on BFCL; earlier summary fields counted only resolved repair
targets. The instrumented BFCL rerun invokes the repair stage on 63/200 examples
with 13.0 s conditional mean latency, yielding 4.1 s measured added repair-stage
wall time per example. The 32.6% and 47.1% token-overhead figures remain valid.

### Baseline patching versus drift recovery

**Response:** Retained the gain-decomposition table and added the exact
decomposition to the conclusion: DICE has 19 drift recoveries and 10 baseline
patches among 29 gains; BFCL has 33 drift recoveries and 12 baseline patches
among 45 gains.

## Review 4

No actionable review text was supplied.

