# System One models: the landscape (checked 2026-10-15)

Two research agents and one direct check collected these facts on 2026-10-15. The area changes every week, so check each fact again before you use it in the paper.

- **[P]** = primary source: the vendor's own site, documents, model card or paper.
- **[S]** = secondary source: news, blogs, aggregators or third-party pages.
- **(vendor-reported)** = a number that the vendor measured. Do not give it as a fact.
- **(verified)** = we opened the source ourselves on 2026-10-15 (arXiv metadata or the OpenRouter model list).

---

## 1. The category

### 1.1 Name and definition

TypeSafe AI named the category on 2026-09-15, when it released Jev. The name comes from Kahneman's System 1: fast, automatic judgment. TypeSafe gives chat LLMs and reasoning models the System 2 role.

Exact definitions:

- Launch post: System One models are "a new class of frontier models built to make fast, structured decisions that software can use directly." Jev is "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out." [P](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- Documents: "System One models are a class of AI models built to make fast, structured decisions that software can use directly." "A System One model evaluates a state and returns typed answers and probabilities." "System One models do not write replies, produce code, or generate explanations of their reasoning." [P](https://docs.typesafe.ai/concepts/system-one)

Other names: "decision models" (OpenAI, Microsoft, Perplexity, Simon Willison) and "typed decision models" (Tang and Zheng). In our documents, we use **System One model** for the category.

### 1.2 What a call looks like

The caller sends an **input** (TypeSafe calls it the "state") and a set of named **questions**. Each question declares its answer type. The model returns a typed answer and probabilities for each question. It writes no text.

```json
{
  "model": "jev-latest",
  "state": "Hi, I was charged twice for order 1182. Please fix this today.",
  "questions": {
    "is_refund_request": {"type": "noul"},
    "team": {"type": "choice", "criteria": {"billing": "...", "shipping": "...", "other": "..."}},
    "urgency": {"type": "score", "criteria": ["low", "medium", "high"]}
  }
}
```

This example shows the shape only. Check the exact field names in each API reference.

### 1.3 Answer types

| Type | Answer | Notes |
|---|---|---|
| `noul` | A probability from 0 to 1 that the statement is true | "Noul" stands for Bernoulli (the TypeSafe CEO, [S](https://simonwillison.net/2026/Sep/21/jev/)). OpenAI calls this type `predicate`. |
| `choice` | One option from a fixed set, a probability for each option and a confidence | Up to 255 options (Jev, OpenAI, Perplexity). |
| `score` | A position on an ordered scale, a probability for each level and a confidence | 2 to 10 levels (Jev). The score is the probability-weighted mean of the level indices, so it can fall between levels. |
| `box` | One bounding box in an image | Jev 1.14 only, behind a header flag. [P](https://typesafe.ai/blog/jev-1-14) |

### 1.4 The interface is becoming a standard

- The Jev request shape (`state` plus a map of named questions) is now used by Microsoft, Perplexity, Cloudflare ("fully Jev-API compatible") and OpenRouter (`POST /api/v1/systemone`).
- OpenAI uses its own shape: `input` plus an array of questions, with the types `predicate`, `choice` and `score`.

---

## 2. The four main systems

| | Jev | OpenAI Decisions API | Microsoft-Decision-1 | pplx-decider v1 / v1.1 |
|---|---|---|---|---|
| Vendor | TypeSafe AI | OpenAI | Microsoft | Perplexity |
| Release | 2026-09-15 | Preview 2026-09-29 (DevDay). Public beta 2026-10-06. | 2026-10-09 (preview) | v1: 2026-10-01. v1.1: 2026-10-06. |
| Endpoint | `POST api.typesafe.ai/v1/systemone` | `POST /v1/decisions` | `POST {foundry}/providers/microsoft/v1/systemone` | `POST api.perplexity.ai/v1/decisions` |
| Backbone | Not disclosed. "A new model architecture" with a "parallel sampler". TechCrunch: transformer-based, "not a large language model", trained "exclusively on synthetic data" [S]. | `gpt-6-luna` only (released 2026-09-22, closed) | Qwen3.5-9B, post-trained on public data and Microsoft synthetic data | Qwen3.8-27B (about 26.1B parameters) |
| Open weights | No | No | No | Yes, Apache-2.0 |
| Questions in one call | "No limit" | 1 to 200 (SDK types) | Not documented | 1 to 128 |
| Context | 64k tokens for each request and 32k for each question. 262k with a "Danger-Zone" header (slower, less accurate). | Not documented (`gpt-6-luna`: 1.05M) | 32,768 tokens | Less than 262,144 tokens (trained on 8,192 or less) |
| Modalities | Text. Images only behind a header flag, "not yet up to our standards". | Text and images (up to 128) | Text | Text and images |
| Input price | $0.042 for each 1M tokens | $0.10 for each 1M tokens | $0.042 for each 1M tokens | $0.02 for each 1M tokens (launch: $0.04) |
| Output price | Free | Free | Free | Free |
| Latency claim (vendor-reported) | "70ms-500ms" end to end, measured "from our laptops on the West Coast" | "About 10x faster than the Responses API", no method given | p50 85 ms, p95 125 ms in Foundry (chart values from [S]) | Less than 2 s for a few hundred tokens |
| Calibration method | "RLCD": Reinforcement Learning for Calibrated Decisions. No paper. | Not disclosed | Not disclosed | One temperature, fit by NLL on a held-out fold (T = 1.0087 for v1.1) |
| Accuracy claim (vendor-reported) | No public benchmark scores. Own workflow evals only. | None published | 83.5% against Jev 82.3% on 36 benchmarks (table copy from [S]) | v1: 85.71% against Jev 84.51% on an 11-task panel. v1.1: Decision Index 61.56 against Jev 57.9. |
| OpenRouter id | `typesafe/jev-1.13`, `typesafe/jev-1.14`, `~typesafe/jev-latest` | `openai/gpt-6-luna-decisions` | `microsoft/microsoft-decision-1` | `perplexity/pplx-decider-v1.1-27b` |

### 2.1 Jev (TypeSafe AI)

- **Team:** Diogo Almeida (CEO, previously at OpenAI), Erik Gafni and Sasha Sheng [S](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/). A $40M seed round, led by DCVC. Bloomberg reported about $870M at a $7.5B valuation on 2026-10-09 [S](https://news.bgov.com/us-law-week/andreessen-horowitz-backs-jev-maker-at-7-5-billion-value).
- **Versions:** `jev-latest` is `jev-1.13.0`. `jev-preview` is `jev-1.14.0`, available from 2026-10-14. TypeSafe will move `jev-latest` to 1.14.0 in about one week [P](https://docs.typesafe.ai/models).
- **Isolation:** the model answers the questions "in parallel and in isolation" [P](https://docs.typesafe.ai/api).
- **Access:** TypeSafe removed the waitlist on 2026-09-20 or 2026-09-21 (sources do not agree). It paused signups on 2026-09-22. OpenRouter gives access without a TypeSafe account.
- **Speed and cost claim:** the home page says "193.6x Faster, 444.6x Cheaper" and does not name the baseline. The chart on the same page gives about 75x and 171x by our arithmetic [P](https://typesafe.ai).
- **Workflow evals (vendor-reported):** TypeSafe staff wrote 4 workflows. The reference labels are the mean of GPT-6 Astra and Claude Fable 5.1 with high thinking. Jev: 67.8% agreement, $0.0004 and 0.4 s for each case. GPT-6 Sol: 74.1%. Claude Opus 5: 73.1% [P](https://evals.typesafe.ai).
- **RLCD:** TypeSafe says that the aim is "epistemically honest probabilities" [P](https://typesafe.ai). The documents say that "probabilities are optimized against outcomes to reflect uncertainty" [P](https://docs.typesafe.ai/concepts/system-one). There is no paper, no reward function and no data description. Do not confuse it with RLCD = Reinforcement Learning from Contrastive Distillation (arXiv:2307.12950).
- **Known failure modes:** TypeSafe lists nine for jev-1.13, for example counting, dates, option order and adversarial content [P](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
- **SDKs:** Python (MIT) and JavaScript. An MIT adapter sends the same request to OpenAI, Anthropic or Gemini models for comparison [P](https://github.com/typesafe-ai/system-one-adapter-python).
- **Pricing note:** "We can't prove it isn't subsidized" [P](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

### 2.2 OpenAI Decisions API

- **Schema:** each `choice` option has a `value` and a `description` (2 to 255 values). Each `score` level has a `label` and a `description`. Any answer can have the type `refusal` [P](https://github.com/openai/openai-python/blob/main/src/openai/types/decision.py).
- **Not available:** streaming and the Batch API. Rate limits are not documented [P](https://developers.openai.com/api/docs/guides/decisions).
- **Status:** OpenAI expects general availability "in the coming weeks" [P](https://developers.openai.com/api/docs/guides/decisions).
- **Conflicts:** the guide says images must be base64 data URLs. The SDK accepts public URLs from 2026-10-11. The guide shows `name` as required, but the SDK makes it optional.
- **Forum reports (anecdotal):** Jev was more accurate on nuanced questions. `choice` outputs were overconfident in a biased-coin test [P](https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877).

### 2.3 Microsoft-Decision-1

- **Announcement:** a blog post by Achint Srivastava on 2026-10-09 [P](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/). Microsoft later added Jev rows to the post.
- **Claims (vendor-reported):** "highest accuracy" over 36 benchmarks and about 150,000 held-out questions. A "calibration score" of 92.2 against 93.7 for Jev (the metric is labeled "100 = perfect" but is not defined). Decisions flip on 1.3% of perturbations. The Jev latency in the chart comes from JevBench, so the latency comparison is not like for like [S](https://cellcog.ai/blog/microsoft-decision-1/).
- **Plans:** Microsoft plans to rebase the model on MAI and OpenAI models [P](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/).
- **Responsible use:** do not use the model as the only decision-maker about people (credit, employment, housing, health, legal). Do not use it for surveillance or profiling. The model gives no rationale. Scores change with the wording and the option order [P](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision).
- **Regions:** 9 US and 7 EU regions. No regions in Asia-Pacific, the Middle East or Africa.

### 2.4 pplx-decider (Perplexity)

- **Head:** the backbone feeds a BF16 readout of shape [255, 5120]. There is no language-model head. v1.1 removes the causal mask in the full-attention layers [P](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b).
- **Training (v1.1):** 1 epoch on 626,033 rows (530,103 from tasksource and 95,930 original), learning rate 2e-6, 8 GPUs. v1 used about 73,000 rows.
- **Development results (vendor-reported):** 90.4% accuracy, ECE 0.016 and Brier 0.139 on 1,497 rows.
- **11-task panel (v1):** WinoGrande, FinancialPhraseBank, RAGTruth, JudgeBench, BBH, JevBench public hard, TabFact, ContractNLI, Circa, Belebele and TruthfulQA binary. 7,210 rows, accuracy only. Jev wins 6 of the 11 tasks. The Decider leads only because the overall score is weighted by sample size. The rows are not public [P](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b).
- **Local use:** BF16 weights need about 49 GiB of GPU memory. SGLang supports the model on `/v1/systemone`. vLLM does not: normal causal serving changes the top answer on 4.7% of items [S](https://github.com/sgl-project/sglang/pull/42645). Community GGUF and MLX builds exist.
- **Rate limit:** 10 requests each second for each organization [P](https://docs.perplexity.ai/docs/decisions/quickstart).
- **Open issue:** a forum bug report on 2026-10-13 says that the probabilities were nearly uniform. Its status is unknown.

---

## 3. Other systems

| System | Vendor | Date (2026) | Description | Source |
|---|---|---|---|---|
| Clef, Clef-flash | Cloudflare | 10-01 | 27B and 9B, Qwen-based, Apache-2.0, "fully Jev-API compatible" | [P](https://blog.cloudflare.com/clef-decision-models) |
| Clef-omni | Cloudflare | 10-09 | Qwen3-Omni-30B-A3B. Adds audio and video. | [P](https://blog.cloudflare.com/clef-faster-cheaper-multimodal/) |
| Strands Decider 2B | AWS Strands Labs | 10-01 | Qwen3.5-2B, pointer head, LoRA, open source | [P](https://strandsagents.com/blog/introducing-strands-decider/) |
| Tev1-4B-experimental | Together AI | 09-23 | Supervised fine-tune of Qwen3.5-4B. It writes an option letter through the chat API. | [S](https://featherless.ai/models/togethercomputer/Tev1-4B-experimental) |
| d1; d1-3B, d1-omni-600M | Liquid AI | About 09-29; 10-07 | A hosted model; open edge models | [S](https://runtimewire.com/article/liquid-ai-open-d1-decision-models-edge) |
| GLiNER2.5-Decide; GLiDE | Fastino Labs | 09-24; 09-30 | A 340M open encoder; a hosted model | [S](https://www.marktechpost.com/2026/10/02/decision-ai-models-explained-typesafe-jev-vs-fastino-glide-gliner2-5-decide-and-open-source-competitors/) |
| Solar Decide, Solar Decide Flash | Upstage | 09-28; 10-08 | Hosted | OpenRouter (verified) |
| Mercury Decide | Inception | 09-30 | Hosted, text and images | OpenRouter (verified) |
| Drex v1.5, Drex v2.0 | Nace.AI | 10-09; 10-12 | Hosted | OpenRouter (verified) |
| Span-01, Span-01-lite | Respan | 09-26 | Hosted | OpenRouter (verified) |
| Classify-1 | ScaleDown | 10-13 | Hosted, 1M context, text, images and files | OpenRouter (verified) |
| Kev-4B | Jared Palmer | 09-25 | Open | OpenRouter (verified) |
| H2O-Lightning-4B, Celeris-1 Decision, Kai 1 | H2O.ai, Celeris, Hanzo AI | Sep–Oct | Hosted or open | [S](https://systemonemodels.org/) |
| Laya, CLM-8B, JEV-27B-VL (distilled from Jev 1.13) | Convai, Contrastive-LM, AutoTrust | Sep–Oct | Open | [S](https://redreamality.com/blog/open-decision-models-2026-survey-benchmarks/) |
| Quyet-1.0-Large | Unknown | Unknown | Appears in the Microsoft results table | [S](https://cellcog.ai/blog/microsoft-decision-1/) |
| AnyJev | Nokia Applied Research | 09-30 | A training-free decision readout for any LLM | arXiv:2610.00831 (verified) |

- **Not found on 2026-10-15:** no System One model from Google, Anthropic, Cohere, Mistral, xAI, Alibaba (Qwen is only the backbone of many systems), Groq or Cerebras.
- **Lists:** the GitHub list `awesome-jev` gives more than 100 open reproductions [S](https://github.com/MrJev/awesome-jev). A community site lists many systems [S](https://systemonemodels.org/).
- **Pattern:** most open systems are fine-tunes of Qwen models with a new readout head. This lets us measure the effect of decision training against the same backbone (experiment E5).

---

## 4. Evaluations that already exist

### 4.1 Leaderboards

| Leaderboard | Date | Scope | Main result | Source |
|---|---|---|---|---|
| Artificial Analysis Decision Index v1.0 | 2026-10-13 | Decision models and LLMs | Jev 1.13: 58, the best System One model. GLiDE and Microsoft-Decision-1: 54. Claude Haiku 5.5 (Low): 61. GPT-6 Luna (Low): 55. Jev median time: 0.13 s. System One models are overconfident on knowledge and reasoning tasks. | [S](https://artificialanalysis.ai/articles/artificial-analysis-decision-index) |
| Jev Decision Index (community, not TypeSafe) v0.3.1 and v0.3.2 | 2026-10-10 and 2026-10-12 | About 40 public datasets in 5 areas, chance-corrected. The full score is 20% public and 80% private items. It has a board for images. The public suite and the [runner](https://github.com/apolinario/decision-index) are open. Perplexity reports its own numbers on earlier editions (section 2). | v0.3.1: pplx-decider v1.1 first (62.75), Jev third (60.11). | [S](https://redreamality.com/blog/open-decision-models-2026-survey-benchmarks/), [Space](https://huggingface.co/spaces/multimodalart/jev-decision-index) |
| JevBench v1.6.1 (community, "Benchmark Heaven") | 2026-10-06 | Routing, adequacy judging, yes/no policy questions, intent and ordinal severity. It measures ECE, Brier and latency. The public items use the MIT license. The held-out items are private. | Microsoft and Perplexity use it for their comparisons with Jev. | [S](https://github.com/fstandhartinger/jevbench) |
| Decision Bench (atlanai, bench-v4) | October | 1,071 cases | pplx-decider v1.1: 94.5%. Jev 1.13: 92.4%. Microsoft-Decision-1: 90.7% (text rows only). GPT-6 Luna Decisions: 90.5%. Median latency from 326 ms to 607 ms. | [S](https://github.com/atlanai/decision-bench/blob/main/results/bench-v4/leaderboard.md) |
| AIMultiple | October | 22 models, 1,655 questions | Jev: 47.9, at $0.039 for each 1,000 decisions. Browser tasks: Jev completes about 17 of 50, GPT-6 Astra 47. | [S](https://aimultiple.com/decision-models) |
| Vendor benchmarks | Sep–Oct | Workflow evals (TypeSafe), the 11-task panel (Perplexity), 36 benchmarks (Microsoft), 10 datasets (Cloudflare). OpenAI published no benchmark. | Each vendor's own model is first, or near first, on its own benchmark. Cloudflare: Clef beats Jev on 8 of 10 datasets, and Jev's median latency is 524.1 ms. | See section 2 |

### 4.2 Papers (all arXiv ids verified on 2026-10-15)

| arXiv | First author, title | Date | Main finding |
|---|---|---|---|
| 2609.32160 | Tang and Zheng, "Typed Decision Models: An Early Evidence Audit and Evaluation Checklist" | 09-26 | Reviews 28 papers from 19 to 24 September. Gives a 14-item checklist. **The closest work to our survey.** It covers only the first nine days of Jev. |
| 2609.37647 | Deußer et al., "Evaluating and Benchmarking the System One Model Jev" | 09-29 | 37 datasets, 346,009 requests, less than $10. Jev beats Qwen3.8-27B on 27 of 37 datasets. The choice probabilities are well calibrated. |
| 2610.00346 | Rafe and Das, "Benchmarking System One decision models against trained classifiers and language models ..." | 09-29 | A fine-tuned DeBERTa-v3-large wins on all labeled benchmarks except one. Jev changes 50.5 of 100 answers when the yes/no wording is swapped. |
| 2610.17205 | Geuter et al., "Releasing the Inner Jev: A Simple Recipe for Open Decision Models" | 10-12 | Zero-shot Gemma 4 31B: 79.6. Jev: 80.1. After LoRA, the open model beats Jev. |
| 2609.29769 | Rao and Callison-Burch, "Decision Models vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places" | 09-24 | LLM judges cost 16 to 357 times more. Decision models are significantly worse in at most 6 of 27 comparisons. More than 85% of the errors are the same. |
| 2610.11392 | Sharma et al., "TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models" | 10-08 | Jev is "systematically underconfident" and sensitive to wording. |
| 2609.30243 | Xu et al., "JevOut: Natural Context Can Flip Decision Models" | 09-24 | Short natural context changes 61.4% to 73.2% of correct decisions. |
| 2610.02076 | Li and Wagle, "LLM-as-Jev: LLMs Are Already Jev-Style Decision Models -- When and How to Fine-Tune Them" | 10-01 | Open LLM logits match or beat Jev in some settings. |
| 2610.00831 | Zhang et al., "AnyJev Technical Report" | 09-30 | A training-free decision readout for any LLM. |
| 2610.03935 | Duan et al., "General Decision Models: Benchmarking and Insights Beyond Jev" | 10-02 | The JEVal benchmark. |
| 2610.19118 | Lan et al., "DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision" | 10-13 | A robustness benchmark for text and images. |
| 2609.31142 | Hu et al., "JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models" | 09-25 | Black-box attacks on System One models. |
| 2610.06425 | Li et al., "SoK: Semantic Decision Engines in Network Control Loops" | 10-05 | A systematization for networking only (139 paper families). |
| 2610.01231 | Hill, "Judgement in the Age of Jev: From Evaluation Scarcity to Evaluation Abundance" | 10-01 | A perspective paper. |
| 2609.30216 | Ling et al., "Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem" | 09-24 | A study of 2,170 GitHub projects. |
| 2610.14912 | Lukauskas and Šarkauskaitė, "Probable but Not Valid: A Psychometric Audit of Decision Models as Synthetic Survey Respondents" | 10-10 | Decision models as synthetic survey respondents. |
| 2610.18773 | Li and Wang, "Jeva: System One Models from First Principles" | 10-13 | Design requirements: null-aware, permutation-invariant, option-scalable, cacheable. |
| 2610.18233 | Huang et al., "Consider, Then Choose: Order-Invariant Decisions in System One Models" | 10-13 | The option order changes single-pass answers. |
| 2610.19811 | Messoudi and Mansar, "Calibration and Risk-Budgeted Decisions with System One Models" | 10-14 | Decision thresholds with a controlled risk. |
| 2609.26550 | Li et al., "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure" | 09-22 | Escalation to a reasoning judge when the confidence is low. |
| 2609.33209 | Li et al., "Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?" | 09-27 | Coherence of the probabilities across linked yes/no questions. |

More papers that we did not open yet: 2610.04345 (wireless) and 2609.28940 (penetration testing). A search for "Jev" on arXiv gave 131 results on 2026-10-15 (verified). The full list of seed papers is in `../literature/seed-papers.md`.

---

## 5. Where the evidence does not agree

These conflicts are the most important input for the survey. The paper must explain them, and our experiments must test them.

| Topic | Source A | Source B | Source C | Possible reasons |
|---|---|---|---|---|
| Calibration of Jev | Overconfident on knowledge and reasoning tasks (Artificial Analysis) | "Systematically underconfident" (TypedBench) | Choice probabilities well calibrated (Deußer et al.) | Different task types, answer types, metrics (top-label ECE or class-wise), binning and versions |
| Sensitivity to wording | 50.5 of 100 answers change when the yes/no wording is swapped (Rafe and Das) | 1.3% of decisions flip under perturbations (Microsoft, vendor-reported, own model) | Natural context changes 61.4% to 73.2% of correct decisions (JevOut) | Different perturbations, systems and definitions of a "flip" |
| Accuracy against open alternatives | Jev beats Qwen3.8-27B on 27 of 37 datasets (Deußer et al.) | A fine-tuned DeBERTa-v3-large beats Jev (Rafe and Das) | Zero-shot Gemma 4 31B is equal to Jev, and LoRA beats it (Geuter et al.) | Zero-shot or trained baselines, the amount of labeled data and the datasets |
| Latency of Jev | 70 to 500 ms (TypeSafe, vendor-reported) | Median 0.13 s (Artificial Analysis) | Median 524.1 ms (Cloudflare, a competitor) | Region, gateway, input length, number of questions |
| Ranking | Jev first among System One models (Artificial Analysis) | pplx-decider v1.1 first (Decision Bench, Community Decision Index) | Each vendor first on its own benchmark | Different tasks and weights. Vendor benchmarks favor the vendor. |

---

## 6. Critiques and responses

Critiques:

- "A classifier with the language understanding of a frontier model" [S](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/).
- Theo Browne: "roughly as intelligent as a switch statement" [S](https://www.digitalocean.com/resources/articles/what-is-jev).
- Armin Ronacher: "it delegates the hallucination problem a little bit to the user" [S](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/).
- Anthony Maio: "Jev constrains the shape of the output. It does not constrain the judgment." He also criticizes the workflows that TypeSafe wrote itself and the labels that LLMs made [S](https://anthonymaio.substack.com/p/jev-the-language-model-that-wont).
- Simon Willison: a score comes with no reason, so bias can hide in it [S](https://simonwillison.net/2026/Sep/21/jev/).
- Several papers find that open LLM logits or trained encoders match or beat Jev (section 4.2).

Responses:

- TypeSafe publishes the known failure modes and the biases of its own evals.
- Diogo Almeida: "I get that people think it's a gold rush, but they might be underestimating the difficulty." [S](https://techcrunch.com/2026/10/01/amazon-releases-its-own-jev-clone-as-decision-models-flood-the-web/)
- Supporters: you define the classes in natural language when you call the model, so you need no labeled data. Rao and Callison-Burch show large cost gaps at nearly equal accuracy.

---

## 7. Access for our experiments

**OpenRouter** lists 23 entries with the output modality "decisions" (verified on 2026-10-15 at `https://openrouter.ai/api/v1/models?output_modalities=decisions`). The default model list hides them. The endpoints are `POST /api/alpha/decisions` and the TypeSafe-compatible `POST /api/v1/systemone`. Responses include the probabilities and `usage.cost`.

| OpenRouter id | Listed | Context | Input price for each 1M tokens | Inputs |
|---|---|---|---|---|
| `typesafe/jev-1.13` | 2026-09-18 | 262,144 | $0.042 | text, image |
| `typesafe/jev-1.14` | 2026-10-14 | 262,144 | $0.042 | text, image |
| `~typesafe/jev-latest` | 2026-09-18 | 262,144 | $0.042 | text, image |
| `openai/gpt-6-luna-decisions` | 2026-10-06 | 1,050,000 | $0.100 | text, image |
| `microsoft/microsoft-decision-1` | 2026-10-09 | 32,768 | $0.042 | text |
| `perplexity/pplx-decider-v1.1-27b` | 2026-10-07 | 262,144 | $0.020 | text, image |
| `~perplexity/pplx-decider-latest` | 2026-10-14 | 262,144 | $0.020 | text, image |
| `cloudflare/clef` | 2026-10-01 | 65,536 | $0.042 | text, image |
| `cloudflare/clef-flash` | 2026-10-01 | 65,536 | $0.021 | text, image |
| `cloudflare/clef-omni` | 2026-10-09 | 65,536 | $0.150 | text, image, audio, video |
| `liquid/d1` | 2026-10-01 | 65,536 | $0.020 | text |
| `upstage/solar-decide` | 2026-09-28 | 524,288 | $0.050 | text |
| `upstage/solar-decide-flash` | 2026-10-08 | 524,288 | $0.050 | text |
| `inception/mercury-decide` | 2026-09-30 | 65,536 | $0.020 | text, image |
| `togethercomputer/tev1-4b-experimental` | 2026-09-30 | 32,768 | $0.042 | text |
| `nace-ai/drex-v1.5` | 2026-10-09 | 131,072 | $0.040 | text |
| `nace-ai/drex-v2.0` | 2026-10-12 | 131,072 | $0.060 | text |
| `jaredpalmer/kev-4b` | 2026-09-25 | 8,192 | $0.042 | text |
| `respan/span-01` | 2026-09-26 | Not given | $0.020 | text |
| `respan/span-01-lite` | 2026-09-26 | Not given | $0.000 | text |
| `scaledown/classify-1` | 2026-10-13 | 1,000,000 | $0.040 | text, image, file |

Two more entries are free variants (`:free`) of Span-01-lite and Mercury Decide.

**Result:** the OpenRouter key that we already use for `better-page-index` can call about 18 different System One models through one API. This includes Jev, although TypeSafe paused its own signups.

**Open weights for local runs:** pplx-decider v1 and v1.1 (Apache-2.0, about 49 GiB in BF16, SGLang), Clef and Clef-flash (Apache-2.0), Strands Decider 2B, GLiNER2.5-Decide (340M) and several community models. Check each license.

---

## 8. Facts to check before we write

1. The exact request and response fields of each API (experiment E0).
2. If OpenRouter changes the probabilities or the latency of the systems that it serves.
3. The terms of service of each vendor about benchmarks and the publication of results.
4. The date when `jev-latest` moves to 1.14.0.
5. The status of OpenAI's general availability.
6. New systems and papers: check the arXiv listings and OpenRouter every week.
