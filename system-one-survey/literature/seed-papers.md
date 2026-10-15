# Seed papers (v0, 2026-10-15)

This is the starting list for the review, not the final list. A research agent found these papers on 2026-10-15. We fetched the metadata of every arXiv paper from the arXiv API on the same day, so the titles, authors and ids are correct. The BibTeX entries are in `../paper/references.bib`, and the key is in the first column.

- Venues come from arXiv metadata or Semantic Scholar. Before submission, replace each arXiv entry with the published entry from DBLP.
- Use this list for snowballing (protocol, section 6).
- Add each new work to Zotero, not to this file.

---

## 0. Papers about System One models (September and October 2026)

On 2026-10-15, a search for "Jev" on arXiv gave 131 papers. These are the most important ones for the survey.

| Key | Paper | Why it matters |
|---|---|---|
| `tang2026typed` | Tang and Zheng, "Typed Decision Models: An Early Evidence Audit and Evaluation Checklist", arXiv:2609.32160 | **The closest work to our survey.** It reviews 28 papers from the first nine days. Jev's clearest gains are latency and cost. It found no independent accuracy advantage over label-probability readouts. |
| `ling2026jev` | Ling et al., "Jev in the Wild", arXiv:2609.30216 | How 2,170 GitHub projects use Jev. |
| `li2026sok` | Li et al., "SoK: Semantic Decision Engines in Network Control Loops", arXiv:2610.06425 | A systematization for networking only. |
| `hill2026judgement` | Hill, "Judgement in the Age of Jev", arXiv:2610.01231 | A perspective on the economics of evaluation. |
| `li2026jeva` | Li and Wang, "Jeva: System One Models from First Principles", arXiv:2610.18773 | Design requirements: null-aware, permutation-invariant, option-scalable, cacheable. Use it for the taxonomy. |
| `duan2026general` | Duan et al., "General Decision Models: Benchmarking and Insights Beyond Jev", arXiv:2610.03935 | JEVal: 36 datasets and 25 model configurations. |
| `deusser2026evaluating` | Deußer et al., "Evaluating and Benchmarking the System One Model Jev", arXiv:2609.37647 | 37 datasets. Jev beats Qwen3.8-27B on 27. Choice probabilities are well calibrated. |
| `rafe2026benchmarking` | Rafe and Das, "Benchmarking System One decision models against trained classifiers and language models for automated decision gates", arXiv:2610.00346 | A trained DeBERTa-v3-large wins. Jev changes 50.5 of 100 answers after a yes/no swap. |
| `geuter2026releasing` | Geuter et al., "Releasing the Inner Jev", arXiv:2610.17205 | Letter logits of open LLMs give similar results without training. LoRA beats Jev. |
| `li2026llmasjev` | Li and Wagle, "LLM-as-Jev", arXiv:2610.02076 | LLMs are already decision models. When and how to fine-tune them. |
| `zhang2026anyjev` | Zhang et al., "AnyJev Technical Report", arXiv:2610.00831 | A training-free decision readout for any LLM. |
| `rao2026decision` | Rao and Callison-Burch, "Decision Models vs. LLMs as Rubric Judges", arXiv:2609.29769 | 16 to 357 times cheaper than LLM judges, with the same errors. |
| `sharma2026typedbench` | Sharma et al., "TypedBench", arXiv:2610.11392 | Jev is "systematically underconfident" and sensitive to framing. |
| `xu2026jevout` | Xu et al., "JevOut: Natural Context Can Flip Decision Models", arXiv:2609.30243 | Short natural context changes 61.4% to 73.2% of correct decisions. |
| `huang2026consider` | Huang et al., "Consider, Then Choose: Order-Invariant Decisions in System One Models", arXiv:2610.18233 | The option order changes single-pass answers. |
| `li2026beyond` | Li et al., "Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?", arXiv:2609.33209 | Coherence across linked yes/no questions. |
| `messoudi2026calibration` | Messoudi and Mansar, "Calibration and Risk-Budgeted Decisions with System One Models", arXiv:2610.19811 | Thresholds with a controlled risk. |
| `li2026jevasajudge` | Li et al., "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure", arXiv:2609.26550 | Escalation to a reasoning judge when the confidence is low. Related to our E6. |
| `hu2026jevadvbench` | Hu et al., "JevAdvBench", arXiv:2609.31142 | Black-box attacks. |
| `lan2026decisionrobust` | Lan et al., "DecisionRobust", arXiv:2610.19118 | Robustness for text and images. |
| `lukauskas2026probable` | Lukauskas and Šarkauskaitė, "Probable but Not Valid", arXiv:2610.14912 | Decision models as synthetic survey respondents. |

---

## 1. Fast and slow systems

| Key | Paper | Why it matters |
|---|---|---|
| `kahneman2011thinking` | Kahneman, *Thinking, Fast and Slow*, 2011 | The source of the System 1 and System 2 terms. |
| `lin2023swiftsage` | Lin et al., "SwiftSage", NeurIPS 2023 | A small model acts. An LLM plans only when necessary. |
| `yu2024distilling` | Yu et al., "Distilling System 2 into System 1", 2024 | Trains direct answers from reasoning outputs. |
| `figure2025helix` | Figure AI, "Helix", 2025 (company report) | A 7B VLM (System 2, 7 to 9 Hz) drives an 80M policy (System 1, 200 Hz). |
| `nvidia2025gr00t` | NVIDIA et al., "GR00T N1", 2025 | A VLM module (System 2) conditions a diffusion action module (System 1). |
| `zhang2025adaptthink` | Zhang et al., "AdaptThink", EMNLP 2025 | Reinforcement learning teaches a model when to skip thinking. |
| `li2025system` | Li et al., "From System 1 to System 2: A Survey of Reasoning Large Language Models", IEEE TPAMI | **Survey.** It treats System 1 as a stage to leave. We treat fast typed decisions as the goal. |

## 2. LLMs as classifiers

| Key | Paper | Why it matters |
|---|---|---|
| `schick2020exploiting` | Schick and Schütze, "Exploiting Cloze Questions ...", EACL 2021 | PET: cloze patterns and verbalizers. |
| `min2021noisy` | Min et al., "Noisy Channel Language Model Prompting ...", ACL 2022 | Scores the input for each label. |
| `zhao2021calibrate` | Zhao et al., "Calibrate Before Use", ICML 2021 | Label bias and contextual calibration. |
| `robinson2022leveraging` | Robinson et al., "Leveraging Large Language Models for Multiple Choice Question Answering", ICLR 2023 | One-pass scoring of option symbols. |
| `zheng2023large` | Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024 | Selection bias toward option ids. |
| `pezeshkpour2023large` | Pezeshkpour and Hruschka, "Large Language Models Sensitivity to The Order of Options ...", 2023 | Option-order sensitivity. |
| `wang2024answer` | Wang et al., "'My Answer is C'", Findings of ACL 2024 | First-token probabilities can disagree with the text answer. |
| `liu2021pretrain` | Liu et al., "Pre-train, Prompt, and Predict", 2021 | **Survey** of prompting, with verbalizers ("answer engineering"). |

## 3. Calibration and uncertainty

| Key | Paper | Why it matters |
|---|---|---|
| `brier1950verification` | Brier, 1950, Monthly Weather Review | The Brier score. |
| `guo2017calibration` | Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 | ECE and temperature scaling (pplx-decider uses it). |
| `kadavath2022language` | Kadavath et al., "Language Models (Mostly) Know What They Know", 2022 | Option probabilities are well calibrated with the right format. |
| `tian2023just` | Tian et al., "Just Ask for Calibration", EMNLP 2023 | After RLHF, stated confidence can be better than token probabilities. |
| `kumar2023conformal` | Kumar et al., "Conformal Prediction with Large Language Models for Multi-Choice Question Answering", 2023 | Option sets with coverage guarantees. |
| `damani2025beyond` | Damani et al., "Beyond Binary Rewards", 2025 | RLCR: reinforcement learning with calibration rewards. A public method with the aim of RLCD. |
| `geng2023survey` | Geng et al., "A Survey of Confidence Estimation and Calibration in Large Language Models", NAACL 2024 | **Survey.** |
| `liu2025uncertainty` | Liu et al., "Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey", KDD 2025 | **Survey.** |
| `wen2024know` | Wen et al., "Know Your Limits: A Survey of Abstention in Large Language Models", TACL | **Survey** of abstention. |

## 4. Structured outputs and constrained decoding

| Key | Paper | Why it matters |
|---|---|---|
| `willard2023efficient` | Willard and Louf, "Efficient Guided Generation for Large Language Models", 2023 | Outlines. |
| `dong2024xgrammar` | Dong et al., "XGrammar", MLSys 2025 | A fast engine for JSON schemas and grammars. |
| `park2024grammaraligned` | Park et al., "Grammar-Aligned Decoding", NeurIPS 2024 | Constrained decoding changes the output distribution. Important when the output is a probability. |
| `tam2024let` | Tam et al., "Let Me Speak Freely?", EMNLP 2024 | Format restrictions lower the accuracy of reasoning. |
| `geng2025jsonschemabench` | Geng et al., "JSONSchemaBench", 2025 | A benchmark of constrained-decoding engines. No verified survey exists for this line. |

## 5. Scalar judges

| Key | Paper | Why it matters |
|---|---|---|
| `zheng2023judging` | Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023 | Defines LLM-as-a-judge and its biases. |
| `liu2023geval` | Liu et al., "G-Eval", EMNLP 2023 | Scores weighted by probabilities: an early form of the score type. |
| `kim2023prometheus` | Kim et al., "Prometheus", ICLR 2024 | An open rubric grader. |
| `lambert2024rewardbench` | Lambert et al., "RewardBench", 2024 | A benchmark for scalar reward models. |
| `malik2025rewardbench` | Malik et al., "RewardBench 2", 2025 | A harder benchmark. We use it in F6. |
| `zhang2024generative` | Zhang et al., "Generative Verifiers", ICLR 2025 | The probability of "Yes" as the reward. |
| `lightman2023let` | Lightman et al., "Let's Verify Step by Step", ICLR 2024 | Process reward models. |
| `gu2024survey` | Gu et al., "A Survey on LLM-as-a-Judge", 2024 | **Survey.** |

## 6. Guard and moderation models

| Key | Paper | Why it matters |
|---|---|---|
| `markov2022holistic` | Markov et al., "A Holistic Approach to Undesired Content Detection in the Real World", AAAI 2023 | The OpenAI moderation classifier. |
| `inan2023llama` | Inan et al., "Llama Guard", 2023 | A classifier with a taxonomy and decision scores. |
| `zeng2024shieldgemma` | Zeng et al., "ShieldGemma", 2024 | One classifier for each type of harm. |
| `han2024wildguard` | Han et al., "WildGuard", NeurIPS 2024 | One model makes three decisions. |
| `padhi2024granite` | Padhi et al., "Granite Guardian", 2024 | Adds jailbreak and groundedness risks. |
| `openai2025gptoss120b` | OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card", 2025 | The base of gpt-oss-safeguard, which reasons over a policy in the prompt. |
| `chennabasappa2025llamafirewall` | Chennabasappa et al., "LlamaFirewall", 2025 | Includes the PromptGuard 2 detector. |
| `dong2024safeguarding` | Dong et al., "Safeguarding Large Language Models: A Survey", Artificial Intelligence Review | **Survey.** |

## 7. Routing and cascades

| Key | Paper | Why it matters |
|---|---|---|
| `chen2023frugalgpt` | Chen et al., "FrugalGPT", TMLR | An LLM cascade with a learned scorer. |
| `ong2024routellm` | Ong et al., "RouteLLM", 2024 | Routes between a strong and a weak model. |
| `gupta2024language` | Gupta et al., "Language Model Cascades: Token-level uncertainty and beyond", ICLR 2024 | Deferral rules from token uncertainty. |
| `nie2024online` | Nie et al., "Online Cascade Learning for Efficient Inference over Streams", ICML 2024 | Cheap classifiers learn from an LLM. |
| `hu2024routerbench` | Hu et al., "RouterBench", 2024 | An early router benchmark. |
| `li2026llmrouterbench` | Li et al., "LLMRouterBench", ACL 2026 | A large router benchmark. |
| `moslem2026dynamic` | Moslem and Kelleher, "Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey", TMLR | **Survey.** |

## 8. Encoder classifiers

| Key | Paper | Why it matters |
|---|---|---|
| `devlin2018bert` | Devlin et al., "BERT", NAACL 2019 | An encoder with a classification head. |
| `he2021debertav3` | He et al., "DeBERTaV3", ICLR 2023 | The strongest common trained baseline. |
| `warner2024smarter` | Warner et al., "Smarter, Better, Faster, Longer" (ModernBERT), ACL 2025 | A modern encoder with 8,192 tokens. |
| `breton2025neobert`, `marone2025mmbert`, `weller2025seq` | NeoBERT, mmBERT, Ettin ("Seq vs Seq") | Newer encoders. |
| `tunstall2022efficient` | Tunstall et al., "Efficient Few-Shot Learning Without Prompts" (SetFit), 2022 | Few-shot baseline in E4. |
| `yin2019benchmarking` | Yin et al., "Benchmarking Zero-shot Text Classification", EMNLP 2019 | NLI as a zero-shot classifier: an early promptable classifier. |
| `nogueira2020document` | Nogueira et al., "Document Ranking with a Pretrained Sequence-to-Sequence Model", Findings of EMNLP 2020 | monoT5: relevance from the probability of "true". |
| `zhang2025bertlike` | Zhang et al., "Do BERT-Like Bidirectional Models Still Perform Better on Text Classification in the Era of LLMs?", EMNLP 2025 | BERT-like models often beat LLMs at classification. |
| `minaee2020deep` | Minaee et al., "Deep Learning Based Text Classification: A Comprehensive Review", ACM CSUR | **Survey** (before LLMs). |

## 9. Efficient one-pass inference

| Key | Paper | Why it matters |
|---|---|---|
| `kwon2023efficient` | Kwon et al., "PagedAttention" (vLLM), SOSP 2023 | Our engine for the readout baselines. |
| `zheng2023sglang` | Zheng et al., "SGLang", NeurIPS 2024 | The engine for local pplx-decider runs. |
| `gim2023prompt` | Gim et al., "Prompt Cache", MLSys 2024 | Reuse of attention states. |
| `cheng2023batch` | Cheng et al., "Batch Prompting", EMNLP 2023 Industry | Many items in one prompt: related to many questions in one call. |
| `du2025prefillonly` | Du et al., "PrefillOnly", SOSP 2025 | An engine for workloads that output one token. |
| `reddy2024first` | Reddy et al., "FIRST", EMNLP 2024 | Ranks all candidates from first-token logits. |
| `li2024survey` | Li et al., "A Survey on Large Language Model Acceleration based on KV Cache Management", TMLR | **Survey.** |

## 10. Robustness and consistency

| Key | Paper | Why it matters |
|---|---|---|
| `greshake2023what` | Greshake et al., "Not what you've signed up for", AISec 2023 | Indirect prompt injection through data. |
| `shi2024optimizationbased` | Shi et al., "Optimization-based Prompt Injection Attack to LLM-as-a-Judge", CCS 2024 | Optimized injections change the choice of a judge. |
| `raina2024llmasajudge` | Raina et al., "Is LLM-as-a-Judge Robust?", EMNLP 2024 | Absolute scores are more vulnerable than comparisons. |
| `zhao2025one` | Zhao et al., "One Token to Fool LLM-as-a-Judge", 2025 | "Master key" tokens. |
| `ye2024justice` | Ye et al., "Justice or Prejudice?", ICLR 2025 | 12 judge biases. |
| `kassner2021beliefbank` | Kassner et al., "BeliefBank", EMNLP 2021 | Consistency of yes/no beliefs. |
| `paleka2024consistency` | Paleka et al., "Consistency Checks for Language Model Forecasters", 2024 | Coherence checks correlate with Brier scores. |
| `novikova2025consistency` | Novikova et al., "Consistency in Language Models", 2025 | **Survey** (workshop). |

---

## Surveys that overlap with ours

| Survey | How our survey is different |
|---|---|
| Tang and Zheng (`tang2026typed`) | It covers 28 papers from one week. We cover all systems until the cut-off, the lineage, a taxonomy, the evidence synthesis and experiments. |
| Ling et al. (`ling2026jev`) | It studies how projects use one product. We compare systems and evidence. |
| Li et al. (`li2025system`) | It treats System 1 as a stage to leave. We treat fast typed decisions as the goal. |
| Sui et al. (`sui2025stop`) | It makes generated reasoning shorter. System One models remove generation completely. |
| Geng et al. (`geng2023survey`) and Liu et al. (`liu2025uncertainty`) | They study confidence for free-text answers. We treat calibrated typed probabilities as an API contract. |
| Gu et al. (`gu2024survey`) | Those judges write rationales. System One judges return only distributions, so evaluation and attacks are different. |
| Moslem and Kelleher (`moslem2026dynamic`) | It routes queries among generative LLMs. A System One model is both the router and the cheap tier. |
