# Related-work facts: System One decision-model papers (verified 2026-10-16)

## How I got these facts

- I read each paper online on 2026-10-16.
- For each paper, I downloaded three sources:
  - the abstract page, `https://arxiv.org/abs/<id>`;
  - the HTML full text, `https://arxiv.org/html/<id>` (latest version);
  - the arXiv API record, `https://export.arxiv.org/api/query?id_list=<id>`.
- I used `curl` and a local HTML-to-text step, not WebFetch. Reason: WebFetch returns a summary from a small model. Exact quotes need the exact text.
- All ten papers have an HTML full text. I read the full text of all ten.
- For paper 2, the HTML does not show the tables of Appendix C. I also read the PDF (`https://arxiv.org/pdf/2610.00346v2`) for Tables C.10 and C.12.
- Quotes are exact. I show math in plain text (for example, "P(yes)" for the LaTeX formula, "±" for `\pm`).
- "Not found" means that the text I read does not give the fact.
- "My calculation" marks a number that I calculated from numbers in the paper. The paper does not print that number.

---

## 1. Deußer, Sparrenberg, Sifa: "Evaluating and Benchmarking the System One Model Jev" (arXiv 2609.37647)

- Read: full text (HTML v1).
- URLs: https://arxiv.org/abs/2609.37647 and https://arxiv.org/html/2609.37647
- arXiv v1: 29 Sep 2026. Primary category: cs.CL.

### Jev version and dates

- Version: "We pinned the model version jev-1.13.0 for all requests."
- The authors accessed the vendor documentation on 2026-09-29 (footnote 1 and reference [6]).
- Exact date of the Jev run: Not found. The paper gives only the duration: "The full evaluation took 5 h 15 min."
- Price: "Jev is priced at US$0.042 per million input tokens (output tokens are free)".
- Size and cost: 346,009 requests; the main run cost US$9.15 (Table 7).

### Datasets (37 datasets in 7 groups, one frozen template for each dataset, full evaluation splits)

- Text classification: AG News, IMDB, Rotten Tomatoes, SST-2, Emotion, Financial PhraseBank, SMS Spam, Language ID, GoEmotions.
- Intent and topic routing: Banking77, CLINC150, SIB-200.
- NLI, paraphrase and grounding: ANLI, AfriXNLI, PAWS, LLM-AggreFact.
- Reading comprehension and knowledge: BoolQ, Belebele, PubMedQA, MMLU, C-Eval.
- Commonsense and reasoning: BIG-bench (MC; 93 tasks, 13,228 examples), HellaSwag, WinoGrande, ARC (E+C), CommonsenseQA, αNLI (ART).
- Safety, moderation and legal: ToxiGen, OpenAI Moderation, ToxicChat, Prompt Injections, AGB-DE, UNFAIR-ToS.
- Rubric scoring: STS-B, SST-5, SummEval, HelpSteer2.
- Reference models: Qwen3.8-27B and Gemma-4-E4B. The authors read the next-token distribution over single-token option codes and renormalize it ("restrict to the codes and renormalize").

### How they measure calibration

- Quote: "Calibration is measured with the ECE over 15 equal-width bins [19], using the top-option probability for Choices and P(yes) for Nouls, and with the Brier score [20]."
- Choice questions: top-label ECE (top-option probability).
- Noul (yes/no) questions: ECE on P(yes). This is a positive-class measure, not a top-label measure.
- Score questions: no ECE. Tables 2 and 3 show "–" for Score-only tasks.
- Tables 2 and 3 report the "ECE of the first Choice or Noul question".
- Figure 2 shows three separate pooled reliability diagrams: Choice, single Noul, and multi-label Noul.
- Class-wise ECE: Not found.
- Brier-score values: Not found. The paper names the Brier score as a metric, but the HTML text gives no Brier number.

### Calibration results

- Choice: "Pooled over 22 Choice datasets and 279,925 answers, the ECE is 0.028."
- Choice: "the median per-dataset ECE is likewise 0.028, while the unweighted mean is 0.061."
- Choice: "Where there is miscalibration, it is almost always overconfidence: the mean top-option probability exceeds accuracy on 20 of the 22 datasets."
- Reference models: "Qwen’s pooled Choice probabilities are less well calibrated (ECE 0.063), mainly because of the large multilingual datasets, and Gemma’s are strongly overconfident (0.208; mean top-option probability 0.936 at 72.8% accuracy)."
- Per-dataset mean: "Averaged over the 33 datasets with a Choice or Noul question (Table 3), Jev (0.074) and Qwen (0.075) are on par, and both are far better calibrated than Gemma (0.184)."
- Single Noul: "Over eight binary datasets, the mean predicted P(yes) is 0.465 against an observed yes-rate of 0.518, and the reliability curve lies above the diagonal (ECE 0.052)." The paragraph heading calls this "slightly under-confident".
- Multi-label Noul: "the mean P(yes) is 0.209 against an observed rate of 0.041 (ECE 0.168)."
- Thresholds (abstract): "thresholds tuned on training data raise micro-F1 on UNFAIR-ToS from 0.50 to 0.75."

### Other facts

- "Jev scores higher than Qwen3.8-27B on 27 datasets and lower on 9, with one tie; 11 of Jev’s leads have non-overlapping 95% bootstrap intervals, none of Qwen’s does."
- MMLU probe (Table 6): calculation-heavy subjects 0.943 for Jev; 0.943 with rotated options; 0.315 with the question withheld.

---

## 2. Rafe, Das: "Benchmarking System One decision models against trained classifiers and language models for automated decision gates" (arXiv 2610.00346)

- Read: full text (HTML v2) and PDF v2 (for Appendix C Tables C.10 and C.12).
- URLs: https://arxiv.org/abs/2610.00346 , https://arxiv.org/html/2610.00346 , https://arxiv.org/pdf/2610.00346v2
- v1: 29 Sep 2026. v2: 6 Oct 2026. Primary category: cs.LG.

### Jev version and dates

- "Jev is accessed through the vendor’s typesafe-sdk pinned to jev-1.13.0, since the alias jev-latest moves without notice."
- Jev run dates (PDF Table C.12): 2026-09-24 to 2026-09-30.

### The yes/no swap: what changes

- The authors swap the option NAMES. They do not negate the question. They do not change the rubric text, which carries the meaning of each option. The question, the rubric text and the option order stay the same.
- Quote: "The option-naming experiment recasts every binary question under four identifier pairs with the rubric text and order fixed. The pairs are the digits zero and one, an aligned no and yes, the same pair swapped against the rubric, and a pair of random five-letter strings."
- Quote: "Attaching the words yes and no to the opposite rubrics moves the hosted model and both Laya checkpoints strongly, Kev-0.8B moderately, and the four larger decoder-based checkpoints little (Figure 3)."
- Quote (Appendix B): "For the binary D3 tasks, the leading word that names the rubric’s own key, such as a description that opens with the word True, is stripped in the four naming conditions. The option name is then the only cue to polarity. D1’s noul questions are recast under the same four naming conditions with their rubric text unchanged, because their rubrics do not name a key"
- The four conditions have the keys "k01, kny, kswap and krand".

### What a "flip" means

- A flip is a change of the chosen rubric (the chosen meaning) relative to the aligned condition. The unit is flips per hundred decisions.
- Quote: "Each condition is scored by the share of decisions whose chosen rubric flips relative to the aligned condition and by the area under the curve of the positive-rubric probability against the reference label, with its paired-bootstrap change. All three quantities are read against the test-retest floor."

### The 50.5 number

- 50.5 is a rate: 50.5 flips per hundred decisions. It is not a count of items (for example, it is not 50.5 of 100 items).
- Quote: "On the binary questions of typed-decisions the swapped names flip 50.5 answers per hundred for the hosted model, with a state-clustered interval from 46.2 to 54.7 and a test-retest floor of 1.3."
- Quote: "Its flip rate reaches 97.2 on conversation derailment and 84.0 on power, against measured floors of 0.4 and 0.8."
- AUC under the swap: "Its area under the curve falls from 0.89 to 0.31".
- Neutral names: "For the hosted model, digits and random strings stay within 2.8 flips per hundred of the aligned condition on all three binary sets".

### Datasets and item counts for the swap

- Three binary sets:
  - D1 typed-decisions, binary (Noul) questions only. D1 has "400 states and 2,000 decisions" and 5 questions for each state. Number of D1 binary questions: Not found.
  - D3 conversation derailment ("conversations gone awry"): 500 scored decisions (Table 4).
  - D3 Wikipedia power: 500 scored decisions (Table 4).
- D1 labels come from a teacher model: "its gold label is the mean of three samples from a teacher endpoint of roughly four-billion-parameter class." The flip rate does not use labels. The AUC uses labels.
- Floor: "It also answers 5 times on a subset of 40 neutral workflow states, which gives a test-retest floor against which any option-naming effect is read."

### Swap results for the other models (typed-decisions, "Swapped" column of PDF Table C.10, flips per hundred)

- Laya (English) 91.3; Laya (multilingual) 86.8; Kev-0.8B 14.3; Kev-9B 6.5; decider-2b 4.8; this-that-model-1.0 4.3; Nimble-9B 4.2.
- Generative comparators: Qwen3.6-27B verbal 47.8, likelihood 48.7; Gemma-4-31B verbal 17.2, likelihood 16.8; Mistral-Small-24B verbal 40.3, likelihood 40.7.
- Abstract: "swapping yes and no flips 50.5 answers per hundred for Jev and at least 16.8 for every generative model tested, against at most 6.5 for four fine-tuned decision checkpoints."
- Earlier study that the paper cites: "sun2026typesafe report that exchanging the rubrics behind no and yes changes 32.5 percent of the hosted model’s answers and 76.92 percent for an open marker head on the encoder family of Laya."

### DeBERTa result

- Training: DeBERTa-v3-large "is fine-tuned on the training split of every benchmark with labels, as a cross-encoder with soft targets on typed-decisions and as a sequence classifier elsewhere".
- Quote: "A DeBERTa-v3-large encoder fine-tuned on each training split has the highest observed accuracy on every task of the three main dataset families except power. It reaches 0.793 on typed-decisions and 0.953 on CLINC-150, against 0.732 and 0.913 for the hosted model, and separates out-of-scope requests with an area of 0.989."
- Quote: "Its leads over the hosted model are significant after Holm correction on typed-decisions, CLINC-150, emotion and politeness, and on derailment and power the two do not differ significantly."
- Further benchmarks: "the fine-tuned DeBERTa-v3-large at 0.756 on GoEmotions and 0.950 on Banking77". Jev gets 0.604 on GoEmotions and 0.821 on Banking77.

### Calibration (for comparison with papers 1 and 3)

- Quote: "Calibration is reported as top-label expected calibration error with 10 equal-mass bins".
- Jev: "The hosted model is close to calibrated on typed-decisions and CLINC-150, with errors of 0.033 and 0.032, and far from it on the social-science tasks, where the pooled error is 0.251."
- Output grid: "The hosted model serves two-decimal probabilities at an undisclosed temperature".
- Risk control: "A guaranteed 5 percent risk leaves Jev 0.528 of the intent decisions, and an in-scope threshold still accepts 0.310 of out-of-scope requests."

---

## 3. Sharma, Duncan, Marceau Caron, Vollmer: "TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models" (arXiv 2610.11392)

- Read: full text (HTML v2).
- URLs: https://arxiv.org/abs/2610.11392 and https://arxiv.org/html/2610.11392
- v1: 8 Oct 2026. v2: 9 Oct 2026. Primary category: cs.AI (cross-list stat.AP).
- WARNING: the abstract page and the HTML full text have different abstracts.
  - Abstract page: "The decoder is least accurate on policy questions and degrades with more options."
  - HTML full text: "The encoder is least accurate on policy questions and degrades with more options."
  - The results support "encoder": "both encoder checkpoints barely beat always answering the most common level (0.34–0.38 against 0.34)."

### Models and version

- Quote: "Jev answered every request as jev-1.13.0; a 200-item canary (1,100 answers) recorded at session start showed no drift."
- Open models: Laya 0.3.4 (English and typed-decisions checkpoints) and Kev 0.8B, 4B, 9B.
- LLM baselines: Gemma 4 31B and Qwen3.5-9B, "restricted to the option letters, with probabilities from their first-token log-probabilities".
- Run date: Not found.
- Cost: "The hosted model’s entire evaluation cost about $5."

### Tasks

- Seven programmatic generators with rule-derived labels: support tickets, phishing email, RAG relevance, log triage, policy compliance, guardrail intent, multilingual tickets.
- Nine suites, A to I. Suite A (calibration) uses "n=500 items per generator" on the ticket and phishing sets.

### How they measure calibration

- Quote: "Top-label ECE with 15 equal-width bins, the binning-free smooth calibration error, Brier score with its Murphy decomposition (Brier, 1950; Murphy, 1973), and NLL with probabilities floored at 10^-4."
- Null ECE: "Every ECE is compared with its null ECE, the expected ECE of a perfectly calibrated model with the same confiedence histogram, from 200 label resamplings". The typo "confiedence" is in the original.
- The ratio ECE/null is "1 on average under perfect calibration, up to about 1.5 by chance".
- Temperature refit: "A temperature T is fitted by NLL on one half of the items and evaluated on the other; T>1 is over-confidence, and the fit is declared degenerate when the fit half is fully correct."
- Answer types: all three primitives. Table 8 lists tickets.priority (score, 5 levels), phish.urgency (score, 4 levels), phish.is_phishing (noul), phish.attack_class (choice, 5 options). The calibration rows use canonical wording.
- Noise floors: "ECE noise floors at n=500 range from 0.002 (near-certain answers) to 0.049 (diffuse score questions), which is why raw ECE is never compared across questions."
- Class-wise ECE: Not found.

### The "underconfident" claim and numbers

- Quote: "the hosted model is under-confident on every Suite A question (all its temperature refits are below 1)".
- Jev numbers (Table 8): tickets.priority ECE/null 2.2, T 0.66; phish.urgency ECE/null 1.9, T "deg."; phish.is_phishing ECE/null 4.3, T 0.21; phish.attack_class ECE/null 1.8, T "deg.". ("deg." means that all items in the fit half were correct.)
- Error ranking: "the under-confident hosted model has the lowest or tied-lowest AURC on every question".
- Unknowable arm: "With the tier line removed, so that two priority levels fit equally well, it stays highly confident (mean mass 0.89 on its top level), whereas the decoders split their probability between the two (0.28–0.51)."
- Exception on the other generators: "except the hosted model is over-confident on guardrail intent and not the best error-ranker on policy severity." Table 10 gives ECE/null 2.8 for Jev on guard.intent, marked over-confident.
- Decision cost on ticket priority (under-prioritising costs 2 per level, over-prioritising 1): Jev argmax 1,080, Bayes 2,300 per 10^4 decisions, value −1,220 (Table 4).
- Phishing: "the hosted model’s under-confidence keeps nearly all legitimate mail above the 4.8% flagging line, so Bayes flags 498 of 500 emails."

### Framing-sensitivity protocol

- Quote: "Prompts include canonical wording, five human-written paraphrases, three criteria variants (label-only, with negative definitions, or deliberately vague), a corruption control with swapped option descriptions, and adversarial wordings written to mislead, all fixed before evaluation to reduce accuracy. We define non-adversarial variants (canonical, paraphrased, and explicit criteria) as friendly framings."
- Quote: "To measure prediction stability, we compute the mean pairwise Jensen–Shannon divergence of the returned output distributions and track the argmax flip rate across framings."
- Option order: "option orders are permuted three times per item".
- Negations (Suite G): each yes/no question has an explicit negation and a paraphrased negation. Phishing example: "Is this email legitimate, i.e. NOT a phishing or social-engineering attack?" and "Is it safe to treat this message as genuine?"

### Framing-sensitivity numbers

- Ticket priority: "Wording alone spans 24 points for the hosted model, 12–18 for the decoder family and 15 and 7 for English and typed-decision encoders."
- Jev on tickets.priority: median accuracy 0.874, range [0.708, 0.946] (Table 8).
- Jev argmax flip rate across friendly framings on tickets.priority: 0.209; mean JSD 0.105 (Table 9).
- Phishing negations (Jev only): the explicit negation "deviates by 0.10 at accuracy 0.99, the paraphrase “Is it safe to treat this message as genuine?” by 0.23 at accuracy 0.76, against 0.99 on the original question."
- Anger negation: Jev deviation 0.015 (explicit) and 0.022 (paraphrase); accuracy 1.00 and 1.00 (Table 12).
- Position bias: "Jev chooses index 0 on 17.9% of 6-option rows".
- Result of the "corruption control with swapped option descriptions": Not found.
- Earlier reports that the introduction cites: "a 33-point accuracy swing from rewording a question (XenoSpectrum, 2026), an expected calibration error roughly four times its finite-sample noise floor (scienthoon, 2026)".

---

## 4. Xu, Song, Zhang, Chen, Liu, Hu, Zhao: "JevOut: Natural Context Can Flip Decision Models" (arXiv 2609.30243)

- Read: full text (HTML v2).
- URLs: https://arxiv.org/abs/2609.30243 and https://arxiv.org/html/2609.30243
- v1: 24 Sep 2026. v2: 8 Oct 2026. The arXiv comment says that v2 "updated author list". Primary category: cs.CL.

### Systems

- Quote: "The targets comprise hosted Jev 1.13.0 (TypeSafe AI, 2025), the open Jev-style implementation OpenSourceJev (OpenSourceJev Contributors, 2026), Von 1.0, and a plain Qwen scorer."
- OpenSourceJev and the plain Qwen scorer use the same frozen Qwen3-1.7B Q8 weights. "Von is a 395M non-autoregressive decision model".
- "Gemma4-12B proposes additions, and a separate invocation checks their constraints without seeing the target option".
- Run date: Not found.

### Datasets

- MMLU-Pro, SuperGPQA, MuSR, ToMBench, LAR-ECHR, SATA-Bench, BFCL V4.
- "Each dataset contributes 50 development and 100 held-out source items." The test set has 700 items.

### What "natural context" is

- Quote: "An answer-preserving context addition supplies background or procedural detail that fits the input without changing its correct answer."
- Quote: "Rather than instructing the model to select an answer, the addition supplies background or procedural detail relevant to the original input. The question and available choices stay unchanged".
- The proposer model sees the gold answer and the fixed wrong target. The checker does not see the target.
- Quote: "constructed contexts contain one to four additions."

### Examples of inserted text (MMLU-Pro item 11233, Appendix I.3)

- Original: Jev selects C with probability 0.97; the fixed target B has 0.01.
- Addition 1: "Gardiner highlights how the delayed ecological consequences of current actions complicate the direct link between present choices and future impacts." Result: Jev keeps C (0.90); B rises to 0.09. No flip.
- Addition 2: "Gardiner argues that the diffusion of responsibility among billions of actors complicates the moral weight of individual actions." Result: C 1.00, B 0.00. No flip.
- The sentence that flips this item (B at 0.54) is in Figure 1, which is an image. The caption says only: "The added sentence discusses a related intergenerational issue". Exact text of the flipping sentence: Not found.

### Where they insert the text

- Rule: "Original characters are copied verbatim between insertions. The question, option identities, option order, and rubric bindings are fixed."
- Question-only items (MMLU-Pro, SuperGPQA): "For question-only items, the permitted boundary is before the original question."
- Passage tasks (MuSR, ToMBench, LAR-ECHR): "These passage tasks place legal boundaries after sentence endings."
- Tool routing (BFCL V4): "Insertion boundaries occur only inside the final user message, so earlier turns and completed tool calls remain fixed."

### How the 61.4%–73.2% is computed

- It is NOT the share of correct decisions that change in any way. It is the share of initially correct decisions that the search moves to ONE wrong option, fixed in advance.
- The fixed target is the most probable wrong option on the original input ("fix the most probable wrong option as the target").
- Quote: "Targeted Flip Rate (TFR) measures the proportion of initially correct decisions for which we uncover at least one such flip within a fixed budget of accepted target evaluations."
- Quote: "A change to another wrong option does not count; rejections and unsuccessful runs retain their place in the original denominator."
- Budget: at most 64 accepted target evaluations for each decision (16 proposal slots, up to four iterations).
- Table 1 TFR: Jev 61.4% (312 of 508); OpenSourceJev 72.6% (238 of 328); Von 73.2% (240 of 328); plain Qwen 64.9% (185 of 285).
- Also with target probability at least 0.7: 45.1%, 64.3%, 54.0%, 60.4%.
- One-shot controls (one sentence): neutral 2.2 / 6.4 / 7.6 / 8.4%; target-aware 16.9 / 16.2 / 21.6 / 18.6%.
- Jev by dataset (Table 8): from 40.0% on LAR-ECHR (32 of 80) to 85.7% on BFCL V4 (60 of 70).
- Confident decisions (original gold probability at least 0.9): Jev TFR 45.5% of 290 (Table 2).
- Short contexts: "A single added sentence suffices for 41.5% of the original initially correct Jev population."
- Baseline without optimization: repeated independent generation reaches "48.0% TFR on the same 508 Jev decisions under the optimizer’s 64-call ceiling."
- Human check: "91.6% of 250 successful contexts sampled across systems meet all three validity criteria by majority judgment". Accepted-but-unsuccessful controls: 93.0%.
- Repeatability on Jev: "97/100 preselected contexts reproduce the targeted flip in at least eight of ten evaluations".

---

## 5. Geuter, Cheng, Alvarez-Melis: "Releasing the Inner Jev: A Simple Recipe for Open Decision Models" (arXiv 2610.17205)

- Read: full text (HTML v1).
- URLs: https://arxiv.org/abs/2610.17205 and https://arxiv.org/html/2610.17205
- v1: 12 Oct 2026. Primary category: cs.AI.

### Readout method

- Letter tokens, one forward pass. The readout does not score the full label text.
- Quote: "Rather than generating a textual answer, we extract the model’s logits for the answer letters after a single forward pass."
- The readout uses the LM-head rows "corresponding to the letters A through P" at the last prompt position, then a softmax over the K options.
- Quote: "Yes/no questions are represented by two options, while scoring questions use ordered answer levels. The readout supports up to 16 options."
- Calibration: one temperature for each question type, fitted on 3,884 held-out questions. Gemma: "The fitted temperatures are 6.0 for choice questions and 7.2 for yes/no questions". Temperature scaling raises the Gemma 4 31B Capability "from 72.9 to 79.6".
- Mass on the letters: "Gemma 4 31B assigns 99.99% of its next-token probability to the available answer letters".
- Number readout of concurrent work (Li & Wagle): "80.5 for numbered options and 79.6 for answer letters".

### Models

- "Gemma 4 31B, Gemma 4 12B, Qwen3.6-27B, and Qwen3.5-4B". Comparison: Jev 1.13.0 through its API, and 13 public systems.

### Metric

- OpenDecisions: 2,352 questions, 11 task families, 23 languages. The authors built it.
- Capability = (I + C)/2. I is chance-corrected accuracy. C uses ECE ("100(1−ECE/0.5)") and, for score items, also the ranked probability score.
- The numbers 79.6 and 80.1 are Capability scores. They are not accuracies.

### Numbers

- Quote: "Without fine-tuning, Gemma 4 31B scores 79.6 on OpenDecisions, compared to Jev’s 80.1." Qwen3.6-27B scores 79.0.
- Table 3: zero-shot Gemma 4 31B I 71.9, C 87.4; Jev I 71.9, C 88.3.
- Seven public decision datasets (9,212 questions, Table 4): macro accuracy Jev 88.0, Gemma 4 31B zero-shot 88.4; mean ECE Jev 0.066, Gemma zero-shot 0.077.

### LoRA result

- Recipe: "rank-32 LoRA adapters (α=64, no dropout) to all attention and MLP projections". A decision head starts from the letter rows of the LM head. 400 steps, "about five GPU-hours per run". Training pool: 417,825 questions.
- Quote: "Our LoRA training raises Gemma 4 31B from 79.6 to 84.4 Capability on OpenDecisions, a gain of 4.8 ± 2.0." The trained model leads "Jev by 4.3 ± 2.5".
- JevBench v1.7 leaderboard: Jeff-1.0-Large "scores 81.5 Capability"; "Jev scores 77.1". Each system gets its own draw of items.
- Source of the gain: "Training only a decision head on frozen model representations gives 78.9 Capability". With LoRA and a frozen head: "training LoRA while keeping the head fixed at its original answer-letter weights gives 84.9".
- Smaller models (Section 5.2): Qwen3.6-27B gains 2.2 ± 2.2. Gemma 4 12B changes by −0.4 ± 1.8. For Qwen3.5-4B the recipe gives 63.0 against 63.6 zero-shot; with three extra changes for this model, "it gains 3.0 ± 2.6".
- Public sets after training: trained 31B macro accuracy 89.0; mean ECE "0.047, against 0.066 for Jev".
- Knowledge-heavy multiple choice: "Jev is ahead: 92.9 macro accuracy against 88.8 for our trained 31B model".

---

## 6. Huang, Teng, Deng, An: "Consider, Then Choose: Order-Invariant Decisions in System One Models" (arXiv 2610.18233)

- Read: full text (HTML v1).
- URLs: https://arxiv.org/abs/2610.18233 and https://arxiv.org/html/2610.18233
- v1: 13 Oct 2026. Primary category: cs.AI.

### Protocol

- Quote: "We report accuracy with the options in their original order (Acc) and the flip rate under one shuffle per question (Flip)."
- Flip: "share of questions (%) whose answer changes when the options are shuffled".
- Datasets: "Banking20 and Newsgroups (20 options, 300 questions each), MMLU-Pro (Wang et al., 2024b) (10 options; 1,400 questions), Typed Decisions (LocalLLaMA, 2026) (600 questions with 4–5 options), and the public items of JevBench (Standhartinger, 2026) (139 questions with 3–6 options)."
- Ten models: Qwen3-8B and Gemma 4 12B (open LLMs with a letter readout); Jev 1.13, Mercury Decide, GPT-6 Luna (closed); Torchcast 12B, Kev 9B, Intern-Decision 4B, Tev1 4B, Laya (open, fine-tuned).
- Baselines: Listwise (one listing in the given order) and cyclic averaging as AnyJev implements it.
- Runoff, step 1 ("Consider"): score each option alone with a yes/no question; keep the top options that hold at least α of the probability (α = 0.99, at least k_min = 3 options).
- Runoff, step 2 ("Choose"): compare the kept options in every cyclic shift of their ranking; average the log-probabilities.

### Main numbers

- Open models (abstract): Runoff "changes no answer under shuffling, against 9.3% for cyclic averaging". Introduction: Listwise changes "18.6%".
- Jev 1.13 (mean over five datasets, Table 2): native flip 5.3%, AnyJev 2.2%, Runoff 0.9%. Accuracy: native 80.6, AnyJev 81.5, Runoff 81.5.
- Jev 1.13 native flips by dataset: Banking20 3.0, Newsgroups 5.0, MMLU-Pro 7.7, Typed Decisions 6.5, JevBench 4.3.
- Remaining Jev flips: "the two APIs can return different probabilities for identical yes/no requests, which can change the consideration set."
- Other closed models: GPT-6 Luna native 13.2% to Runoff 0.0%; Mercury Decide 9.8% to 1.4%.
- Fine-tuned models (Table 3, native flip): Torchcast 12B 15.6, Kev 9B 12.5, Intern-Decision 4B 14.7, Tev1 4B 12.3, Laya 30.6. AnyJev on Laya: 15.4. Runoff: 0.0 on all five.
- Option count (Qwen3-8B, Listwise, Banking20): flip rate 4.7% at K = 2 and 17.0% at K = 20 (Section 3).
- Speed: "Averaged over the five datasets, Runoff is 2.1 times faster than AnyJev on both models". On Banking20 with Qwen3-8B: 77 ms against 308 ms.
- Agent test (OSWorld, 316 tasks): "27 against 21 with Qwen3.5-35B-A3B, and 14 against 9 with Qwen3.5-9B."

---

## 7. Li, He, Li: "Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?" (arXiv 2609.33209)

- Read: full text (HTML v1).
- URLs: https://arxiv.org/abs/2609.33209 and https://arxiv.org/html/2609.33209
- v1: 27 Sep 2026. Primary category: cs.CL.

### Protocol

- 160 items: 100 ChaosNLI items (entailment, neutral, contradiction) and 60 PubMedQA items (labels renamed to affirmative, negative, inconclusive). Each item has three mutually exclusive labels.
- For each label k, the authors ask three yes/no questions. The paper shows them in a math display; here they are in plain text:
  - S_k: Is it true that the label is X_k?
  - N_k: Is it true that the label is not X_k?
  - D_k: Is it true that the label is either X_i or X_j?
- For each item, they also ask one choice question (CH): Which label applies?
- This gives 10 questions for each item and 480 negation pairs (160 items × 3 labels).
- Quote: "S_k and N_k differ only by the word “not”." (math shown in plain text)
- The checks need no gold labels. Identities: negation complement, partition additivity, disjunctive complement, disjunction additivity, description invariance, monotonicity, cross-format consistency.
- Jev: "We query Jev (typesafe/jev-1.13) through OpenRouter’s decisions endpoint. All 3,360 responses report the same version, jev-1.13-20260917."
- The authors send each question alone and repeat it once. A bundling check sends all ten questions of an item in one request.
- Comparison LLM: Qwen3.8-27B (official BF16 weights, thinking disabled) with two readouts: first-token probability and verbalized JSON probability.
- Cost: "The Jev calls cost US$0.076 in total."

### Main numbers

- Abstract: "On 480 negation pairs, Jev’s probabilities for “the label is X” and “the label is not X” miss summing to one by 0.064 on average (95% CI 0.055 to 0.072)."
- Qwen3.8-27B: 0.293 (first-token readout) and 0.122 (verbalized readout).
- Repeat noise: Jev noise floor NF = 0.013 (Section 5.1).
- Confidence-matched pairs (253 pairs): Jev 0.028, Qwen 0.257 (Table 2).
- Partition: "Asked separately whether each of the three labels applies, Jev gives probabilities that sum to 1.14 on average and to more than one on 159 of 160 items."
- Cross-format: yes/no versus choice for the same label differ by 0.091 (Jev) and 0.131 (Qwen first-token) (Table 3).
- Same side: Qwen first-token "answers “no” to both a statement and its negation in 196 of 480 pairs". Jev "answers “yes” to both questions of a pair in 31 of 480 pairs".
- Bundling: "its negation violation is 0.064 in bundled and single requests".
- Grid: "all 4,320 of its yes/no answers, across both repeats and the bundled requests, lie in [0.01,0.99]".

---

## 8. Messoudi, Mansar: "Calibration and Risk-Budgeted Decisions with System One Models" (arXiv 2610.19811)

- Read: full text (HTML v1).
- URLs: https://arxiv.org/abs/2610.19811 and https://arxiv.org/html/2610.19811
- v1: 14 Oct 2026. Primary category: stat.ML.
- Jev endpoint: "The evaluated proprietary interfaces are typesafe/jev-1.13, respan/span-01, upstage/solar-decide, and openai/gpt-6-luna-decisions."

### Method and result in two sentences

1. The authors use Learn Then Test (LTT, δ = 0.10) to pick a score threshold; uncertain inputs get a safe fallback action (block or retain).
2. On six datasets, naive and empirical thresholds exceed the risk budget in 42% and 51.7% of settings, but LTT exceeds it in only 7.8%.

Details:

- The risk is the false omission rate (FOR) among the inputs that get the useful action (allow an input or discard a document). The budget is α.
- The naive threshold is τ = α. It assumes exact calibration.
- The six datasets cover moderation (WildGuardMix, Aegis), safety (S-Labs, ATBench) and reranking (SciDocs, StackOverflow).
- Jev's ECE changes from 0.034 on WildGuardMix to 0.29 on SciDocs.

### Supporting quotes

- "Table 3 shows that Empirical selection exceeds the target budget in 51.7% of settings (mean excess 1.17%). The Naive rule similarly exceeds the budget 42% of the time"
- "LTT produces substantially fewer and smaller risk exceedances (just 7.8% frequency, mean excess 0.1%)"
- "Jev’s ECE degrades from 0.034 on WildGuardMix to 0.29 on SciDocs."
- Calibration metric: "10-bin expected calibration error (ECE) and Brier score".

---

## 9. Li, Miao, Krishnan, Padman: "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure" (arXiv 2609.26550)

- Read: full text (HTML v5).
- URLs: https://arxiv.org/abs/2609.26550 and https://arxiv.org/html/2609.26550
- Versions: v1 22 Sep, v2 27 Sep, v3 29 Sep, v4 6 Oct, v5 9 Oct 2026. Primary category: cs.AI.
- Version and price: "At collection time JEV 1.13.0 charged $0.042 per million input tokens and nothing for output".

### Escalation protocol

- Confidence: "we take a judge’s confidence to be the probability of its verdict" (q = the largest label probability).
- For a pair, JEV judges both orders, and the gate averages the two aligned probabilities. If the larger averaged probability is at least τ, the gate accepts JEV's verdict. If not, GPT-6 (low reasoning effort) decides.
- Threshold choice: the "point rule" on 96 pilot selection pairs (64 RewardBench, 32 JudgeBench). The rule takes the most permissive τ whose accuracy loss against the fallback stays within two points. The authors fixed the rule and the grid before the main round. The frozen threshold for JEV → GPT-6 is τ = 0.9.
- Held-out test: 1,610 preference pairs (1,340 RewardBench, 270 JudgeBench).
- Later "lower-bound rule": the one-sided 95% lower confidence bound of the accuracy loss must stay within two points.

### Main numbers

- Held-out pairs: the policy "accepts 68.5% of the 1,610 held-out pairs and escalates the other 31.5%, which contain 85% of JEV’s errors."
- Held-out accuracy and fee: "It scores 93.4% against GPT-6’s 92.5%" (+0.93 points, 95% interval 0.24 to 1.66) "at 41.4% of its fee, or 44.0% under conservative reservations."
- RewardBench (1,340 pairs): escalates 25% and beats GPT-6 by 1.27 points at 26.7% of its fee.
- JudgeBench (270 pairs): escalates 65%; accuracy rises from 81.3% (JEV) to 92.2% against 93.0% for GPT-6; "the cascade closes 94% of the JudgeBench gap but saves only a third of the fee there."
- Live rerun (510 pairs): escalated 46.5%; 92.5% against 93.1% for GPT-6; 57.2% of the fee; "Its median latency per pair was 0.27 seconds against 2.10 for GPT-6 alone".
- Prospective live test (570 held-out pairs from PPE and the JudgeBench Claude split; lower-bound rule on 100 labels for each workload; τ = 0.95 for PPE and 0.99 for JudgeBench): "the live cascade matched GPT-6’s accuracy exactly on both workloads (88.2% on PPE and 94.7% on JudgeBench; Table 5). It escalated 68% and 89% of pairs, caught 93% of JEV’s errors, and cost 75.5% of GPT-6’s fee".
- JEV alone: "JEV ranks eighth of fifteen LLM judges, 1.7 points behind GPT-6, yet it is 277 times cheaper ($0.044 per 1,000 judgments) and answers in 0.15 seconds at the median".
- Confidence: "Where JEV’s confidence (averaged over both orders for pairs) is at least 0.9, the two judges are nearly tied (94.7% versus 94.4% on 3,744 items)".

---

## 10. Lai, Ou, Wang, Wang, Yang, Chen, Chen: "BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation" (arXiv 2602.09383), the source of JudgeBench-Pro

- Read: full text (HTML v1).
- URLs: https://arxiv.org/abs/2602.09383 and https://arxiv.org/html/2602.09383
- v1: 10 Feb 2026. arXiv comment: "Accepted to ICLR 2026". Primary category: cs.CL.

### How JudgeBench-Pro was built

- The bias goes into the WRONG (rejected) response. In BiasScope, a teacher model rewrites each rejected response to add one bias and keeps its outcome. Method text: "We apply a sampled bias ... to each rejected response" and the teacher must "generate its biased variant ... while preserving the original outcome as much as possible" (math symbols removed).
- The injection prompt asks that the "terminal conclusion/result stays identical word-for-word".
- Construction quote: "Based on the 620 original samples from JudgeBench, we generated 10 biased variants for each sample via the bias injection module of BiasScope, resulting in 6,200 synthetic instances. We employed a powerful model Qwen3-32B for adversarial filtering. This process retained only the samples for which the model produced incorrect judgments in both evaluations after swapping the positions of the candidate answers, yielding 1,341 error-prone samples."
- So yes: the authors kept only items that ONE judge model (Qwen3-32B) got wrong in both answer orders. The five models that they evaluate later did not take part in this filter. But models with similar names (Kimi-K2 and DeepSeek-V3) helped to check the labels of ambiguous cases (see the next item).
- Manual check: "Next, we manually verified that misjudgments stemmed from bias." Four annotators; "The final IAA reached 0.92". For ambiguous cases, they also used "DeepSeek-R1, Kimi-K2, DeepSeek-V3".
- Final set: "Finally, 163 samples with consistent outcomes between the two answers were removed, resulting in a refined set of 1,178 high-quality samples that constitute JudgeBench-Pro."
- Length: "The new rejected responses are only 8.4% longer than the original ones".
- Bias source: "the ten biases used to construct JudgeBench-Pro were initially discovered in Qwen2.5-1.5B-Instruct".
- Teacher model that wrote the JudgeBench-Pro variants: Not found in the JudgeBench-Pro section. The default teacher of the experiments is "Qwen 2.5-72B-Instruct".

### Accuracy of strong LLM judges on JudgeBench-Pro

- The paper reports error rate (Err %), not accuracy. Table 9, overall Err on JudgeBench-Pro (JudgeBench in brackets):
  - GPT-4o 74.7 (36.7)
  - Deepseek-v3 67.5 (37.4)
  - Deepseek-v3.1 47.5 (23.4)
  - Doubao-seed-1-6-250615 20.4 (11.9)
  - Kimi-k2-Instruct 56.0 (27.4)
- Accuracy = 100 − Err (my calculation): GPT-4o 25.3%; Deepseek-v3 32.5%; Deepseek-v3.1 52.5%; Doubao-seed-1-6-250615 79.6%; Kimi-k2-Instruct 44.0%.
- Mean Err of the five models (my calculation): 53.2% on JudgeBench-Pro and 27.4% on JudgeBench. The difference is 25.9 points. This agrees with the introduction: "an average error rate 25.9% higher than on JudgeBench".
- Section 5: "most models perform close to or even worse than random guessing (50%) on JudgeBench-Pro".
- Section 5: "GPT-4o exhibits the highest error rate of 74.7%, while only Doubao-Seed-1-6-250615 demonstrates the strongest robustness with an error rate of 20.4%."
- Abstract: "even powerful LLMs as evaluators show error rates above 50% on JudgeBench-Pro". Note: in Table 9, only three of the five models have an error rate above 50%.

---

## BibTeX (data from the arXiv API, 2026-10-16)

Key rule: first-author last name (lowercase ASCII) + 2026 + first word of the title (lowercase, without punctuation). I removed the hyphens in "JEV-as-a-Judge" and the colons in "TypedBench:", "JevOut:" and "BiasScope:".

```bibtex
@misc{deusser2026evaluating,
  title         = {{Evaluating and Benchmarking the System One Model Jev}},
  author        = {Deu{\ss}er, Tobias and Sparrenberg, Lorenz and Sifa, Rafet},
  year          = {2026},
  eprint        = {2609.37647},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.37647}
}

@misc{rafe2026benchmarking,
  title         = {{Benchmarking System One decision models against trained classifiers and language models for automated decision gates}},
  author        = {Rafe, Amir and Das, Subasish},
  year          = {2026},
  eprint        = {2610.00346},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.00346}
}

@misc{sharma2026typedbench,
  title         = {{TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models}},
  author        = {Sharma, Rahul and Duncan, Andrew B. and Marceau Caron, Ga{\'e}tan and Vollmer, Sebastian J.},
  year          = {2026},
  eprint        = {2610.11392},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.11392}
}

@misc{xu2026jevout,
  title         = {{JevOut: Natural Context Can Flip Decision Models}},
  author        = {Xu, Zixiang and Song, Zirui and Zhang, Chiyu and Chen, Xiuying and Liu, Xi and Hu, Xiyang and Zhao, Yue},
  year          = {2026},
  eprint        = {2609.30243},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.30243}
}

@misc{geuter2026releasing,
  title         = {{Releasing the Inner Jev: A Simple Recipe for Open Decision Models}},
  author        = {Geuter, Jonathan and Cheng, Andrew and Alvarez-Melis, David},
  year          = {2026},
  eprint        = {2610.17205},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.17205}
}

@misc{huang2026consider,
  title         = {{Consider, Then Choose: Order-Invariant Decisions in System One Models}},
  author        = {Huang, Jianguo and Teng, Yao Long and Deng, Yanchen and An, Bo},
  year          = {2026},
  eprint        = {2610.18233},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.18233}
}

@misc{li2026beyond,
  title         = {{Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?}},
  author        = {Li, Keyi and He, Yihao and Li, Quanyi},
  year          = {2026},
  eprint        = {2609.33209},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.33209}
}

@misc{messoudi2026calibration,
  title         = {{Calibration and Risk-Budgeted Decisions with System One Models}},
  author        = {Messoudi, Soundouss and Mansar, Youness},
  year          = {2026},
  eprint        = {2610.19811},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url           = {https://arxiv.org/abs/2610.19811}
}

@misc{li2026jevasajudge,
  title         = {{JEV-as-a-Judge: Accept When Confident, Escalate When Unsure}},
  author        = {Li, Yubo and Miao, Yidi and Krishnan, Ramayya and Padman, Rema},
  year          = {2026},
  eprint        = {2609.26550},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.26550}
}

@misc{lai2026biasscope,
  title         = {{BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation}},
  author        = {Lai, Peng and Ou, Zhihao and Wang, Yong and Wang, Longyue and Yang, Jian and Chen, Yun and Chen, Guanhua},
  year          = {2026},
  eprint        = {2602.09383},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.09383}
}
```
