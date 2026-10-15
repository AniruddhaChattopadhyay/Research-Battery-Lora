# Survey plan: System One models (v0, 2026-10-15)

> **Status (2026-10-15): this is now the plan for paper 2.** Aniruddha decided to write the experiments first, as a standalone research paper (`2026-10-15-paper1-plan.md`). The survey starts in January 2027. It reuses this plan, but its experiments section becomes a short summary of paper 1, plus E1 and E4. Change the timeline (section 10) and the venues (section 11) in January.

**Companion notes**

- `2026-10-15-model-landscape.md`: the facts about the systems, benchmarks and papers, with sources.
- `2026-10-15-experiment-design.md`: the experiments.
- `2026-10-15-venues.md`: where to submit, and the deadlines.
- `2026-10-15-survey-paper-guide.md`: what a survey paper is, and the steps.
- `../literature/protocol.md`: the review protocol.
- `../literature/seed-papers.md`: the first list of papers for the research lines.

---

## 0. Summary

1. **Topic.** System One models are AI models that return typed answers with probabilities, not text. Examples: TypeSafe Jev, the OpenAI Decisions API, Microsoft-Decision-1 and Perplexity pplx-decider.
2. **Why now.** The category started on 2026-09-15. In one month, about 25 systems and 5 leaderboards appeared, and a search for "Jev" on arXiv gives 131 papers. Their results do not agree. No complete survey exists. The closest work covers only the first nine days.
3. **Our angle.** We will do five things:
   - give a clear definition and a taxonomy;
   - show the older research lines that lead to these models;
   - collect all published results in one evidence table, and explain where they disagree;
   - run targeted experiments that test the disagreements under the same conditions;
   - give specific open problems.
4. **When.** Do the work until 2026-12-20. Submit in January 2027 to ARR (for ACL 2027), the IJCAI 2027 survey track or JAIR.
5. **Cost.** About $310 to $340 for API calls and GPU time. Plan for $500 (estimate, section 13).
6. **Decisions for you now:** the team, a senior co-author and the budget (section 16).

---

## 1. What System One models are

- A System One model gets an **input** (for example, a support message) and a set of **questions** that the caller writes at call time.
- Each question has an answer type:
  - a **predicate** (yes or no; TypeSafe calls it `noul`);
  - a **choice** (one option from a fixed set);
  - a **score** (a level on an ordered scale).
- The model returns a typed answer and a probability for each question, in one call. It writes no text.
- The vendors say that it takes 70 to 500 ms and costs about $0.02 to $0.10 for each million input tokens. Output is free.
- The name comes from Kahneman's System 1 (fast judgment). Chat LLMs and reasoning models take the System 2 role. In our documents, we call them **System Two models**.

The full facts are in `2026-10-15-model-landscape.md`.

---

## 2. My assessment

**Is this a good survey topic? Yes, if we move fast.**

Reasons for:

1. The area grows quickly, and the results conflict. A survey that puts them in order has high value.
2. No complete survey exists on 2026-10-15. Tang and Zheng (arXiv:2609.32160) review only 28 papers from the first nine days.
3. The experiments are cheap. One OpenRouter key gives access to about 18 System One models.
4. The topic fits your earlier work:
   - ThinkBudget asked when a model must think more. A System One model with a System Two model is the same question.
   - The tool-drift paper studied API changes. `jev-latest` changes version this month.

Risks:

1. **The area changes every week.** Set a cut-off date (2026-12-01), record all versions and dates, and plan a second version.
   - On 2026-10-15, arXiv has 131 papers that mention Jev. If the rate stays the same, expect 250 to 350 works at the cut-off. Plan the screening work for this number.
2. **Other teams will write a survey too.** Our evidence synthesis and conflict tests make our survey different. Speed is still important.
3. **arXiv CS does not take a survey before a venue accepts it.** Use a public GitHub repository for early visibility (venues note, section 2).
4. **Conference routes need a senior co-author** (ARR reviewer rule, IJCAI expert rule).
5. **Most facts come from vendors and blogs.** Label every vendor number. Check grey sources with the AACODS checklist.

---

## 3. The paper

**Working title:** "System One Models: A Survey of Typed Decision Models, Their Evidence and Open Problems".

**Type:** a multivocal survey plus targeted experiments.

**In scope:**

- Models and APIs that fit the definition in section 4: hosted and open, text and images.
- The research lines that lead to them, as background (section 4.2). We do not survey each line completely.
- The use of System One models with System Two models.

**Out of scope:**

- General surveys of LLM routing, text classification or reward models.
- Robotics "System 1" policies (for example, Figure Helix). Mention them only in the background, because the name is the same.
- Training methods of LLMs in general.

**Survey questions:**

- **SQ1. Definition.** What is a System One model? How is it different from earlier model types?
- **SQ2. Landscape.** Which systems exist? How are they different (taxonomy)?
- **SQ3. Lineage.** Which research lines lead to them?
- **SQ4. Evidence.** What do the published evaluations show? Where do they disagree, and why?
- **SQ5. Experiments.** Do the disagreements stay when all systems run under the same conditions?
- **SQ6. Open problems.** What is not solved?

**Planned contributions:**

- **C1.** A definition, and a comparison with 7 earlier model types (section 4).
- **C2.** A taxonomy of all systems that exist at the cut-off date (section 5).
- **C3.** A map of the research lines that lead to System One models.
- **C4.** An evidence table of all published results, and an analysis of the disagreements.
- **C5.** Targeted experiments on calibration, robustness, the value of decision training, cascades with System Two models and version changes.
- **C6.** Specific open problems, and a public repository with the list of systems and our results.

---

## 4. Definition and boundary (draft)

### 4.1 Definition

A **System One model** M gets an input x and a set of questions Q = {q1, ..., qm}. The caller writes each question at call time, in natural language, with an answer space Aj:

- a **predicate**: Aj = {true, false};
- a **choice**: Aj = {a1, ..., ak}, a finite set of named options;
- a **score**: Aj = (l1 < ... < lk), a finite ordered set of levels.

For each question, M returns a probability distribution pj over Aj, in one call. Four properties together make a System One model:

1. **Promptable.** The caller defines the questions and the options at call time. A new task needs no training data.
2. **Typed.** Each answer is always a member of its answer space. The model writes no free text.
3. **Probabilistic.** The model returns a probability for each answer. The vendor intends the probabilities to be calibrated.
4. **Prefill-bound.** The model does not generate the answer token by token. The cost depends mainly on the input length, not on the output.

### 4.2 Comparison with earlier model types

| Model type | Promptable | Typed | Probabilities | No text generation | Example |
|---|---|---|---|---|---|
| Task-specific classifier | No | Yes | Yes | Yes | A fine-tuned DeBERTa-v3 |
| NLI zero-shot classifier | Yes (labels become hypotheses) | Yes | Yes | Yes | Yin et al. (2019) |
| LLM with label-likelihood scoring | Yes | Yes | Yes, often not calibrated | Yes | AnyJev, LLM-as-Jev |
| LLM with structured outputs | Yes | Yes, with constrained decoding | Only verbalized | No | JSON-schema mode |
| LLM-as-a-judge | Yes | No, the answer is parsed from text | No | No | MT-Bench judges |
| Reward model | No, one fixed question | Yes, one scalar | No, an unnormalized score | Yes | RewardBench models |
| Guard model | Partly, a fixed or custom policy | Mostly | Through token probabilities | Partly | Llama Guard |
| **System One model** | **Yes** | **Yes** | **Yes, intended to be calibrated** | **Yes** | Jev, pplx-decider |

**Our stance (to test in the paper):** each property is old. The new things are the combination of all four, many questions in one call, a common API and a price for input only. This stance answers the critique "it is only a classifier", and it gives the survey a clear point of view.

---

## 5. Taxonomy (draft v0)

| Dimension | Values (examples) |
|---|---|
| Interface | Jev-compatible (`state` and named questions); OpenAI-style (`input` and an array of questions) |
| Answer types | Predicate (`noul`), choice, score, box |
| Question handling | Isolated (Jev: "in parallel and in isolation"); shared context; not disclosed |
| Readout | A new head on a decoder (pplx-decider: a [255, 5120] readout and no LM head; Strands: a pointer head); LM-head token probabilities (AnyJev, Tev1); an encoder (GLiNER2.5-Decide); not disclosed (Jev, OpenAI) |
| Attention | Causal; bidirectional in some layers (pplx-decider v1.1) |
| Backbone | Qwen family (most open systems); GPT-6 Luna; not disclosed |
| Training data | Synthetic (Jev, reported); public task collections (tasksource for pplx-decider); distillation from another System One model (JEV-27B-VL from Jev 1.13); vendor synthetic data (Microsoft) |
| Calibration | Reinforcement learning (RLCD, Jev); temperature scaling (pplx-decider); not disclosed |
| Modalities | Text; text and images; text, images, audio and video (Clef-omni) |
| Openness | Closed API; open weights (Apache-2.0); training-free method |
| Intended use | Routing, moderation, labeling, grading, agent action selection |

We will fill one row for each system in the extraction spreadsheet. Then we will change the dimensions until each system fits.

**System One with System Two (design patterns):**

1. **Gate:** the System One model routes or filters before the System Two model.
2. **Verifier:** the System One model checks the output of the System Two model.
3. **Cascade:** the System One model answers. When its confidence is low, the System Two model answers.
4. **Labeler:** the System One model labels large data sets alone.
5. **Policy:** the System One model selects the next action of an agent.

---

## 6. Paper outline

Long version: about 30 pages (journal). Short version: 8 pages (ARR) or 7 pages plus 2 pages of references (IJCAI).

| Section | Content | Long | Short |
|---|---|---|---|
| 1. Introduction | Motivation, scope, contributions, a table that compares our survey with other surveys | 2.5 | 1 |
| 2. Background | System 1 and System 2; System Two models; decisions in production software | 2 | 0.5 |
| 3. Review method | Sources, search strings, inclusion rules, PRISMA counts | 1.5 | 0.3 |
| 4. Definition and lineage | Section 4 of this plan; the research lines from 2019 to 2026 | 3 | 1 |
| 5. Taxonomy and systems | The taxonomy, a figure and the comparison table | 5 | 1.5 |
| 6. Evidence synthesis | The evidence table and the disagreements | 4 | 1.2 |
| 7. Experiments | Set-up and results of E2 to E8 | 6 | 2 |
| 8. Applications and design patterns | Uses, System One with System Two | 2 | 0.3 |
| 9. Open problems | Specific gaps and directions | 3 | 0.7 |
| 10. Conclusion | | 0.5 | 0.2 |

The LaTeX skeleton is in `../paper/main.tex`.

---

## 7. Literature review method

- Method: a multivocal literature review with forward and backward snowballing. The protocol is in `../literature/protocol.md`.
- Sources: arXiv, the ACL Anthology, OpenReview, Semantic Scholar and Google Scholar. For grey literature: vendor documents, model cards, OpenRouter, Hugging Face and technical blogs.
- Two people screen 10% of the works and calculate Cohen's kappa.
- Report the counts in a PRISMA 2020 diagram.
- Check each grey source with the AACODS checklist.
- Save a copy of each vendor page on the day that you read it.

---

## 8. Evidence synthesis

This is the part that makes our survey different.

1. **Evidence table.** Make one row for each published result: source, source type (vendor or independent), system, version, dataset, task type, answer type, metric, value, conditions (region, gateway, date) and notes.
2. **Normalization.** Map the metrics to shared names. Record when a metric is not defined (for example, Microsoft's "calibration score").
3. **Analysis:**
   - Where do System One models win, and where do they lose, against System Two models, LLM readouts and trained encoders? Group by task type and amount of labeled data.
   - Do vendor benchmarks favor the vendor? Compare the rank of each system on vendor benchmarks and on independent benchmarks.
   - Why do the calibration results disagree? Group by task type, answer type, metric and version.
4. **Output:** an evidence table in the appendix, one summary figure and a list of the disagreements that E2 to E8 test.

The disagreements that we know now are in section 5 of the landscape note.

---

## 9. Experiments (summary)

Details: `2026-10-15-experiment-design.md`.

| ID | Question | Why |
|---|---|---|
| E0 | Which systems can we call, and what do they return? | Access, schemas, terms of service |
| E1 | What does the published evidence say? | The evidence synthesis (section 8). No API calls. |
| E2 | Are System One models overconfident or underconfident? | Three sources disagree. |
| E3 | How sensitive are they to wording, negation, option order, other questions and injected text? | Sources report from 1.3% to 73% changed answers. |
| E4 | When is a System One model the right tool, and when is a small trained model better? | Sources disagree on accuracy against trained models. |
| E5 | How much does decision training add to the same backbone? | Most open systems are Qwen fine-tunes. |
| E6 | How much cost does a System One to System Two cascade save? | One paper tests this for judging only (arXiv:2609.26550). It connects to ThinkBudget. |
| E7 | What changes between Jev 1.13 and Jev 1.14? | `jev-latest` changes this month. It connects to the tool-drift paper. |
| E8 | What are the real latencies under one protocol? | The claims range from 0.13 s to 0.52 s for the same model. |

---

## 10. Timeline

| Week | Dates | Work | Gate |
|---|---|---|---|
| 0 | 10-15 to 10-18 | Read the plan. Decide the team and the senior co-author. Top up the OpenRouter credit. | Team agreed |
| 1 | 10-19 to 10-25 | E0 (access audit, terms of service). Final protocol. Zotero library. Evidence table template. First searches. E7 baseline run on Jev 1.13 and 1.14. | 3 or more major systems work |
| 2 | 10-26 to 11-01 | Screening and extraction. Experiment code. Datasets. | |
| 3 | 11-02 to 11-08 | Extraction. Taxonomy v1. Pilot runs of E2 and E3 on 2 tasks. | The pilot works |
| 4 | 11-09 to 11-15 | Main runs: E2, E3, E5. | |
| 5 | 11-16 to 11-22 | E4, E6. Evidence synthesis (E1). | |
| 6 | 11-23 to 11-29 | E8. Analysis. Figures. | |
| — | 12-01 | **Decision point:** the cut-off search, the route (A, B or C) and one paper or two | |
| 7 to 8 | 11-30 to 12-13 | Write the long version. | Full draft |
| 9 | 12-14 to 12-20 | Internal review. One external reader. | |
| 10 to 11 | 12-21 to 01-03 | Short version. Last arXiv check. Submit to ARR on 2027-01-04 (route A). | Submitted |
| — | 2027-01 | IJCAI 2027 survey track (route B, expected about 01-11) or JAIR (route C) | |

---

## 11. Venues (summary)

Details and sources: `2026-10-15-venues.md`.

- **Route A:** ARR to ACL 2027, deadline 2027-01-04, 8 pages. One author must be a qualified reviewer.
- **Route B:** the IJCAI 2027 survey track, expected about 2027-01-11, 7 pages plus 2 pages of references. A "recognized expert" must be an author.
- **Route C:** JAIR (free, open access, Surveys section) or ACM TIST (welcomes surveys of new areas). No deadline.
- **Do not use:** TMLR (no surveys from 2026-09-01) and FnT in Machine Learning (no literature surveys).
- **arXiv:** only after acceptance, for a survey. Use GitHub for early visibility.

---

## 12. Team and roles (proposal)

| Role | Tasks |
|---|---|
| Lead (Aniruddha) | Scope, taxonomy, evidence synthesis, writing, coordination |
| Literature (2 people) | Search, screening (with kappa), extraction, Zotero |
| Experiments (1 or 2 people) | Code, runs, analysis, figures |
| Senior co-author | Scope review, positioning, the ARR reviewer rule or the IJCAI expert rule |

Agree on the author order and the tasks before week 1.

---

## 13. Budget (estimate)

| Item | Assumption | Estimate |
|---|---|---|
| System One models (E2, E3, E7, E8) | 18 systems, about 20,000 calls each, 600 input tokens for each call, about $0.04 for each 1M tokens | About $14 |
| System Two baselines | `gpt-6-sol` and `claude-sonnet-5-5` with reasoning low (about $75 each), plus 3 small models (about $15 together) | About $165 |
| Local GPU | 30 to 40 hours on one 80 GB GPU at about $2 for each hour | About $60 to $80 |
| Margin | 30% for pilots and repeated runs | About $70 to $80 |
| **Total** | | **About $310 to $340. Plan for $500.** |

- The details are in section 11 of the experiment design.
- The OpenRouter key had about $50 of credit left on 2026-10-12. Add credit before week 1.
- Check the prices again in E0. The prices come from vendor and OpenRouter pages on 2026-10-15.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A complete survey appears before ours. | Move fast. Keep our angle: evidence synthesis and conflict tests. Check arXiv every week. |
| Vendors change models, prices or terms during the study. | Record versions and dates. Run E7. Do all runs of one experiment in 2 weeks. |
| A vendor forbids the publication of benchmark results. | Check the terms in E0. Remove that system, or ask the vendor. |
| No senior co-author joins. | Use route C (JAIR). |
| The survey becomes a list of papers. | Write each section around the taxonomy and the disagreements, not around single papers. |
| OpenRouter changes the results of a system. | Compare 200 items through OpenRouter and through the direct API for each major system (E0). |
| The budget grows. | Run the System Two baselines on a subset first. |

---

## 15. Actions for this week (2026-10-15 to 2026-10-25)

1. Read this plan and the guide. Send questions to the team.
2. Decide the team, the author order and the roles.
3. Ask a senior researcher to join as a co-author (for example, Prof. Kaushik Roy).
4. Add credit to OpenRouter. Approve the budget.
5. Make a shared Zotero group library and a shared spreadsheet (extraction form, evidence table, screening counts).
6. Do E0: send 20 test items to each major system. Record the schemas, versions and prices. Read the terms of service.
7. Run the E7 baseline on Jev 1.13 and Jev 1.14 with the same 1,000 items.
8. Start the grey-literature search. Save a copy of each vendor page.

---

## 16. Decisions for you

1. **Team:** who writes the paper with you? Who is the senior co-author?
2. **Budget:** do you approve a limit of $500?
3. **Scope of inputs:** text only, or text and images? My recommendation: text for the main results and one small image experiment.
4. **One paper or two (decide on 2026-12-01):** one survey with the experiments inside, or a survey plus a separate research paper with the experiments. A research paper can go on arXiv at once.

---

## 17. Glossary

Use these terms in all documents and in the paper.

| Term | Meaning |
|---|---|
| System One model | A model that fits the definition in section 4.1. Vendors also say "decision model". |
| System Two model | A chat LLM or a reasoning model that generates text. |
| Input | The content that the model judges. TypeSafe calls it the "state". |
| Question | One item that the caller asks about the input. It has an answer type. |
| Predicate | A yes/no question. TypeSafe and Microsoft call the type `noul`. OpenAI calls it `predicate`. |
| Choice | A question with a fixed set of named options. |
| Score | A question with an ordered set of levels. |
| Backbone | The pretrained model under a System One model, for example Qwen3.8-27B. |
| Readout | The part that changes the hidden states into answer probabilities. |
| Calibration | The agreement between the probabilities and the real frequencies of correct answers. |
| ECE | Expected calibration error: the mean gap between confidence and accuracy over bins. |
| Brier score | The mean squared error of the probabilities. |
| Selective prediction | The model answers only when its confidence is above a threshold. |
| Cascade | A System One model answers first. A System Two model answers the items with low confidence. |
| Evidence table | Our table of all published results. |
| Grey literature | Sources that are not peer-reviewed: vendor documents, blogs, model cards. |
| Multivocal review | A literature review that includes grey literature. |
