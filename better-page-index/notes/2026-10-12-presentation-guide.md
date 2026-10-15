# How to present the pilot (meeting of 2026-10-12)

Deck: `slides/pilot-results.pdf`. Numbers: `notes/2026-10-12-pilot-results-brief.md`. PageIndex background: `notes/2026-10-10-pageindex-clevrer-analysis.md`.

## 1. The goal of the meeting

You do not only report results. You want five decisions from Prof. Roy (slide 12). Every slide before slide 12 builds the case for those decisions. Plan about 30 minutes: 20 minutes for slides 1 to 11, and 10 minutes for the decisions.

## 2. Before the meeting (15 minutes)

1. Open `slides/pilot-results.pdf` in full-screen mode.
2. Open this guide and the brief in a second window.
3. Open a terminal in `~/Documents/research/better-page-index` for the optional live demo (section 6).
4. Learn the six numbers in section 3.
5. Read the limits (slide 11) and the "do not say" list (section 7) once more.

## 3. Six numbers to know

| What | Number |
|---|---|
| Overall accuracy: PageIndex, full context, graph agent | 87.5%, 95.2%, 93.1% |
| Counterfactual: PageIndex against the other two | 70.9% against 85.5% |
| Input tokens per question: PageIndex, full context, graph agent | 110,010, 17,649, 2,872 |
| Paired test: full context right and PageIndex wrong, and the reverse | 20 and 1 (p = 2 × 10⁻⁵) |
| PageIndex-only errors: evidence not read, and evidence read but still wrong | 4 and 11 (of 20) |
| Operators on 3,000 held-out videos: predictive, counterfactual | 95.2%, 88.4% (graph alone: 29.6%, 52.1%) |

## 4. The story in three sentences

Use these three sentences to open, and again to close.

1. PageIndex finds the right part of the document, but it gives the model partial pages and a large tree of lossy summaries.
2. With the right evidence, the same model answers well, so the problem is evidence access, not reasoning.
3. A dual-view graph with two operators gives that evidence exactly, at a small part of the cost.

## 5. Slide by slide

| Slide | Time | What to say | Point at |
|---|---|---|---|
| 1. Title | 0.5 min | "Last Monday we agreed that I test PageIndex on CLEVRER. Today I show the first pilot: three results, one correction to our hypothesis, and five decisions that I need from you." | Nothing. |
| 2. The pilot in one slide | 2 min | Say the story in three sentences (section 4). Give the three accuracies and the token numbers. | The 70.9% and the 110,010. |
| 3. Setup | 2 min | "I made the test fair to PageIndex: the same model for all, the index model that its README recommends, and a PDF with clean headings and bookmarks. I used the ground truth on purpose, so each error is a retrieval error or a reasoning error. Full context is the upper bound: perfect retrieval." | The three colored systems. |
| 4. Accuracy | 2 min | "The largest gap is on counterfactual questions. This matches what you saw on the manufacturing data. The paired test is strong: 20 against 1." | The counterfactual bars. |
| 5. Cost | 1.5 min | "PageIndex is also the most costly and the slowest. Its tree for 30 runs is about 90,000 tokens, larger than the evidence, and the agent reads parts of it again in every turn." | The $0.224 bar. |
| 6. Why 1 | 2 min | "Navigation works: PageIndex never read a wrong run. That is why it does well on FinanceBench, where each question names the company and the year. The problem starts inside the chapter: for most descriptive questions it reads one page. In 4 errors, the evidence was on pages that it did not read." | 0 wrong runs; the 4. |
| 7. Why 2 | 2 min | "Even from a clean PDF, the tree is wrong for this data: merged headings, wrong nesting, summaries that drop the numbers. And this is the bigger effect: in 11 errors PageIndex had the evidence and still answered wrong, while the same model with only the run's pages got all 11 right. Events and states do not fit a table of contents. This is your argument for a data graph and a process graph." | The last bullet. |
| 8. Examples | 1.5 min | One error of each kind. Left: it did not read the evidence. Right: it read the event log with the answer on it, and still said 3. Optional: run the live demo (section 6). | "page 57 only"; "answered 3". |
| 9. Correction | 2.5 min | "I want to be open about one result. We expected PageIndex to fail because the answers are in no document. But with all pages of the run, the model does the physics itself. So I propose a new claim: tree retrieval does not deliver clean, complete evidence for these questions. This claim does not depend on how good the model is, so it will not go out of date." | The revised claim. |
| 10. Operators | 2 min | "Then I tested two operators for the process graph: intervention and world simulation. With no LLM, they get 95.2% and 88.4% on held-out videos. That is near the best published CLEVRER result, but our perception is perfect, so it is not a fair comparison. These operators are different from the search rollouts in your figure: rollouts decide whether to keep a path; operators compute new events." | The two green bars. |
| 11. Meaning and limits | 2 min | Say the limits before he asks. Then say what the pilot supports. | The limits column. |
| 12. Decisions | 10 min | Ask each question. Give your recommendation (section 8). Write down each answer. | Each decision. |

## 6. Optional live demo (no API cost)

Run these if Prof. Roy asks for evidence, or after slide 8:

```bash
cd ~/Documents/research/better-page-index
uv run python analysis/show_example.py 10572 1     # descriptive: PageIndex read 1 of 9 pages
uv run python analysis/show_example.py 10475 9     # descriptive: PageIndex read the event log and still answered 3
uv run python analysis/show_example.py --list      # all 14 questions that only PageIndex missed
```

Each command shows the question, the correct answer, the pages that PageIndex read, and the tool calls of our graph agent.

## 7. Do not say these things

| Do not say | Why | Say this instead |
|---|---|---|
| "PageIndex fails on video." | No document system reads video. It is not a finding. | "Tree retrieval misses evidence that is spread over time." |
| "Our system beats full context." | It does not: 93.1% against 95.2%. | "Our agent matches perfect retrieval on counterfactual questions with 1/6 of the tokens." |
| "We beat the state of the art on CLEVRER." | Our perception is ground truth. | "Simple operators reach near the published scores, with perfect perception." |
| "LLMs cannot reason counterfactually." | Our pilot shows that they can, with the right evidence. | "The bottleneck is evidence access." |
| "PageIndex is useless." | It navigates to the right chapter every time. | "It works at chapter level and fails inside the chapter." |

## 8. Your recommendations for the five decisions

1. **The role of CLEVRER:** use it as a diagnostic probe with exact ground truth. Make the manufacturing corpus the main benchmark.
2. **Operators:** add intervention and world simulation to the process graph as tools, next to the search rollouts.
3. **The claim:** adopt "evidence access, not reasoning".
4. **Models and budget:** keep one strong model for all systems. PageIndex costs about $0.13 for each question after caching ($0.22 at list price), so 1,000 PageIndex questions cost about $130, and a larger corpus costs more because the tree grows. Full context costs about $0.04 and our agent about $0.007 for each question.
5. **Ash:** one option is that Ash builds the offline phase (LLM extraction of the two graphs), and you scale the experiments and the baselines.

## 9. Likely questions and short answers

**"Is this fair to PageIndex?"**
Yes. It used the same model as the others, the index model that its README recommends, and a clean PDF with bookmarks. We tested the open-source SDK in local mode. The cloud product uses a different search that the docs describe without details, so we can test it later with a PageIndex API key.

**"If full context is so good, why do we need retrieval?"**
For one run, we do not. A real plant archive does not fit in one context window, and the cost grows with its size. Our agent gets the same counterfactual accuracy with 1/6 of the tokens. The next step is a 300-run corpus that does not fit.

**"Is the simulator not cheating?"**
No. The operators are part of the proposed design. They use only the facts in the report and two constants that we tuned on other videos. The LLM still decides which tool to call. We also report the operators alone, with no LLM.

**"Are 248 questions enough?"**
For a pilot, yes. The overall paired difference is significant (p = 2 × 10⁻⁵). The intervals for each type are wide, mainly for the 22 predictive questions. Scaling is the first next step.

**"Why did PageIndex do well on predictive questions?"**
The answer depends on the last frames, and the last page of a run is easy to find. It read 2.4 pages on average. The sample is small (22 questions).

**"What about the MCTS and the value function that we discussed?"**
The open-source code has neither. The docs say that the cloud API uses MCTS and a hybrid search with embeddings, but they give no details. Our result is for the open-source PageIndex.

**"Where are the search rollouts and the sufficiency check?"**
They are not built yet. The pilot tested the graph and the operators first. Rollout gating saves cost, so we must measure it on a large corpus.

**"How does this transfer to manufacturing?"**
The failure comes from evidence that is spread over time and entities. Run logs, sensor records and assembly steps have the same shape. In manufacturing documents, counterfactual answers often come from written rules, so the intervention operator on the process graph is the key part there.

**"How did you build the graph?"**
Code built it directly from the CLEVRER annotation files, which hold the same facts as the PDF. The graph agent never read the PDF. In the real system, an LLM extracts the two graphs from documents. That is a next step, and it is a good task for Ash.

**"What did it cost?"**
About $47 for everything today, on OpenRouter.

## 10. After the meeting

1. Write the five decisions at the end of the brief.
2. Send the deck and the brief to Prof. Roy and Ash.
3. Update the plan in section 8 of the brief with the agreed next steps.
