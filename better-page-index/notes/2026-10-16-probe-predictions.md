# Probe predictions (written 2026-10-16, before any run)

This file states what we expect before we run the probe. The probe tests the five assumptions of tree retrieval (A1 to A5 in `notes/2026-10-15-research-plan.md`) on two public datasets. The results go into a separate file.

## 1. Setup

Three systems answer every question. All three use `gpt-5.6-sol` with reasoning effort "medium" and the same task instructions.

| System | What it gets |
|---|---|
| PageIndex 0.4.1 | Local mode. The Flash index uses `gpt-5.6-luna` for the summaries. The agent has at most 16 turns. |
| Full context | The text of the whole PDF, in one prompt. |
| Keyword agent | Two tools over the pages of the same PDF: BM25 search, and page reading. At most 16 turns. |

## 2. Samples

| Dataset | Questions | Metric |
|---|---|---|
| DesignQA, Compilation | All 30. Each asks for all rules about one topic. The gold lists have 3 to 240 rules (median 38). | F1 over rule numbers (the official metric) |
| DesignQA, Retrieval | 20, random (seed 7). Each asks for the text of one rule, by its rule number. | Token F1 (the official metric) |
| LongMemEval S | 5 of each type, random (seed 7): single-session-user, multi-session, knowledge-update, temporal-reasoning, abstention | The official judge prompt for each type, with `gpt-4o-2024-08-06` |

The DesignQA document is one PDF of 140 pages with 588 bookmarks. For LongMemEval, we render the history of each question into one PDF, with one chapter and one bookmark for each session, in date order.

## 3. Predictions

| # | Prediction | Assumption that it tests |
|---|---|---|
| P1 | On DesignQA Retrieval, the mean F1 of PageIndex is within 0.05 of full context. | A1 and A2 are true: the rule number is in a bookmark title, and the evidence is one rule. |
| P2 | On DesignQA Compilation, the mean F1 of PageIndex is at least 0.10 below full context. | A2 is false: the evidence is in many sections. |
| P3 | On Compilation, the F1 of PageIndex falls as the number of gold rules grows (Spearman correlation below 0). | A2 |
| P4 | On Compilation, PageIndex reads fewer than half of the pages that hold gold rules (median over the 30 questions). | A2 and the search policy (it stops early) |
| P5 | On Compilation, the keyword agent has a higher mean F1 than PageIndex. | A1: keyword search finds the topic in the page text, not only in the labels. |
| P6 | On LongMemEval single-session-user, PageIndex is wrong on at most one more question than full context. | A1 and A2 are mostly true: one session holds the evidence. |
| P7 | On LongMemEval multi-session, knowledge-update and temporal-reasoning together (15 questions), PageIndex is wrong on more questions than full context. | A2 and A4 are false: the evidence is in several sessions, or the answer is the latest value or an order of events. |
| P8 | In most PageIndex errors of P7, PageIndex did not read all the answer sessions. | A1 and A2 |
| P9 | On abstention questions, PageIndex reads more sessions than on single-session-user questions. | A4: to show absence, the agent must read more. |
| P10 | The labels of the DesignQA tree hold more than 15% of the words of the PDF. | The trade-off between A1 and A5: a fine tree has large labels. |

## 4. What would show that the model is wrong

- If P2 and P7 are both false, PageIndex does as well as full context where the model predicts failure. Then the model is wrong for these datasets, and we must revise it before we build the method.
- If P1 or P6 is false, PageIndex fails even where the model predicts success. Then something other than A1 to A5 causes failures, and we must find it in the traces.

## 5. Limits of this probe

- LongMemEval has only 5 questions of each type. So P6 to P9 are weak tests. They show direction, not significance.
- The Compilation gold comes from a fixed rule: a rule is relevant when its text contains the topic word, or when it is a sub-rule of such a rule, or when such a rule cites it. The question does not state this rule. So all systems lose F1 for the same reason, and only the differences between systems matter.
- The budget is about $35 of the $50 credit. A guard stops the runs when less than $5 of credit is left.
