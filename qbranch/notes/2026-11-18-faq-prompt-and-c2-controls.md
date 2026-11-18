# QBranch: third set of experiments (2026-11-18)

Status: results after the meeting with Prof. Roy on 2026-11-13. Author: Aniruddha. The earlier results are in
`notes/2026-11-18-results.md`. The raw outputs are in `runs/`. Code: `src/`, `analysis/`.

## 1. Summary

1. **The C2 control of the earlier runs was too easy.** The open slots did not keep the subject type, so the
   "random question of the same entity type" was a random question of any entity. It often did not fit the
   entity ("When did Municipal Stadium die?"). With a strict control (a random open question of the same
   type that the checker confirms can be asked about the entity), the gap is **not significant** on news,
   full MuSiQue and 2Wiki. The point estimates are +0.5 to +2.4: news +2.1 [−1.7, +5.8],
   MuSiQue +2.4 [−0.7, +5.5], 2Wiki +0.5 [−3.0, +4.0]. A check of the control (section 3.1) does not change
   this. The open slots predict which *kinds* of facts arrive for an entity type. The specific question of
   the passage adds little over a typical question for that type.
2. **The Q&A examples in the build prompt (the meeting decision) give no significant gain.** On news, the
   predicted questions of the Q&A variant beat the strict control by +4.9 [+0.8, +8.8] (+5.5 without the
   entities of the examples). But the direct test against the current prompt is +2.9 [−2.6, +8.1], so the
   gain over the current prompt is not shown. On 2Wiki, the examples change nothing (C1, C2, goal 2). On
   MuSiQue (only 23 clean examples), they do not help, and goal 2 is a little lower (−1.9 [−3.6, −0.2]).
3. **Goal 2 has a first measure.** 9.7% (MuSiQue) and 8.7% (2Wiki) of the test questions have a gold hop that
   an open slot asked for before the answer passage arrived. Only 0.3% (MuSiQue) and 1.3% (2Wiki) of the open
   slots ask for a fact that a test question needs.
4. **The open-event prompt changes the kind of slots, not the prediction.** On news, event slots go from 2%
   to 21% of the slots. They bind more often than background slots (19.4% against 10.3%), but not more often
   than the strict control (18.3%).
5. **The anchor fix does not help.** The plan with the fix is 1.3 points below the plan without anchors.
   Drop the anchor step.

## 2. What I did

| # | Work | Files |
|---|---|---|
| 1 | Build variant `qbranch_qa`: 8 held-out question-answer pairs of the dataset in each build prompt (an FAQ) | `prompts.BUILD_QBRANCH_QA`, `data.faq_pool`, `runs/*/build_qbranch_qa` |
| 2 | Build variant `qbranch_events` (news): the open questions can also be open outcomes of events | `prompts.BUILD_QBRANCH_EVENTS`, `runs/multihoprag_nall/build_qbranch_events` |
| 3 | C2 version 2: fix of the type bug, three controls, leak check for the examples | `analysis/c2_v2.py`, `runs/*/c2_<variant>.json` |
| 4 | Goal 2 measure: do the test questions need the facts that the open slots ask for? | `analysis/goal2.py`, `runs/*/goal2.json` |
| 5 | Anchor fix: the anchored facts go to the reader, but they do not choose the passages | `answer.GraphRetriever.retrieve(anchor_mode="facts")`, system `graph_qbranch_plan_factanchor` |
| 6 | A check if the strict control is itself a prediction (paraphrase or another open slot of the entity) | `analysis/c2_overlap.py`, `runs/*/c2_overlap_<variant>.json` |
| 7 | A sheet of 50 C2 checker decisions for a hand check (news) | `analysis/handcheck_sheet.py`, `runs/multihoprag_nall/handcheck_c2_qbranch.csv` |

Before the full builds, each new prompt ran on 30 passages (gate). No JSON failures; the counts of
answered and open questions were close to the baseline.

## 3. C2 with three controls (current prompt)

For each sampled open slot (600 slots that have later facts about the entity), the checker (Claude Haiku 5.5)
decides if one of the 30 nearest later facts answers the question. Each control puts the name of the entity
into another open question and sees the same later facts:

- **any type**: a random open question of any other entity (the control of the 2026-11-10 runs).
- **same type**: a random open question of another entity of the same type.
- **same type, fits**: as "same type", but the checker first confirms that a true answer can exist for this
  entity (found for 97–99% of the slots). This is the strict control.

| Stream | Predicted | Any type | Same type | Same type, fits | Predicted − fits |
|---|---|---|---|---|---|
| News, MultiHop-RAG | 14.7% | 4.7% | 9.9% | 12.8% | +2.1 [−1.7, +5.8] |
| MuSiQue full | 11.7% | 4.0% | 7.4% | 9.0% | +2.4 [−0.7, +5.5] |
| 2Wiki full | 16.3% | 8.5% | 12.6% | 15.8% | +0.5 [−3.0, +4.0] |

The gap against "same type" is +3.7 to +4.7 and excludes zero on all three streams. The gap against
"same type, fits" does not. Reading: a question that fits the entity and is typical for its type is
answered later almost as often as the question that the passage raised. So the open slots act as
type-level schema expectations ("a person has a birthplace"), not as predictions from this passage.
The "any type" control reproduces the old result (news +10.0 here, +9.0 before).

### 3.1 Is the strict control also a prediction?

The control rejects a random question only when the relation name is the same. Relation names are free text,
so the control can be a paraphrase of the predicted question, or a question that the build already opened
about the same entity (up to the batch of the slot). `analysis/c2_overlap.py` asks the checker about each row.

| Run | Control = the predicted question | Control = another open slot of the entity | Predicted − fits, rows without overlap |
|---|---|---|---|
| News, current prompt | 2% | 35% | −4.6 [−9.5, +0.0] (n = 368) |
| News, Q&A variant | 4% | 39% | +1.5 [−3.5, +6.7] (n = 343) |
| News, event prompt | 1% | 20% | −0.9 [−4.3, +2.8] (n = 462) |
| MuSiQue, current prompt | 3% | 29% | +3.3 [+0.0, +6.8] (n = 396) |
| MuSiQue, Q&A variant | 5% | 31% | +0.8 [−3.4, +5.0] (n = 379) |
| 2Wiki, current prompt | 1% | 19% | +0.6 [−3.4, +4.7] (n = 473) |
| 2Wiki, Q&A variant | 2% | 24% | −1.1 [−5.0, +3.0] (n = 439) |

The control is seldom a paraphrase of the predicted question (1–5%). It often asks a question that the
build raised about the entity in another passage (19–39%), because the sample has entities with later facts,
and these entities have many open slots. Without these rows, the gap stays near zero. Only MuSiQue reaches
the edge of significance (+3.3 [+0.0, +6.8]). So the overlap does not explain the result of this section.

## 4. The Q&A-example variant

**Design.** Each build prompt gets 8 question-answer pairs, sampled per passage (seed = passage id) from a
held-out pool. The prompt says that users asked such questions, and asks for questions about the same kinds
of facts. Leak rules:

- The pool comes from the first 100 questions (MuSiQue, 2Wiki) or from the news questions outside the
  test sample. All results use only the other questions (900, 900 and 300).
- A pool question is dropped when it shares a gold passage, a MuSiQue sub-question or a 2Wiki evidence
  triple with a test question. Pool sizes: MuSiQue 23 (most MuSiQue questions share a sub-question with
  another question), 2Wiki 82, news 100.
- C2 is also given without the slots whose entity is named in a pool question or answer.

**Results (test questions only).**

| Measure | MuSiQue | 2Wiki | News |
|---|---|---|---|
| C1 judge, `graph_qbranch_qa` − `graph_qbranch` | −1.4 [−3.7, +0.7] | +0.2 [−1.1, +1.6] | – |
| C1 recall | +0.8 [−0.5, +2.0] | +1.1 [+0.4, +1.9] | – |
| C2, predicted − fits (QA variant) | +0.3 [−2.9, +3.4] | +1.0 [−2.4, +4.4] | **+4.9 [+0.8, +8.8]** |
| C2, the same without pool entities | +0.2 [−2.9, +3.4] | +1.0 [−2.4, +4.5] | **+5.5 [+1.2, +10.1]** |
| C2, (predicted − fits) QA variant − current prompt | −2.1 [−6.5, +2.2] | +0.5 [−4.4, +5.6] | +2.9 [−2.6, +8.1] |
| C2, predicted rate (baseline → QA) | 11.7% → 11.0% | 16.3% → 16.2% | 14.7% → 17.5% |
| Goal 2, questions asked before (baseline → QA) | 9.7% → 7.8% | 8.7% → 9.4% | – |
| Goal 2, difference | −1.9 [−3.6, −0.2] | +0.8 [−0.8, +2.4] | – |

Reading:
1. The examples do not improve QA. Both QBranch graphs still beat dense retrieval on 2Wiki
   (`graph_qbranch_qa` − dense = +3.9 [+2.0, +5.9]; − dense_k10 = +2.3 [+0.6, +4.2]).
2. On news, the Q&A variant is the only run where the predicted question beats the strict control, and the
   entities of the examples do not cause it (+5.5 without them). But the direct test against the current
   prompt (+2.9 [−2.6, +8.1]) is not significant, and the gap falls to +1.5 without the overlap rows of
   section 3.1. This is one stream and one run, and it is 1 of 7 strict comparisons. Treat it as a lead, not
   a result.
3. On MuSiQue, the test is weak: only 23 clean examples. The MuSiQue training set has no shared
   sub-questions with the test set, so it can give a larger clean pool.

## 5. Goal 2: do the test questions need the open slots?

Prof. Roy's goals for the open slots: (1) later documents give valid bindings (C2), (2) enough dataset
questions use those bindings. The answer step does not use the open slots, so this measure uses the gold hops:
MuSiQue sub-questions (with the answers of earlier steps put in), and 2Wiki evidence triples. For each hop,
the checker looks at the 10 open slots with the nearest question text (not from the support passage of the
hop) and decides which ask for the same fact about the same entity.

| | MuSiQue | 2Wiki |
|---|---|---|
| Open slots per passage | 2.99 | 2.95 |
| Hops that some open slot asks for | 5.3% | 9.9% |
| Hops asked before the support passage arrived | 3.8% | 4.2% |
| Test questions with at least one such hop | 9.7% | 8.7% |
| Open slots that a test hop needs | 0.3% | 1.3% |

Example (2Wiki): the passage "That Nazty Nuisance" (batch 2) names Glenn Tryon and does not describe him.
The build opens "When was Glenn Tryon born?". A later batch gives his birth date, and a test question needs it.

Reading: about one test question in ten uses a fact that an open slot waited for. Most open slots ask for
facts that no test question needs. This is the expected shape for a benchmark, because the test questions
cover a small part of each passage.

## 6. Open-event prompt (news)

| | Current prompt | Event prompt |
|---|---|---|
| Event slots (share of the sample) | 2% (12) | 21% (124) |
| Predicted rate, event slots | 41.7% | 19.4% |
| Predicted rate, background slots | 14.1% | 10.3% |
| Strict control, event slots | 8.3% | 18.3% |
| All slots: predicted − fits | +2.1 [−1.7, +5.8] | −0.7 [−4.1, +2.7] |

The event prompt finds many more event slots. With 124 slots, they bind no more often than the strict
control. The high rate for the 12 events of the current prompt came from a small sample.

## 7. Anchor fix (full MuSiQue, 1,000 questions)

| System | Judge | F1 | Recall |
|---|---|---|---|
| graph_qbranch_plan (anchored facts choose the passages) | 0.685 | 0.622 | 0.870 |
| graph_qbranch_plan_factanchor (the fix) | 0.691 | 0.623 | 0.872 |
| graph_qbranch_plan_noanchor | 0.704 | 0.631 | 0.872 |
| dense_plan | 0.706 | 0.625 | 0.910 |

The fix gives the same passages as the plan without anchors (recall 0.872), but the judge is 1.3 points lower
(−1.3 [−2.6, −0.1]). So the anchored facts in the evidence also distract the reader. The anchor step has no
value in this form.

## 8. What this means

1. **C2 in its old form ("the open slots predict later facts better than chance") is not shown against a
   strict control.** The gap is not significant on three streams (point estimates +0.5 to +2.4). Note that the
   strict control is itself a typical prediction for the type (section 10), so this tests if the passage adds
   information, not if the system predicts better than chance. What holds: (a) the open slots are typed schema
   expectations, and later documents fill 11–16% of those that have later facts; (b) a lead, not a result:
   with FAQ examples on the news stream, the predicted question beats the strict control (+4.9), but not
   significantly more than with the current prompt.
2. **C1 holds.** The inferred-question index is the part of QBranch that works (earlier: +3.6 MuSiQue,
   +2.6 2Wiki over plain triples; 2Wiki above dense retrieval).
3. **For ICWM (2026-11-30)**, two options:
   - C1 as the main claim, with C2 as an honest analysis (typed expectations, the strict control, the FAQ
     result on news).
   - Wait, and test the FAQ result on a second dated stream with recurring entities (FinanceBench) and a
     second seed before we claim it.

Proposed next steps:
1. A hand check of the 50 decisions in `runs/multihoprag_nall/handcheck_c2_qbranch.csv` by a person.
2. Replicate the news FAQ result: a second seed for the example sample, and a second dated stream.
3. A larger clean MuSiQue pool from the MuSiQue training set.
4. Use the open slots at answer time (a filled slot as a fact with its question), so that goal 2 shows in QA.

## 9. Costs

| Item | Cost (USD) |
|---|---|
| Builds: MuSiQue QA 8.88, 2Wiki QA 4.48, news QA 4.02, news events 3.44 | 20.81 |
| Answers (QA variant on MuSiQue and 2Wiki, anchor fix) | 1.23 |
| C2 version 2 (7 runs), control overlap (7 runs), goal 2 (3 runs), event kinds, gates | about 2.3 |
| **This set** | **about 24** |
| Project total | about 59 |

## 10. Limits

- One run per configuration. The intervals cover the sample of slots or questions, not the run-to-run
  variation. Two runs of the same measure gave slightly different numbers, and the cause is not known: goal 2
  on 2Wiki gave 9.7% and 9.9% of hops asked, and the news predicted rate was 14.0% in C2 version 1 and 14.7%
  in version 2. Take about one point as the size of this noise.
- One checker model (Claude Haiku 5.5) decides C2, the fit of the control and goal 2. Nobody has checked it by
  hand yet.
- Entity matching by name merges different entities with the same name (for example two people called
  Edsel Ford, a film and a person called Eileen).
- The strict control draws from the open questions of other entities, so it is itself a typical prediction for
  the type. It tests if the passage adds information, not if the system predicts better than chance.
