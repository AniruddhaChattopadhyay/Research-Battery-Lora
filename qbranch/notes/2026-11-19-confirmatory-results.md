# QBranch: results of the confirmatory runs (2026-11-19)

Status: results. Author: Aniruddha (with Claude). The rules are in
`notes/2026-11-19-confirmatory-plan.md`. I wrote them before the runs. The earlier results are in
`notes/2026-11-18-faq-prompt-and-c2-controls.md`.

## 1. Summary

1. **T1 (does the passage add anything?): a pass at the edge, and the size is small.** On 8,366 fresh
   open slots of three streams, the predicted question beats the strict control by +0.9 points
   [+0.0, +1.9] (mean of the streams). By the rule of the plan, a gap below +3 points is "real but
   small" and does not make a stronger paper.
2. **T2 (the news FAQ lead): not replicated.** Q&A variant − current prompt is +0.6 [−1.7, +2.9] on
   3,118 and 3,145 fresh slots. The first sample of 600 gave +2.9 [−2.6, +8.1]. The estimate is below
   +1.5, so the stop rule applies: no new build (T3 is not run).
3. **No strong positive C2 result exists in these data.** The open slots are type-level expectations.
   The question of the passage adds about one point over a typical question for the entity type.
4. I did not run the exploratory variant `qbranch_fb`. Section 5 gives the reason.

## 2. T1: predicted − strict control, current prompt, fresh rows

Intervals resample entities (5,000 draws).

| Stream | Fresh rows | Entities | Predicted | Strict control | Gap | Gap, new facts only (secondary) |
|---|---|---|---|---|---|---|
| News, MultiHop-RAG | 3,145 | 991 | 11.9% | 10.5% | +1.4 [−0.5, +3.3] | +1.3 [−0.0, +2.7] |
| MuSiQue full | 2,918 | 1,630 | 10.7% | 11.0% | −0.3 [−1.7, +1.2] | +1.8 [+0.6, +3.0] |
| 2Wiki full | 2,303 | 1,242 | 13.8% | 12.1% | +1.7 [+0.1, +3.3] | +2.1 [+0.5, +3.7] |
| **Mean of the streams** | 8,366 | | | | **+0.9 [+0.0, +1.9]** | +1.7 [+0.9, +2.6] |

- The lower end of the pooled interval is +0.02 points. Read the result as "about one point, at the
  edge of zero", not as a clear effect.
- The MuSiQue estimate of 2026-11-18 (+2.4) did not come back (−0.3).
- **New facts only** counts an answer only when memory did not link the entity to the value at the
  batch of the slot. This gap is more stable (+1.3 to +2.1 on all three streams), but it is a secondary
  number and it is below the +3 bar. A likely cause on MuSiQue (not tested): the build LLM saw the
  passage, so its question does not ask for a fact that the passage gives. The control question can ask
  for such a fact, and then a later repeat counts as "answered" but not as "new". On news, this cause
  does not apply: the known share is about 5 points for the predicted question and for the control.
  A fair test needs a control that memory cannot answer. I did not run it.
- **The first samples were high by chance.** The predicted rate is lower on the fresh rows in all four
  runs (news 14.7% → 11.9%, news Q&A 17.5% → 15.9%, MuSiQue 11.7% → 10.7%, 2Wiki 16.3% → 13.6%, all
  sampled rows). The checker is not the cause: a new call for the first 600 news slots gives the same
  decision in 99.5% of the rows (14.7% → 14.5%, `analysis/c2_recheck.py`). The share of empty checker
  answers is the same in the old and the fresh files (2% to 5%).

## 3. T2: the Q&A variant on news, fresh rows

| | Current prompt | Q&A variant | Q&A − current |
|---|---|---|---|
| Fresh rows (entities) | 3,145 (991) | 3,118 (969) | |
| Predicted (rows with a control) | 11.9% | 16.0% | +4.1 [+2.1, +6.0] |
| Strict control, own agenda | 10.5% | 14.1% | |
| **Predicted − control (primary)** | +1.4 [−0.5, +3.3] | +2.0 [−0.1, +4.2] | **+0.6 [−1.7, +2.9]** |
| Predicted − control, shared control pool | +1.4 | +1.2 [−0.9, +3.4] | −0.2 [−2.6, +2.2] |
| Predicted − control, without example entities | +1.4 [−0.7, +3.5] | +2.3 [+0.1, +4.5] | +0.9 [−1.6, +3.3] |
| Predicted, new facts only | 6.9% | 8.4% | +1.5 [−0.2, +3.1] |

Reading:
- The Q&A examples raise the share of open questions that later documents answer (+4.1 points,
  the interval excludes zero, also without the entities of the examples).
- The strict control rises by the same amount. A control question from the agenda of the
  current-prompt build is answered almost as often as the predicted question of the Q&A build (the gap
  with the shared pool is +1.2). So the gain does not come from the open questions.
- The gain does not come from other entities. On the 694 entities that both samples share, the
  predicted rate rises by +4.2 [+2.2, +6.4] (each entity with the same weight). The entity types and
  the number of later facts per slot are the same in both arms (17.0 and 16.8).
- The likely cause is the fact store (not tested directly). Each arm is checked against the facts of
  its own build. The Q&A build stores fewer facts (32,177 against 35,375), but the examples also steer
  its answered questions, so its facts answer a typical question more often.
- Most of the gain is facts that memory already has. For new facts, the gain is +1.5 [−0.2, +3.1].
- So the examples change which facts the build stores, and the collection repeats many of these facts.
  They do not make the predictions more specific to the passage. This is a secondary result; the
  primary test failed.

## 4. What the two tests settle

- The claim "the open slots predict later facts better than a typical question for the type" is at
  most a one-point effect. Larger samples will not make it larger.
- The claim "Q&A examples make the open slots predict more than the type" is not supported.
- Every larger number that these runs can show (any-type control +10, predicted rate +4.1, new facts
  +1.7) has a simpler reading that a reviewer will find: a control that does not fit the entity, a
  change of the fact store, or a control that can ask for what the passage said.
- The news and 2Wiki builds have no unseen slots with later facts left. A later test needs a new build
  or a new stream.

## 5. Why I did not run the exploratory variant

The plan allowed one exploratory variant (`qbranch_fb`, feedback from the stream's own bindings). T2
shows what a change of the prompt examples does: it moves the full agenda, and the strict control moves
with it. Feedback examples are the same kind of change. I expect a higher answered share, the same gap
against the control, and mostly repeated facts. That result would look positive and would not hold
against the control. I estimate the chance of a real "strong" result as low, so I stopped. The build
costs about USD 5 and about 2 hours if the team wants it.

## 6. Costs

| Item | Cost (USD) |
|---|---|
| Four confirmatory runs (11,757 sampled slots, checker Claude Haiku 5.5) | 1.48 |
| Checker test-retest (600 slots) | 0.04 |
| **This set** | **about 1.5** |
| Project total | about 61 |

## 7. Limits

- One checker model decides all numbers. It is stable (99.5% test-retest), but nobody has checked by
  hand if it is correct (`runs/multihoprag_nall/handcheck_c2_qbranch.csv`).
- One build per configuration. The fresh rows come from the same builds as the 2026-11-18 rows.
- The strict control is a typical question for the type from the system's own agenda.
- Files: `runs/*/c2_qbranch_confirm.json`, `runs/multihoprag_nall/c2_qbranch_qa_confirm.json`,
  `runs/c2_confirm_T1.json`, `runs/multihoprag_nall/c2_confirm_T2.json`. Code: `analysis/c2_v2.py`
  (new options), `analysis/c2_confirm.py`, `analysis/c2_recheck.py`.
