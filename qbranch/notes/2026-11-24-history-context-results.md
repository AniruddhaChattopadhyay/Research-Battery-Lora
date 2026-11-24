# QBranch: results of the history-context and oracle runs (2026-11-24)

Status: results. Author: Aniruddha (with Claude). The rules are in
`notes/2026-11-24-history-context-plan.md`. I wrote them before the runs. Stream: news (MultiHop-RAG),
10 batches in date order. Rows: open slots of batches 1 to 8 that have a later fact. Intervals: 95%,
bootstrap over entities.

## 1. Summary

1. **By the rule of the plan, history context passes (H1).** Predicted − strict control is +7.9 points
   for the history build and +2.9 for zero knowledge. The difference is +5.0 [+1.5, +8.7].
2. **The gain does not come from better questions.** A history build shows its open questions to later
   passages, and later passages then store the facts that answer them. When I check the same questions
   against the facts of the zero-knowledge build (no question steered them), the difference is
   −1.2 [−3.6, +1.4]. This check was not in the plan. It is exploratory, but it decides the reading.
3. **The oracle on top of history does not pass (H2):** +1.3 [−1.5, +4.1].
4. **The checker rule changes the old news number.** Without self-facts such as
   `Taylor Swift | full_name | Taylor Swift`, the zero-knowledge gap on news is +2.9 [+1.3, +4.6]
   (it was +2.1 [+0.4, +3.8] on the same rows without the rule).

So the correct statement is: **history does not make the system predict better; it makes the system
close its own open questions.** The first is the claim that we wanted. The second is a real property
of the system (the agenda directs what later reading stores), and it needs its own test.

## 2. Primary numbers (each build against its own facts)

| Build | Rows (entities) | Predicted | Strict control | Gap | Gap, new facts only |
|---|---|---|---|---|---|
| Zero knowledge (`qbranch`) | 2,736 (888) | 11.0% | 8.1% | +2.9 [+1.3, +4.6] | +2.5 [+1.4, +3.6] |
| Examples (`qbranch_qa`) | 2,718 (859) | 13.0% | 9.2% | +3.8 [+1.9, +5.8] | +2.1 [+0.9, +3.5] |
| History (`qbranch_hist`) | 3,401 (834) | 17.0% | 9.0% | +7.9 [+4.8, +11.4] | +6.6 [+3.9, +10.0] |
| History and examples (`qbranch_hist_qa`) | 3,333 (832) | 19.4% | 10.2% | +9.2 [+6.1, +13.0] | +7.4 [+4.4, +11.2] |

| Test | Difference of the gaps | With the shared control pool | New facts only | Result |
|---|---|---|---|---|
| H1: history − zero knowledge | +5.0 [+1.5, +8.7] | +6.4 [+2.4, +10.3] | +4.2 [+1.2, +7.6] | pass, "strong" size |
| H2: history and examples − history | +1.3 [−1.5, +4.1] | +1.4 [−1.6, +4.5] | +0.7 [−1.8, +3.3] | no pass |
| Examples − zero knowledge (old rows, new rule) | +0.9 [−1.2, +3.1] | | | no pass |
| History and examples − examples | +5.4 [+1.7, +9.6] | | | (secondary) |

Other secondary numbers:
- Without the checker rule: H1 +5.9 [+2.2, +9.7], H2 +0.3 [−2.7, +3.4].
- With batch 0: H1 +5.3 [+2.1, +8.7], H2 +0.7 [−1.9, +3.1].
- Open slots per passage: 3.10 (zero knowledge), 3.33 (history), 3.38 (history and examples).
- Share of all open slots that a later fact answers (sampled rows, checker rule not applied):
  3.7%, 5.2% and 5.8%.
- Name-only answers that the rule removed (predicted, control): zero knowledge 27 and 66; history 18
  and 22; history and examples 35 and 93; examples 91 and 124.

## 3. Where the gain comes from (exploratory, after the primary numbers)

- In the history build, 592 predicted rows are answered. For 431 of them (73%), the passage that gave
  the answering fact had the question in its memory block. For history and examples: 464 of 681.
  A control question is never in a memory block.
- **Check against facts that no question steered.** I checked the open questions of the two history
  builds against the later facts of the zero-knowledge build (`--facts_from qbranch`):

| Questions from | Rows | Predicted | Strict control | Gap | Gap − zero-knowledge gap |
|---|---|---|---|---|---|
| Zero knowledge | 2,736 | 11.0% | 8.1% | +2.9 [+1.3, +4.6] | |
| History | 3,402 | 9.9% | 8.1% | +1.7 [−0.2, +3.7] | −1.2 [−3.6, +1.4] |
| History and examples | 3,206 | 11.9% | 8.1% | +3.7 [+1.2, +6.6] | +0.8 [−2.1, +4.0] |

- So the questions of a history build are not better predictions of what later documents say. The gain
  of section 2 comes from the later passages: with the question in view, the model stores the fact
  that answers it.
- **Are the steered facts real?** Not decided. I asked the checker model if the source passage states
  the answering fact (300 rows per build). The rate is the same for steered facts (41%), other facts
  of the history build (39%) and facts of the zero-knowledge build (38%), so there is no sign that
  steering adds invented facts. But the absolute rate is not usable: I read five rejected facts, and
  the passage stated all five. A person must check a sample.

## 4. What this means for the paper

- "Open slots predict later facts" (C2 as a prediction claim): on news, about +3 points over the
  strict control with the corrected checker. History and examples do not increase it.
- "The open slots direct later reading" (a new claim): with history, later passages fill 17% of the
  open slots with later facts, against 11%, and the control stays at 9%. This is an effect of the
  write path, not a forecast. A fair test needs (a) a hand check that the bound facts are in the
  passages, and (b) a test that the extra facts help a task, for example question answering with the
  history graph against the zero-knowledge graph.
- The budgeted oracle adds nothing that these tests can see, with or without history.

## 5. Not done

- MuSiQue and 2Wiki history builds. The stop rule of the plan allows them (H1 is above +1.5), but
  section 3 changes what they can show, so the team must decide first. About USD 8 each.
- The checker rule on the MuSiQue and 2Wiki rows of 2026-11-19.
- The hand check of the checker (50 decisions) and of the steered facts.

## 6. Costs

| Item | Cost (USD) |
|---|---|
| Build `qbranch_hist` | 4.85 |
| Build `qbranch_hist_qa` | 5.22 |
| C2 checks of the two builds (own facts) | 1.57 |
| C2 checks against the zero-knowledge facts | 0.83 |
| Source-passage check and smoke test | 0.07 |
| **This set** | **about 12.5** |
| Project total | about 74 |

Files: `runs/multihoprag_nall/build_qbranch_hist*`, `c2_qbranch_hist*_h.json`,
`c2_qbranch_hist*_h_basefacts.json`, `c2_hist_H1.json`, `c2_hist_H2.json`, `c2_hist_basefacts_*.json`,
`*_faithful.json`. Code: `src/build.py` (class `Memory`), `src/prompts.py`, `analysis/c2_v2.py`
(`--facts_from`), `analysis/c2_confirm.py` (`--min_batch`, `--raw`), `analysis/c2_faithful.py`.

## 7. Added on 2026-11-25: the other streams, and question answering

Section 5 listed these as not done. The team decided to run them. Only the history setting, because the
oracle did not pass on news. MuSiQue and 2Wiki have no time order: their batches are random.

### 7.1 History against zero knowledge on three streams (batches 1 to 8)

Gap = predicted − strict control. "Own facts": each build against its own later facts (the planned
test). "Unsteered facts": the questions of the history build against the later facts of the
zero-knowledge build (exploratory, section 3).

| Stream | Zero-knowledge gap | History gap, own facts | Difference, own facts | History gap, unsteered facts | Difference, unsteered facts |
|---|---|---|---|---|---|
| News | +2.9 [+1.3, +4.6] | +7.9 [+4.8, +11.4] | +5.0 [+1.5, +8.7] | +1.7 [−0.2, +3.7] | −1.2 [−3.6, +1.4] |
| 2Wiki | +2.0 [+0.3, +3.8] | +6.7 [+4.9, +8.5] | +4.6 [+2.3, +7.0] | +2.9 [+1.0, +4.9] | +0.9 [−1.7, +3.4] |
| MuSiQue | −0.2 [−1.8, +1.3] | +0.9 [−0.8, +2.6] | +1.1 [−1.0, +3.2] | −4.0 [−5.8, −2.2] | −3.8 [−6.0, −1.7] |

Rates (predicted, control). 2Wiki: zero knowledge 13.5%, 11.5%; history, own facts 19.2%, 12.5%;
history, unsteered facts 16.5%, 13.6%. MuSiQue: zero knowledge 11.0%, 11.2%; history, own facts 12.6%,
11.7%; history, unsteered facts 9.8%, 13.8%.

- The planned test passes on news and 2Wiki and does not pass on MuSiQue.
- With unsteered facts, history gives no gain on any stream. On MuSiQue the history questions are
  worse than their control. A possible cause (not tested): the prompt forbids questions that memory
  can answer, and in MuSiQue many later passages repeat known facts.
- So the reading of section 1 holds on three streams: the gain is steering of later reading, not
  better prediction.

### 7.2 The checker rule on the zero-knowledge rows of 2026-11-19 (all batches)

| Stream | Predicted | Strict control | Gap | Gap, new facts only |
|---|---|---|---|---|
| News | 11.1% | 8.4% | +2.6 [+1.0, +4.4] | +2.0 [+0.8, +3.2] |
| MuSiQue | 10.5% | 10.9% | −0.4 [−1.8, +1.0] | +1.7 [+0.5, +3.0] |
| 2Wiki | 13.7% | 12.0% | +1.7 [+0.1, +3.3] | +2.1 [+0.5, +3.7] |
| Mean | | | +1.3 [+0.4, +2.3] | +1.9 [+1.2, +2.7] |

Without the rule the mean gap was +0.9 [+0.0, +1.9]. File: `runs/c2_confirm_T1_rule.json`.

### 7.3 Question answering on news (300 questions, paired bootstrap over questions)

| Graph | Judge accuracy | Recall@5 | Judge, against zero knowledge |
|---|---|---|---|
| Zero knowledge | 74.0% | 31.1% | |
| History | 72.0% | 31.4% | −2.0 [−6.7, +2.7] |
| History and examples | 73.0% | 30.6% | −1.0 [−5.0, +3.0] |
| Examples | 69.3% | 27.3% | −4.7 [−9.3, +0.0] |

The history build stores about as many facts as zero knowledge (35,203 against 35,375). Steering
changes which facts the build stores, and that does not change the answers. News is a weak test: the
reader without evidence gets 86.7%. I did not run question answering on the other streams.

### 7.4 Costs of this addition

2Wiki build and checks 5.26; MuSiQue build and checks 10.06; question answering 0.61. About USD 16.
Project total: about USD 90.
