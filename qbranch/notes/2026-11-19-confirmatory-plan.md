# QBranch: plan for the confirmatory runs (written 2026-11-19, before the runs)

Status: rules that I fixed before the first paid run of this set. Author: Aniruddha (with Claude).
The earlier results are in `notes/2026-11-18-faq-prompt-and-c2-controls.md`.

Aim: find out if a positive C2 result exists that can carry a paper. The rules below make sure that a
"pass" is a real result and not the best of many tries.

## 1. Rules for all tests

- **Fresh rows only.** No test uses the 600 slots per run that the 2026-11-18 analysis saw.
  `analysis/c2_v2.py --exclude` removes them by key.
- **Intervals by entity.** Many slots share an entity and its later facts. All intervals are 95%
  bootstrap intervals that resample entities, not rows (5,000 draws).
- **Primary numbers use all fresh rows.** The rows without control overlap and the rows without example
  entities are secondary.
- **No file of the 2026-11-18 set is overwritten.** New outputs have the tag `confirm` or a new build
  directory. Each example pool has its own file.
- **Two confirmatory tests (T1, T2).** A third (T3) runs only under the stop rule. One exploratory
  variant at most (section 4). I report every test, pass or fail.
- **The checker is still not checked by a person.** A pass is provisional until the hand check is done.

## 2. T1: does the passage add anything? (current prompt, three streams)

- Quantity: predicted − strict control (same type, fits), per stream, on fresh rows. The pooled value is
  the mean of the three stream values (news, full MuSiQue, full 2Wiki).
- Pass: the interval of the pooled value excludes zero.
- Size: a pooled gap below +3 points is "real but small". It does not make a stronger paper.
  The 2026-11-18 point estimates were +2.1, +2.4 and +0.5.

## 3. T2 and T3: the news FAQ lead

- Quantity (the same as the lead of 2026-11-18, which was +2.9 [−2.6, +8.1]):
  (predicted − strict control) of the Q&A variant − (predicted − strict control) of the current prompt.
  Each arm uses the control pool of its own build. News stream, all fresh rows of both builds.
- Pass (T2): the interval excludes zero.
- Size: "strong" needs a difference of +3 points or more, and predicted − control of the Q&A variant of
  +4 points or more.
- Secondary: the same with one shared control pool (the agenda of the current-prompt build) for both
  arms; the predicted rates of the two arms; the rows without the entities of the examples.
  Reason: the control of each arm comes from its own agenda, so a prompt that makes the full agenda
  better also lifts its control.
- Stop rule: if the T2 point estimate is below +1.5 points, I do not build again.
- T3 (only if T2 passes or its estimate is +1.5 or more): a new news build with a new example pool
  (pool seed 2). The same quantity. The claim "Q&A examples make the open slots predict more than the
  type" needs both T2 and T3 with estimates of +2 or more and a pooled interval that excludes zero.
  After that, a second dated stream is necessary before a paper claims it.

## 4. The one exploratory variant

Name: `qbranch_fb` (feedback). Decision to run: after T1 and T2.

- Idea: the system learns from its own bindings, with no task labels. At the start of each batch, the
  build prompt shows open questions of earlier batches that later documents answered, and some that
  they did not answer. This replaces the dataset Q&A examples by feedback of the stream itself.
- Quantity: the share of open slots (with later facts) that a later fact answers, feedback build −
  current build, for slots of batches 2 to 8. Also predicted − shared strict control, and confirmed
  open slots per passage.
- Split: the entities are split in two halves by a hash of the name. I look at the first half. The
  second half is the confirmation.
- Size for "strong": +4 points or more in the answered share on the second half, the interval excludes
  zero, and confirmed open slots per passage not lower than for the current prompt.

## 5. Exploratory look before this plan (no model calls)

I counted only the answers that are new to memory at the batch of the slot (memory did not link the
entity to the value before). On the 2026-11-18 rows, predicted − strict control for new answers is:
news +1.0 [−1.9, +3.8], news Q&A +3.6 [+0.7, +6.6], news events +1.2 [−1.9, +4.1],
MuSiQue +3.8 [+1.0, +6.5], MuSiQue Q&A +0.0 [−2.9, +2.9], 2Wiki +1.0 [−2.4, +4.5],
2Wiki Q&A +0.7 [−2.7, +4.0]. This is a mixed result (2 of 7 intervals exclude zero, and the paired
variants do not agree). It is not a finding. The confirmatory runs report this number as a secondary
value.
