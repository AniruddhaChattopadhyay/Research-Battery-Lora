# QBranch: plan for the history-context and oracle runs (written 2026-11-24, before the runs)

Status: rules that I fixed before the first paid run of this set. Author: Aniruddha (with Claude).
Source: the meeting with Prof. Roy of 2026-11-23. The earlier results are in
`notes/2026-11-19-confirmatory-results.md`.

## 1. Decisions of the meeting

- The strict control stays as it is (an open question of another entity of the same type, with the fit
  check). No new anchor.
- The metric stays: the share of open questions that a later document answers, against the control.
- Three knowledge settings:
  1. **Zero knowledge.** The build LLM sees one passage (and the relation names). Done (`qbranch`).
  2. **History context.** The build LLM also sees the related facts and open questions of memory.
  3. **History context and budgeted oracle.** As 2, and the prompt also shows some dataset questions.
- The test Q&A has no separate role. The oracle setting is its only use.
- Later, not in this set: other build models, prompt sensitivity, prompts that several LLMs design.

## 2. The new builds

- `qbranch_hist`: the prompt of `qbranch` plus a memory block. Before each batch, the code embeds the
  facts and the open questions of all earlier batches. For each passage, it puts the 20 nearest facts
  and the 8 nearest open questions in the prompt (dense retrieval with the passage text). The prompt
  tells the model to bind a memory question that the passage answers, not to ask what a known fact
  answers, and not to repeat a memory question. No other text of the prompt changes.
- `qbranch_hist_qa`: `qbranch_hist` plus the example block of `qbranch_qa`, with the same pool
  (100 held-out questions) and the same 8 examples per passage. So the four builds make a 2 × 2 design
  (history: no or yes; examples: no or yes).
- Memory holds only earlier batches. A passage does not see the other passages of its own batch.
  Batch 0 has no memory. No later fact can enter the prompt.
- Stream: news (MultiHop-RAG), 10 batches in date order, the same batches as the earlier builds.
  Build model, checker and embeddings do not change.

## 3. Checker rule (fixed before the runs)

On 2026-11-19 I found that the builds store facts such as `Taylor Swift | full_name | Taylor Swift`.
The checker accepts them as answers, and on news this helps the control (66 of 331 answered control
rows against 27 of 375 predicted rows). New rule, in code and not in the checker prompt:
**an answer does not count when the object of the fact is the entity of the slot itself, or when the
answer text is only the name of that entity** (after normalization). The rule applies to all arms and
to the old rows. The checker prompt does not change, so the old decisions stay comparable.

## 4. Quantities

All intervals are 95% bootstrap intervals that resample entities (5,000 draws), as on 2026-11-19.

- Rows: open slots of batches 1 to 8 that have a later fact (batch 0 has no memory; batch 9 has no
  later batch). New builds: all such slots, up to 4,000 per build. Zero-knowledge arm: the rows of
  `c2_qbranch_confirm.json` (fresh rows of 2026-11-19), the same batches, with the new rule.
  Examples arm without history: `c2_qbranch_qa_confirm.json`.
- **H1 (history), primary:** (predicted − strict control) of `qbranch_hist` − (predicted − strict
  control) of `qbranch`. Each arm uses the control pool of its own build.
- **H2 (oracle on top of history), primary:** the same difference, `qbranch_hist_qa` − `qbranch_hist`.
- Pass: the interval excludes zero. "Strong": the difference is +3 points or more, and predicted −
  control of the new build is +4 points or more. (The same sizes as on 2026-11-19.)
- Secondary, reported for each build: predicted rate; control rate; the gap with the shared control
  pool (agenda of `qbranch`); the gap for answers that are new to memory; open slots per passage;
  the share of all open slots that a later fact answers (ceiling × predicted rate); the gap without
  the entities of the examples (`qbranch_hist_qa`); the values without the checker rule of section 3;
  the values with batch 0.
- I report every number, pass or fail. A higher predicted rate with a control that rises by the same
  amount is not a pass (this happened with the Q&A examples on 2026-11-19).

## 5. Order and stop rule

1. Build `qbranch_hist` and `qbranch_hist_qa` on news (about USD 5 each). Run the C2 check (about
   USD 1 each).
2. MuSiQue and 2Wiki have no time order (their batches are random). I build them only if H1 on news has
   a point estimate of +1.5 points or more, or if the team decides to report all streams.
3. One build per setting. No second prompt version in this set. A change of the memory prompt after I
   see the result is a new exploratory run, and I label it as such.

## 6. Limits known before the runs

- The news is from 2023. The build model can know some outcomes from its training data.
- History changes the stored facts as well as the open questions, and each arm is checked against the
  facts of its own build. The shared control pool and the new-facts number are the checks for this.
- The rule "do not repeat a memory question" removes repeated open questions from later batches. Those
  can be the questions that later documents answer most often. So the share of all open slots that get
  an answer is a necessary secondary number.
- One checker model. The hand check of 50 decisions is still open.
