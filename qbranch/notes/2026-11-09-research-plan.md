# QBranch research plan, v0 (2026-11-09)

Status: draft by Aniruddha, for Prof. Roy and Ashraf. Source: the kickoff meeting
(`notes/2026-11-06-kickoff-meeting.md`) and the figure (`figures/2026-11-06-qbranch-architecture.png`).
Pilot results go in `notes/2026-11-09-pilot-log.md`.

## 1. Claim to test

An LLM can build a **symbolic world model** of a growing document collection: a knowledge graph plus
an agenda of the questions that the collection answers or raises. The system optimizes this
representation with its own inferred questions. It does not use the labels of the downstream task.
We claim three things:

1. **C1 (representation).** A graph built for the inferred questions answers the real user questions
   better, or at a lower cost, than a graph built without questions (plain triple extraction).
2. **C2 (prediction).** The open slots (unbound schema elements) predict the facts that later documents
   bring. This is the "world model" property: the model predicts what the world will show next.
3. **C3 (view management).** Specialized views, kept only when they help on the inferred questions,
   improve accuracy or cost on the real questions without a regression on earlier questions.

## 2. Map from the figure to the code

| Figure part | Component | Code (v0) | Status |
|---|---|---|---|
| (a) Persistent memory | Unified graph, agenda, open slots, source store | `src/graph.py` | v0 done |
| (b) Steps 1–3 | Question inference, schema, fact binding (one LLM call per passage) | `src/build.py`, `src/prompts.py` | v0 done |
| (b) Schema normalization | Merge synonym relations and entity aliases | — | needed: 1,220 relation names after 994 passages |
| (c) Answering | Seed facts + inferred-question match + 1-hop expansion + reader | `src/answer.py` | v0 done, no LLM path plan yet |
| (c) Judge | Independent judge (other model family) | `src/evaluate.py` | v0 judges correctness only |
| (d) View management | Keep, merge or create a view, tested on inferred questions | — | not started |
| (a) Planner, controller | Budget allocation, routing | — | not started (the ICAPS angle) |

Models (v0, OpenRouter, checked on 2026-11-09):

- Build: `deepseek/deepseek-v4.1-flash` with reasoning off. It takes 3–5 s and costs about $0.0005 per passage.
- Reader: `openai/gpt-6-luna`, reasoning effort low.
- Judge: `anthropic/claude-haiku-5.5` (a different family from the reader).
- Embeddings: `google/gemini-embedding-2`. On the 100-question HotpotQA slice, Recall@5 was 0.97
  (voyage-4: 0.90, cohere embed-5-fast: 0.82).

## 3. Experimental protocol

### Datasets

1. **HotpotQA, MuSiQue, 2WikiMultiHopQA** in the HippoRAG 2 setting (1,000 questions each, pooled
   corpus: 9,811 / 11,656 / 6,119 passages). This setting lets us compare with published graph-RAG numbers.
   Files: `data/hipporag2/` (from `osunlp/HippoRAG_2` on Hugging Face).
2. **FinanceBench** (150 open questions over 10-K/10-Q filings). The filings have dates, so the stream
   order is real time. This dataset was named in the meeting. It is not downloaded yet.

### Online stream

Shuffle the corpus (fixed seed) and cut it into T batches (T = 5 for the pilot, T = 10 later). The
system reads one batch at a time. The relation names that are most frequent so far go into the next
prompt. After the last batch, we answer the benchmark questions. For C2 and the abstention test, we also
answer after each batch. A question whose gold passages have not arrived yet should get "insufficient
evidence" (figure panel c).

### Systems

| System | What it is |
|---|---|
| closed_book | Reader only, no evidence. Measures memorization. |
| dense | Top-5 passages by embedding similarity. |
| oracle | Gold supporting passages. Upper bound. |
| graph_openie | Graph from plain triple extraction (no questions). Ablation for C1. |
| graph_qbranch_noq | QBranch graph without the inferred-question index. |
| graph_qbranch | QBranch graph + inferred-question index (full v0). |
| HippoRAG 2, LightRAG, RAPTOR | Published numbers first; rerun later with the same reader. |

### Metrics

- EM and F1 (HotpotQA normalization), judge accuracy, passage Recall@5.
- Build cost (tokens, USD) and reader tokens per question.
- C2: the fraction of open slots that a later batch binds, and how many batches later. Compare with a
  chance baseline (random (entity, relation) pairs of the same types).
- Anticipation: for each user question, does an inferred question match it (embedding similarity, then
  an LLM check on a sample)?
- Abstention: precision and recall of "insufficient evidence" on questions whose evidence is incomplete.

## 4. Risks found in the pilot (so far)

1. **Memorization.** Closed-book gets EM 0.45, judge 0.72 on 100 HotpotQA questions. HotpotQA alone
   cannot show the value of retrieval. Use MuSiQue and FinanceBench, and report closed-book always.
2. **Dense retrieval almost saturates HotpotQA.** Recall@5 is 0.97 on the 994-passage slice. The full
   9,811-passage corpus is harder, but the room is small. The main claims must come from MuSiQue,
   2Wiki, the online setting and cost, not from HotpotQA accuracy.
3. **Schema explosion.** 1,220 distinct relation names after 994 passages (OpenIE variant). Without
   normalization, open slots cannot bind (C2), and views cannot form (C3).
4. **Novelty.** See `notes/2026-11-09-related-work-scan.md`. The architecture (a), the judge (c) and
   the "KG as world model" title are not new (Zep, AriGraph, RAGChecker). The new, testable parts are:
   open slots that persist over a stream and bind later (C2), and view management driven by the
   system's own questions with a no-regression gate at a fixed budget (C3). The closest prior work for
   open slots is HopRAG (ACL 2025 Findings). For views, it is StructRAG (ICLR 2025). Make C2 a main result.
5. **C2 needs a stream where entities come back.** On the HotpotQA slice, only 11.5% of the slot
   entities get any later fact. Add a dataset with recurring entities (FinanceBench, dated news).

### Changes to the protocol from the scan

- Baselines: add HopRAG and PropRAG or HGRAG next to HippoRAG 2. In the HippoRAG 2 setting, the
  published F1 (MuSiQue / 2Wiki / HotpotQA) is 48.6 / 71.0 / 75.5 for HippoRAG 2 and 53.9 / 75.3 / 76.1
  for PropRAG.
- Ablation: remove the open slots (keep only the answered questions). This separates the new part of
  (b) from question indexing (doc2query, PIKE-RAG).
- View management: compare with a StructRAG-style router and with "always rebuild", at the same token budget.

## 5. Venues and deadlines (checked on 2026-11-09)

| Venue | Deadline | Format | Notes |
|---|---|---|---|
| ICWM 2027 (icwm.cc) | Paper: 2026-11-30 (marked "provisional") | Not stated | Open review on OpenReview. Reviews 2027-01-11, decisions 2027-02-08. Late May 2027, SF Bay Area. Archival status and dual-submission rule not stated. |
| ICAPS 2027 | Abstract: 2026-12-07; paper: 2026-12-14 (UTC-12) | 8 pages + refs, AAAI-27 kit, double-blind | No arXiv from 2 weeks before the deadline to the notification (2027-02-26). Papers "already under review" elsewhere are not permitted. Special track "Planning Under a Different Name" lists "learning world models". |

The review periods overlap, so one paper cannot go to both. The team must choose this week.

- **ICWM** fits a fixed (predetermined) plan. The deadline is in 3 weeks. The venue is new, and the
  format is not published yet.
- **ICAPS** needs a real planning formulation: view management and budget allocation as a sequential
  decision problem, compared with heuristic policies. The deadline is in 5 weeks.

## 6. Timeline (if ICWM, 2026-11-30)

| Dates | Work |
|---|---|
| Nov 9–12 | Pilot on 100 HotpotQA and 100 MuSiQue questions. Schema normalization. C2 measurement. |
| Nov 13–19 | Full runs on 3 datasets (1,000 questions each). View management v1 (fixed policy). Baselines. |
| Nov 20–26 | Online experiment, ablations, FinanceBench pilot. Write the paper. |
| Nov 27–30 | Internal review and submission. |

## 7. View management v1: a proposal (figure panel d)

This is a proposal for discussion. It is not implemented yet.

1. **A view** is a relation path pattern P over entity types, for example
   `film -starring-> person -born_in-> place`. The system materializes P as one relation `r_P`. Each
   path instance in the graph gives one view fact (s, r_P, o). Its provenance is the union of the
   passages of the path facts.
2. **Candidate views** come from the agenda. When an open slot (e, r2, ?) sits on an entity e that is
   the object of a bound fact (x, r1, e), the path r1 → r2 is a candidate. The most frequent paths in
   the agenda become candidates (figure panel b, Q4).
3. **Self-supervised test set.** For each candidate, compose questions from bound path instances. The
   reference answer comes from the graph path, so the test needs no task labels. Split the questions
   into "earlier" (from earlier batches) and "new".
4. **Compare three options at the same reader token budget:** keep the unified graph, merge
   (normalize the relations of the path), or add the view and route the questions that match P to it.
5. **Gate.** Keep a view only if the gain on the "new" questions is at least δ, and the loss on the
   "earlier" questions is at most ε (non-inferiority), and the upkeep cost is in the budget.
6. **Routing.** An unseen question goes to the view whose pattern is nearest (embedding of the
   pattern text), else to the unified graph. This is the one-step plan from the meeting.
7. **ICAPS form.** State = graph + agenda + views + budget left. Actions = keep, merge, create view,
   infer more questions for a region, bind a slot. Reward = gain on the inferred questions minus a
   regression penalty. Policies to compare: greedy, a UCB bandit over candidate views, and MCTS with
   rollouts on the inferred questions.

## 8. Questions for Prof. Roy

1. Which venue? ICWM (Nov 30) or ICAPS (Dec 14)?
2. Is the main metric accuracy on the real questions, or accuracy at a fixed build budget?
3. What is a "view" in the first version: a sub-graph for a relation path (for example
   book → author → birthplace), or a materialized relation (book → author's birth country)?
4. May the system use the benchmark questions as "user queries" during the stream (feedback), or must
   it only use its own inferred questions?
5. FinanceBench filings are long (100+ pages). Is a subset of companies enough?
