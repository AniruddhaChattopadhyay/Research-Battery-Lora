# Probe results (2026-10-17)

- Predictions, written before the runs: `notes/2026-10-16-probe-predictions.md`
- Code: `src/probe_data.py`, `src/probe_run.py`, `analysis/probe_report.py`
- Raw records: `runs/probe/`
- Numbers: `results/probe_summary.json`
- Figures: `results/probe_fig1_designqa.png`, `results/probe_fig2_spread.png`, `results/probe_fig3_labels.png`

## 1. Summary

- On two public datasets, PageIndex was as accurate as full context. This was true also for the question types where we predicted failure.
- So our failure model is wrong for single documents of about 150 pages. The predictions file stated this rule before the runs.
- PageIndex did not fail, because it read widely. On DesignQA Compilation, it read a median of 19 pages, including 80% of the pages with gold rules. It used 2.1 times the tokens of the whole document.
- The model needs one change. Spread alone does not cause errors. Errors occur when the evidence is out of the agent's reach. Two conditions cause this: dense sections whose labels drop the facts (our CLEVRER pilot), and corpora that are too large to read (the RIT-RAG result).
- The cost penalty is real. On the rules PDF, a PageIndex question cost 5 to 7 times as much as a full-context question with a prompt cache.

## 2. What we ran

| | DesignQA | LongMemEval S |
|---|---|---|
| Document | One PDF of Formula SAE rules: 140 pages, 588 bookmarks, about 81,000 tokens | One PDF for each question: 41 to 53 dated chat sessions, about 165 pages and 115,000 tokens |
| Questions | 20 Retrieval (the text of one rule, by its number); 30 Compilation (all rules about a topic) | 3 of each type: single-session-user, multi-session, knowledge-update, temporal-reasoning, abstention |
| Metric | The official F1 (no judge) | The official judge prompts, with `gpt-4o-2024-08-06` |
| Chat model | `gpt-5.6-sol` | `gpt-5.6-luna`, for all three systems (to save budget) |
| Systems | PageIndex 0.4.1 and full context | PageIndex 0.4.1, full context and the keyword agent |

- PageIndex: local mode, a Flash index with `gpt-5.6-luna` summaries, at most 16 turns.
- Full context: the text of the whole PDF in one prompt.
- Keyword agent: BM25 search over the pages and page reading, at most 16 turns.
- All systems got the same instructions and reasoning effort "medium". The budget stopped the keyword agent before it ran on DesignQA.

## 3. Results

### 3.1 DesignQA (`gpt-5.6-sol`)

| Question type | n | PageIndex F1 | Full-context F1 | Pages that PageIndex read (median) | Share of gold pages that PageIndex read (median) | Input tokens: PageIndex / full context (median) |
|---|---|---|---|---|---|---|
| Retrieval | 20 | 0.976 | 0.975 | 2 | 1.00 | 80,726 / 81,418 |
| Compilation | 30 | 0.679 | 0.673 | 19 | 0.80 | 171,710 / 81,440 |

More detail:

- On Compilation, PageIndex was better on 11 questions, full context on 10, and the two were equal on 9.
- Compilation recall: PageIndex 0.637, full context 0.652. Precision: 0.784 and 0.769.
- The size of the gold list did not change the result. Spearman correlation of F1 with the gold-list size: PageIndex −0.08, full context −0.18.
- Cost for each question:

| Question type | PageIndex | Full context with a prompt cache |
|---|---|---|
| Retrieval | $0.164 | $0.033 |
| Compilation | $0.363 | about $0.05, computed from the tokens |

- The PageIndex tree has 207 nodes. Its labels hold 28,909 words, which is 55% of the words in the PDF.

Both systems reach only about 0.67 on Compilation. The gold list follows a fixed rule (sub-rules and cited rules), and the question does not state this rule. This limit can hide differences in retrieval.

### 3.2 LongMemEval S (`gpt-5.6-luna` for all three systems)

| Question type | PageIndex | Full context | Keyword agent |
|---|---|---|---|
| single-session-user (3) | 3 | 3 | 3 |
| multi-session (3) | 3 | 3 | 3 |
| knowledge-update (3) | 3 | 3 | 2 |
| temporal-reasoning (3) | 2 | 1 | 2 |
| abstention (3) | 2 | 2 | 2 |
| **Correct, of 15** | **13** | **12** | **12** |
| Input tokens for each question (median) | 42,327 | 118,403 | 4,672 |
| Cost for each question | $0.0062 | $0.0237 | $0.0013 |

- PageIndex read all the answer sessions in 10 of 15 questions. For the three knowledge-update questions, it read only the later session, and that session holds the latest value.
- All three systems missed the same temporal question. PageIndex had read both of its answer sessions.
- The labels name the key. Example: the question "What size is my new Samsung TV?" has the answer "55-inch". The label of the answer session says: "The user describes organizing cables on a Samsung 55-inch 4K TV". So the label holds the key and also the answer.
- The labels of a history hold a median of 11% of its words.

### 3.3 The predictions

| # | Prediction | Result | Evidence |
|---|---|---|---|
| P1 | Retrieval: PageIndex within 0.05 F1 of full context | Held | 0.976 against 0.975 |
| P2 | Compilation: PageIndex at least 0.10 F1 below full context | **False** | 0.679 against 0.673 |
| P3 | Compilation: PageIndex F1 falls as the gold list grows | Held as written, but the effect is near zero | Spearman −0.08 (full context −0.18) |
| P4 | Compilation: PageIndex reads less than half of the gold pages | **False** | Median 0.80 |
| P5 | Compilation: the keyword agent beats PageIndex | Not tested | Budget |
| P6 | Single-session-user: PageIndex at most one error more than full context | Held | 0 and 0 errors (n = 3) |
| P7 | Multi-session, update and temporal: PageIndex wrong more often than full context | **False** | 1 against 2 errors (n = 9) |
| P8 | Most of those PageIndex errors: it did not read all the answer sessions | Not supported | The only error had read both answer sessions |
| P9 | Abstention: PageIndex reads more than on single-session-user | Held | 11.0 against 5.3 pages |
| P10 | The DesignQA labels hold more than 15% of the PDF words | Held | 55% |

## 4. What this means

### 4.1 The model is wrong for single documents of this size

- **A2** says that the evidence is in one or two sections. It was false for Compilation and multi-session questions, but PageIndex did not lose accuracy. It read most of the spread.
- **A1** says that the key is in the labels. It was true on both datasets:
  - A LongMemEval session is about one topic, so a 150-word summary names its main facts.
  - The DesignQA labels hold more than half of the words of the PDF.

### 4.2 A revised model

PageIndex loses accuracy only when the evidence is out of its reach. Two conditions cause this:

1. **Dense sections.** The labels cannot name the facts.
   - In the CLEVRER pilot, the labels held only 6.4% of the distinct numbers of a run.
   - PageIndex got 87.5%, and full context got 95.2%.
2. **Large corpora.** The agent cannot read the spread, or the tree does not fit in the context.
   - RIT-RAG measured 66.7–75.3% on FinanceBench with all 364 filings in the corpus.
   - PageIndex could not run on 2.84 million pages.

In the other cases, PageIndex pays for accuracy with tokens. It reads more, and its cost grows.

### 4.3 What the probe does not show

- The samples are small: 15 LongMemEval questions and one run for each question.
- PageIndex results vary from run to run. One Compilation question got F1 0.11 in a check run and 0.69 in the main run.
- LongMemEval used `gpt-5.6-luna`, and DesignQA used `gpt-5.6-sol`.
- The keyword agent ran only on LongMemEval.
- The Compilation limit (about 0.67) can hide differences.

## 5. Next tests (a proposal for Monday)

1. **Dense records.** Render the AssetOpsBench work orders and sensor tables as a maintenance log. Ask count, latest-value and absence questions with exact answers. Prediction: PageIndex loses accuracy against full context, as in the pilot.
2. **Scale.** Ask the same LongMemEval questions on the M variant: about 500 sessions and 1.5 million tokens for each question. Full context does not fit in the context window there. Prediction: PageIndex loses accuracy and its cost grows, but the keyword agent and the graph method keep their accuracy.
3. **Baseline.** Keep the keyword agent as a main baseline. On LongMemEval, it matched PageIndex at one fifth of the cost.

Estimated cost of tests 1 and 2 with `gpt-5.6-luna`, on 15 to 30 questions each: about $10–15.

## 6. Cost of this probe

| Item | Estimated cost (from token counts) |
|---|---|
| Check runs | $1.80 |
| PageIndex, DesignQA (`gpt-5.6-sol`) | $14.18 |
| Full context, DesignQA (`gpt-5.6-sol`) | $14.46 |
| PageIndex indexes, LongMemEval (`gpt-5.6-luna`) | $1.02 |
| Three systems, LongMemEval (`gpt-5.6-luna`) | $0.47 |
| Judge (`gpt-4o-2024-08-06`) | $0.02 |
| **Total** | **$31.94** |

- The full-context runs on DesignQA cost about 10 times more than necessary. Parallel calls with the same document did not use the prompt cache. Sequential calls cost $0.033 instead of $0.33.
- The OpenRouter key is shared with other projects. Other work used about $83 of it between 2026-10-15 and 2026-10-17. A separate key for this project would make costs clear.

## 7. How to reproduce

```bash
PYTHONPATH=src uv run python src/probe_data.py runs/probe/items.jsonl      # items and LongMemEval PDFs
PYTHONPATH=src uv run python src/probe_run.py index designqa
PYTHONPATH=src uv run python src/probe_run.py pageindex designqa
PROBE_WORKERS=1 PYTHONPATH=src uv run python src/probe_run.py full designqa   # sequential: uses the prompt cache
PYTHONPATH=src uv run python src/probe_run.py index longmemeval 3
for s in pageindex full keyword; do
  PROBE_CHAT_MODEL=openrouter/openai/gpt-5.6-luna PYTHONPATH=src uv run python src/probe_run.py $s longmemeval 3
done
PYTHONPATH=src uv run python src/probe_run.py judge
uv run python analysis/probe_report.py
```
