# Overnight log (2026-10-16)

Aniruddha asked for the experiments and a paper draft by the morning. OpenRouter has auto-reload, so cost is not a limit. Do not commit. Never print the API key.

## Status

| Step | State | Where |
|---|---|---|
| Interface limits in code (`accepts`) | Done | `src/openrouter.py` |
| 4 new tasks: when2call, aegis, judgebench (JudgeBench-Pro), arxiv (fresh) | Done | `src/tasks.py`, `src/fetch_arxiv.py`, `fixtures/` |
| E2: 19 models x 8 tasks x 500 items | Done | `runs/e2/` |
| B1: 5 System Two models, structured outputs | Done | `runs/b1/`, `src/llm.py` |
| B3: GPT-6 Luna without reasoning | Done | `runs/b3-none/` |
| B2: Qwen3.5-9B and Qwen3.8-27B letter readout | Done | `runs/b2/` |
| B2 local: Qwen3.5-4B with MLX | Done at 03:48 (3,500 items; one restart after the 2-hour limit) | `runs/b2-local/`, `src/local_readout.py` |
| E3: 12 perturbations and 4 repeats, 16 models | Done | `runs/e3/`, `runs/e3-names/`, `src/perturb.py` |
| Related-work facts (subagent, full texts) | Done | `notes/2026-10-16-related-work-facts.md` |
| Analysis E2, E3, E5, E6, E7, E8 | Done (E5 4B updates when the local run ends) | `analysis/`, `results/` |
| Paper (ACL format) | Draft done, 8 main pages | `paper1/main.pdf` |
| Internal review (subagent as ARR reviewer): 28 MUST, 54 SHOULD, 15 COULD | Done; all MUST fixed (02:00 to 02:40 IST) | `notes/2026-10-16-internal-review.md` |
| E7 later run (141 min after the pilot) | Done | `runs/e7-later-2026-10-16/` |
| E7 second later run (5.3 h after the pilot) | Done at 04:57 IST | `runs/e7-later2-2026-10-16/` |

## Decisions

- The judging task is JudgeBench-Pro (2026, Apache-2.0), not RewardBench 2. It is adversarial by design (BiasScope).
- Aegis 2.0: prompts only, human prompt labels, one predicate.
- When2Call: the MCQ test file. "direct" is an option but never correct.
- arXiv: q-bio.NC has only 11 papers in the window; the other categories have 32 to 40.
- System Two prompts: JSON schema with one probability for each option. For more than 26 options, a list of likely options (Claude rejects a strict schema with 77 properties).
- Backbone readout: option letters, top-20 logprobs, reasoning off, temperature 0, one pinned provider (Parasail bf16 for 9B, Cerebras fp16 for 27B).
- Gemma 4 31B readout dropped: the Novita provider returned no logprobs for many calls.
- After the related-work facts showed that Rafe and Das swap option names (not a negation), we added their name-swap test (`namealign`, `nameswap`).
- GPT-6 Luna without reasoning was added after the first E5 result (exploratory, reported in the appendix).

## Problems and fixes

- The rtk hook rewrites `grep`, `cat`, `wc` and `tail` in pipelines. Use Python for filters and checks. One `grep -v` made a wrong model list; those calls returned 400 and cost nothing; I deleted them.
- Claude, Haiku and Gemini runs stopped once on a response with empty content. `src/llm.py` now records it as `parse_error`.
- Some items always fail: the OpenAI Decisions API refuses some questions; Tev1-4B sometimes writes text.

## Costs

$62.43 for all runs (395,131 records). System One calls: $20.86 (360,549 calls).

## Changes after the internal review (02:00 to 02:40 IST)

- New analyses (offline): task-mean group calibration with bootstrap; majority baselines; Civil Comments positive rates and a sensitivity check without severe toxicity; variance splits for E2 and E3; matched floors and intervals for all E3 cells; E5 bootstrap gains and cross-fitted temperature scaling; cascade AUROC and a question-level check; System Two reasoning tokens and failures; Jev under the metrics of three studies (`analysis/e2_metric_replication.py`); E7 with real time gaps and bootstrap intervals.
- New run: E7 later run, 141 minutes after the pilot ($0.08).
- Text: hypothesis table; H1 and H2 "partly supported"; H4 "not testable"; H6 "not for Jev"; E7 "exploratory"; exact protocols of the other papers; complete deviations list; STE fixes; new appendix tables (comparison with published numbers, metric replication, System Two details).
- Layout: the cascade and version tables moved to the appendix to keep 8 main pages.

## Second review and final fixes (02:56 to 03:55 IST)

- Second review (`notes/2026-10-16-internal-review-2.md`): 20 of 28 MUST fixed, 8 partly, 7 new MUST. All fixed: the 8-page limit, the arXiv counts, the 4B ceiling claim, the Civil Comments cascade explanation (severe toxicity), the AUROC range, finding 1, and the Rafe and Das explanation.
- New analysis: the Civil Comments cascade without severe toxicity (`results/e6_civil_wo.csv`), an interval for the 2.7-times ratio (`results/e7_ratio.csv`), the System Two failure rule of E2 in `analysis/s2_details.py`.
- All analyses ran again with the complete 4B data at 03:50. The main text ends on page 8.

## End of the night (05:00 IST)

- Second E7 later run at 04:57 IST: the same version flips 1.8% to 2.5% at gaps up to 5.3 hours; 1.13 against 1.14 flips 6.1% to 6.9% in all 4 runs.
- Final build: `paper1/main.pdf`, 20 pages, main text on 8 pages, no LaTeX errors, no undefined references, no overfull boxes.
- Nothing is committed. All new files are untracked (git ignores `runs/`, `data/` and `paper1/tmp/`).
