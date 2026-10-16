# Paper 1: results of the overnight runs (2026-10-16)

The draft paper is `paper1/main.pdf` (ACL format, anonymous review mode). The main text has 8 pages. The limitations, the ethics statement, the references and the appendix follow.

Build:

```
for s in e2_calibration e2_metric_replication e3_robustness e5_training e6_cascade e7_versions e8_latency s2_details; do uv run python analysis/$s.py > results/$s.md; done
uv run python analysis/e2_reliability.py
uv run python analysis/make_paper_assets.py     # tables, figures and numbers.tex from results/*.csv
cd paper1 && latexmk -pdf -outdir=tmp main.tex && cp tmp/main.pdf main.pdf
```

All numbers in the text come from `paper1/numbers.tex` (macros). Do not type numbers into `main.tex` by hand: change the analysis, then run the commands above.

**Added on 2026-10-16 (afternoon, exploratory):** the direction of the order flips, the name test with the true rubric second, the image test and a length probe for Clef. Status, results and the extra build commands: `notes/2026-10-16-extensions-log.md`.

---

## 1. What ran

| Run | What | Calls | Cost | Folder |
|---|---|---:|---:|---|
| E2 | 19 System One models x 8 tasks x 500 items | 69,529 | $3.97 | `runs/e2/` |
| B1 | 5 System Two models, structured outputs, reasoning low, the same items | 20,000 | $37.24 | `runs/b1/` |
| B3 | GPT-6 Luna without reasoning, the same items | 4,000 | $0.49 | `runs/b3-none/` |
| B2 | Letter readout of Qwen3.5-9B and Qwen3.8-27B (OpenRouter, logprobs) | 7,002 items | $3.81 | `runs/b2/` |
| B2 local | Letter readout of Qwen3.5-4B with MLX on this Mac | 3,500 items | $0 | `runs/b2-local/` |
| E3 | 16 System One models x 8 tasks x 200 items x 12 perturbations and 4 repeats | 266,400 | $16.09 | `runs/e3/` |
| E3 names | The yes/no name swap of Rafe and Das, 14 models x 3 predicate tasks | 16,800 | $0.48 | `runs/e3-names/` |
| E7 later | Jev 1.13 and 1.14 on the E7 pilot items, 141 minutes after the pilot | 2,000 | $0.08 | `runs/e7-later-2026-10-16/` |
| E7 later 2 | The same, 5.3 hours after the pilot | 2,000 | $0.08 | `runs/e7-later2-2026-10-16/` |
| **All runs** | | **395,131 records** | **$62.43** | |

---

## 2. Main findings (after the internal review)

Table 3 of the paper gives the status of each hypothesis.

1. **H1, partly supported.** The task type changes the calibration most. All 17 System One models with judging tasks are overconfident on them. On classification, 13 of 19 are slightly overconfident (we predicted calibrated or underconfident). The metric choice reproduces each published verdict: a TypedBench-style temperature refit gives underconfidence for Jev on BoolQ (T = 0.85) and overconfidence on the other 7 tasks.
2. **H2, partly supported.** The perturbation explains 57% of the variance of the robustness table, the system 7%, and their interaction 35%. The name test below is exploratory (added after the main runs).
   - Name swap (the protocol of Rafe and Das): Jev follows the names in 98.1% of the answers (they report 50.5 on one set, 97.2 and 84.0 on two others). Liquid d1: 0.3%.
   - Negation: 3.9% for Jev; more than 20% for 11 of 16 systems.
   - Reviewer note against instruction: Jev follows the note (26.2%) but not the instruction (5.3%). Microsoft-Decision-1 follows the instruction (47.6%) but not the note (1.0%).
3. **Noise floor.** 9 of 16 systems flip some answers on identical requests (up to 3.1%).
4. **H6.** Not supported for Jev. The three Clef models flip 12% to 18% of the answers when a question is alone in the call; 5 more systems show small effects (exploratory).
5. **H4, not testable** with three vendor pairs. Decision training adds 5.0 (4B), 7.5 (9B) and 8.0 (27B) points over the same open backbone, so the gains rise with size. One temperature gives most of the calibration gain. On the fresh arXiv task, the gain is small or negative.
6. **H5, mostly not supported.** The cascade keeps the System Two accuracy at less than half the cost in 19 of 89 settings, 17 of them on When2Call, arXiv and BANKING77. On Civil Comments, most of the System Two advantage comes from one attribute without positive items (severe toxicity).
7. **H7, supported, exploratory.** Jev 1.13 to 1.14 flips 6.1% of the answers in E2, 2.7 (2.3 to 3.3) times the noise floor of 2.3%, and 6.1% to 6.9% in all 4 runs. The same version flips 1.8% to 2.5% at gaps from 24 minutes to 5.3 hours, so the floor does not grow with time.

---

## 3. What the internal reviews changed

The first review (`notes/2026-10-16-internal-review.md`) had 28 MUST findings. The second review (`notes/2026-10-16-internal-review-2.md`) found 20 fixed, 8 partly fixed and 7 new MUST items. I fixed all of them in the text or in the analysis. The most important changes:

- The old "next day" rows of the version table were wrong: those calls were 35 minutes apart. I ran a new check 141 minutes after the pilot, and the table now gives the real gaps.
- H1 and H2 are now "partly supported", with a hypothesis table in the main text.
- Group calibration values are now means over tasks (each task counts once).
- Majority baselines are in the paper. On JudgeBench-Pro, all 17 System One models are below "always A" (56.0%).
- The severe-toxicity attribute of Civil Comments has no positive item; the text gives the numbers with and without it.
- The cascade rule (item level) is described, trivial settings are excluded, and the AUROC of the confidence is reported.
- Bootstrap intervals for E5 and E7, matched noise floors in the robustness table (grey cells), and a temperature-scaled ECE for the backbone pairs.
- All comparisons with other papers now use their exact protocols (appendix table).
- The deviations list in Appendix A is complete.
- The second review found a wrong explanation of the Rafe and Das numbers (now removed), a wrong ceiling claim for the 4B backbone, a wrong count of arXiv papers for each category, and an AUROC range that left out HelpSteer2. All are fixed.
- Open SHOULD items that I did not do: the terms-of-service check, intervals for the cascade results, and merging the short appendix sections.

---

## 4. What you must check before submission

1. **Read the paraphrases and the negations** in `src/perturb.py` (Appendix E of the paper). I wrote them. The paper says "we wrote", so you must agree with each one.
2. **Terms of service.** Check that each vendor permits the publication of benchmark results. The ethics statement says that we will check this.
3. **Prof. Roy.** Ask him to be a co-author and the qualified reviewer for ARR.
4. **Hu et al., Lan et al. and Zhang et al.** are cited only for their topics. Open them once before submission.
5. **Back up `runs/`** (1.6 GB, git ignores it). The Jev 1.13 data cannot be made again after TypeSafe removes 1.13.
6. **Rotate the OpenRouter key** (it was printed partly on 2026-10-15).

---

## 5. Limits of these results

- One route (OpenRouter), one region (Bangalore), one 2-hour window at night.
- 500 items for each task, 200 for each perturbation. No correction for multiple tests.
- English, text only.
- The System Two prompts are simple, and the System Two models have no robustness runs.
- The backbone readout uses letters.
- Latency includes the network.
