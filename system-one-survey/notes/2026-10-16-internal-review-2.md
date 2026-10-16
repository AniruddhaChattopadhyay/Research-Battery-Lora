# Internal review 2 of paper 1 (strict ARR reviewer, second pass), 2026-10-16

Paper: "Same Question, Different Answer? A Controlled Study of System One Decision Models" (`paper1/`).
First review: `notes/2026-10-16-internal-review.md`.

## 0. Scope, version and method

Reviewed version:

- `paper1/main.tex`, MD5 `7e30956cbc9e72eca57391b7dbd45a21`, saved 02:20:07 IST.
- `paper1/numbers.tex`, MD5 `557d06875d7a4c8b186ab3c34fc5c473`, saved 02:20:37 IST.
- `paper1/main.pdf`, MD5 `e59b2b93e0680942aea6f67aec12aee9`, built 02:18:29 IST. This PDF is older than `main.tex` and `numbers.tex`.

I built a copy of the current source in `/tmp/rev2_build/`, outside the repository. The copy has 19 pages, the same as `main.pdf`. The two PDFs differ only in Section 4.6 (l. 514–519) and in Table 14 (row labels and caption). "p. 8, l. 471" means page 8, ARR line 471 of the current build. Page and line numbers are the same in `main.pdf`, except l. 514–519.

The 4B pair (`runs/b2-local`) is still running. Its last record is from 03:10 IST. The overnight log expects 500 items at about 03:35 IST. I do not report 4B values as errors. Section 3.5 lists the text claims about 4B that the final data can contradict.

What I did:

1. I gave a status to each of the 28 MUST findings of the first review.
2. I compared each literal number in the main text with `numbers.tex`, the tables and `results/*.csv`. I also compared 21 E3 macros directly with `results/e3_summary.csv`. All 21 agree.
3. I read the changed parts of `src/` and `analysis/` again, and I compared them with the method text.
4. I compared each statement about another paper with `notes/2026-10-16-related-work-facts.md` and `notes/2026-10-15-model-landscape.md`.
5. I compared Table 3 with section 2 of `notes/2026-10-15-experiment-design.md` and with the results.
6. I read all 19 pages of the PDF.
7. I read the commit time from the file `.git/logs/HEAD`. This is a file read. I did not run a git command.

I did not edit other files, and I did not call an API. My scratch scripts are in `/tmp/rev2_scripts/`. Appendix Z gives the numbers that I computed.

Labels (as in the first review):

- **MUST**: the text is wrong or misleading. Fix it before submission.
- **SHOULD**: the problem weakens the paper. An ARR reviewer will probably raise it.
- **COULD**: polish.

New findings continue the numbering of the first review: N33 and up, R14 and up, M17 and up, L10 and up, S16 and up. Findings about Table 3 have the prefix T.

---

## Summary

The revision is a large improvement. Of the 28 MUST findings, 20 are fixed and 8 are partly fixed. No finding is fully open. The hypothesis table, the majority baselines, the matched floors, the bootstrap intervals and the deviations appendix answer the main concerns of the first review.

But the revision added new errors, and some fixes are not complete.

New MUST findings (7):

1. **L10.** The main text ends on page 9, not page 8. The last line of Recommendation 6 is on p. 9, l. 544.
2. **N33.** "about 31 for each category" is wrong. The 500 arXiv items have 32 or 33 papers for each category (11 for q-bio.NC).
3. **N34.** "the backbones already reach 87% to 89% on arXiv" is wrong for the 4B backbone (79.7%, Table 12).
4. **N35.** Section 4.5 explains the Civil Comments escalation by a 5.7-point gap. Most of this gap comes from `severe_toxicity`, which has no positive item. Without it, the median gap is 1.0 point and the median escalation is 30%, not 79%.
5. **N36.** "separates right from wrong answers on most tasks (AUROC 0.67 to 0.90), but not on JudgeBench-Pro" leaves out HelpSteer2 (AUROC 0.54 to 0.67).
6. **N37.** Finding 1 says "Under the metric of each published study, our data give the verdict of that study". This is true only for some tasks.
7. **R14.** The new explanation of the Rafe and Das values disagrees with the facts file. Their typed-decisions rubrics also do not name a key.

MUST findings of the first review that are only partly fixed (8): N3, N5, N32, R1, W1, W2, W3, S2 (section 1).

---

## 1. Status of the 28 MUST findings of the first review

| ID | Status | Evidence |
|---|---|---|
| N1 | FIXED | Table 14 rows "1.14, 35 min apart" and "1.14, 2.4 h apart"; l. 513–515 "at gaps from 24 minutes to 2.4 hours" (`results/e7_versions.csv`, `gap_min` 35 and 141). |
| N2 | FIXED | Table 14 row "1.14, 24 min apart" (the median gap between the E2 call and the E3 repeats). |
| N3 | PARTLY | The 4B run is not finished. Table 4 uses 3,259 questions for Qwen3.5-4B (`e2_metrics.csv`, 02:00 IST), and Table 6 uses 3,342 (`e5_training.md`, 02:04 IST). When the run ends, run `e2_calibration.py`, `e5_training.py` and `make_paper_assets.py` again, then check section 3.5. |
| N4 | FIXED | l. 333 "Thus, H1 is partly supported", and Table 3 row H1 gives each part. |
| N5 | PARTLY | Table 2 "Maj.", Table 12 row "Majority answer (B7)", the bound on l. 268–269 and l. 305–307 ("all 17 ... below the majority answer (56.0%)") are in. Still open (problem 3): the abstract (l. 017–019) and l. 316–318 say "overconfident on them", but on HelpSteer2 alone Liquid d1 (−2.3) and Kev-4B (−2.6) are underconfident (Figure 1). Write "on the judging group". |
| N7 | FIXED | l. 383–385 "the size of the effect depends mostly on the perturbation"; variance split 57/7/35; new title of Section 4.2. |
| N8 | FIXED | "confidently wrong" is gone; l. 497–504 give the AUROC values. New problems in the same paragraph: N35 and N36. |
| N9 | FIXED | l. 489–496: 69, 10, 89, 69 and 19 settings; "17 of these 19 are on When2Call, arXiv and BANKING77" (`e6_cascade.csv` agrees). |
| N10 | FIXED | l. 474–477 "a similar mean accuracy (70.3% and 71.3%), but they differ by up to 8.3 points on single tasks, in both directions (exploratory)". |
| N11 | FIXED | l. 310–314 give GPT-6 Luna with and without reasoning, and the Claude Sonnet 5.5 split; Table 17 gives the reasoning tokens. New inconsistency: N38. |
| N12 | FIXED | l. 375 "1.9 to 3.7 seconds". |
| N13 | FIXED | l. 432 "9 of 16 systems follow the note more often than the instruction" (`nHintAboveInject`). |
| N14 | FIXED | l. 396–397 "14.1% to 19.6%". |
| N15 | FIXED | l. 198 "Five systems from three vendors limit the interface". |
| N16 | FIXED | l. 465–467 "At 27B, the gains are largest on the judging tasks, and at 9B on When2Call." |
| N17 | FIXED | l. 437 "the published range from 1.3% to 61.4%". |
| N32 | PARTLY | l. 240–244 give the positive rate of each attribute, and l. 356–359 give the Jev values without severe toxicity. Still open: the attribute stays in all Civil Comments numbers (Tables 4, 12, 13, 16, Figure 1). The text does not say that it causes most of the System One against System Two gap on Civil Comments (N35). |
| R1 | PARTLY | l. 398–400 give 50.5, 97.2 and 84.0. But the new explanation of the difference disagrees with the facts file (R14). |
| R2 | FIXED | l. 417–422 give all four pairs and the protocol difference ("one shuffle for each question on other datasets"). Naming problem: N44. |
| R4 | FIXED | l. 249–253 "Its builders added a bias to the incorrect response and kept the items that Qwen3-32B judged wrongly in both orders". |
| M1 | FIXED | l. 479–482 describe the item-level rule and the single System Two call. |
| M7 | FIXED | Table 4 caption "All columns are means over tasks, and each task counts once"; `analysis/e2_calibration.py`, lines 81–110, compute task means. |
| W1 | PARTLY | Majority baselines and the bound are in (see N5). Not done: "within 1 level" accuracy and MAE for HelpSteer2 (`e2_metrics.csv` has them; Appendix Z12), and the removal of `severe_toxicity` or the use of soft labels (N32, N35). |
| W2 | PARTLY | Table 3, Appendix A and the E7 timing note are in. The commit time agrees with "about 10 minutes" (Appendix Z2). Still open: (1) the abstract and finding 2 use the name test, which came after the main runs, as the H2 result without "exploratory" (N45, T2); (2) Appendix A does not list all deviations (section 2, item 3). |
| W3 | PARTLY | The E5 gains, the E7 flips and all E3 cells have bootstrap intervals, and the grey cells use the matched floor. Still open: no interval for the cascade accuracy, the cost ratio or the ratio "2.7 times". The "within 1 point" rule of E6 is smaller than the sampling error of a 500-item paired difference. |
| S2 | PARTLY | The abstract now says "signed calibration error", so the term is fixed. The sentence still has 33 words (S16, item 2). |
| S9 | FIXED | l. 084–085 define "System Two models", and the text uses only this term. |
| L7 | FIXED | Table 6 now has 7 tasks for 4B, the Table 14 labels are correct, and Table 16 has no "–" for 4B. The Table 4 caption "500 items for each task" becomes true for 4B only when the run ends (N3). |

---

## 2. SHOULD findings of the first review that are still open and important

1. **W12 (terms of service).** l. 576–578 still say "Before publication, we will check the terms of service". The plan made this check a gate of E0. Do the check for the 13 vendors, and report the result in the Ethics statement.
2. **W7 and W3 (cascades).** The cascade accuracy and the cost ratio have no interval. The question-level check exists, but the paper gives no result (N39).
3. **W2 (deviations list).** Appendix A does not list these planned items:
   - the two Cloudflare pairs in E5 (Clef-flash with Qwen3.5-9B, Clef with Qwen3.8-27B) and AnyJev;
   - the planned datasets that the paper does not use: ClaimDB, CLINC150, SummEval, MetaTool, LlavaGuard, POPE and the CFPB complaints;
   - the robustness probe sets (ANLI, `deepset/prompt-injections`, NotInject, `llmail-inject-challenge`);
   - the 200-item development split and the sampling stratified by label;
   - the weekly runs of the two "latest" aliases (only one run on 15 October);
   - most planned metrics (macro-F1, AUPRC, quadratic weighted kappa, Brier score, NLL, AURC, risk-coverage curves).
4. **W1, item 3 (HelpSteer2).** The exact-match accuracy is below the majority baseline for 25 of 26 systems. "Within 1 level", the System One models get 76.0% to 94.5% and the System Two models 81.3% to 86.9% (Appendix Z12). The judging conclusions use HelpSteer2, so report this value.
5. **N6 (now N41).** The variance split is in, but the conclusion does not follow it.
6. **W10 (definitions).** l. 271–273 say "A value for a group of tasks is the mean over its tasks". E7 pools the questions (Table 14 caption), and the abstract uses the pooled values. Add "except E7" to the Metrics paragraph. With task means, the ratio is 2.8, not 2.7.
7. **L2 and L3 (layout).** See L12 and L13.

---

## 3. New errors in the revision

### 3.1 Numbers and claims in the text

**N33 [MUST] p. 4, l. 258–259 ("We use 500 papers, about 31 for each category (11 for q-bio.NC)").**

- Problem: The fixture has 603 papers: 11 for q-bio.NC, 32 for one category and 40 for each of 14 categories. The task takes the first 500 items of an order that alternates the categories (`src/tasks.py`, `arxiv()`). These 500 items have 32 papers for 6 categories, 33 for 9 categories and 11 for q-bio.NC (Appendix Z3). The first review suggested "about 31". That suggestion was wrong.
- Fix: Write "32 or 33 for each category (11 for q-bio.NC)".

**N34 [MUST] p. 8, l. 468–471 ("or from a ceiling, because the backbones already reach 87% to 89% on arXiv").**

- Problem: This is true only for the 9B (87.2%) and 27B (88.8%) backbones. The 4B backbone gets 79.7% (Table 12) and 83.0% on the E5 items (`results/e5_training.md`, 147 items). At 4B, Tev1-4B is 4.8 points below its backbone (Table 16), and a ceiling cannot explain a loss. The 4B run is not finished, but a final value of 87% or more is unlikely.
- Fix: Write "the 9B and 27B backbones already reach 87% to 89% on arXiv". Check the 4B value when the run ends.

**N35 [MUST] p. 8, l. 501–504 ("On Civil Comments, the AUROC is 0.76 to 0.90, but the System Two models are more accurate (median gap 5.7 points), so the cascade sends a median of 79% of the items").**

- Problem: Most of this gap comes from `severe_toxicity`, the attribute with no positive item (l. 243–244). The System One models often say "yes" to it (Jev 1.14: 51.2%). I ran the code of `analysis/e6_cascade.py` again, offline, without this attribute (Appendix Z4):
  - median gap (System Two minus System One): 5.7 → 1.0 points;
  - settings where the System Two model is more than 1 point better: 20 of 21 → 10 of 21;
  - median escalation: 79% → 30%.
- For Jev 1.14 against Claude Sonnet 5.5, the Civil Comments gap falls from 12.1 to 5.6 points without the attribute (`results/e2_civil_sensitivity.csv`).
- Fix: Add one sentence: "Most of this gap comes from severe toxicity, which has no positive item. Without it, the median gap is 1.0 point, and the cascade sends a median of 30% of the items." Better: give all Civil Comments results without this attribute, or with the soft labels (first review, N32).

**N36 [MUST] p. 8, l. 497–500 ("separates right from wrong answers on most tasks (AUROC 0.67 to 0.90), but not on JudgeBench-Pro (0.29 to 0.63)").**

- Problem: The macro `aurocOther` leaves out both HelpSteer2 and JudgeBench-Pro (`analysis/make_paper_assets.py`, `numbers_e6`). The text names only JudgeBench-Pro. On HelpSteer2, the AUROC is 0.54 to 0.67 (median 0.62), near the JudgeBench-Pro values (Appendix Z5). A reader concludes that the range 0.67 to 0.90 covers the other 7 tasks.
- Fix: Write "on 6 of 8 tasks (AUROC 0.67 to 0.90), but not on JudgeBench-Pro (0.29 to 0.63) and HelpSteer2 (0.54 to 0.67)".

**N37 [MUST] p. 2, l. 100–102 (finding 1: "Under the metric of each published study, our data give the verdict of that study").**

- Problem: This is true only if we also choose the task. Table 8 shows:
  - TypedBench refit: underconfident only on BoolQ (T = 0.85). On the other 7 tasks, the same metric gives the opposite verdict (T = 1.11 to 17.50).
  - Deußer et al. metric: their pooled choice ECE is 0.028. Our four choice tasks give 4.1 to 19.2 points.
  - The leaderboard says "overconfident on knowledge". Jev 1.14 has an SCE of −1.7 on BoolQ, with an interval that contains 0.
- Section 4.1 says it correctly ("Each comes from a different task", l. 347–348). Only the finding generalizes too much. The first review asked for "can explain part of" (N21). The new text is stronger than the old text.
- Fix: Write "Each published verdict appears in our data, but only on some tasks (Section 4.1)."

**N38 [SHOULD] Table 17 disagrees with Tables 4 and 12 and with l. 311–312.**

- Problem: `analysis/s2_details.py` counts as "Failed" only the calls with a status other than 200. It counts a status-200 answer without a usable distribution as a wrong answer. E2 drops these answers (`analysis/e2_calibration.py`, `usable`). Results (Appendix Z7):
  - "Failed": Claude Sonnet 5.5 50 (E2 drops 51 items), Gemini 3.8 Flash 2 (5), Claude Haiku 5.5 4 (6), GPT-6 Luna 0 (2).
  - "Mean acc., answered": 79.7, 79.4 and 78.9, against 79.8, 79.5 and 79.0 in Table 4.
  - GPT-6 Luna on JudgeBench-Pro: Table 17 gives 83.0 (415 of 500). Table 12 and l. 311 give 83.3% (415 of 498). All its JudgeBench-Pro calls used reasoning, so the two values must be equal.
  - Claude Sonnet 5.5 without reasoning tokens: l. 312 gives 73.0% on 259 calls. One of these calls has no usable answer. With the E2 rule, the value is 73.3% on 258 calls.
- Fix: Use the E2 rule in `s2_details.py`: drop the answers with no distribution, and count their items as failed. Then build Table 17 and the two Sonnet macros again. Also correct the caption: "Failed: items without a usable answer (error, empty content or no valid probabilities)".

**N39 [SHOULD] p. 11, l. 792–793 (Appendix A lists "the question-level cascade check", but the paper gives no result).**

- Data: `results/e6_cascade.csv` has the columns `acc_cascade_q`, `escalated_q` and `cost_ratio_q`. With the question-level rule, the cascade is kept in 71 settings (item level: 69) and cheap in 20 (19). On Civil Comments, the median escalation rises to 90% (item level: 79%). Thus, the item-level rule does not cause the high escalation (Appendix Z6). This result closes N8 of the first review.
- Fix: Add one sentence to Section 4.5, or remove the item from Appendix A.

**N40 [SHOULD] p. 2, l. 108–109 (finding 3: "The Clef models do not isolate the questions of one call, but Jev does") and Table 3, row H6.**

- Problem: In Table 5, five more systems have a Crowd. or Alone flip rate above the matched floor: Mercury Decide and Drex v2.0 in both tests, and Liquid d1, Drex v1.5 and Kev-4B in Alone (Appendix Z10). Their effects are small (0.7% to 3.6%), but they are above the floor. Thus, 8 of 16 systems show interference.
- Fix: Write "The three Clef models change 12% to 18% of the answers when a question is alone in the call. Five more systems show small changes above the floor. Jev shows none."

**N41 [SHOULD] p. 5, l. 329–335 ("Without the judging tasks, the task explains 32% and the system 39%. Thus, H1 is partly supported. The task type matters most ..."), and the abstract (l. 015–017) and finding 1.**

- Problem: The conclusion "The task type matters most" is true only with the judging tasks. Without them, the system explains more of the variance than the task. The abstract and finding 1 give the conclusion without this condition.
- Fix: Write "With the judging tasks, the task type matters most. Without them, the system matters slightly more."

**N42 [SHOULD] p. 5, l. 334–335 ("judging gives overconfidence, as we predicted").**

- Problem: H1 predicted overconfidence on "knowledge and reasoning tasks". The pre-registered grouping (design note, E2, step 1) has no reasoning group. It has knowledge, classification, judging and agent. The paper maps "reasoning" to judging after the runs.
- Fix: Write "judging, our nearest group to reasoning, gives overconfidence". Say the same in Table 3 (T1).

**N44 [SHOULD] p. 7, l. 420 ("GPT-6 Luna 10.9% (13.2%)").**

- Problem: The 10.9% is the macro `orderLuna`, the OpenAI Decisions API, a System One model. Everywhere else, "GPT-6 Luna" is the System Two chat model, which has no E3 runs (l. 560–563). Rule 6: use one term for one thing.
- Fix: Write "OpenAI Decisions (Luna) 10.9% (their 'GPT-6 Luna': 13.2%)".

**N45 [SHOULD] Abstract (l. 021–024 and l. 028–030) and findings 2 and 6 (l. 103–107, l. 114–115).**

- Problem: Two of the six headline results are exploratory by the rules of the paper: the name test (Appendix A: added after the main runs) and E7 (Section 4.6: exploratory). The abstract and the findings do not say this. Finding 2 also links the name test to H2, but H2 did not include it (T2).
- Fix: Add "(exploratory)" to findings 2 and 6, and add the word to the abstract.

**N46 [SHOULD] Abstract (l. 026–027), finding 4 (l. 110) and Table 3, row H4 ("adds 4.2 to 8.0 points").**

- Problem: The text uses the macros `\gainAbsFour` and `\gainAbsTwentySeven` as the ends of a range. This assumes that the 4B gain is the smallest and the 27B gain the largest. The 4B pair is not final. If its gain is above 8.0 or below 0, the text becomes wrong, for example "9.1 to 8.0".
- Fix: Add macros for the minimum and the maximum gain of the three open pairs, and use them. Check the sign.

**N47 [COULD] p. 4, l. 308–310 ("The System Two models are 25 to 38 points above the best System One model on JudgeBench-Pro").** The range is correct for the five configurations with reasoning effort "low" (24.8 to 37.7). GPT-6 Luna without reasoning is 12.0 points below. Write "The five System Two models with low reasoning effort".

**N48 [COULD] p. 9, l. 579–580 ("The 357,000 calls to System One models").** The later E7 run added 2,000 calls. The runs now have about 358,600 System One calls for USD 20.78. The total cost is USD 62.35 (Appendix Z8). Write "about 358,600 calls".

**N49 [COULD] p. 4, l. 254–255 ("603 papers submitted from 10 to 15 October 2026").** The query window is 10 to 15 October, but `src/fetch_arxiv.py` keeps the first 40 papers of each category in date order. The newest paper is from 14 October, 16:49 UTC. Write "submitted from 10 to 14 October 2026".

**N50 [COULD] p. 6, l. 370–373 ("Jev 1.14 ... within 2 points of it").** The mean ECE values are 9.148 (Jev 1.14) and 7.143 (Claude Sonnet 5.5), a gap of 2.005 points. Write "about 2 points".

**N51 [COULD] p. 3, l. 205–206 ("one check run followed at 02:00 IST").** The run was from 01:57 to 01:59 IST. The overnight log plans a second later run at 05:45 IST. If that run happens, change this sentence and the Limitations ("one 2-hour window", l. 546–547). The E7 macros change automatically.

**N52 [COULD] p. 6, l. 360–365 (item 3: "A small ECE can hide a consistent direction").** Our two values (+8.1 with an ECE of 8.3; −0.1 with an ECE of 4.1) do not show this effect. Only the numbers of Deußer et al. show it. Say that the item uses their numbers, or add a pooled value of ours.

**N53 [COULD] Abstract, l. 012–014 ("All systems get the same 8 tasks with 500 items each").** The five limited systems get 3 or 7 tasks, and the three backbones get 7. Write "the same tasks (up to 8, with 500 items each)".

**N54 [COULD] p. 11, l. 787–790.** Appendix A puts "GPT-6 Luna without reasoning" under "Added after the main runs". This run was from 00:48 to 00:59 IST, during the main runs (E3 ended at 01:06 IST). Write "Added during the main runs".

### 3.2 Statements about other papers

**R14 [MUST] p. 6, l. 398–404, and Table 7, row 1, column "Protocol difference".**

- Text: "On those two sets, the rubric text did not name the key, so the name was the only cue to polarity. Our rubrics are also neutral, and our value for Jev is close to their two values." Table 7: "their typed-decisions rubrics carry text; their other two sets and our rubrics do not name the key".
- Facts file, section 2 (quote from their Appendix B): on the D3 sets, the authors removed the leading word that names the key, and "The option name is then the only cue to polarity". For D1 (typed-decisions): "their rubric text unchanged, because their rubrics do not name a key".
- Problems:
  1. The D1 rubrics also do not name a key. Thus, this property does not separate D1 (50.5) from D3 (97.2 and 84.0). The text and Table 7 say that it does.
  2. Our rubrics are "This is true for the input." and "This is not true for the input." They do not contain "yes" or "no", but they state the polarity. Thus, in our test the name is not the only cue to polarity.
  3. 98.1% is close to 97.2, but it is 14 points above 84.0.
- Fix: Write "Rafe and Das report 50.5 flips per hundred on their typed-decisions set, and 97.2 and 84.0 on two other sets. Our 98.1% for Jev 1.14 is near their highest value. The data and the rubric texts differ, so we do not explain the differences." In Table 7, write "other data and rubric texts; our rubrics say 'true' or 'not true'".

**R15 [SHOULD] p. 3, l. 166–169 ("we add a floor for each system, the comparison with the backbone, the interference between questions and version changes"), and Section 4.3.**

- Facts file, section 7: Li et al. (2026a) send each question alone and also all ten questions of an item in one request. The negation violation is 0.064 in both cases. This is an interference test for Jev 1.13, and it agrees with our Jev result.
- Fix: Cite this result in Section 4.3. In Section 2, write "the interference between questions for many systems".

**R16 [SHOULD] p. 7, l. 426–431, and Table 7, row "Xu et al. (2026)".**

- Problem: JevOut uses the most probable wrong option as the target, and its denominator is the initially correct decisions (facts file, section 4). Our note uses a random wrong option (`src/perturb.py`, `_wrong`), and our follow rate uses all answers that were not X before. Our nearest value to their denominator is the "broken" rate: 24.7% for Jev (Appendix Z11).
- Fix: Add the two differences to Table 7, and give 24.7% next to 26.2%.

**R17 [COULD] p. 3, l. 165–166 ("Each study tests one or a few systems").** Huang et al. test 8 decision models, and Rafe and Das test 8 (facts file, sections 2 and 6). Write "at most 8 decision models".

**R18 [COULD] Table 7, row "Deußer et al. (2026)".** Their 37 datasets include Banking77, BoolQ and HelpSteer2 (facts file, section 1). A comparison on the same datasets is closer than the pooled 0.028, if their Table 3 gives the values (the facts file does not). The row also mixes units: 0.028 against "8.3 and 4.1 points". Write "2.8 points".

**R19 [COULD] Reference "Lai et al. (2026)".** The facts file says "Accepted to ICLR 2026". The reference says "Preprint".

### 3.3 Method text against the code

**M17 [COULD] Table 8 caption ("one temperature fitted by NLL on one half of the items and tested on the other half, the geometric mean of the two folds").** `analysis/e2_metric_replication.py` fits one temperature on each half and gives the geometric mean. It does not test on the other half. Write "the geometric mean of the temperatures that NLL gives on each half of the items".

**M18 [COULD] Figure 3.** `analysis/e2_reliability.py` pools all questions of a group. Civil Comments gives 70% of the classification answers, and HelpSteer2 gives 83% of the judging answers. Elsewhere, a group value is a mean over tasks (l. 271–273). Say this in the caption.

**M19 [COULD] Table 11 caption.** The caption says "<q>", but the table shows "{q}".

**M20 [COULD] Table 12 caption ("–: the system does not accept the task").** For the backbones on BANKING77, "–" means that the readout cannot express 77 options. Use the wording of the Figure 1 caption.

I found no other disagreement between the method text and the code. The provider pins, the interface limits, the retries, the negative-value rule, the BANKING77 list form, the top-20 letter readout, the length limits, the hint and injection texts, the name-test rubrics and the item-level cascade rule agree with `src/` and `analysis/`.

### 3.4 Logic

The logic problems are N35 (wrong cause on Civil Comments), N36 (selective range), N37 and N41 (conclusions stronger than the data), N42 (a mapping after the runs) and R14 (an unsupported cause). I found no other logic error.

### 3.5 Text claims about 4B that the final data can contradict

1. N34 [MUST]: "the backbones already reach 87% to 89% on arXiv" (wrong now).
2. N46 [SHOULD]: the range "4.2 to 8.0" in the abstract, finding 4 and Table 3.
3. l. 459 "It also lowers the ECE." (4B now: 14.9 against 18.7). Check it again.
4. l. 461–462 "Thus, one temperature gives most of the raw calibration gain." The text gives 9B and 27B only. Check that 4B agrees.
5. l. 467–468 "There is no clear gain on the fresh arXiv task". At 4B, the value is now −4.8, a loss.
6. Table 4 caption "500 items for each task" and Table 6 caption "on the same items". Now the two tables use different 4B item sets (N3). Run E2, E5 and the assets together after the run.

---

## 4. Hypotheses table (Table 3) against the pre-registration and the results

Source of the pre-registration: `notes/2026-10-15-experiment-design.md`, section 2. The file has the save time 23:41:31 IST, and the commit `3747117` has the time 00:02:26 IST. The first E2 call was at 00:12:06 IST (Appendix Z2). Thus, "written before the main runs" is true. But the E7 pilot ended at 23:38 IST, before the last save of H7. Table 3 marks H7 as exploratory, which agrees.

| H | Pre-registered text | What Table 3 says | What the data show | Finding |
|---|---|---|---|---|
| H1 | "Calibration depends on the task type. System One models are overconfident on knowledge and reasoning tasks, and well calibrated or underconfident on classification tasks." | Same text. "Partly: judging overconfident (17/17); BoolQ mixed; classification mostly slightly overconfident (13/19)". | Judging 17 of 17 overconfident (group mean). BoolQ: 9 over, 8 contain 0, 2 under. Classification: 13 over, 1 contains 0, 5 under. Variance: task 64%, system 14%; without judging, task 32%, system 39%. | **T1 [SHOULD]**. The numbers and "Partly" are correct. But no pre-registered group is "reasoning", and the table maps it to judging without a note (N42). The "task type" part holds only with the judging tasks (N41). |
| H2 | "The size of the wording effect depends on the type of perturbation. Polarity swaps and natural context change many answers. Option order and synonyms change few." | Drops the first sentence. "Partly: order and neutral names change few; negation many for most systems, few for Jev". | The perturbation explains 57% of the variance (supports sentence 1). Negation: more than 20% for 11 of 16, Jev 3.9%. Note (replaces natural context): median follow 17.1%, Jev 26.2%. Order: median 4.6%. Neutral names (replace synonyms): median 5.2%. | **T2 [SHOULD]**. (1) Add sentence 1 and its result. (2) Add the result for the note, and say that it replaces natural context. (3) Say that neutral names replace synonyms. (4) The name test is not part of H2, but the abstract and finding 2 use it as the H2 result (N45). "Partly" is acceptable. |
| H3 | "A System One model is better than a trained encoder only when fewer than about 1,000 labeled examples exist." | "Trained encoders win above 1,000 labels"; "Not tested (paper 2)". | Not run. | **T7 [COULD]**. "paper 2" is an internal plan. Write "Not tested here". |
| H4 | "Decision training adds more to small backbones than to large backbones." | Same text. "Not testable with three vendor pairs; gains of 4.2 to 8.0 points". | Gains 4.2 (4B, partial), 7.5 (9B), 8.0 (27B). The observed order is the opposite of the prediction. | **T6 [COULD]**. "Not testable" is fair. Also say that the observed gains rise with the size. The two Cloudflare pairs of the plan share one vendor; Appendix A does not list them (section 2, item 3). |
| H5 | "A cascade from a System One model to a System Two model keeps the accuracy of the System Two model at less than half of its cost." | Same text. "Partly: 19 of 89 settings". | Kept in 69 of 89 settings, kept and cheap in 19 of 89 (21%). 17 of the 19 are on When2Call, arXiv and BANKING77. | **T3 [SHOULD]**. "Partly" overstates a 21% success. Write "Not supported in general: 19 of 89 settings, 17 of them on three tasks". |
| H6 | "Answers to the same question change when other questions are in the same call, also for systems that claim isolation." | Same text. "Not for Jev; yes for the Clef models (exploratory)". | Jev: no change above the floor. Clef models: 12% to 18% (Alone). Five more systems: small changes above the floor (Appendix Z10). | **T4 [SHOULD]**. Add the five systems (N40). The paper names only Jev as a system that claims isolation, so the key part of H6 fails. Say this in one clause. |
| H7 | "A new version changes a measurable part of the answers, also on easy items." | Drops "also on easy items". "Supported, exploratory: 6.1% vs. 2.3% noise". | Pooled: 6.1% against 2.3%. Easy tasks: BoolQ 2.2% [1.0, 3.6] against 0.1% [0.0, 0.4]; Aegis 2.0 2.2% [1.0, 3.6] against 0.5% [0.1, 1.0] (Appendix Z13). | **T5 [SHOULD]**. Put the clause back, and give the BoolQ and Aegis 2.0 values. They support it. The status is correct. |

Two more points about Table 3:

- **T8 [COULD]** The table has no column for "confirmatory or exploratory". Only H6 and H7 say "exploratory" in the result cell. The first review asked for such a column (W2).
- **T9 [COULD]** The first header cell is empty. Write "H".

---

## 5. ASD-STE100: 10 remaining sentences that break the rules badly

**S16 [SHOULD]** The 10 worst sentences, with rewrites:

1. Abstract, l. 005–008. Rule 3 (passive) and rule 2.
   - Original: "More than 20 such models were released in four weeks, and the first evaluations disagree about their calibration and their sensitivity to wording."
   - Rewrite: "Vendors released more than 20 such models in four weeks. The first evaluations disagree about their calibration and their sensitivity to wording."
2. Abstract, l. 015–020. Rule 1 (33 words).
   - Original: "The task type changes the calibration most: all 17 System One models that accept our judging tasks are overconfident on them, but on BoolQ their median signed calibration error is only +2.0 points."
   - Rewrite: "The task type changes the calibration most. On the judging group, all 17 System One models that accept it are overconfident. But on BoolQ, their median signed calibration error is only +2.0 points."
3. Abstract, l. 008–012. Rule 1 (28 words).
   - Original: "We test 19 System One models from 13 vendors, 6 configurations of System Two models (general language models with structured outputs) and 3 open backbones under one protocol."
   - Rewrite: "We test 19 System One models from 13 vendors under one protocol. We also test 6 configurations of System Two models (general language models with structured outputs) and 3 open backbones."
4. Introduction, l. 091–094. Rule 1 (29 words).
   - Original: "We also send each of these requests 4 more times without a change, because a flip (a change of the top answer) means something only above this noise floor."
   - Rewrite: "We also send each of these requests 4 more times without a change. These repeats give a noise floor. A flip (a change of the top answer) means something only above this floor."
5. Section 3, l. 284–287. Rule 1 (34 words without the citation).
   - Original: "To test the claim of Rafe and Das (2026), we also ask each predicate as a choice between two options with neutral rubrics ("This is true for the input." and "This is not true for the input.")."
   - Rewrite: "To test the claim of Rafe and Das (2026), we also ask each predicate as a choice between two options. The two rubrics are "This is true for the input." and "This is not true for the input."" (Also see R14 for the word "neutral".)
6. Section 4.2, l. 386–390. Rules 1 and 11 (30 words, a list in a sentence).
   - Original: "9 of 16 systems flip some answers between identical requests, for example Solar Decide Flash (3.1%), OpenAI Decisions (Luna) (2.2%), Jev 1.14 (1.9%), Microsoft-Decision-1 (1.8%), Liquid d1 (1.6%), Tev1-4B (1.1%)."
   - Rewrite: "9 of 16 systems flip some answers between identical requests (Table 5, column Floor). The highest floor is 3.1% (Solar Decide Flash), and Jev 1.14 has 1.9%." (This also gives back about 2 lines for L10.)
7. Section 4.4, l. 463–465. Rule 3 (passive "are confounded").
   - Original: "The three pairs come from three vendors with different recipes, so size and recipe are confounded, and the pairs cannot test H4."
   - Rewrite: "The three pairs come from three vendors with different recipes. Thus, the pairs cannot separate the effect of the size from the effect of the recipe, and they cannot test H4."
8. Section 4.5, l. 492–495. Rule 1 (31 words).
   - Original: "In the other 89 settings, the cascade keeps the System Two accuracy within 1 point in 69, and only 19 of them cost less than half (Table 13 in the appendix)."
   - Rewrite: "In the other 89 settings, the cascade keeps the System Two accuracy within 1 point in 69 settings. Only 19 of these cost less than half of the System Two model alone (Table 13)."
9. Section 4.5, l. 501–504. Rule 1 (31 words). The rewrite also fixes N35.
   - Original: "On Civil Comments, the AUROC is 0.76 to 0.90, but the System Two models are more accurate (median gap 5.7 points), so the cascade sends a median of 79% of the items."
   - Rewrite: "On Civil Comments, the AUROC is 0.76 to 0.90. But the System Two models are more accurate there (median gap 5.7 points), so the cascade sends a median of 79% of the items. Most of this gap comes from severe toxicity, which has no positive item."
10. Appendix A, l. 802–809. Rules 1 and 11 (58 words, a list in a sentence).
    - Original: "Planned but not run: trained encoders (E4, H3), the zero-shot encoders, guards and reward models (B4 to B6), the 2B pair and the Gemma 4 31B readout, the implication test, the threshold-transfer matrix, the cascade on the confidence of Claude Haiku 5.5, the comparison with the direct APIs (E0) and controlled latency runs in two cloud regions (E8)."
    - Rewrite as a vertical list: "Planned but not run:" then one item each for "trained encoders (E4, H3)", "zero-shot encoders, guards and reward models (B4 to B6)", "the 2B pair, the two Clef pairs and the Gemma 4 31B readout (E5)", "the implication test (E3)", "the threshold-transfer matrix (E2)", "the cascade on the confidence of Claude Haiku 5.5 (E6)", "the comparison with the direct APIs (E0)" and "latency runs in two cloud regions (E8)". Do the same for "Changed:" (50 words) and "Added after the main runs:" (46 words).

Other sentences above 25 words, less bad: l. 370–373 (27 words), l. 485–488 (28 words) and l. 549–552 (26 words).

**S17 [SHOULD] Rule 10 (6 sentences or fewer in a paragraph).** These paragraphs are too long:

- Section 4.1, paragraph 1 (l. 316–337): 12 sentences. Start a new paragraph at "For the 14 System One models".
- Section 3, "Perturbations" (l. 277–297): 11 sentences. Start a new paragraph at "To test the claim".
- Limitations (l. 546–569): 13 sentences. Make a vertical list.
- Section 3, "Metrics" (l. 262–276), Section 4.4, paragraph 1, Section 4.5, paragraph 1, and Section 4.6: 9 sentences each.

A new paragraph costs space (L10). Make the splits only together with the cuts in L10.

**S18 [COULD] Rule 6.** l. 408–413 use "decisions" for our data ("flips 3.9% of the decisions", "flip more than 20% of the decisions"). The rest of the paper uses "answers". Write "answers". Keep "decisions" in statements about other papers.

Note about force:

- Item 2: Three sentences make the contrast weaker. Keep "But" at the start of the third sentence.
- Item 9: The rewrite adds content (N35). This is a content fix, not only a style fix.
- The title is still a question. As in the first review, I recommend that you keep it.

---

## 6. PDF layout

**L10 [MUST] Pages 8 and 9: the main text is longer than 8 pages.**

- The last line of Recommendation 6 ("on your development data.", l. 544) is at the top of page 9. ARR counts Section 5 as main text. Both `main.pdf` and my build of the current source show this. The overnight log says "8 main pages", but this is not correct.
- The fixes in this review add about 4 to 6 lines (N35, N36, N39, N40, R14). The final 4B numbers can also move line breaks. Thus, cut about 8 to 10 lines. Candidates:
  1. l. 386–391, "Noise floor": keep one value (S16, item 6). This gives back about 2 lines.
  2. l. 373–375: the latency sentence repeats Appendix F. Keep a short clause. This gives back about 2 lines.
  3. l. 170–187, "Calibration and robustness of LLM classifiers": shorten. This gives back 2 to 3 lines.
  4. Table 3: shorten the prediction cells. The float then takes fewer lines of page 4.

**L11 [COULD] `main.pdf` is out of date.** It has the build time 02:18:29 IST, before `main.tex` (02:20:07) and `numbers.tex` (02:20:37). It still shows "141 min apart", "later" and "in three runs" in Table 14, and "24 to 141 minutes" on l. 514. Build it again before you share it.

**L12 [SHOULD] Short appendix sections (first review L2).** Appendix B (l. 810–812), D (l. 823–824) and E (l. 825–828) have one sentence each. Their tables are 1 to 4 pages later: Table 7 on page 12, Table 10 on page 14 and Table 11 on page 15. Merge B, D and E into Appendix F, or put each table directly after its heading.

**L13 [SHOULD] Figure 1 (page 6) is too small to read on paper (first review L3).** The code sets the cell numbers to 5.6 pt and the system names to 6.4 pt (`make_paper_assets.py`, `figure_sce`). The figure is 7.0 in wide, and LaTeX scales it to the text width (about 6.3 in). Thus, the printed sizes are about 5.0 pt and 5.8 pt. Use 7 pt or more, or move the numbers to an appendix table. The grey hatching for empty cells is now in.

**L14 [COULD] Empty areas in the appendix.** Pages 12, 13, 14, 16, 17 and 19 have large empty areas. On page 17, Tables 13 and 14 fill only the left column, and on page 19 Table 18 does the same. Put the one-column tables side by side, or use `\raggedbottom` in the appendix.

**L9 [COULD] Abstract length.** The abstract has 231 words (first version: about 220). Check the current ARR limit.

Checked and correct:

- The log of the current build has no overfull box. The Table 10 box of the first review is gone.
- No floats overlap. Tables 4, 5, 7, 12 and 15 fit the text width, and Tables 3, 6, 8, 13, 14 and 18 fit the column.
- No section is empty. The Limitations section, the Ethics statement, the references and the appendix come after the main text, as ARR requires.

---

## 7. Numbers that I checked and found correct

- 95.8% (Clef-flash on BANKING77, Table 12).
- 83.3% and 41.4% (GPT-6 Luna on JudgeBench-Pro, Table 12; but see N38 for Table 17).
- 25 to 38 points (24.8 to 37.7 for the five configurations with low effort; see N47).
- 7.1 points (Claude Sonnet 5.5, mean ECE 7.14; see N50).
- 0.3 to 0.9 s (295 to 853 ms for 17 of 19 System One models) and 1.9 to 3.7 s (1,904 to 3,705 ms).
- 8.3 points (Civil Comments, −8.3, Table 16).
- "two thirds" (Solar Decide Flash: 62.7% and 67.7%).
- "about 10 minutes" (commit at 00:02:26 IST, first E2 call at 00:12:06 IST).
- "23:30 to 01:30 IST" (all API calls from 23:32 to 01:08 IST) and "02:00 IST" (01:57 to 01:59 IST; see N51).
- Huang et al.: 5.3%, 13.2%, 12.3% and 9.8% (facts file, section 6).
- Li et al. (2026a): 31 of 480 pairs = 6.5% (section 7).
- Li et al. (2026b): 65%, "a third of the fee", 26.7% on RewardBench (section 9).
- Rafe and Das: 50.5, 97.2, 84.0 and the floor 1.3 (section 2).
- JevOut: 16.9%, 61.4% and "up to 64 tries" (section 4).
- Deußer et al.: 0.028, 20 of 22, 33 of 37, 27 of 37 and 11 (section 1).
- TypedBench: T = 0.21 and 0.66, and the exception for guardrail intent (section 3).
- Geuter et al.: 79.6, 80.1 and 84.4 (section 5).
- The OpenRouter alias `~typesafe/jev-latest` resolved to `jev-1.14-20261014` on 15 October at 23:38 IST (`runs/e7-weekly-2026-10-15`).
- The counts 17 of 17; 9, 8 and 2 of 19 (BoolQ); 13 and 5 of 19 (classification); 16 (agent); 64/14 and 32/39; 122 of 148; 57/7/35; 9 of 16 nondeterministic; 8 of 14; 11 of 16; 9 of 16 (note above instruction); 69, 10, 89, 69, 19 and 17; 6.1% to 6.4% in 3 runs; 1.8% to 2.3%.
- The bound SCE ≥ 100/K − accuracy (l. 268–269).
- USD 62 in total (62.35).
- 21 E3 macros against `results/e3_summary.csv`: all agree.

---

## Appendix Z. Evidence that I computed

All values come from `runs/`, `results/` and `fixtures/` on 2026-10-16 between 02:56 and 03:15 IST (read only).

**Z1. Run times (IST, first and last call of each folder).**

| Run | Time (IST) |
|---|---|
| E0 | 15 Oct, 23:32–23:34 |
| E7 pilot | 23:36–23:38 |
| E7 alias check (`e7-weekly-2026-10-15`) | 23:38–23:39 |
| E7 repeat (not used) | 23:39–23:40 |
| Commit `3747117` "system paper 1" | 16 Oct, 00:02:26 |
| E2 | 00:12–00:53 |
| B1 (System Two models) | 00:27–01:02 |
| B2 (9B and 27B readouts) | 00:27–00:54 |
| B3 (GPT-6 Luna without reasoning) | 00:48–00:59 |
| E3 | 00:30–01:06 |
| E3 names | 01:06–01:08 |
| E7 later run | 01:57–01:59 |
| B2 local (4B) | 00:56 to now (last record 03:10) |

**Z2. Commit time.** The file `.git/logs/HEAD` has "3747117 commit: system paper 1" at 2026-10-16 00:02:26 IST. The design note has the save time 23:41:31 IST. The first E2 call has the time 18:42:06 UTC (00:12:06 IST).

**Z3. arXiv categories (`fixtures/arxiv-fresh-2026-10.jsonl` and the order of `tasks.arxiv()`).**

- The fixture: 603 papers; 11 for q-bio.NC, 32 for one category and 40 for 14 categories. Dates: 2026-10-10 00:06 UTC to 2026-10-14 16:49 UTC.
- The first 500 items: 11 for q-bio.NC, 32 for 6 categories, 33 for 9 categories (mean of the 15 categories: 32.6).

**Z4. Civil Comments cascade with and without `severe_toxicity`.** 21 settings (7 first stages × 3 second stages). I used the functions of `analysis/e6_cascade.py` (item-level rule, 2-fold cross-fitting) and removed the attribute from both stages.

| | All 7 attributes | Without severe_toxicity |
|---|---|---|
| Median gap, System Two minus System One (points) | 5.7 | 1.0 |
| System Two better by more than 1 point | 20 of 21 | 10 of 21 |
| Median escalation | 79% | 30% |
| Kept / cheap | 20 / 1 | 10 / 2 |

**Z5. Confidence AUROC of the first stage, by task (minimum to maximum over 21 settings, `results/e6_cascade.csv`).** BoolQ 0.72–0.86; Aegis 2.0 0.67–0.83; Civil Comments 0.76–0.90; BANKING77 0.83–0.90; arXiv 0.77–0.89; When2Call 0.71–0.80; JudgeBench-Pro 0.29–0.63; HelpSteer2 0.54–0.67.

**Z6. Question-level cascade check (`acc_cascade_q`, `escalated_q`, `cost_ratio_q`).** Of the 89 settings, kept 71 and cheap 20 (item level: 69 and 19). Civil Comments: median escalation 90% (item level: 79%), cheap settings 2 (item level: 1).

**Z7. System Two items without a usable answer.**

| Model | Items that E2 drops | Table 17 "Failed" |
|---|---|---|
| Claude Sonnet 5.5 | 51 (Aegis 2.0 9, JudgeBench-Pro 38, HelpSteer2 4) | 50 |
| Gemini 3.8 Flash | 5 (Aegis 2.0 2, JudgeBench-Pro 3) | 2 |
| Claude Haiku 5.5 | 6 (Aegis 2.0 3, BANKING77 1, arXiv 2) | 4 |
| GPT-6 Luna (low) | 2 (JudgeBench-Pro 2) | 0 |

- GPT-6 Luna on JudgeBench-Pro: 415 correct. E2: 415 of 498 = 83.3%. Table 17: 415 of 500 = 83.0%.
- Claude Sonnet 5.5 on JudgeBench-Pro without reasoning tokens: 189 correct. E2 rule: 189 of 258 = 73.3%. Text and Table 17: 189 of 259 = 73.0%.

**Z8. Calls and cost (all of `runs/`).** 392,015 records (the 4B run still adds records), USD 62.35. System One calls (E0, E2, E3, E3 names and the E7 runs): 358,549 for USD 20.78.

**Z9. Mean ECE over tasks (`results/e2_metrics.csv`).** Claude Sonnet 5.5 7.143; Drex v2.0 7.491; Claude Haiku 5.5 8.334; pplx-decider v1.1 8.831; Jev 1.14 9.148.

**Z10. Interference above the matched floor (Table 5, cells that are not grey).** Clef, Clef-flash and Clef-omni (Crowd. and Alone); Mercury Decide and Drex v2.0 (Crowd. and Alone); Liquid d1, Drex v1.5 and Kev-4B (Alone only). Jev 1.14: none.

**Z11. Jev 1.14 and the reviewer note (`results/e3_rows.csv`).** Follow rate 26.2% (mean over 6 tasks). "Broken" rate (originally correct answers that became wrong): 24.7%.

**Z12. HelpSteer2 (`results/e2_metrics.csv`).** The exact-match accuracy is below the majority baseline (54.9%) for 25 of 26 systems (all except Liquid d1). "Within 1 level": System One models 76.0% to 94.5%, System Two models 81.3% to 86.9%, backbones 75.2% to 89.1%.

**Z13. E7 on easy tasks (`results/e7_versions.csv`).** BoolQ: version 2.2% [1.0, 3.6], repeats 0.1% [0.0, 0.4]. Aegis 2.0: 2.2% [1.0, 3.6] against 0.5% [0.1, 1.0]. arXiv: 1.4% [0.6, 2.6] against 0.8% [0.0, 1.9].
