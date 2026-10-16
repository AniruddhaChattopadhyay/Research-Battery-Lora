# Internal review of paper 1 (strict ARR reviewer), 2026-10-16

Paper: "Same Question, Different Answer? A Controlled Study of System One Decision Models" (`paper1/`).

## 0. Scope, version and method

Reviewed version:

- `paper1/main.tex`, MD5 `ef601a07718c943b4d1225e21ebb5f14`, saved 01:27 IST.
- `paper1/numbers.tex`, MD5 `c7cdffdb404f4ae76abfe18ba2f239ae`, saved 01:28 IST.
- `paper1/main.pdf`, built 01:28 IST. The PDF now has 16 pages, not 15.

The paper changed while I worked. New text came into Sections 4.4, 4.6, the Limitations and Appendix D. I reviewed the version above. "l. 327" means line 327 in the ARR line numbers of this PDF.

Update at 01:52 IST: `main.tex` changed again at 01:50 IST (MD5 `510688491e8775f4591547c6c263f15e`, PDF not rebuilt). The new text fixes N12 and N29, and parts of N11 and N19. I mark the status at these findings. All other findings still apply to the 01:50 version.

What I did:

1. I compared each number in the main text with `numbers.tex`, the tables and `results/*.csv`.
2. I read the raw runs in `runs/` (read only, offline) for claims that the CSV files do not cover.
3. I compared each statement about another paper with `notes/2026-10-16-related-work-facts.md`.
4. I compared the method text with the code in `src/` and `analysis/`.
5. I read all 16 pages of the PDF.

I did not edit other files, run git or call an API. My scratch scripts are in `/tmp/rev_scripts/`, outside the repository. Appendix Z gives the numbers that I computed.

Labels:

- **MUST**: the text is wrong or misleading. Fix it before submission.
- **SHOULD**: the problem weakens the paper. An ARR reviewer will probably raise it.
- **COULD**: polish.

Terms in this review:

- "S1 model": a System One model. "S2 model": a System Two model (a general LLM with structured outputs).
- "Flip": a change of the top answer.
- "Floor": the noise floor, that is, the flip rate between identical requests.

---

## Summary and the 10 most important MUST findings

The idea is good: test the published disagreements under one protocol. The coverage is wide. The noise floor, the backbone pairs and the release of all responses are real contributions. The numbers in the text agree with `numbers.tex` and the tables in almost all places.

But several headline claims are wrong or stronger than the data. The 10 most important MUST findings are:

1. **N1, N2.** Table 7 rows "next day" compare calls that are 35 minutes apart, not one day. "Minutes apart" is 18 to 30 minutes.
2. **N3.** The 4B result of E5 (abstract, findings, Tables 3, 5 and 13) comes from an unfinished local run with 2 of 7 tasks.
3. **N4.** The paper says "H1 holds". The pre-registered H1 predicted calibrated or underconfident classification. In fact, 16 of 19 S1 models are overconfident on classification.
4. **N5, W1.** On JudgeBench-Pro, all 17 S1 models are below the "always A" baseline (56.0%) and 14 are below 50%. With 2 options, this forces a positive SCE. The paper has no trivial baselines.
5. **N7.** "Even more on the system" is wrong. The perturbation explains 57% of the variance in Table 4, the system 7%.
6. **N8, M1.** Civil Comments is not "confidently wrong" (AUROC 0.77 to 0.90). The high escalation comes from an undisclosed item-level rule (the lowest of 7 confidences).
7. **N10.** "The OpenAI Decisions API equals its base model without reasoning" hides task differences of up to 8.3 points in both directions (pooled McNemar p = 1.8e-35).
8. **N11.** "Only with reasoning" comes from one model. Claude Sonnet 5.5 is correct on 73.3% of the JudgeBench-Pro calls with zero reasoning tokens.
9. **R1.** The text compares our 98.1% only with the 50.5 of Rafe and Das. It does not give their 97.2 and 84.0 on the other two binary sets, which agree with us.
10. **W2.** Appendix E is incomplete. The headline name-swap test came after the main runs. H7 was probably final only after the E7 pilot on the same items (check the git history).

---

## 1. Numbers and claims in the main text (abstract to Recommendations)

### 1.1 Wrong numbers and wrong labels

**N1 [MUST] Section 4.6, l. 455–457, and Table 7 rows "1.14, next day" and "1.13, next day".**

- Problem: The text says "2.2% between two days". The "next day" runs (`runs/e7-2026-10-15/`) ran at 18:06–18:08 UTC. The E2 runs ran at 18:42–18:50 UTC on the same UTC day. The median gap between the two calls of an item is 35.1 minutes (1.14) and 36.2 minutes (1.13). The label comes from the IST date: the runs cross midnight IST. The docstring of `analysis/e7_versions.py` repeats the error ("on 2026-10-15 ... and on 2026-10-16").
- Fix: Rename the rows "1.14, 35 minutes apart" and "1.13, 36 minutes apart". Delete "between two days". Say that all E7 data come from one 2-hour window.

**N2 [MUST] Section 4.6, l. 455–456, and Table 7 row "1.14, minutes apart".**

- Problem: This row compares the E2 call with the 4 E3 repeats. The gap is 18 to 30 minutes (median 24). Thus "a few minutes apart" is not correct. Also, this row and the "next day" rows have almost the same gap, so Table 7 does not show two time scales.
- Fix: Write "18 to 30 minutes apart". Then write one statement: "Identical requests within 40 minutes change 1.9% to 2.3% of the top answers."

**N3 [MUST] Abstract l. 024–026; findings l. 103–105; Section 4.4 l. 404–406; Table 3 row "Qwen3.5-4B readout"; Table 5 row 1; Table 13 row 1.**

- Problem: The 4B pair uses an unfinished local run. At 01:30 IST, `runs/b2-local/` was still running (BoolQ 200, HelpSteer2 200, Civil Comments 138 items). E5 used 200 BoolQ items and 81 HelpSteer2 items (`results/e5_training.md`: n = 200 and 405 questions). Table 3 used 200 BoolQ items and 113 HelpSteer2 items (`results/e2_metrics.csv`: n = 565 questions). Thus Tables 3 and 5 use different items for the same model. The Table 5 caption says "macro mean over the 7 tasks", but the 4B row has 2 tasks. In Table 13, "–" for 4B means "not run yet", but Table 11 defines "–" as "the system does not accept the task". The overnight log confirms that the run is not finished.
- Fix: Let the run finish. Then run `analysis/e2_calibration.py`, `analysis/e5_training.py` and `analysis/make_paper_assets.py` again. If the run cannot finish in time, remove the 4B pair from the abstract and the findings. Show it in the appendix as "partial: 2 tasks, 281 items".

**N12 [MUST] Section 4.1, l. 327–329 ("the System Two models in 1.8 to 3.6 seconds").**

- Problem: Tables 3 and 14 give 1.90 s (Claude Haiku 5.5) to 3.70 s (Gemini 3.8 Flash). Appendix D (l. 756–757) says "1.9 to 3.7 seconds".
- Fix: Write "1.9 to 3.7 seconds".
- Status: fixed in the 01:50 version.

**N13 [MUST] Section 4.2, l. 379–380 ("5 systems follow the note more often than the instruction").**

- Problem: In Table 4, 9 systems have Hint > Inject, and 5 have Inject > Hint. The macro `nHintOverInject` counts only differences of more than 5 points (`analysis/make_paper_assets.py`: `hint[x] > inject[x] + 0.05`). The text does not say this.
- Fix: Write "9 of 16 systems follow the note more often than the instruction, 5 of them by more than 5 points".

**N14 [MUST] Section 4.2, l. 348–349 ("The Clef models follow the names in only 14.9% to 19.6%").**

- Problem: Clef-omni follows the names in 14.1% (Table 4). The range uses only Clef and Clef-flash.
- Fix: Write "14.1% to 19.6%".

**N15 [MUST] Section 3, l. 203–206 ("Three systems limit the interface").**

- Problem: Five systems have limits: Span-01, Span-01-lite, Tev1-4B, Solar Decide and Solar Decide Flash (`src/openrouter.py`, `accepts`).
- Fix: Write "Five systems from three vendors limit the interface".

**N17 [MUST] Section 4.2, l. 383–384 ("the published range from 1.3% to 73%").**

- Problem: The paper never introduces 73%. The intro and Table 1 give 61.4% for Jev. The value 73.2% is the JevOut result for Von, a different system.
- Fix: Write "from 1.3% to 61.4%". Or add the Von value to Table 1 and say that it is not Jev.

**N18 [SHOULD] Abstract l. 027–028; Section 4.6 l. 458–459; Table 4 column "Floor".**

- Problem: The paper gives two floors for Jev 1.14. Table 4 gives 1.9% (mean over 8 tasks). Table 7 and the abstract give 2.3% (pooled over 3,600 questions, so Civil Comments and HelpSteer2 weigh more). "2.7 times the noise floor" uses the pooled value. A reader who divides 6.1 by 1.9 gets 3.2.
- Fix: Use one definition, or say in both captions which mean you use. With task means, the ratio is 5.5 / 1.95 = 2.8, so the message does not change.

### 1.2 Claims that the data do not support, and overclaims

**N4 [MUST] Section 4.1 l. 318–319 ("and H1 holds"); findings l. 088–093.**

- Problem: The pre-registered H1 has directional parts: "System One models are overconfident on knowledge and reasoning tasks, and well calibrated or underconfident on classification tasks" (`notes/2026-10-15-experiment-design.md`, line 22). The data:
  - Classification: 16 of 19 S1 models have a 95% interval above 0 (median +4.9 points). Only 3 are underconfident. This contradicts the prediction.
  - Knowledge (BoolQ): 9 of 19 are overconfident, 8 contain 0 and 2 are underconfident. This is mixed.
  - Judging: all 17 are overconfident on the pooled group. This agrees.
- The paper keeps only the first sentence of H1 ("Calibration depends on the task type").
- Fix: Write "H1 is partly supported". Give the status of each part. Add a hypothesis table (prediction, result, status) to the main text.

**N5 [MUST] Section 4.1 l. 287–290; abstract l. 014–018; findings l. 088–090.**

- Problem 1: The overconfidence on the judging tasks is partly a result of low accuracy. JudgeBench-Pro has 2 options, so the top probability is always at least 0.5, and SCE ≥ 50 − accuracy (in points). 14 of 17 S1 models are below 50% on JudgeBench-Pro. All 17 are below the "always A" baseline (56.0%: 280 of the 500 gold answers are "A").
- Problem 2: On HelpSteer2, the majority level of each attribute gives 54.9% accuracy. Only Liquid d1 (59.5%) is above it. All S2 models (37.7% to 45.9%) and all backbones are below it.
- Problem 3: "Overconfident on them" is true only for the pooled judging group. On HelpSteer2 alone, Liquid d1 (−2.3) and Kev-4B (−2.6) have a negative SCE.
- Fix: Add a majority row and an "always A" row to Table 11 (the plan had them as baseline B7). State the bound SCE ≥ 1/K − accuracy for K options. Write "on the pooled judging group". Report H1 also without JudgeBench-Pro (see N6).

**N32 [MUST] Section 4.1 l. 304–309 and l. 315–317 (Jev on Civil Comments).**

- Problem: At the 0.5 rule, `severe_toxicity` has 0 positives in the 500 items. Jev 1.14 says "yes" to it on 51.2% of the items. This one attribute drives the Civil Comments numbers in the text:
  - top-label ECE 5.6 points (1.9 without `severe_toxicity`);
  - SCE +5.6 points (+1.8 without it);
  - ECE of P(yes) 20.5 points (15.6 without it).
- `severe_toxicity` alone causes 36% of the Jev errors on Civil Comments. Four more attributes have fewer than 5% positives (threat 0.6%, sexual_explicit 1.2%, obscene 3.6%, identity_attack 4.8%). Thus the majority baseline on Civil Comments is 86.1%, and Jev 1.14 (79.7%) is below it.
- Fix: Report the positive rate of each attribute. Drop `severe_toxicity` or use the soft labels (the release already has a soft Brier score). Run E2 again offline. The claim "Jev says yes too often on the rare attributes" stays true; support it with numbers (identity_attack: 17.6% "yes" against 4.8% positives).

**N6 [SHOULD] Section 4.1 l. 285–287 ("the rows of the figure (tasks) differ more than its columns (systems)").**

- Problem: The claim is true, but it depends on the judging tasks. A two-way split of SCE for the 14 S1 models with all 8 tasks gives: task 64%, system 14%, residual 22%. Without JudgeBench-Pro and HelpSteer2, it gives: task 32%, system 39%.
- Fix: Report the split and its sensitivity in one sentence.

**N7 [MUST] Section 4.2 l. 383–386 ("The size of the effect depends on the perturbation, as H2 predicts, and even more on the system"); title of Section 4.2; findings l. 094–099.**

- Problem: A two-way split of Table 4 (14 systems × 12 perturbations) gives: perturbation 57%, system 7%, interaction and residual 35%. Thus the effect depends much more on the perturbation than on the system. The large interaction supports the abstract ("the effect of a perturbation depends on the system"). It does not support "Robustness depends on the system". Also, H2 predicted "polarity swaps and natural context change many answers; option order and synonyms change few". For Jev, the negation changes only 3.9%.
- Fix: Write "The size of the effect depends mostly on the perturbation, and the same perturbation acts very differently on different systems." Report H2 as partly supported. Rename Section 4.2 "Robustness depends on the perturbation and the system".

**N8 [MUST] Section 4.5 l. 445–449 ("On JudgeBench-Pro and Civil Comments, the System One models are confidently wrong, and the cascade must send almost all items to the System Two model").**

- Problem: This is true for JudgeBench-Pro (confidence AUROC 0.30 to 0.61). It is not true for Civil Comments. There, the confidence separates right from wrong answers well: AUROC 0.77 to 0.90 for the 7 first stages (When2Call: 0.71 to 0.81). The escalation on Civil Comments goes from 13% to 100% (median 79%). Two of the 28 cheap cases are on Civil Comments (pplx-decider to GPT-6 Sol: cost ratio 0.14). The high escalation comes from the cascade rule: the code sends an item up if any of its 7 answers is below the threshold (see M1).
- Fix: Remove "and Civil Comments" from this sentence. Explain the item-level rule. Add a question-level variant for Civil Comments as a check (offline, from saved outputs).

**N9 [MUST] Findings l. 109–113; Section 4.5 l. 438–445; Table 6.**

- Problem 1: 8 of the 28 "cheap" cases are trivial. In these cases, the S1 model alone is already within 1 point of the S2 model, so a cascade is not necessary. Example: Jev 1.14 to Claude Sonnet 5.5 on BoolQ (89.8% against 90.4%, 0% escalated).
- Problem 2: The cheap cases are not only on When2Call, arXiv and BANKING77. Six of 28 are on BoolQ (3), Civil Comments (2) and Aegis 2.0 (1).
- Fix: Count only the 90 cases where the S2 model is more than 1 point better. There, the cascade keeps the accuracy in 70 and is cheap in 20. Write "mostly on When2Call, arXiv and BANKING77 (22 of 28)".

**N10 [MUST] Findings l. 106–107 ("The OpenAI Decisions API equals its base model without reasoning"); Section 4.4 l. 421–424 ("almost the same accuracy").**

- Problem: The macro difference is −1.1 points. But the task differences are large and go in both directions: Civil Comments −8.3, HelpSteer2 −8.2, JudgeBench-Pro +7.7 (Table 13). The pooled McNemar test in `results/e5_pairs.csv` gives p = 1.8e-35. The pooled difference is about −5.3 points.
- Fix: Write "The macro accuracy is similar (70.3% and 71.3%), but the two differ by up to 8 points on single tasks, in both directions." Also mark this pair as exploratory in the main text (it is deviation 6).

**N11 [MUST] Section 4, l. 280–283 ("The System Two models lead on JudgeBench-Pro by 30 to 60 points, but only with reasoning").**

- Problem 1: "Only with reasoning" comes from one model (GPT-6 Luna). Claude Sonnet 5.5 used zero reasoning tokens on 258 of its 462 JudgeBench-Pro answers, and it was correct on 189 of them (73.3%). Over all tasks, Sonnet used reasoning tokens in 5% of the calls, and Haiku in 18%. (The model chooses when it reasons, so this split is not random.)
- Problem 2: "30 to 60 points" has no clear reference. Against the best S1 model (pplx-decider, 53.4%), the lead is 24.8 to 37.7 points.
- Fix: Write "GPT-6 Luna leads only with reasoning (83.3% against 41.4%)." Give the lead against the best S1 model. Report the reasoning tokens of each S2 model (see M9).
- Status: the 01:50 version fixes Problem 2 ("25 to 38 points above the best System One model"). Problem 1 ("but only with reasoning") is still open.

**N16 [MUST] Section 4.4 l. 411–412 ("The gains are largest on the judging tasks").**

- Problem: This is true only at 27B (JudgeBench-Pro +18.2, HelpSteer2 +16.7). At 9B, the largest gain is on When2Call (+15.0), and JudgeBench-Pro (+5.4) is below the mean (+7.5). At 4B, BoolQ (+8.5) is above HelpSteer2 (+4.0).
- Fix: Write "At 27B, the gains are largest on the judging tasks."

**N19 [SHOULD] Intro l. 078–080 ("so no system saw them in training"); Section 3 l. 238–240.**

- Problem 1: Three systems came out inside the window of 10 to 15 October: Jev 1.14 (14 October), Classify-1 (13 October) and Drex v2.0 (12 October).
- Problem 2: The text says "603 arXiv papers", but the task uses 500 of them, balanced over the 16 categories (`src/tasks.py`, `arxiv()`). The abstract says "500 items each".
- Fix: Write "so most systems were released before these papers existed". Write "We sample 500 of the 603 papers, about 31 for each category (11 for q-bio.NC)".
- Status: the 01:50 version writes "unlikely to have seen them" and "We use 500 of them". Its new sentence about version dates is correct: by the date suffix, only Jev 1.14 is newer than the window start (Clef and Clef-flash have no date). Still open: the category balance, and the OpenRouter listing dates of Classify-1 and Drex v2.0 inside the window.

**N20 [SHOULD] Intro l. 081–084 ("We also send each request 4 more times without a change").**

- Problem: The repeats exist only for the first 200 items of each task, only in E3, and only for the 16 S1 models in E3. The S2 models and the backbones have no repeats.
- Fix: Write "For the 16 S1 models in E3, we send each of the 200 base requests 4 more times".

**N21 [SHOULD] Findings l. 091–093 and Section 4.1 l. 296 ("explains much of the published disagreement").**

- Problem: The paper does not compute the metrics of the other studies on our data. The plan had this step ("Calculate the metric in the three ways that the three papers used"). Our explanation for TypedBench (accuracy near 1) fits at most one of its two informative refits (phish.is_phishing, accuracy 0.99). It does not fit tickets.priority (Jev median accuracy 0.874, T = 0.66).
- Fix: Compute these three metrics offline from the saved outputs:
  1. a temperature refit on half of the items, as in TypedBench;
  2. top-label ECE with 10 equal-mass bins, as in Rafe and Das;
  3. ECE of P(yes) for predicates, as in Deußer et al.
- Then show which verdict each metric gives. If time is short, write "can explain part of".

**N22 [SHOULD] Section 4.1 l. 315–317 ("our small positive SCE for Jev on Aegis, Civil Comments and BANKING77").**

- Problem: The sentence leaves out arXiv, the fourth classification task (−0.1). Deußer et al. report choice questions, but Aegis and Civil Comments are predicates here. Our two choice tasks give +8.1 (BANKING77) and −0.1 (arXiv). Also see N32 for Civil Comments.
- Fix: Give all four values and mark the two choice tasks.

**N23 [SHOULD] Section 4.3 l. 399–401 ("Thus, H6 holds for the Clef models ... and related questions change them most").**

- Problem 1: H6 predicts interference "also for systems that claim isolation". Jev claims isolation and shows none. On the same tasks, Crowd gives 1.5% against a floor of 1.6%, and Alone gives 2.5% against 2.9%. Thus H6 fails for the system that it names.
- Problem 2: "Related questions change them most" compares Alone (Civil Comments, HelpSteer2) with Crowd (six other tasks). The tasks and the changes differ, so the comparison does not isolate relatedness.
- Fix: Write "H6 is not supported for Jev. It holds for the three Clef models (exploratory)." Delete the last clause, or say that the tasks differ.

**N24 [SHOULD] Recommendation 3, l. 477–479 ("A threshold from a classification task is too low for a judging task").**

- Problem: The paper does not test threshold transfer. The plan had a threshold-transfer matrix (E2, step 4).
- Fix: Compute the matrix offline from the saved outputs. Or write "Do not reuse a threshold from another task without a check."

**N25 [SHOULD] Limitations l. 490–491 ("one region and two days"); Section 3 l. 208–209.**

- Problem: All main calls ran between 18:02 and 20:02 UTC on 15 October 2026 (23:32 to 01:32 IST). This is one 2-hour window at night.
- Fix: Write "one route, one region and one 2-hour window (23:30 to 01:30 IST, 15 to 16 October 2026)".

**N26 [SHOULD] Section 4.6 l. 463–468 (the `jev-latest` alias).**

- Problem: The OpenRouter alias is `~typesafe/jev-latest` (with a tilde). The TypeSafe documentation talks about the TypeSafe alias `jev-latest`. These can be two different aliases. Our runs show only the OpenRouter alias.
- Fix: Write "The OpenRouter alias `~typesafe/jev-latest` already pointed to 1.14 on 15 October. The TypeSafe documentation said that its own alias would move in about one week."

**N27 [SHOULD] Section 4.4 l. 413–415 ("part of the gain on the public tasks can come from training data that is similar to these tasks").**

- Problem: The backbones already reach 87.2% (9B) and 88.8% (27B) on arXiv. A ceiling can also explain the small gain. The test does not separate the two causes.
- Fix: Add "or from a ceiling, because the backbones already reach 87% to 89% on arXiv".

**N28 [SHOULD] Section 4.5 l. 449–451 ("Li et al. report the same effect on JudgeBench").**

- Problem: The effect there is weaker. Li et al. escalated 65% and saved a third of the fee. Here, JudgeBench-Pro escalates 95% to 100% and saves nothing (cost ratio 0.96 to 1.12). Their Jev accuracy on JudgeBench is 81.3%; ours on JudgeBench-Pro is 51.2%. Their main result is positive: on RewardBench, the cascade beat GPT-6 at 26.7% of its fee.
- Fix: Write "a similar but weaker effect", and give their RewardBench result.

**N29 [COULD] Section 4, l. 276–278 ("the best System One models are as accurate as the System Two models").**

- Problem: On Civil Comments (90.7% against 92.1%) and arXiv (90.8% against 92.4%), the best S1 model is 1.4 to 1.6 points lower. The text gives no test.
- Fix: Write "about as accurate as", and give a paired test.
- Status: fixed in the 01:50 version ("at most 2 points below the best System Two model or better").

**N30 [COULD] Intro l. 051–054 ("On 16 October 2026, OpenRouter served 19 of them").**

- Problem: The landscape note checked the list on 15 October (23 entries: 19 models, 2 aliases and 2 free variants).
- Fix: Use the date of the check, or check again.

**N31 [COULD] Findings l. 114 ("A minor version changes many answers").**

- Problem: "Many" is vague. The change is 6.1%, which is 3.8 points above the floor.
- Fix: Write "A minor version changes 6.1% of the answers".

### 1.3 Numbers that I checked and found correct

All other numbers in the abstract, the findings and Sections 3 to 5 agree with `numbers.tex`, the tables and `results/*.csv`. Examples:

- the medians in Section 4.1 (+17.4, +2.0, +4.9, +12.8);
- the counts 8 of 19, 3, 17, 9 of 16, 7, 11 of 16 and 8 of 14;
- all medians in Section 4.2;
- the E5 and E6 counts (69, 168, 99, 79, 28);
- the E7 values (2.3, 6.1, 15.4, 12.1, 5.2, 4.6);
- the costs (USD 62.27 in total; 356,577 S1 calls for USD 20.70).

---

## 2. Statements about other papers

I checked each statement against `notes/2026-10-16-related-work-facts.md`. Most statements are correct. These are the problems.

**R1 [MUST] Section 4.2 l. 349–354 (Rafe and Das).**

- Problem: The text gives only "50.5 flips per hundred decisions" and says that our generic rubrics cause our higher rate (98.1%). But Rafe and Das also report 97.2 (conversation derailment) and 84.0 (power) for Jev (facts file, section 2). These values are close to ours. On those two sets, the authors removed the leading key word from the rubrics, so the name is the only cue.
- Fix: Give all three values. Say that our 98.1% is near their two D3 sets. Remove the causal claim about our rubrics, or mark it as a hypothesis.

**R2 [MUST] Section 4.2 l. 363–368 (Huang et al., "close to").**

- Problem: The text compares three systems and calls the values "close". It leaves out Mercury Decide, which both studies test: 9.8% in Huang et al. against 4.4% here. Our Jev value (2.8%) is about half of theirs (5.3%). The protocols differ: Huang et al. use one shuffle for each question and other datasets (Banking20, Newsgroups, MMLU-Pro and others).
- Fix: Give all four pairs. Write "lower than, but in the same order as". State the protocol differences. Also check that the "GPT-6 Luna" of Huang et al. is the decisions endpoint; the facts file does not say this.

**R3 [SHOULD] Section 4.2 l. 357–360 (Li et al., "This is close to").**

- Problem: Li et al. count only the pairs with "yes" to both questions (31 of 480, that is, 6.5%). Our 3.9% counts every changed decision. The tasks also differ (ChaosNLI and PubMedQA labels against BoolQ, Aegis 2.0 and Civil Comments), and so does the version (1.13 against 1.14).
- Fix: Write "Li et al. find that Jev 1.13 says 'yes' to a statement and to its negation in 6.5% of the pairs (31 of 480). This count includes only one type of error."

**R4 [MUST] Section 3 l. 235–237 ("kept the items on which LLM judges often fail").**

- Problem: The BiasScope authors added a bias to the wrong response. Then they kept the items that one judge, Qwen3-32B, got wrong in both answer orders. A manual check reduced the 1,341 items to 1,178. The Limitations (l. 498–499) now say this correctly, so the two sections disagree.
- Fix: Write "Its builders added a bias to the wrong response and kept the items that Qwen3-32B judged wrongly in both orders".

**R5 [SHOULD] Section 4.2 l. 374–377 ("the one-shot control of JevOut (16.9%), whose sentence does not name the answer").**

- Problem: The facts file says only that the control is "target-aware". It does not say that the sentence does not name the answer. Also, our 26.2% is a mean over tasks with a wide range: the note moves 2.4% (Aegis 2.0) to 71.6% (JudgeBench-Pro) of the Jev answers. On JudgeBench-Pro, one note moves more answers than the JevOut search (61.4%).
- Fix: Check the JevOut text. Then write, for example, "whose sentence supports the wrong option but does not tell the model to choose it". Give the per-task range.

**R6 [SHOULD] Table 1 row "Value" (Geuter et al.).**

- Problem: 79.6, 80.1 and 84.4 are "Capability" scores on OpenDecisions (chance-corrected accuracy and calibration combined). They are not accuracies. The 79.6 also uses a fitted temperature (72.9 without it).
- Fix: Write "Capability score (accuracy and calibration combined)".

**R7 [SHOULD] Intro l. 058–060 and Table 1 (TypedBench).**

- Problem: The text is correct for Suite A. But TypedBench also reports that Jev is over-confident on guardrail intent (ECE/null 2.8). Also, two of the four Suite A refits are degenerate. This exception supports the main point of the paper: the verdict depends on the task.
- Fix: Add "(but over-confident on guardrail intent)" to Table 1.

**R8 [SHOULD] Section 4.4 l. 408–411 ("but it does not change its accuracy (Geuter et al.)").**

- Problem: The facts file does not give this statement for Geuter et al. It is a property of the math: one temperature does not change the argmax.
- Fix: State it without the citation. Cite Geuter et al. for the gain from temperature scaling (Capability 72.9 to 79.6).

**R9 [SHOULD] Section 2 l. 188–189 ("can fall for injected biases (Tan et al., 2025; Lai et al., 2026)").**

- Problem: JudgeBench (Tan et al.) tests hard response pairs for correctness. It does not inject biases. This paper is not in the facts file, so please confirm.
- Fix: Keep only Lai et al., or cite a bias-injection study.

**R10 [COULD] Table 1 row "Value" ("Jev beats Qwen3.8-27B on 27 of 37 datasets").**

- Problem: Only 11 of the 27 leads have non-overlapping 95% intervals.
- Fix: Add "(11 with non-overlapping intervals)".

**R11 [COULD] Section 2 l. 139–143 ("use ECE with 15 equal-width bins on 37 datasets").**

- Problem: They report ECE on 33 datasets. The score-only datasets have no ECE.
- Fix: Write "on the 33 datasets with a choice or predicate question".

**R12 [COULD] Intro l. 067–069 ("61.4% of correct decisions changed by an adversarial search").**

- Problem: JevOut counts only moves to one fixed wrong option. Table 1 says this correctly. The intro does not.
- Fix: Write "moved to a fixed wrong option".

**R13 [COULD] Statements outside the facts file.**

- The facts file does not cover Artificial Analysis, Microsoft (1.3%), TypeSafe, Perplexity, Cloudflare, OpenRouter, Hu et al., Lan et al., Zhang et al. and the older papers. The vendor statements agree with `notes/2026-10-15-model-landscape.md`. Check Hu et al., Lan et al. and Zhang et al. before submission.

---

## 3. Method description against the code

**M1 [MUST] Section 4.5 l. 429–431 (cascade rule). Code: `analysis/e6_cascade.py`, lines 38–49 and 52–66.**

- Text: "A cascade keeps the System One answer if its confidence is at or above a threshold".
- Code: The confidence of an item is the lowest top probability over all questions of the item. If one question is below the threshold, the code sends the whole item to the S2 model and replaces all its answers. This changes the results for Civil Comments (7 questions) and HelpSteer2 (5 questions). See N8.
- Fix: Describe the item-level rule, and give the reason (one S2 call answers all questions of an item).

**M2 [SHOULD] Section 4.5 l. 434–435 ("The threshold comes from 2-fold cross-fitting").**

- Code: The code splits the items by a hash of the item id. On one half, it picks the lowest threshold that reaches the S2 accuracy of that half. The grid goes from 0 to 0.85 in steps of 0.05, then from 0.90 to 1.00 in steps of 0.01. If no threshold reaches the target, the code sends all items. The cost always includes the S1 call, so the cost ratio can be above 1 (up to 1.12). Figure 2 shows in-sample curves on all items, not the cross-fitted result.
- Fix: Add these details in two or three sentences.

**M3 [SHOULD] Section 3 "Perturbations" l. 268–273 (hint and injection). Code: `src/perturb.py`, lines 161–169 and 209–217; `analysis/e3_robustness.py`, line 97.**

- Code: The hint and the injection exist only for the 6 single-question tasks (not Civil Comments or HelpSteer2). X is a random wrong option, seeded by the item id. It is not the most probable wrong option, as in JevOut. For choices, X is the option text (for example "Response B is correct"). The follow rate counts only the items whose original top answer is not X.
- Fix: State the tasks, the choice of X and the denominator.

**M4 [SHOULD] Section 3 l. 256–257 ("We apply 12 perturbations to the first 200 items of each task") and the Table 4 caption.**

- Code: Each perturbation applies to some tasks only:
  - Neg., Fmt. and Names: BoolQ, Aegis 2.0, Civil Comments.
  - Order and Codes: BANKING77, arXiv, When2Call.
  - Rev.: HelpSteer2. Swap: JudgeBench-Pro.
  - Crowd., Hint and Inject: the 6 single-question tasks.
  - Alone: Civil Comments and HelpSteer2.
- Fix: Add this list as a footnote to Table 4.

**M5 [SHOULD] Table 4 column "Floor". Code: `analysis/e3_robustness.py`, lines 157–158.**

- Code: The code computes a floor for each perturbation on the same tasks (column `floor` in `results/e3_summary.csv`). Table 4 shows only the floor over all tasks. For Jev, the matched floor is 5.2% for Swap (JudgeBench-Pro) and 4.6% for Rev. (HelpSteer2), not 1.9%.
- Fix: Show the matched floor, for example in Table 12.

**M6 [SHOULD] Section 3 l. 197–199 ("We pin each model version").**

- Code: The code sends a model id without a date (for example `cloudflare/clef`), and it records the resolved version. For Clef and Clef-flash, the response has no version suffix. Thus their version is not known.
- Fix: Write "We use fixed model ids (no aliases) and record the resolved version. Clef and Clef-flash report no version."

**M7 [MUST] Table 3 caption; Section 3 "Metrics" l. 251–252; Section 4.1.**

- Code: The group SCE values pool all questions (`analysis/e2_calibration.py`, lines 82–96). Civil Comments gives 3,500 of the 5,000 classification questions (70%). HelpSteer2 gives 2,500 of the 3,000 judging questions (83%). In the same table, "Acc." and "ECE" are means over tasks. The caption does not say this.
- Fix: Say in the caption that the SCE columns pool the questions. Better: report the mean over tasks, or both.

**M8 [SHOULD] Section 3 "Baselines" l. 210–215. Code: `src/llm.py`, lines 95–117, 141 and 144–147.**

- Code: Three details are missing:
  1. For BANKING77 (77 options), the S2 model lists only the likely options (probability 0.01 or more), not all options.
  2. The code sets negative values to 0 before it normalizes.
  3. A response with empty content becomes `parse_error`, and the analysis drops it. Claude Sonnet 5.5 lost 50 items in this way (37 on JudgeBench-Pro, 9 on Aegis 2.0, 4 on HelpSteer2). Claude Haiku 5.5 lost 4 and Gemini 3.8 Flash lost 2.
- Fix: Add the three details. Add a sensitivity check that counts the dropped items as wrong.

**M9 [SHOULD] Section 3 "Baselines" ("reasoning effort low"). Data: `runs/b1/*.jsonl`, field `usage.completion_tokens_details.reasoning_tokens`.**

- Data: The code asks every S2 model for effort "low". The share of calls with any reasoning tokens is: GPT-6 Sol 38%, Claude Sonnet 5.5 5%, Gemini 3.8 Flash 58%, Claude Haiku 5.5 18%, GPT-6 Luna 82%. Thus "low" gives different amounts of reasoning.
- Fix: Report the mean reasoning tokens of each model on each task in the appendix.

**M10 [SHOULD] Section 4.4 and the Table 5 caption. Code: `analysis/e5_training.py`, lines 49–64.**

- Code: In Table 5, the accuracy is a mean over tasks, but the ECE pools the questions (so Civil Comments and HelpSteer2 weigh more). The McNemar test also pools the questions, but the 7 questions of one Civil Comments item are not independent. The paper does not report the p-values, although Section 3 says that it uses the test.
- Fix: Say which mean each column uses. Report a cluster-bootstrap interval for each gain (resample items).

**M11 [SHOULD] Section 3 "Baselines" l. 219–222 (letter readout). Code: `src/llm.py`, lines 151–205; `src/local_readout.py`.**

- Code: The readout prompt shows the instruction and the option texts with letters. It does not show the option names. A letter outside the 20 most probable tokens gets probability 0. If no letter is in the top 20, the code drops the answer. The text calls this prompt "the question card".
- Fix: Add these details in one or two sentences.

**M12 [SHOULD] Section 3 "Tasks" (When2Call and the length limits). Code: `src/tasks.py`.**

- Code: When2Call has 4 options, but "direct" is never correct (3 gold classes). The code drops items above 6,000 characters (12,000 for JudgeBench-Pro). The Civil Comments sample alternates toxic and clean comments.
- Fix: State these three details.

**M13 [COULD] Appendix A l. 733–736 ("Tev1-4B sometimes wrote text instead of an option letter").**

- Data: The error messages say that Tev1-4B answered "C", "D" or "H" to a 2-option question. These are letters outside the options, not text. Tev1-4B lost 14 JudgeBench-Pro items in this way.
- Fix: Write "answered with a letter that is not an option".

**M14 [COULD] Retries. Code: `src/openrouter.py`, lines 17 and 57–74; `src/run.py`, lines 51–61.**

- Code: The client tries status 502 up to 5 times, and a second run repeats the failed items. The OpenAI refusals return 502. Thus some refusals became answers on a later try (2 items got an answer on the 4th attempt). The latency is the time of the last attempt only.
- Fix: Say this in Appendix A and in the Table 14 caption.

**M15 [COULD] Section 3 "Perturbations" (Fmt. and Names). Code: `src/perturb.py`, lines 202–207.**

- Code: The true rubric is always the first option in both conditions. Thus the test cannot separate "follows the rubric" from "follows the position". Rafe and Das have the same design ("rubric text and order fixed").
- Fix: Say this in one sentence.

**M16 [COULD] Section 3 "Metrics" l. 253–254 ("500 resamples").**

- Code: The code uses 500 resamples and no multiple-comparison correction. The plan had 1,000 resamples and a Holm correction.
- Fix: Add this to Appendix E.

---

## 4. The most important weaknesses for an ARR reviewer, with fixes for tonight

All fixes below use text changes or offline analysis of the saved runs. None needs a large new experiment.

**W1 [MUST] No trivial baselines; the task difficulty drives calibration.**

- Why a reviewer cares: Without a majority baseline, readers cannot interpret accuracies of 30% to 55%. Below-chance accuracy forces a positive SCE on 2-option tasks (N5). The Civil Comments labels are degenerate for some attributes (N32). The plan had these baselines (B7).
- Fix tonight:
  1. Add majority and "always A" rows to Table 11, from the gold labels.
  2. Add the bound SCE ≥ 1/K − accuracy to Section 3.
  3. Report "within 1 level" accuracy and MAE for HelpSteer2 (already in `results/e2_metrics.csv`).
  4. Report the positive rate of each Civil Comments attribute. Drop `severe_toxicity` or use the soft labels.

**W2 [MUST] The paper does not follow its own pre-registration closely enough.**

- Why: ARR reviewers check pre-registered claims. H1 fails for classification (N4). H2 is partly supported (N7). H6 fails for Jev (N23). The E7 pilot ran at 18:06–18:10 UTC. The E7 note with the 2.7× result was last saved at 18:11:21 UTC, 10 seconds before the last save of the design file. Table 7 reuses those pilot runs, and the E2 comparison uses the same first 250 items of four tasks. Thus H7 is probably not a blind test. I could not check the git history (no git commands); please check it. The name-swap test (Fmt., Names) came after the main runs (overnight log), but it is the H2 headline in the abstract.
- Fix tonight: Add a hypothesis table (prediction, result, status, confirmatory or exploratory). Complete Appendix E with these changes:
  - added after the main runs: the name-swap test, Rev., Swap;
  - planned but not run (experiments): the implication test, baselines B4 to B7, the 2B and Gemma 4 31B pairs and the Clef pairs;
  - planned but not run (analyses): the threshold-transfer matrix, the metric replication, the direct-API check (E0) and the Haiku cascade;
  - changed: 2 option orders instead of 3; thresholds by cross-fitting instead of a development split; 500 resamples instead of 1,000; no Holm correction;
  - timing: the E7 pilot ran before the hypotheses were final.

**W3 [MUST] Weak statistics.**

- Why: Most headline numbers have no interval: the E5 gains, the E6 accuracy and cost, the E7 ratio and the S1-against-S2 accuracy claims. The "within 1 point" rule is smaller than the sampling error of a 500-item paired difference. Table 4 compares 16 systems × 12 perturbations with the floor and uses no correction.
- Fix tonight: Use the bootstrap code that you have (resample items, keep all questions of an item together). Apply it to the E5 gains, the cascade accuracy and cost ratio, and the E7 ratio. Mark the cells of Table 4 that are not above the matched floor after a Holm correction.

**W4 [SHOULD] The S2 baselines are not fully fair.**

- Why: The S2 models get one prompt and stated probabilities only. They have no repeats (no floor) and no E3 runs. "Reasoning effort low" did not give equal reasoning (M9). Dropped refusals can raise their accuracy (M8).
- Fix tonight: Report the reasoning tokens and the failures of each model on each task. Add a sensitivity check that counts failures as wrong. State that all robustness claims apply to S1 models only. Optional small run: repeat the 5 S2 models once on 200 items of 2 tasks to get a floor.

**W5 [SHOULD] The backbone comparison (E5, H4) cannot test the size effect.**

- Why: The three sizes come from three vendors with different recipes, so size and recipe are confounded. The letter readout has no temperature, and its prompt differs from the native card. The 4B pair is partial (N3).
- Fix tonight: Let the 4B run finish, or remove 4B from the headline. Write "Three vendor pairs cannot test H4", not "H4 not supported". Fit one temperature for each backbone on half of the items, and report the ECE on the other half. The readout logs store the letter probabilities, so this is offline.

**W6 [SHOULD] Task choice and task groups.**

- Why: Two groups have one task each. The judging group has an adversarial task (filtered with a Qwen judge) and an exact-match task with 5 levels. Most S1 models use Qwen backbones.
- Fix tonight: Report the results for each task in the main text. Use the groups only as a summary. Report H1 also without JudgeBench-Pro.

**W7 [SHOULD] The cascade design changes the conclusion.**

- Why: The item-level rule (M1), the trivial wins (N9) and the selection target (M2) change the H5 result. The paper has no latency analysis for cascades. The plan also had a cascade on the confidence of Claude Haiku 5.5.
- Fix tonight: Describe the rule. Exclude the trivial cases. Add a question-level variant for Civil Comments and HelpSteer2 (offline). Add bootstrap intervals.

**W8 [SHOULD] Time, route and version validity.**

- Why: All data come from one 2-hour window at night through one router. Two systems report no version. The E0 plan to compare the direct APIs did not happen.
- Fix tonight: Correct the text (N1, N2, N25, M6). Add the time of day to the Limitations.

**W9 [SHOULD] The comparisons with other papers select agreeing numbers.**

- Why: See R1, R2, R3 and N28. A reviewer who wrote one of these papers will see this at once.
- Fix tonight: Add one appendix table with these columns: study, Jev version, task, metric, their value, our closest value, protocol difference. Use "close" or "same" only when the protocols match.

**W10 [SHOULD] The metric definitions are not consistent.**

- Why: "Calibration error" means SCE in the abstract and ECE in the intro. Jev has two floors (1.9% and 2.3%, N18). Table 3 mixes pooled and task-mean columns (M7). The follow-rate denominator is not defined (M3).
- Fix tonight: Add one definitions paragraph. Write "task mean" or "pooled" in each caption.

**W11 [SHOULD] The contamination control is weaker than stated.**

- Why: The arXiv task is easy (85% to 91% for most systems) and balanced over categories. Some systems came out inside the window (N19). Category prediction does not need memory of the paper.
- Fix tonight: Soften the claim, and give the release dates against the window.

**W12 [SHOULD] Terms of service.**

- Why: The plan made the terms check a gate of E0 ("continue if ... their terms permit publication"). The Ethics statement says that the check is still to do. The paper publishes vendor outputs.
- Fix tonight: Do the check for the 13 vendors, and report the result in the Ethics statement.

---

## 5. ASD-STE100 check: the 15 worst violations

The text uses the active voice almost everywhere (I found one passive: "were submitted", l. 238–239). The main problems are long sentences, long paragraphs, synonyms and a few idioms.

**S1 [SHOULD] Abstract, l. 009–014. Rule 1 (42 words).**

- Original: "We test 19 System One models from 13 vendors, 6 configurations of general language models and 3 open backbones under one protocol: 8 tasks with 500 items each, the same question cards, 12 perturbations and a noise floor from repeated identical calls."
- Rewrite: "We test 19 System One models from 13 vendors, 6 configurations of System Two models and 3 open backbones. All systems get the same 8 tasks with 500 items each and the same question cards. We apply 12 perturbations and repeat identical calls to measure a noise floor."

**S2 [MUST] Abstract, l. 014–020. Rules 1 and 6 (37 words; "calibration error" here means SCE, but l. 061 uses it for ECE).**

- Original: "Calibration depends more on the task than on the system: all 17 System One models that accept our judging tasks are overconfident on them, but their median calibration error on yes/no knowledge questions is only +2.0 points."
- Rewrite: "Calibration depends more on the task than on the system. On the judging tasks, all 17 System One models that accept them are overconfident. But on BoolQ, their median signed calibration error (SCE) is only +2.0 points."

**S3 [SHOULD] Intro, l. 062–069. Rules 1 and 11 (44 words).**

- Original: "For sensitivity to wording, the reports go from 1.3% of changed decisions (vendor-reported, …) to 50.5 changes per hundred decisions when the names "yes" and "no" change places (…), and to 61.4% of correct decisions changed by an adversarial search for natural context (…)."
- Rewrite: "The reports about sensitivity to wording also disagree:
  - Microsoft reports that 1.3% of the decisions of its model change (vendor-reported).
  - Rafe and Das report 50.5 changed decisions per hundred when the names "yes" and "no" change places.
  - In JevOut, an adversarial search moves 61.4% of the correct decisions to a fixed wrong option."

**S4 [SHOULD] Intro, l. 073–078. Rules 1, 6 and 11 (39 words).**

- Original: "We send the same 8 tasks (500 items each), with the same question cards, to 19 System One models from 13 vendors, to 5 general LLMs with structured outputs, and to the open backbones of three System One models."
- Rewrite: "We send the same 8 tasks (500 items each) and the same question cards to three groups:
  - 19 System One models from 13 vendors;
  - 5 System Two models with structured outputs;
  - the open backbones of 3 System One models."

**S5 [SHOULD] Section 2, l. 158–166. Rule 11 (41 words in one list sentence).**

- Original: "Other studies examine the order of options (…), the coherence of probabilities across linked questions (…), adversarial attacks (…), thresholds with a controlled risk (…), open decision models from LLM readouts (…), and escalation from Jev to an LLM judge (…)."
- Rewrite: "Other studies examine the order of options, adversarial attacks and thresholds with a controlled risk. Others examine the coherence of probabilities across linked questions, open decision models from LLM readouts and escalation from Jev to an LLM judge." (Keep each citation next to its topic.)

**S6 [SHOULD] Section 2, paragraph "Evaluations of System One models", l. 138–172. Rule 10 (15 sentences).**

- Fix: Make three paragraphs:
  1. Calibration metrics: Deußer et al., Rafe and Das, TypedBench (5 sentences).
  2. Definitions of a changed answer: Rafe and Das, JevOut (3 sentences).
  3. Other studies and our contribution (4 sentences).

**S7 [SHOULD] Section 4.1, second paragraph, l. 296–319. Rules 10 and 11 (13 sentences).**

- Fix: Make a numbered list:
  "Three design choices decide the result of a study:
  1. The tasks. (2 or 3 sentences.)
  2. The probability that a study bins. (3 sentences.)
  3. The summary statistic. (2 sentences.)"

**S8 [SHOULD] Section 3, paragraph "Tasks", l. 228–243. Rule 10 (13 sentences).**

- Fix: Make one paragraph about the sampling (3 sentences). Then give one list item for each task with special rules: Civil Comments, HelpSteer2, Aegis 2.0, JudgeBench-Pro, arXiv and When2Call.

**S9 [MUST] Rule 6: three terms for one thing.**

- Original: "general language models" (l. 010), "general LLMs with structured outputs" (l. 076) and "System Two models" (l. 210 and later).
- Fix: Define "System Two model" once in the intro: "a general LLM that writes a probability for each option in JSON". Then use only this term.

**S10 [SHOULD] Rule 6: four terms for a changed answer.**

- Original: "changed decisions" and "changes per hundred decisions" (l. 063–066), "flips per hundred decisions" (Table 1, l. 350), "changed top answers" (Table 4), "changes … of the answers" (Section 4.6).
- Fix: Define "flip" once ("a change of the top answer"), and use it everywhere. Use "answer" for our data, and keep "decision" only in quotes from other papers.

**S11 [SHOULD] Rule 6: "cases" has two meanings.**

- Original: "in 98.1% of the cases" (l. 345–349; l. 382) means answers. "In 69 of 168 cases" (l. 435–445) means combinations of first stage, second stage and task.
- Fix: Write "of the answers" for the first meaning, and "settings" for the second meaning.

**S12 [SHOULD] Rules 8 and 12: idioms, metaphors and complex words.**

- "came to market" (l. 005–006) → "were released".
- "can fall for injected biases" (l. 188–189) → "Injected biases can mislead LLM judges."
- "the names win more often" (l. 353–354) → "the systems follow the names more often".
- "three choices decide the verdict of a study" (l. 297–298) → "three design choices decide the result of a study".
- "The changes concentrate on the judging tasks" (l. 460–461) → "Most changes are on the judging tasks."

**S13 [SHOULD] Section 4.5, l. 438–441. Rule 1 (33 words).**

- Original: "In the other 99 cases, the cascade keeps the accuracy of the System Two model within 1 point in 79, but at less than half of its cost in only 28 (Table 6)."
- Rewrite: "In the other 99 settings, the cascade keeps the accuracy of the System Two model within 1 point in 79. Only 28 of them cost less than half of the System Two model alone (Table 6)." (After fix N9, the numbers become 90, 70 and 20.)

**S14 [SHOULD] Section 4.4, l. 421–425 (34 words), and Section 4.6, l. 463–468 (31 words). Rule 1.**

- Rewrite 4.4: "The OpenAI Decisions API and GPT-6 Luna without reasoning have a similar mean accuracy (70.3% and 71.3%). GPT-6 Luna with low reasoning is 8.3 points more accurate."
- Rewrite 4.6: "The OpenRouter alias `~typesafe/jev-latest` already pointed to 1.14 on 15 October, one day after the release. The TypeSafe documentation said that its own alias would move in about one week."

**S15 [SHOULD] Section 5, Recommendations 2, 3 and 5 (l. 474–479, 483–485). Rule 2 (one instruction in one sentence).**

- "Pin the version, and record the resolved version string of each response." → "Pin the version. Record the resolved version of each response."
- "Calibrate and choose thresholds for each task." → "Calibrate each task separately. Choose a threshold for each task."
- "Ask related questions in separate calls when the system does not isolate them, or test this first." → "Test if your system isolates the questions of one call. If it does not, ask related questions in separate calls."

Note about force: The rewrites keep the force of the text with two small exceptions.

- S2: Two sentences make the contrast weaker. Keep "But" at the start of the second sentence.
- The title is a question ("Same Question, Different Answer?"). STE prefers a statement. But the question form has rhetorical force, and a title is not body text. I recommend that you keep it, unless the venue style says otherwise.

---

## 6. Layout problems in the PDF

**L1 [SHOULD] Page count and page 8.**

- The PDF now has 16 pages. The main text (through Section 5) ends on page 8, so the limit holds. But each fix in this review adds text. Keep a budget: the main text has no free lines on page 8.

**L2 [SHOULD] Appendix order (pages 11–16).**

- The text of Appendix E (l. 763–777) is on page 11. The floats of Appendices B, C and D (Tables 9 to 14, Figure 3) come after it, on pages 12 to 16. Appendices B and C have only one sentence each, and their tables appear 1 to 2 pages later.
- Fix: Put `\clearpage` before Appendix E, or move Appendix E before Appendix A. Merge the short texts of B, C and D into one section with the tables.

**L3 [SHOULD] Figure 1 (heatmap), page 6.**

- The cell numbers are about 4.6 pt, and the axis labels are 5.6 to 6 pt. They are too small to read on paper. The figure is on page 6, two pages after its first mention (page 4).
- Empty cells ("not accepted") and cells with SCE near 0 are both white in the `RdBu_r` map.
- Fix: Use fonts of 7 pt or more, or move the numbers to an appendix table. Use grey hatching for "not accepted", and say this in the caption.

**L4 [COULD] Float order in Section 4.**

- Table 3 (page 5), Figure 1 (page 6) and Table 4 (page 7) are full-width floats. LaTeX puts each one at the top of the next free page. Thus Table 4 appears 3 pages after its first mention (page 4).
- Fix: Put `\input{tables/e3_flips}` before Section 4.1 in the source, or move Figure 1 to the appendix.

**L5 [COULD] White space.**

- Page 5, left column: a gap of about 3 lines between l. 341 and l. 342 (the log shows `Underfull \vbox` warnings).
- Page 12: Table 9 is alone in the middle of a float page.
- Page 15: a large gap between Table 12 and Table 13.
- Page 16: Table 14 (one column wide) is alone on the page.
- Fix: Use `\raggedbottom` in the appendix. Make Table 14 a `table*`, or put Tables 13 and 14 on one page.

**L6 [COULD] Overfull box in Table 10.**

- The log shows `Overfull \hbox (1.93994pt too wide)` for `correct_response` in the 1.9 cm column (JudgeBench-Pro row).
- Fix: Make the column 2.1 cm wide, or allow a break after the underscore.

**L7 [MUST] Captions that disagree with the content.**

- Table 3: "(500 items each)" and "the tasks that the system accepts" are wrong for the 4B row (N3).
- Table 5: "macro mean over the 7 tasks" is wrong for the 4B row (N3).
- Table 7: the row labels "minutes apart" and "next day" are wrong (N1, N2).
- Table 13: "–" for 4B means "not run yet", but Table 11 defines "–" as "not accepted".
- Fix: Correct the captions and labels.

**L8 [SHOULD] Other captions.**

- Table 4: "All changes are measured against the original request (E2)" disagrees with "Names: … against Fmt." in the same caption. Write "All other changes …". The sentence is also passive (rule 3): "We compare all other changes with the original request (E2)."
- Table 6: the brackets show the median over the kept tasks only (`make_paper_assets.py`). Say this.
- Table 8: "The date suffix of the resolved version identifies the model version" is not true for Clef and Clef-flash (no suffix).
- Figure 2: the curves are in-sample threshold sweeps over all items, not the cross-fitted result. Say this.
- Figure 3 uses 10 bins, but the ECE uses 15 bins. Say why, or use 15.

**L9 [COULD] Abstract length.**

- The abstract has about 220 words. Many ACL venues ask for 200 words or fewer. Check the current ARR guide.

---

## Appendix Z. Evidence that I computed

All values come from `runs/` and `results/` on 2026-10-16 between 01:30 and 01:45 IST.

**Z1. Run times (UTC, 15 October 2026).**

| Run | Start | End |
|---|---|---|
| E0 | 18:02 | 18:04 |
| E7 pilot (`e7-2026-10-15`) | 18:06 | 18:08 |
| E7 repeat (not used in the paper) | 18:09 | 18:10 |
| E2 (most systems) | 18:42 | 18:51 |
| B1 (S2 models) | 18:57 | 19:32 |
| E3 | 19:00 | 19:37 |
| E3 names | 19:36 | 19:38 |
| B2 local (4B) | 19:26 | still running at 20:02 |

- Gap "next day" 1.14: 34.7 to 35.6 minutes (median 35.1). Gap "next day" 1.13: 35.8 to 36.6 minutes (median 36.2).
- Gap "noise" (E2 call to E3 repeats): 18.0 to 29.8 minutes (median 24.1).
- `notes/2026-10-15-e7-first-results.md` saved 23:41:21 IST; `notes/2026-10-15-experiment-design.md` saved 23:41:31 IST (18:11 UTC).

**Z2. Majority baselines (gold labels of the 500 items; majority for each question).**

| Task | Majority accuracy (%) |
|---|---|
| BoolQ | 61.4 |
| Aegis 2.0 | 53.6 |
| Civil Comments | 86.1 |
| BANKING77 | 2.2 |
| arXiv | 6.6 |
| When2Call | 40.2 |
| JudgeBench-Pro ("always A") | 56.0 |
| HelpSteer2 | 54.9 |

- JudgeBench-Pro: 14 of 17 S1 models are below 50%. All 17 are below 56.0%.
- HelpSteer2: only Liquid d1 (59.5%) is above 54.9%.
- Civil Comments: 5 of 19 S1 models are above 86.1% (Microsoft-Decision-1, pplx-decider, Drex v2.0, Classify-1, Liquid d1).

**Z3. Civil Comments, positive rate at the 0.5 rule (500 items).**

- toxicity 50.0%, insult 36.8%, identity_attack 4.8%, obscene 3.6%, sexual_explicit 1.2%, threat 0.6%, severe_toxicity 0.0%.
- Jev 1.14 says "yes" to severe_toxicity on 51.2% of the items. This attribute gives 36% of the Jev errors on the task.
- Jev 1.14 without severe_toxicity: accuracy 84.8% (79.7% with it), top-label ECE 1.9 (5.6), SCE +1.8 (+5.6), class-wise ECE 15.6 (20.5).

**Z4. Confidence AUROC (right against wrong answers), 7 cascade first stages.**

- Civil Comments 0.77 to 0.90; JudgeBench-Pro 0.30 to 0.61; When2Call 0.71 to 0.81; arXiv 0.77 to 0.89; BANKING77 0.83 to 0.91.

**Z5. Cascades (`results/e6_cascade.csv`).**

- 99 settings where the S2 model is better: 9 by at most 1 point (all kept, 8 cheap) and 90 by more than 1 point (70 kept, 20 cheap).
- The 28 cheap settings: When2Call 9, arXiv 7, BANKING77 6, BoolQ 3, Civil Comments 2, Aegis 2.0 1.
- Escalation on Civil Comments: 13% to 100% (median 79%). On JudgeBench-Pro: 95% to 100%, cost ratio 0.96 to 1.12.

**Z6. Reasoning tokens of the S2 models (`runs/b1/`).**

| Model | Calls with reasoning tokens > 0 | Mean reasoning tokens | JudgeBench-Pro: calls with reasoning |
|---|---|---|---|
| GPT-6 Sol | 38% | 20 | 93% |
| Claude Sonnet 5.5 | 5% | 59 | 44% |
| Gemini 3.8 Flash | 58% | 123 | 86% |
| Claude Haiku 5.5 | 18% | 89 | 98% |
| GPT-6 Luna (low) | 82% | 79 | 100% |
| GPT-6 Luna (none) | 0% | 0 | 0% |

- Claude Sonnet 5.5 on JudgeBench-Pro: 181 of 204 correct with reasoning tokens (88.7%); 189 of 258 correct without (73.3%).
- GPT-6 Luna without reasoning: p50 1,744 ms and USD 0.122 for 1,000 calls (Table 3 shows "–"). OpenAI Decisions: 295 ms and USD 0.085.

**Z7. Failed or dropped items.**

- Claude Sonnet 5.5: 50 `parse_error` (JudgeBench-Pro 37, Aegis 2.0 9, HelpSteer2 4). Claude Haiku 5.5: 4. Gemini 3.8 Flash: 2.
- OpenAI Decisions: 6 items never answered (JudgeBench-Pro 5, BANKING77 1). Two items got an answer on the 4th attempt.
- Tev1-4B: 16 items never answered (JudgeBench-Pro 14, with answers "C", "D" or "H"; Aegis 2.0 2).

**Z8. Two-way variance splits (main effects only).**

- E2, SCE, 14 S1 models × 8 tasks: task 64%, system 14%, residual 22%. Without JudgeBench-Pro and HelpSteer2: task 32%, system 39%.
- E3, Table 4, 14 systems × 12 perturbations: perturbation 57%, system 7%, residual (interaction) 35%.

**Z9. Hypothesis H1 by group (S1 models, 95% bootstrap interval).**

- Classification: 16 of 19 above 0, 3 below 0.
- Knowledge (BoolQ): 9 above 0, 8 contain 0, 2 below 0.
- Judging (pooled): 17 of 17 above 0. HelpSteer2 alone: Liquid d1 −2.3 and Kev-4B −2.6 (point estimates).

**Z10. E3 details for Jev 1.14.**

- Floors on the same tasks: Swap 5.2%, Rev. 4.6%, Neg. 0.6%, Crowd./Hint/Inject 1.6%, Alone 2.9%.
- Hint follow rate by task: Aegis 2.0 2.4%, BoolQ 4.5%, arXiv 21.7%, When2Call 26.0%, BANKING77 31.0%, JudgeBench-Pro 71.6%.

**Z11. E5 closed pair (Decisions API against GPT-6 Luna without reasoning).**

- Task differences: BoolQ +1.2, Aegis 2.0 −0.2, Civil Comments −8.3, arXiv +0.6, When2Call −0.4, JudgeBench-Pro +7.7, HelpSteer2 −8.2.
- Pooled difference about −5.3 points; pooled McNemar p = 1.8e-35.

**Z12. Calls and cost.**

- 388,153 records in `runs/` (the local 4B run still adds records). Total cost USD 62.27.
- S1 calls: 356,577 for USD 20.70. The Ethics statement (357,000 calls, USD 21, USD 62) is correct.
