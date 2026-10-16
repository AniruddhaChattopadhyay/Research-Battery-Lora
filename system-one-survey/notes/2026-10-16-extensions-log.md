# Extensions log (started 2026-10-16)

Aniruddha approved these extensions on 2026-10-16. He is travelling, and his network can stop. Thus, update this file after each step. A new session can resume from the "Status" table and the "Resume" section.

Rules (as for the overnight work): do not commit; never print the API key; use Python, not grep/cat/wc/tail, for filters (the rtk hook); use uv. All new results are exploratory, because they come after the main runs. The paper must say this.

## Scope

1. **Option order, direction of the flips.** No new calls. Use the saved E3 calls (order1, order2).
2. **Name test, true rubric second.** Add 2 variants (`namealign-second`, `nameswap-second`). Then the analysis can separate the name, the rubric and the position. Tag `e3-names-second`; 14 systems (the systems of `runs/e3-names`); BoolQ, Civil Comments and Aegis 2.0; 200 items.
3. **Clef 2,000-token limit.** The OpenRouter image guide says that Clef and Clef-flash read only about the first 2,000 text tokens. Test it: put the fact that decides the answer at different positions in a long input.
4. **Image pilot.** One test call for each of the 10 systems that list images. Then 50 Civil Comments items and 50 BANKING77 items, as text and as an image of the same text.
5. **Paper.** Add the results that hold. Mark them as exploratory. Keep the main text on 8 pages.

## Decision rule for the image pilot (written before the pilot)

- **Go** (full image run, and an "image" column in E3): at least 3 systems flip more than 5% of their answers between the text and the image of the same input (pooled over the 2 pilot tasks), or at least 3 systems change their accuracy by more than 5 points.
- **No go:** report the pilot in one sentence. Keep a full image study for the survey.
- The flip compares the image answer with the text answer **of the same run** (versions change, E7).

## Status

| Step | State | Where |
|---|---|---|
| 0. This log | Done | `notes/2026-10-16-extensions-log.md` |
| 1. Direction of order flips | Done (not yet in the paper) | `analysis/e3_position.py`, `results/e3_position.csv` |
| 2. Name test, true rubric second | Done (not yet in the paper) | `src/perturb.py`, `analysis/e3_names2x2.py`, `runs/e3-names-second/`, `results/e3_names2x2.csv` |
| 3. Clef length probe | Done: no limit found | `src/probe_length.py`, `analysis/probe_length.py`, `runs/probe-length/` |
| 4. Image pilot | Done: **go** (5 systems above 5% flips) | `src/render.py`, `src/run_image.py`, `analysis/e3_image.py`, `runs/image-pilot/` |
| 4b. Full image run | Running: 4 tasks (civil, aegis, banking77, boolq) x 200 items x 2 x 10 systems | `runs/e3-image/` |
| 4c. System Two image reference | Done (15:00 UTC): image flips 2.8% to 6.1%; text-to-text drift 1.3% to 3.9% (they are not deterministic) | `runs/e3-image-s2/`, `src/llm.py` (image states) |
| 4d. List-form control | Done: the same text as `["<text>"]` (Classify-1: `{"text": ...}`), 10 systems, tag `e3-image`, variant `list` | `src/run_image.py --variants list` |
| 5. Paper | **Done** (2026-10-16, about 15:10 UTC): final build, 23 pages, main text ends on page 8, no LaTeX errors, no undefined references | `paper1/main.pdf`, `paper1/main.tex`, `paper1/numbers.tex` |

**All steps are done.** Total of the added runs: 52,388 calls (with failed calls), USD 15.38 (`results/added_runs.csv`), 18:25 to 20:29 IST.

## Resume

- The runners skip the calls that already returned status 200. Thus, run the same command again after a network break.
- Set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`: the datasets are in the local cache, and this stops network checks.
- `src/retry.sh <passes> <command>` runs a command several times and waits for the network before each pass. Each run has logs in `runs/<tag>/logs/`.
- Check the progress: count the status-200 lines in `runs/<tag>/*.jsonl` (use Python).
- Commands (one background process for each model; `<m>` is the model id):
  - Step 2: `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src nohup bash src/retry.sh 4 uv run python src/run_e3.py --tag e3-names-second --n 200 --tasks boolq,civil,aegis --only namealign-second,nameswap-second --models <m> --workers 8`. Models: `runs/e3-names-second/models.txt`.
  - Step 3: `PYTHONPATH=src nohup bash src/retry.sh 3 uv run python src/probe_length.py --models <m> --sets base,long,array --workers 4`.
  - Step 4: `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src nohup bash src/retry.sh 3 uv run python src/run_image.py --tag image-pilot --tasks civil,banking77 --n 50 --models <m> --workers 4`. Models: `runs/image-pilot/models.txt`.

## Results

### Step 2: the new variants

- `namealign-second`: {"no": not true, "yes": true}. `nameswap-second`: {"yes": not true, "no": true}.
- Check: the 19,600 old E3 requests of Jev 1.14 are byte-identical to `perturb.variants()` after the change.
- Result (`uv run python analysis/e3_names2x2.py`; flips of the chosen rubric, macro over BoolQ, Civil, Aegis):

| System | Names, rubric first | Names, rubric second | Position, names match | Position, names swapped |
|---|---:|---:|---:|---:|
| Jev 1.14 | 98.1 | 98.7 | 0.6 | 1.8 |
| Microsoft-Decision-1 | 95.2 | 95.2 | 0.8 | 2.0 |
| Clef / Clef-flash / Clef-omni | 14.9 / 19.6 / 14.1 | the same | 0.0 | 0.0 |
| Liquid d1 | 0.3 | 0.5 | 3.2 | 3.4 |
| OpenAI Decisions | 76.3 | 77.0 | 5.2 | 29.9 |
| pplx-decider | 79.7 | 54.5 | 1.6 | 35.1 |
| Mercury Decide | 96.6 | 79.8 | 1.2 | 20.4 |
| Tev1-4B | 91.7 | 65.0 | 7.8 | 31.3 |
| Drex v1.5 / v2.0 | 34.4 / 64.6 | 11.5 / 49.6 | 8.2 / 11.4 | 32.2 / 24.0 |
| Kev-4B | 41.2 | 48.0 | 4.0 | 29.1 |
| Solar Decide Flash | 86.0 | 77.4 | 9.6 | 21.9 |

- Reading: 6 systems are stable (Jev, Microsoft, 3 Clef, Liquid d1). For the other 8, the position decides many answers when the names and the rubrics disagree (20.4% to 35.1%).
- **Correction (same day):** my first reading ("the first option wins") was wrong for most systems. The direction depends on the system. Position flips with swapped names, first both times : second both times (all answers; `first`, `second`, `p_sign` in `results/e3_names2x2.csv`): OpenAI 267:49 and pplx 564:101 (first wins); Kev-4B 136:545, Drex v1.5 230:419, Drex v2.0 49:331, Tev1-4B 50:542, Solar Decide Flash 94:142 (second wins); Mercury 70:101 (p = 0.02). The paper now says: the first option wins for 2 systems and the second for 5 (p < 0.01).
- Note: in the 2-option name test, Solar Decide Flash prefers the second option, but in the order test with 4 to 77 options it prefers the earlier option. The direction depends on the task format.

### Step 3: Clef length probe (base set)

- Exact sentence of the guide: "Workers AI reads roughly the first 2,000 tokens of text in `state` and drops the rest without an error." (bullet "Text length on Clef and Clef Flash").
- Base set: a plain text input of about 3,050 tokens (16,000 characters of arXiv abstracts); the fact at about 20 to 2,850 tokens; 120 items for each system.
- Result: all 6 systems (Clef, Clef-flash, Clef-omni, Jev 1.14, pplx-decider, Microsoft-Decision-1) are correct on 100% of the items at every position. Thus, the limit does not show for plain text inputs of this length. Our E2 inputs for Clef are at most 3,300 tokens.
- My characters-per-token estimate was too low (about 5.2 characters for each token, not 4.2). The analysis uses the real token counts.
- Long set (plain text, about 6,850 tokens; Clef models only): 100% at every position, up to about 6,670 tokens.
- Array set (the base inputs as `["<text>"]`, the form of the guide; Clef models only): 100% at every position, up to about 2,870 tokens.
- **Conclusion:** through the pinned Cloudflare provider, the Clef models read the full input in our range. The 2,000-token sentence of the guide does not apply to our runs (2026-10-16). No change to the E2 or E3 results is necessary. `results/probe_length.csv`.

### Step 1: direction of the order flips (no new calls)

`analysis/e3_position.py` -> `results/e3_position.csv`. Pooled over BANKING77, When2Call and arXiv:

| System | Items | Flips | Earlier | Later | Sign test p | Wrong answers in first place, W2C (chance 25%) |
|---|---:|---:|---:|---:|---:|---:|
| Solar Decide Flash | 400 | 105 | 73 | 2 | 1.5e-19 | 40.7% |
| OpenAI Decisions | 598 | 69 | 40 | 5 | 7.9e-08 | 33.3% |
| Tev1-4B | 400 | 38 | 19 | 8 | 0.052 | 26.9% |
| Jev 1.14 | 600 | 19 | 6 | 4 | 0.75 | 24.1% |
| Clef models | 600 | 0 | 0 | 0 | - | about 25% |

The other systems: no clear direction (p >= 0.25).

### Step 4: image test calls (2026-10-16)

- Format: `state` is an array; an image is `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`. Our image state has only the image.
- Clef, Clef-flash, Clef-omni, OpenAI Decisions, pplx-decider, Mercury Decide, Liquid d1: accepted the image and answered correctly.
- Jev 1.13 and 1.14 rejected 1,000 x 256 pixels ("Invalid image dimensions") but accepted 1,000 x 300 and more. Thus, `src/render.py` makes each image at least 400 pixels high.
- Classify-1 rejects `image_url` parts ("ScaleDown does not accept "image_url" parts in state"). It accepts `{"document": "<raw base64 PNG>"}` and reads it with OCR. Other forms gave 422 ("OCR processing failed" or "'state.text' or 'state.document' must be provided"). `src/run_image.py` uses the document form for ScaleDown.
- Rendering: DejaVu Sans, 24 pixels, 1,000 pixels wide. Of the first 200 items, these render: Civil Comments 200, BANKING77 200, BoolQ 199 (Korean characters), Aegis 2.0 195 (emoji, Chinese).

### Step 4: image pilot result (2026-10-16)

`uv run python analysis/e3_image.py --tag image-pilot` -> `results/image-pilot.csv`. 50 Civil Comments items (350 answers) and 50 BANKING77 items for each system. Flip = text against image in the same run. Drift = today's text against E2.

| System | Pooled flip | Accuracy change (points) | Drift |
|---|---:|---:|---:|
| Clef-omni | 13.5% | -1.1 | 0% |
| OpenAI Decisions | 10.1% | +2.5 | 0% |
| Jev 1.14 | 7.0% | +2.9 | 2% |
| Jev 1.13 | 5.5% | -0.1 | 0.6 to 2% |
| Classify-1 (OCR) | 5.2% (BANKING77 16%) | -3.1 | 0% |
| Mercury Decide | 4.5% | +0.6 | 0% |
| Clef | 4.0% | 0.0 | 0% |
| Liquid d1 | 3.5% | -1.7 | 0.6 to 2% |
| Clef-flash | 2.8% | -0.4 | 0% |
| pplx-decider | 2.2% | -1.0 | 0% |

### Step 5: paper changes (2026-10-16)

- Main text: Section 2 (Lan et al. on images), Section 3 (the added tests, exploratory), Section 4.2 (names with the rubric second; direction of order flips; new paragraph "Image of the input"), Table 5 (new "Image" column), Recommendation 3, Limitations (image test; System Two image test), Ethics (cost of the added tests).
- Appendix: Deviations ("Added on 16 October 2026"), new section "Added tests (exploratory)" with Tables app-position, app-names, app-image and the length probe.
- To keep 8 pages, these details moved from the main text to the appendix (no content lost): E5 detail per task, Civil Comments cascade detail, E7 detail per task, the jev-latest alias sentence, the System Two BANKING77 list form and reasoning notes, the time of the E7 check runs. Cut from the main text: the Li et al. negation comparison (it stays in Table 7), the Solar Decide Flash hint sentence (Table 5 shows it), the JudgeBench-Pro cascade sentence (the AUROC shows it). The intro list of system groups is now inline. The Table 5 caption is shorter.
- Rebuild: `uv run python analysis/e3_image.py --tag e3-image; uv run python analysis/e3_image.py --tag e3-image-s2 --ref b1; uv run python analysis/added_runs.py; uv run python analysis/make_paper_assets.py; cd paper1 && latexmk -pdf -outdir=tmp -interaction=nonstopmode main.tex && cp tmp/main.pdf main.pdf`.
- Not changed (still waiting for Aniruddha's yes from the earlier session): the "2-hour window" sentence in Section 3 and the reason for 16 of 19 models in E3.
- Not in the abstract or the list of findings in the introduction: the image test and the position results. The main text has no free line. Aniruddha decides if one of them replaces other text there.
- Later cuts for the 8-page limit (all moved to the appendix or to tables): the comparison with Rafe and Das is shorter (Table 7 has it), the Huang et al. comparison gives 2 systems (not 4), the E7 timing note points to Appendix A, and the closed-pair sentence of E5 has no numbers (Table 6 has them).

### Step 4: full image run and the list-form control (2026-10-16)

`uv run python analysis/e3_image.py --tag e3-image` -> `results/e3-image.csv`, `results/e3-image_summary.csv`. 200 items each of Civil, Aegis, BANKING77, BoolQ. Macro flips (%):

| System | Image | List (control) | Drift (vs E2) | E3 floor |
|---|---:|---:|---:|---:|
| Jev 1.13 | 3.3 | 0.8 | 0.4 | - |
| Jev 1.14 | 3.4 | 1.0 | 0.6 | 0.8 |
| OpenAI Decisions | 7.8 | 0.0 | 0.0 | 1.0 |
| pplx-decider | 2.8 | 1.5 | 0.0 | 0.0 |
| Clef | 4.3 | 1.8 | 0.0 | 0.0 |
| Clef-flash | 5.4 | 1.7 | 0.0 | 0.0 |
| Clef-omni | 7.4 | 1.4 | 0.0 | 0.0 |
| Mercury Decide | 5.3 | 2.7 | 0.1 | 0.1 |
| Liquid d1 | 4.7 | 2.8 | 0.7 | 0.8 |
| Classify-1 (OCR) | 7.8 | 0.2 | 0.5 | - |

- All 8 systems of E3 are above their floor. Accuracy changes by at most 1.5 points.
- **The list form alone changes answers** (0.0% to 2.8%), also for deterministic systems. Thus, part of the image effect comes from the form of the state. The image flips more than the list for every system. The paper says this in the main text.
- System Two (partial at 13:25 UTC): image 2.3% to 7.4%; text-to-text drift 1.4% to 4.1%.
- System Two final (`results/e3-image-s2_summary.csv`): GPT-6 Sol 2.8%, Claude Sonnet 5.5 4.6%, Gemini 3.8 Flash 5.4%, Claude Haiku 5.5 6.1%, GPT-6 Luna 5.9%. Two Aegis items get no content from Claude Sonnet 5.5, as text and as an image (as in B1); the analysis leaves them out.
- Network break from about 14:42 to 14:55 UTC. `src/retry.sh` waited ("no network, wait 60 s") and started pass 2 at 14:55. Pass 2 re-sends the 207 Gemini calls that failed in the break. Everything else was complete before the break.

Decision: 5 systems flip more than 5% -> **go** (the rule above). Accuracy changes are small; the flips go both ways. Full run: tag `e3-image`, the same command as the pilot with `--tag e3-image --tasks civil,aegis,banking77,boolq --n 200`.
