# PageIndex on CLEVRER: pilot results (brief for the meeting of 2026-10-12)

Slides: `slides/pilot-results.pdf` (12 slides). Figures and numbers: `results/`. Earlier analysis: `notes/2026-10-10-pageindex-clevrer-analysis.md`.

---

## 1. Summary

1. We ran the real PageIndex SDK on CLEVRER for the first time. We compared it with the same chat model in two other set-ups: "full context" (the model gets all pages of the right run) and a small proof of concept of the dual-view graph design.
2. **PageIndex is less accurate.** Overall, PageIndex got 87.5%, full context 95.2% and our graph agent 93.1% per question. The largest gap is on counterfactual questions: 70.9% for PageIndex, against 85.5% for both other systems.
3. **PageIndex is also the most expensive and the slowest.** It used 110,010 input tokens for each question. That is 6 times full context and 38 times our graph agent.
4. **The cause is how the evidence reaches the model, not its reasoning.** PageIndex always finds the right run. Then it reads too little (4 errors), or it reads the evidence and still answers wrong next to its large tree of summaries (11 errors). With only the pages of the run, the same model answers 95.5% of predictive and 85.5% of counterfactual questions.
5. **Our two operators work.** Intervention and world simulation answer 95.2% of predictive and 88.4% of counterfactual questions on 3,000 held-out videos, with no LLM. A causal graph of the observed events alone gets 29.6% and 52.1%.

---

## 2. What we ran

| Item | Setting |
|---|---|
| Data | 30 random CLEVRER validation runs. Ground-truth annotations, so perception is perfect. |
| Document | One 317-page "plant log" PDF with one chapter for each run: overview, objects, event log, and the state of every object in every frame. Real headings and PDF bookmarks give PageIndex its best case. |
| Questions | 248: all multiple-choice questions of the 30 runs (51 explanatory, 22 predictive, 55 counterfactual) and 4 random descriptive questions for each run (120). Each question names its run. |
| Chat model | `gpt-5.6-sol`, reasoning effort "medium", through OpenRouter. The same model, instructions and output format for every system. |
| PageIndex | SDK v0.2.22, local mode, Flash index built with `gpt-5.6-luna`. This is the setup that the PageIndex README recommends. Agent turn limit: 16 (no question hit it). |
| Full context | The model gets all pages of the named run (text from the same PDF) and nothing else. This is perfect retrieval. |
| Graph agent (ours) | The model gets six tools over a graph of each run: objects, states at a frame, event log, causal past of an event, `predict_future` (world simulation) and `remove_object` (intervention plus world simulation). Code builds the graph directly from the annotation files, which hold the same facts as the PDF. |
| Metric | CLEVRER per-question accuracy: all options of a question must be right. |

---

## 3. Results

### 3.1 Accuracy (per question, with 95% Wilson intervals)

| Type | n | PageIndex | Full context | Graph agent (ours) |
|---|---:|---|---|---|
| Descriptive | 120 | 89.2% [82.3, 93.6] | 97.5% [92.9, 99.1] | 95.0% [89.5, 97.7] |
| Explanatory | 51 | 98.0% [89.7, 99.7] | 100% [93.0, 100] | 98.0% [89.7, 99.7] |
| Predictive | 22 | 95.5% [78.2, 99.2] | 95.5% [78.2, 99.2] | 90.9% [72.2, 97.5] |
| Counterfactual | 55 | 70.9% [57.9, 81.2] | 85.5% [73.8, 92.4] | 85.5% [73.8, 92.4] |
| **All** | 248 | **87.5%** [82.8, 91.1] | **95.2%** [91.7, 97.2] | **93.1%** [89.3, 95.7] |

Paired on the same questions:

- Full context was right and PageIndex wrong on 20 questions. The reverse happened on 1 question (McNemar exact p = 2 × 10⁻⁵).
- Our graph agent was right and PageIndex wrong on 20 questions. The reverse happened on 6 questions (p = 0.009).

### 3.2 Cost and time (per question)

| System | Input tokens | Output tokens | Cost (list price) | Time |
|---|---:|---:|---:|---:|
| PageIndex | 110,010 | 404 | $0.224 | 12.1 s |
| Full context | 17,649 | 224 | $0.038 | 4.8 s |
| Graph agent (ours) | 2,872 | 136 | $0.007 | 5.4 s |

- The costs use the OpenRouter list price of `gpt-5.6-sol` ($2 for each million input tokens, $10 for each million output tokens). Prompt caching made the real bill lower.
- The real total spend for all pilot work today (indexing, three full runs and tests) was about $47.4. OpenRouter posts some costs with a delay.
- The tree has 361 nodes and about 90,000 tokens. The tree parts that the agent reads stay in its conversation, so the model reads them again in every turn. One run report has only about 17,000 tokens.

---

## 4. Why PageIndex fails

### 4.1 It finds the right run, then reads only part of it

- PageIndex never read a page of a wrong run (0 of 248 questions). The table of contents works at chapter level, as on FinanceBench, where each question names the company and the year.
- On average, it read 1.3 pages for a descriptive question, 0.9 for an explanatory, 2.4 for a predictive and 7.2 for a counterfactual question. One run has 10.5 pages.
- 10% of the answers came from the tree summaries alone, with no page read.
- The tool description tells the agent: "Use tight, targeted page ranges — never the whole document at once."

The 31 PageIndex errors, by what the agent read:

| What PageIndex read | Errors | Full context right on the same question |
|---|---:|---:|
| One page only | 13 | 11 |
| Some pages of the run | 15 | 6 |
| All pages of the run | 3 | 3 |

Full context was also wrong on 11 of these 31 questions, so they are hard questions. The other 20 errors are PageIndex-only errors. We checked where their evidence was:

| PageIndex-only errors (20) | Errors | Examples |
|---|---:|---|
| Evidence not read: the question needed state tables on pages that PageIndex did not read | 4 | "How many moving spheres are there?" |
| Evidence read, still wrong: the answer was on a page that PageIndex read (the event log or frame 0) | 8 | "How many rubber objects enter the scene?" |
| Whole run read, still wrong | 3 | Counterfactual questions |
| Counterfactual, part of the run read: the cause is not clear | 5 | In one case (run 12995), its reasoning said that an option was true, but its answer said false. |

So partial reading is real, but the more common failure is a wrong answer with the evidence in hand (11 of 20). The same model with only the run's pages got all 11 right. The difference is the agent context: the tree text of all 30 runs (tens of thousands of tokens of lossy summaries) and several agent turns.

### 4.2 The tree is a table of contents, not a model of events

We built the PDF with clean, numbered headings and bookmarks. The PageIndex tree still had these defects:

- **Merged headings.** One node has the title `Run 10307: scene report 10307.1 Overview`.
- **Wrong nesting.** All state tables became children of `Event log`. A table header (`Frame Event`) became a node.
- **Lossy summaries.** One summary says that the frames "contain object identities only". The pages hold positions, speeds and states.
- **Overlapping page ranges.** One page holds parts of two time windows, so a question about one time needs more than one node.
- **Evidence in hand, still wrong.** In 11 errors, PageIndex had read the evidence. Full context got all 11 right. The likely cause is the large tree of summaries in the context of the agent.

### 4.3 Two examples, one of each kind

- **Run 10572, descriptive:** "How many moving spheres are there?" The correct answer is 2, because the spheres move later in the video. PageIndex read page 57 only (frames 0 to 15) and answered 0. Full context and the graph agent answered 2.
- **Run 10475, descriptive (evidence read, still wrong):** "How many rubber objects enter the scene?" The correct answer is 2. PageIndex read page 24, whose event log shows the two rubber objects that enter. It answered 3. Full context answered 2. The graph agent made one `event_log` call and answered 2.

---

## 5. A correction to our hypothesis

- **We expected:** predictive and counterfactual answers are in no document, so PageIndex fails on these questions.
- **The pilot shows:** with all state tables of the run, the model does the physics itself. Full context got 95.5% on predictive and 85.5% on counterfactual questions.
- **This part is still true:** retrieval alone cannot do it. 100% of the correct predictive answers and the new events of 44.8% of the counterfactual questions are not in the video. Something must compute them: the LLM from the raw numbers, or an operator.
- **Revised claim:** tree retrieval does not deliver clean, complete evidence for causal and counterfactual answers, and it pays a large token cost for its index.

This revision is good for the paper. A claim of the form "LLMs cannot reason counterfactually" is weak and goes out of date fast. A claim about evidence access is measurable and stays true when models improve.

---

## 6. Our operators, with no LLM

Measured on 3,000 held-out validation videos (12000 to 14999) with `analysis/eval_operators.py`. Two constants (contact distance and time horizon) were tuned on videos 10000 to 11999.

| Method | Predictive per question | Counterfactual per question |
|---|---:|---:|
| Causal graph of observed events only | 29.6% (n = 2,138) | 52.1% (n = 5,597) |
| + intervention + world simulation | **95.2%** | **88.4%** |

- **Intervention:** remove object X, and find every object whose motion depends on X and the frame where it changes.
- **World simulation:** move objects on straight lines with the measured friction (calibrated from the data: 0.0104 speed units per frame), and detect contacts (center distance below 0.42).
- For context: the best published CLEVRER counterfactual score that we found is 90.7% per question (CRCG, WACV 2024, test split). That system perceives the scene from video. Ours uses ground-truth perception, so this is not a direct comparison.

---

## 7. What this means, and the limits

**Support for the dual-view design:**

- The data graph gives exact state lookups at any frame.
- The process graph gives the causal past of an event in one call.
- The operators let the agent ask for a simulation, instead of reading 11 pages of numbers.
- The graph agent matched perfect retrieval on counterfactual questions with 1/6 of the tokens.

**Limits of this pilot:**

1. It is small: 248 questions, and only 22 predictive questions.
2. Perception is ground truth, not video.
3. Code builds the graph directly from the CLEVRER annotation files. These files hold the same facts as the PDF, but the graph agent never reads the PDF. The offline phase of the design (an LLM extracts the graph from documents) is not built yet.
4. The sufficiency check and the rollout-gated abandonment are not built yet.
5. We used one model and one corpus size (30 runs).
6. 15 of the 30 pilot runs are in the tuning range of the operator constants (10000 to 11999). The held-out operator results in section 6 do not have this overlap.
7. One tool note of the graph agent changed after one pilot question failed. The note now says that the simulated collisions happen before the run ends.

---

## 8. Decisions and next steps

**Decisions for today:**

1. Is CLEVRER a diagnostic probe or the main benchmark?
2. Do the operators (intervention, world simulation) become part of the design?
3. Do we adopt the revised claim (evidence access, not reasoning)?
4. Which models and which budget do we use?
5. Which tasks does Ash take?

**Proposed for the next two weeks:**

1. Scale to 500 runs, and build a 300-run corpus that does not fit in one context window.
2. Build the graph from the reports with an LLM (the offline phase).
3. Add VLM captions as a perception test on a subset.
4. Run a lookup benchmark (FinanceBench) to show that our design does not lose there.
5. Start the manufacturing corpus.

---

## 9. Talking points for each slide

1. **Title.** This is the first pilot of the plan from last Monday.
2. **The pilot in one slide.** Same model and same questions; only the evidence access changes. PageIndex loses about 8 points overall and 15 points on counterfactual questions. It also uses the most tokens.
3. **Setup.** Ground truth removes perception errors. Headings and bookmarks give PageIndex its best case. Full context is the perfect-retrieval reference.
4. **Accuracy.** The largest gap is on counterfactual questions, as Prof. Roy saw on the manufacturing data. The paired test is strong (20 against 1).
5. **Cost.** The tree is larger than the evidence, and the agent reads it again in every turn.
6. **Why 1.** Navigation to the run works. Inside the run, PageIndex reads little: one page for most descriptive questions. In 4 errors, the evidence was on pages that it did not read.
7. **Why 2.** Even a clean document gives a wrong tree and lossy summaries. In 11 errors, PageIndex had the evidence and still answered wrong; full context got all 11 right.
8. **Examples.** One error of each kind: evidence not read (run 10572), and evidence read but still wrong (run 10475).
9. **Correction.** Be open about it: the model can do the physics. The claim moves from "reasoning" to "evidence access".
10. **Operators.** Simple operators, no LLM, near the best published scores, with the caveat about perception.
11. **Meaning and limits.** Say the limits before Prof. Roy asks.
12. **Decisions.** Ask for the five decisions.

---

## 10. How to reproduce

All commands run from `better-page-index/`. The scripts read the OpenRouter key from `.env` in this folder (see `.env.example`). To change the key, edit `.env`. Its values win over the shell.

```bash
uv sync                                                           # pageindex 0.2.22, fpdf2, pypdf, matplotlib

# data (not in git): CLEVRER validation questions and annotations
curl -L -o data/clevrer/validation.json http://data.csail.mit.edu/clevrer/questions/validation.json
curl -L -o data/clevrer/annotation_validation.zip http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip
(cd data/clevrer && mkdir -p annotations && cd annotations && unzip -q ../annotation_validation.zip)

uv run python analysis/calibrate_moving.py                        # constants: moving threshold, time step, contact, friction
PYTHONPATH=src uv run python analysis/eval_operators.py           # operators on 3,000 held-out videos (no LLM)

SIDS=$(PYTHONPATH=src uv run python -c "import qa; print(','.join(map(str, qa.pilot_runs())))")
PYTHONPATH=src uv run python src/scene_report.py runs/pilot/plant_log_30_runs.pdf "$SIDS"
PYTHONPATH=src uv run python src/run_pageindex.py index runs/pilot/plant_log_30_runs.pdf runs/pilot/pi_store runs/pilot/pi_index.json
PYTHONPATH=src uv run python src/run_pageindex.py ask runs/pilot/pi_store <doc_id> runs/pilot/pageindex.jsonl
PYTHONPATH=src uv run python src/run_fullcontext.py runs/pilot/fullcontext.jsonl
PYTHONPATH=src uv run python src/run_graph_agent.py runs/pilot/graph_agent.jsonl

PYTHONPATH=src uv run python analysis/make_figures.py             # results/ figures and results.json
(cd slides && latexmk -pdf -outdir=tmp pilot-results.tex)         # the deck
```

Notes:

- `PYTHONPATH=src` also loads `src/sitecustomize.py`. It makes Python try IPv4 first, because IPv6 connections time out on this network.
- The question files keep the full PageIndex agent traces (every tool call and result) in `runs/pilot/pageindex.jsonl`. `analysis/pageindex_traces.py` summarizes them.
