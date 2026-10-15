# PageIndex on CLEVRER: how PageIndex works, why it fails, and what to do next

Status: draft for the Monday 11 AM meeting. Date: 2026-10-10.

**Update 2026-10-12:** the pilot changed some predictions of section 4.2. With all pages of a run, the model does well on predictive and counterfactual questions; PageIndex loses because it reads only part of the run. See `2026-10-12-pilot-results-brief.md`.

Evidence in this note comes from three sources:

- The PageIndex source code (GitHub `VectifyAI/PageIndex`, main branch, commit of 2026-10-08; the latest release tag is v0.2.22).
- The CLEVRER validation split (5,000 videos, 76,368 questions), analyzed with the scripts in `../analysis/`.
- The public documents of PageIndex and CLEVRER (see "Sources").

---

## 0. Summary

1. PageIndex cannot read a video. In local mode it accepts PDF files only. PageIndex Cloud also accepts scanned and image-rich documents. Thus we must convert each CLEVRER video into a document before PageIndex can index it.
2. Use the CLEVRER ground-truth annotations to write this document (a "scene report"). Then PageIndex gets perfect perception, and each failure comes from the index or from the reasoning, not from vision.
3. A scene report for one video is short. It is about 80 tokens with events only, and about 12,000 tokens (about 22 pages) with all object states. The PageIndex instructions tell the agent to read a document of 20 pages or fewer directly, without the tree. So for one video, PageIndex is almost the same as "give the full document to the LLM". We must run that baseline too. Without it, we cannot say that the tree causes a failure.
4. The data show a structural limit before we run any model:
   - 100% of the correct predictive answers are events that do not occur in the video.
   - 44.8% of the counterfactual questions contain at least one event that does not occur in the video.
   - No retrieval method can find these events, because no document contains them.
5. A perfect causal graph over the observed events answers 99.9% of the explanatory questions. But it gets only 29.2% (predictive) and 52.0% (counterfactual) per-question accuracy.
6. Thus the dual-view graph of Prof. Roy can close the explanatory gap. It cannot close the predictive and counterfactual gap without a simulation step. The rollouts in the figure are search rollouts: they only decide if a path is abandoned. They do not make new events.
7. Recommendation: use CLEVRER as a diagnostic probe for the failure modes of PageIndex. Then either add a simulation tool to our system, or limit our CLEVRER claims to the observed events. Test the counterfactual claim on the manufacturing data, where the process rules are written in the documents.

---

## 1. How PageIndex works

### 1.1 The basic idea

RAG (retrieval-augmented generation) has two steps. First, the system retrieves the relevant text. Then an LLM writes the answer from that text.

- **Vector RAG** cuts each document into chunks, converts each chunk into an embedding, and retrieves the chunks with the highest similarity to the question.
- **PageIndex** has no embeddings and no chunks. It builds a tree that is like a table of contents. At question time, an LLM agent reads the tree (titles and summaries) and decides which pages to read. The authors call this "vectorless, reasoning-based RAG".

### 1.2 Offline phase: build the tree index

The SDK has two indexing modes. Flash is the default since August 2026.

**Flash mode (default):**

1. Parse the PDF layout: fonts, line positions and blocks. Detect the headings and the printed table of contents. Use the embedded PDF bookmarks if they are good. This step uses no LLM.
2. Build the tree from the headings. Each node covers one contiguous page range.
3. Optimize the tree (default "full"). A deterministic step merges small subtrees. An LLM step expands large nodes into subsections. The cost model counts the pages that the agent must scan to reach one target in the worst branch.
4. An LLM writes a summary for each node (default maximum: 150 words) and one description of the document.

Two rules are important for log-like data:

- If the layout has no headings, Flash makes a flat tree: one node for each page, with the title "Page N". If this flat tree has more than 10 nodes, the local client refuses the document and tells you to use the standard mode.
- If the PDF has no text layer, Flash refuses it and tells you to run OCR first.

**Standard mode (the original LLM pipeline):**

1. An LLM checks the first 20 pages for a table of contents.
2. The pipeline tries three paths, with fallback: a table of contents with page numbers, a table of contents without page numbers, and no table of contents. With no table of contents, the LLM reads the pages in groups of about 20,000 tokens and writes the section hierarchy ("1", "1.1", ...) with the start page of each section.
3. An LLM checks that each title appears on its page. At 100% accuracy, the pipeline accepts the result. Above 60%, it fixes the wrong entries (maximum 3 attempts). Below 60%, it falls back to the next path.
4. Each leaf node with more than 10 pages and at least 20,000 tokens is split again with the "no table of contents" path. This repeats recursively.
5. An LLM writes the node summaries and the document description.

**Node format:** `title`, `node_id`, `start_index` and `end_index` (1-based pages), `summary`, `key_items` (titles of merged subsections) and `nodes` (children).

### 1.3 Online phase: answer a question

1. An LLM agent (OpenAI Agents SDK through LiteLLM, so most model providers work) gets four read tools: `browse_documents`, `get_document`, `get_document_structure` and `get_page_content`.
2. The agent instructions say:
   - For a document of more than 20 pages, get the structure first, then read targeted page ranges.
   - For a document of 20 pages or fewer, read the pages directly.
3. The agent can call the tools many times. It can go back and read a different branch. This is the "backtracking": the LLM decides at each turn. No value function controls it in the open-source code.
4. The unit of retrieval is a page.
5. If citations are on, the prompt says: "Never fill a gap from general knowledge. When the documents do not answer the question, say so."

### 1.4 Corrections to the meeting notes

| Meeting note | What the code and the documents show |
|---|---|
| "Monte Carlo tree search and an RL-based value function decide the expansion." | The open-source code has no MCTS, no value function and no RL. An LLM agent with tools does the search. The docs say: "In our dashboard and retrieval API, we use a combination of LLM tree search and value function-based Monte Carlo Tree Search (MCTS). More details will be released soon." No details were released. The documented "hybrid tree search" uses a value function that is a vector search: it embeds chunks of each node and scores the node as (1/√(N+1)) · Σ chunk scores. The docs call this hybrid search "the default search method used in our retrieval API". Thus the cloud retrieval is not strictly vectorless, and no document mentions RL. |
| "Hierarchical clustering merges a million documents into a tree." | The open-source code has no document clustering. The only "clustering" groups text lines into blocks in the PDF parser. The cloud-only "PageIndex File System" (blog of 2026-05-03) says: "Documents are clustered into topic nodes by topic models or LLM-driven grouping". A document can be under more than one parent, and the hierarchy can change with the query. The blog gives no accuracy, latency or scale results. |
| "Decompose until semantic value is exhausted." | Standard mode splits by size (more than 10 pages and at least 20,000 tokens). Flash splits and merges by a search-cost rule. |
| "98% on finance, against 60-70% for vector RAG." | The README shows 98.7% on FinanceBench against 50% for vector RAG. The 98.7% is for Mafin 2.5 (a finance product on PageIndex) on the 150 public questions. It counts benchmark errors and other valid methods as correct after a human review. A strict match is 136/150 (90.7%). The 50% bar equals the GPT-4-Turbo "single vector store" result of the 2023 FinanceBench paper, so the two results use different models and years. |
| "CLEVR dataset." | The link and the screenshot show CLEVRER (video). CLEVR is a different dataset of still images. |
| "Petronus AI." | The company is Patronus AI. |

### 1.5 Cost and published evidence

- Indexing: about $0.001 for each page, and 13 seconds to 4.5 minutes for documents of 9 to 1,098 pages (README numbers).
- Questions: the cost depends on the chat model and the number of agent turns. The meeting estimate was about 60 seconds for each question.
- The own open-source benchmark of PageIndex has 62 questions on 34 PDFs. The README says: "Every question's answer is a fact stated in running text, so a wrong answer is a retrieval or reading failure, not a reasoning one." Thus the published evidence is for lookup questions.
- Cloud pricing (from 2026-09-16): $0.01 for each page to index, then $0.001 for each active page each month. You also pay your LLM provider. Local mode is free.

**Independent tests (not by VectifyAI):**

- RIT-RAG (arXiv 2610.11370, 2026-10-08) ran PageIndex on FinanceBench with three chat models. It got 66.7% to 75.3%, not 98.7%. With Claude Haiku, each FinanceBench question cost 12.6 cents and took 14 seconds. A vanilla RAG cost 0.2 to 0.3 cents. On a corpus of 2.84 million pages, PageIndex could not run, because the tree did not fit in the context.
- A PwC study (arXiv 2511.18177) on 1,200 SEC filings found that a vector-based agentic RAG won 68% of the comparisons against a hierarchical node-based system. It reported a "retrieval bottleneck at the table-of-contents level".
- We found no independent test of PageIndex on manufacturing, causal or counterfactual questions. Our CLEVRER baseline can be the first one.

---

## 2. CLEVRER in numbers

### 2.1 The dataset

- 20,000 synthetic videos of moving and colliding objects. Each video is 5 seconds long and has 128 annotated frames (480 x 320). The license is CC0.
- The Bullet physics engine simulates 7 seconds, and the video shows the first 5 seconds. The paper says: "The remaining two seconds are held-out for predictive tasks."
- Splits: train (videos 0-9,999), validation (10,000-14,999) and test (15,000-19,999). The test answers are hidden. The test phase of the EvalAI server closed on 2026-01-16, so recent papers report validation results. We must do the same.
- Annotations exist for train and validation. Each video has a JSON file with:
  - `object_property`: color, material and shape of each object.
  - `motion_trajectory`: for each of the 128 frames, the location, orientation, velocity, angular velocity and `inside_camera_view` flag of each object.
  - `collision`: object ids, frame id and location of each collision.
- Each question also has a `program` (a list of reasoning steps). The scripts in `../analysis/` use the programs to find the objects and events of each answer choice.

### 2.2 The validation split (measured by `analysis/clevrer_stats.py`)

- 5,000 videos with 3 to 6 objects (mean 4.9) and 1 to 5 observed collisions (mean 2.5).
- The annotations stop at frame 127. The last collision is at frame 125. Thus the annotations do not contain future events.

| Type | Questions | Share | Answer format |
|---|---:|---:|---|
| Descriptive | 54,990 | 72.0% | One word (count, color, shape, material, yes/no) |
| Explanatory | 8,488 | 11.1% | Multiple choice, 3-4 options, 0-4 correct |
| Predictive | 3,557 | 4.7% | Multiple choice, 2 options, exactly 1 correct |
| Counterfactual | 9,333 | 12.2% | Multiple choice, 1-4 options, 0-3 correct |
| Total | 76,368 | 100% | |

### 2.3 The metrics

- Descriptive: accuracy.
- Multiple choice: each option is a "correct or wrong" decision.
  - Per-option accuracy: the share of options with the right decision.
  - Per-question accuracy: the share of questions where all options have the right decision.
- About 50% of the options are "correct". Thus a majority-label baseline gets about 50% per-option accuracy.

### 2.4 Published results

Cells show per-option / per-question accuracy. Descriptive has one value. Test split unless the row says otherwise.

| Method | Descriptive | Explanatory | Predictive | Counterfactual |
|---|---|---|---|---|
| Random (by question type) | 29.2 | 50.1 / 8.1 | 50.7 / 25.5 | 50.1 / 10.3 |
| MAC (V+), a video QA model | 86.4 | 70.5 / 22.3 | 59.7 / 42.9 | 63.5 / 25.1 |
| NS-DR (neuro-symbolic, learned dynamics) | 88.1 | 87.6 / 79.6 | 82.9 / 68.7 | 74.1 / 42.2 |
| DCL | 90.7 | 89.6 / 82.8 | 90.5 / 82.0 | 80.4 / 46.5 |
| ALOE (object attention) | 94.0 | 98.5 / 96.0 | 93.5 / 87.5 | 91.4 / 75.6 |
| VRDP (differentiable physics, extra labels) | 93.4 | 96.3 / 91.9 | 95.7 / 91.4 | 94.8 / 84.3 |
| CRCG(VRDP) (WACV 2024, best found) | 96.5 | 99.3 / 98.8 | 96.1 / 92.3 | 96.6 / 90.7 |

**LLMs and VLMs (validation split or a sample of it):**

| System | Descriptive | Explanatory | Predictive | Counterfactual |
|---|---|---|---|---|
| GPT-4o, 8 frames, zero-shot (arXiv 2412.08619) | 62.7 | 65.5 / 30.7 | 47.6 / 30.3 | 60.2 / 18.7 |
| Gemini 1.5 Pro, full video, zero-shot (same paper) | 58.6 | 61.2 / 15.7 | 49.6 / 32.0 | 55.6 / 17.6 |
| Gemini 2.0 Flash, full video, our PMLR 2025 paper (50 videos, 620 questions; one LLM-judged accuracy for each type) | 58.8 | 30.8 | 70.6 | 34.7 |
| Symbolic interpreter on ground-truth annotations, no LLM (arXiv 2605.15967) | 98.0 | 99.9 / 99.9 | 84.1 / 69.5 | 86.7 / 59.9 |

Four facts from these tables:

- The models that reach high counterfactual scores (VRDP, CRCG, ALOE) all learn or use the physics of the scene.
- Frontier VLMs are weak on CLEVRER, mainly per question.
- Our own PMLR 2025 paper (Chattopadhyay, Dandekar, Roy, "Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems") already gives a video-native LLM reference. Counterfactual and explanatory questions were the weakest types.
- The symbolic interpreter on the ground truth uses straight-line motion for the future. With this simple projection, it gets 69.5% per-question on predictive questions. Our observed-graph ceiling, with no projection, is 29.2% (section 4.1).

**Known biases:**

- The ALOE paper says that "about 47% of the counterfactual questions ask about the effect of removing an object that did not collide". It also says that "about 45% can be answered perfectly" as if they were descriptive questions. ComPhy (ICLR 2022) removed such questions for this reason.
- CLEVRER-Humans (NeurIPS 2022) found that humans agree with only 62% of the rule-based causal pairs of CLEVRER. Thus the explanatory labels follow a rule ("responsible" means "causal ancestor"). We must give this rule to every LLM system in the prompt. Otherwise, an LLM can fail because it uses a different meaning of "responsible", not because of retrieval.

---

## 3. How to give a video to PageIndex

| Option | What PageIndex reads | Advantage | Problem |
|---|---|---|---|
| A. Ground-truth annotations to text (recommended first) | A scene report from the JSON annotations | Perfect perception. Each failure comes from the index or the reasoning. Free and deterministic. | Not a real "video" pipeline. Reviewers can call it an oracle setting. |
| B. Video-language model captions to text | A scene report written by a VLM from the frames | A realistic pipeline. | Caption errors mix with retrieval errors. You cannot tell which part failed. |
| C. Video frames as images in a PDF | Page images in PageIndex Cloud | The nearest thing to "PageIndex on video" as it is. | Local mode refuses image-only PDFs. Cloud OCR is for documents, not for motion. The result shows nothing about retrieval. |

**Recommendation:** use option A for the main baseline. Add option B later on a small sample, to show the effect of perception errors. Skip option C, or run it only on a few videos as a sanity check.

### 3.1 Two document formats (same facts, different layout)

The report must contain the object states over time, not only the events. In the validation split, 45.4% of the descriptive questions ask if objects are "moving" or "stationary", and 66.1% have a time condition ("when the cube enters", "before", "last").

- **Log format:** an object table, then one line for each object in each frame, and one line for each collision, in time order. It has almost no headings, like a machine log in a factory. Flash will probably make a flat tree and refuse it, so this format also tests the standard mode.
- **Report format:** sections with real headings and PDF bookmarks. For example: "1 Objects" (one subsection for each object), "2 Timeline" (one subsection for each window of 16 frames) and "3 Collisions". This is the best case for PageIndex.

If PageIndex fails on the report format too, the result does not depend on the layout.

### 3.2 Two retrieval settings

- **One video for each document.** The document is short, so the tree is almost not used (see 1.3). This setting measures the reasoning limit of each question type.
- **A corpus of many videos.** For example, 100 scene reports, or one long "plant log" with one chapter for each run. Each question names the run ("In run 10023, ..."). This setting makes the tree search necessary, like FinanceBench, where each question names the company and the year.

---

## 4. Why PageIndex fails on CLEVRER: the exact reasons

| # | Mechanism in PageIndex | Effect on CLEVRER | Symptom to measure |
|---|---|---|---|
| R1 | Input is text in a PDF. | Perception must happen before PageIndex. PageIndex cannot see motion. | Not a retrieval result; we remove it with option A. |
| R2 | The tree is a table of contents. Each node is one contiguous page range with one parent. There are no links between nodes. | A collision belongs to two objects and to one time. It can go under only one branch. Causal links (A hits B, so B hits C) are not in the tree. | Explanatory errors that grow with the length of the causal chain. |
| R3 | The agent reads documents of 20 pages or fewer without the tree. | For one video, PageIndex is almost the same as full-context reading. | PageIndex and the full-context baseline give almost the same scores in setting 3.2 (one video). |
| R4 | The tree optimizer measures the cost to reach one target. The retrieval unit is a page. | A question that needs several events needs several branches. The agent must find each one. | In the corpus setting: missing evidence when the answer needs events from several time windows. |
| R5 | The agent selects nodes by titles and summaries (maximum 150 words). | A summary cannot keep the numbers of a state table (positions, velocities). | Wrong page choices for questions about a state at one time ("How many objects are moving when the cube enters?"). |
| R6 | Retrieval can only return text that exists. | All correct predictive answers and the new events of 44.8% of the counterfactual questions are not in any document. | Low predictive and counterfactual scores. Answers like "the document does not say". |
| R7 | The tools only read. There is no "intervene" or "simulate" tool. | "What happens if the cylinder is removed?" needs an intervention, then a new simulation. | Counterfactual errors on events that the removal prevents or creates. |
| R8 | In a corpus, the agent finds documents by name and description. | All scene reports have almost the same description. | Wrong-document errors in the corpus setting, unless the run id is in the name. |

### 4.1 What the observed events allow (measured by `analysis/clevrer_observed_graph.py`)

The script builds a perfect causal graph from the observed events only (entrances and collisions, with time-ordered causal links). It uses no LLM and no simulation.

| Type | Per-option | Per-question | Events in the asked world that are not in the video |
|---|---:|---:|---|
| Explanatory | 99.9% | 99.9% | (not applicable) |
| Predictive | 64.6% | 29.2% | 100% of the events that happen; 100% of the questions |
| Counterfactual | 83.9% | 52.0% | 32.1% of the events that happen; 44.8% of the questions |

How to read this table:

- **Explanatory** questions are fully answerable from the observed events, if the system has the causal links. A tree has no causal links, so the LLM must rebuild them from the text. In 71.1% of the explanatory questions, the causal chain has at least one earlier collision. In 14.2%, it has two or more.
- **Predictive** questions are never answerable from the observed events. The 29.2% comes only from a shortcut: "an event that already happened is not the next event".
- **Counterfactual** questions are answerable from the observed events only for the observed part. A system without simulation is capped near 52% per-question accuracy.

### 4.2 Our predictions for PageIndex (one video for each document)

| Type | Prediction | Reason |
|---|---|---|
| Descriptive | High, if the report has the object states | Lookup, counting and time filters. Errors come from counting at one time point. |
| Explanatory | Medium | The LLM must rebuild the causal chain from the collision list (R2). |
| Predictive | Low | The answer is not in the document (R6). |
| Counterfactual | Low per question | New events need simulation (R6, R7). Ceiling near 52% without simulation. |

This pattern matches what Prof. Roy saw on the manufacturing data: good on direct questions, medium on abductive questions, and a large drop on counterfactual questions.

---

## 5. What the dual-view graph fixes, and what it does not fix

| Failure | Component in the proposed system | Fixed? |
|---|---|---|
| R2: no causal links | Process graph (events, causes, goals) with typed edges | Yes. Explanatory questions become a graph traversal (99.9% ceiling). |
| R4: evidence in several branches | LLM-guided search over two graph views, with several paths | Yes, in principle. |
| R5: lossy summaries | Data graph with typed facts and measurements, linked to evidence spans | Yes. State questions become typed lookups. |
| Cost and latency | Sufficiency check and rollout-gated abandonment | Partly. This must be measured. |
| R6, R7: unseen events | Nothing in the current figure | **No.** Search rollouts only decide if a path is abandoned. They do not make new events. |

### 5.1 Suggested extension

Use two different terms, because the figure uses "rollout" for search:

- **Search rollout:** simulate the next steps of the search, to decide if a path is still worth expansion (as in the figure).
- **World simulation:** simulate the scene, to make the events that the documents do not contain.

Add two operators to the process graph:

1. **Intervention:** remove node X and mark all events that depend on X. This operator alone reaches the observed-graph ceiling (52% per question on counterfactual questions).
2. **World simulation tool:** produce the new events. In CLEVRER, this can be a simple physics model that starts from the object states in the data graph (positions and velocities), or a physics engine. In manufacturing, this is a process simulator or a set of written process rules. Even straight-line motion helps a lot: a published symbolic interpreter with straight-line projection gets 69.5% per question on predictive questions, against our 29.2% ceiling without projection.

### 5.2 A risk for the paper

- In CLEVRER, the counterfactual answer is not written anywhere. It needs physics.
- In manufacturing documents, a counterfactual answer is often written as a rule (for example, "if the torque step is skipped, the seal fails"). Thus rule propagation over a process graph can answer it.
- These are two different tasks. If we report only CLEVRER counterfactual scores, our system can look weak for a reason that is not related to the goal of the paper.
- Published CLEVRER models with dynamics models get much higher counterfactual scores: 84.3% (VRDP) and 90.7% (CRCG) per question (see 2.4). Reviewers will compare with them.

---

## 6. Plan

### 6.1 Before the Monday meeting

1. Confirm the open questions in section 7 with Prof. Roy and Ash.
2. Install the PageIndex SDK with `uv` and run it on one example PDF. Record the model, the cost and the time.
3. Write the converter from annotations to scene reports (log format and report format). Make real PDFs with headings and bookmarks, because local mode reads PDFs only.
4. Run a pilot on 20 videos (about 300 questions) with PageIndex and the full-context baseline. Use the same chat model for both. Save every agent trace.
5. Bring the tables in 4.1 and the pilot traces to the meeting.

### 6.2 The main baseline experiment

- **Sample:** 500 validation videos with a fixed random seed. Use all multiple-choice questions of these videos (about 850 explanatory, 350 predictive and 950 counterfactual) and a random sample of 1,000 descriptive questions.
- **Systems (all with the same chat model):**
  1. Majority label and the observed-graph ceiling (free, already done).
  2. Full-context LLM: the full scene report in the prompt, no retrieval.
  3. Vector RAG: chunks, embeddings and top-k retrieval.
  4. PageIndex local SDK: Flash mode and standard mode.
  5. Optional: a video-native LLM that reads the MP4 file, as a perception reference. Our PMLR 2025 paper already has one (Gemini 2.0 Flash).
- **Prompt rule:** give every LLM system the CLEVRER meaning of "responsible" (causal ancestor) and the same output format.
- **Settings:** one video for each document, and a corpus of 100 videos.
- **Metrics:** CLEVRER accuracy (descriptive), per-option and per-question accuracy (multiple choice), cost, latency, number of agent turns, pages read, and the share of "the document does not say" answers.
- **Error analysis:** for each wrong answer, record the reason: evidence not retrieved, evidence retrieved but wrong reasoning, unseen event, wrong time step, or counting error. For explanatory questions, plot accuracy against the length of the causal chain.

### 6.3 After the baseline

1. Build the dual-view graph from the same scene reports.
2. Add the intervention operator and test it against the 52% ceiling.
3. Decide with Prof. Roy if we add a world simulation tool.

---

## 7. Questions for Prof. Roy

1. Do we use CLEVRER (video) or CLEVR (still images)? The notes say CLEVR, but the screenshot shows CLEVRER.
2. Is CLEVRER a diagnostic probe for PageIndex, or the main benchmark of the paper?
3. Do we give PageIndex ground-truth text (option A) or VLM captions (option B)?
4. Does our system get a world simulation tool for predictive and counterfactual questions?
5. Which chat model and which budget do we use? Do we use the local SDK (MIT license) or PageIndex Cloud?
6. Which tasks does Ash take (converter, baselines or graph construction)?

---

## Sources

- PageIndex repository and README: https://github.com/VectifyAI/PageIndex
- Code read for this note: `pageindex/local_api.py`, `pageindex/flash/api.py`, `pageindex/page_index_classic.py`, `pageindex/tree_optimize.py`, `pageindex/agent_tools.py`, `pageindex/local_chat.py`, `pageindex/config.yaml`.
- PageIndex tree search docs: https://docs.pageindex.ai/tutorials/tree-search/llm and https://docs.pageindex.ai/tutorials/tree-search/hybrid
- PageIndex multi-document search docs: https://docs.pageindex.ai/tutorials/doc-search
- PageIndex File System blog: https://pageindex.ai/blog/pageindex-filesystem
- PageIndex Flash blog: https://pageindex.ai/blog/pageindex-flash
- PageIndex pricing: https://docs.pageindex.ai/pricing
- Mafin 2.5 FinanceBench results: https://pageindex.ai/blog/Mafin2.5 and https://github.com/VectifyAI/Mafin2.5-FinanceBench
- RIT-RAG (independent PageIndex test): https://arxiv.org/abs/2610.11370
- PwC study (vector agentic RAG against node-based RAG): https://arxiv.org/abs/2511.18177
- CLEVRER website and data: http://clevrer.csail.mit.edu/ and http://data.csail.mit.edu/clevrer/README.txt
- CLEVRER paper: Yi et al., "CLEVRER: CoLlision Events for Video REpresentation and Reasoning", ICLR 2020, https://arxiv.org/abs/1910.01442
- CLEVRER evaluation server: https://eval.ai/api/challenges/challenge/667/challenge_phase
- DCL: https://arxiv.org/abs/2103.16564 ; VRDP: https://arxiv.org/abs/2110.15358 ; ALOE: https://arxiv.org/abs/2012.08508 ; CRCG: https://arxiv.org/abs/2506.10753
- Physics Context Builders (GPT-4o and Gemini on CLEVRER): https://arxiv.org/abs/2412.08619
- Symbolic interpreter and Llama on ground-truth text: https://arxiv.org/abs/2605.15967
- Our PMLR 2025 paper: https://neurosymbolic-ai-journal.com/system/files/nai-paper-883.pdf
- CLEVRER-Humans: https://arxiv.org/abs/2310.03635 ; ComPhy: https://arxiv.org/abs/2205.01089
- FinanceBench paper: https://arxiv.org/abs/2311.11944
