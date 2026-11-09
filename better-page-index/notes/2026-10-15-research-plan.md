# Research plan after the pilot (2026-10-15)

> **Update 2026-10-17.** The probe (`notes/2026-10-17-probe-results.md`) tested this model on DesignQA and LongMemEval. PageIndex matched full context where the model predicted failure (predictions P2 and P7 were false). The revised model: PageIndex loses accuracy only when the evidence is out of its reach. Two conditions cause this: dense sections whose labels drop the facts (A1), and corpora that are too large to read (A5). Spread alone (A2) did not cause errors on single documents of about 150 pages, because PageIndex read widely and paid with tokens. Sections 3, 8 and 10 below still use the old model. Revise them after the meeting on 2026-10-19.

Inputs: the pilot (`notes/2026-10-12-pilot-results-brief.md`), the PageIndex code (v0.2.22, installed in `.venv`), and a search for papers and datasets on 2026-10-15. Section 13 lists the sources.

## 1. The goal

We do not try to solve CLEVRER. We want to show three things:

1. PageIndex fails in cases that we can predict from how it works.
2. These failures occur on several public datasets, and the traces explain each failure.
3. One general graph method, with no change for each dataset, works better than PageIndex in these cases. In the other cases, it works as well as PageIndex.

## 2. What was specific to CLEVRER in the pilot

The pilot graph agent is not general. Five parts of it depend on CLEVRER:

| Part | In the pilot |
|---|---|
| Graph source | Code built the graph from the ground-truth annotation files, not from the PDF. So the graph is an oracle. |
| Schema | Fixed CLEVRER types: objects with color, material and shape; enter, exit and collision events; a state for each frame. |
| Tools | Six CLEVRER queries, for example `states_at` and `remove_object`. |
| Operators | A physics simulation with tuned constants (friction, contact distance, horizon). |
| Causal edges | A CLEVRER rule: the previous event of an object causes its next event. |

Two results carry over to the general work:

- With clean and complete evidence, the model reasons well. So the problem is evidence access, not reasoning.
- Small typed query results can replace a large tree in the context (2,872 against 110,010 input tokens for each question).

## 3. A model of PageIndex from first principles

PageIndex has two parts:

- **An index.** The index is a tree of sections. Each node has a title, an LLM summary of at most about 150 words, and a page range. The only edge type is "section contains subsection".
- **A search policy.** An LLM reads the titles and the summaries, selects nodes, reads their pages and decides when to stop.

Two facts from the PageIndex code support this model:

- In local mode, the agent has four tools: `browse_documents`, `get_document`, `get_document_structure` and `get_page_content`. No tool searches the page text.
- The agent prompt says: "1 match or 1 clearly best match → proceed to read and answer without asking. Multiple equally relevant → ask user to pick." So the agent assumes that one document answers each question.

Two terms in this plan:

- The **label** of a node is its title and its summary.
- The **key** of a question is the entity, the time or the attribute that identifies its evidence. Example: in "When did pump P-101 last fail?", the keys are "P-101" and "fail".

PageIndex can reach a page only when the labels on the path to that page tell the LLM to go there. So PageIndex works when five assumptions are true.

| # | Assumption | It fails when | Example question | Predicted behavior of PageIndex |
|---|---|---|---|---|
| A1 | The key of the question is in the labels. | The document uses a different organizing axis. Or a section holds more facts than its summary can name. Or many sibling sections have the same title. | "When did pump P-101 last fail?" in a plant log with one chapter for each day. | It reads many sections, or it guesses one section. |
| A2 | The evidence is in one or two sections. | The question counts, lists or compares over many sections, or asks for a trend. | "How many times did P-101 fail in 2025?" | It reads some sections, stops, and gives a count that is too low. |
| A3 | The parts of the evidence connect through the tree, or through an explicit pointer such as "see Note 12". | The parts connect only through a shared entity, a time order or a cause. | "Which supplier made the seal that failed on the line that operator Lee runs?" | It finds the first part, but it does not find the next part. |
| A4 | A passage in the document states the answer. | The answer is a property of a set: "never", "how many", "the latest value", "which came first". | "Did P-101 ever fail?" "What is the current setpoint of oven 3?" | It cannot show absence without reading all pages. It returns an old value. |
| A5 | The tree is small compared with the context budget. | The corpus is large. | 300 runs, 500 chat sessions, all filings of 40 companies. | The tree fills the context, the cost grows, and the model answers wrong with the evidence in hand. |

### 3.1 The main point: a tree cannot have both faithful labels and a small index

A1 and A5 work against each other:

- Small sections give faithful labels, but the tree becomes large. Then A5 fails.
- Large sections give a small tree, but the labels lose most facts. Then A1 fails.
- When every fact is in a label, the tree is as large as the document. That is full context.

Our pilot tree shows this trade-off:

- The labels of the tree have 46,309 words. That is 22% of the words in the PDF.
- But the labels of a run contain only 6.4% of the distinct numbers on the pages of that run (median of 30 runs).

A graph index does not have this trade-off:

- It makes each entity, event and time a node with its own label. A lookup by key returns a small result, at any corpus size (A1, A5).
- Typed edges give the links that A3 needs: same entity, before and after, cause.
- Set queries (count, list, exists, latest value) give the answers that A4 needs, and they make A2 a single query.

### 3.2 A second cause: the search policy

Two failures come from the search policy, not from the index:

- **The agent stops too early.** Nothing tells it that the evidence is complete. In the pilot, it read 1 of about 10 pages for many descriptive questions.
- **The agent does not go back.** After a wrong branch, it seldom returns to an earlier node.

Prof. Roy's search design treats these two failures: the sufficiency check treats the first, and the rollout gate treats the second.

## 4. The evidence so far

### 4.1 Our pilot fits the model

| Pilot result | Assumption |
|---|---|
| PageIndex never read a wrong run. | A1 is true at the chapter level, because the run ID is in the chapter title. |
| For many descriptive questions, it read 1 of about 10 pages. In 4 errors, the evidence was on pages that it did not read. | A1 is false inside a chapter, because frame-level states are not in the labels. |
| The largest gap was on counterfactual questions (70.9% against 85.5%). | A2 is false, because these questions need the whole run. |
| In 11 errors, it had the evidence and still answered wrong. Full context answered all 11 correctly. | A5 is under strain, because the tree of 30 runs is about 90,000 tokens. |

### 4.2 Other studies

- **RIT-RAG** (IBM, arXiv 2610.11370, October 2026): the corpus had all 364 FinanceBench filings. PageIndex got 66.7–75.3% with three models. It chose one filing too early (evidence recall 56.8%). It could not run on a corpus of 2.84 million pages, because the tree did not fit in the context (A5).
- **The 98.7% FinanceBench claim** (Mafin 2.5, a commercial system on PageIndex) counts 12 relabeled answers as correct. Without them, the score is 90.7% (136 of 150).
- **The vendor's open-source benchmark** has 62 look-up questions. It excludes tables, charts, counting and arithmetic. Counting questions break A2.
- **Jeon et al.** (arXiv 2606.29399): on nuclear regulatory documents, a PageIndex re-implementation got 43.5%, and a planning agent got 81.5%.
- **RAPTOR** (ICLR 2024): its authors found that a flat search over all nodes works better than a top-down walk of the tree. **MemWalker** took a wrong branch in 15–19% of cases.
- **Absence questions:** the search found no study of absence questions for tree navigation. We can fill this gap.

## 5. Where PageIndex should win

FinanceBench is the best case for PageIndex:

- Each question names the company, the year and often the financial statement.
- These names are in the document title and in the section titles.
- The evidence is usually on one or two pages. In the open set, 35 of 150 questions cite more than one page, and no question cites more than one filing.

So all five assumptions are true. A second example is the DesignQA question "What does rule V.1 state exactly?". The rule ID is a section title, so A1 is true.

Our method must not be worse on such data, so it keeps the PageIndex tree as one of its views. Some questions need prose, not extracted facts. Example: "Why did management lower its guidance?" For such questions, our method must read the source pages.

## 6. What is new and what is not

Parts of the method already exist:

| Part | Prior work |
|---|---|
| An entity graph and an event graph | E²RAG (EACL 2026), with small gains. Zep/Graphiti, a temporal knowledge graph that its authors tested on LongMemEval. |
| A summary tree and an entity graph | E2GraphRAG (arXiv 2505.24226) |
| Passage nodes and entity nodes | HippoRAG 2 (ICML 2025) |
| MCTS and rollouts on graphs | ReKG-MCTS (ACL 2025 Findings), RTSoG, KBQA-o1 (ICML 2025), Fast MCTS (CIKM 2026) |
| A check that the context is sufficient | Sufficient Context (ICLR 2025) |

The search found none of these:

1. A first-principles model of tree retrieval with testable assumptions.
2. A measured failure analysis of PageIndex over several datasets, with features for each question.
3. A study of absence questions for tree navigation.
4. A method that keeps the tree and adds keyed views with set and time operators, tested against each assumption with one fixed configuration.

**Recommendation:** make items 1 to 3 the main contribution. The method then shows that the analysis gives a fix. Discuss this framing with Prof. Roy, because his design is the method.

Four results from GraphRAG studies set limits on the method:

- Graphs help on multi-hop, comparison and temporal questions. They give little or no gain on simple fact questions (Han et al.; Xiang et al., ICLR 2026).
- Graph methods can fail on absence questions. On MultiHop-RAG null queries, global GraphRAG got 19.3% and plain RAG got 96.0% (Han et al.).
- In one study, the knowledge graphs held only about 65% of the answer entities (Han et al.). So our method must keep the source text.
- An agent with dense search closes most of the gap to GraphRAG (Fan et al., 2026). So this agent is a necessary baseline.

## 7. The general method

### 7.1 What changes from the pilot

| Part | Pilot (CLEVRER) | General method |
|---|---|---|
| Graph source | Code, from the ground-truth annotations | An LLM extracts the graph from the same PDF that PageIndex reads |
| Schema | Objects, collisions, frame states | Entity; fact (attribute, value, unit, time); event (type, time, participants); relation (precedes, causes, prevents, refers to, part of) |
| Tools | Six CLEVRER queries | Eight generic tools (section 7.3) |
| Tree | Not used | Kept as the structure view, so the method includes everything that PageIndex has |
| Source text | Not used | Kept, with a text search, because the extraction misses facts |
| Search | LLM tool calls | LLM-guided path expansion, a rollout gate and a sufficiency check |
| Domain operators | Physics simulation | Not in the main claim. An optional plug-in, tested only as an ablation |

### 7.2 Three views, built from the same PDF

- **Structure view:** the PageIndex tree.
- **Data view:** entities and their facts (attribute, value, unit, time when the fact is true).
- **Process view:** events (type, time, participants) and the relations between events.

Each node keeps its source: the document, the page and the text span.

Index build:

1. Parse the pages with the PageIndex parser.
2. Extract entities, facts, events and relations from each page with an LLM. Use one fixed prompt and one fixed schema.
3. Merge duplicate entities: exact name match, then embeddings, then an LLM check for unclear pairs.
4. Convert all times to absolute times.
5. Index the page text for BM25 and embedding search.

### 7.3 Tools (the same for every dataset)

| Tool | Returns | Assumption that it treats |
|---|---|---|
| `search(text, type)` | Entity, event or passage nodes that match the text | A1; also the fallback when the extraction missed a fact |
| `node(id)` | The facts and the edge counts of one node | A1 |
| `neighbors(id, relation, time range)` | The linked nodes | A3 |
| `timeline(entity, event type, time range)` | The events of an entity, in time order | A2, A4 |
| `value_at(entity, attribute, time)` | The latest fact before that time, with its source | A4 |
| `select(type, filters, operation)` | count, list, min, max, sum or exists over all matching nodes | A2, A4 |
| `read(document, pages)` | The source text | Prose questions; checks of extracted facts |
| `tree(document)` | The structure view | A1, when the key is a section name |

### 7.4 Search policy (Prof. Roy's design)

1. Start from the nodes that `search` returns.
2. The LLM proposes path expansions.
3. The rollout gate gives each open path a score with a cheap lookahead. The system abandons the paths with a score below a threshold.
4. The sufficiency check decides when to stop. For count and absence questions, it needs a complete `select` result.

Use Fast MCTS (arXiv 2502.13428) as the cost reference for the rollouts.

### 7.5 The rule for generalization

Use one schema, one extraction prompt, one tool set and one set of hyperparameters for all datasets. Fix them before the test runs. Do not change them for a dataset.

### 7.6 The design separates the index from the search

| | LLM agent search | Rollout-gated search |
|---|---|---|
| Tree index | PageIndex | Ablation |
| Graph, tree and text index | Ablation | Our method |

This grid shows how much of the gain comes from the index and how much comes from the search.

## 8. Datasets

### 8.1 Tier 1: the first probe (about 265 questions)

| Dataset | Content | Questions to sample | Assumptions | Gold evidence | Metric |
|---|---|---|---|---|---|
| FinanceBench, open set | 150 questions over 84 filings (PDF) | 40 | Control: all five are true | `evidence[].evidence_page_num` (0-based), `doc_name` | Correct or incorrect (LLM judge, checked by hand) |
| DesignQA | Formula SAE 2024 rules, one PDF of about 140 pages | 40 Retrieval, 30 Compilation | A1 true (Retrieval) against A2 false (Compilation), in the same document | Rule IDs | F1, no judge |
| MMLongBench-Doc, text-only | 135 PDFs, 47.5 pages on average | 30 single-page, 30 cross-page, 30 unanswerable | A2, A4 | `evidence_pages` (1-based) | The official answer extraction and score |
| LongMemEval S | Dated chat sessions: about 48 sessions and 115,000 tokens for each question | 15 each of multi-session, knowledge-update, temporal-reasoning and abstention; 5 single-session | A1 (the record uses dates, the questions use topics), A2, A4 | `answer_session_ids`, turn-level `has_answer` | The official judge prompt for each type |

Notes:

- DesignQA is the cleanest test of the model. In the same document, PageIndex should do well on Retrieval and badly on Compilation. Even GPT-4o with all rules in its context got only 0.424 F1 on Compilation, so our `select` tool can also beat full context there.
- For MMLongBench-Doc, use the VectifyAI V2 labels to remove the questions that need images. V2 has 234 count questions, 122 enumerate questions and 89 compare questions.
- LongMemEval gives each question its own history. So the probe needs one PDF and one PageIndex tree for each question.

### 8.2 Tier 2: manufacturing and scale

| Dataset | Content | Role |
|---|---|---|
| AssetOpsBench work orders (IBM, Apache-2.0) | About 4,200 real work orders over 11 years, with timestamps and FMEA rows. It has no evidence-labeled QA. | Our manufacturing test. Render the work orders as a maintenance log, write template questions, and compute the exact answers from the CSV files. Check the data files first. |
| ViDoRe V3 Industrial | 27 USAF maintenance technical orders, 5,244 pages, 283 English queries (multi-hop, compare-contrast, enumerative and others) | Industrial PDFs at corpus scale, with page-level gold |
| CorporateBench (EMNLP Findings 2026) | 4 synthetic companies with 90 days of email each, from 354 to 232,693 documents; 3,000 questions; a gold temporal knowledge graph | Scale (A5), and a gold graph to score our extraction |
| LongMemEval M (a subset) | About 500 sessions and 1.5 million tokens for each question | Scale (A5) |
| FinanceBench, whole corpus | All filings in one corpus | Document selection at scale; compare with RIT-RAG |
| Prof. Roy's manufacturing data | Unknown | The real case study, if he can share it |

The AssetOpsBench log gives five controlled tests from one real source:

- **Axis flip (A1):** render the same work orders by date and by asset.
- **Spread (A2):** questions over one work order, and questions over many work orders.
- **Absence and latest value (A4):** "Did asset X have a work order of type Y in 2017?" "What was the last work order on asset X before date D?"
- **Multi-hop (A3):** work order → failure mode (FMEA) → sensor, if the FMEA rows link to the assets.
- **Scale (A5):** 1, 3 and 11 years of records.

### 8.3 Tier 3: optional

- **MultiHop-RAG:** 609 news articles and 2,556 questions, with 301 null questions. The articles are short and have no structure, so this set tests document selection more than tree navigation.
- **TechManualQA-700:** 700 questions over 20 product manuals, with an unanswerable class and a gold page for each question.
- **TIDE** (arXiv 2608.08512): 3,050 "version in force" questions over 644 dated instruments. This set tests the latest value (A4). Check it before use.
- **Oolong** (arXiv 2511.02817): aggregation questions over long transcripts (A2).
- **CLEVRER:** keep it only for one experiment. Compare the oracle graph from the annotations with the graph that the LLM extracts from the same PDF. The gap measures the extraction loss.

### 8.4 A fair input for PageIndex

- When a dataset has PDFs, use them as they are.
- Render text datasets (LongMemEval, AssetOpsBench) into PDFs with one chapter for each record and clean bookmarks, as in the pilot.
- Sort the LongMemEval sessions by `haystack_dates` before you render them. In 211 of the 500 histories, the file does not keep the sessions in date order.
- Pin one PageIndex version (v0.4.0 came out on 2026-10-12), and report it. Read its agent prompt again before the runs.
- For corpus datasets, test PageIndex twice: with the gold document only, and with the full corpus. The difference measures the document-selection failure.

## 9. How to show "generalizes better"

1. **Write the predictions first.** For each dataset category, write "PageIndex wins", "ties" or "loses", and the assumption that it breaks. Commit this file before the runs.
2. **Measure four things for each system:** accuracy, evidence recall (did the system read the gold evidence?), input tokens and dollars for each question, and the cost of the index build.
3. **Classify each PageIndex error automatically** with the gold evidence:
   - wrong branch: it never opened the section with the evidence;
   - stopped early: it read part of the evidence;
   - read but wrong: it read all of the evidence;
   - absence: no evidence exists, and it gave an answer;
   - tree error: the tree split or merged the evidence section.
4. **Measure the features of each question:** spread (the number of tree nodes with evidence), key in label (yes or no), the similarity of the evidence node to its siblings, the tree size, and the question type. Fit a logistic regression of PageIndex correctness on these features. This turns the failure-mode discussion into a measured result.
5. **Define "generalizes better" with three numbers:** higher mean accuracy over all datasets, higher accuracy on the worst category, and a flatter slope of accuracy against spread and corpus size.
6. **Use these baselines:**
   - PageIndex, with the version pinned;
   - full context, where the document fits;
   - an agent with hybrid search (BM25 and embeddings) over the page text;
   - HippoRAG 2, and E2GraphRAG or E²RAG if their code runs;
   - PageIndex with a keyword search tool. This baseline tests A1 alone.
7. **Control the judge.** Use exact match or F1 when the dataset allows it. Hide the system names from the judge. Check 50 judgments by hand. One study found that judge bias doubles the reported gains of graph methods.

## 10. Next steps

### 10.1 This week (before Monday 2026-10-19)

1. Send Prof. Roy and Ash a one-page summary: the five assumptions, the novelty risk (section 6), the datasets and three requests:
   - his manufacturing data and the PageIndex failures that he saw on it;
   - an API budget (section 11);
   - his opinion on the framing: the analysis first, and the method as the fix.
2. Read three papers in full: RIT-RAG (arXiv 2610.11370), E²RAG (EACL 2026) and E2GraphRAG (arXiv 2505.24226). Write how our method differs.
3. Write the predictions file (section 9, step 1) and commit it.
4. Pin PageIndex v0.4.0 and read its agent prompt.
5. Write loaders for the four Tier 1 datasets. Each loader writes the PDFs, the questions and the gold evidence pages.
6. Build the baseline agent with hybrid search.

### 10.2 Milestones (one for each Monday meeting)

| Monday | Result |
|---|---|
| 2026-10-19 | Summary sent; predictions committed; Tier 1 loaders ready |
| 2026-10-26 | Probe done: PageIndex, full context and hybrid search on about 265 questions; error classes; regression of the features. **Decision point:** if PageIndex matches full context on the categories where we predict failure, the model is wrong. Then revise it before you build more. |
| 2026-11-02 | Generic extractor done. Extraction recall measured on CLEVRER (against the annotations) and on CorporateBench (against the gold graph). AssetOpsBench log built. |
| 2026-11-09 | Graph agent with the generic tools, on Tier 1 |
| 2026-11-16 | Rollout gate and sufficiency check; absence tests |
| 2026-11-23 | Tier 2 scale runs: LongMemEval M subset, ViDoRe Industrial, FinanceBench whole corpus, AssetOpsBench by years |
| 2026-11-30 | Graph baselines and ablations |
| 2026-12-07 | First draft of the paper |

A possible split of the work:

- **Aniruddha:** the probe, the PageIndex failure analysis and the baselines.
- **Ash:** the extractor and the graph store.
- **Prof. Roy:** the design of the rollout gate and the sufficiency check, and the manufacturing data.

## 11. Budget

| Item | Estimate |
|---|---|
| OpenRouter credit left on 2026-10-15 | $50.17 of $100 |
| Tier 1 probe (about 265 questions, three systems) | About $80–120 |
| Full study (Tier 1 and Tier 2, all systems and ablations) | About $500–1,000 (a rough estimate) |

These estimates use the pilot prices: about $0.13–0.22 for each PageIndex question, and about $2.3 for each million input tokens of full context. Full context on one LongMemEval S history (115,000 tokens) costs about $0.26.

To reduce the cost, use `gpt-5.6-luna` for development runs. Use the strong model only for the final runs.

## 12. Risks

| Risk | What to do |
|---|---|
| Parts of the method are not new. | Make the analysis the main contribution (section 6). |
| The extraction misses facts. In one study, graphs held only about 65% of the answer entities. | Keep the source text and the text search. Measure the extraction recall on two datasets with gold graphs. |
| Graph methods can fail on absence questions. | The sufficiency check needs a complete `select` result. Test absence questions early. |
| An agent with dense search closes most of the gap. | Use it as a main baseline. |
| LLM judges favor graph methods. | Use exact match or F1 where possible. Hide the system names. Check 50 judgments by hand. |
| PageIndex changes fast: from v0.2.10 to v0.4.0 in about two months. | Pin the version and report it. |
| Long-context models make retrieval unnecessary for small corpora. | Show the results at scale, and report the cost. |
| Some licenses limit use. FinanceBench and MMLongBench-Doc are CC BY-NC. DesignQA and GlobalQA state no license. | Use them for evaluation only. Do not publish copies of the data. |

## 13. Sources

PageIndex and its evaluations:

- PageIndex: https://github.com/VectifyAI/PageIndex
- Mafin 2.5 on FinanceBench: https://github.com/VectifyAI/Mafin2.5-FinanceBench
- PageIndex open-source benchmark: https://github.com/VectifyAI/PageIndex-OSS-Benchmark
- RIT-RAG: arXiv 2610.11370
- Jeon et al., planning agent for regulatory documents: arXiv 2606.29399
- Lumer et al., "Rethinking Retrieval": arXiv 2511.18177

Tree retrieval and GraphRAG:

- RAPTOR: arXiv 2401.18059
- MemWalker: arXiv 2310.05029
- MemTree: arXiv 2410.14052
- Han et al., RAG vs. GraphRAG: arXiv 2502.11371
- Xiang et al., When to use Graphs in RAG: arXiv 2506.05690
- Zeng et al., judge bias in GraphRAG evaluation: arXiv 2506.06331
- Fan et al., Do We Still Need GraphRAG?: arXiv 2604.09666

Methods close to ours:

- E²RAG: https://aclanthology.org/2026.eacl-long.90
- E2GraphRAG: arXiv 2505.24226
- HippoRAG 2: arXiv 2502.14802
- Zep/Graphiti: arXiv 2501.13956
- ReKG-MCTS: https://aclanthology.org/2025.findings-acl.484
- Fast MCTS: arXiv 2502.13428
- Sufficient Context: arXiv 2411.06037

Datasets:

- FinanceBench: https://huggingface.co/datasets/PatronusAI/financebench
- DesignQA: https://github.com/anniedoris/design_qa
- MMLongBench-Doc: https://github.com/mayubo2333/MMLongBench-Doc
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- AssetOpsBench: https://github.com/IBM/AssetOpsBench
- ViDoRe V3 Industrial: https://huggingface.co/datasets/vidore/vidore_v3_industrial
- CorporateBench: arXiv 2608.27391
- MultiHop-RAG: https://huggingface.co/datasets/yixuantt/MultiHopRAG
- TechManualQA-700: https://zenodo.org/records/17410809
