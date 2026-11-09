# QBranch related-work scan (2026-11-09)

Status: first full scan, by Aniruddha (with Claude), for Prof. Roy and Ashraf.
Method: web and arXiv search on 2026-11-09. I read the arXiv abstract page, the proceedings page or
the PDF for each main item. The benchmark numbers come from the result tables of the PDFs.

Marks:

- **not confirmed**: I could not check the claim on a primary source (the paper, the arXiv page, the
  proceedings page or the official venue site).
- "Overlap" names the QBranch part: **(a)** agent and persistent memory, **(b)** question-driven schema
  evolution and open slots, **(c)** evidence-grounded answering and the judge, **(d)** view management,
  **(P)** the planning angle, **(W)** the world-model framing.
- Many 2026 papers are preprints. Their numbers are self-reported.
- A sentence such as "we found no paper that does X" is a search result. It is not a proof.

---

## 0. Venue facts (checked on 2026-11-09)

| Venue | Facts | Source |
|---|---|---|
| ICWM 2027 | Paper deadline 2026-11-30 (provisional). Reviews 2027-01-11, rebuttal 2027-01-11 to 01-18, decisions 2027-02-08. Late May 2027, SF Bay Area. Public, real-time review on OpenReview. The site has no topic list. The site does not define "world model". Creators: Randall Balestriero (Brown / AMI), Amir Zadeh (Lambda), Yann LeCun (NYU / AMI). Format and archival status: not stated. | [icwm.cc/call-for-papers](https://icwm.cc/call-for-papers), [icwm.cc/about](https://icwm.cc/about) |
| ICAPS 2027 | Abstract 2026-12-07, paper 2026-12-14 (11:59 PM UTC-12). Author response 2027-02-04 to 02-08. Notification 2027-02-26. Columbia, SC, USA, 2027-06-27 to 07-02. Special track "Planning Under a Different Name" accepts work from neighbor AI fields that makes a real contribution to planning. | [icaps27 news page](https://icaps27.icaps-conference.org/news) |

Note: a LeCun co-founded venue will likely have reviewers who prefer predictive, JEPA-style world
models. They will ask what QBranch predicts. See Section 5 and Section 9.

---

## 1. Question-driven and query-anticipating indexing

| Paper | Authors | Venue / arXiv / date | One-line summary | Overlap |
|---|---|---|---|---|
| [Document Expansion by Query Prediction (doc2query)](https://arxiv.org/abs/1904.08375) | Nogueira, Yang, Lin, Cho | arXiv 1904.08375, Apr 2019 (venue not confirmed) | A seq2seq model predicts 10 queries per document and appends them before BM25 indexing. | (b) question inference at write time |
| [Doc2Query--: When Less is More](https://arxiv.org/abs/2301.03266) | Gospodinov, MacAvaney, Macdonald | ECIR 2023; arXiv 2301.03266 | A relevance filter removes hallucinated generated queries before indexing. | (b) keep only source-supported questions |
| [PAQ: 65 Million Probably-Asked Questions](https://aclanthology.org/2021.tacl-1.65) | Lewis, Wu, Liu, Minervini, Küttler, Piktus, Stenetorp, Riedel | TACL 2021; arXiv 2102.07033 | 65M generated QA pairs act as a semi-structured knowledge base; RePAQ retrieves over them. | (a) question store, (b) |
| [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) | Gao, Ma, Lin, Callan | arXiv 2212.10496, Dec 2022 (ACL 2023: not confirmed) | The LLM writes a hypothetical answer document at query time. It is the query-side inverse of QBranch. | (c) contrast only |
| [Question-Based Retrieval using Atomic Units for Enterprise RAG](https://arxiv.org/abs/2405.12363) | Raina, Gales | arXiv 2405.12363, May 2024 | Chunks become atomic statements; an LLM writes questions on them; retrieval matches the query to these questions. | (b) |
| [PIKE-RAG](https://arxiv.org/abs/2501.11551) | J. Wang, Fu, R. Wang, Song, Bian (MSRA) | arXiv 2501.11551, Jan 2025 | "Knowledge atomizing": the LLM writes the questions each chunk can answer and stores them as index tags. Tested on HotpotQA, 2Wiki, MuSiQue. | (b) questions as the index |
| [QuOTE: Question-Oriented Text Embeddings](https://arxiv.org/abs/2502.10976) | Neeser, Latimer, Khatri, Latimer, Ramakrishnan | arXiv 2502.10976, Feb 2025 | It embeds each chunk together with the questions that the chunk can answer. | (b) |
| [HopRAG](https://arxiv.org/abs/2502.12442) | Hao Liu et al. | ACL 2025 Findings; arXiv 2502.12442 | Passage graph with LLM pseudo-queries as edges. An "out-coming" question that passage A raises but cannot answer links to a passage B whose "in-coming" question matches it. | **(b) closest prior work for open slots** (static corpus only) |
| [QCG-RAG: Query-Centric Graph RAG](https://arxiv.org/abs/2509.21237) | Wu, Bo, Zhang, Liang, Liu | arXiv 2509.21237, Sep 2025 | Doc2Query questions become graph nodes; edges are chunk–query and query–query links; retrieval goes through the questions. | (b), (d) |
| [T-Mem: Memory That Anticipates, Not Archives](https://arxiv.org/abs/2606.15405) | Guo, D. Wang, Z. Wang, Liu, Xu (Tencent) | arXiv 2606.15405, Jun 2026 | At write time, the LLM makes "triggers": cues for the future situations where a memory will matter. | (b) anticipation at write time |
| [Which questions should I answer? (QSALIENCE)](https://arxiv.org/abs/2404.10917) | Wu, Mangla, Dimakis, Durrett, Li | arXiv 2404.10917, Apr 2024 (EMNLP 2024: not confirmed) | Salient "inquisitive" questions that a text raises are more likely to get an answer later in the same article. | (b) theory: questions that documents raise |

Summary for (b), first half: "infer the questions that a chunk answers and index them" is old (2019)
and common (2024–2026). HopRAG already links "raised but not answered" questions to the passages that
answer them. QBranch must not claim question inference or question indexing as new.

---

## 2. Schema induction and canonicalization for LLM-built KGs

| Paper | Authors | Venue / arXiv / date | One-line summary | Overlap |
|---|---|---|---|---|
| [EDC: Extract, Define, Canonicalize](https://aclanthology.org/2024.emnlp-main.548/) | Bowen Zhang, Harold Soh | EMNLP 2024; arXiv 2404.03868 | Open IE, then the LLM defines each schema element, then canonicalization by definition embeddings. It works with or without a target schema. | (b) schema normalization tool |
| [AutoSchemaKG](https://arxiv.org/abs/2505.23628) | Jiaxin Bai, …, Yangqiu Song (20 authors, HKUST) | arXiv 2505.23628, May 2025 (venue not confirmed) | Joint triple and schema induction for entities and events, with conceptualization. ATLAS: 900M+ nodes, 5.9B edges; 92% alignment with human schemas. | (b) |
| [Ontology-grounded KG construction under Wikidata schema](https://arxiv.org/abs/2412.20942) | Xiaohan Feng, Xixin Wu, Helen Meng | HI-AI@KDD 2024 workshop, CEUR 3841; arXiv 2412.20942 | The LLM writes competency questions, takes relations from them, builds an ontology, aligns it to Wikidata, then extracts the KG. | **(b) closest on "questions → typed schema"** |
| [IN-VKGE: Information-Needs-Guided VKG Enrichment](https://www.ijcai.org/proceedings/2026/0448.pdf) | Lin Ren, Guohui Xiao, Guilin Qi, et al. | IJCAI 2026 | For each new information need, the LLM tests "supportable?" with SPARQL feedback. If not, it proposes ontology and mapping changes. | (b) question-driven schema evolution (databases, expert in the loop) |
| [Youtu-GraphRAG](https://arxiv.org/abs/2508.19855) | Junnan Dong et al. | arXiv 2508.19855, Aug 2025 | A seed schema bounds extraction; the schema expands under a confidence threshold; the same schema guides agentic retrieval. | (a), (b) |
| [Agentic-KGR](https://arxiv.org/abs/2510.09156) | Jing Li et al. | arXiv 2510.09156, Oct 2025 | Multi-round multi-agent RL co-evolves the LLM and the KG, with dynamic schema expansion. | (b), (P) |
| [DIAL-KG](https://arxiv.org/abs/2603.20059) | Weidong Bao et al. | DASFAA 2026; arXiv 2603.20059, Mar 2026 | Closed loop for incremental KG construction: extraction, governance adjudication, then schema induction from validated facts for the next batch. | (a), (b) streaming schema evolution |
| [TRACE-KG](https://arxiv.org/abs/2604.03496) | Abolhasani, Ba, He, Pan | LoG 2026 (PMLR 338); arXiv 2604.03496, Apr 2026 | Joint construction of a context-enriched KG and an induced schema, with qualifiers and traceability to source text. | (a) provenance, (b) |
| [MemGraphRAG](https://arxiv.org/abs/2606.00610) | Chuanjie Wu, Zhishang Xiang, Yunbo Tang, Zerui Chen, Qinggang Zhang, Jinsong Su | KDD 2026; arXiv 2606.00610, May 2026 | Agents share a memory of schemas, facts and passages. Candidate schemas become "stable" when their frequency passes a threshold. Agents detect and fix conflicts. | (a), **(b) candidate → stable schema**, (d) |
| [Executable Schema Contracts](https://arxiv.org/abs/2606.05415) | Jonnalagedda, Yao, Gao, Hasson, Das | arXiv 2606.05415, Jun 2026 | An induced schema is the contract for KG construction and retrieval. A query-time agent can extend it by a "monotonic protocol". | (b) |
| [EvoOntology](https://arxiv.org/abs/2609.15779) | Chong, Zhang, Fan, Du | arXiv 2609.15779, Sep 2026 | A self-evolving ontology layer for data agents. Typed edits are accepted only after a paired evaluation. | **(d) "keep only if it helps"** |
| [iText2KG](https://arxiv.org/abs/2409.03284) | Lairgi, Moncla, Cazabet, Benabdeslem, Cléau | WISE 2024; arXiv 2409.03284 | Zero-shot, incremental KG construction as documents arrive. | (a), (b) |
| [Never-Ending Learning (NELL)](https://doi.org/10.1145/3191513) | Tom Mitchell et al. | CACM 61(5), 2018 | A system reads the web since 2010. It invents new relational predicates to extend its ontology. | (a), (b) lineage of "growing schema" |
| [LLM-empowered KG construction: a survey](https://arxiv.org/abs/2510.20345) | not confirmed | arXiv 2510.20345, Oct 2025 | It covers competency-question pipelines (CQ → TBox → population). | (b) background |

Summary for (b), second half: schema induction, schema expansion and frequency-based promotion are
well covered. Feng et al. and IN-VKGE already go from questions to schema. **We found no paper that
keeps unanswered questions as unbound typed schema elements across a document stream and binds them
when a later document arrives.** HopRAG does the linking in a static corpus. Pro-Memory (Section 4)
keeps pending intentions linked to evidence, not questions.

---

## 3. Graph RAG with many structures, views, routing, and adaptive structure

| Paper | Authors | Venue / arXiv / date | One-line summary | Overlap |
|---|---|---|---|---|
| [From Local to Global: A Graph RAG Approach (GraphRAG)](https://arxiv.org/abs/2404.16130) | Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Metropolitansky, Ness, Larson | arXiv 2404.16130, Apr 2024 | Entity KG plus community summaries for global questions. | (a), baseline |
| [RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html) | Sarthi et al. (Stanford) | ICLR 2024; arXiv 2401.18059 | Recursive cluster-and-summarize tree. | (d) one alternative structure |
| [HippoRAG](https://arxiv.org/abs/2405.14831) | Gutiérrez et al. | arXiv 2405.14831, May 2024 (NeurIPS 2024: not confirmed) | OpenIE KG plus Personalized PageRank for single-step multi-hop retrieval. | baseline |
| [HippoRAG 2: From RAG to Memory](https://proceedings.mlr.press/v267/gutierrez25a.html) | Gutiérrez, Shu, Qi, Zhou, Su | ICML 2025 (PMLR 267); arXiv 2502.14802 | Phrase + passage graph, recognition-memory filter, PPR. Framed as non-parametric continual learning. | (a), (c), **benchmark setting** |
| [LightRAG](https://arxiv.org/abs/2410.05779) | Guo et al. | EMNLP 2025 Findings; arXiv 2410.05779 | Dual-level (local/global) graph retrieval with incremental update. | (a) |
| [StructRAG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) | Zhuoqun Li et al. | ICLR 2025; arXiv 2410.08815 | A DPO-trained router picks table, graph, algorithm, catalogue or chunk per task; a structurizer builds it at inference time. | **(d) closest on "choose the representation"** |
| [RAG vs. GraphRAG: A Systematic Evaluation](https://arxiv.org/abs/2502.11371) | Haoyu Han et al. | arXiv 2502.11371, Feb 2025 | RAG and GraphRAG are complementary; "Selection" and "Integration" hybrids. | (d) |
| [When to use Graphs in RAG (GraphRAG-Bench)](https://arxiv.org/abs/2506.05690) | Zhishang Xiang et al. | ICLR 2026 (per arXiv v3 label); arXiv 2506.05690 | Graphs help only for some task profiles; the authors call adaptive graph construction crucial. | (c) evaluation, (d) motivation |
| [GraphRAG-Bench (domain reasoning)](https://arxiv.org/abs/2506.02404) | Xiao, Dong, Zhou, Dong, Zhang, Yin, Sun, Huang | arXiv 2506.02404, Jun 2025 | College-level multi-hop questions in 16 fields; evaluates construction, retrieval and reasoning. | evaluation |
| [EA-GraphRAG: Use Graph When It Needs](https://arxiv.org/abs/2602.03578) | Su Dong, Qinggang Zhang, Yilin Xiao, Shengyuan Chen, Chuang Zhou, Xiao Huang | arXiv 2602.03578, Feb 2026 | A syntactic complexity score routes each query to dense RAG or graph RAG. | (d) routing |
| [Think Parallax (ParallaxRAG)](https://arxiv.org/abs/2510.15552) | Liu, Bai, Zeng | ACL 2026; arXiv 2510.15552 | Multi-view KG-RAG with head-specific views. A single (monolithic) representation lets early-hop signals interfere with deeper hops. | **(c) "representation interference"**, (d) views |
| [PropRAG](https://arxiv.org/abs/2504.18070) | Jingjin Wang, Jiawei Han | EMNLP 2025; arXiv 2504.18070 | Proposition graph plus LLM-free beam search over proposition paths. | benchmark leader (Section 7) |
| [HGRAG: Cross-Granularity Hypergraph RAG](https://ojs.aaai.org/index.php/AAAI/article/view/40623) | Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, Ning Jiang | AAAI 2026; arXiv 2508.11247 | Entity–passage hypergraph with semantic and structural enhancement. | benchmark leader (Section 7) |
| [LinearRAG](https://arxiv.org/abs/2510.10114) | Zhuang, Chen, Xiao, Zhou, Zhang, Chen, Zhang, Huang (PolyU) | arXiv 2510.10114, Oct 2025 | Relation-free "Tri-Graph" (entity, sentence, passage); indexing time drops by more than 77%. | (d) a cheap view type |
| [EraRAG](https://arxiv.org/abs/2506.20963) | Fangyuan Zhang et al. | arXiv 2506.20963, Jun 2025 | LSH multi-layer graph for growing corpora; local insert; merge-and-split of groups by size thresholds. | (a) streaming, (d) merge/split (structural, not performance-driven) |
| [EvoRAG](https://arxiv.org/abs/2604.15676) | Zhenbo Fu et al. | ICDE 2027 (per arXiv comments); arXiv 2604.15676, Apr 2026 | Response feedback flows back to retrieved paths and then to single triplets, which the system refines. | **(d) feedback-driven graph change** |
| [EvoGraph-R1](https://arxiv.org/abs/2607.12764) | Jiashi Lin et al. (15 authors) | CVPR 2026; arXiv 2607.12764 | An agent with a GraphEdit action reshapes a multimodal knowledge hypergraph from feedback. | (d), (P) |
| [AutoGraph-R1](https://aclanthology.org/2026.acl-long.1070/) | Hong Ting Tsang et al. | ACL 2026; arXiv 2510.15339 | RL trains the KG constructor with a reward from the graph's use in RAG. It uses task QA data (HotpotQA, MuSiQue) for training. | (d), (P); contrast: QBranch uses no task labels |
| [SelfGraphRAG](https://arxiv.org/abs/2608.25123) | Ben Lagnese, Manas Gaur | arXiv 2608.25123, Aug 2026 (under review) | The LLM generates QA pairs from graph substructures and trains a graph retriever with them. | (d) label-free self-questions (for training, not for structure choice) |
| [ERM: RAG without Forgetting](https://arxiv.org/abs/2602.05152) | Hu, Li, Ramakrishnan, Zhao | arXiv 2602.05152, Feb 2026 | Correctness-gated feedback updates the document keys of the index; the update persists. | (d) |
| [KET-RAG](https://arxiv.org/abs/2502.09304) | Huang, Zhang, Xiao | arXiv 2502.09304, Feb 2025 (KDD 2025: not confirmed) | KG skeleton on key chunks plus a text–keyword bipartite graph; it cuts indexing cost. | (d) compute budget |

Summary for (d): StructRAG chooses a structure per query at inference time. EvoRAG, EvoGraph-R1, ERM
and EvoOntology change the structure from feedback. EraRAG merges and splits by size. **We found no
paper that (1) scores keep / merge / create-view actions on the system's own inferred questions,
(2) applies a no-regression gate on earlier questions, and (3) compares the actions at a matched
compute budget.** EvoOntology is the nearest (paired evaluation before an edit is accepted), but it
works on data-agent ontologies, not on a document stream.

---

## 4. Online, streaming and continual KG memory for agents

| Paper | Authors | Venue / arXiv / date | One-line summary | Overlap |
|---|---|---|---|---|
| [MemGPT](https://arxiv.org/abs/2310.08560) | Charles Packer et al. | arXiv 2310.08560, Oct 2023 | OS-style memory tiers; the LLM pages data in and out of context. | (a) |
| [Zep (Graphiti)](https://arxiv.org/abs/2501.13956) | Preston Rasmussen et al. (Zep AI) | arXiv 2501.13956, Jan 2025 | Bi-temporal KG with episode, entity and community subgraphs; incremental update; episodes keep the source. | **(a) unified graph + source store** |
| [A-MEM](https://papers.nips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html) | Xu, Liang, Mei, Gao, Tan, Zhang | NeurIPS 2025; arXiv 2502.12110 | Zettelkasten-style notes; new notes link to old notes and update them. | (a) |
| [Mem0 / Mem0g](https://arxiv.org/abs/2504.19413) | Chhikara, Khant, Aryan, Singh, Yadav | arXiv 2504.19413, Apr 2025 | Extract–update memory pipeline with a graph variant; tested on LoCoMo. | (a) |
| [Memory-R1](https://aclanthology.org/2026.acl-long.583/) | Sikuan Yan et al. | ACL 2026; arXiv 2508.19828 | RL trains a memory manager with ADD / UPDATE / DELETE / NOOP actions. | (a), (P) |
| [Mem-α](https://arxiv.org/abs/2509.25911) | Yu Wang et al. | arXiv 2509.25911, Sep 2025 | RL trains an agent to build a core / episodic / semantic memory with tools. | (a), (P) |
| [AriGraph](https://www.ijcai.org/proceedings/2025/0002) | Anokhin, Semenov, Sorokin, Evseev, Kravchenko, Burtsev, Burnaev | IJCAI 2025; arXiv 2407.04363 | The agent builds a semantic + episodic KG "world model" while it explores; also tested on multi-hop QA. | (a), **(W)** |
| HippoRAG 2 | see Section 3 | ICML 2025 | Non-parametric continual learning framing. | (a) |
| [Pro-Memory: Prospective Memory for Proactive LLM Agents](https://arxiv.org/abs/2611.05731) | Bei, Liu, Wei, Zhang, Zeng, Yu, Hamann, He, Tong | arXiv 2611.05731, Nov 2026 | Keeps pending objectives linked to their evidence; it updates them when evidence of completion arrives. | **(b) open-item tracking** (intentions, not questions) |
| [StreamingQA](https://proceedings.mlr.press/v162/liska22a.html) | Liška et al. (DeepMind) | ICML 2022; arXiv 2205.11388 | Dated questions over 14 years of dated news; measures adaptation and forgetting. | evaluation for the stream |
| [StreamMemBench](https://arxiv.org/abs/2606.14571) | Guanming Liu et al. | EMNLP 2026 Findings; arXiv 2606.14571 | Streaming evaluation of agent memory on EgoLife; eight memory systems often fail to reuse evidence. | evaluation |
| [GAM: Hierarchical Graph-based Agentic Memory](https://arxiv.org/abs/2604.12285) | not confirmed | arXiv 2604.12285, Apr 2026 | Hierarchical graph memory that trades new input against old knowledge. | (a) |

Summary for (a): a persistent agent memory with a KG, provenance and tools is standard in 2025–2026.
The **question agenda** (resolved and open questions as first-class objects) is the only part of (a)
that we did not find in agent-memory work. Pro-Memory is the nearest, for intentions.

---

## 5. LLM-built KGs or text memory as a "world model"

| Paper | Authors | Venue / arXiv / date | One-line summary | Overlap |
|---|---|---|---|---|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | Yann LeCun | OpenReview position paper, 2022 (link and date not confirmed: the page blocked the fetch) | JEPA: learn abstract representations and predict their change; LLMs alone are not world models. | (W) the critique |
| [AMI Labs funding news](https://siliconangle.com/2026/03/10/yann-lecuns-new-startup-ami-labs-raises-1-03b-train-world-models/) | SiliconANGLE | News, 2026-03-10 | LeCun's AMI Labs raised $1.03B to build JEPA world models. | (W) venue context |
| [Critiques of World Models](https://arxiv.org/abs/2507.05169) | Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu | arXiv 2507.05169, Jul 2025 | A world model simulates the actionable possibilities of the world for reasoning and acting; proposes PAN. | (W) definition to answer |
| [Evaluating the World Model Implicit in a Generative Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f6a6317bada76b26a4f61bb70a7db59-Abstract.html) | Vafa, Chen, Rambachan, Kleinberg, Mullainathan | NeurIPS 2024; arXiv 2406.03689 | Myhill–Nerode-based tests show that "good" generative models can have incoherent world models. | (W) evaluation method |
| [Learning KG-based World Models of Textual Environments (Worldformer)](https://arxiv.org/abs/2106.09608) | Ammanabrolu, Riedl | NeurIPS 2021; arXiv 2106.09608 | Predicts the change of a KG world state from text actions in text games. | (W), (a) |
| AriGraph | see Section 4 | IJCAI 2025 | KG world model with episodic memory. | **(W) closest on the title claim** |
| [WALL-E 2.0](https://arxiv.org/abs/2504.15785) | Siyu Zhou, Tianyi Zhou, Yang, Long, Ye, Jiang, Zhang | arXiv 2504.15785, Apr 2025 | Neurosymbolic learning of action rules, KGs and scene graphs aligns an LLM world model to an environment. | (W) |
| [PoE-World](https://arxiv.org/abs/2505.10819) | Piriyakulkij et al. | NeurIPS 2025; arXiv 2505.10819 | World model as a product of LLM-written programmatic experts; used for planning in Atari. | (W) symbolic world models |
| [Neurosymbolic Graph Enrichment for Grounded World Models](https://arxiv.org/abs/2411.12671) | De Giorgis, Gangemi, Russo | Information Processing & Management, 2025; arXiv 2411.12671 | LLMs add implicit knowledge to KGs to make "grounded world models". | (W) |
| [Knowledge Graphs as Grounded World Models for Neuro-Symbolic AI](https://www.researchgate.net/publication/400386797_Knowledge_Graphs_as_Grounded_World_Models_for_Neuro-Symbolic_Artificial_Intelligence) | Sören Auer (per search snippet) | ResearchGate, Feb 2026 (**not confirmed**: no arXiv copy found; the page blocked the fetch) | Argues that a KG is a stateful world model that an LLM writes to and reads from; RAG only approximates grounding. | (W) |
| [Towards a Belief-Based World Model for LLM Agents](https://arxiv.org/abs/2609.00455) | S. Kumar, H. Kumar, Ahuja, Jha | arXiv 2609.00455, Aug 2026 | A queryable belief of what is known and what is uncertain helps an LLM policy under partial observability. | (W), (b) open slots as uncertainty |
| [MemWM: Memory-Augmented Text-Based World Model](https://arxiv.org/abs/2608.07107) | not confirmed | arXiv 2608.07107, Aug 2026 | World memory of transition rules, state caches and hard facts helps next-state prediction. | (W), (a) |

Summary for (W): "KG as world model" already has a title-level precedent (Worldformer 2021,
AriGraph 2025). In those papers, the world model predicts the effect of actions. A document KG has
no actions by default. A reviewer will ask: "What does this model predict?" The plan's claim C2
(open slots predict the facts that later documents bring) is the right answer. Make it a measured,
central result, not a side metric.

---

## 6. Planning formulations of index or memory management

| Paper | Authors | Venue / arXiv / date | One-line summary | Overlap |
|---|---|---|---|---|
| [MBA-RAG](https://arxiv.org/abs/2412.01572) | Xiaqiang Tang et al. | arXiv 2412.01572, Dec 2024 (COLING 2025: not confirmed) | Multi-armed bandit picks a retrieval strategy per query; the reward penalizes extra retrieval steps. | (P) bandit over strategies |
| [ERRAND: Budgeted Maintenance of Agent Memory](https://arxiv.org/abs/2609.29545) | Beining Wu, Zihao Ding, Jun Huang | arXiv 2609.29545, Sep 2026 | Rechecks of stored memory compete with task actions for one budget; a running "wage" sets when a recheck pays. | **(P) closest on budgeted memory maintenance** |
| [Inference-Time Budget Control for LLM Search Agents](https://arxiv.org/abs/2605.05701) | Zhengru Fang et al. | arXiv 2605.05701, May 2026 | Scores retrieval, decomposition and answer actions by value per unit of budget. | (P) query-time only |
| AutoGraph-R1, Agentic-KGR, Memory-R1, Mem-α | see Sections 3 and 4 | 2025–2026 | RL policies for KG construction or memory operations. | (P) learned policies, not planning |
| [Budget-aware Index Tuning with Reinforcement Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/06/mcts-full.pdf) | Microsoft Research (authors and venue not confirmed) | 2022 | Database index configuration search as an MDP solved with MCTS under a budget. | (P) classical analogue for "index maintenance as budgeted planning" |
| [Momoka-RAG](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.183/) | not confirmed | ACL 2026 Findings | MCTS finds links among chunks and builds semantic paths for long documents. | (P), (d) |
| KET-RAG | see Section 3 | 2025 | Indexing cost versus quality trade-off. | (P) budget |

Summary for (P): we found no paper that formulates "split / merge / keep a view of a document KG"
as a sequential decision problem under a compute budget. ERRAND and MBA-RAG are the nearest. ICAPS
reviewers will want a formal model (states, actions, costs, transitions, objective), a planner, and
a comparison with greedy, bandit and RL policies.

---

## 7. Benchmarks and published numbers

### 7.1 HippoRAG 2 setting (direct comparison)

Setting: 1,000 questions per dataset; corpus = all supporting and distractor passages of these
questions; retriever NV-Embed-v2; reader Llama-3.3-70B-Instruct. PropRAG and HGRAG copy the baseline
rows from HippoRAG 2 and use the same encoder and reader. QA metric: F1. Retrieval metric: passage
Recall@5. GraphRAG and LightRAG have no passage recall.

| Method | Source | MuSiQue F1 | 2Wiki F1 | HotpotQA F1 | MuSiQue R@5 | 2Wiki R@5 | HotpotQA R@5 |
|---|---|---|---|---|---|---|---|
| NV-Embed-v2 (dense) | HippoRAG 2, Tables 2–3 | 45.7 | 61.5 | 75.3 | 69.7 | 76.5 | 94.5 |
| RAPTOR | HippoRAG 2 | 28.9 | 52.1 | 69.5 | 57.8 | 66.2 | 86.9 |
| GraphRAG | HippoRAG 2 | 38.5 | 58.6 | 68.6 | – | – | – |
| LightRAG | HippoRAG 2 | 1.6 | 11.6 | 2.4 | – | – | – |
| HippoRAG | HippoRAG 2 | 35.1 | 71.8 | 63.5 | 53.2 | 90.4 | 77.3 |
| HippoRAG 2 | [arXiv 2502.14802](https://arxiv.org/abs/2502.14802) | 48.6 | 71.0 | 75.5 | 74.7 | 90.4 | 96.3 |
| PropRAG (Lmax = 3, B = 4) | [arXiv 2504.18070](https://arxiv.org/abs/2504.18070), Tables 1–2 | **53.9** | 75.3 | 76.1 | **78.3** | **94.1** | **97.4** |
| HGRAG | [arXiv 2508.11247](https://arxiv.org/abs/2508.11247), Tables 1–2 | 53.8 | **78.3** | **76.8** | 74.1 | 93.0 | 95.5 |

HippoRAG 2 also reports NQ, PopQA, LV-Eval and NarrativeQA. Its average F1 over 7 datasets is 59.8
(NV-Embed-v2: 57.0).

### 7.2 Other settings (not directly comparable)

| Method | Setting | Numbers (method vs HippoRAG 2 in the same paper) | Source |
|---|---|---|---|
| LinearRAG | GPT-4o-mini reader and judge; all-mpnet-base-v2; top-5; 1,000 questions | Contain-Acc / GPT-Acc: HotpotQA 64.3 / 66.5 vs 62.9 / 64.3; 2Wiki 70.2 / 63.7 vs 62.7 / 55.0; MuSiQue 33.9 / 37.0 vs 31.0 / 35.0 | [arXiv 2510.10114](https://arxiv.org/abs/2510.10114) |
| MemGraphRAG | LLM-Acc; 1,000 validation questions each; reader not confirmed | HotpotQA 71.6 vs 67.2; 2Wiki 69.8 vs 57.9; MuSiQue 37.9 vs 38.3 | [arXiv 2606.00610](https://arxiv.org/abs/2606.00610) |

Other systems claim gains over HippoRAG 2 on these datasets: Youtu-GraphRAG, AutoGraph-R1, GFM-RAG
(NeurIPS 2025), TIGRAG ([arXiv 2606.30093](https://arxiv.org/abs/2606.30093)) and G³RAG
([arXiv 2609.19622](https://arxiv.org/abs/2609.19622)). I did not confirm their numbers.

### 7.3 Other standard benchmarks

- Graph RAG: GraphRAG-Bench (two different benchmarks with this name: arXiv 2506.05690 and
  2506.02404), MultiHop-RAG, LiHuaWorld (QCG-RAG), NQ, PopQA, LV-Eval, NarrativeQA.
- Agent memory: LongMemEval, LoCoMo, Deep Memory Retrieval (Zep), StreamMemBench.
- Streams: StreamingQA (dated news). The plan's FinanceBench stream also has real dates.
- Growing-corpus protocol: EraRAG uses 50% of the corpus as the start and adds the other 50% later.

---

## 8. Novelty verdict per QBranch part

| Part | Risk | Closest work | What is still new (based on this scan) |
|---|---|---|---|
| (a) Agent with persistent memory and tools | **High** | Zep/Graphiti, A-MEM, Mem0g, HippoRAG 2, MemGraphRAG, Youtu-GraphRAG, AriGraph | Only the question agenda (open and resolved questions as objects). Do not claim the architecture as a contribution. |
| (b) Question-driven schema evolution with open slots | **Medium** | HopRAG (raiser → solver questions), PIKE-RAG, QCG-RAG, Feng et al. (CQ → schema), IN-VKGE, MemGraphRAG, DIAL-KG, AutoSchemaKG, EDC | Unbound typed schema elements that persist over a document stream and bind later, plus a test that they predict later facts. "Questions → schema" alone is medium-high risk. |
| (c) Evidence-grounded answering + independent judge | **High** | RAGChecker ([NeurIPS 2024 D&B](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract.html)), RAG error taxonomy ([Leung et al., EACL 2026, arXiv 2510.13975](https://arxiv.org/abs/2510.13975)), PAGE-RAG ([arXiv 2607.19301](https://arxiv.org/abs/2607.19301)) abstention, ParallaxRAG | Use (c) as the measurement tool. The "representation interference" label is new as a name, but ParallaxRAG describes a similar effect. |
| (d) Performance-driven view management | **Medium** | StructRAG, EvoRAG, EvoOntology, ERM, EraRAG, AutoGraph-R1, SelfGraphRAG | The combination: label-free objective (own inferred questions) + no-regression gate on earlier questions + matched compute + keep / merge / create-view actions on a growing KG. |
| (P) Planning angle (ICAPS) | **Low novelty risk, high rigor risk** | ERRAND, MBA-RAG, Budget-aware index tuning (MCTS), RL memory managers | No paper found that plans view split / merge / keep under a compute budget. ICAPS needs a formal model and planner baselines. |
| (W) World-model framing (ICWM) | **Medium-high (framing risk)** | Worldformer, AriGraph, Belief-Based WMs, De Giorgis et al., Xing et al. critique, LeCun | The title claim is not new. The new, testable part is prediction: open slots forecast what later documents will say. |

Overall: the paper is safe only as a **combination** paper. The core claim must be: "a KG plus a
question agenda that predicts future evidence and restructures itself on its own questions, without
task labels, at a fixed compute budget." Each piece alone has close prior work.

---

## 9. Papers that a reviewer will most likely cite against QBranch

1. **HippoRAG 2** (ICML 2025): the same benchmark setting and the "non-parametric continual learning"
   claim. Reviewers will ask for a direct comparison.
2. **HopRAG** (ACL 2025 Findings): LLM pseudo-queries link a passage that raises a question to a
   passage that answers it. This is the strongest prior for open slots.
3. **StructRAG** (ICLR 2025): it already chooses among table, graph, tree and chunk structures. This
   is the strongest prior for views.
4. **MemGraphRAG** (KDD 2026): shared memory, candidate-to-stable schema promotion and conflict
   resolution. It beats HippoRAG 2 on 2 of 3 datasets (LLM-Acc).
5. **AutoSchemaKG** (2025) together with **EDC** (EMNLP 2024): schema induction and canonicalization.

Second-tier papers to cite:

- For (d): EvoRAG (ICDE 2027), EvoOntology (2026).
- For (b): PIKE-RAG (2025), Feng et al. (2024), IN-VKGE (IJCAI 2026).
- For ICWM: AriGraph (IJCAI 2025), Worldformer (NeurIPS 2021).
- For ICAPS: ERRAND (2026), MBA-RAG (2024).

---

## 10. What this scan means for the plan

1. Add HopRAG and PropRAG (or HGRAG) as baselines next to HippoRAG 2. They hold the best numbers in
   the HippoRAG 2 setting (Section 7.1).
2. Add an ablation that removes the open slots (keep only answered questions). This isolates the new
   part of (b) from PIKE-RAG-style question indexing.
3. Make C2 (slot binding predicts later facts) a main result. It is the answer to "what does the
   world model predict?" for ICWM reviewers.
4. Report view management against a StructRAG-style router and against "always re-build", at the
   same token budget.
5. For ICAPS: write the MDP (state = graph + agenda + views; actions = keep, merge, create view;
   cost = tokens; reward = gain on inferred questions with a regression penalty). Compare a planner
   with greedy and bandit policies.
6. Schema explosion (1,220 relation names, plan Section 4) blocks C2 and C3. EDC-style canonicalization
   is the standard fix. Cite it and use it.

---

## 11. Limits of this scan

- Search coverage: web search and arXiv pages only. I did not search Semantic Scholar citation graphs.
- I read only the abstract page for most 2026 preprints. Method details can differ from the abstract.
- I did not confirm the numbers of Youtu-GraphRAG, AutoGraph-R1, GFM-RAG, TIGRAG and G³RAG.
- ACE-GraphRAG (arXiv 2608.01269) is withdrawn, so this scan does not use it.
