# QBranch: question-driven neurosymbolic world model for RAG

Team: Prof. Kaushik Roy (University of Alabama), Ashraf Abdul-Muumin, Aniruddha Chattopadhyay.
Started on 2026-11-06. The figure is `figures/2026-11-06-qbranch-architecture.png`.

Read these files first:

1. `notes/2026-11-06-kickoff-meeting.md`: the idea, the decisions and the next steps.
2. `notes/2026-11-09-research-plan.md`: the claims (C1–C3), the protocol, the venues and the timeline.
3. `notes/2026-11-09-pilot-log.md`: the pilot on 100-question slices.
4. `notes/2026-11-10-results.md`: the full runs, path planning, C2 on four streams (read this first for results).
5. `notes/2026-11-09-related-work-scan.md`: the closest work and the novelty risk.

## Set-up

```bash
cp .env.example .env        # then put the lab OpenRouter key in .env (git ignores .env)
uv sync
```

The data is in `data/hipporag2/`, from the `osunlp/HippoRAG_2` dataset on Hugging Face, and in
`data/multihoprag/`, from the `yixuantt/MultiHopRAG` dataset (dated news, for C2).
Git ignores `data/` and `runs/`.

## Run the pilot

```bash
uv run python src/run_build.py --dataset hotpotqa --n 100 --variant qbranch --batches 5
uv run python src/run_build.py --dataset hotpotqa --n 100 --variant openie --batches 5
uv run python src/run_answer.py --dataset hotpotqa --n 100
uv run python analysis/anticipation.py runs/hotpotqa_n100          # C2, raw relation names
```

The C2 analysis with `--norm` needs `runs/<slice>/relmap_qbranch.json`, made by `normalize.normalize`.

## Run the full setting

```bash
uv run python src/run_build.py --dataset musique --n 0 --variant qbranch --batches 10 --workers 32
uv run python src/run_answer.py --dataset musique --n 0 --systems dense,dense_k10,dense_plan,graph_qbranch,graph_qbranch_plan
uv run python analysis/compare.py runs/musique_nall graph_qbranch:dense --exclude runs/musique_n100
uv run python analysis/c2_semantic.py runs/musique_nall --n 600     # C2 with an LLM check
uv run python analysis/bridges.py runs/musique_nall dense_plan graph_qbranch_plan
```

`--n 0` means all 1,000 questions and the full pooled corpus. For `multihoprag`, `--n 0` means the fixed
300-question sample and all 4,574 passages, in time order. Do not run two plan systems on the same
questions in two processes at the same time: the planner calls must hit the cache, so that all systems
get the same plan.

## Code

| File | Part of the figure | What it does |
|---|---|---|
| `src/llm.py` | – | OpenRouter chat and embeddings, a disk cache (`runs/llm_cache.sqlite`), and cost counts |
| `src/data.py` | inputs | HippoRAG 2 datasets, pilot slices, the stream batches |
| `src/prompts.py` | (b), (c) | build, reader and judge prompts |
| `src/build.py` | (b) | online build: infer questions, name the schema, bind facts, keep open slots |
| `src/graph.py` | (a) | unified graph, question agenda, open slots |
| `src/normalize.py` | (a) schema normalization | canonical relations with a direction |
| `src/answer.py` | (c) | dense, oracle, closed-book and graph retrieval (with anchors), and the reader |
| `src/plan.py` | (c) step 2 | path-planned answering: sub-questions, one retrieval and one read per hop |
| `src/evaluate.py` | (c) judge | EM, F1, LLM judge, passage recall |
| `analysis/anticipation.py` | world-model test (C2) | do open slots predict later facts? (same relation name) |
| `analysis/c2_semantic.py` | world-model test (C2) | the same, with an LLM check and a random-question baseline |
| `analysis/c2_kinds.py` | C2 | open slots by kind: background fact or open event |
| `analysis/compare.py` | – | paired bootstrap between systems, held-out view, by question type |
| `analysis/bridges.py` | (c) step 2 | MuSiQue: does the chain find the gold bridge entities? |
| `analysis/route.py` | (d) | label-free routing by the plan length |

`src/sitecustomize.py` makes Python try IPv4 first, because IPv6 times out on this network.
