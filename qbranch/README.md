# QBranch: question-driven neurosymbolic world model for RAG

Team: Prof. Kaushik Roy (University of Alabama), Ashraf Abdul-Muumin, Aniruddha Chattopadhyay.
Started on 2026-11-06. The figure is `figures/2026-11-06-qbranch-architecture.png`.

Read these files first:

1. `notes/2026-11-06-kickoff-meeting.md`: the idea, the decisions and the next steps.
2. `notes/2026-11-09-research-plan.md`: the claims (C1–C3), the protocol, the venues and the timeline.
3. `notes/2026-11-09-pilot-log.md`: the results so far.
4. `notes/2026-11-09-related-work-scan.md`: the closest work and the novelty risk.

## Set-up

```bash
cp .env.example .env        # then put the lab OpenRouter key in .env (git ignores .env)
uv sync
```

The data is in `data/hipporag2/`, from the `osunlp/HippoRAG_2` dataset on Hugging Face.
Git ignores `data/` and `runs/`.

## Run the pilot

```bash
uv run python src/run_build.py --dataset hotpotqa --n 100 --variant qbranch --batches 5
uv run python src/run_build.py --dataset hotpotqa --n 100 --variant openie --batches 5
uv run python src/run_answer.py --dataset hotpotqa --n 100
uv run python analysis/anticipation.py runs/hotpotqa_n100          # C2, raw relation names
```

The C2 analysis with `--norm` needs `runs/<slice>/relmap_qbranch.json`, made by `normalize.normalize`.

## Code

| File | Part of the figure | What it does |
|---|---|---|
| `src/llm.py` | – | OpenRouter chat and embeddings, a disk cache (`runs/llm_cache.sqlite`), and cost counts |
| `src/data.py` | inputs | HippoRAG 2 datasets, pilot slices, the stream batches |
| `src/prompts.py` | (b), (c) | build, reader and judge prompts |
| `src/build.py` | (b) | online build: infer questions, name the schema, bind facts, keep open slots |
| `src/graph.py` | (a) | unified graph, question agenda, open slots |
| `src/normalize.py` | (a) schema normalization | canonical relations with a direction |
| `src/answer.py` | (c) | dense, oracle, closed-book and graph retrieval, and the reader |
| `src/evaluate.py` | (c) judge | EM, F1, LLM judge, passage recall |
| `analysis/anticipation.py` | world-model test (C2) | do open slots predict later facts? |

`src/sitecustomize.py` makes Python try IPv4 first, because IPv6 times out on this network.
