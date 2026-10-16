# system-one-survey

Research about System One models: AI models that return typed answers with probabilities instead of text. Examples: TypeSafe Jev, the OpenAI Decisions API, Microsoft-Decision-1 and Perplexity pplx-decider.

Two papers:

- **Paper 1 (now):** an experiments paper. Target: ARR on 2027-01-04 (ACL 2027).
- **Paper 2 (from January 2027):** the full survey. It cites paper 1.

Status on 2026-10-16: all experiments of paper 1 ran overnight, and the draft of paper 1 is `paper1/main.pdf` (ACL format, 8 main pages). Read `notes/2026-10-16-paper1-results.md` first.

## Start here

0. `notes/2026-10-16-paper1-results.md`: what ran overnight, the findings and what you must check. Then `paper1/main.pdf`.
1. `notes/2026-10-15-paper1-plan.md`: the plan for paper 1.
2. `notes/2026-10-15-experiment-design.md`: the hypotheses and the experiments E0 to E8.
3. `notes/2026-10-15-api-audit.md`: the results of E0 (access, schemas, latency, cost).
4. `notes/2026-10-15-e7-first-results.md`: the first results (Jev 1.13 against 1.14, and a noise control).
5. `notes/2026-10-15-model-landscape.md`: the facts about the systems, the benchmarks and the papers, with sources.
6. For paper 2: `notes/2026-10-15-survey-plan.md`, `notes/2026-10-15-survey-paper-guide.md` and `notes/2026-10-15-venues.md`.

## Files

| File | What it contains |
|---|---|
| `notes/2026-10-15-paper1-plan.md` | Paper 1: venues, scope, outline, timeline, budget and the papers to read first. |
| `notes/2026-10-15-api-audit.md` | E0: which System One models work through OpenRouter, and what they return. |
| `src/tasks.py` | The datasets, the question cards and the fixed samples of items. |
| `src/openrouter.py` | One call to the OpenRouter decisions API, with retries. |
| `src/run.py` | Runs models on tasks. Writes one JSON line for each call to `runs/<tag>/`. |
| `analysis/audit.py` | Summarizes one run as a table (used for E0). |
| `analysis/compare_versions.py` | Compares two runs on the same items (used for E7). |
| `notes/2026-10-15-e7-first-results.md` | E7: Jev 1.13 against 1.14, and the noise control. |
| `notes/2026-10-15-survey-plan.md` | Paper 2: scope, survey questions, definition, taxonomy, outline, timeline, budget, risks, actions and glossary. |
| `notes/2026-10-15-survey-paper-guide.md` | A guide for team members who did not write a survey before. |
| `notes/2026-10-15-model-landscape.md` | About 25 systems, the leaderboards, 21 papers, the disagreements and the OpenRouter ids. |
| `notes/2026-10-15-experiment-design.md` | Hypotheses H1 to H7, systems, baselines, task suite, metrics, experiments, budget and code plan. |
| `notes/2026-10-15-venues.md` | Journals and conferences, the next deadlines, the preprint rules and our recommendation. |
| `literature/protocol.md` | The review protocol: sources, search strings, inclusion rules, screening and quality checks. |
| `literature/seed-papers.md` | The first list of papers by research line, with BibTeX keys. |
| `src/llm.py` | System Two baselines (structured outputs) and the letter readout of open backbones, through OpenRouter. |
| `src/perturb.py` | The 12 perturbations of E3 and their texts. |
| `src/run_e3.py` | Runs the perturbed requests of E3. |
| `src/local_readout.py` | The letter readout of Qwen3.5-4B with MLX on this Mac. |
| `src/fetch_arxiv.py` | Makes the fresh arXiv task (`fixtures/arxiv-fresh-2026-10.jsonl`). |
| `analysis/e2_calibration.py` ... `e8_latency.py` | One script for each experiment. Each writes `results/*.csv` and `results/*.md`. |
| `analysis/make_paper_assets.py` | Makes the tables, the figures and `paper1/numbers.tex` from `results/`. |
| `analysis/status.py` | The progress and the cost of runs. |
| `paper1/main.tex` | Paper 1 (ACL format). All numbers come from `paper1/numbers.tex`. |
| `notes/2026-10-16-related-work-facts.md` | Verified protocols and numbers of 10 related papers, with quotes. |
| `notes/2026-10-16-internal-review.md` | An internal ARR-style review of the draft. |
| `notes/2026-10-16-internal-review-2.md` | The second review: the status of each finding of the first review, and new findings. |
| `notes/2026-10-16-overnight-log.md` | The log of the overnight work: decisions, problems and costs. |
| `paper/main.tex` | The survey skeleton for paper 2 (ACM `acmsmall`, the CSUR and TIST format). |
| `paper/references.bib` | 119 entries. The arXiv entries come from the arXiv API. |
| `paper/main.pdf` | The built draft (11 pages). |

## Build paper 1

```
for s in e2_calibration e2_metric_replication e3_robustness e5_training e6_cascade e7_versions e8_latency s2_details; do uv run python analysis/$s.py > results/$s.md; done
uv run python analysis/e2_reliability.py
uv run python analysis/make_paper_assets.py
cd paper1 && latexmk -pdf -outdir=tmp main.tex && cp tmp/main.pdf main.pdf
```

## Build the survey skeleton (paper 2)

```
cd paper
latexmk -pdf -outdir=tmp main.tex
cp tmp/main.pdf main.pdf
```

## Rules

- Write in ASD-STE100 Simplified Technical English.
- Write "vendor-reported" next to each number from a vendor.
- Record the access date of each web source. Vendors change their pages.
- Use the terms in the glossary of the plan (section 17).
- Do not add a reference that you did not open yourself.

## Run the experiments

- `uv sync` installs the dependencies.
- Copy `.env.example` to `.env` and set your OpenRouter key. Git ignores `.env`, `data/` and `runs/`.
- Always set `PYTHONPATH=src`. Then `src/sitecustomize.py` makes Python try IPv4 first, because IPv6 connections time out on this network.

```
# E0: 5 items from each task on two models
PYTHONPATH=src uv run python src/run.py --tag e0 --n 5 --models typesafe/jev-1.13,microsoft/microsoft-decision-1
uv run python analysis/audit.py runs/e0
```

- `runs/` is not in git (1.6 GB on 2026-10-16). Make a backup copy: nobody can repeat a run on an old version after the vendor removes it.
