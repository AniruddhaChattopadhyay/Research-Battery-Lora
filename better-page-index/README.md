# better-page-index

Research code: test PageIndex on CLEVRER, and compare it with full context and with a small dual-view graph agent.

## Start here

1. `notes/2026-10-12-pilot-results-brief.md`: the results and what they mean.
2. `notes/2026-10-12-presentation-guide.md`: how to present them.
3. `notes/2026-10-10-pageindex-clevrer-analysis.md`: how PageIndex works, and facts about CLEVRER.

## Data flow

```
data/clevrer/annotations/*.json        data/clevrer/validation.json
(one file per video: objects,          (questions and answers)
 states in each frame, collisions)               |
          |                                      |
     src/clevrer.py  (Run: one video)       src/qa.py  (248 pilot questions, shared
          |                                      |      instructions, answer parsing, scoring)
    +-----+-------------------+                  |
    |                         |                  |
src/scene_report.py      src/world.py (graph + 2 operators)
    |                         |
runs/pilot/plant_log_30_runs.pdf (317 pages)    |
    |               |                            |
src/run_pageindex.py   src/run_fullcontext.py   src/run_graph_agent.py
(PageIndex SDK:        (pages of the named run  (LLM with 6 graph tools;
 tree index + agent)    in one prompt)           never reads the PDF)
    |               |                            |
    +--------- runs/pilot/*.jsonl (one line per question) ---------+
                         |
     src/summarize.py, analysis/pageindex_traces.py, analysis/make_figures.py
                         |
     results/ (figures, results.json)  ->  slides/pilot-results.tex (the deck)
```

## Files

| File | What it does |
|---|---|
| `src/clevrer.py` | Loads one CLEVRER video from its annotation file: objects, the state of each object in each frame, entrances, exits and collisions. Every other file starts from this. |
| `src/scene_report.py` | Writes the runs as one PDF ("plant log") with fpdf2: one chapter for each run, with headings and PDF bookmarks. |
| `src/qa.py` | Picks the 30 pilot runs and their 248 questions, holds the instructions that every system gets, parses the JSON answer and scores it. |
| `src/run_pageindex.py` | System 1. Indexes the PDF with the PageIndex SDK, then asks each question through the PageIndex agent and saves its full trace. |
| `src/run_fullcontext.py` | System 2. Puts the text of the named run's pages into one prompt. This is perfect retrieval. |
| `src/world.py` | The dual-view graph of one run and the two operators: intervention and world simulation. No LLM. |
| `src/run_graph_agent.py` | System 3. The LLM answers with six tools over `world.py`. |
| `src/summarize.py` | Scores a result file by question type. |
| `src/env.py` | Loads the API key from `.env`. |
| `src/sitecustomize.py` | Makes Python try IPv4 first. IPv6 connections time out on this network. |
| `src/usage_tracker.py`, `src/or_usage.py` | Count tokens and read the OpenRouter spend. |
| `analysis/clevrer_stats.py` | Statistics of the CLEVRER validation split. |
| `analysis/clevrer_observed_graph.py` | Shows how many answers need events that the video does not show, and the ceiling of a causal graph with no computation. |
| `analysis/calibrate_moving.py` | Measures the constants: the "moving" threshold, the time step, the contact distance, the friction. |
| `analysis/eval_operators.py` | Scores the two operators on 3,000 held-out videos. |
| `analysis/pageindex_traces.py` | Summarizes what the PageIndex agent read for each question. |
| `analysis/show_example.py` | Shows one question as the three systems saw it (for the meeting demo). |
| `analysis/make_figures.py` | Makes the figures and `results/results.json`. |

## Reading order for the code

1. `src/clevrer.py`: the data model. It is short.
2. `src/scene_report.py`, with `runs/pilot/plant_log_30_runs.pdf` open next to it.
3. `src/qa.py`: the questions, the instructions and the scoring.
4. The three systems: `src/run_fullcontext.py` (the simplest), then `src/run_pageindex.py`, then `src/world.py` and `src/run_graph_agent.py`.
5. The analysis scripts, when you need a number.

## Setup

- `uv sync` installs the dependencies.
- Copy `.env.example` to `.env` and set your OpenRouter key.
- `data/` and `runs/` are not in git. The brief (section 10) has the download and run commands.
