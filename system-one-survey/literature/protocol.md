# Review protocol (draft v0, 2026-10-15)

This file is the written protocol for the literature review. Write it before the search, and change it only with a dated note at the end. A fixed protocol lets another team repeat the review and get the same set of works.

Method: a **multivocal literature review** (academic papers plus grey literature), with forward and backward snowballing. Report the counts in a PRISMA 2020 flow diagram.

---

## 1. Review questions

- **LQ1.** Which systems are System One models, and what are their interfaces, mechanisms and claims?
- **LQ2.** Which earlier research lines lead to System One models, and what did each line learn?
- **LQ3.** How do vendors and researchers evaluate these systems now, and what is missing?
- **LQ4.** What risks and open problems do the sources give?

---

## 2. Sources

| Source | What we search | How |
|---|---|---|
| arXiv (cs.CL, cs.LG, cs.AI) | Preprints | Search strings below, plus a weekly check of the new listings |
| ACL Anthology | NLP papers | Search strings below |
| OpenReview | ICLR, NeurIPS, ICML, COLM and TMLR papers | Search strings below |
| Semantic Scholar and Google Scholar | All papers | Search strings below, and forward snowballing |
| Vendor sites | Announcements, documents, API references, changelogs and pricing pages | Manual check of each known vendor |
| Hugging Face and OpenRouter | Model cards and model pages | Search "decider", "decision" and the vendor names |
| News and technical blogs | Only to find primary sources | Do not cite a blog for a fact that a primary source gives |

---

## 3. Search strings

### 3.1 Grey literature (System One models)

- `"System One model"` OR `"System One models"`
- `"decision model"` AND (`API` OR `LLM`)
- `"Decisions API"`
- `"typed decision"` OR `"probabilistic decision"` AND `model`
- The names of the known systems: `Jev` AND `TypeSafe`; `pplx-decider`; `"Microsoft-Decision-1"`; `"OpenAI" "Decisions API"`

### 3.2 Academic research lines

| Line | Search string |
|---|---|
| L1. LLMs as classifiers | (`"zero-shot classification"` OR `"text classification"` OR `"in-context classification"`) AND (`"language model"` OR `LLM`) |
| L2. Label scoring | (`"label likelihood"` OR `verbalizer` OR `"noisy channel"` OR `cloze`) AND `prompt` |
| L3. Calibration | (`calibration` OR `"uncertainty"` OR `"verbalized confidence"`) AND (`"language model"` OR `LLM`) |
| L4. Selective prediction | (`"selective prediction"` OR `abstention` OR `"conformal prediction"`) AND (`"language model"` OR `LLM`) |
| L5. Structured outputs | (`"structured output"` OR `"constrained decoding"` OR `"JSON schema"`) AND (`"language model"` OR `LLM`) |
| L6. Scalar judges | (`"reward model"` OR `"LLM-as-a-judge"` OR `"rubric"`) AND (`score` OR `benchmark`) |
| L7. Guard models | (`"guard model"` OR `"content moderation"` OR `"safety classifier"`) AND (`LLM` OR `"language model"`) |
| L8. Routing and cascades | (`router` OR `routing` OR `cascade`) AND (`LLM` OR `"language model"`) |
| L9. Encoder baselines | (`ModernBERT` OR `DeBERTa` OR `SetFit` OR `"cross-encoder"`) AND `classification` |
| L10. Dual-process AI | (`"dual-process"` OR `"System 1"` OR `"fast and slow"`) AND (`LLM` OR `agent` OR `robot`) |

---

## 4. Date range

- Grey literature: from 2026-09-01 to the cut-off date.
- Academic papers: from 2018-01-01 to the cut-off date. Older seminal works enter through snowballing.
- Cut-off date for version 1: **2026-12-01** (proposal). Record a new cut-off date for each later version.

---

## 5. Inclusion and exclusion rules

Include a work if one of these rules is true:

- **I1.** It describes, proposes or evaluates a model or an API that returns closed-set typed answers (binary, categorical or ordinal), with probabilities or scores, for questions that the user gives at inference time.
- **I2.** It is a direct technical ancestor of I1 (lines L1 to L10). For these lines, keep the seminal works, the surveys and the most relevant works from 2024 to 2026.
- **I3.** It is a benchmark or a metric that we use in the evaluation.

Exclude a work if one of these rules is true:

- **E1.** It is a classifier for one fixed task, with no questions at inference time. Exception: we use it as a baseline.
- **E2.** It only generates free text.
- **E3.** It is a marketing page with no technical content. Exception: it is the only primary source for a system.
- **E4.** It is a duplicate. Keep the most complete version.
- **E5.** It is not in English.
- **E6.** We cannot get the full text.

---

## 6. Screening

1. Two people screen the same random 10% of the works by title and abstract.
2. Calculate Cohen's kappa for the two decisions. If kappa is less than 0.6, discuss the rules and repeat the step with a new sample.
3. One person screens the remaining works. A second person checks each exclusion that is not clear.
4. Read the full text of the kept works. Record the reason for each exclusion (E1 to E6).
5. Snowball once forward and once backward from the kept works. Screen the new works with the same steps.
6. Record all counts for the PRISMA diagram: found, duplicates, screened, excluded (by reason) and included.

---

## 7. Quality check for grey literature

Use the AACODS checklist (Tyndall, 2010) for each grey source:

- **Authority:** Who wrote it? Is it the vendor itself?
- **Accuracy:** Does it give a method for its numbers?
- **Coverage:** Does it give the limits of the system?
- **Objectivity:** Does the author sell the system?
- **Date:** When did the author publish or change it? When did we read it?
- **Significance:** Is it the primary source for a fact?

---

## 8. Data extraction

Use the form in `notes/2026-10-15-survey-paper-guide.md`, section 3. Keep one row for each work in the shared spreadsheet. Save a PDF or a web archive copy of each vendor page on the day that you read it, because vendors change their pages.

---

## Change log

- 2026-10-15: Draft v0.
