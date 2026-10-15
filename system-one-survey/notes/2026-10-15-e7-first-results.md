# E7: first results, Jev 1.13 against Jev 1.14 (2026-10-15)

These are the first results of paper 1. They are preliminary: one day, one route (OpenRouter from Bangalore), 4 tasks.

---

## 1. What we ran

| Run | Model | Items | Questions | Folder |
|---|---|---|---|---|
| Baseline A | `typesafe/jev-1.13` (resolved `jev-1.13-20260917`) | 1,000 (250 for each task) | 3,500 | `runs/e7-2026-10-15/` |
| Baseline B | `typesafe/jev-1.14` (resolved `jev-1.14-20261014`) | the same 1,000 | 3,500 | `runs/e7-2026-10-15/` |
| Noise control | `typesafe/jev-1.14` again, the same requests, about 20 minutes later | the same 1,000 | 3,500 | `runs/e7-repeat-2026-10-15/` |
| Week-0 probe | `~typesafe/jev-latest` and `~perplexity/pplx-decider-latest` | 200 (50 for each task) | 700 | `runs/e7-weekly-2026-10-15/` |

All 4,400 calls returned status 200. All the runs today (E0, tests and E7) cost $0.16 in total (the sum of `usage.cost` over 3,808 calls).

Commands:

```
uv run python analysis/compare_versions.py runs/e7-2026-10-15/typesafe__jev-1.13.jsonl runs/e7-2026-10-15/typesafe__jev-1.14.jsonl
uv run python analysis/compare_versions.py runs/e7-2026-10-15/typesafe__jev-1.14.jsonl runs/e7-repeat-2026-10-15/typesafe__jev-1.14.jsonl
```

---

## 2. Results

**Version change: Jev 1.13 (A) against Jev 1.14 (B)**

| Task | Questions | Top change | Mean dist | Acc A | Acc B | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 250 | 2.8% | 0.046 | 79.6% | 80.4% | 0.625 |
| boolq | 250 | 3.2% | 0.029 | 90.8% | 89.2% | 0.289 |
| civil | 1750 | 2.7% | 0.027 | 79.5% | 79.3% | 0.665 |
| helpsteer2 | 1250 | 12.7% | 0.076 | 41.1% | 41.8% | 0.493 |
| **all** | 3500 | **6.3%** | 0.046 | 66.6% | 66.7% | 0.938 |

**Noise control: Jev 1.14 against Jev 1.14, the same requests**

| Task | Questions | Top change | Mean dist | Acc A | Acc B | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 250 | 1.6% | 0.017 | 80.4% | 79.6% | 0.5 |
| boolq | 250 | 0.4% | 0.008 | 89.2% | 89.6% | 1 |
| civil | 1750 | 0.9% | 0.009 | 79.3% | 79.3% | 1 |
| helpsteer2 | 1250 | 4.9% | 0.024 | 41.8% | 41.1% | 0.229 |
| **all** | 3500 | **2.3%** | 0.015 | 66.7% | 66.4% | 0.212 |

"Top change": the top answer changed (predicate: the side of 0.5; choice: the option; score: the most probable level). "Mean dist": the mean absolute change of the predicate probability, or the total variation distance for choice and score. HelpSteer2 accuracy is low because the most probable level must be exactly the human level (5 levels).

---

## 3. What the results show

1. **Jev is not deterministic.** Identical requests to the same version changed the top answer on 2.3% of the questions, and on 4.9% of the scores. The API has no temperature or seed setting.
2. **The version update changed answers without a change in accuracy.** 1.13 and 1.14 disagree on 6.3% of the questions. This is about 2.7 times the noise floor (2.3%), and the mean distance is 3 times larger (0.046 against 0.015). The accuracy is the same (66.6% and 66.7%, McNemar p = 0.94).
3. **Scores are the least stable answer type.** HelpSteer2 scores changed most, both across versions (12.7%) and across repeated calls (4.9%).
4. **The aliases point to the newest versions.** On 2026-10-15, `~typesafe/jev-latest` resolved to `jev-1.14-20261014`, and `~perplexity/pplx-decider-latest` resolved to `pplx-decider-v1.1-27b-20261006`.

---

## 4. Consequences for the design

- **Every flip rate needs a noise floor.** E3 must compare each perturbation with repeated identical calls. Published flip rates (for example, 50.5% after a yes/no swap) do not report this floor.
- **Repeat the noise control at least 3 times**, at different times of day, and for every system in E2. Some systems can be deterministic.
- **Pin versions in all runs** (`typesafe/jev-1.14`, not `~typesafe/jev-latest`). Record the resolved version from each response.

---

## 5. Limits of these results

- One day, one route, one region.
- One repeat of the noise control.
- 250 items for BoolQ and Banking77. Small differences are not significant.
- We did not compare OpenRouter with the TypeSafe API directly (TypeSafe paused signups).

---

## 6. Next steps

1. Run the week-0 probe again every week (command below), and compare it with this week.
2. Repeat the noise control for Jev 1.13 and for the other tier-1 systems.

```
PYTHONPATH=src uv run python src/run.py --tag e7-weekly-YYYY-MM-DD --n 50 --models '~typesafe/jev-latest,~perplexity/pplx-decider-latest'
```
