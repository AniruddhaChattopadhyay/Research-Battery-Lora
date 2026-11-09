# How to present the probe (meeting of 2026-10-19)

- Deck: `slides/2026-10-19-probe.pdf` (8 slides and 1 backup slide).
- Numbers: `notes/2026-10-17-probe-results.md`.
- Predictions, written before the runs: `notes/2026-10-16-probe-predictions.md`.

## 1. The goal of the meeting

1. Report an honest negative result: the failure model failed its own test.
2. Get agreement on the revised model.
3. Get three decisions: the revised model, the budget, and a separate API key.

Plan about 20 minutes.

## 2. The story in three sentences

1. We wrote ten predictions before the runs, and two key ones failed: PageIndex matched full context on aggregation questions and on questions about a dated history.
2. PageIndex did this by reading widely, so it pays with tokens: on Compilation, it read 2.1 times the whole document.
3. So PageIndex fails only when the evidence is out of its reach, in dense sections or in large corpora. The next tests go there.

## 3. Slide by slide

| Slide | Time | What to say |
|---|---|---|
| 1. Title | 0.5 min | "Last week I proposed a failure model with five assumptions. This week I tested it on two public datasets. It failed where it predicted failure, and that tells us where to look." |
| 2. The probe in one slide | 2 min | Give the three accuracy pairs and the token numbers. Then say the main message. |
| 3. Predictions | 2 min | "I wrote ten predictions on Friday, before any run, with a rule: if P2 and P7 both fail, the model is wrong." Retrieval asks for one rule by its number. Compilation asks for all rules about a topic. |
| 4. DesignQA | 2 min | "The accuracy is the same. But on Compilation, PageIndex read 172,000 tokens for an 81,000-token document. With a prompt cache, full context costs about $0.05 for each question, and PageIndex costs $0.36." |
| 5. LongMemEval | 2 min | "13 against 12. The reason is the labels. A chat session has one topic, so its summary names the key and sometimes the answer." Point at the Samsung example. |
| 6. Scorecard | 1.5 min | "P2, P4 and P7 are false. By our own rule, the model is wrong for single documents of this size." |
| 7. Revised model | 3 min | "Spread alone does not cause errors. Errors need evidence that is out of reach: dense sections, where the labels cannot name the facts, or large corpora. We have evidence for both: our pilot and RIT-RAG." Point at the figure: a fine tree has labels that hold half the document. |
| 8. Next tests and decisions | 5 min | Ask each decision. Give your recommendation (section 6). |

## 4. Do not say these things

| Do not say | Why | Say this instead |
|---|---|---|
| "PageIndex works well." | It matched full context only on single documents of about 150 pages, at a high token cost. | "PageIndex keeps its accuracy on one mid-size document, because it reads widely." |
| "The theory is dead." | Only the predictions about spread failed. The label trade-off and the pilot result stand. | "The model was wrong about spread, so we revised it." |
| "The keyword agent is as good as PageIndex." | It ran only on 15 LongMemEval questions, with `gpt-5.6-luna`. | "On 15 LongMemEval questions, it matched PageIndex at one fifth of the cost." |
| "PageIndex always costs 7 times more." | This is true only for DesignQA Compilation, against full context with a cache. | "On the rules PDF, PageIndex cost 5 to 7 times as much as full context with a cache." |

## 5. Likely questions and short answers

**"Why did you use a weaker model on LongMemEval?"**
The budget. The OpenRouter key is shared with other projects. The same model ran all three systems, so the comparison is fair. The DesignQA runs used `gpt-5.6-sol`.

**"Are 15 questions enough?"**
No. They show a direction only. DesignQA, with 50 questions, is the main result.

**"Is Compilation a fair test?"**
Partly. The gold list follows a fixed rule that the question does not state, so both systems stop at about 0.67. But PageIndex read 80% of the gold pages, so it did find the evidence.

**"What changed in PageIndex 0.4.1?"**
The agent still has the same four tools and the same prompt. For the rules PDF, PageIndex merged the 588 bookmarks into a tree of 207 nodes.

**"Does this end the graph idea?"**
No. The graph helps where the revised model predicts failure: dense records and large corpora. These are also the manufacturing cases: logs, sensor tables, and work orders over many years.

**"What did it cost?"**
About $31.94, estimated from the token counts. The parallel full-context runs did not use the prompt cache, so they cost about 10 times more than necessary.

## 6. Your recommendations for the decisions

1. **The revised model:** accept it, and test it first on dense records and at scale.
2. **Budget:** about $10–15 for the next two tests with `gpt-5.6-luna`. Ask for the main-study budget after these tests.
3. **API key:** a separate OpenRouter key for this project, with its own spending limit.

## 7. Optional live demo (no API cost)

```bash
cd ~/Documents/research/better-page-index
uv run python analysis/probe_report.py | tail -12     # the scorecard, computed from the raw records
```
