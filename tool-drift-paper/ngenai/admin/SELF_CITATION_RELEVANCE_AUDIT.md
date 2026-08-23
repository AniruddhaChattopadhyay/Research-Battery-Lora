# Self-citation relevance audit

## Verdict

The paper is real and the citation is **technically defensible as related feedback-based recovery work**, provided the manuscript states the important scope difference. It should be used once in Related Work—not as evidence about function calling or interface drift.

## Verified record

- **Title:** *Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems*
- **Authors (ordered):** Aniruddha Chattopadhyay; Raj Dandekar; Kaushik Roy
- **OpenReview record:** NeSy 2025 Poster; paper type “Long Paper”; track “Neurosymbolic Generative Models”; published 20 April 2025 and last modified 29 August 2025 ([official OpenReview record](https://openreview.net/forum?id=4T8HP1SHOW&noteId=dBwuQuAVnG)).
- **Archival proceedings record:** *Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning*, PMLR volume 284, pages 957–976, 2025 ([official PMLR record](https://proceedings.mlr.press/v284/chattopadhyay25a.html)). The supplied [OpenReview PDF](https://openreview.net/pdf?id=4T8HP1SHOW) is the same paper.

For the camera-ready bibliography, the archival PMLR record is more stable and complete than citing only the OpenReview discussion page:

```bibtex
@inproceedings{chattopadhyay2025modelgrounded,
  title     = {Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems},
  author    = {Chattopadhyay, Aniruddha and Dandekar, Raj and Roy, Kaushik},
  booktitle = {Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning},
  pages     = {957--976},
  year      = {2025},
  editor    = {H. Gilpin, Leilani and Giunchiglia, Eleonora and Hitzler, Pascal and van Krieken, Emile},
  volume    = {284},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v284/chattopadhyay25a.html}
}
```

## Relevance to the interface-drift manuscript

The cited paper frames instruction-tuned LLMs as model-grounded symbolic systems and proposes “metatuning”: an external judge detects errors or inconsistencies, produces symbolic natural-language corrections, and iteratively updates the prompt/context state. This feedback pattern is genuinely related to SchemaShield-Lite's validation-feedback-and-repair pattern ([official archival paper](https://proceedings.mlr.press/v284/chattopadhyay25a.html), Sections 3.1–3.2).

The scope is nevertheless different. Metatuning is an iterative task-learning procedure evaluated on mathematical reasoning, using training-set interactions and retained prompt/history state. It does **not** study tools, API schemas, interface drift, constrained decoding, or one-shot deployment-time repair. The camera-ready text should make that contrast explicit.

## Least-promotional insertion

Place the citation in the Related Work paragraph on robustness and recovery, immediately after the discussion of Fission-GRPO and before the paragraph on constrained decoding. This is a more direct fit than inserting it as general neurosymbolic framing.

Suggested wording:

> Chattopadhyay et al.~\cite{chattopadhyay2025modelgrounded} study a related feedback pattern for mathematical reasoning: an external judge identifies errors, generates symbolic corrections, and iteratively revises the prompt state. \method instead uses deterministic schema validation and a single inference-time repair step for tool calls, without a training-set loop or persistent prompt optimization.

Use this single citation only. Do not place it in the contributions or experiments, and do not describe it as prior work on interface drift or tool-call repair.
