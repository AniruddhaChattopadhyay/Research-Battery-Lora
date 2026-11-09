# Kickoff meeting, 2026-11-06

Attendees: Prof. Kaushik Roy (kroy2@ua.edu), Ashraf Abdul-Muumin, Aniruddha Chattopadhyay.
Source: the Gemini meeting notes and the figure "Fig. 1. QBranch"
(`figures/2026-11-06-qbranch-architecture.png`). The Gemini notes can have errors.
Corrections are in the last section.

## The idea (Prof. Roy)

1. A world model is the "brain" of an agent in a closed loop (model predictive control). It changes
   observations into a representation. The representation predicts, and it lets the agent plan and act.
   Feedback from the environment changes the plan and the representation.
2. LeCun's JEPA learns an energy landscape. Prof. Roy says that energy manifolds are fickle and
   hypersensitive. They need a very strong training signal.
3. Proposal: use a foundation model to build a **neurosymbolic** world model. The neural model writes a
   symbolic backbone (a knowledge graph). The world model is not a pure neural model.
4. Setting: online-learning RAG with text observations. Documents arrive over time. The foundation model
   builds a knowledge graph and infers the questions that the corpus can answer.
5. Schemas and bindings: a schema element that no document supports yet stays **unbound**. It binds
   later, when a relevant document arrives.
6. The representation is optimized with the **inferred questions**, not with downstream task success.
7. Second refinement step, like **database views**: answer the inferred questions with the graph and
   existing RAG methods. When an answer is wrong, keep the schema or split a view. The policy that
   decides when to split or merge is not designed yet.
8. Question from Aniruddha: how does the system route an unseen question to a view? Answer: that is a
   research objective. It can be a one-step plan.

## Decisions

- Aligned: use the neurosymbolic world model method (knowledge graph + inferred questions).
- Open: the venue. ICWM if the plan is fixed (predetermined). ICAPS if the paper adds an explicit
  planning angle (split and merge under a budget).

## Datasets named

HotpotQA and FinanceBench.

## Next steps

| Owner | Task |
|---|---|
| Prof. Roy | Upload the world-model figure to the Google Chat. |
| Prof. Roy | Share the OpenRouter key (done: the key is in `qbranch/.env`, which git ignores). |
| Prof. Roy | Theory and experimental design, after he gets the decision paper draft. |
| Aniruddha | Run the experiments on the architecture. |
| Aniruddha | Post progress in the group chat. |
| Aniruddha | Make a shared Google Doc to track the project and its milestones. |
| Aniruddha | Send the decision paper draft (System One, paper 1) to Prof. Roy for review. |
| Ashraf + Aniruddha | Deadline-driven work. |

Other items: Prof. Roy looks for Amazon funding. Prof. Roy also proposed a summer visit. Aniruddha
will discuss the visit at work. Aniruddha will give this project priority over the
curiosity-driven experiments with Jev.

## Corrections to the Gemini notes

- "IAPS" means ICAPS, the International Conference on Automated Planning and Scheduling.
- "Lun and colleagues at Brown University and Lambda": the ICWM site lists Randall Balestriero
  (Brown / AMI), Amir Zadeh (Lambda) and Yann LeCun (NYU / AMI) as the creators.
- "experiments with Jeff" means the experiments with Jev (the System One / decision model project).
