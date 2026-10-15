# How to write a survey paper: a guide for our team

This guide is for team members who did not write a survey paper before. It explains what a survey paper is, what reviewers look for and the steps that we will do. The plan for our survey is in `2026-10-15-survey-plan.md`.

---

## 1. What a survey paper is

A survey paper collects the work in one research area, puts it in order and explains it. Usually, it does not propose a new method. It gives value to the reader in five ways:

1. It gives a clear definition of the area and its limits.
2. It gives a **taxonomy**: a set of categories that groups the work by its important properties.
3. It compares the work on the same dimensions, usually in tables.
4. It finds the open problems and suggests research directions.
5. Some surveys also run new experiments that compare the methods under the same conditions. We call this an **empirical survey**.

A good survey saves time for a new researcher. After one paper, the reader knows the main ideas, the main works and the gaps.

### Types of surveys

| Type | What it does | Strength of the method |
|---|---|---|
| Narrative review | An expert summarizes the area from personal knowledge. | Weak. Reviewers can say that it is not complete. |
| Systematic literature review (SLR) | The authors follow a written protocol: sources, search strings and inclusion rules. They report the counts at each step (a PRISMA diagram). | Strong. Another team can do it again and get the same set of works. |
| Multivocal literature review (MLR) | An SLR that also includes "grey literature": vendor documents, API references, blogs and talks. | Strong, and necessary when industry leads the area. |
| Empirical survey | A survey that also runs experiments to compare the methods. | Strong, if the experiments are fair. |
| Tutorial | It teaches a method step by step. | It is not a review of all the work. |
| Position paper | It argues for one point of view. | It is not a review of all the work. |

**Our survey is a multivocal review plus targeted experiments.** System One models started in September 2026. Many facts about them are only in vendor documents and blogs. In the first month, more than 15 papers and several leaderboards measured these models, and their results do not agree. Our experiments will test the main disagreements under the same conditions.

---

## 2. What reviewers look for

Reviewers ask these questions:

1. **Is the survey necessary?** Is there an existing survey on the same topic? Why is a new survey necessary now?
2. **Is the scope clear?** What is in the survey, and what is not?
3. **Is the coverage complete?** Did the authors miss important work?
4. **Is the method clear?** How did the authors find and select the work? Can another team do it again?
5. **Does it give new insight?** A taxonomy, comparisons, trends and conflicts give insight. A list of paper summaries does not.
6. **Is it critical?** Does it show the weaknesses of each approach, not only the strengths?
7. **Are the open problems specific?** "More research is necessary" is not specific. "No benchmark measures calibration under prompt injection" is specific.
8. **Is it up to date?**

### Frequent reasons for rejection

- The paper summarizes the papers one by one ("paper A did X; paper B did Y"). This is the most frequent reason.
- The taxonomy is missing, or it does not help the reader.
- The paper misses important work, or it is out of date.
- The paper repeats claims from authors or vendors as facts.
- The experiments are not fair: different prompts, different data or different settings.
- The scope is too wide, so the paper is shallow.

---

## 3. The steps

We will do these steps in this order. Some steps overlap.

1. **Define the scope and the research questions.** Write down what is in the survey and what is out.
2. **Do a novelty check.** Search for existing surveys on the same topic. If one exists, find a different angle.
3. **Write the review protocol.** Write the sources, the search strings, the inclusion rules and the date range before you search.
4. **Search.** Run each search string on each source. Save all the results in Zotero (a reference manager).
5. **Screen.** First, read the title and the abstract. Then read the full text of the remaining works. Record how many works you keep and remove at each step.
6. **Snowball.** Read the reference lists of the kept works (backward snowballing). Find the works that cite them (forward snowballing). Screen the new works.
7. **Extract the data.** Fill the same form for each work, in a shared spreadsheet.
8. **Build the taxonomy.** Group the works by the values in the form. Name the dimensions. Change the taxonomy until each work fits in one place.
9. **Synthesize.** Make the comparison tables. Find the trends, the agreements and the conflicts.
10. **Run the experiments.** See `2026-10-15-experiment-design.md`.
11. **Write the open problems.** Use the gaps from steps 9 and 10.
12. **Write the paper.** Then do an internal review. Ask one expert outside the team to read it.
13. **Publish.** Put the paper on arXiv. Submit it to a venue. Keep the GitHub list of works up to date.

### The extraction form (example fields)

For each work, record:

- The bibliographic data and the access date. Vendor pages change.
- The source type: paper, preprint, vendor document, blog or API reference.
- The system name, the vendor and the release date.
- The answer types: predicate, choice or score.
- The interface: questions in each call, context length and modalities.
- The mechanism, if the source gives it: backbone model, output head and training method.
- The calibration claims, and how the authors measured them.
- The latency, price and accuracy claims, with the benchmark name. Mark each vendor number as "vendor-reported".
- The limits and risks that the authors give.

---

## 4. The parts of a survey paper

| Section | What it contains |
|---|---|
| Title and abstract | The topic, the scope and the 3 to 5 main contributions. |
| 1. Introduction | Why the topic is important now. The scope. The contributions. A table that compares our survey with the existing surveys. |
| 2. Background | The concepts that the reader needs. For us: System 1 and System 2, classifiers and LLMs. |
| 3. Review method | The sources, the search strings, the inclusion rules and the PRISMA diagram. |
| 4. Definition and taxonomy | A formal definition, and the taxonomy with a figure. |
| 5. Review of the works | One subsection for each taxonomy category. Compare the works; do not list them. |
| 6. Empirical evaluation | Our experiments. |
| 7. Applications | Where people use the systems, and the design patterns. |
| 8. Open problems | Specific gaps and research directions. |
| 9. Conclusion | A short summary. |
| Appendix | The extraction form, the prompts and the extra results. |

A journal survey usually has 25 to 45 pages. A conference survey track usually has 7 to 9 pages plus references. We will write the long version first. If necessary, we will make a short version from it.

---

## 5. Tools

- **Zotero** with the Better BibTeX plug-in: one shared group library for all the works. Export it to `paper/references.bib`.
- **A shared spreadsheet** for the extraction form and the screening counts.
- **Semantic Scholar, Google Scholar and Connected Papers** for the search and the snowballing.
- **The arXiv listings** for cs.CL, cs.LG and cs.AI. Check them every week, because the area changes quickly.
- **This git repository** for the LaTeX source, the notes and the experiment code.

---

## 6. Rules for our team

1. Record the access date for each web source.
2. Write "vendor-reported" next to each number from a vendor. Do not give it as a fact.
3. Quote definitions exactly, with the source.
4. Paraphrase all other text and cite it. Do not copy sentences.
5. Use one term for one thing. Use the terms in the glossary of the plan.
6. Agree on the author order and the tasks of each person before the work starts.
