

NGEN-AI 2026 Camera-Ready Plan and Checklist

## Paper

- **Paper:** 15
- **Title:** Robust Function Calling for LLM Agents Under Interface Drift
- **Conference:** International Conference on Next Generation AI Systems (NGEN-AI 2026)
- **Conference dates:** September 1–4, 2026
- **Author-registration deadline:** August 3, 2026
- **Internal registration target:** August 2, 2026
- **Camera-ready deadline:** August 10, 2026, Anywhere on Earth (AoE)
- **Internal submission target:** August 8, 2026
- **Working manuscript:** `ngenai/main.tex`
- **Maximum length:** 16 LNCS pages

## Definition of Done

The camera-ready is complete only when:

- At least one author is registered.
- The author names, order, affiliations, emails, and corresponding author are final and consistent everywhere.
- Every reviewer comment has either a manuscript change or an explicit limitation/future-work response.
- The final paper follows the NGEN-AI and Springer requirements and is at most 16 pages.
- The final PDF has passed content, reference, build, and visual checks.
- The complete ZIP has been rebuilt and tested in a clean directory.
- The ZIP has been uploaded and the submission confirmation has been saved.

## Current Status — August 8, 2026

The reviewer revisions, numerical audit, bibliography audit, Springer-template
checks, visual QA, signed-license QA, and final package preflight are complete.
The camera-ready PDF is 16 pages, has embedded fonts, and has no margin overflow.
The final eight-file ZIP extracts without errors and rebuilds using only its
included files; the rebuilt PDF has identical extracted text and pixel-identical
page renders. The handwritten License-to-Publish form is five readable pages and
matches the manuscript title, author order, corresponding author, and contact
details. The required Disclosure of Interests is included, and no ethics or
consent statement is applicable.

The remaining author-controlled items are:

- Written coauthor confirmation of the author order and approval of the final
  manuscript, metadata, and corresponding author's authority to sign.
- Upload the tested ZIP through PayQlick, submit it, and save the confirmation.

The conference camera-ready page now directs authors to sign in to PayQlick,
open “View My Tickets,” and upload one ZIP through the NGEN-AI 2026 ticket.

## Working Rules

- Complete the checklist in order unless an item explicitly says it can run in parallel.
- Treat EasyChair submission metadata as the authoritative source for author order.
- Do not begin a new large experiment for the camera-ready version.
- Re-analysis of existing logs is allowed only if it is quick, reproducible, and does not delay required revisions.
- Do not add claims that are broader than the evidence.
- Keep the final paper within 16 pages without changing LNCS margins, fonts, or spacing.
- Do not upload old manuscript versions, unused figures, or LaTeX build artifacts.

---

## Phase 1 — Resolve Metadata and Registration

### 1.1 Confirm the authoritative author order

- [X] Record the accepted submission author order from the paper 15 decision
  notice: Aniruddha Chattopadhyay, Kanishka Halder, Partha Sarathi Roy.
- [X] Resolve the manuscript discrepancy using the author-supplied order, which
  matches the acceptance metadata:
  - Acceptance metadata: Aniruddha Chattopadhyay, Kanishka Halder, Partha Sarathi Roy.
  - Revised manuscript: Aniruddha Chattopadhyay, Kanishka Halder, Partha Sarathi Roy.
- [ ] Obtain written confirmation from all authors that the final order is correct.
- [X] Verify that the accepted-submission order is identical in the manuscript
  and prefilled License-to-Publish form.

**Verification:** All three locations show the same names in the same order.

### 1.2 Finalize author information

- [X] Confirm the exact spelling of every author's full name.
- [X] Confirm each author's affiliation.
- [X] Add the confirmed shared affiliation location: Bangalore, India.
- [X] Confirm each author's publication email address.
- [X] Omit ORCID identifiers because the authors do not have ORCIDs.
- [X] Confirm that `Independent Researcher` is the affiliation for all three authors.

**Verification:** Every author has approved the complete author block.

### 1.3 Designate the corresponding author

- [X] Select Aniruddha Chattopadhyay as the corresponding author.
- [X] Add a clear corresponding-author marker to the manuscript.
- [X] Include the corresponding author's valid email address.
- [X] Record Aniruddha Chattopadhyay as the person who will hand-sign the
  License-to-Publish form.
- [X] Record that the corresponding author is responsible for reviewing the
  Springer proof within 72 hours of receipt.

**Verification:** The manuscript and License-to-Publish form identify the same corresponding author.

### 1.4 Complete author registration

- [X] Choose Aniruddha Chattopadhyay as the registered author.
- [X] Complete author registration by the official August 3 deadline (paid on
  August 3, 2026).
- [X] Save the payment receipt and registration confirmation at
  `ngenai/admin/registration/Receipt_PQ9300882026.pdf`.
- [X] Verify that the receipt explicitly associates paper 15 with the
  registration.
- [X] Record the presentation mode as online/remote.
- [X] Record Aniruddha Chattopadhyay as the presenter.

**Verification:** Receipt PQ9300882026 records Aniruddha Chattopadhyay's author
registration as paid, explicitly covers paper 15, and records online attendance.
The receipt's billing address ends in “Canada” despite otherwise listing
Bangalore, Karnataka 560075; request a corrected receipt if that country entry
is unintended.

### 1.5 Confirm the submission route

- [X] Check the NGEN-AI camera-ready page for the final upload link.
- [X] Confirm that the official route is PayQlick “View My Tickets,” not an
  EasyChair camera-ready link.
- [X] Record the upload location as
  `https://events.payqlick.com/event/153/ngen-ai-2026` and the required ZIP
  contents in Phase 8.
- [X] Determine that the August 5 escalation is unnecessary because the upload
  route is now published.

**Verification:** The team knows exactly where and how the final package will be submitted.

---

## Phase 2 — Address Reviewer 1

### 2.1 Clarify candidate-set retry

- [X] Replace the difficult sentence:
  - “a candidate-set retry over only the visible tools resolves a target before repair.”
- [X] Explain the process in chronological order:
  1. The initial output omits a tool name.
  2. The model receives only the runtime-visible tool list.
  3. The model issues a new candidate tool call.
  4. If a tool is named, the call is validated.
  5. If necessary, repair is constrained to that selected tool.
- [X] Check all other descriptions of this mechanism for consistent wording.

**Verification:** A reader can understand candidate-set retry without already knowing the implementation.

### 2.2 Replace “load-bearing” terminology

- [X] Replace “The candidate-set retry is load-bearing” with plain technical language such as “The candidate-set retry is essential to the non-oracle protocol.”
- [X] Replace other uses of “load-bearing” or “does the heavy lifting” where they reduce clarity.
- [X] Preserve the supporting numerical comparisons after the rewording.

**Verification:** The manuscript contains no unexplained metaphorical use of “load-bearing.”

### 2.3 Define oracle and non-oracle protocols

- [X] Add a definition at the first substantive use of “non-oracle” and “oracle-constrained.”
- [X] State that the non-oracle protocol uses only deployment-available information:
  - User context.
  - Initial model output.
  - Runtime-visible tool set.
- [X] State that the oracle-constrained protocol additionally receives the benchmark's gold target tool.
- [X] State explicitly that the gold target is unavailable in deployment.
- [X] Describe oracle-constrained results as upper bounds rather than operational evidence.

**Verification:** The distinction is defined before the first results comparison and used consistently afterward.

### 2.4 Clarify the latency and stronger-model tradeoff

- [X] State that SchemaShield-Lite has meaningful token and latency overhead.
- [X] Retain the measured DICE and BFCL token-overhead numbers.
- [X] Retain the measured expected BFCL latency increase and label it as coming from one instrumented rerun.
- [X] State that cross-provider, cross-model latency and dollar cost remain unmeasured.
- [X] State that GPT-4o-mini suffers less drift damage and receives a small, non-significant repair gain on the evaluated slice.
- [X] Avoid implying that the repair layer is equally beneficial for every base model.

**Verification:** The paper presents the method as a gated, model-dependent fallback rather than a universally free improvement.

---

## Phase 3 — Address Reviewer 2

### 3.1 Make the existing model coverage visible

- [X] Add GPT-4o-mini to the Experimental Setup model description.
- [X] Clearly separate:
  - Primary non-oracle Qwen3.5-9B evidence.
  - Same-family Qwen3.5-35B-A3B replication.
  - Cross-family Llama-4-Scout portability result.
  - GPT-4o-mini closed-model sanity check.
  - Oracle-constrained Mistral evidence.
- [X] Check that the abstract, introduction, results, discussion, limitations, and conclusion use the same scope.
- [X] Do not describe the Qwen-35B result alone as broad cross-family generalization.

**Verification:** Readers can identify which models were tested under which protocol without reconstructing it from several sections.

### 3.2 Scope the real-API-history evidence correctly

- [X] Retain the 11-item Notion and Stripe changelog audit.
- [X] Describe the audit as an ecological-validity sanity check.
- [X] State that the drift operators remain author-designed.
- [X] State that the audit is not a mined empirical distribution of production API changes.
- [X] Check that the discussion and conclusion do not overstate real-world representativeness.

**Verification:** The paper clearly distinguishes real examples supporting plausibility from an empirically sampled drift distribution.

### 3.3 Add the multi-step and tool-ecosystem limitation

- [X] Clarify that DICE provides multi-round conversational context but the evaluation scores an individual function-call decision.
- [X] Clarify that BFCL multiple-function categories do not constitute a complete multi-step agent trajectory.
- [X] Add sequential tool execution, compounding failures, and production-scale candidate sets as untested settings.
- [X] Identify multi-step and large-tool-ecosystem evaluation as future work.
- [X] Keep suitable future benchmarks in Related Work without claiming they were evaluated.

**Verification:** The limitations section directly addresses both multi-step scalability and larger tool ecosystems.

---

## Phase 4 — Address Reviewer 3

### 4.1 Preserve the portability limitation

- [X] State that the strongest successful non-oracle evidence remains concentrated in the Qwen family.
- [X] Retain the Llama-4-Scout missing-tool-name failure mode.
- [X] Explain that the repair layer depends on the base model producing a usable tool name.
- [X] Avoid treating same-family size replication as complete portability evidence.

**Verification:** Portability claims match the strongest cross-family result rather than the strongest same-family result.

### 4.2 Refine cost reporting

- [X] Verify all invocation-rate and token-overhead calculations against the released run artifacts.
- [X] Verify the reported original-call, drifted-call, and repair-stage latency means.
- [X] State that the latency evidence comes from one BFCL rerun.
- [X] Add cross-provider, cross-model, and dollar-cost analysis to the limitations.
- [X] Use existing logs to correct repair-stage invocation and expected-latency accounting.
- [X] Do not launch a new provider sweep.

**Verification:** Every reported cost number is reproducible, appropriately scoped, and not generalized beyond the measured run.

### 4.3 Disentangle drift recovery from baseline patching

- [X] Retain the gain-decomposition table.
- [X] Retain the originally-correct-slice metrics.
- [X] Add the decomposition directly to the conclusion:
  - DICE: 19 of 29 gains are drift recovery; 10 are baseline patching.
  - BFCL: 33 of 45 gains are drift recovery; 12 are baseline patching.
- [X] State that repaired-minus-drifted accuracy is not a pure drift-recovery estimate.
- [X] Describe baseline patching as a secondary deployment benefit, not the central causal claim.
- [X] Check that the abstract's recovery counts remain consistent with this framing.

**Verification:** No headline gain can reasonably be read as entirely caused by drift recovery.

### 4.4 Record Review 4

- [X] Confirm that Review 4 contains no comment beyond the blank review body.
- [X] Record “No actionable comment supplied” in the internal reviewer-response notes.

**Verification:** All supplied reviews have an explicit disposition.

---

## Phase 5 — Tighten Claims and Internal Consistency

### 5.1 Audit headline claims

- [X] Scope “candidate-set drift dominates” to the Qwen3.5-9B DICE-300 ablation.
- [X] Replace “practical robustness layer” in the abstract with model-dependent fallback wording.
- [X] Ensure “no observed repair harms” is never shortened to an unqualified “no harms.”
- [X] Ensure GPT-4o-mini is described as a sanity check rather than confirmatory evidence.
- [X] Ensure oracle-constrained ablations are never presented as deployment-facing results.

**Verification:** Every major claim states or clearly implies its model, benchmark, and protocol scope.

### 5.2 Cross-check all numbers

- [X] Verify DICE-300 Original, Drifted, and Repaired scores in every occurrence.
- [X] Verify BFCL-200 Original, Drifted, and Repaired scores in every occurrence.
- [X] Verify drift-recovery, baseline-patching, harms, unresolved-target, and invocation-rate counts.
- [X] Verify Qwen3.5-35B-A3B, Llama-4-Scout, Mistral, and GPT-4o-mini values.
- [X] Verify all reported p-values and confidence intervals against paired run artifacts.
- [X] Confirm that differences between separate runs are clearly labeled as run-level variance.

**Verification:** Each metric has one authoritative value per run and all manuscript occurrences agree.

### 5.3 Check terminology and model naming

- [X] Use one spelling for `Qwen3.5-35B-A3B`.
- [X] Use one spelling for `Llama-4-Scout`.
- [X] Use one spelling for `Mistral-Small-3.2-24B`.
- [X] Use one spelling for `GPT-4o-mini`.
- [X] Use “function calling” and “tool calling” consistently where they refer to the same concept.
- [X] Define every abbreviation at first use.

**Verification:** A full-text search finds no conflicting names or undefined abbreviations.

### 5.4 Verify reproducibility links

- [X] Open the public artifact repository link.
- [X] Open each manuscript-linked artifact or source file.
- [X] Confirm that no obsolete line-anchored artifact links remain in the manuscript.
- [X] Confirm that the repository is public without authentication.
- [X] Pin the exact public release using immutable commit
  `fbac15632f1ef8133fe83ce8eccb2ee071f0891e`; no mutable branch link is used.
- [X] Link the manuscript to verified commit `fbac156` rather than mutable `main`.

**Verification:** A reader can retrieve the exact code and artifacts used by the paper.

---

## Phase 6 — Springer and NGEN-AI Compliance

### 6.1 Verify the official template

- [X] Download the current Springer Computer Science proceedings LaTeX template.
- [X] Compare the included `llncs.cls` and `splncs04.bst` with the official versions and TeX Live 2026.
- [X] Retain local `llncs.cls` v2.26 because it matches TeX Live 2026 and is newer than the conference download's v2.24; `splncs04.bst` matches exactly.
- [X] Confirm that the paper compiles with the verified official files.
- [X] Remove unused LaTeX packages introduced by earlier manuscript versions.

**Verification:** The final source compiles using the current official Springer template without layout overrides.

### 6.2 Remove headers and page numbers

- [X] Remove the `runningheads` document-class option.
- [X] Remove unused `\titlerunning` and `\authorrunning` declarations.
- [X] Confirm that the current PDF contains no running headers.
- [X] Confirm that the current PDF contains no page numbers.

**Verification:** Visual inspection of every page shows neither running heads nor page numbers.

### 6.3 Add the mandatory Disclosure of Interests

- [X] Add a section immediately before the references titled `Disclosure of Interests`.
- [X] The authors confirmed that there are no competing interests; use the required statement:

  > The authors have no competing interests to declare that are relevant to the content of this article.
  >
- [X] Determine that a competing-interest disclosure beyond the default statement is not applicable.
- [X] Place the GenAI disclosure before the Disclosure of Interests so that Disclosure of Interests remains immediately before the references.

**Verification:** “Disclosure of Interests” is the final manuscript section before References.

### 6.4 Confirm ethics and consent requirements

- [X] Confirm that the research involves no human participants, animals, personal data, or biological material.
- [X] Record the no-involvement determination internally.
- [X] Determine that ethics committee, approval, consent-to-participate, and consent-to-publish statements are not applicable.

**Verification:** All applicable ethics and consent disclosures are present.

### 6.5 Prepare alternative text

- [X] Create `Alt-Text.txt`.
- [X] Add a clearly labeled description for Fig. 1.
- [X] Describe the inputs, base-model call, validation branch, pass-through path, repair context, constrained repair call, and repaired output.
- [X] Keep the description functional and concise rather than repeating the caption.
- [X] Confirm that there are no other figures or image-based tables requiring alternative text.

**Verification:** Every visual item has a corresponding entry in `Alt-Text.txt`.

### 6.6 Check figures and tables

- [X] Confirm that every figure and table is numbered and cited in the text.
- [X] Confirm that figure captions are below figures.
- [X] Confirm that table captions are above tables.
- [X] Confirm that tables and equations remain editable rather than embedded as images.
- [X] Confirm that Fig. 1 is readable in grayscale.
- [X] Confirm that figure and table text is at least 6 pt; Table 1 now uses native
  `\scriptsize` (7 pt) rather than scaled-down text.
- [X] Confirm that no table or figure crosses a page margin.
- [X] Confirm that Fig. 1 is generated from author-controlled TikZ source and uses no third-party artwork.

**Verification:** Every figure and table satisfies the NGEN-AI production checklist.

### 6.7 Audit references

- [X] Confirm that citations render as numbered square brackets.
- [X] Confirm that every in-text citation has a bibliography entry.
- [X] Confirm that every rendered bibliography entry is cited.
- [X] Verify all 22 rendered references against authoritative primary records;
  no hallucinated or materially incorrect reference was found.
- [X] Add DOI information wherever available.
- [X] Replace incomplete `and others` author metadata where complete metadata is available.
- [X] Preserve capitalization of terms such as LLM, API, BFCL, DICE-BENCH, ToolLLM, and ReAct.
- [X] Confirm that all references use the Latin alphabet.
- [X] Check titles, venues, years, page ranges, URLs, and access dates.
- [X] Rebuild the `.bbl` after the bibliography changes.

**Verification:** The bibliography is complete, correctly capitalized, and free of missing available DOIs.

### 6.8 Check page limit and template integrity

- [X] Confirm that the disclosure-complete PDF is exactly 16 pages.
- [X] Do not reduce margins, fonts, line spacing, or standard LNCS spacing to make it fit.
- [X] Determine that no shortening is required because the paper does not exceed 16 pages.
- [X] Confirm that the required GenAI and competing-interest disclosures are included within the 16-page PDF.

**Verification:** `pdfinfo` reports 16 or fewer pages and the source contains no layout hacks.

---

## Phase 7 — Build and Visual Quality Assurance

### 7.1 Perform a clean LaTeX build

- [X] Isolate old generated build artifacts by compiling from a new temporary directory.
- [X] Compile from a clean directory using the disclosure-complete source files.
- [X] Run the required LaTeX/BibTeX passes until references stabilize.
- [X] Confirm that there are no undefined citations.
- [X] Confirm that there are no undefined references.
- [X] Confirm that there are no multiply defined labels.
- [X] Fix the previously reported overfull table line.
- [X] Inspect and resolve every remaining overfull-box warning; the current log
  contains no overfull boxes.
- [X] Review the two remaining underfull-page warnings; page-by-page inspection
  confirms that neither visibly damages the layout.

**Verification:** The clean build succeeds and the log contains no unresolved citation, reference, label, or margin-overflow issue.

### 7.2 Inspect the PDF mechanically

- [X] Confirm the current page count (16) and US-Letter page dimensions
  (612 by 792 points), matching the documentation PDF bundled with Springer's
  current LNCS template.
- [X] Confirm that all fonts are embedded.
- [X] Confirm that text can be selected and copied.
- [X] Confirm that all external hyperlinks resolve to the intended targets
  (HTTP 200 or accepted 202 response on July 29, 2026).
- [X] Confirm that no hidden page numbers or headers remain.
- [X] Confirm that the PDF metadata title and authors are correct and follow the
  supplied author order.

**Verification:** PDF inspection tools report embedded fonts and correct metadata, with no structural issue.

### 7.3 Inspect every page visually

- [X] Render all 16 disclosure-complete PDF pages to images.
- [X] Inspect every page at normal reading size.
- [X] Check the revised title and author block visually; all three names and
  emails fit cleanly on page 1.
- [X] Check equations, symbols, and special characters.
- [X] Check all tables for clipping and minimum text size.
- [X] Check Fig. 1 for legibility and grayscale contrast.
- [X] Check paragraph and section breaks, including the new disclosure section.
- [X] Check references for broken URLs, awkward wrapping, and capitalization.
- [X] Confirm that the final page is not accidentally blank.

**Verification:** Every rendered page has been inspected and approved.

### 7.4 Conduct the final content review

- [ ] Ask each coauthor to review the complete final PDF.
- [ ] Obtain explicit approval of the author block.
- [ ] Obtain explicit approval of the scientific claims and reviewer changes.
- [X] Run a pre-final spelling, grammar, and LaTeX style check; rerun after
  author metadata and disclosures are inserted.
- [X] Confirm consistent US English throughout.
- [ ] Freeze the manuscript after all authors approve it.

**Verification:** All authors have approved the same frozen PDF.

---

## Phase 8 — License and Submission Package

### 8.1 Complete the License-to-Publish form

- [X] Download the official form linked by NGEN-AI to `ngenai/admin/License-to-Publish-Form.docx`.
- [X] Copy the final paper title exactly into the prefilled form.
- [X] Copy all final author names exactly and in the correct order into the
  prefilled form.
- [X] Identify Aniruddha Chattopadhyay as the corresponding author in both the
  manuscript and prefilled form.
- [X] Prefill Aniruddha Chattopadhyay's print name, affiliation/address, and
  email while leaving the handwritten signature and actual signing date blank.
- [X] Prepare an internal coauthor approval and authorization form at
  `output/pdf/NGEN-AI-2026-Paper-15-Coauthor-Approval.pdf`; keep completed
  copies in the project archive and exclude them from the camera-ready ZIP.
- [ ] Obtain approval from all coauthors for the corresponding author to sign.
- [X] Have the corresponding author sign the form by hand.
- [X] Scan the signed form as a readable PDF.
- [X] Confirm that the form does not contain a typed or digital signature.

**Verification:** The signed form exactly matches the frozen manuscript metadata.

### 8.2 Assemble a clean submission directory

- [X] Create the clean staging directory at `staging/ngenai-camera-ready`.
- [X] Add the frozen final PDF.
- [X] Add the final `.tex` source.
- [X] Add the final `.bib`.
- [X] Add the regenerated `.bbl`.
- [X] Add the required `llncs.cls` to staging.
- [X] Add the required `splncs04.bst` to staging.
- [X] Confirm that no additional style or font file is required to compile.
- [X] Confirm that no external figure file is required; Fig. 1 is native TikZ.
- [X] Add `Alt-Text.txt` to staging.
- [X] Add the signed License-to-Publish PDF.
- [X] Confirm that no copyright-permission document is required because the only
  figure is author-controlled TikZ and no third-party material is reused.
- [X] Confirm that no supplementary material has been approved for inclusion.

**Verification:** Every included file is required by the final submission instructions.

### 8.3 Remove prohibited or unnecessary files

- [X] Exclude anonymous manuscript versions from staging.
- [X] Exclude the root and IntelliSys manuscript variants from staging.
- [X] Exclude unused `accuracy_bars.pdf` and `drift_ablation.pdf` files.
- [X] Exclude `.aux`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, and `.out` files.
- [X] Exclude editor settings, caches, `.DS_Store`, and repository metadata.
- [X] Exclude previous PDFs and previous ZIP files.

**Verification:** The package contains no previous version, unused asset, or local build artifact.

### 8.4 Test the final ZIP

- [X] Create the ZIP from the clean submission directory.
- [X] Extract it into a new temporary directory.
- [X] Pre-test the staging source in an isolated temporary directory; it compiles
  using only staged files.
- [X] Compare the pre-test rebuild with the current PDF; extracted text and all
  rendered page images are identical.
- [X] Open every non-source document in the extracted package.
- [X] Confirm that `Alt-Text.txt` and the signed license are readable.
- [X] Record the final ZIP filename, size, and checksum:
  - Filename: `NGEN-AI-2026_Paper-15_Camera-Ready.zip`
  - Size: 674,415 bytes
  - SHA-256: `48dab9f20405091d85d369db66e5b0afebb5e1b0cb2d3fb9a658171c086343d0`

**Verification:** The extracted package reproduces the frozen PDF without accessing files outside the ZIP.

---

## Phase 9 — Submit and Archive

### 9.1 Upload the camera-ready package

- [ ] Upload the final ZIP by the internal target of August 8.
- [ ] Complete every required metadata field.
- [ ] Confirm that the uploaded title, author names, order, corresponding author, and abstract match the PDF.
- [ ] Download or reopen the uploaded package if the portal permits.
- [ ] Confirm that the uploaded file is the final tested ZIP.
- [ ] Submit rather than leaving the upload in draft state.

**Verification:** The portal reports a completed camera-ready submission for paper 15.

### 9.2 Save submission evidence

- [ ] Save the submission confirmation page or email.
- [X] Save the final ZIP checksum in this checklist.
- [X] Save the registration receipt at
  `ngenai/admin/registration/Receipt_PQ9300882026.pdf`.
- [X] Save the signed License-to-Publish form at
  `ngenai/admin/License-to-Publish-Form.pdf` and in the final ZIP.
- [ ] Save the frozen PDF and exact source package in a permanent archive.
- [ ] Notify all coauthors that submission is complete.

**Verification:** The team can prove exactly what was registered and submitted.

### 9.3 Monitor post-submission communication

- [ ] Monitor the corresponding-author inbox and spam folder.
- [ ] Respond to Springer proof requests within 72 hours.
- [ ] Restrict proof-stage changes to production corrections rather than new scientific changes.
- [ ] Save the final publication proof and approval confirmation.

**Verification:** The Springer proof has been checked and approved on time.

---

## Phase 10 — Presentation Readiness

- [ ] Confirm the assigned presentation format and duration.
- [X] Confirm that Aniruddha Chattopadhyay will present remotely over Zoom.
- [ ] Prepare the slide deck using the final camera-ready claims and numbers.
- [ ] Include the main result, oracle/non-oracle distinction, overhead, and limitations.
- [ ] Rehearse within the assigned time.
- [ ] Test the remote connection or in-person presentation equipment as applicable.
- [ ] Ensure that at least one author presents the paper.

**Verification:** The designated presenter can deliver the accepted paper in the assigned format and duration.

---

## Reviewer Traceability

| Reviewer | Comment                                 | Checklist items |
| -------- | --------------------------------------- | --------------- |
| 1        | Candidate-set sentence unclear          | 2.1             |
| 1        | “Load-bearing” unclear                | 2.2             |
| 1        | Oracle needs explanation                | 2.3             |
| 1        | Latency overhead                        | 2.4, 4.2        |
| 1        | Less benefit for a stronger model       | 2.4, 5.1        |
| 2        | Broader model evaluation                | 3.1, 4.1        |
| 2        | Real API version histories              | 3.2             |
| 2        | Multi-step agents and larger ecosystems | 3.3             |
| 3        | Qwen-family concentration               | 4.1             |
| 3        | Llama tool-name omission                | 4.1             |
| 3        | Synthetic ecological validity           | 3.2             |
| 3        | Systematic latency and API cost         | 4.2             |
| 3        | Baseline patching versus drift recovery | 4.3             |
| 4        | No actionable review text supplied      | 4.4             |

## Official Links

- NGEN-AI camera-ready instructions: [https://ngen-ai.org/cameraReady.php](https://ngen-ai.org/cameraReady.php)
- NGEN-AI registration: [https://ngen-ai.org/registration.php](https://ngen-ai.org/registration.php)
- Official PayQlick event registration portal:
  [https://events.payqlick.com/event/153/ngen-ai-2026](https://events.payqlick.com/event/153/ngen-ai-2026)
- NGEN-AI submission guidelines: [https://ngen-ai.org/submission.php](https://ngen-ai.org/submission.php)
- Springer Computer Science proceedings templates: [https://link.springer.com/series/558/information-for-authors-and-editors](https://link.springer.com/series/558/information-for-authors-and-editors)
