# Submission Checklist

- Confirm the target venue page limit and bibliography policy against
  [main.pdf](/Users/aniruddha/Documents/research/tool-drift-paper/main.pdf).
- Verify anonymization in [main.tex](/Users/aniruddha/Documents/research/tool-drift-paper/main.tex): no author names, affiliations, acknowledgments, or self-identifying links.
- Freeze the paper around the current evidence set unless a new run materially changes the reviewer story.
- If rerunning experiments, use the corrected runner behavior in
  [run_pilot_bfcl.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_bfcl.py:423)
  and [run_pilot_dice.py](/Users/aniruddha/Documents/research/tool-drift/scripts/run_pilot_dice.py:425), which now honor `evaluation.sample_count`.
- For the closed-model sanity check, use
  [bfcl_stage3_200_gpt4omini_non_oracle.yaml](/Users/aniruddha/Documents/research/tool-drift/configs/bfcl_stage3_200_gpt4omini_non_oracle.yaml:1)
  together with provider-safe tool-name aliasing in
  [openrouter_client.py](/Users/aniruddha/Documents/research/tool-drift/inference/openrouter_client.py:17).
- Double-check that all cited result files still exist under `tool-drift/outputs/`.
- Rebuild with `latexmk -pdf -interaction=nonstopmode main.tex` immediately before submission.
- Upload the final PDF and any required source bundle.
