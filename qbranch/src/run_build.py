"""Run the online build for one dataset slice and one variant.

Example: uv run python src/run_build.py --dataset hotpotqa --n 100 --variant qbranch --batches 5
Output: runs/<dataset>_n<n>/build_<variant>/batch_XX.jsonl and usage.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build  # noqa: E402
import data  # noqa: E402
import llm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="hotpotqa")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--variant", choices=["qbranch", "qbranch_noopen", "openie", "qbranch_qa", "qbranch_events"],
                default="qbranch")
ap.add_argument("--k_examples", type=int, default=8, help="qbranch_qa: pool pairs per passage")
ap.add_argument("--batches", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()

qs, ps = data.load(a.dataset, a.n or None)
run_dir = os.path.join(llm.ROOT, "runs", "%s_n%s" % (a.dataset, a.n or "all"))
out_dir = os.path.join(run_dir, "build_" + a.variant)
print("%s: %d questions, %d passages, %d batches -> %s" % (a.dataset, len(qs), len(ps), a.batches, out_dir))
examples_fn = None
if a.variant == "qbranch_qa":
    pool, eval_ids = data.faq_pool(a.dataset, qs)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"pool": pool, "eval_qids": eval_ids, "k": a.k_examples},
              open(os.path.join(run_dir, "faq_pool.json"), "w"), indent=1, default=list)
    print("example pool: %d questions, k = %d per passage, evaluate on %d questions" % (len(pool), a.k_examples,
                                                                                         len(eval_ids)))
    examples_fn = lambda pid: data.faq_examples(pool, pid, a.k_examples)  # noqa: E731
build.build(data.stream_batches(ps, a.batches, a.seed), a.variant, out_dir, workers=a.workers,
            examples_fn=examples_fn)
recs = build.load_records(out_dir)
cost = sum((r.get("usage") or {}).get("cost", 0) for r in recs)
rep = {**llm.usage_report(), "build_cost_from_records_usd": round(cost, 4)}
json.dump(rep, open(os.path.join(out_dir, "usage.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
