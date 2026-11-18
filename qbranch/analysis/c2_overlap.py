"""Check the strict C2 control (random_fit of c2_v2.py): is the control question also a prediction?

The control rejects a draw only when the relation name is the same. Relation names are free text, so the
control question can be a paraphrase of the predicted question (born_on against birth_date), or a question
that the build already opened about the same entity (in this passage or in another passage, up to the batch
of the slot). Then the control is also a prediction, and predicted - control goes toward zero by design.

For each row with a control question, the checker sees the control question and the open questions that the
build raised about the entity up to the batch of the slot (the predicted question first, then the 29 nearest
others). It names the ones that ask for the same fact. A row "overlaps" when at least one does.
Output: the overlap share, and predicted - control on the rows that do not overlap. With two files, also the
difference between the variants (unpaired bootstrap, because the two samples have different slots).

Usage: uv run python analysis/c2_overlap.py runs/multihoprag_nall qbranch [qbranch_qa]
Output: <run_dir>/c2_overlap_<variant>.json
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import answer  # noqa: E402
import build  # noqa: E402
import llm  # noqa: E402
from graph import Graph, norm_entity  # noqa: E402

CHECK_MODEL = "anthropic/claude-haiku-5.5"
PROMPT = """Control question: {control}

Open questions that a system raised about the same entity:
{slots}

Which open questions ask for the same fact as the control question (the same answer would answer both)? Small differences in wording do not matter.
Return only a JSON object: {{"same": [<numbers>]}}"""


def hit(r, key):
    return float(bool(r.get(key) and r[key]["answered"]))


def boot(d, seed=0, n=5000):
    bs = np.random.default_rng(seed)
    b = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def run(run_dir, variant):
    d = json.load(open(os.path.join(run_dir, "c2_%s.json" % variant)))
    g = Graph(build.load_records(os.path.join(run_dir, "build_" + variant)), variant)
    by_ent = collections.defaultdict(list)
    for s in g.open_slots:
        if s["iqid"] is not None:
            by_ent[s["s_norm"]].append(s)
    rows = [r for r in d["rows"] if r.get("q_fit")]

    def one(r):
        own = [s for s in by_ent.get(norm_entity(r["slot"][0]), []) if s["batch"] <= r["batch"]]
        others = list(dict.fromkeys(g.agenda[s["iqid"]]["q"] for s in own if g.agenda[s["iqid"]]["q"] != r["q"]))
        if len(others) > 29:
            sc = answer.embed_queries(others) @ answer.embed_queries([r["q_fit"]])[0]
            others = [others[i] for i in np.argsort(-sc)[:29]]
        qs = [r["q"]] + others
        lines = "\n".join("%d. %s" % (i + 1, q) for i, q in enumerate(qs))
        t = llm.chat(CHECK_MODEL, PROMPT.format(control=r["q_fit"], slots=lines), temperature=None,
                     max_tokens=150)["text"]
        same = [k for k in ((llm.parse_json(t) or {}).get("same") or []) if isinstance(k, int) and 0 < k <= len(qs)]
        return {"same_as_predicted": 1 in same, "same_as_other_slot": any(k > 1 for k in same),
                "n_entity_slots": len(qs)}

    with ThreadPoolExecutor(8) as ex:
        marks = list(ex.map(one, rows))
    for r, m in zip(rows, marks):
        r.update(m)
    clean = [r for r in rows if not (r["same_as_predicted"] or r["same_as_other_slot"])]
    diff = lambda rs: np.array([hit(r, "pred") - hit(r, "random_fit") for r in rs])  # noqa: E731
    out = {"run_dir": run_dir, "variant": variant, "rows": len(rows),
           "overlap_predicted": float(np.mean([r["same_as_predicted"] for r in rows])),
           "overlap_other_slot": float(np.mean([r["same_as_other_slot"] for r in rows])),
           "overlap_any": 1 - len(clean) / len(rows),
           "all_rows": {"pred": float(np.mean([hit(r, "pred") for r in rows])),
                        "control": float(np.mean([hit(r, "random_fit") for r in rows])), "diff": boot(diff(rows))},
           "rows_without_overlap": {"n": len(clean), "pred": float(np.mean([hit(r, "pred") for r in clean])),
                                    "control": float(np.mean([hit(r, "random_fit") for r in clean])),
                                    "diff": boot(diff(clean))},
           "examples_overlap": [[r["q"], r["q_fit"], r["same_as_predicted"]] for r in rows
                                if r["same_as_predicted"] or r["same_as_other_slot"]][:10]}
    json.dump(out, open(os.path.join(run_dir, "c2_overlap_%s.json" % variant), "w"), indent=1)
    return out, rows, clean


run_dir = sys.argv[1]
res = {}
for v in sys.argv[2:]:
    out, rows, clean = run(run_dir, v)
    res[v] = (rows, clean)
    print(json.dumps({k: x for k, x in out.items() if k != "examples_overlap"}), flush=True)
if len(res) == 2:  # variant 2 - variant 1 of (pred - control), unpaired bootstrap
    (a, (ra, ca)), (b, (rb, cb)) = list(res.items())
    bs = np.random.default_rng(1)
    for name, xa, xb in (("all_rows", ra, rb), ("rows_without_overlap", ca, cb)):
        da = np.array([hit(r, "pred") - hit(r, "random_fit") for r in xa])
        db = np.array([hit(r, "pred") - hit(r, "random_fit") for r in xb])
        boots = [db[bs.integers(0, len(db), len(db))].mean() - da[bs.integers(0, len(da), len(da))].mean()
                 for _ in range(5000)]
        print("%s - %s, %s: %+.3f [%+.3f, %+.3f]" % (b, a, name, db.mean() - da.mean(), np.percentile(boots, 2.5),
                                                     np.percentile(boots, 97.5)))
print("cost", llm.usage_report()["total_cost_usd"])
