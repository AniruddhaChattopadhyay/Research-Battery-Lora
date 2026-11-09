"""Analyze the probe: scores, PageIndex reading behavior, labels, cost, and the predictions P1 to P10.

Reads runs/probe/*.jsonl; writes results/probe_summary.json and the probe figures.
Usage: uv run python analysis/probe_report.py
"""
import collections
import glob
import json
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROBE = os.path.join(ROOT, "runs", "probe")
OUT = os.path.join(ROOT, "results")
BLUE, ORANGE, AQUA, MUTE, INK = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b"
STOP = set("a an the of to in on for and or my i me do did does what when where which who how is was were am are "
           "have has had with from at by it its this that these those many much be been about as than ago last "
           "since total currently current previous previously".split())


def load(name):
    path = os.path.join(PROBE, name)
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def pages_of(spec):
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        m = re.match(r"^(\d+)(?:-(\d+))?$", part)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
    return out


def pi_trace(rec):
    calls = [e for e in rec.get("events", []) if e.get("type") == "tool_call"]
    pages = set()
    for c in calls:
        if c.get("name") == "get_page_content":
            pages |= pages_of((c.get("arguments") or {}).get("pages", ""))
    return {"calls": len(calls), "struct": sum(c.get("name") == "get_document_structure" for c in calls),
            "reads": sum(c.get("name") == "get_page_content" for c in calls), "pages": pages}


def tree_of(pdf):
    docs = json.load(open(os.path.join(PROBE, "pi_docs.json")))
    d = docs[pdf]
    return json.load(open(os.path.join(d["store"], "docs", d["doc_id"], "tree.json"))), \
        json.load(open(os.path.join(d["store"], "docs", d["doc_id"], "pages.json")))


def walk(n):
    yield n
    for c in n.get("nodes", []):
        yield from walk(c)


def label(n):
    return " ".join([n.get("title", ""), n.get("summary", ""), n.get("prefix_summary", "")])


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def tree_stats(pdf):
    tree, pages = tree_of(pdf)
    nodes = [n for t in tree for n in walk(t)]
    lw = sum(len(label(n).split()) for n in nodes)
    pw = sum(len(p["markdown"].split()) for p in pages)
    return {"nodes": len(nodes), "label_words": lw, "page_words": pw, "label_share": lw / pw}


def leaf_labels(pdf, first, last):
    """Label of the session node: the node that starts on the first page of the session (each session starts a page)."""
    tree, _ = tree_of(pdf)
    return " ".join(label(n) for t in tree for n in walk(t)
                    if n.get("start_index") == first and n.get("title", "").startswith("Session"))


def main():
    items = {i["id"]: i for i in map(json.loads, open(os.path.join(PROBE, "items.jsonl")))}
    pi, full, kw = load("pageindex.jsonl"), load("full.jsonl"), load("keyword.jsonl")
    pil, fulll, kwl = load("pageindex_luna.jsonl"), load("full_luna.jsonl"), load("keyword_luna.jsonl")
    judge = collections.defaultdict(dict)
    for r in map(json.loads, open(os.path.join(PROBE, "judge.jsonl"))):
        judge[r["id"]][r["system"]] = r["correct"]
    S = {}

    # ---------------- DesignQA (gpt-5.6-sol)
    dqa = [i for i in items.values() if i["dataset"] == "designqa" and i["id"] in pi and i["id"] in full]
    doc_tokens = st.median(full[i["id"]]["tokens"]["prompt"] for i in dqa)
    S["designqa"] = {"document_tokens": doc_tokens, "tree": tree_stats(dqa[0]["pdf"])}
    rows = []
    for i in dqa:
        tr = pi_trace(pi[i["id"]])
        gold = set(i["gold_pages"])
        rows.append({"id": i["id"], "type": i["type"], "n_gold": len(i["gold"]) if i["type"] == "compilation" else 1,
                     "pi_f1": pi[i["id"]]["score"]["f1"], "full_f1": full[i["id"]]["score"]["f1"],
                     "pi_recall": pi[i["id"]]["score"].get("recall"), "full_recall": full[i["id"]]["score"].get("recall"),
                     "pi_pages": len(tr["pages"]), "pi_gold_page_recall": len(tr["pages"] & gold) / len(gold),
                     "pi_prompt": pi[i["id"]]["tokens"]["prompt"], "full_prompt": full[i["id"]]["tokens"]["prompt"],
                     "pi_cost": pi[i["id"]]["cost"], "full_cost": full[i["id"]]["cost"], "pi_struct": tr["struct"]})
    for t in ("retrieval", "compilation"):
        rs = [r for r in rows if r["type"] == t]
        diffs = [r["pi_f1"] - r["full_f1"] for r in rs]
        S["designqa"][t] = {
            "n": len(rs), "pageindex_f1": st.mean(r["pi_f1"] for r in rs), "full_f1": st.mean(r["full_f1"] for r in rs),
            "mean_diff": st.mean(diffs), "pi_better": sum(d > 1e-9 for d in diffs), "full_better": sum(d < -1e-9 for d in diffs),
            "pi_pages_median": st.median(r["pi_pages"] for r in rs),
            "pi_gold_page_recall_median": st.median(r["pi_gold_page_recall"] for r in rs),
            "pi_prompt_median": st.median(r["pi_prompt"] for r in rs),
            "full_prompt_median": st.median(r["full_prompt"] for r in rs),
            "pi_tree_parts_median": st.median(r["pi_struct"] for r in rs)}
    comp = [r for r in rows if r["type"] == "compilation"]

    def spearman(x, y):
        rank = lambda v: {k: r for r, k in enumerate(sorted(range(len(v)), key=lambda j: v[j]))}
        rx, ry = rank(x), rank(y)
        n = len(x)
        mx = my = (n - 1) / 2
        cov = sum((rx[j] - mx) * (ry[j] - my) for j in range(n))
        return cov / (sum((rx[j] - mx) ** 2 for j in range(n)) * sum((ry[j] - my) ** 2 for j in range(n))) ** 0.5

    S["designqa"]["compilation"]["spearman_pi_f1_vs_gold_size"] = spearman([r["n_gold"] for r in comp], [r["pi_f1"] for r in comp])
    S["designqa"]["compilation"]["spearman_full_f1_vs_gold_size"] = spearman([r["n_gold"] for r in comp], [r["full_f1"] for r in comp])
    S["designqa"]["rows"] = rows

    # ---------------- LongMemEval (gpt-5.6-luna for all three systems)
    lme = [i for i in items.values() if i["dataset"] == "longmemeval" and i["id"] in pil]
    lrows = []
    for i in lme:
        tr = pi_trace(pil[i["id"]])
        gold = set(i["gold_pages"])
        sess_read = sum(1 for a, b in i["answer_sessions"].values() if tr["pages"] & set(range(a, b + 1)))
        qwords = [w for w in words(i["raw_question"]) if w not in STOP and len(w) > 2]
        labels = " ".join(leaf_labels(i["pdf"], a, b) for a, b in i["answer_sessions"].values())
        key_in_label = sum(w in set(words(labels)) for w in qwords) / max(1, len(qwords))
        gwords = [w for w in words(str(i["gold"])) if w not in STOP and len(w) > 1]
        answer_in_label = sum(w in set(words(labels)) for w in gwords) / max(1, len(gwords))
        lrows.append({"id": i["id"], "type": i["type"], "n_sessions": i["n_sessions"],
                      "answer_sessions": len(i["answer_sessions"]), "answer_sessions_read": sess_read,
                      "pi_pages": len(tr["pages"]), "pi_tree_parts": tr["struct"],
                      "key_words_in_answer_session_labels": round(key_in_label, 2),
                      "answer_words_in_answer_session_labels": round(answer_in_label, 2),
                      "correct": {s: judge[i["id"]].get(s) for s in ("pageindex_luna", "full_luna", "keyword_luna")},
                      "pi_prompt": pil[i["id"]]["tokens"]["prompt"], "full_prompt": fulll[i["id"]]["tokens"]["prompt"],
                      "kw_prompt": kwl[i["id"]]["tokens"]["prompt"],
                      "cost": {"pageindex": pil[i["id"]]["cost"], "full": fulll[i["id"]]["cost"], "keyword": kwl[i["id"]]["cost"]}})
    S["longmemeval"] = {"rows": lrows, "trees": [tree_stats(i["pdf"]) for i in lme]}
    S["longmemeval"]["label_share_median"] = st.median(t["label_share"] for t in S["longmemeval"]["trees"])
    by = collections.defaultdict(lambda: collections.Counter())
    for r in lrows:
        for s, ok in r["correct"].items():
            by[r["type"]][s] += bool(ok)
        by[r["type"]]["n"] += 1
    S["longmemeval"]["by_type"] = {t: dict(c) for t, c in by.items()}
    for k in ("pi_prompt", "full_prompt", "kw_prompt"):
        S["longmemeval"][k + "_median"] = st.median(r[k] for r in lrows)

    # ---------------- Predictions
    d, c = S["designqa"], S["designqa"]["compilation"]
    ssu = [r for r in lrows if r["type"] == "single-session-user"]
    hard = [r for r in lrows if r["type"] in ("multi-session", "knowledge-update", "temporal-reasoning")]
    absq = [r for r in lrows if r["type"] == "abstention"]
    wrong = lambda rs, s: sum(1 for r in rs if not r["correct"][s])
    P = {
        "P1": (abs(d["retrieval"]["mean_diff"]) <= 0.05, "Retrieval F1: PageIndex %.3f, full %.3f" % (d["retrieval"]["pageindex_f1"], d["retrieval"]["full_f1"])),
        "P2": (c["mean_diff"] <= -0.10, "Compilation F1: PageIndex %.3f, full %.3f" % (c["pageindex_f1"], c["full_f1"])),
        "P3": (c["spearman_pi_f1_vs_gold_size"] < 0, "Spearman(PageIndex F1, gold size) = %.2f" % c["spearman_pi_f1_vs_gold_size"]),
        "P4": (c["pi_gold_page_recall_median"] < 0.5, "Median share of gold pages read: %.2f" % c["pi_gold_page_recall_median"]),
        "P5": (None, "Keyword agent not run on DesignQA (budget)"),
        "P6": (wrong(ssu, "pageindex_luna") <= wrong(ssu, "full_luna") + 1, "Single-session-user wrong: PageIndex %d, full %d (n=%d)" % (wrong(ssu, "pageindex_luna"), wrong(ssu, "full_luna"), len(ssu))),
        "P7": (wrong(hard, "pageindex_luna") > wrong(hard, "full_luna"), "Multi-session, update, temporal wrong: PageIndex %d, full %d (n=%d)" % (wrong(hard, "pageindex_luna"), wrong(hard, "full_luna"), len(hard))),
        "P8": (None, "No PageIndex errors of the P7 kind to classify" if not [r for r in hard if not r["correct"]["pageindex_luna"]] else
               "PageIndex errors: answer sessions read %s" % [(r["answer_sessions_read"], r["answer_sessions"]) for r in hard if not r["correct"]["pageindex_luna"]]),
        "P9": (st.mean(r["pi_pages"] for r in absq) > st.mean(r["pi_pages"] for r in ssu),
               "Pages read: abstention %.1f, single-session-user %.1f" % (st.mean(r["pi_pages"] for r in absq), st.mean(r["pi_pages"] for r in ssu))),
        "P10": (d["tree"]["label_share"] > 0.15, "DesignQA labels = %.0f%% of the PDF words" % (100 * d["tree"]["label_share"])),
    }
    S["predictions"] = {k: {"held": v[0], "evidence": v[1]} for k, v in P.items()}
    json.dump(S, open(os.path.join(OUT, "probe_summary.json"), "w"), indent=1, default=str)

    # ---------------- print
    print("DesignQA (gpt-5.6-sol); document = %d tokens; tree %s" % (doc_tokens, {k: round(v, 2) for k, v in d["tree"].items()}))
    for t in ("retrieval", "compilation"):
        print(" ", t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d[t].items()})
    print("LongMemEval (gpt-5.6-luna); median label share %.2f" % S["longmemeval"]["label_share_median"])
    for t, cnt in sorted(S["longmemeval"]["by_type"].items()):
        print("  %-20s %s" % (t, dict(cnt)))
    for r in sorted(lrows, key=lambda r: r["type"]):
        print("  %-22s %-20s sessions read %d/%d pages %3d key-in-label %.2f answer-in-label %.2f %s" % (
            r["id"], r["type"], r["answer_sessions_read"], r["answer_sessions"], r["pi_pages"],
            r["key_words_in_answer_session_labels"], r["answer_words_in_answer_session_labels"],
            {k.split("_")[0]: v for k, v in r["correct"].items()}))
    print("tokens median: PageIndex %d, full %d, keyword %d" % (S["longmemeval"]["pi_prompt_median"],
                                                              S["longmemeval"]["full_prompt_median"], S["longmemeval"]["kw_prompt_median"]))
    print("Predictions:")
    for k, v in S["predictions"].items():
        print("  %-4s %-6s %s" % (k, {True: "held", False: "FALSE", None: "n/a"}[v["held"]], v["evidence"]))
    figures(S)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")


def figures(S):
    d = S["designqa"]
    # Figure 1: DesignQA F1 and input tokens
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    xs = range(2)
    for k, (key, name, col) in enumerate((("pageindex_f1", "PageIndex", BLUE), ("full_f1", "Full context", ORANGE))):
        vals = [d["retrieval"][key], d["compilation"][key]]
        ax.bar([x + (k - 0.5) * 0.36 for x in xs], vals, 0.34, color=col, label=name)
        for x, v in zip(xs, vals):
            ax.text(x + (k - 0.5) * 0.36, v + 0.02, "%.2f" % v, ha="center", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Retrieval (n=%d)\none rule, by its number" % d["retrieval"]["n"],
                        "Compilation (n=%d)\nall rules about a topic" % d["compilation"]["n"]])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mean F1 (official metric)")
    ax.set_title("Accuracy: the same", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    ax = axes[1]
    vals = [(d["retrieval"]["pi_prompt_median"], d["retrieval"]["full_prompt_median"]),
            (d["compilation"]["pi_prompt_median"], d["compilation"]["full_prompt_median"])]
    for k, (name, col) in enumerate((("PageIndex", BLUE), ("Full context", ORANGE))):
        ax.bar([x + (k - 0.5) * 0.36 for x in xs], [v[k] / 1000 for v in vals], 0.34, color=col, label=name)
        for x, v in zip(xs, vals):
            ax.text(x + (k - 0.5) * 0.36, v[k] / 1000 + 3, "%dk" % round(v[k] / 1000), ha="center", fontsize=10)
    ax.axhline(d["document_tokens"] / 1000, color=MUTE, ls="--", lw=1)
    ax.text(0.5, d["document_tokens"] / 1000 + 3, "whole document", color=MUTE, ha="center", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Retrieval", "Compilation"])
    ax.set_ylabel("input tokens per question (median, thousands)")
    ax.set_title("Cost: PageIndex reads up to 2x the document", loc="left", fontweight="bold")
    style(ax)
    fig.suptitle("DesignQA: one 140-page rules PDF, gpt-5.6-sol for both systems", x=0.01, ha="left", fontsize=11, color="#52514e")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe_fig1_designqa.png"), dpi=170)
    plt.close(fig)

    # Figure 2: compilation F1 against gold size
    rows = [r for r in d["rows"] if r["type"] == "compilation"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter([r["n_gold"] for r in rows], [r["pi_f1"] for r in rows], color=BLUE, label="PageIndex", s=28)
    ax.scatter([r["n_gold"] for r in rows], [r["full_f1"] for r in rows], color=ORANGE, label="Full context", s=28, marker="s")
    ax.set_xscale("log")
    ax.set_xlabel("gold rules in the answer (log scale)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Compilation: spread does not separate the systems", loc="left", fontweight="bold", fontsize=11)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe_fig2_spread.png"), dpi=170)
    plt.close(fig)

    # Figure 3: label share of the three trees
    lab = [("CLEVRER pilot\n(30 runs, numeric tables)", 0.22), ("LongMemEval\n(one chat history)", S["longmemeval"]["label_share_median"]),
           ("DesignQA\n(rules PDF)", d["tree"]["label_share"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([n for n, _ in lab], [100 * v for _, v in lab], color=BLUE)
    for k, (_, v) in enumerate(lab):
        ax.text(100 * v + 1, k, "%.0f%%" % (100 * v), va="center", fontsize=10)
    ax.set_xlabel("words in the tree labels, as % of the words in the PDF")
    ax.set_xlim(0, 70)
    ax.set_title("The size of the PageIndex index", loc="left", fontweight="bold", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe_fig3_labels.png"), dpi=170)
    plt.close(fig)
    print("wrote results/probe_fig1_designqa.png, probe_fig2_spread.png, probe_fig3_labels.png")


if __name__ == "__main__":
    main()
