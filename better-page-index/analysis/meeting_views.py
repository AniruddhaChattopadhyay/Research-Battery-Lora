"""Write readable HTML views of the probe for a meeting. Reads the saved runs; makes no API call.

Writes:
  results/view_tree_designqa.html     the PageIndex tree of the DesignQA rules PDF
  results/view_tree_longmemeval.html  the PageIndex tree of one LongMemEval history ("What size is my new Samsung TV?")
  results/view_questions.html         the probe questions, the gold answers and the result of each system
Usage: uv run python analysis/meeting_views.py
"""
import html
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_report import OUT, PROBE, ROOT, load, tree_of, tree_stats  # noqa: E402

CSS = """
body{font:15px/1.45 -apple-system,Helvetica,Arial,sans-serif;max-width:1000px;margin:28px auto;padding:0 20px;color:#1a1c1e;background:#fff}
h1{font-size:22px;margin:0 0 8px} h2{font-size:18px;margin:30px 0 8px}
.note{color:#555;margin:3px 0} code{font:13px Menlo,monospace;background:#f2f3f4;padding:1px 4px;border-radius:3px}
details details{margin-left:20px} summary{cursor:pointer;padding:2px 0}
.pg{display:inline-block;min-width:80px;color:#74787e;font-size:13px}
.sum{margin:3px 0 6px 24px} .keys{margin:0 0 8px 24px;color:#74787e;font-size:13px}
.hit>summary{background:#fff2b3}
table{border-collapse:collapse;width:100%;font-size:14px;margin-top:8px}
th,td{border-bottom:1px solid #e3e5e8;padding:5px 8px;text-align:left;vertical-align:top} th{background:#f7f8f9}
td.n{text-align:right;font-variant-numeric:tabular-nums} tr.mean td{font-weight:600}
.ok{color:#2e6b3f;font-weight:600} .no{color:#8a3a2c;font-weight:600}
pre{white-space:pre-wrap;background:#f7f8f9;padding:10px;border-radius:4px;font-size:13px}
"""
BUTTONS = ('<p><button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=true)">Open all</button> '
           '<button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=false)">Close all</button></p>')


def page(title, body):
    return "<!doctype html><meta charset=utf-8><title>%s</title><style>%s</style>%s" % (html.escape(title), CSS, body)


def node_html(n, hits):
    a, b = n["start_index"], n["end_index"]
    pages = "p. %d" % a if a == b else "pp. %d–%d" % (a, b)
    inner = "<p class=sum>%s</p>" % html.escape(n.get("summary", ""))
    if n.get("key_items"):
        inner += "<p class=keys>Key items: %s</p>" % html.escape(", ".join(n["key_items"]))
    inner += "".join(node_html(c, hits) for c in n.get("nodes", []))
    # Same rule as leaf_labels() in probe_report.py: the session node that starts on the first page of the session.
    cls = " class=hit" if a in hits and n.get("title", "").startswith("Session") else ""
    return "<details%s><summary><span class=pg>%s</span>%s</summary>%s</details>" % (
        cls, pages, html.escape(n.get("title", "")), inner)


def tree_view(item, title, notes, hits=()):
    tree, _ = tree_of(item["pdf"])
    s = tree_stats(item["pdf"])
    d = json.load(open(os.path.join(PROBE, "pi_docs.json")))[item["pdf"]]
    src = os.path.relpath(os.path.join(d["store"], "docs", d["doc_id"], "tree.json"), ROOT)
    body = "<h1>%s</h1>" % html.escape(title)
    body += "".join("<p class=note>%s</p>" % n for n in notes)
    body += "<p class=note>%d nodes. The labels (titles and summaries) hold %s words: %.0f%% of the %s words in the PDF.</p>" % (
        s["nodes"], format(s["label_words"], ","), 100 * s["label_share"], format(s["page_words"], ","))
    body += "<p class=note>Source file: <code>%s</code>. Click a node to see its summary and its children.</p>" % src
    return page(title, body + BUTTONS + "".join(node_html(n, set(hits)) for n in tree))


def mark(v):
    return "<span class=ok>✓</span>" if v else "<span class=no>✗</span>"


def questions_view(items, pi, full, judge):
    body = "<h1>The probe questions</h1>"
    body += ("<p class=note>All questions: <code>runs/probe/items.jsonl</code>. <code>src/probe_data.py</code> made it "
             "from <code>data/designqa/rule_retrieval_qa.csv</code>, <code>data/designqa/rule_compilation_qa.csv</code> "
             "and <code>data/longmemeval/longmemeval_s_cleaned.json</code>.</p>"
             "<p class=note>Answers: <code>runs/probe/pageindex.jsonl</code> and <code>full.jsonl</code> (DesignQA, "
             "gpt-5.6-sol); <code>*_luna.jsonl</code> and <code>judge.jsonl</code> (LongMemEval, gpt-5.6-luna).</p>")
    for t, heading in (("retrieval", "DesignQA Retrieval: one rule, by its number"),
                       ("compilation", "DesignQA Compilation: all rules about a topic")):
        rows = [i for i in items.values() if i["dataset"] == "designqa" and i["type"] == t and i["id"] in pi]
        ex = rows[0]
        gold = ex["gold"] if t == "retrieval" else ", ".join(ex["gold"])
        body += "<h2>%s (%d questions)</h2>" % (heading, len(rows))
        body += "<p class=note>Example question, word for word:</p><pre>%s</pre>" % html.escape(ex["question"])
        body += "<p class=note>The gold answer of this example:</p><pre>%s</pre>" % html.escape(gold)
        head = "<th>Rule</th>" if t == "retrieval" else "<th>Topic</th><th>Gold rules</th>"
        body += "<table><tr>%s<th>PageIndex F1</th><th>Full-context F1</th></tr>" % head
        for i in rows:
            first = "<td>%s</td>" % i["rule"] if t == "retrieval" else "<td>%s</td><td class=n>%d</td>" % (
                html.escape(i["topic"]), len(i["gold"]))
            body += "<tr>%s<td class=n>%.2f</td><td class=n>%.2f</td></tr>" % (
                first, pi[i["id"]]["score"]["f1"], full[i["id"]]["score"]["f1"])
        pad = "<td></td>" if t == "retrieval" else "<td></td><td></td>"
        body += "<tr class=mean>%s<td class=n>%.3f</td><td class=n>%.3f</td></tr></table>" % (
            pad.replace("<td></td>", "<td>Mean</td>", 1), st.mean(pi[i["id"]]["score"]["f1"] for i in rows),
            st.mean(full[i["id"]]["score"]["f1"] for i in rows))
    rows = [i for i in items.values() if i["dataset"] == "longmemeval" and i["id"] in judge]
    systems = ("pageindex_luna", "full_luna", "keyword_luna")
    body += "<h2>LongMemEval: questions about a dated chat history (%d questions)</h2>" % len(rows)
    body += ("<p class=note>Each question has its own history PDF in <code>runs/probe/lme_pdfs/</code>. "
             "The prompt starts with the current date. The official LongMemEval judge prompts decide ✓ or ✗.</p>")
    body += "<table><tr><th>Type</th><th>Question</th><th>Gold answer</th><th>PageIndex</th><th>Full context</th><th>Keyword agent</th></tr>"
    for i in rows:
        body += "<tr><td>%s</td><td>%s</td><td>%s</td>%s</tr>" % (
            i["type"], html.escape(i["raw_question"]), html.escape(str(i["gold"])),
            "".join("<td class=n>%s</td>" % mark(judge[i["id"]].get(s)) for s in systems))
    body += "<tr class=mean><td>Correct</td><td></td><td></td>%s</tr></table>" % "".join(
        "<td class=n>%d of %d</td>" % (sum(bool(judge[i["id"]].get(s)) for i in rows), len(rows)) for s in systems)
    return page("The probe questions", body)


def main():
    items = {i["id"]: i for i in map(json.loads, open(os.path.join(PROBE, "items.jsonl")))}
    pi, full = load("pageindex.jsonl"), load("full.jsonl")
    judge = {}
    for r in map(json.loads, open(os.path.join(PROBE, "judge.jsonl"))):
        judge.setdefault(r["id"], {})[r["system"]] = r["correct"]

    dqa = items["dqa-comp-00"]
    views = {"view_tree_designqa.html": tree_view(dqa, "PageIndex tree: the DesignQA rules PDF", [
        "Formula SAE Rules 2024, 140 pages, 588 bookmarks. PageIndex 0.4.1 built this tree in Flash mode; "
        "gpt-5.6-luna wrote the summaries.",
        "The agent gets these fields through the tool <code>get_document_structure</code>, in 4 parts."])}

    lme = items["lme-76d63226"]
    views["view_tree_longmemeval.html"] = tree_view(lme, "PageIndex tree: one LongMemEval chat history", [
        "Question: “%s” Gold answer: “%s”." % (html.escape(lme["raw_question"]), html.escape(lme["gold"])),
        "One chapter for each dated session. The yellow node is the session with the answer."],
        hits=[first for first, _ in lme["answer_sessions"].values()])

    views["view_questions.html"] = questions_view(items, pi, full, judge)
    for name, text in views.items():
        with open(os.path.join(OUT, name), "w") as f:
            f.write(text)
        print("wrote", os.path.relpath(os.path.join(OUT, name), ROOT))


if __name__ == "__main__":
    main()
