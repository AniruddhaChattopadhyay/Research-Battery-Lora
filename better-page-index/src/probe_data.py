"""Probe datasets: DesignQA (one rules PDF) and LongMemEval S (one rendered PDF for each question).

Each item has: id, dataset, type, pdf, question, gold, gold_pages (the PDF pages that hold the
evidence), and extra fields for scoring. All systems read the same PDF.

Usage: PYTHONPATH=src uv run python src/probe_data.py runs/probe/items.jsonl
"""
import ast
import csv
import json
import os
import random
import re
import string
import sys
from collections import Counter
from datetime import datetime

from fpdf import FPDF
from pypdf import PdfReader

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DQA = os.path.join(ROOT, "data", "designqa")
DQA_PDF = os.path.join(DQA, "FSAE_Rules_2024_V1.pdf")
LME = os.path.join(ROOT, "data", "longmemeval", "longmemeval_s_cleaned.json")
LME_PDFS = os.path.join(ROOT, "runs", "probe", "lme_pdfs")
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
SEED = 7
RULE_ID = re.compile(r"\b([A-Z]{1,3}(?:\.\d+){1,4})\b")

INSTRUCTIONS = {
    "designqa": "Answer from the attached document only. Follow the answer format that the question asks for, "
                "exactly. Give only the answer, with no other words.",
    "longmemeval": "The document is a history of chat sessions between a user and an AI assistant, in date order. "
                   "Answer the user's question from this history only. Use the current date that the question "
                   "gives. If the history does not contain the information, say that you do not know.",
}


# ---------------------------------------------------------------- DesignQA

def rule_pages():
    """Rule number -> the first page (1-based) where a line starts with that rule number."""
    reader = PdfReader(DQA_PDF)
    first = {}
    for i, page in enumerate(reader.pages):
        for line in (page.extract_text() or "").splitlines():
            m = re.match(r"^\s*([A-Z]{1,3}(?:\.\d+){1,4})\b", line)
            if m and m.group(1) not in first:
                first[m.group(1)] = i + 1
    return first


def designqa_items(n_retrieval=20):
    pages = rule_pages()
    items = []
    comp = list(csv.DictReader(open(os.path.join(DQA, "rule_compilation_qa.csv"))))
    for k, row in enumerate(comp):
        gold = ast.literal_eval(row["ground_truth"])
        topic = re.search(r"relevant to `([^`]+)`", row["question"]).group(1)
        items.append({"id": "dqa-comp-%02d" % k, "dataset": "designqa", "type": "compilation", "pdf": DQA_PDF,
                      "question": row["question"].strip(), "gold": gold, "topic": topic,
                      "gold_pages": sorted({pages[r] for r in gold if r in pages})})
    ret = list(csv.DictReader(open(os.path.join(DQA, "rule_retrieval_qa.csv"))))
    for k in sorted(random.Random(SEED).sample(range(len(ret)), n_retrieval)):
        row = ret[k]
        rule = re.search(r"What does rule (\S+) state", row["question"]).group(1)
        items.append({"id": "dqa-ret-%04d" % k, "dataset": "designqa", "type": "retrieval", "pdf": DQA_PDF,
                      "question": row["question"].strip(), "gold": row["ground_truth"], "rule": rule,
                      "gold_pages": [pages[rule]] if rule in pages else []})
    return items


def _normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(gold)
    return 2 * p * r / (p + r)


def score_designqa(item, answer):
    """The official DesignQA metrics. Compilation also gets a robust variant (rule numbers found by regex)."""
    if item["type"] == "retrieval":
        return {"f1": _f1(_normalize(answer).split(), _normalize(item["gold"]).split())}
    official = _f1([x.strip() for x in answer.strip().split(", ") if x.strip()], item["gold"])
    found = sorted(set(RULE_ID.findall(answer)))
    gold = set(item["gold"])
    hit = len(gold & set(found))
    return {"f1": _f1(found, list(gold)), "f1_official": official, "n_pred": len(found), "n_gold": len(gold),
            "recall": hit / len(gold), "precision": hit / len(found) if found else 0.0}


# ---------------------------------------------------------------- LongMemEval

def _when(s):
    return datetime.strptime(s[:10] + " " + s[-5:], "%Y/%m/%d %H:%M")


def _clean(text):
    text = text.replace("\t", "    ").replace("\r", "")
    return "".join(ch for ch in text if (ch == "\n" or ord(ch) >= 32) and ord(ch) <= 0xFFFF)


class HistoryPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, "Page %d" % self.page_no(), align="C")


def render_history(q, path):
    """One chapter (and one bookmark) for each session, in date order. Returns session id -> (first, last) page."""
    pdf = HistoryPDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("U", "", FONT)
    order = sorted(range(len(q["haystack_sessions"])), key=lambda i: _when(q["haystack_dates"][i]))
    pages = {}
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.start_section("Chat history", level=0)
    pdf.multi_cell(0, 8, "Chat history", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("U", "", 10)
    pdf.multi_cell(0, 5.5, "Sessions between a user and an AI assistant, in date order. %d sessions, from %s to %s."
                   % (len(order), q["haystack_dates"][order[0]], q["haystack_dates"][order[-1]]),
                   new_x="LMARGIN", new_y="NEXT")
    for n, i in enumerate(order, 1):
        pdf.add_page()
        first = pdf.page_no()
        title = "Session %d - %s" % (n, q["haystack_dates"][i])
        pdf.start_section(title, level=1)
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for turn in q["haystack_sessions"][i]:
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.multi_cell(0, 5, "User:" if turn["role"] == "user" else "Assistant:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("U", "", 9.5)
            pdf.multi_cell(0, 4.8, _clean(turn["content"]) or " ", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        pages[q["haystack_session_ids"][i]] = (first, pdf.page_no())
    pdf.output(path)
    return pages


def longmemeval_items(per_type=5):
    data = json.load(open(LME))
    rng = random.Random(SEED)
    groups = {}
    for q in data:
        t = "abstention" if q["question_id"].endswith("_abs") else q["question_type"]
        groups.setdefault(t, []).append(q)
    os.makedirs(LME_PDFS, exist_ok=True)
    items = []
    for t in ("single-session-user", "multi-session", "knowledge-update", "temporal-reasoning", "abstention"):
        for q in sorted(rng.sample(groups[t], per_type), key=lambda x: x["question_id"]):
            path = os.path.join(LME_PDFS, "%s.pdf" % q["question_id"])
            pages = render_history(q, path)
            gold_pages = sorted({p for sid in q["answer_session_ids"] if sid in pages
                                 for p in range(pages[sid][0], pages[sid][1] + 1)})
            items.append({"id": "lme-%s" % q["question_id"], "dataset": "longmemeval", "type": t, "pdf": path,
                          "question": "Current date: %s\nQuestion: %s" % (q["question_date"], q["question"]),
                          "gold": q["answer"], "lme_type": q["question_type"],
                          "abstention": q["question_id"].endswith("_abs"),
                          "raw_question": q["question"], "gold_pages": gold_pages,
                          "answer_sessions": {sid: pages[sid] for sid in q["answer_session_ids"] if sid in pages},
                          "n_sessions": len(pages)})
            print(items[-1]["id"], t, "sessions", len(pages), "pages", PdfReader(path).get_num_pages())
    return items


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    items = designqa_items() + longmemeval_items()
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print("wrote %d items to %s" % (len(items), out))
