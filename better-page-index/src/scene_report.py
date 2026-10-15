"""Render CLEVRER runs as scene reports in one PDF (a "plant log" with one chapter per run).

The report holds raw facts only: objects, an event log (entrances, exits, collisions) and the
state of every object in every frame. It holds no causal links. The layout is the best case
for PageIndex: real headings, numbered sections and PDF bookmarks.
"""
import json
import os

from fpdf import FPDF

import clevrer

WINDOW = 16  # frames in each state-table subsection


def _fmt(v):
    return "%.2f" % v


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, "Page %d" % self.page_no(), align="C")


def _heading(pdf, text, level):
    sizes = {0: 15, 1: 12.5, 2: 10.5}
    pdf.start_section(text, level=level)
    pdf.set_font("Helvetica", "B", sizes[level])
    pdf.ln(2 if level else 0)
    pdf.multi_cell(0, 7 if level == 0 else 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _lines(pdf, lines):
    pdf.set_font("Helvetica", "", 9.5)
    for line in lines:
        pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _table(pdf, header, rows, widths, size=8):
    pdf.set_font("Helvetica", "", size)
    with pdf.table(col_widths=widths, line_height=size * 0.48, first_row_as_headings=True,
                   borders_layout="MINIMAL", text_align="LEFT") as t:
        r = t.row()
        for h in header:
            r.cell(h)
        for row in rows:
            r = t.row()
            for v in row:
                r.cell(str(v))
    pdf.ln(2)


def render_run(pdf, run):
    sid = run.sid
    pdf.add_page()
    _heading(pdf, "Run %d: scene report" % sid, 0)

    _heading(pdf, "%d.1 Overview" % sid, 1)
    _lines(pdf, [
        "- Video file: video_%05d.mp4. Frames 0 to %d (about 5 seconds)." % (sid, clevrer.LAST_FRAME),
        "- Objects in the scene: %d. Collisions recorded in the video: %d." % (len(run.objects), len(run.collisions)),
        "- Positions (x, y) are on the floor plane, in scene units. Velocity (vx, vy) and speed are in scene units per second.",
        "- State is 'moving' when the speed is 0.02 or more, else 'stationary'. 'In view' means inside the camera view.",
    ])

    _heading(pdf, "%d.2 Objects" % sid, 1)
    _table(pdf, ["Object", "Color", "Material", "Shape"],
           [[oid, o["color"], o["material"], o["shape"]] for oid, o in sorted(run.objects.items())],
           (1, 2, 2, 2), size=9)

    _heading(pdf, "%d.3 Event log" % sid, 1)
    rows = [[0, "In view at the start: %s." % (", ".join("the " + run.name(o) for o in sorted(run.in_view_at_start)) or "no object")]]
    for e in run.events():
        names = [run.name(o) for o in e["objects"]]
        if e["kind"] == "enter":
            text = "The %s enters the view." % names[0]
        elif e["kind"] == "exit":
            text = "The %s exits the view." % names[0]
        else:
            text = "Collision between the %s and the %s." % (names[0], names[1])
        rows.append([e["frame"], text])
    _table(pdf, ["Frame", "Event"], rows, (1, 9), size=9)

    _heading(pdf, "%d.4 Object states" % sid, 1)
    for start in range(0, clevrer.LAST_FRAME + 1, WINDOW):
        end = min(start + WINDOW - 1, clevrer.LAST_FRAME)
        _heading(pdf, "%d.4.%d Frames %d to %d" % (sid, start // WINDOW + 1, start, end), 2)
        rows = []
        for f in range(start, end + 1):
            for oid, s in sorted(run.states[f].items()):
                rows.append([f, run.name(oid), _fmt(s["x"]), _fmt(s["y"]), _fmt(s["vx"]), _fmt(s["vy"]),
                             _fmt(s["speed"]), "moving" if s["moving"] else "stationary",
                             "yes" if s["in_view"] else "no"])
        _table(pdf, ["Frame", "Object", "x", "y", "vx", "vy", "Speed", "State", "In view"], rows,
               (1.1, 3.6, 1.1, 1.1, 1.1, 1.1, 1.1, 1.6, 1.1))


def render_corpus(sids, path):
    """Write one PDF with one chapter per run. Return the page range of each run."""
    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    _heading(pdf, "Plant log: %d recorded runs" % len(sids), 0)
    _lines(pdf, [
        "This log holds the scene reports of %d recorded runs. Each run is one chapter." % len(sids),
        "Each chapter has an overview, the objects, an event log and the state of every object in every frame.",
        "Runs in this log: " + ", ".join(str(s) for s in sids) + ".",
    ])
    pages = {}
    for sid in sids:
        first = pdf.page_no() + 1
        render_run(pdf, clevrer.Run(sid))
        pages[sid] = (first, pdf.page_no())
    pdf.output(path)
    return pages


if __name__ == "__main__":
    import sys
    out, sids = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    pages = render_corpus(sids, out)
    json.dump(pages, open(os.path.splitext(out)[0] + "_pages.json", "w"))
    print("wrote", out, "pages:", pages)
