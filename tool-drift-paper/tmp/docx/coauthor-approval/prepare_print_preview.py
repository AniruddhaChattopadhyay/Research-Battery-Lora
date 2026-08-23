from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "tmp"
    / "docx"
    / "coauthor-approval"
    / "render"
    / "NGEN-AI-2026-Paper-15-Coauthor-Approval.docx.qlpreview"
    / "Preview.html"
)
OUTPUT = SOURCE.with_name("PrintPreview.html")

PRINT_CSS = """
<style type="text/css">
@page { size: Letter portrait; margin: 0; }
html, body { margin: 0 !important; padding: 0 !important; }
.s1 { min-height: 648px !important; }
table { width: 100% !important; table-layout: fixed !important; }
</style>
"""


def main():
    html = SOURCE.read_text(encoding="utf-8")
    html = html.replace("</head>", PRINT_CSS + "</head>", 1)
    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
