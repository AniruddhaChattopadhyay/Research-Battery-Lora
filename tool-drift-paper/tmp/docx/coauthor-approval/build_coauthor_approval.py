from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "docx" / "NGEN-AI-2026-Paper-15-Coauthor-Approval.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def run(text, *, bold=False, italic=False, size=22, color="000000", font="Arial"):
    props = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>',
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
        f'<w:color w:val="{color}"/>',
    ]
    if bold:
        props.append("<w:b/><w:bCs/>")
    if italic:
        props.append("<w:i/><w:iCs/>")
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(
    content="",
    *,
    style=None,
    align=None,
    before=0,
    after=120,
    line=264,
    keep_next=False,
    keep_lines=False,
):
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    props.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>')
    if keep_next:
        props.append("<w:keepNext/>")
    if keep_lines:
        props.append("<w:keepLines/>")
    body = content if content.startswith("<w:r") else run(content)
    return f'<w:p><w:pPr>{"".join(props)}</w:pPr>{body}</w:p>'


def cell(paragraphs, width, *, fill=None, grid_span=None, valign="center"):
    tc_props = [f'<w:tcW w:w="{width}" w:type="dxa"/>']
    if grid_span:
        tc_props.append(f'<w:gridSpan w:val="{grid_span}"/>')
    if fill:
        tc_props.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>')
    tc_props.extend(
        [
            f'<w:vAlign w:val="{valign}"/>',
            '<w:tcMar><w:top w:w="100" w:type="dxa"/><w:start w:w="120" w:type="dxa"/>'
            '<w:bottom w:w="100" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tcMar>',
        ]
    )
    return f'<w:tc><w:tcPr>{"".join(tc_props)}</w:tcPr>{"".join(paragraphs)}</w:tc>'


def table(rows, widths, *, border_color="DADCE0", indent=120):
    total = sum(widths)
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    tbl_pr = (
        f'<w:tblW w:w="{total}" w:type="dxa"/><w:tblInd w:w="{indent}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="100" w:type="dxa"/>'
        '<w:start w:w="120" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/>'
        '<w:end w:w="120" w:type="dxa"/></w:tblCellMar>'
        f'<w:tblBorders>{borders}</w:tblBorders>'
    )
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    body = "".join(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(row)}</w:tr>' for row in rows)
    return f'<w:tbl><w:tblPr>{tbl_pr}</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>'


def label_value_row(label, value, *, mono=False):
    label_p = paragraph(run(label, bold=True, size=20, color="1F3A5F"), after=0)
    value_font = "Courier New" if mono else "Arial"
    value_size = 16 if mono else 20
    value_p = paragraph(run(value, size=value_size, font=value_font), after=0)
    return [cell([label_p], 2880, fill="F2F4F7"), cell([value_p], 6480)]


def signature_block(role, name, email):
    header = cell(
        [paragraph(run(role.upper(), bold=True, size=19, color="1F3A5F"), after=0)],
        9360,
        fill="E8EEF5",
        grid_span=2,
    )
    identity = cell(
        [
            paragraph(run("Printed name: ", bold=True, size=19) + run(name, size=19), after=20),
            paragraph(run("Email: ", bold=True, size=19) + run(email, size=19), after=0),
        ],
        9360,
        grid_span=2,
    )
    signature = cell(
        [
            paragraph(run("Signature (handwritten or electronic)", bold=True, size=18), after=200),
            paragraph(run(" ", size=18), after=0),
        ],
        6480,
        valign="top",
    )
    date = cell(
        [
            paragraph(run("Date", bold=True, size=18), after=200),
            paragraph(run(" ", size=18), after=0),
        ],
        2880,
        valign="top",
    )
    return table([[header], [identity], [signature, date]], [6480, 2880], border_color="9AA7B4")


def build_document_xml():
    body = []
    body.append(paragraph(run("NGEN-AI 2026 | PAPER 15", bold=True, size=18, color="667085"), after=80))
    body.append(paragraph(run("COAUTHOR APPROVAL AND AUTHORIZATION", bold=True, size=36, color="0B2545"), after=50, keep_next=True))
    body.append(paragraph(run("Camera-ready publication record", size=22, color="667085"), after=200))

    notice = paragraph(
        run("INTERNAL RECORD ONLY. ", bold=True, size=19, color="1F3A5F")
        + run("Keep the completed form with the authors' records. Do not include it in the NGEN-AI camera-ready ZIP.", size=19),
        after=0,
    )
    body.append(table([[cell([notice], 9360, fill="F4F6F9")]], [9360], border_color="DADCE0"))
    body.append(paragraph("", after=100))

    body.append(paragraph(run("Paper details", bold=True, size=26, color="2E74B5"), before=80, after=80, keep_next=True))
    details = [
        label_value_row("Conference", "International Conference on Next Generation AI Systems (NGEN-AI 2026)"),
        label_value_row("Paper", "15"),
        label_value_row("Title", "Robust Function Calling for LLM Agents Under Interface Drift"),
        label_value_row("Authors", "Aniruddha Chattopadhyay; Kanishka Halder; Partha Sarathi Roy"),
        label_value_row("Corresponding author", "Aniruddha Chattopadhyay (studyaniruddha@gmail.com)"),
        label_value_row("Final PDF SHA-256", "9146b562f022eabdc474fa27d825aa2c138feb4d2ac9ef8d225b535d91698d7f", mono=True),
    ]
    body.append(table(details, [2880, 6480]))

    body.append(paragraph(run("Approval", bold=True, size=26, color="2E74B5"), before=160, after=70, keep_next=True))
    body.append(paragraph("By signing below, I confirm the following:", after=80, keep_lines=True))
    declarations = [
        ("Manuscript review. ", "I reviewed the camera-ready paper identified above and approve its scientific content, reviewer-response revisions, author block, and submission."),
        ("Authorship details. ", "My name, author order, shared affiliation (Independent Researcher, Bangalore, India), and email address are correct."),
        ("Corresponding-author authorization. ", "I authorize Aniruddha Chattopadhyay to approve and submit the final paper and to sign the Springer License-to-Publish form on behalf of all authors."),
        ("Disclosures. ", "To the best of my knowledge, the statements that there are no competing interests and that no human participants, personal data, animals, or biological material are involved are accurate."),
        ("Publication. ", "I consent to submission and publication of the paper in the Springer conference proceedings, subject to the publisher's applicable terms."),
    ]
    for label, text in declarations:
        body.append(paragraph(run(label, bold=True, size=20) + run(text, size=20), after=55, keep_lines=True))

    body.append(paragraph(run("Signatures", bold=True, size=26, color="2E74B5"), before=100, after=70, keep_next=True))
    body.append(signature_block("Coauthor approval 1", "Kanishka Halder", "kanishkahalder@gmail.com"))
    body.append(paragraph("", after=80))
    body.append(signature_block("Coauthor approval 2", "Partha Sarathi Roy", "psarathi012@gmail.com"))

    body.append(
        paragraph(
            run("Return the completed form to studyaniruddha@gmail.com and retain it with the project archive.", italic=True, size=18, color="667085"),
            before=120,
            after=0,
        )
    )
    body.append(
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" '
        'w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="708"/><w:docGrid w:linePitch="360"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>{"".join(body)}</w:body></w:document>'
    )


CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

DOC_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>'''

STYLES = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
</w:styles>'''

SETTINGS = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/><w:compat/></w:settings>'''

CORE = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>NGEN-AI 2026 Paper 15 Coauthor Approval and Authorization</dc:title>
  <dc:subject>Internal coauthor approval record</dc:subject>
  <dc:creator>Aniruddha Chattopadhyay</dc:creator>
  <cp:keywords>NGEN-AI 2026; paper 15; coauthor approval</cp:keywords>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-08-08T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-08-08T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''

APP = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>
  <Company>Independent Researcher</Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>
</Properties>'''


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": ROOT_RELS,
        "word/document.xml": build_document_xml(),
        "word/_rels/document.xml.rels": DOC_RELS,
        "word/styles.xml": STYLES,
        "word/settings.xml": SETTINGS,
        "docProps/core.xml": CORE,
        "docProps/app.xml": APP,
    }
    with ZipFile(OUT, "w", ZIP_DEFLATED) as docx:
        for name, content in parts.items():
            docx.writestr(name, content.encode("utf-8"))
    print(OUT)


if __name__ == "__main__":
    main()
