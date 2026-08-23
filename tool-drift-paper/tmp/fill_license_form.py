from __future__ import annotations

from copy import deepcopy
import tempfile
import zipfile
import xml.etree.ElementTree as etree
from pathlib import Path

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def visible_text(element: etree.Element) -> str:
    return "".join(node.text or "" for node in element.iter(qn("t")))


def fill_control(control: etree.Element, value: str) -> None:
    properties = control.find(qn("sdtPr"))
    if properties is not None:
        for placeholder_flag in properties.findall(qn("showingPlcHdr")):
            properties.remove(placeholder_flag)

    content = control.find(qn("sdtContent"))
    if content is None:
        raise ValueError("Content control has no w:sdtContent")

    # The official template highlights placeholder runs in light gray. Remove
    # that actual text highlighting when replacing a placeholder with final text
    # so it cannot appear in the printed form.
    for run_properties in content.iter(qn("rPr")):
        for highlight in list(run_properties.findall(qn("highlight"))):
            run_properties.remove(highlight)

    text_nodes = list(content.iter(qn("t")))
    if not text_nodes:
        raise ValueError("Content control has no text node to replace")

    text = text_nodes[0]
    text.text = value
    if value.startswith(" ") or value.endswith(" "):
        text.set(XML_SPACE, "preserve")
    else:
        text.attrib.pop(XML_SPACE, None)

    for extra_text in text_nodes[1:]:
        extra_text.text = ""
        extra_text.attrib.pop(XML_SPACE, None)


def direct_children(element: etree.Element, local_name: str) -> list[etree.Element]:
    return [child for child in element if child.tag == qn(local_name)]


def replace_cell_text(cell: etree.Element, value: str) -> None:
    paragraphs = direct_children(cell, "p")
    if not paragraphs:
        raise ValueError("Expected a paragraph in form cell")

    first = paragraphs[0]
    template_properties = None
    for run in first.iter(qn("r")):
        run_properties = run.find(qn("rPr"))
        if run_properties is not None:
            template_properties = deepcopy(run_properties)
            break

    for paragraph in paragraphs:
        paragraph_properties = paragraph.find(qn("pPr"))
        paragraph_run_properties = (
            paragraph_properties.find(qn("rPr"))
            if paragraph_properties is not None
            else None
        )
        if paragraph_run_properties is not None:
            for color in list(paragraph_run_properties.findall(qn("color"))):
                paragraph_run_properties.remove(color)
        for child in list(paragraph):
            if child.tag != qn("pPr"):
                paragraph.remove(child)

    if value:
        run = etree.SubElement(first, qn("r"))
        if template_properties is not None:
            run.append(template_properties)
        else:
            template_properties = etree.SubElement(run, qn("rPr"))
        for color in list(template_properties.findall(qn("color"))):
            template_properties.remove(color)
        color = etree.SubElement(template_properties, qn("color"))
        color.set(qn("val"), "000000")
        text = etree.SubElement(run, qn("t"))
        text.text = value


def fill_signature_area(root: etree.Element, values: dict[str, str]) -> None:
    tables = list(root.iter(qn("tbl")))
    signature_table = next(
        (table for table in tables if "Signed for and on behalf" in visible_text(table)),
        None,
    )
    if signature_table is None:
        raise ValueError("Signature table not found")

    signature_rows = direct_children(signature_table, "tr")
    if len(signature_rows) < 2:
        raise ValueError("Signature table has an unexpected row structure")
    signature_cells = direct_children(signature_rows[1], "tc")
    if len(signature_cells) != 5:
        raise ValueError("Signature table has an unexpected column structure")

    # Keep the actual signature and signing date empty for handwriting.
    replace_cell_text(signature_cells[0], "")
    replace_cell_text(signature_cells[2], values["print_name"])
    replace_cell_text(signature_cells[4], "")

    for text in signature_table.iter(qn("t")):
        if text.text == "[Ha":
            text.text = ""

    contact_table = next(
        (
            table
            for table in tables
            if "Address:" in visible_text(table) and "Email:" in visible_text(table)
        ),
        None,
    )
    if contact_table is None:
        raise ValueError("Address/email table not found")

    contact_rows = direct_children(contact_table, "tr")
    if len(contact_rows) < 2:
        raise ValueError("Address/email table has an unexpected row structure")
    address_cells = direct_children(contact_rows[0], "tc")
    email_cells = direct_children(contact_rows[1], "tc")
    if len(address_cells) != 3 or len(email_cells) != 3:
        raise ValueError("Address/email table has an unexpected column structure")
    replace_cell_text(address_cells[2], values["address"])
    replace_cell_text(email_cells[2], values["email"])


def register_namespaces(document_path: Path) -> None:
    for _, namespace in etree.iterparse(document_path, events=("start-ns",)):
        prefix, uri = namespace
        if prefix != "xml":
            etree.register_namespace(prefix or "", uri)


def main() -> None:
    source = Path("ngenai/admin/License-to-Publish-Form.docx")
    destination = Path("ngenai/admin/License-to-Publish-Form-prefilled.docx")
    values = {
        "contribution_title": "Robust Function Calling for LLM Agents Under Interface Drift",
        "authors": "Aniruddha Chattopadhyay, Kanishka Halder, Partha Sarathi Roy",
        "corresponding": "Aniruddha Chattopadhyay",
        "print_name": "Aniruddha Chattopadhyay",
        "address": "Independent Researcher, Bangalore, India",
        "email": "studyaniruddha@gmail.com",
    }

    with tempfile.TemporaryDirectory(prefix="ngenai_license_") as temp_dir:
        unpacked = Path(temp_dir) / "unzipped"
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(unpacked)

        document_path = unpacked / "word" / "document.xml"
        register_namespaces(document_path)
        tree = etree.parse(document_path)
        root = tree.getroot()

        filled = set()
        for control in root.iter(qn("sdt")):
            properties = control.find(qn("sdtPr"))
            tag_element = properties.find(qn("tag")) if properties is not None else None
            tag = tag_element.get(qn("val"), "") if tag_element is not None else ""
            current = visible_text(control)
            if tag == "Title" and current == "Click here to enter text.":
                fill_control(control, values["contribution_title"])
                filled.add("contribution_title")
            elif tag == "Author(s)":
                fill_control(control, values["authors"])
                filled.add("authors")
            elif tag == "Corresponding Author":
                fill_control(control, values["corresponding"])
                filled.add("corresponding")

        fill_signature_area(root, values)
        filled.update({"print_name", "address", "email"})

        missing = set(values) - filled
        if missing:
            raise ValueError(f"Expected controls not found: {sorted(missing)}")

        tree.write(document_path, xml_declaration=True, encoding="UTF-8")

        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(unpacked.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(unpacked).as_posix())

    print(destination)


if __name__ == "__main__":
    main()
