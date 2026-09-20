---
title: "DOCX Document Creation, Styling, and Annotation Mastery"
description: "Comprehensive guide to creating, styling, editing, and annotating professional Microsoft Word (.docx) documents in Python using python-docx."
category: "document_engineering"
tags: [docx, python-docx, report, document_design, annotation, comments, word]
visibility: loadable
modifiable: true
---

# Professional DOCX Engineering with Python

This skill teaches how to design and generate executive-grade, publication-ready Word documents (`.docx`), as well as safely edit and annotate existing `.docx` files using Python.

---

## 🎨 Part 1: High-Level Document Design & Typographic Principles

A well-crafted Word document communicates authority, clarity, and visual hierarchy before a single word is read.

### 1. Typography & Hierarchy
* **Primary / Body Font**: Use clean, modern serifs or sans-serifs (e.g., `Calibri`, `Aptos`, `Arial`, `Segoe UI`, or `Georgia` / `Garamond` for formal reports).
* **Headings**: Complementary font with distinct weight and color (e.g., Navy `#1F4E79` or Slate `#2B3E50`).
* **Scale Guideline**:
  - Title: `24–28 pt`, Bold
  - Subtitle: `14–16 pt`, Regular or Italic, Muted Gray (`#595959`)
  - Heading 1: `18–20 pt`, Bold, Colored
  - Heading 2: `14–16 pt`, Semi-bold
  - Heading 3: `12–13 pt`, Medium / Bold
  - Body: `10.5–11.5 pt`, Line spacing `1.15–1.25`, Space after `4–6 pt`.

### 2. Palette & Theme Consistency
Always define a coherent 4-color palette for the document:
- **Primary Accent**: Main brand color (e.g., Deep Blue `#1B365D`) for H1s, table header fills, callout left-bars.
- **Secondary Accent**: Secondary color (e.g., Teal `#008080` or Steel Blue `#4A90E2`) for H2s, table highlights.
- **Neutral Dark**: Main text body (e.g., Charcoal `#262626`, never pure `#000000` for eye comfort).
- **Neutral Light**: Backgrounds for callout boxes and alternating table rows (e.g., Warm White / Soft Gray `#F2F4F7`).

### 3. Structural Archetypes
1. **Executive Cover Page**: Minimalist or Accent Header Band layout with Title, Subtitle, Author, Organization, Version, and Date.
2. **Callout / Note Boxes**: 1×1 shaded tables with a thick colored left border for warnings, takeaways, or executive summaries.
3. **Publication-Quality Tables**:
   - Header row with primary fill and bold white text.
   - Alternating row shading (`zebra striping`) with `#F9FAFB`.
   - Right-aligned numbers, left-aligned text, centered dates/status badges.
   - Explicit column widths and cell vertical centering.
4. **Header & Footer**:
   - Header: Right-aligned Document Title / Chapter in `8.5 pt` muted gray.
   - Footer: Left-aligned Confidentiality/Copyright, Right-aligned dynamic Page Numbering.

---

## 💻 Part 2: Building Documents from Scratch in Python

### Required Dependency
```bash
pip install python-docx
```

### Complete End-to-End Generator Template

```python
import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color: str):
    """Sets background fill color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color.replace("#", "")}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding (in twips) for a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_left_border_only(cell, hex_color: str, size: int = 24):
    """Sets a thick colored left border and removes top/bottom/right borders (for Callout Boxes)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="{size}" w:space="0" w:color="{hex_color.replace("#", "")}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)

def add_page_number(run):
    """Inserts a dynamic Word PAGE field."""
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    run._r.append(fldSimple)

def create_styled_document(output_path: str = "report.docx"):
    doc = Document()

    # 1. Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Enable different first page for clean cover page
        section.different_first_page_header_footer = True
        
        # Header / Footer setup
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        f_run = f_p.add_run("Page ")
        f_run.font.size = Pt(9)
        f_run.font.color.rgb = RGBColor(128, 128, 128)
        add_page_number(f_run)

    # 2. Base Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Segoe UI'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(6)

    # 3. Title / Cover Block
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(36)
    title_p.paragraph_format.space_after = Pt(8)
    t_run = title_p.add_run("Autonomous System Verification Report")
    t_run.font.size = Pt(24)
    t_run.font.bold = True
    t_run.font.color.rgb = RGBColor(31, 78, 121) # Deep Blue

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(28)
    s_run = sub_p.add_run("Architecture Audit, Benchmark Analysis & Quality Assurance")
    s_run.font.size = Pt(13)
    s_run.font.italic = True
    s_run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_page_break()

    # 4. Heading 1
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(6)
    h1_run = h1.add_run("1. Executive Summary & Key Findings")
    h1_run.font.size = Pt(16)
    h1_run.font.bold = True
    h1_run.font.color.rgb = RGBColor(31, 78, 121)

    # 5. Callout Box (1x1 Table with Left Border)
    callout = doc.add_table(rows=1, cols=1)
    callout.alignment = WD_TABLE_ALIGNMENT.CENTER
    callout.autofit = False
    callout.columns[0].width = Inches(6.5)

    c_cell = callout.cell(0, 0)
    set_cell_background(c_cell, "F0F4F8")
    set_cell_left_border_only(c_cell, "1F4E79", size=32)
    set_cell_margins(c_cell, top=140, bottom=140, left=200, right=150)

    cp = c_cell.paragraphs[0]
    cp.paragraph_format.space_after = Pt(0)
    c_bold = cp.add_run("Key Takeaway: ")
    c_bold.bold = True
    c_bold.font.color.rgb = RGBColor(31, 78, 121)
    cp.add_run("All 14 core test suites completed with 100% pass rate. System throughput increased by 28.4% under load.")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 6. Styled Data Table
    h2 = doc.add_paragraph()
    h2_run = h2.add_run("1.1 Benchmark Performance Metrics")
    h2_run.font.size = Pt(13)
    h2_run.font.bold = True
    h2_run.font.color.rgb = RGBColor(43, 62, 80)

    headers = ["Component", "Throughput (req/s)", "Latency p95 (ms)", "Error Rate (%)", "Status"]
    data = [
        ["API Gateway", "4,250", "12.4", "0.00%", "Optimal"],
        ["Inference Pipeline", "1,820", "45.1", "0.02%", "Optimal"],
        ["Vector Search", "980", "18.7", "0.00%", "Optimal"],
        ["Background Sync", "540", "110.2", "0.05%", "Within Target"],
    ]

    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(1.8), Inches(1.4), Inches(1.3), Inches(1.1), Inches(0.9)]

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(title)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9.5)

    # Data Rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].width = col_widths[c_idx]
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_idx in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if c_idx == 4:
                r.bold = True
                r.font.color.rgb = RGBColor(0, 128, 0)

    doc.save(output_path)
    print(f"✅ Generated styled document at {output_path}")

if __name__ == "__main__":
    create_styled_document()
```

---

## 🔍 Part 3: Editing, Replacing & Annotating Existing DOCX Files

When modifying an existing document, preserve exact paragraph styles, heading structures, and table formatting without rewriting from scratch.

### 1. Safe Search and Replace in Paragraphs
Word stores paragraph text broken into multiple `Runs`. A single search string might span across consecutive runs.

```python
from docx import Document

def safe_replace_text_in_doc(doc_path: str, search_text: str, replacement_text: str, output_path: str = None):
    doc = Document(doc_path)
    replaced_count = 0

    for p in doc.paragraphs:
        if search_text in p.text:
            # Simple single-run replace if contained in one run
            for r in p.runs:
                if search_text in r.text:
                    r.text = r.text.replace(search_text, replacement_text)
                    replaced_count += 1
            # Multi-run span fallback
            if search_text in p.text:
                full_text = p.text.replace(search_text, replacement_text)
                p.text = full_text
                replaced_count += 1

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if search_text in p.text:
                        p.text = p.text.replace(search_text, replacement_text)
                        replaced_count += 1

    save_path = output_path or doc_path
    doc.save(save_path)
    print(f"✅ Replaced {replaced_count} occurrences of '{search_text}' in {save_path}")
```

### 2. Highlighting Text in Existing Documents
```python
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

def highlight_search_matches(doc_path: str, search_keyword: str, highlight_color=WD_COLOR_INDEX.YELLOW):
    doc = Document(doc_path)
    count = 0

    for p in doc.paragraphs:
        if search_keyword.lower() in p.text.lower():
            for r in p.runs:
                if search_keyword.lower() in r.text.lower():
                    r.font.highlight_color = highlight_color
                    count += 1

    doc.save(doc_path)
    print(f"✅ Highlighted {count} occurrences in {doc_path}")
```

---

## ⚡ Part 4: Native Tool Integration in lollms-code

If document editing tools are active, you can call them directly rather than writing custom python scripts:
- `tool_inspect_document(file_name="report.docx")` — Inspect paragraphs, headings, and tables.
- `tool_read_document_content(file_name="report.docx", max_chars=8000)` — Read sections.
- `tool_edit_document_text(file_name="report.docx", operation="update", search_text="old", replacement_text="new")` — Surgical edits.
- `tool_annotate_document(file_name="report.docx", annotation_type="comment", search_text="anchor", comment="Review needed")` — Comments & highlights.