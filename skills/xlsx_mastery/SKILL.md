---
title: "XLSX Spreadsheet Modeling, Financial Dashboards, and openpyxl Mastery"
description: "Comprehensive guide to designing, creating, calculating, formatting, and charting professional Excel spreadsheets (.xlsx) in Python using openpyxl."
category: "spreadsheet_engineering"
tags: [xlsx, openpyxl, excel, financial_modeling, spreadsheet, charts, formulas]
visibility: loadable
modifiable: true
---

# Professional Spreadsheet & Financial Model Engineering with Python

This skill teaches how to build clean, formula-driven, auditable Excel workbooks (`.xlsx`) using `openpyxl`, with corporate design standards, dynamic formulas, conditional formatting, and charts.

---

## 🎨 Part 1: High-Level Spreadsheet Design & Financial Standards

A production spreadsheet is a software artifact. It must be clean, readable, self-documenting, and free of hardcoded formula inputs.

### 1. Financial Color-Coding Conventions (FAST Standard)
* **Blue text on Soft Gray / White background** (`#0000FF` on `#F2F4F8`): **User Input / Hardcoded Assumptions**.
* **Black text on White** (`#000000`): **Formulas & Calculations**.
* **Green text** (`#008000`): **External Workbook / Database Link**.
* **Dark Header Fills** (Navy `#1F4E79`, Charcoal `#262626`): **Table headers with White Bold Text**.
* **Total / Summary Rows**: Double bottom border (`accounting underline`) and top thin border.

### 2. Number Formatting Matrix
Never leave raw unformatted numbers. Always assign explicit number formats:
- **Currency**: `"$#,##0"` (Standard) or `"$#,##0.00"` (Cents required)
- **Accounting (Zero as Dash)**: `_($* #,##0.00_);_($* (#,##0.00);_($* "-"??_);_(@_)`
- **Percentage**: `"0.0%"` or `"0.00%"`
- **Integer Quantities**: `"#,##0"`
- **Dates**: `"YYYY-MM-DD"` or `"DD-MMM-YYYY"`

### 3. Layout Architecture
- **Tab 1: `Summary / Dashboard`**: High-level KPI summary cards, KPI trends, dynamic executive charts.
- **Tab 2: `Assumptions`**: Input variables (growth rates, cost factors, tax percentages) with named ranges.
- **Tab 3: `Model / Calculations`**: Formulaic time-series projection (Monthly / Quarterly / Annual).
- **Freeze Panes**: Always freeze top header rows and left identification columns so data stays anchored on scroll.
- **Gridlines**: Always ensure `ws.views.sheetView[0].showGridLines = True` is set explicitly.

---

## 💻 Part 2: Building Professional Workbooks with openpyxl

### Required Dependency
```bash
pip install openpyxl
```

### Complete Dashboard & Financial Model Template

```python
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference, Series

def create_financial_model(output_path: str = "financial_model.xlsx"):
    wb = openpyxl.Workbook()
    
    # ── 1. Tab 1: Executive Summary & Dashboard ──
    ws = wb.active
    ws.title = "Financial Summary"
    ws.views.sheetView[0].showGridLines = True

    # Palette Constants
    NAVY_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    ACCENT_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    ZEBRA_FILL = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")
    
    WHITE_BOLD = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    TITLE_FONT = Font(name="Calibri", size=16, bold=True, color="1F4E79")
    SECTION_FONT = Font(name="Calibri", size=12, bold=True, color="1F4E79")
    BOLD_FONT = Font(name="Calibri", size=10.5, bold=True)
    REGULAR_FONT = Font(name="Calibri", size=10.5)
    
    THIN_BORDER_SIDE = Side(style='thin', color='D3D3D3')
    THIN_BORDER = Border(left=THIN_BORDER_SIDE, right=THIN_BORDER_SIDE, top=THIN_BORDER_SIDE, bottom=THIN_BORDER_SIDE)
    DOUBLE_BOTTOM = Border(top=Side(style='thin', color='000000'), bottom=Side(style='double', color='000000'))

    # Title Block
    ws["B2"] = "Quarterly Revenue & Margin Projections"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = "Forecast Model — Fiscal Year 2026"
    ws["B3"].font = Font(name="Calibri", size=11, italic=True, color="595959")

    # KPI Card 1: Total Revenue
    ws["B5"] = "Projected FY26 Revenue"
    ws["B5"].font = Font(size=9, color="595959")
    ws["B6"] = "=SUM(C11:F11)"
    ws["B6"].font = Font(size=16, bold=True, color="1F4E79")
    ws["B6"].number_format = "$#,##0"
    ws["B5"].fill = ACCENT_FILL
    ws["B6"].fill = ACCENT_FILL

    # KPI Card 2: Net Margin Average
    ws["D5"] = "Average Net Margin"
    ws["D5"].font = Font(size=9, color="595959")
    ws["D6"] = "=AVERAGE(C14:F14)"
    ws["D6"].font = Font(size=16, bold=True, color="1F4E79")
    ws["D6"].number_format = "0.0%"
    ws["D5"].fill = ACCENT_FILL
    ws["D6"].fill = ACCENT_FILL

    # Data Table Headers
    headers = ["Financial Metric", "Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026", "FY2026 Total"]
    start_row = 10
    
    for col_idx, h in enumerate(headers, start=2):
        cell = ws.cell(row=start_row, column=col_idx, value=h)
        cell.font = WHITE_BOLD
        cell.fill = NAVY_FILL
        cell.alignment = Alignment(horizontal="left" if col_idx == 2 else "right", vertical="center")

    # Financial Rows Setup
    row_specs = [
        ("Revenue", [1200000, 1350000, 1500000, 1800000], "$#,##0", False),
        ("Cost of Goods Sold (COGS)", [480000, 520000, 580000, 690000], "$#,##0", False),
        ("Gross Profit", ["=C11-C12", "=D11-D12", "=E11-E12", "=F11-F12"], "$#,##0", True),
        ("Gross Margin %", ["=C13/C11", "=D13/D11", "=E13/E11", "=F13/F11"], "0.0%", False),
        ("Operating Expenses (OPEX)", [320000, 340000, 370000, 410000], "$#,##0", False),
        ("Operating Income (EBIT)", ["=C13-C15", "=D13-D15", "=E13-E15", "=F13-F15"], "$#,##0", True),
        ("Net Income", ["=C16*0.75", "=D16*0.75", "=E16*0.75", "=F16*0.75"], "$#,##0", True),
    ]

    for idx, (label, values, num_fmt, is_total) in enumerate(row_specs, start=11):
        # Label
        lbl_cell = ws.cell(row=idx, column=2, value=label)
        lbl_cell.font = BOLD_FONT if is_total else REGULAR_FONT
        lbl_cell.border = DOUBLE_BOTTOM if is_total and "Net Income" in label else THIN_BORDER
        if idx % 2 == 1 and not is_total:
            lbl_cell.fill = ZEBRA_FILL

        # Quarter values
        for q_idx, val in enumerate(values, start=3):
            cell = ws.cell(row=idx, column=q_idx, value=val)
            cell.font = BOLD_FONT if is_total else REGULAR_FONT
            cell.number_format = num_fmt
            cell.alignment = Alignment(horizontal="right", vertical="center")
            cell.border = DOUBLE_BOTTOM if is_total and "Net Income" in label else THIN_BORDER
            if idx % 2 == 1 and not is_total:
                cell.fill = ZEBRA_FILL

        # FY Total Column (SUM or Average depending on percentage)
        tot_cell = ws.cell(row=idx, column=7)
        if "%" in num_fmt:
            tot_cell.value = f"=AVERAGE(C{idx}:F{idx})"
        else:
            tot_cell.value = f"=SUM(C{idx}:F{idx})"
        tot_cell.font = BOLD_FONT
        tot_cell.number_format = num_fmt
        tot_cell.alignment = Alignment(horizontal="right", vertical="center")
        tot_cell.border = DOUBLE_BOTTOM if is_total and "Net Income" in label else THIN_BORDER

    # Freeze panes below headers
    ws.freeze_panes = "C11"

    # ── 2. Add Native Bar Chart ──
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "Quarterly Revenue vs. Gross Profit"
    chart.y_axis.title = "USD ($)"
    chart.x_axis.title = "Quarter"
    chart.width = 15
    chart.height = 8

    data_ref = Reference(ws, min_col=3, min_row=11, max_col=6, max_row=13)
    cats_ref = Reference(ws, min_col=3, min_row=10, max_col=6, max_row=10)
    
    chart.add_data(data_ref, titles_from_data=False)
    chart.set_categories(cats_ref)
    chart.legend.position = "b"

    ws.add_chart(chart, "B20")

    # ── 3. Auto-fit column widths ──
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            if cell.number_format and "$" in cell.number_format:
                val_str += "   $"
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)
    ws.column_dimensions["B"].width = 30

    wb.save(output_path)
    print(f"✅ Generated financial workbook at {output_path}")

if __name__ == "__main__":
    create_financial_model()
```

---

## 🔍 Part 3: Reading, Updating & Annotating Existing Workbooks

### 1. Safe Updating Cells without Breaking Formulas
```python
import openpyxl

def update_spreadsheet_values(xlsx_path: str, updates: dict):
    """
    Updates specific cells in a spreadsheet while keeping formatting and formulas intact.
    updates example: {'Sheet1': {'C11': 1400000, 'D11': 1550000}}
    """
    wb = openpyxl.load_workbook(xlsx_path)

    for sheet_name, cell_dict in updates.items():
        if sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for coord, new_val in cell_dict.items():
                ws[coord].value = new_val
                print(f"Updated {sheet_name}!{coord} -> {new_val}")

    wb.save(xlsx_path)
    print(f"✅ Saved updates to {xlsx_path}")
```

### 2. Adding Cell Comments / Annotations
```python
import openpyxl
from openpyxl.comments import Comment

def add_cell_annotation(xlsx_path: str, sheet_name: str, cell_coord: str, comment_text: str, author: str = "Auditor"):
    wb = openpyxl.load_workbook(xlsx_path)
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        comment = Comment(comment_text, author)
        comment.width = 300
        comment.height = 100
        ws[cell_coord].comment = comment
        wb.save(xlsx_path)
        print(f"✅ Added comment to {sheet_name}!{cell_coord}")
```

---

## ⚡ Part 4: Native Tool Integration in lollms-code

When working in `lollms_code`, you can also use:
- `tool_inspect_document(file_name="model.xlsx")` — Inspect sheet names and row counts.
- `tool_read_document_content(file_name="model.xlsx", page_or_sheet="Summary", max_chars=8000)` — Read tabular data.
- `tool_execute_python_code` / `tool_execute_python_file` — Run `openpyxl` / `pandas` scripts to transform and calculate models.