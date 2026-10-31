"""
as_is_document_tools.py
=======================
Tools for inspecting, reading, and selectively searching native document formats
(.pdf, .docx, .xlsx, .pptx, .txt, .csv, .json) in the workspace without prompt context bloat.
"""

import os
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from ascii_colors import ASCIIColors

os.environ["PYMUPDF_MESSAGE"] = "fd:2"
TOOL_LIBRARY_NAME = "As-Is Document Tools"
TOOL_LIBRARY_DESC = "Inspect, search, and extract selective content from binary and structured workspace documents."
TOOL_LIBRARY_ICON = "📄"

_VISION_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def init_tools_library() -> None:
    """Ensure optional parsing dependencies are available."""
    try:
        import pipmaster as pm
        pm.ensure_packages({
            "pypdf": ">=3.0.0",
            "python-docx": ">=0.8.11",
            "openpyxl": ">=3.0.0",
            "python-pptx": ">=0.6.21"
        })
    except Exception:
        pass


def _extract_pdf_images_for_vlm(pdf_path: Path, target_pages: List[int], max_images: int = 5) -> tuple[List[str], List[str]]:
    """
    Extracts embedded raster images from specified PDF pages as base64-encoded PNG strings.
    Uses PyMuPDF (fitz) if available; returns (images, media_types).
    """
    images_b64: List[str] = []
    media_types: List[str] = []

    try:
        import fitz
    except ImportError:
        return images_b64, media_types

    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in target_pages:
            if len(images_b64) >= max_images:
                break
            if page_num >= len(doc):
                continue
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_info in image_list:
                if len(images_b64) >= max_images:
                    break
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img.get("image")
                    if not img_bytes:
                        continue
                    img_ext = base_img.get("ext", "png").lower()
                    if img_ext not in _VISION_IMAGE_EXTS:
                        img_ext = "png"
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    images_b64.append(b64_str)
                    media_types.append(f"image/{img_ext}")
                except Exception:
                    continue
    except Exception:
        pass
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    return images_b64, media_types


def tool_inspect_document(file_name: str) -> Dict[str, Any]:
    """
    Inspects a document file to return structural metadata (format, page/sheet count, outline/headings).

    Args:
        file_name (str): Relative path to the document file in the workspace.
    """
    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    ext = path.suffix.lower()
    file_size = path.stat().st_size

    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                return {"success": False, "error": "pypdf is not installed. Please install it via 'pip install pypdf' to inspect PDF files."}
            try:
                reader = PdfReader(str(path))
                return {
                    "success": True,
                    "file_name": file_name,
                    "format": "PDF",
                    "size_bytes": file_size,
                    "pages_count": len(reader.pages),
                    "output": f"PDF '{file_name}': {len(reader.pages)} page(s), {file_size:,} bytes."
                }
            except Exception as pdf_err:
                return {"success": False, "error": f"Failed to read PDF '{file_name}': {pdf_err}. The file may be corrupted, encrypted, or require a password."}

        elif ext == ".docx":
            import docx
            doc = docx.Document(str(path))
            headings = [p.text.strip() for p in doc.paragraphs if p.style.name.startswith("Heading") and p.text.strip()]
            return {
                "success": True,
                "file_name": file_name,
                "format": "DOCX",
                "size_bytes": file_size,
                "paragraphs_count": len(doc.paragraphs),
                "headings": headings[:15],
                "output": f"DOCX '{file_name}': {len(doc.paragraphs)} paragraph(s), {len(headings)} heading(s)."
            }

        elif ext in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(str(path), read_only=True)
            sheets = wb.sheetnames
            wb.close()
            return {
                "success": True,
                "file_name": file_name,
                "format": "Excel",
                "size_bytes": file_size,
                "sheets": sheets,
                "output": f"Excel '{file_name}': {len(sheets)} sheet(s) ({', '.join(sheets)})."
            }

        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(str(path))
            return {
                "success": True,
                "file_name": file_name,
                "format": "PPTX",
                "size_bytes": file_size,
                "slides_count": len(prs.slides),
                "output": f"PPTX '{file_name}': {len(prs.slides)} slide(s)."
            }

        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            return {
                "success": True,
                "file_name": file_name,
                "format": ext.upper().lstrip(".") or "Text",
                "size_bytes": file_size,
                "lines_count": len(lines),
                "output": f"Text file '{file_name}': {len(lines)} line(s), {file_size:,} bytes."
            }

    except Exception as e:
        import traceback as _tb
        return {"success": False, "error": f"Failed to inspect document '{file_name}': {e}", "traceback": _tb.format_exc()}


def tool_read_document_content(
    file_name: str,
    page_or_sheet: Optional[str] = None,
    max_chars: int = 8000
) -> Dict[str, Any]:
    """
    Extracts readable text content from a document (supports specific pages or sheets).

    Args:
        file_name (str): Relative path to the file in the workspace.
        page_or_sheet (str, optional): Specific page number (for PDF/PPTX) or sheet name (for Excel).
        max_chars (int, optional): Maximum characters to extract. Defaults to 8000.
    """
    if isinstance(max_chars, str):
        try:
            max_chars = int(max_chars)
        except ValueError:
            max_chars = 8000

    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                return {"success": False, "error": "pypdf is not installed. Please install it via 'pip install pypdf' to read PDF files."}
            try:
                reader = PdfReader(str(path))
                total_pages = len(reader.pages)
            except Exception as pdf_err:
                return {"success": False, "error": f"Failed to open PDF '{file_name}': {pdf_err}. The file may be corrupted, encrypted, or require a password."}

            target_pages = list(range(total_pages))
            if page_or_sheet:
                target_pages = []
                for part in str(page_or_sheet).split(','):
                    part = part.strip()
                    if '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            target_pages.extend(range(max(1, start) - 1, min(total_pages, end)))
                        except ValueError:
                            pass
                    elif part.isdigit():
                        p = int(part) - 1
                        if 0 <= p < total_pages:
                            target_pages.append(p)

            if not target_pages:
                return {"success": False, "error": f"No valid pages found in range '{page_or_sheet}'. Document has {total_pages} pages."}

            pages_text = []
            current_chars = 0
            for page_num in target_pages:
                if current_chars >= max_chars:
                    break
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                pages_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                current_chars += len(page_text)

            full_text = "\n\n".join(pages_text)
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + f"\n\n... [truncated, {len(full_text) - max_chars} more chars]"

            images_b64: List[str] = []
            media_types: List[str] = []
            has_no_text = not full_text.strip()

            if has_no_text or "[truncated" in full_text:
                images_b64, media_types = _extract_pdf_images_for_vlm(path, target_pages, max_images=5)

            if has_no_text and not images_b64:
                return {"success": False, "error": f"Pages {page_or_sheet or '1-' + str(total_pages)} extracted but contained no readable text or extractable images. The PDF may require OCR."}
            if has_no_text and images_b64:
                return {
                    "success": True,
                    "total_pages": total_pages,
                    "pages_read": [p + 1 for p in target_pages],
                    "content": f"[Pages {page_or_sheet} contain no extractable text. {len(images_b64)} embedded figure(s)/image(s) were extracted for visual analysis.]",
                    "output": f"[Pages {page_or_sheet} contain no extractable text. {len(images_b64)} embedded figure(s)/image(s) were extracted for visual analysis.]",
                    "images": images_b64,
                    "image_media_types": media_types,
                    "prompt_injection": f"\n\n[SYSTEM: {len(images_b64)} image(s) from page(s) {page_or_sheet} have been extracted and attached as vision content for your analysis. Describe what you see in the figures, including any equations, charts, or diagrams.]"
                }

            result = {"success": True, "total_pages": total_pages, "pages_read": [p + 1 for p in target_pages], "content": full_text, "output": full_text}
            if images_b64:
                result["images"] = images_b64
                result["image_media_types"] = media_types
                result["prompt_injection"] = f"\n\n[SYSTEM: {len(images_b64)} embedded figure(s)/image(s) from page(s) {page_or_sheet} have been extracted and attached as vision content. Reference them in your analysis of the text.]"
            return result

        elif ext == ".docx":
            import docx
            doc = docx.Document(str(path))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())[:max_chars]
            return {"success": True, "content": text, "output": text}

        elif ext in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            sheet_name = page_or_sheet if page_or_sheet in wb.sheetnames else wb.sheetnames[0]
            ws = wb[sheet_name]

            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
                if sum(len(r) for r in rows) >= max_chars:
                    break

            wb.close()
            content = f"### Sheet: {sheet_name}\n\n" + "\n".join(rows)
            return {"success": True, "sheet": sheet_name, "content": content[:max_chars], "output": content[:max_chars]}

        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(str(path))
            slides_text = []
            for i, slide in enumerate(prs.slides):
                parts = [f"## Slide {i + 1}"]
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for p in shape.text_frame.paragraphs:
                            if p.text.strip():
                                parts.append(p.text.strip())
                slides_text.append("\n".join(parts))
                if sum(len(s) for s in slides_text) >= max_chars:
                    break

            content = "\n\n---\n\n".join(slides_text)[:max_chars]
            return {"success": True, "slides_count": len(prs.slides), "content": content, "output": content}

        else:
            text = path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
            if not text.strip():
                return {"success": False, "error": f"File '{file_name}' exists but contains no readable text."}
            return {"success": True, "content": text, "output": text}

    except Exception as e:
        return {"success": False, "error": f"Failed to read content from '{file_name}': {e}"}


def _ensure_docx_comments_part(doc: Any) -> Any:
    """Return the document's comments part, creating it when absent (idempotent)."""
    for rel in doc.part.rels.values():
        if "comments" in rel.reltype:
            return rel.target_part

    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.opc.packuri import PackURI
    from docx.opc.part import Part
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    comments_part = Part(
        PackURI("/word/comments.xml"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
        parse_xml(f'<w:comments {nsdecls("w")}/>'),
        doc.part.package,
    )
    doc.part.relate_to(comments_part, RT.COMMENTS)
    return comments_part


def _insert_word_comment(
    doc: Any,
    paragraph: Any,
    text: str,
    author: str = "Assistant",
    run: Optional[Any] = None,
) -> str:
    """Insert a native OOXML comment anchored on `run` (or paragraph end). Returns the comment id."""
    from datetime import datetime, timezone
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    comments_part = _ensure_docx_comments_part(doc)
    comments_root = comments_part.element

    existing_ids = [
        int(c.get(qn("w:id")))
        for c in comments_root.findall(qn("w:comment"))
        if c.get(qn("w:id")) is not None and str(c.get(qn("w:id"))).lstrip("-").isdigit()
    ]
    comment_id = str(max(existing_ids) + 1 if existing_ids else 0)

    initials = "".join(ch for ch in author if ch.isalpha())[:4].upper() or "AI"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    comment_el = OxmlElement("w:comment")
    comment_el.set(qn("w:id"), comment_id)
    comment_el.set(qn("w:author"), author)
    comment_el.set(qn("w:initials"), initials)
    comment_el.set(qn("w:date"), date_str)

    p_el = OxmlElement("w:p")
    r_el = OxmlElement("w:r")
    t_el = OxmlElement("w:t")
    t_el.set(qn("xml:space"), "preserve")
    t_el.text = text
    r_el.append(t_el)
    p_el.append(r_el)
    comment_el.append(p_el)
    comments_root.append(comment_el)

    range_start = OxmlElement("w:commentRangeStart")
    range_start.set(qn("w:id"), comment_id)
    range_end = OxmlElement("w:commentRangeEnd")
    range_end.set(qn("w:id"), comment_id)

    ref_run = OxmlElement("w:r")
    ref_el = OxmlElement("w:commentReference")
    ref_el.set(qn("w:id"), comment_id)
    ref_run.append(ref_el)

    if run is not None:
        run._element.addprevious(range_start)
        run._element.addnext(range_end)
        range_end.addnext(ref_run)
    else:
        paragraph._p.append(range_start)
        paragraph._p.append(range_end)
        paragraph._p.append(ref_run)

    return comment_id


def _find_docx_run_by_text(doc: Any, search_text: str) -> Optional[Any]:
    """Return the first run whose text contains `search_text` (exact, then case-insensitive)."""
    if not search_text:
        return None
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if search_text in run.text:
                return run
    needle = search_text.lower()
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if needle in run.text.lower():
                return run
    return None


def _read_docx_comments(doc: Any) -> List[Dict[str, Any]]:
    """Extract all native comments from the document. Returns [] on any failure."""
    from docx.oxml.ns import qn

    try:
        comments_part = None
        for rel in doc.part.rels.values():
            if "comments" in rel.reltype:
                comments_part = rel.target_part
                break
        if comments_part is None:
            return []

        results: List[Dict[str, Any]] = []
        for c in comments_part.element.findall(qn("w:comment")):
            text_parts = [t.text or "" for t in c.iter(qn("w:t"))]
            results.append({
                "id": c.get(qn("w:id")),
                "author": c.get(qn("w:author")),
                "initials": c.get(qn("w:initials")),
                "date": c.get(qn("w:date")),
                "text": "".join(text_parts),
            })
        return results
    except Exception:
        return []


def tool_add_word_comment(
    file_name: str,
    comment: str,
    search_text: Optional[str] = None,
    author: str = "Assistant",
    paragraph_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Adds a native Word review comment (Review-pane margin box) to a .docx file.

    Args:
        file_name (str): Relative path to the .docx file in the workspace.
        comment (str): The comment text.
        search_text (str, optional): Text to anchor the comment on. Preferred anchor.
        author (str, optional): Comment author name. Defaults to "Assistant".
        paragraph_index (int, optional): 0-based index into non-empty paragraphs when search_text is absent.
    """
    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}
    if path.suffix.lower() not in (".docx", ".docm"):
        return {"success": False, "error": f"File '{file_name}' is not a Word document (.docx/.docm)."}

    try:
        import docx
        doc = docx.Document(str(path))

        target_run = _find_docx_run_by_text(doc, search_text) if search_text else None
        target_paragraph = None

        if target_run is not None:
            target_paragraph = target_run._parent
        else:
            non_empty = [p for p in doc.paragraphs if p.text.strip()]
            if not non_empty:
                return {"success": False, "error": f"Document '{file_name}' has no non-empty paragraphs to anchor a comment on."}
            if paragraph_index is not None:
                if not (0 <= paragraph_index < len(non_empty)):
                    return {"success": False, "error": f"paragraph_index {paragraph_index} out of range (0..{len(non_empty) - 1})."}
                target_paragraph = non_empty[paragraph_index]
            else:
                target_paragraph = non_empty[0]

        comment_id = _insert_word_comment(doc, target_paragraph, comment, author=author, run=target_run)
        doc.save(str(path))

        return {
            "success": True,
            "comment_id": comment_id,
            "author": author,
            "file_name": file_name,
            "workspace_file": str(path.resolve()),
            "output": f"Added native Word comment (id={comment_id}) to '{file_name}' by '{author}'.",
        }
    except Exception as e:
        import traceback as _tb
        return {"success": False, "error": f"Failed to add comment to '{file_name}': {e}", "traceback": _tb.format_exc()}


def tool_get_word_comments(file_name: str) -> Dict[str, Any]:
    """
    Lists all native Word review comments in a .docx file.

    Args:
        file_name (str): Relative path to the .docx file in the workspace.
    """
    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}
    if path.suffix.lower() not in (".docx", ".docm"):
        return {"success": False, "error": f"File '{file_name}' is not a Word document (.docx/.docm)."}

    try:
        import docx
        doc = docx.Document(str(path))
        comments = _read_docx_comments(doc)
        if not comments:
            output = f"No native comments found in '{file_name}'."
        else:
            lines = [f"Found {len(comments)} comment(s) in '{file_name}':"]
            for c in comments:
                lines.append(f"  [id={c['id']}] {c['author']} ({c['date']}): {c['text']}")
            output = "\n".join(lines)
        return {
            "success": True,
            "file_name": file_name,
            "count": len(comments),
            "comments": comments,
            "output": output,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to read comments from '{file_name}': {e}"}


def tool_grep_document(
    file_name: str,
    pattern: str,
    max_matches: int = 20
) -> Dict[str, Any]:
    """
    Searches for a regular expression or keyword within a document without loading the whole file into context.

    Args:
        file_name (str): Relative path to the file in the workspace.
        pattern (str): Search keyword or regex pattern.
        max_matches (int, optional): Maximum matching excerpts to return. Defaults to 20.
    """
    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    regex = re.compile(pattern, re.IGNORECASE)
    matches = []

    try:
        doc_result = tool_read_document_content(file_name, max_chars=100000)
        if not doc_result.get("success"):
            return doc_result

        lines = doc_result.get("content", "").splitlines()
        for idx, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"Line {idx}: {line.strip()[:300]}")
                if len(matches) >= max_matches:
                    break

        output_str = f"Found {len(matches)} match(es) for '{pattern}' in '{file_name}':\n" + "\n".join(matches) if matches else f"No matches found for '{pattern}' in '{file_name}'."
        return {
            "success": True,
            "matches_count": len(matches),
            "matches": matches,
            "output": output_str
        }

    except Exception as e:
        return {"success": False, "error": f"Grep failed on '{file_name}': {e}"}


def get_tools() -> List[Dict[str, Any]]:
    """Return the tool descriptors exposed by this library."""
    return [
        {
            "name": "tool_inspect_document",
            "description": "Inspects a document file to return structural metadata (format, page/sheet count, outline/headings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the document file in the workspace."},
                },
                "required": ["file_name"],
            },
            "output": {"type": "object"},
            "callable": tool_inspect_document,
            "_source": "as_is_document_tools",
            "_source_file": __file__,
        },
        {
            "name": "tool_read_document_content",
            "description": "Extracts readable text content from a document (supports specific pages or sheets).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the file in the workspace."},
                    "page_or_sheet": {"type": "string", "description": "Specific page number (PDF/PPTX) or sheet name (Excel)."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to extract. Defaults to 8000."},
                },
                "required": ["file_name"],
            },
            "output": {"type": "object"},
            "callable": tool_read_document_content,
            "_source": "as_is_document_tools",
            "_source_file": __file__,
        },
        {
            "name": "tool_grep_document",
            "description": "Searches for a regular expression or keyword within a document without loading the whole file into context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the file in the workspace."},
                    "pattern": {"type": "string", "description": "Search keyword or regex pattern."},
                    "max_matches": {"type": "integer", "description": "Maximum matching excerpts to return. Defaults to 20."},
                },
                "required": ["file_name", "pattern"],
            },
            "output": {"type": "object"},
            "callable": tool_grep_document,
            "_source": "as_is_document_tools",
            "_source_file": __file__,
        },
        {
            "name": "tool_add_word_comment",
            "description": "Adds a native Word review comment (Review-pane margin box) to a .docx file, anchored on search_text or paragraph_index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the .docx file in the workspace."},
                    "comment": {"type": "string", "description": "The comment text."},
                    "search_text": {"type": "string", "description": "Text to anchor the comment on (preferred)."},
                    "author": {"type": "string", "description": "Comment author name. Defaults to 'Assistant'."},
                    "paragraph_index": {"type": "integer", "description": "0-based index into non-empty paragraphs when search_text is absent."},
                },
                "required": ["file_name", "comment"],
            },
            "output": {"type": "object"},
            "callable": tool_add_word_comment,
            "_source": "as_is_document_tools",
            "_source_file": __file__,
        },
        {
            "name": "tool_get_word_comments",
            "description": "Lists all native Word review comments in a .docx file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the .docx file in the workspace."},
                },
                "required": ["file_name"],
            },
            "output": {"type": "object"},
            "callable": tool_get_word_comments,
            "_source": "as_is_document_tools",
            "_source_file": __file__,
        },
    ]