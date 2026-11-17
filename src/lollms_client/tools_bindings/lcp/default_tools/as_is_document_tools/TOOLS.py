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
# File types this library handles. The LCP binding auto-mounts this toolset
# whenever one of these extensions is present in the workspace, so dropping a
# .pdf or .docx into the workspace makes it immediately available without
# requiring enable_office_files.
TOOL_LIBRARY_FILE_TYPES = [".pdf", ".docx", ".docm", ".xlsx", ".xls", ".pptx"]

_VISION_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

# ---------------------------------------------------------------------------
# PDF page-rendering defaults
# ---------------------------------------------------------------------------
# These constants control how `tool_render_pdf_pages` turns PDF pages into
# vision content that the model can inspect for one single round.
#
# _PDF_RENDER_MAX_DIM : longest side (px) an image is downscaled to before
#                       base64 encoding. Keeps tokens-per-image predictable
#                       across scanned vs. vector PDFs.
# _PDF_RENDER_DPI     : default rasterization resolution. 150 DPI is the sweet
#                       spot for readable scanned text and legible chart axes
#                       without exploding image-token budgets. Clamped to
#                       [72, 300] inside the tool.
# _PDF_RENDER_JPEG_Q  : JPEG quality for RGB pages (85 is visually lossless for
#                       document content while keeping payloads small).
# _MAX_RENDER_PAGES   : HARD CEILING on how many pages a single call may render.
#                       This protects the context window from a model that asks
#                       for "all 400 pages" of a scanned report. The model can
#                       still iterate over a large document by narrowing the
#                       `pages` selector across multiple calls.
# ---------------------------------------------------------------------------
_PDF_RENDER_MAX_DIM   = 1600
_PDF_RENDER_DPI       = 150
_PDF_RENDER_JPEG_Q    = 85
_MAX_RENDER_PAGES     = 12


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


def _parse_paragraph_selector(selector: Optional[str], total: int) -> List[int]:
    """Parses a 0-based paragraph selector into a sorted list of indices.

    Grammar: "10" | "10-50" | "10,20,30-40" | "" (all).
    Out-of-range values are clamped; unparseable tokens are ignored.
    """
    if not selector or not str(selector).strip():
        return list(range(total))
    indices: set[int] = set()
    for part in str(selector).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            try:
                start, end = int(start_s.strip()), int(end_s.strip())
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for i in range(max(0, start), min(total - 1, end) + 1):
                indices.add(i)
        elif part.isdigit():
            i = int(part)
            if 0 <= i < total:
                indices.add(i)
    return sorted(indices) if indices else list(range(total))


def _parse_page_selector(selector: Optional[str], total_pages: int) -> List[int]:
    """
    Parse a 1-based page selector into a sorted list of 0-based page indices.

    Accepted grammar (case-insensitive, whitespace-tolerant):
        "3"            -> [2]                    single page
        "2-6"          -> [1, 2, 3, 4, 5]        inclusive range
        "1,4,9-11"     -> [0, 3, 8, 9, 10]       mixed list + ranges
        "" / None      -> list(range(total_pages))  all pages

    Behaviour:
        * Values outside [1, total_pages] are silently clamped/dropped so a
          hallucinated page number never raises.
        * Reversed ranges ("6-2") are normalized to ascending order.
        * Unparseable tokens are ignored; if every token is invalid, the
          function falls back to "all pages" so the caller never gets an
          empty selection unexpectedly.
        * Output is sorted and de-duplicated for deterministic rendering.

    Returns:
        List[int]: 0-based page indices ready to feed `doc[i]`.
    """
    if not selector or not str(selector).strip():
        return list(range(total_pages))

    pages: set[int] = set()
    for part in str(selector).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            try:
                start = int(start_s.strip())
                end = int(end_s.strip())
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for p in range(max(1, start), min(total_pages, end) + 1):
                pages.add(p - 1)
        elif part.isdigit():
            p = int(part)
            if 1 <= p <= total_pages:
                pages.add(p - 1)

    return sorted(pages) if pages else list(range(total_pages))


def _resize_and_encode_image(img_bytes: bytes, max_dim: int = _PDF_RENDER_MAX_DIM) -> tuple[str, str]:
    """
    Downscale an image and encode it for vision consumption.

    Pipeline: Pillow open -> (mode normalize) -> LANCZOS downscale ->
    format-specific encode -> base64-str.

    Encoding rules:
        * RGBA  -> PNG (preserves alpha for figures with transparency).
        * other -> JPEG @ _PDF_RENDER_JPEG_Q (small payload, document-friendly).

    Robustness:
        * Any Pillow/decode failure falls back to returning the ORIGINAL bytes
          base64-wrapped as `image/png`. This guarantees the caller always gets
          a usable (image_b64, media_type) pair and never crashes the round on
          an exotic image codec embedded inside a PDF.
        * `max_dim` <= 0 is treated as "no downscale" (only re-encoding).

    Returns:
        (base64_str, media_type) where media_type is "image/jpeg" or "image/png".
    """
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB")
        width, height = img.size
        if max(width, height) > max_dim:
            ratio = max_dim / float(max(width, height))
            img = img.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        if img.mode == "RGBA":
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"
        img.save(buf, format="JPEG", quality=_PDF_RENDER_JPEG_Q)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
    except Exception:
        return base64.b64encode(img_bytes).decode("utf-8"), "image/png"


def tool_render_pdf_pages(
    file_name: str,
    pages: Optional[str] = None,
    max_pages: int = _MAX_RENDER_PAGES,
    include_embedded_figures: bool = False,
    dpi: int = _PDF_RENDER_DPI,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Render PDF page(s) to images for ONE-SHOT visual inspection.

    PURPOSE
    -------
    `tool_read_document_content` returns text extracted from a PDF. That is
    perfect for born-digital documents, but it fails on:

        * Scanned pages (no selectable text / OCR noise).
        * Mathematical curves, plots, and scientific figures.
        * Architectural / engineering drawings.
        * Handwriting, signatures, stamps, diagrams.
        * Any layout where "picture = meaning".

    `tool_render_pdf_pages` rasters the requested pages and returns them as
    base64 images attached to the NEXT LLM round. A vision-capable model can
    then literally see the page.

    EPHEMERAL-BY-DESIGN
    -------------------
    Rendered images are NOT persisted across rounds. They live only in the
    vision-hydration payload of the very next generation. This is deliberate:
    a 200-page scanned PDF would otherwise permanently bloat the context.
    The returned `prompt_injection` block explicitly tells the model to write
    every extracted fact into the scratchpad (or an artifact) BEFORE the round
    ends; otherwise that knowledge is lost.

    WHEN TO USE
    -----------
        ✔ Scanned pages / image-only pages.
        ✔ Charts, curves, plots, spectrograms, heat-maps.
        ✔ Diagrams, flowcharts, schematics.
        ✔ Visual verification of layout (columns, tables, forms).
        ✘ Pure text extraction  ->  use `tool_read_document_content`.
        ✘ Editing PDF text      ->  use `tool_edit_document_text`.

    ARGUMENTS
    ---------
        file_name (str):
            Relative path to the PDF file inside the workspace.
        pages (str, optional):
            1-based selector. Accepts "3", "2-6", "1,4,9-11".
            Defaults to ALL pages (capped by `max_pages`).
        max_pages (int, optional):
            Safety cap per call. Defaults to 12, hard ceiling of 12.
            If the selection is longer, it is truncated with a `warning`
            key returned in the payload so the model knows to call again
            with a narrower selector.
        include_embedded_figures (bool, optional):
            When True, ALSO extracts the raw raster figures embedded inside
            each selected page (xref images). Useful when a page contains
            high-resolution charts whose raster survives downscaling better
            than the rendered page. Each extracted figure is appended to the
            `images` list, so `image_count` may exceed `len(pages_rendered)`.
        dpi (int, optional):
            Rasterization resolution. Defaults to 150. Clamped to [72, 300].
            Raise to 220-300 for dense scanned text; lower to 100-120 for
            large plot overviews to save image tokens.
        env (Any, optional):
            Reserved for future use (workspace root override / sandbox info).
            Currently unused.

    RETURNS (dict)
    --------------
        success (bool)
        file_name (str)
        total_pages (int)         Total page count of the PDF.
        pages_rendered (List[int]) 1-based page numbers actually rendered.
        image_count (int)         Total images returned (pages + figures).
        content / output (str)    Human-readable summary + per-page previews.
        images (List[str])        Base64-encoded image payloads.
        image_media_types (List[str]) Parallel media types ("image/jpeg"
                                  or "image/png") for `images`.
        ephemeral (bool)          Always True — signals one-round lifetime.
        prompt_injection (str)    Instruction block injected into the next
                                  model turn, telling it that the images
                                  are ephemeral and that findings must be
                                  persisted to the scratchpad.
        warning (str, optional)   Present iff the selection was truncated.
        error / traceback         Present only when `success` is False.

    EXAMPLES
    --------
        # Look at a single scanned page to OCR it visually:
        tool_render_pdf_pages("reports/q3_scan.pdf", pages="7")

        # Inspect a 3-page batch of a scientific paper (fig. 2-4 are pages 5-7):
        tool_render_pdf_pages("papers/attention.pdf", pages="5-7")

        # Inspect a plot-heavy page AND pull the original figure rasters:
        tool_render_pdf_pages(
            "papers/charts.pdf",
            pages="12",
            include_embedded_figures=True,
            dpi=220,
        )

        # Walk a 40-page scan in safe chunks of 10:
        tool_render_pdf_pages("scan.pdf", pages="1-10")
        tool_render_pdf_pages("scan.pdf", pages="11-20")
        # ... persist findings into the scratchpad between calls.
    """
    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}
    if path.suffix.lower() != ".pdf":
        return {"success": False, "error": f"File '{file_name}' is not a PDF (got '{path.suffix}')."}

    # --- clamp max_pages defensively ---
    try:
        max_pages = int(max_pages)
    except (TypeError, ValueError):
        max_pages = _MAX_RENDER_PAGES
    max_pages = max(1, min(max_pages, _MAX_RENDER_PAGES))

    # --- clamp dpi ---
    try:
        dpi = int(dpi)
    except (TypeError, ValueError):
        dpi = _PDF_RENDER_DPI
    dpi = max(72, min(dpi, 300))

    try:
        import fitz
    except ImportError:
        return {
            "success": False,
            "error": "PyMuPDF (fitz) is required to render PDF pages. "
                     "Install it with: pip install pymupdf",
        }

    if hasattr(fitz, "TOOLS"):
        try:
            fitz.TOOLS.mupdf_display_errors(False)
            fitz.TOOLS.mupdf_display_warnings(False)
        except Exception:
            pass

    doc = None
    try:
        doc = fitz.open(str(path))
        total_pages = len(doc)
        if total_pages == 0:
            return {"success": False, "error": f"PDF '{file_name}' contains no pages."}

        selected = _parse_page_selector(pages, total_pages)
        if len(selected) > max_pages:
            truncated = selected[:max_pages]
            warning = (
                f"Requested {len(selected)} page(s); rendering only the first "
                f"{max_pages} to protect the context. Call again with a narrower "
                f"'pages' selector for the rest."
            )
            selected = truncated
        else:
            warning = None

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        rendered_images: List[str] = []
        media_types: List[str] = []
        page_summaries: List[str] = []

        for page_index in selected:
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            b64, mt = _resize_and_encode_image(pix.tobytes("jpeg"))
            rendered_images.append(b64)
            media_types.append(mt)

            snippet = (page.get_text() or "").strip().replace("\n", " ")
            snippet = (snippet[:120] + "…") if len(snippet) > 120 else snippet
            page_summaries.append(
                f"- Page {page_index + 1}: rendered"
                + (f" — text preview: \"{snippet}\"" if snippet else " — no selectable text (image-only page)")
            )

            if include_embedded_figures:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        base_img = doc.extract_image(xref)
                        raw = base_img.get("image")
                        if not raw:
                            continue
                        b64_fig, mt_fig = _resize_and_encode_image(raw)
                        rendered_images.append(b64_fig)
                        media_types.append(mt_fig)
                        page_summaries.append(f"  · embedded figure from page {page_index + 1}")
                    except Exception:
                        continue

        doc.close()
        doc = None

        page_list = ", ".join(str(p + 1) for p in selected)
        summary = (
            f"Rendered {len(rendered_images)} image(s) covering page(s) {page_list} "
            f"of '{file_name}' ({total_pages} page(s) total)."
        )
        output = summary + "\n" + "\n".join(page_summaries)
        if warning:
            output = warning + "\n\n" + output

        result: Dict[str, Any] = {
            "success": True,
            "file_name": file_name,
            "total_pages": total_pages,
            "pages_rendered": [p + 1 for p in selected],
            "image_count": len(rendered_images),
            "content": output,
            "output": output,
            "images": rendered_images,
            "image_media_types": media_types,
            "ephemeral": True,
            "prompt_injection": (
                f"\n\n[SYSTEM: {len(rendered_images)} page/figure image(s) from "
                f"'{file_name}' (page(s) {page_list}) have been attached as "
                f"vision content for THIS round only. Inspect them carefully — "
                f"read scanned text, interpret curves, charts, and diagrams. "
                f"Before the round ends, persist every useful finding into the "
                f"scratchpad (or an artifact) so it survives: these images will "
                f"NOT be available in subsequent rounds.]"
            ),
        }
        if warning:
            result["warning"] = warning
        return result

    except Exception as e:
        import traceback as _tb
        return {
            "success": False,
            "error": f"Failed to render PDF pages from '{file_name}': {e}",
            "traceback": _tb.format_exc(),
        }
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


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
            total_paragraphs = len(doc.paragraphs)

            # `page_or_sheet` is interpreted as a 0-based paragraph selector
            # ("10", "10-50", "10,20,30-40") so callers can read a chunk of a
            # large document without loading the whole body.
            if page_or_sheet and str(page_or_sheet).strip():
                selected = _parse_paragraph_selector(str(page_or_sheet), total_paragraphs)
                selected_set = set(selected)
                paragraphs = [
                    p.text for idx, p in enumerate(doc.paragraphs)
                    if idx in selected_set and p.text.strip()
                ]
            else:
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            full_text = "\n\n".join(paragraphs)
            total_chars = len(full_text)
            text = full_text[:max_chars]
            truncated = total_chars > len(text)

            # Image metadata (no bytes loaded) so the caller knows what's inside.
            images_meta: List[Dict[str, Any]] = []
            try:
                from docx.oxml.ns import qn
                for para_idx, para in enumerate(doc.paragraphs):
                    for run in para.runs:
                        for drawing in run._element.findall(f".//{qn('w:drawing')}"):
                            for blip in drawing.findall(f".//{qn('a:blip')}"):
                                embed_id = blip.get(qn("r:embed"))
                                if not embed_id:
                                    continue
                                try:
                                    image_part = doc.part.related_parts[embed_id]
                                    images_meta.append({
                                        "paragraph_index": para_idx,
                                        "filename": Path(str(image_part.partname)).name,
                                        "size_bytes": len(image_part.blob),
                                        "content_type": image_part.content_type,
                                    })
                                except Exception:
                                    continue
            except Exception:
                pass

            result: Dict[str, Any] = {
                "success": True,
                "content": text,
                "output": text,
                "total_chars": total_chars,
                "total_paragraphs": total_paragraphs,
                "truncated": truncated,
                "images": images_meta,
                "image_count": len(images_meta),
            }
            if truncated:
                result["next_offset_hint"] = (
                    f"Body truncated at {len(text)} of {total_chars} chars. "
                    f"Use tool_read_word(mode='chunk', offset={len(text)}) for the rest, "
                    f"or narrow with page_or_sheet='<paragraph_range>'."
                )
            return result

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
            "description": (
                "Extracts readable text content from a document. For PDF/PPTX, "
                "`page_or_sheet` selects pages ('3', '2-6', '1,4,9-11'). For Excel, "
                "it selects the sheet name. For DOCX, it selects a 0-based PARAGRAPH "
                "range ('10', '10-50', '10,20,30-40'). Every response includes "
                "total_chars and (for DOCX) total_paragraphs + images metadata "
                "(count + per-image info) so you can plan chunked reads. When the "
                "body is truncated, `next_offset_hint` tells you how to continue "
                "with tool_read_word(mode='chunk', offset=...)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the file in the workspace."},
                    "page_or_sheet": {"type": "string", "description": "PDF/PPTX: page selector ('3', '2-6', '1,4,9-11'). Excel: sheet name. DOCX: 0-based paragraph range ('10-50')."},
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
            "name": "tool_render_pdf_pages",
            "description": (
                "Renders PDF page(s) to images and attaches them as EPHEMERAL vision "
                "content for the NEXT LLM round only — ideal for scanned pages, "
                "curves, charts, plots, and diagrams. Optionally also extracts "
                "embedded figures. Images are NOT persisted across rounds: save "
                "findings into the scratchpad or an artifact before the round ends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Relative path to the PDF file in the workspace."},
                    "pages": {"type": "string", "description": "1-based page selector, e.g. '3', '2-6', '1,4,9-11'. Defaults to all pages."},
                    "max_pages": {"type": "integer", "description": "Safety cap on pages rendered per call (default 12, hard max 12)."},
                    "include_embedded_figures": {"type": "boolean", "description": "Also extract raw embedded raster figures/plots from the selected pages."},
                    "dpi": {"type": "integer", "description": "Render resolution in DPI (default 150, clamped to 72-300)."},
                },
                "required": ["file_name"],
            },
            "output": {"type": "object"},
            "callable": tool_render_pdf_pages,
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