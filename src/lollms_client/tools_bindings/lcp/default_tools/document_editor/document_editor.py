import os
import re
import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOL_LIBRARY_NAME = "Document Editor"
TOOL_LIBRARY_DESC = "Surgically edits and annotates PDF, DOCX, and PPTX files. Supports text insertion, replacement, removal, comments, and highlighting."
TOOL_LIBRARY_ICON = "📝"

_DEFAULT_COMMENTER = getpass.getuser()

_LIGATURE_MAP = {
    '\ufb00': 'ff',
    '\ufb01': 'fi',
    '\ufb02': 'fl',
    '\ufb03': 'ffi',
    '\ufb04': 'ffl',
    '\ufb05': 'ft',
    '\ufb06': 'st',
}

_LIGATURE_TRANS = str.maketrans(_LIGATURE_MAP)


def init_tools_library(config: dict = None) -> None:
    try:
        import pipmaster as pm
        pm.ensure_packages("pymupdf")
        pm.ensure_packages("python-docx")
        pm.ensure_packages("python-pptx")
    except Exception as e:
        import ascii_colors
        ascii_colors.ASCIIColors.warning(f"[Document Editor] Failed to ensure dependencies: {e}")

def _get_output_path(file_name: str, suffix: str = "_edited") -> str:
    p = Path(file_name)
    stem = p.stem
    for existing_suffix in ["_annotated", "_edited", "_inserted", "_updated", "_removed"]:
        if stem.endswith(existing_suffix):
            stem = stem[:-len(existing_suffix)]
            break
    return str(p.with_name(f"{stem}{suffix}{p.suffix}"))

def _check_file_exists(file_name: str) -> Optional[Dict[str, Any]]:
    if not Path(file_name).is_file():
        return {
            "success": False,
            "error": f"File '{file_name}' not found."
        }
    return None

def _parse_page_numbers(pages_str: Optional[str], total_pages: int) -> List[int]:
    if not pages_str or not pages_str.strip():
        return list(range(total_pages))
    
    pages = set()
    for part in pages_str.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                pages.update(range(max(1, start) - 1, min(total_pages, end)))
            except ValueError:
                pass
        elif part.isdigit():
            p = int(part) - 1
            if 0 <= p < total_pages:
                pages.add(p)
    return sorted(list(pages))


def _normalize_text_for_search(text: str) -> str:
    text = text.translate(_LIGATURE_TRANS)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_keywords(search_text: str, min_len: int = 4) -> List[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "and", "or", "but", "not", "no", "if", "then", "so",
        "i", "you", "he", "she", "it", "we", "they",
    }
    normalized = _normalize_text_for_search(search_text)
    words = re.findall(r'[A-Za-z0-9+\-]+', normalized)
    return [w for w in words if len(w) >= min_len and w.lower() not in stop_words]


def _fuzzy_search_pdf_page(page, search_text: str, flags: int = 0) -> List:
    """
    Three-tier search for a text fragment on a PDF page using PyMuPDF.
    
    Tier 1: Exact literal search (fastest, matches PyMuPDF's built-in).
    Tier 2: Normalized search — collapses whitespace and fixes ligatures.
    Tier 3: Keyword proximity search — finds the region with the highest
            concentration of significant keywords from the search text.
    
    Returns a list of fitz.Rect objects (possibly empty).
    """
    exact_hits = page.search_for(search_text, flags=flags)
    if exact_hits:
        return exact_hits

    norm_search = _normalize_text_for_search(search_text)
    if not norm_search:
        return []

    norm_hits = page.search_for(norm_search, flags=flags)
    if norm_hits:
        return norm_hits

    partial_fragments = page.search_for(norm_search[:80], flags=flags)
    if partial_fragments:
        return partial_fragments

    keywords = _extract_keywords(search_text, min_len=4)
    if not keywords:
        return []

    page_rect = page.rect
    keyword_rects: List = []
    for kw in keywords:
        kw_hits = page.search_for(kw, flags=flags)
        if kw_hits:
            keyword_rects.extend(kw_hits)

    if not keyword_rects:
        return []

    best_center: Optional[Tuple[float, float]] = None
    best_score = 0

    for anchor_rect in keyword_rects:
        center = (anchor_rect.x0 + anchor_rect.x1) / 2, (anchor_rect.y0 + anchor_rect.y1) / 2
        score = 0
        for other_rect in keyword_rects:
            oc = (other_rect.x0 + other_rect.x1) / 2, (other_rect.y0 + other_rect.y1) / 2
            dist = ((center[0] - oc[0]) ** 2 + (center[1] - oc[1]) ** 2) ** 0.5
            if dist < 200:
                score += 1
        if score > best_score:
            best_score = score
            best_center = center

    if best_center is None:
        return []

    cx, cy = best_center
    half_w = min(150, page_rect.width / 3)
    half_h = min(40, page_rect.height / 4)
    fuzzy_rect = page_rect & type(page.rect)(
        cx - half_w, cy - half_h, cx + half_w, cy + half_h
    )
    return [fuzzy_rect] if fuzzy_rect else []


def _fuzzy_find_in_paragraphs(paragraphs_text: List[str], search_text: str) -> List[int]:
    """
    Fuzzy search across a list of paragraph text strings.
    Returns indices of paragraphs that match.
    
    Tier 1: Exact substring match.
    Tier 2: Normalized match (whitespace/ligature insensitive).
    Tier 3: Keyword coverage match (>=60% of significant keywords present).
    """
    norm_search = _normalize_text_for_search(search_text)
    keywords = _extract_keywords(search_text, min_len=4)
    
    matched_indices: List[int] = []
    
    for idx, ptext in enumerate(paragraphs_text):
        if search_text in ptext:
            matched_indices.append(idx)
            continue
        
        norm_ptext = _normalize_text_for_search(ptext)
        if norm_search and norm_search in norm_ptext:
            matched_indices.append(idx)
            continue
        
        if keywords:
            present = sum(1 for kw in keywords if kw.lower() in norm_ptext.lower())
            if present / len(keywords) >= 0.6:
                matched_indices.append(idx)
                continue
    
    return matched_indices


def tool_edit_document_text(
    file_name: str,
    operation: str,
    search_text: str,
    replacement_text: str = "",
    pages: str = "",
    match_case: bool = False,
    whole_word: bool = False
) -> Dict[str, Any]:
    """
    Surgically edits text content in a PDF, DOCX, or PPTX document.
    Uses fuzzy matching to locate the text and applies the specified operation.

    Args:
        file_name (str): The path to the document file (PDF, DOCX, PPTX).
        operation (str): The edit operation to perform. Options: "insert", "update", "remove".
        search_text (str): The exact text to search for. For "insert", this is the anchor text after which to insert.
        replacement_text (str, optional): The new text. Required for "update" and "insert". Ignored for "remove".
        pages (str, optional): Pages to apply the edit (PDF only). E.g., "1-3, 5". Empty means all pages.
        match_case (bool, optional): Whether to match the exact case. Defaults to False.
        whole_word (bool, optional): Whether to match whole words only. Defaults to False.
    """
    err = _check_file_exists(file_name)
    if err:
        return err

    if operation not in ("insert", "update", "remove"):
        return {"success": False, "error": "Invalid operation. Must be 'insert', 'update', or 'remove'."}
    
    if operation in ("update", "insert") and not replacement_text and operation != "remove":
        return {"success": False, "error": "replacement_text is required for 'update' and 'insert' operations."}

    output_path = _get_output_path(file_name, f"_{operation}ed")
    file_ext = Path(file_name).suffix.lower()

    try:
        if file_ext == ".pdf":
            import fitz
            doc = fitz.open(file_name)
            target_pages = _parse_page_numbers(pages, len(doc))
            total_matches = 0
            
            flags = 0
            if not match_case:
                flags |= fitz.TEXT_DEHYPHENATE
            if whole_word:
                flags |= fitz.TEXT_FIND_WHOLEWORDS

            for page_num in target_pages:
                page = doc[page_num]
                
                if operation == "insert":
                    text_instances = _fuzzy_search_pdf_page(page, search_text, flags=flags)
                    if text_instances:
                        rect = text_instances[0]
                        point = fitz.Point(rect.x1, rect.y0)
                        page.insert_text(point, " " + replacement_text, fontsize=11, fontname="helv")
                        total_matches += 1
                else:
                    text_instances = _fuzzy_search_pdf_page(page, search_text, flags=flags)
                    for inst in text_instances:
                        if operation == "remove":
                            page.add_redact_annot(inst, fill=(1, 1, 1))
                        else:
                            page.add_redact_annot(inst, text=replacement_text, fill=(1, 1, 1), fontsize=11)
                        total_matches += 1
                    page.apply_redactions()

            if total_matches == 0:
                doc.close()
                return {"success": False, "error": f"Search text not found in the specified pages."}

            doc.save(output_path)
            doc.close()
            
        elif file_ext == ".docx":
            import docx
            doc = docx.Document(file_name)
            total_matches = 0
            
            paragraph_texts = [p.text for p in doc.paragraphs]
            matched_indices = _fuzzy_find_in_paragraphs(paragraph_texts, search_text)
            
            for idx in matched_indices:
                paragraph = doc.paragraphs[idx]
                if operation == "remove":
                    paragraph.clear()
                    total_matches += 1
                elif operation == "update":
                    paragraph.text = paragraph.text.replace(search_text, replacement_text)
                    if search_text not in paragraph.text:
                        norm_search = _normalize_text_for_search(search_text)
                        norm_ptext = _normalize_text_for_search(paragraph.text)
                        if norm_search in norm_ptext:
                            paragraph.text = norm_ptext.replace(norm_search, replacement_text)
                    total_matches += 1
                elif operation == "insert":
                    paragraph.add_run(" " + replacement_text)
                    total_matches += 1

            if total_matches == 0:
                return {"success": False, "error": "Search text not found in document."}

            doc.save(output_path)

        elif file_ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(file_name)
            total_matches = 0
            
            for slide in prs.slides:
                for shape in slide.shapes:
                    if not shape.has_text_frame:
                        continue
                    for paragraph in shape.text_frame.paragraphs:
                        full_text = "".join(run.text for run in paragraph.runs)
                        matched = False
                        
                        if search_text in full_text:
                            matched = True
                        else:
                            norm_search = _normalize_text_for_search(search_text)
                            norm_full = _normalize_text_for_search(full_text)
                            if norm_search and norm_search in norm_full:
                                matched = True
                            else:
                                keywords = _extract_keywords(search_text, min_len=4)
                                if keywords:
                                    present = sum(1 for kw in keywords if kw.lower() in full_text.lower())
                                    if present / len(keywords) >= 0.6:
                                        matched = True
                        
                        if matched:
                            if operation == "remove":
                                for run in paragraph.runs:
                                    run.text = ""
                                total_matches += 1
                            elif operation == "update":
                                for run in paragraph.runs:
                                    if search_text in run.text:
                                        run.text = run.text.replace(search_text, replacement_text)
                                        total_matches += 1
                                    else:
                                        run.text = _normalize_text_for_search(run.text).replace(
                                            _normalize_text_for_search(search_text),
                                            replacement_text
                                        )
                                        if replacement_text in run.text:
                                            total_matches += 1
                            elif operation == "insert":
                                if paragraph.runs:
                                    paragraph.runs[-1].text += " " + replacement_text
                                    total_matches += 1

            if total_matches == 0:
                return {"success": False, "error": "Search text not found in presentation."}

            prs.save(output_path)
        else:
            return {"success": False, "error": f"Unsupported file extension: {file_ext}"}

        return {
            "success": True,
            "output": f"Successfully applied '{operation}' to {total_matches} instance(s). Saved to {output_path}",
            "file_name": output_path
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Document editing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

def tool_annotate_document(
    file_name: str,
    annotation_type: str,
    search_text: str,
    comment: str = "",
    pages: str = "",
    highlight_color: str = "yellow",
    commenter_name: str = _DEFAULT_COMMENTER
) -> Dict[str, Any]:
    """
    Annotates a PDF or DOCX document by highlighting text or adding comments.

    Args:
        file_name (str): The path to the document file (PDF or DOCX).
        annotation_type (str): The type of annotation. Options: "comment", "highlight".
        search_text (str): The text to locate for annotation.
        comment (str, optional): The comment text. Required for "comment" type.
        pages (str, optional): Pages to apply the annotation (PDF only). E.g., "1-3, 5".
        highlight_color (str, optional): Color for highlighting. Options: "yellow", "red", "green", "blue".
        commenter_name (str, optional): The name of the user adding the comment. Defaults to current OS user.
    """
    err = _check_file_exists(file_name)
    if err:
        return err

    if annotation_type not in ("comment", "highlight"):
        return {"success": False, "error": "Invalid annotation_type. Must be 'comment' or 'highlight'."}

    if annotation_type == "comment" and not comment:
        return {"success": False, "error": "comment is required when annotation_type is 'comment'."}

    if not commenter_name:
        commenter_name = _DEFAULT_COMMENTER

    output_path = _get_output_path(file_name, "_annotated")
    file_ext = Path(file_name).suffix.lower()

    color_map = {
        "yellow": (1, 1, 0),
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1)
    }
    rgb_color = color_map.get(highlight_color.lower(), (1, 1, 0))

    try:
        if file_ext == ".pdf":
            import fitz
            doc = fitz.open(file_name)
            target_pages = _parse_page_numbers(pages, len(doc))
            total_annot = 0
            
            for page_num in target_pages:
                page = doc[page_num]
                text_instances = _fuzzy_search_pdf_page(page, search_text)
                
                for inst in text_instances:
                    if annotation_type == "highlight":
                        annot = page.add_highlight_annot(inst)
                        annot.set_colors(stroke=rgb_color)
                        annot.update()
                    else:
                        annot = page.add_text_annot(
                            fitz.Point(inst.x0, inst.y0),
                            comment,
                            icon="Comment"
                        )
                        annot.set_info(title=commenter_name, content=comment)
                        annot.update()
                    total_annot += 1

            if total_annot == 0:
                doc.close()
                return {"success": False, "error": "Search text not found for annotation."}

            doc.save(output_path)
            doc.close()

        elif file_ext == ".docx":
            import docx
            from docx.oxml.ns import qn
            from docx.oxml import OxmlElement

            doc = docx.Document(file_name)
            total_annot = 0
            
            paragraph_texts = [p.text for p in doc.paragraphs]
            matched_indices = _fuzzy_find_in_paragraphs(paragraph_texts, search_text)
            
            for idx in matched_indices:
                paragraph = doc.paragraphs[idx]
                if annotation_type == "highlight":
                    for run in paragraph.runs:
                        if search_text in run.text or _normalize_text_for_search(search_text) in _normalize_text_for_search(run.text):
                            rPr = run._element.get_or_create_rPr()
                            highlight = OxmlElement('w:highlight')
                            highlight.set(qn('w:val'), highlight_color)
                            rPr.append(highlight)
                            total_annot += 1
                    if total_annot == 0 and matched_indices:
                        for run in paragraph.runs:
                            rPr = run._element.get_or_create_rPr()
                            highlight = OxmlElement('w:highlight')
                            highlight.set(qn('w:val'), highlight_color)
                            rPr.append(highlight)
                            total_annot += 1
                else:
                    paragraph.add_run(f" [COMMENT by {commenter_name}: {comment}]")
                    total_annot += 1

            if total_annot == 0:
                return {"success": False, "error": "Search text not found for annotation."}

            doc.save(output_path)
        else:
            return {"success": False, "error": f"Unsupported file extension: {file_ext}"}

        return {
            "success": True,
            "output": f"Successfully added {total_annot} {annotation_type}(s). Saved to {output_path}",
            "file_name": output_path
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Document annotation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }