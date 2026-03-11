# lollms_discussion/_repl_tools.py
# ---------------------------------------------------------------------------
# REPL Text Tools — format-aware in-session text REPL for the agentic loop.
#
# The Problem
# -----------
# MCP tools and RAG pipelines can return very large payloads — a list of
# scientific articles, a CSV export, a JSONL feed, a wall of Markdown.
# Stuffing the whole payload into the context window as a tool_result is
# wasteful and may overflow the budget.  Instead:
#
#   1. The LLM calls  text_store(handle, content)  immediately after receiving
#      a large result.  The payload is indexed server-side and a compact
#      summary (record count, schema, first few items) is returned.
#   2. The LLM then navigates the data with targeted calls:
#      text_search / text_get_range / text_get_record / text_list_records
#   3. Structured data can be filtered and aggregated without ever re-injecting
#      the full payload.
#   4. When done, text_to_artefact persists the (possibly filtered) result as a
#      discussion artefact so the user can download it.
#
# Format auto-detection (in priority order)
# -----------------------------------------
#   JSON_ARRAY   – top-level JSON array of dicts   → each dict is a record
#   JSONL        – one JSON object per line          → each line is a record
#   CSV          – comma / tab / semicolon delimited → each row is a record
#   MD_TABLE     – Markdown pipe-table               → each body row is a record
#   NUMBERED_LIST– "1. …" / "- …" lines             → each item is a record
#   SECTIONS     – "## Heading" separated blocks     → each section is a record
#   LINES        – fallback: every non-blank line    → each line is a record
#
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["TextBuffer", "register_repl_tools"]


# ── constants ─────────────────────────────────────────────────────────────────

_FMT_JSON_ARRAY   = "json_array"
_FMT_JSONL        = "jsonl"
_FMT_CSV          = "csv"
_FMT_MD_TABLE     = "md_table"
_FMT_NUMBERED     = "numbered_list"
_FMT_SECTIONS     = "sections"
_FMT_LINES        = "lines"

_PREVIEW_RECORDS  = 3    # how many records to include in the store() summary
_MAX_RECORD_CHARS = 800  # soft cap per record in get_record() output


# ── helpers ───────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int = _MAX_RECORD_CHARS) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"… [+{len(s)-n} chars]"


def _sniff_csv_dialect(text: str) -> Optional[csv.Dialect]:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|")
        return dialect
    except csv.Error:
        return None


def _detect_format(content: str) -> Tuple[str, Optional[Any]]:
    """
    Returns (format_name, parsed_data_or_None).
    parsed_data is the fully-parsed representation when cheap to produce upfront
    (JSON / CSV); None for line-based formats where we index lazily.
    """
    stripped = content.strip()

    # ── JSON array ────────────────────────────────────────────────────────────
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                return _FMT_JSON_ARRAY, data
        except json.JSONDecodeError:
            pass

    # ── JSONL ─────────────────────────────────────────────────────────────────
    lines = [l for l in content.splitlines() if l.strip()]
    if len(lines) >= 2:
        jsonl_hits = 0
        for ln in lines[:10]:
            try:
                obj = json.loads(ln.strip())
                if isinstance(obj, dict):
                    jsonl_hits += 1
            except json.JSONDecodeError:
                break
        if jsonl_hits >= min(3, len(lines[:10])):
            parsed = []
            for ln in lines:
                try:
                    parsed.append(json.loads(ln.strip()))
                except json.JSONDecodeError:
                    pass
            if parsed:
                return _FMT_JSONL, parsed

    # ── CSV ───────────────────────────────────────────────────────────────────
    dialect = _sniff_csv_dialect(content)
    if dialect is not None:
        try:
            reader = csv.DictReader(io.StringIO(content), dialect=dialect)
            rows = list(reader)
            if rows and len(rows[0]) > 1:
                return _FMT_CSV, rows
        except Exception:
            pass

    # ── Markdown table ────────────────────────────────────────────────────────
    table_lines = [l for l in content.splitlines() if "|" in l]
    if len(table_lines) >= 3:
        # must have a separator row (|---|)
        sep = [l for l in table_lines if re.match(r"^\|[-:| ]+\|", l.strip())]
        if sep:
            header_line = table_lines[0]
            headers = [h.strip() for h in header_line.strip("|").split("|")]
            body = [l for l in table_lines if l not in sep and l != header_line]
            rows = []
            for row_line in body:
                cells = [c.strip() for c in row_line.strip("|").split("|")]
                rows.append(dict(zip(headers, cells)))
            if rows:
                return _FMT_MD_TABLE, rows

    # ── Numbered / bulleted list ──────────────────────────────────────────────
    num_pat = re.compile(r"^(\d+[\.\)]|[-*•])\s+(.+)", re.MULTILINE)
    num_matches = num_pat.findall(content)
    if len(num_matches) >= 5:
        records = [{"index": i + 1, "text": m[1].strip()}
                   for i, m in enumerate(num_matches)]
        return _FMT_NUMBERED, records

    # ── Sections (## Heading) ─────────────────────────────────────────────────
    sec_pat = re.compile(r"^#{1,4}\s+(.+)", re.MULTILINE)
    sec_titles = sec_pat.findall(content)
    if len(sec_titles) >= 3:
        parts = sec_pat.split(content)
        # parts: [preamble, title1, body1, title2, body2, …]
        records = []
        i = 1
        while i + 1 < len(parts):
            records.append({"title": parts[i].strip(), "body": parts[i + 1].strip()})
            i += 2
        if records:
            return _FMT_SECTIONS, records

    # ── Fallback: raw lines ───────────────────────────────────────────────────
    return _FMT_LINES, None


def _schema_summary(records: List[Dict]) -> Dict[str, Any]:
    """Infer a lightweight schema from a list of dicts."""
    if not records:
        return {}
    all_keys: dict = {}
    for r in records[:50]:
        for k, v in r.items():
            if k not in all_keys:
                all_keys[k] = type(v).__name__
    return all_keys


def _record_summary(record: Any, max_chars: int = 120) -> str:
    """One-line summary of a record for paginated listing."""
    if isinstance(record, dict):
        # Try common title-like keys first
        for key in ("title", "name", "id", "subject", "heading", "text",
                    "abstract", "summary", list(record.keys())[0] if record else ""):
            if key and key in record:
                return _truncate(str(record[key]), max_chars)
    return _truncate(str(record), max_chars)


# ── TextBuffer ────────────────────────────────────────────────────────────────

class TextBuffer:
    """
    In-session indexed store for large text payloads.

    Each buffer is identified by a caller-chosen *handle* string.
    The store is a plain dict held in the chat() local scope — it never
    persists across sessions, which is intentional: it's scratch space.
    """

    def __init__(self) -> None:
        # handle → { format, records, raw_content, meta }
        self._store: Dict[str, Dict[str, Any]] = {}

    # ── ingestion ─────────────────────────────────────────────────────────────

    def store(self, handle: str, content: str) -> Dict[str, Any]:
        fmt, parsed = _detect_format(content)

        # Normalise everything to a list of records
        if parsed is not None:
            records = parsed if isinstance(parsed, list) else [parsed]
        else:
            # _FMT_LINES: split on non-blank lines
            records = [{"line": i + 1, "text": l}
                       for i, l in enumerate(content.splitlines())
                       if l.strip()]

        schema = _schema_summary(records) if records and isinstance(records[0], dict) else {}
        preview = [_record_summary(r) for r in records[:_PREVIEW_RECORDS]]

        entry = {
            "format":      fmt,
            "records":     records,
            "raw_content": content,
            "schema":      schema,
            "total":       len(records),
        }
        self._store[handle] = entry

        return {
            "success":       True,
            "handle":        handle,
            "format":        fmt,
            "total_records": len(records),
            "schema":        schema,
            "preview":       preview,
            "hint":          (
                f"Buffer '{handle}' ready. Use text_search, text_get_range, "
                f"text_get_record, text_list_records to navigate it."
            ),
        }

    # ── navigation ────────────────────────────────────────────────────────────

    def get(self, handle: str) -> Optional[Dict[str, Any]]:
        return self._store.get(handle)

    def list_handles(self) -> List[str]:
        return list(self._store.keys())

    def search(
        self,
        handle: str,
        query: str,
        max_results: int = 10,
        field: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._store.get(handle)
        if entry is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}

        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        hits = []
        for i, rec in enumerate(entry["records"]):
            target_text = (
                str(rec.get(field, "")) if field and isinstance(rec, dict)
                else json.dumps(rec, ensure_ascii=False)
                if isinstance(rec, dict)
                else str(rec)
            )
            if pattern.search(target_text):
                hits.append({
                    "index":   i,
                    "summary": _record_summary(rec),
                    "snippet": _truncate(target_text, 300),
                })
                if len(hits) >= max_results:
                    break

        return {
            "success":      True,
            "handle":       handle,
            "query":        query,
            "hits":         hits,
            "hit_count":    len(hits),
            "total_scanned": entry["total"],
            "truncated":    len(hits) == max_results,
        }

    def get_range(
        self,
        handle: str,
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        """Return records[start..end] inclusive, 0-based."""
        entry = self._store.get(handle)
        if entry is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}
        total = entry["total"]
        start = max(0, start)
        end   = min(end, total - 1)
        if start > end:
            return {"success": False, "error": f"start ({start}) > end ({end})."}
        slice_ = entry["records"][start : end + 1]
        return {
            "success": True,
            "handle":  handle,
            "start":   start,
            "end":     end,
            "records": [_truncate(json.dumps(r, ensure_ascii=False)
                                  if isinstance(r, dict) else str(r))
                        for r in slice_],
            "count":   len(slice_),
            "total":   total,
        }

    def get_record(self, handle: str, index: int) -> Dict[str, Any]:
        """Return one record by 0-based index, in full (with per-field soft cap)."""
        entry = self._store.get(handle)
        if entry is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}
        total = entry["total"]
        if index < 0 or index >= total:
            return {"success": False, "error": f"Index {index} out of range [0, {total-1}]."}
        rec = entry["records"][index]
        if isinstance(rec, dict):
            # Truncate per-field to keep the result readable
            display = {k: _truncate(str(v)) for k, v in rec.items()}
        else:
            display = _truncate(str(rec))
        return {
            "success": True,
            "handle":  handle,
            "index":   index,
            "record":  display,
            "total":   total,
        }

    def list_records(
        self,
        handle: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Paginated one-line-per-record listing."""
        entry = self._store.get(handle)
        if entry is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}
        total = entry["total"]
        start = (page - 1) * page_size
        end   = min(start + page_size, total)
        if start >= total:
            return {
                "success":    False,
                "error":      f"Page {page} out of range (total {total} records).",
            }
        items = [
            {"index": i, "summary": _record_summary(entry["records"][i])}
            for i in range(start, end)
        ]
        return {
            "success":     True,
            "handle":      handle,
            "page":        page,
            "page_size":   page_size,
            "total":       total,
            "total_pages": (total + page_size - 1) // page_size,
            "items":       items,
        }

    # ── structured operations ─────────────────────────────────────────────────

    def filter_records(
        self,
        handle: str,
        field: str,
        op: str,
        value: Any,
        new_handle: str,
    ) -> Dict[str, Any]:
        """
        Filter structured records and save to *new_handle*.

        ops: eq  ne  gt  lt  gte  lte  contains  startswith  regex
        """
        entry = self._store.get(handle)
        if entry is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}

        results = []
        op = op.lower()
        for rec in entry["records"]:
            if not isinstance(rec, dict):
                continue
            cell = rec.get(field)
            if cell is None:
                continue
            cell_s = str(cell).lower()
            val_s  = str(value).lower()
            try:
                cell_n = float(cell)
                val_n  = float(value)
                num_ok = True
            except (ValueError, TypeError):
                cell_n = val_n = 0.0
                num_ok = False

            keep = False
            if op == "eq":
                keep = cell_s == val_s
            elif op == "ne":
                keep = cell_s != val_s
            elif op in ("gt", ">"):
                keep = num_ok and cell_n > val_n
            elif op in ("lt", "<"):
                keep = num_ok and cell_n < val_n
            elif op in ("gte", ">="):
                keep = num_ok and cell_n >= val_n
            elif op in ("lte", "<="):
                keep = num_ok and cell_n <= val_n
            elif op == "contains":
                keep = val_s in cell_s
            elif op == "startswith":
                keep = cell_s.startswith(val_s)
            elif op == "regex":
                try:
                    keep = bool(re.search(str(value), str(cell), re.IGNORECASE))
                except re.error:
                    keep = False
            if keep:
                results.append(rec)

        # Store the filtered result
        filtered_content = "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
        self.store(new_handle, filtered_content if filtered_content else "[]")

        return {
            "success":         True,
            "source_handle":   handle,
            "new_handle":      new_handle,
            "filter":          f"{field} {op} {value!r}",
            "matched":         len(results),
            "total_checked":   entry["total"],
        }

    def aggregate(
        self,
        handle: str,
        operation: str,
        field: str,
    ) -> Dict[str, Any]:
        """
        Aggregate over a field.

        operations: count  sum  min  max  avg  unique  unique_count
        """
        entry = self._store.get(handle)
        if entry is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}

        op = operation.lower()
        values = []
        for rec in entry["records"]:
            if isinstance(rec, dict) and field in rec:
                values.append(rec[field])

        if not values and op not in ("count",):
            return {
                "success": False,
                "error":   f"Field '{field}' not found in any record of '{handle}'.",
            }

        try:
            if op == "count":
                result = len(values)
            elif op == "sum":
                result = sum(float(v) for v in values)
            elif op == "min":
                try:
                    result = min(float(v) for v in values)
                except ValueError:
                    result = min(str(v) for v in values)
            elif op == "max":
                try:
                    result = max(float(v) for v in values)
                except ValueError:
                    result = max(str(v) for v in values)
            elif op == "avg":
                nums = [float(v) for v in values]
                result = sum(nums) / len(nums) if nums else 0.0
            elif op in ("unique", "unique_count"):
                unique = sorted(set(str(v) for v in values))
                if op == "unique_count":
                    result = len(unique)
                else:
                    result = unique[:100]  # cap at 100 for context safety
            else:
                return {"success": False, "error": f"Unknown operation '{operation}'."}
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {
            "success":   True,
            "handle":    handle,
            "field":     field,
            "operation": op,
            "result":    result,
            "value_count": len(values),
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def to_artefact_content(self, handle: str) -> Optional[Tuple[str, str]]:
        """Return (content_str, detected_format) or None if handle not found."""
        entry = self._store.get(handle)
        if entry is None:
            return None
        fmt = entry["format"]
        records = entry["records"]

        if fmt in (_FMT_JSON_ARRAY, _FMT_JSONL):
            content = json.dumps(records, indent=2, ensure_ascii=False)
        elif fmt == _FMT_CSV:
            if records and isinstance(records[0], dict):
                out = io.StringIO()
                w = csv.DictWriter(out, fieldnames=list(records[0].keys()))
                w.writeheader()
                w.writerows(records)
                content = out.getvalue()
            else:
                content = "\n".join(str(r) for r in records)
        else:
            content = entry["raw_content"]

        return content, fmt


# ── tool registration ─────────────────────────────────────────────────────────

def register_repl_tools(
    tool_registry: Dict[str, Any],
    tool_descriptions: List[str],
    buffer: "TextBuffer",
    artefacts_manager: Any,
) -> None:
    """
    Register all 8 REPL text tools into *tool_registry* and append their
    compact descriptions to *tool_descriptions*.

    Parameters
    ----------
    tool_registry       : the chat() tool_registry dict
    tool_descriptions   : the chat() tool_descriptions list
    buffer              : a TextBuffer instance (one per chat() call)
    artefacts_manager   : self.artefacts (for text_to_artefact)
    """

    # ── 1. text_store ─────────────────────────────────────────────────────────
    def _text_store(handle: str, content: str) -> Dict[str, Any]:
        """
        Ingest a large text blob into a named in-session buffer.
        Auto-detects format (JSON array, JSONL, CSV, Markdown table,
        numbered list, section-based doc, or raw lines) and builds an index.
        Returns a compact summary — NOT the full content — so the context
        stays small.
        Call this immediately after receiving any large tool output.
        """
        if not handle or not handle.strip():
            return {"success": False, "error": "handle must not be empty."}
        if not content or not content.strip():
            return {"success": False, "error": "content must not be empty."}
        return buffer.store(handle.strip(), content)

    tool_registry["text_store"] = _text_store
    tool_descriptions.append(
        "- text_store(handle: str, content: str): "
        "Ingest a large text payload into a named in-session buffer; "
        "auto-detects format (JSON array / JSONL / CSV / Markdown table / "
        "numbered list / sections / raw lines) and returns a compact summary. "
        "Call this first on any large MCP tool output before further processing."
    )

    # ── 2. text_search ────────────────────────────────────────────────────────
    def _text_search(
        handle: str,
        query: str,
        max_results: int = 10,
        field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Keyword / regex search across all records in a buffer.
        If *field* is given (for structured formats), searches only that field.
        Returns index positions, one-line summaries, and snippets.
        """
        return buffer.search(handle, query, max_results=max_results, field=field)

    tool_registry["text_search"] = _text_search
    tool_descriptions.append(
        "- text_search(handle: str, query: str, max_results: int = 10, field: str = None): "
        "Keyword/regex search in a text buffer; returns matching record indices and snippets."
    )

    # ── 3. text_get_range ─────────────────────────────────────────────────────
    def _text_get_range(handle: str, start: int, end: int) -> Dict[str, Any]:
        """Return records[start..end] inclusive (0-based). Use to read contiguous blocks."""
        return buffer.get_range(handle, start, end)

    tool_registry["text_get_range"] = _text_get_range
    tool_descriptions.append(
        "- text_get_range(handle: str, start: int, end: int): "
        "Return records [start..end] inclusive (0-based) from a text buffer."
    )

    # ── 4. text_get_record ────────────────────────────────────────────────────
    def _text_get_record(handle: str, index: int) -> Dict[str, Any]:
        """Return one full record by 0-based index."""
        return buffer.get_record(handle, index)

    tool_registry["text_get_record"] = _text_get_record
    tool_descriptions.append(
        "- text_get_record(handle: str, index: int): "
        "Return one full record by 0-based index."
    )

    # ── 5. text_list_records ──────────────────────────────────────────────────
    def _text_list_records(
        handle: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Paginated one-line-per-record directory listing."""
        return buffer.list_records(handle, page=page, page_size=page_size)

    tool_registry["text_list_records"] = _text_list_records
    tool_descriptions.append(
        "- text_list_records(handle: str, page: int = 1, page_size: int = 20): "
        "Paginated listing of record summaries (one line each)."
    )

    # ── 6. text_filter ────────────────────────────────────────────────────────
    def _text_filter(
        handle: str,
        field: str,
        op: str,
        value: Any,
        new_handle: str,
    ) -> Dict[str, Any]:
        """
        Filter structured records where field <op> value.
        ops: eq  ne  gt  lt  gte  lte  contains  startswith  regex
        Saves the result to *new_handle* for further processing.
        """
        return buffer.filter_records(handle, field, op, value, new_handle)

    tool_registry["text_filter"] = _text_filter
    tool_descriptions.append(
        "- text_filter(handle: str, field: str, op: str, value: any, new_handle: str): "
        "Filter structured records where field <op> value "
        "(ops: eq ne gt lt gte lte contains startswith regex) and save to new_handle."
    )

    # ── 7. text_aggregate ────────────────────────────────────────────────────
    def _text_aggregate(
        handle: str,
        operation: str,
        field: str,
    ) -> Dict[str, Any]:
        """
        Aggregate over a field.
        operations: count  sum  min  max  avg  unique  unique_count
        """
        return buffer.aggregate(handle, operation, field)

    tool_registry["text_aggregate"] = _text_aggregate
    tool_descriptions.append(
        "- text_aggregate(handle: str, operation: str, field: str): "
        "Aggregate a field: count / sum / min / max / avg / unique / unique_count."
    )

    # ── 8. text_to_artefact ───────────────────────────────────────────────────
    def _text_to_artefact(
        handle: str,
        title: str,
        artefact_type: str = "document",
        language: str = "",
    ) -> Dict[str, Any]:
        """
        Persist a (possibly filtered) buffer as a discussion artefact.
        The format is preserved: JSON arrays stay JSON, CSV stays CSV, etc.
        """
        result = buffer.to_artefact_content(handle)
        if result is None:
            return {"success": False, "error": f"No buffer named '{handle}'."}
        content, fmt = result

        # map detected format to a reasonable artefact_type when caller omits
        _fmt_to_type = {
            _FMT_JSON_ARRAY: "json",
            _FMT_JSONL:      "json",
            _FMT_CSV:        "csv",
            _FMT_MD_TABLE:   "document",
            _FMT_NUMBERED:   "document",
            _FMT_SECTIONS:   "document",
            _FMT_LINES:      "document",
        }
        try:
            from ._artefacts import ArtefactType as _AT
            resolved_type = (
                artefact_type
                if artefact_type in _AT.ALL
                else _fmt_to_type.get(fmt, "document")
            )
        except ImportError:
            resolved_type = artefact_type or _fmt_to_type.get(fmt, "document")

        new_art = artefacts_manager.add(
            title         = title,
            artefact_type = resolved_type,
            content       = content,
            language      = language or None,
            active        = True,
        )

        entry = buffer.get(handle)
        return {
            "success":       True,
            "handle":        handle,
            "artefact_title": title,
            "artefact_type": resolved_type,
            "records_saved": entry["total"] if entry else 0,
            "artefact_id":   new_art.get("id"),
        }

    tool_registry["text_to_artefact"] = _text_to_artefact
    tool_descriptions.append(
        "- text_to_artefact(handle: str, title: str, artefact_type: str = 'document', "
        "language: str = ''): "
        "Save a text buffer (or its filtered subset) as a persistent discussion artefact."
    )

    # ── 9. text_list_buffers ─────────────────────────────────────────────────
    def _text_list_buffers() -> Dict[str, Any]:
        """List all active in-session text buffers and their record counts."""
        handles = buffer.list_handles()
        summary = []
        for h in handles:
            entry = buffer.get(h)
            if entry:
                summary.append({
                    "handle":  h,
                    "format":  entry["format"],
                    "records": entry["total"],
                })
        return {"success": True, "buffers": summary, "count": len(summary)}

    tool_registry["text_list_buffers"] = _text_list_buffers
    tool_descriptions.append(
        "- text_list_buffers(): List all active in-session text buffers with format and record count."
    )
