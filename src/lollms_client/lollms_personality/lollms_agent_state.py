from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import sys
import uuid
import traceback
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

try:
    from lollms_client.lollms_types import MSG_TYPE, EventMode
except ImportError:
    class MSG_TYPE:
        MSG_TYPE_CHUNK = "chunk"
        MSG_TYPE_INFO = "info"
        MSG_TYPE_NEW_MESSAGE = "new_message"
        MSG_TYPE_THOUGHT_CHUNK = "thought"

try:
    from lollms_client.lollms_memory import FailureMemory
except ImportError:
    class FailureMemory:
        def __init__(self):
            self.failures = []
            self._signatures = set()
        def record_failure_by_signature(self, sig, error):
            self.failures.append({"signature": sig, "error": error})
            self._signatures.add(sig)


_BASE64_RE = re.compile(r'^[A-Za-z0-9+/=\s]{500,}$')
_BINARY_BLOB_KEYS = {
    "plot_b64", "image_b64", "audio_b64", "video_b64", "file_b64",
    "screenshot_b64", "pdf_b64", "thumbnail_b64", "base64",
    "binary", "raw_image", "image_data", "raw_data",
}
_MAX_TOOL_RESULT_CHARS = 4000

def _is_large_base64(v: str) -> bool:
    sample = v.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(sample) < 500:
        return False
    return bool(_BASE64_RE.match(sample[:1000]))

def _sanitize_tool_result(tool_res: Any, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> str:
    def _find_prompt_injection(obj: Any, depth: int = 0) -> Optional[str]:
        if depth > 4:
            return None
        if isinstance(obj, dict):
            pinj = obj.get("prompt_injection")
            if isinstance(pinj, str) and pinj.strip():
                return pinj.strip()
            for v in obj.values():
                hit = _find_prompt_injection(v, depth + 1)
                if hit:
                    return hit
        elif isinstance(obj, list):
            for v in obj:
                hit = _find_prompt_injection(v, depth + 1)
                if hit:
                    return hit
        return None

    def _walk(obj: Any, depth: int = 0) -> Any:
        if depth > 6:
            return "[truncated: depth limit]"
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            if _is_large_base64(obj):
                approx_kb = len(obj) * 3 / 4 / 1024
                return f"[base64 blob stripped: {approx_kb:.1f}KB]"
            if len(obj) > max_chars:
                return obj[:max_chars] + f"\n... [truncated, {len(obj) - max_chars} more chars]"
            return obj
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in _BINARY_BLOB_KEYS:
                    if isinstance(v, str) and v:
                        approx_kb = len(v) * 3 / 4 / 1024
                        cleaned[k] = f"[base64 blob stripped: {approx_kb:.1f}KB]"
                    else:
                        cleaned[k] = None
                else:
                    cleaned[k] = _walk(v, depth + 1)
            return cleaned
        if isinstance(obj, (list, tuple)):
            walked = [_walk(v, depth + 1) for v in obj[:50]]
            if len(obj) > 50:
                walked.append(f"... [truncated, {len(obj) - 50} more items]")
            return walked
        return str(obj)

    if isinstance(tool_res, dict) and tool_res.get("success") is False:
        error_msg = tool_res.get("error", "Unknown error")
        inner = tool_res.get("output")
        if isinstance(inner, dict):
            error_msg = inner.get("error", error_msg)
        return f"⚠ Tool Failed\nError: {error_msg}"

    pinj = _find_prompt_injection(tool_res)
    if pinj:
        success = True
        inner = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
        if isinstance(inner, dict):
            success = inner.get("success", True)
        if isinstance(tool_res, dict) and tool_res.get("success") is False:
            success = False
        success_status = "✓ Success" if success else "⚠ Tool Failed"
        error_msg = ""
        if not success and isinstance(tool_res, dict):
            error_msg = tool_res.get("error", "")
        if error_msg:
            return f"{success_status}\nError: {error_msg}\n{pinj}"
        return f"{success_status}\n{pinj}"

    unwrapped = tool_res
    if isinstance(tool_res, dict):
        if "output" in tool_res:
            unwrapped = tool_res["output"]
            if isinstance(unwrapped, dict):
                for key in ("content", "text", "result", "data", "page_content", "summary"):
                    if key in unwrapped:
                        unwrapped = unwrapped[key]
                        break
        elif "content" in tool_res:
            unwrapped = tool_res["content"]
        elif "result" in tool_res:
            unwrapped = tool_res["result"]
        elif "data" in tool_res:
            unwrapped = tool_res["data"]

    if unwrapped is None:
        return "Tool executed successfully but returned no output content."

    def _replace_none(obj):
        if obj is None:
            return "[No output returned by tool]"
        if isinstance(obj, dict):
            return {k: _replace_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace_none(v) for v in obj]
        return obj

    sanitized = _walk(_replace_none(unwrapped))
    
    if isinstance(tool_res, dict) and "files" in tool_res and tool_res["files"]:
        files_list = tool_res["files"]
        if isinstance(files_list, list):
            files_str = "\n".join(files_list[:50])
            if isinstance(sanitized, str):
                sanitized = sanitized + f"\nMatching Files:\n{files_str}"
            else:
                try:
                    sanitized_str = json.dumps(sanitized, indent=2, default=str, ensure_ascii=False)
                    sanitized = sanitized_str + f"\nMatching Files:\n{files_str}"
                except Exception:
                    pass

    if isinstance(sanitized, str):
        if len(sanitized) > max_chars:
            return sanitized[:max_chars] + f"\n... [truncated, {len(sanitized) - max_chars} more chars]"
        return sanitized

    try:
        text = json.dumps(sanitized, indent=2, default=str, ensure_ascii=False)
    except Exception:
        text = str(sanitized)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
    return text

class _ToolsManager:
    SYSTEM_TOOLS_DIR = Path("app/tools")
    USER_TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"

    def __init__(self, extra_dirs: Optional[List[Union[str, Path]]] = None):
        self._extra_dirs: List[Path] = [Path(d) for d in (extra_dirs or [])]
        self._loaded_modules: Dict[str, ModuleType] = {}

    def list_available_files(self) -> List[Path]:
        files: set = set()
        for directory in [self.SYSTEM_TOOLS_DIR, self.USER_TOOLS_DIR] + self._extra_dirs:
            if directory.exists():
                for fp in directory.glob("*.py"):
                    if fp.name != "__init__.py":
                        files.add(fp.resolve())
        return sorted(files, key=lambda p: p.name.lower())

    def load_file(self, file_path: Union[str, Path]) -> ModuleType:
        fp = Path(file_path).resolve()
        key = str(fp)
        if key in self._loaded_modules:
            return self._loaded_modules[key]
        content = fp.read_text(encoding="utf-8")
        module_name = f"lollms_tools_{fp.stem}_{uuid.uuid4().hex[:8]}"
        module = ModuleType(module_name)
        module.__file__ = str(fp)
        exec(compile(content, str(fp), "exec"), module.__dict__)
        self._loaded_modules[key] = module
        return module

    def build_inline_tools_dict(self, sources: List[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        tools_dict: Dict[str, Dict[str, Any]] = {}
        for src in sources:
            if isinstance(src, dict):
                name = src.get("name", "unknown")
                tools_dict[name] = src
                continue
            fp = Path(src)
            if not fp.exists():
                continue
            module = self.load_file(fp)
            callables = {name: getattr(module, name) for name in dir(module) if name.startswith("tool_") and callable(getattr(module, name))}
            for tool_name, fn in callables.items():
                doc = (fn.__doc__ or "").strip()
                params: List[Dict[str, Any]] = []
                arg_pattern = re.compile(r'^\s*-\s+([\w_]+)\s*\(([\w_]+)(?:,\s*optional)?\):\s*(.*)', re.MULTILINE | re.IGNORECASE)
                for m in arg_pattern.finditer(doc):
                    pname, ptype, pdesc = m.groups()
                    is_optional = "optional" in m.group(0).lower()
                    p_entry: Dict[str, Any] = {"name": pname, "type": ptype.lower(), "description": pdesc.strip()}
                    if is_optional:
                        p_entry["optional"] = True
                    params.append(p_entry)
                tools_dict[tool_name] = {"name": tool_name, "callable": fn, "parameters": params, "description": doc.split('\n\n')[0].strip() if doc else f"Execute {tool_name}"}
        return tools_dict


class _AgentStreamState:
    def __init__(self, callback: Optional[Callable] = None, event_mode: int = 0):
        self.callback = callback
        self._event_mode = event_mode
        self.content = ""
        self.completed_actions: List[Dict[str, Any]] = []

        self._is_accumulating_tool = False
        self._is_accumulating_artifact = False

        self._is_accumulating_context = False
        self._context_tag_name = ""

        self._tool_buffer = ""
        self._pending_buffer = ""
        self._raw_stream_buffer = ""

        self._in_code_fence = False
        self._code_fence_buffer = ""
        self._code_fence_hold_buffer = ""
        self._in_inline_code = False

        self._in_think_block = False
        self._think_buffer = ""

        self.context_trigger = False
        self.artifact_trigger = False
        self.tool_trigger = False
        self.live_artifact_meta: Optional[Dict[str, Any]] = None
        self._done_intercepted: bool = False

    def _cb(self, text: str, msg_type=None, meta: Optional[Dict] = None):
        if self.callback is None:
            return
        try:
            mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
            self.callback(text, mt, meta or {})
        except Exception:
            pass

    def _check_exact_control_tag_match(self, buffer: str) -> int:
        if not buffer:
            return -1

        lines = buffer.splitlines(keepends=True)
        if not lines:
            return -1

        last_line = lines[-1].strip()
        if not last_line:
            return -1

        for tag in ["<done/>", "<done>"]:
            if last_line.lower() == tag:
                return len(buffer) - len(last_line)

        for tag in ["<tool>", "<artifact>", "<artefact>", "<refactor_history>"]:
            if last_line.lower() == tag:
                return len(buffer) - len(last_line)

        for tag in ["</tool>", "</artifact>", "</artefact>", "</refactor_history>", "</processing>"]:
            if last_line.lower() == tag:
                return len(buffer) - len(last_line)

        return -1

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str) or not chunk:
            return True

        self._raw_stream_buffer += chunk
        self._pending_buffer += chunk

        if "<think>" in self._pending_buffer and not self._in_think_block:
            idx = self._pending_buffer.find("<think>")
            text_before = self._pending_buffer[:idx]
            if text_before:
                self.content += text_before
                self._cb(text_before)
            self._pending_buffer = self._pending_buffer[idx + 7:]
            self._in_think_block = True
            self._think_buffer = ""

        if self._in_think_block:
            close_idx = self._pending_buffer.find("</think>")
            if close_idx != -1:
                self._think_buffer += self._pending_buffer[:close_idx]
                self._pending_buffer = self._pending_buffer[close_idx + 8:]
                self._in_think_block = False
            else:
                self._think_buffer += self._pending_buffer
                self._pending_buffer = ""
                return True

        if not self._is_accumulating_tool and not self._is_accumulating_artifact and not self._in_code_fence and not self._in_inline_code:
            done_match = re.search(r'(?m)^\s*<done\s*/?>', self._pending_buffer, re.IGNORECASE)
            if done_match:
                text_before = self._pending_buffer[:done_match.start()].strip()
                if text_before:
                    self.content += text_before
                    self._cb(text_before)

                self._pending_buffer = ""
                self._done_intercepted = True
                self._cb("", MSG_TYPE.MSG_TYPE_INFO, {"done_intercepted": True})
                return False

        if self._done_intercepted and (self._is_accumulating_artifact or self._is_accumulating_tool):
            if self._is_accumulating_artifact:
                self._tool_buffer += self._pending_buffer
                self._pending_buffer = ""
                self._try_complete_artifact()
            elif self._is_accumulating_tool:
                self._tool_buffer += self._pending_buffer
                self._pending_buffer = ""
                self._try_complete_tool()
            return True

        if not self._is_accumulating_tool and not self._is_accumulating_artifact and not self._in_code_fence and not self._in_inline_code:
            proc_match = re.search(r'(?m)^\s*<processing', self._pending_buffer, re.IGNORECASE)
            if proc_match:
                self._pending_buffer = re.sub(r'(?m)^\s*<processing[^>]*>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        if not self._is_accumulating_tool and not self._is_accumulating_artifact:
            if "```" in self._pending_buffer:
                self._code_fence_buffer += self._pending_buffer
                self._pending_buffer = ""

                while "```" in self._code_fence_buffer:
                    idx = self._code_fence_buffer.find("```")
                    before = self._code_fence_buffer[:idx]
                    self._code_fence_buffer = self._code_fence_buffer[idx + 3:]

                    if not self._in_code_fence:
                        self._in_code_fence = True
                        self.content += before + "```"
                        self._cb(before + "```")
                    else:
                        self._in_code_fence = False
                        if self._code_fence_hold_buffer:
                            self.content += self._code_fence_hold_buffer
                            self._cb(self._code_fence_hold_buffer)
                            self._code_fence_hold_buffer = ""
                        if before:
                            self.content += before
                            self._cb(before)
                        self.content += "```"
                        self._cb("```")

                if self._in_code_fence:
                    self._code_fence_hold_buffer += self._code_fence_buffer
                    self._code_fence_buffer = ""
                    return True
                else:
                    self._pending_buffer = self._code_fence_buffer
                    self._code_fence_buffer = ""
            elif self._in_code_fence:
                self._code_fence_hold_buffer += self._pending_buffer
                self._pending_buffer = ""
                return True

        if not self._is_accumulating_tool and not self._is_accumulating_artifact and not self._in_code_fence:
            if "`" in self._pending_buffer:
                if self._in_inline_code:
                    idx = self._pending_buffer.find("`")
                    if idx != -1:
                        self._in_inline_code = False
                        inline_content = self._pending_buffer[:idx]
                        self.content += inline_content + "`"
                        self._cb(inline_content + "`")
                        self._pending_buffer = self._pending_buffer[idx + 1:]
                    else:
                        self.content += self._pending_buffer
                        self._cb(self._pending_buffer)
                        self._pending_buffer = ""
                        return True
                else:
                    idx = self._pending_buffer.find("`")
                    before = self._pending_buffer[:idx]
                    remainder = self._pending_buffer[idx + 1:]
                    closing_idx = remainder.find("`")
                    if closing_idx != -1:
                        inline_content = remainder[:closing_idx]
                        self.content += before + "`" + inline_content + "`"
                        self._cb(before + "`" + inline_content + "`")
                        self._pending_buffer = remainder[closing_idx + 1:]
                    else:
                        self._in_inline_code = True
                        self.content += before + "`"
                        self._cb(before + "`")
                        self._pending_buffer = remainder
                        return True
            elif self._in_inline_code:
                idx = self._pending_buffer.find("`")
                if idx != -1:
                    self._in_inline_code = False
                    inline_content = self._pending_buffer[:idx]
                    self.content += inline_content + "`"
                    self._cb(inline_content + "`")
                    self._pending_buffer = self._pending_buffer[idx + 1:]
                else:
                    self.content += self._pending_buffer
                    self._cb(self._pending_buffer)
                    self._pending_buffer = ""
                    return True

        if self._is_accumulating_tool:
            self._tool_buffer += self._pending_buffer
            if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                self._cb(self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK, {"live_tool_chunk": True})
            self._pending_buffer = ""
            self._try_complete_tool()
            return True

        if self._is_accumulating_artifact:
            self._tool_buffer += self._pending_buffer

            if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                self._cb(self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK, {"live_artifact_chunk": True, "artifact_title": self.live_artifact_meta.get("title", "artifact") if self.live_artifact_meta else "artifact", "artifact_lang": self.live_artifact_meta.get("language", "") if self.live_artifact_meta else ""})
            elif self._event_mode == EventMode.PROCESSING_TAG_MODE:
                clean_chunk = self._pending_buffer
                if "<<<<<<< SEARCH" in clean_chunk:
                    clean_chunk = clean_chunk.replace("<<<<<<< SEARCH", "\n[🔍 SEARCH BLOCK]\n")
                if "=======" in clean_chunk:
                    clean_chunk = clean_chunk.replace("=======", "\n[✏️ REPLACE BLOCK]\n")
                if ">>>>>>> REPLACE" in clean_chunk:
                    clean_chunk = clean_chunk.replace(">>>>>>> REPLACE", "\n[✅ END REPLACE]\n")

                extracted_sections = []
                if self.live_artifact_meta and not self.live_artifact_meta.get("is_patch"):
                    lines_to_check = self._tool_buffer.splitlines()
                    for line in lines_to_check:
                        line_str = line.strip()
                        if not line_str:
                            continue

                        md_h = re.match(r'^(#{1,6})\s+(.+)', line_str)
                        if md_h:
                            extracted_sections.append({"type": "markdown_header", "name": md_h.group(2).strip(), "level": len(md_h.group(1))})
                        else:
                            py_c = re.match(r'^(?:class|def)\s+([A-Za-z0-9_]+)', line_str)
                            if py_c:
                                extracted_sections.append({"type": "python_symbol", "name": py_c.group(1), "symbol_type": py_c.group(0).split()[0]})
                            else:
                                js_c = re.match(r'^(?:class|function|def)\s+([A-Za-z0-9_]+)', line_str)
                                if js_c:
                                    extracted_sections.append({"type": "js_symbol", "name": js_c.group(1), "symbol_type": js_c.group(0).split()[0]})
                                else:
                                    generic_c = re.match(r'^(?:class|def|function)\s+([A-Za-z0-9_]+)', line_str, re.IGNORECASE)
                                    if generic_c:
                                        extracted_sections.append({"type": "symbol", "name": generic_c.group(1), "symbol_type": generic_c.group(0).split()[0]})

                last_reported_section = self.live_artifact_meta.get("last_reported_section") if self.live_artifact_meta else None
                current_last_section = extracted_sections[-1] if extracted_sections else None

                if current_last_section and current_last_section != last_reported_section:
                    self.live_artifact_meta["last_reported_section"] = current_last_section
                    status_icon = "📝" if current_last_section.get("type") == "markdown_header" else "🛠️"
                    status_msg = f"{status_icon} {current_last_section.get('symbol_type', 'Section').capitalize()}: {current_last_section.get('name')}\n"

                    self._cb(status_msg, MSG_TYPE.MSG_TYPE_CHUNK, {
                        "was_processed": True, 
                        "event_type": "artifact_chunk",
                        "artifact_title": self.live_artifact_meta.get("title", "artifact") if self.live_artifact_meta else "artifact",
                        "is_patch": self.live_artifact_meta.get("is_patch", False) if self.live_artifact_meta else False,
                        "sections": extracted_sections,
                        "live_artifact_chunk": True
                    })
                else:
                    self._cb(clean_chunk, MSG_TYPE.MSG_TYPE_CHUNK, {
                        "was_processed": True, 
                        "event_type": "artifact_chunk",
                        "artifact_title": self.live_artifact_meta.get("title", "artifact") if self.live_artifact_meta else "artifact",
                        "is_patch": self.live_artifact_meta.get("is_patch", False) if self.live_artifact_meta else False,
                        "sections": extracted_sections,
                        "live_artifact_chunk": True
                    })

            self._pending_buffer = ""
            self._try_complete_artifact()

            if self._is_accumulating_artifact and len(self._tool_buffer) > 500:
                attrs_match = re.search(r'<art(?:ifact|efact)[^>]*>', self._tool_buffer, re.IGNORECASE)
                if attrs_match:
                    attrs_str = attrs_match.group(0)
                    new_title = self.live_artifact_meta.get("title", "artifact") if self.live_artifact_meta else "artifact"
                    new_lang = self.live_artifact_meta.get("language", "") if self.live_artifact_meta else ""

                    updated = False
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                        if m.group(1).lower() in ("name", "title") and m.group(2) != new_title:
                            new_title = m.group(2)
                            updated = True
                        elif m.group(1).lower() == "language" and m.group(2) != new_lang:
                            new_lang = m.group(2)
                            updated = True

                    if updated and self.live_artifact_meta:
                        self.live_artifact_meta["title"] = new_title
                        self.live_artifact_meta["language"] = new_lang
                        try:
                            self._cb("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, self.live_artifact_meta)
                        except Exception:
                            pass
            return True

        if self._is_accumulating_context:
            self._tool_buffer += self._pending_buffer
            self._pending_buffer = ""
            self._try_complete_context_tag()
            return True

        if not self._in_code_fence and not self._in_inline_code:
            tool_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)<tool>', self._pending_buffer, re.IGNORECASE)
            if tool_match:
                tag_start_idx = tool_match.start()
                text_before = self._pending_buffer[:tag_start_idx]
                if text_before:
                    self.content += text_before
                    self._cb(text_before)

                self._is_accumulating_tool = True
                self.tool_trigger = True
                self._tool_buffer = self._pending_buffer[tag_start_idx:]
                self._pending_buffer = ""

                if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                    self._cb("<tool>", MSG_TYPE.MSG_TYPE_TOOL_START, {"tool_name": "pending", "parameters": {}})
                
                if self._event_mode == EventMode.PROCESSING_TAG_MODE:
                    self._cb('\n<processing type="tool" title="pending">\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                self._try_complete_tool()
                return True

            artifact_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)<art(?:ifact|efact)\b', self._pending_buffer, re.IGNORECASE)
            if artifact_match:
                tag_start_idx = artifact_match.start()
                partial_tag_buffer = self._pending_buffer[tag_start_idx:]

                full_tag_match = re.search(r'<art(?:ifact|efact)[^>]*>', partial_tag_buffer, re.IGNORECASE)
                if not full_tag_match:
                    text_before = self._pending_buffer[:tag_start_idx]
                    if text_before:
                        self.content += text_before
                        self._cb(text_before)
                    self._pending_buffer = partial_tag_buffer
                    return True

                text_before = self._pending_buffer[:tag_start_idx]
                if text_before:
                    self.content += text_before
                    self._cb(text_before)

                self._is_accumulating_artifact = True
                self.artifact_trigger = True
                self._tool_buffer = partial_tag_buffer
                self._pending_buffer = ""

                attrs_str = full_tag_match.group(0)
                title = "artifact"
                lang = "python"
                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                    if m.group(1).lower() in ("name", "title"):
                        title = m.group(2)
                    elif m.group(1).lower() == "language":
                        lang = m.group(2)

                parsed_art_type = "code"
                type_match = re.search(r'type=["\']([^"\']*)["\']', self._tool_buffer, re.IGNORECASE)
                if type_match:
                    parsed_art_type = type_match.group(1)

                is_patch_start = "<<<<<<< SEARCH" in self._tool_buffer
                operation_type = "patch" if is_patch_start else "full_rewrite"

                self.live_artifact_meta = {
                    "title": title, 
                    "art_type": parsed_art_type, 
                    "language": lang, 
                    "is_patch": is_patch_start,
                    "operation": operation_type
                }

                if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                    try:
                        self._cb("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {
                            "title": title,
                            "art_type": parsed_art_type,
                            "language": lang,
                            "is_patch": is_patch_start,
                            "operation": operation_type,
                            "stream_complete": False
                        })
                    except Exception:
                        pass

                if self._event_mode == EventMode.PROCESSING_TAG_MODE:
                    op_icon = "🔧" if operation_type == "patch" else "✍️"
                    op_label = "Patching" if operation_type == "patch" else "Writing"
                    start_status = f'\n{op_icon} {op_label} {parsed_art_type} artifact: {title} (operation: {operation_type})...\n'
                    self.ai_message_content_buffer += start_status if hasattr(self, 'ai_message_content_buffer') else start_status

                    self._cb(start_status, MSG_TYPE.MSG_TYPE_CHUNK, {
                        "was_processed": True,
                        "event_type": "artifact_start",
                        "artifact_title": title,
                        "art_type": parsed_art_type,
                        "is_patch": is_patch_start,
                        "operation": operation_type
                    })

                self._try_complete_artifact()
                return True

            refactor_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)<refactor_history\b', self._pending_buffer, re.IGNORECASE)
            if refactor_match:
                tag_start_idx = refactor_match.start()
                text_before = self._pending_buffer[:tag_start_idx]
                if text_before:
                    self.content += text_before
                    self._cb(text_before)

                self._is_accumulating_context = True
                self._context_tag_name = "refactor_history"
                self._tool_buffer = self._pending_buffer[tag_start_idx:]
                self._pending_buffer = ""

                if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                    self._cb("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": "refactor_history", "files": [], "status": "streaming"})

                if self._event_mode == EventMode.PROCESSING_TAG_MODE:
                    self._cb(f'\n<processing type="context" title="refactor_history">\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                self._try_complete_context_tag()
                return True

            context_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)<(unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder|scratchpad_append|scratchpad_patch|scratchpad_clear|user_profile_update|user_profile_clear|mem_new|mem_update)\b', self._pending_buffer, re.IGNORECASE)
            if context_match:
                tag_start_idx = context_match.start()
                tag_name = context_match.group(1).lower()
                text_before = self._pending_buffer[:tag_start_idx]
                if text_before:
                    self.content += text_before
                    self._cb(text_before)

                self._is_accumulating_context = True
                self.context_trigger = True
                self._context_tag_name = tag_name
                self._tool_buffer = self._pending_buffer[tag_start_idx:]
                self._pending_buffer = ""

                if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                    self._cb("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "streaming"})

                if self._event_mode == EventMode.PROCESSING_TAG_MODE:
                    self._cb(f'\n<processing type="context" title="{tag_name}">\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                self._try_complete_context_tag()
                return True

        def _ends_with_partial_tag(buffer: str) -> int:
            tags_to_check = ["<tool", "<done", "<artifact", "<artefact", "<unlock_file", "<lock_file", "<hide_file", "<refactor_history", "<collapse_folder", "<uncollapse_folder", "<scratchpad_append", "<scratchpad_patch", "<scratchpad_clear", "<user_profile_update", "<user_profile_clear", "<mem_new", "<mem_update", "<think"]
            for tag in tags_to_check:
                for i in range(1, len(tag)):
                    if buffer.endswith(tag[:i]):
                        start_idx = len(buffer) - i
                        j = start_idx - 1
                        while j >= 0 and buffer[j] != '\n':
                            if not buffer[j].isspace():
                                return -1
                            j -= 1
                        return start_idx
            return -1

        def _ends_with_partial_tag_anywhere(buffer: str) -> int:
            tags_to_check = ["<tool", "<done", "<artifact", "<artefact", "<unlock_file", "<lock_file", "<hide_file", "<refactor_history", "<collapse_folder", "<uncollapse_folder", "<scratchpad_append", "<scratchpad_patch", "<scratchpad_clear", "<user_profile_update", "<user_profile_clear", "<mem_new", "<mem_update", "<think"]
            for tag in tags_to_check:
                for i in range(1, len(tag)):
                    if buffer.endswith(tag[:i]):
                        return len(buffer) - i
            return -1

        exact_match_idx = self._check_exact_control_tag_match(self._pending_buffer)
        if exact_match_idx != -1:
            partial_idx = -1
        else:
            partial_idx = _ends_with_partial_tag(self._pending_buffer)
            if partial_idx == -1:
                partial_idx = _ends_with_partial_tag_anywhere(self._pending_buffer)

        if partial_idx != -1:
            text_before = self._pending_buffer[:partial_idx]
            if text_before:
                self.content += text_before
                self._cb(text_before)
            self._pending_buffer = self._pending_buffer[partial_idx:]
            return True

        self.content += self._pending_buffer
        self._cb(self._pending_buffer)
        self._pending_buffer = ""
        return True


    def _try_complete_tool(self) -> None:
        close_match = re.search(r'</tool>\s*', self._tool_buffer, re.IGNORECASE)
        if not close_match:
            return

        end_idx = close_match.start()
        end_len = len(close_match.group(0))

        full_tool_call = self._tool_buffer[:end_idx + end_len]
        json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE)
        json_body = re.sub(r'</tool>\s*$', '', json_body, flags=re.IGNORECASE).strip()

        self._is_accumulating_tool = False
        remaining = self._tool_buffer[end_idx + end_len:]
        self._tool_buffer = ""
        self._code_fence_hold_buffer = ""

        if remaining:
            self._pending_buffer = remaining

        raw_data = None
        resolved_tool_name = "pending"
        resolved_params = {}
        try:
            raw_data = json.loads(json_body)
            if isinstance(raw_data, dict):
                resolved_tool_name = raw_data.get("name", "malformed_tool_call")
                resolved_params = raw_data.get("parameters", {})
        except json.JSONDecodeError:
            repaired = json_body
            while repaired.count('{') > repaired.count('}'):
                repaired += '}'
            while repaired.count('[') > repaired.count(']'):
                repaired += ']'
            try:
                raw_data = json.loads(repaired)
                json_body = repaired
                if isinstance(raw_data, dict):
                    resolved_tool_name = raw_data.get("name", "malformed_tool_call")
                    resolved_params = raw_data.get("parameters", {})
            except json.JSONDecodeError:
                raw_data = None
                resolved_tool_name = "malformed_tool_call"

        if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
            try:
                self._cb("", MSG_TYPE.MSG_TYPE_TOOL_START, {"tool_name": resolved_tool_name, "parameters": resolved_params, "stream_complete": True})
            except Exception:
                pass

        if self._event_mode == EventMode.PROCESSING_TAG_MODE:
            self._cb('\n</processing>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

        normalized_json = json_body
        if isinstance(raw_data, dict):
            if "parameters" in raw_data and isinstance(raw_data["parameters"], dict):
                if not raw_data.get("name"):
                    params_dict = raw_data["parameters"]
                    nested_name = params_dict.get("tool_name") or params_dict.get("name")
                    if nested_name:
                        params_cleaned = {k: v for k, v in params_dict.items() if k not in ("tool_name", "name")}
                        normalized = {"name": nested_name, "parameters": params_cleaned}
                        normalized_json = json.dumps(normalized)
            else:
                tool_name = raw_data.get("name", "")
                params = {k: v for k, v in raw_data.items() if k != "name"}
                normalized = {"name": tool_name, "parameters": params}
                normalized_json = json.dumps(normalized)
        else:
            normalized_json = json.dumps({
                "name": "malformed_tool_call",
                "parameters": {"raw_body": json_body[:500]}
            })

        self.completed_actions.append({"type": "tool", "json": normalized_json})

    def _try_complete_artifact(self) -> None:
        close_match = re.search(r'</art(?:ifact|efact)>\s*', self._tool_buffer, re.IGNORECASE)
        if not close_match:
            return

        end_idx = close_match.start()
        end_len = len(close_match.group(0))

        full_artifact_call = self._tool_buffer[:end_idx + end_len]

        self._is_accumulating_artifact = False
        remaining = self._tool_buffer[end_idx + end_len:]
        self._tool_buffer = ""
        self._code_fence_hold_buffer = ""

        if remaining:
            self._pending_buffer = remaining + self._pending_buffer

        if self.live_artifact_meta:
            attrs_match = re.search(r'<art(?:ifact|efact)[^>]*>', full_artifact_call, re.IGNORECASE)
            if attrs_match:
                attrs_str = attrs_match.group(0)
                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                    if m.group(1).lower() in ("name", "title"):
                        self.live_artifact_meta["title"] = m.group(2)
                    elif m.group(1).lower() == "language":
                        self.live_artifact_meta["language"] = m.group(2)

        if self.live_artifact_meta:
            is_patch_stream = "<<<<<<< SEARCH" in full_artifact_call
            self.live_artifact_meta["is_patch"] = is_patch_stream
            self.live_artifact_meta["operation"] = "patch" if is_patch_stream else "full_rewrite"

            type_match_end = re.search(r'type=["\']([^"\']*)["\']', full_artifact_call, re.IGNORECASE)
            if type_match_end:
                self.live_artifact_meta["art_type"] = type_match_end.group(1)

        art_type_end = self.live_artifact_meta.get("art_type", "code") if self.live_artifact_meta else "code"
        title_end = self.live_artifact_meta.get("title", "artifact") if self.live_artifact_meta else "artifact"
        is_patch_end = self.live_artifact_meta.get("is_patch", False) if self.live_artifact_meta else False
        operation_end = self.live_artifact_meta.get("operation", "full_rewrite") if self.live_artifact_meta else "full_rewrite"

        if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
            try:
                self._cb("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {
                    "title": title_end,
                    "art_type": art_type_end,
                    "success": True,
                    "error": None,
                    "stream_complete": True,
                    "is_patch": is_patch_end,
                    "operation": operation_end
                })
            except Exception:
                pass

        if self._event_mode == EventMode.PROCESSING_TAG_MODE:
            self._cb('\n</processing>\n', MSG_TYPE.MSG_TYPE_CHUNK, {
                "was_processed": True,
                "event_type": "artifact_complete",
                "artifact_title": title_end,
                "art_type": art_type_end,
                "is_patch": is_patch_end,
                "operation": operation_end
            })

        self.completed_actions.append({"type": "artifact", "xml": full_artifact_call})

        if self.live_artifact_meta:
            self.live_artifact_meta = None

        self._code_fence_hold_buffer = ""
        
         

    def _try_complete_context_tag(self) -> None:
        closing_tag = f"</{self._context_tag_name}>"
        close_match = re.search(re.escape(closing_tag) + r'\s*', self._tool_buffer, re.IGNORECASE)
        if not close_match:
            return

        end_idx = close_match.start()
        end_len = len(close_match.group(0))

        full_tag_call = self._tool_buffer[:end_idx + end_len]

        self._is_accumulating_context = False
        remaining = self._tool_buffer[end_idx + end_len:]
        self._tool_buffer = ""
        self._code_fence_hold_buffer = ""

        if remaining:
            self._pending_buffer = remaining

        if self._event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
            try:
                self._cb("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": self._context_tag_name, "files": [], "status": "stream_complete"})
            except Exception:
                pass

        if self._event_mode == EventMode.PROCESSING_TAG_MODE:
            self._cb('\n</processing>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

        self.completed_actions.append({"type": "context", "tag_name": self._context_tag_name, "xml": full_tag_call})
        
         
        
    def flush_remaining_buffer(self):
        if self._in_code_fence:
            self._in_code_fence = False
            hold = self._code_fence_hold_buffer
            self._code_fence_hold_buffer = ""
            if hold:
                self.feed(hold)

        if self._is_accumulating_tool:
            self._tool_buffer += self._pending_buffer
            self._pending_buffer = ""
            self._try_complete_tool()
            if self._is_accumulating_tool:
                self._is_accumulating_tool = False
            return

        if self._is_accumulating_artifact:
            self._tool_buffer += self._pending_buffer
            self._pending_buffer = ""
            self._try_complete_artifact()

            if self._is_accumulating_artifact:
                self._is_accumulating_artifact = False

                if "<artifact" in self._tool_buffer.lower() or "<artefact" in self._tool_buffer.lower():
                    content_match = re.search(r'<art(?:ifact|efact)[^>]*>(.*)', self._tool_buffer, re.DOTALL | re.IGNORECASE)
                    if content_match:
                        body_content = content_match.group(1).strip()
                    else:
                        body_content = self._tool_buffer

                    if body_content:
                        full_artifact_call = self._tool_buffer
                        if "</artifact" not in full_artifact_call.lower() and "</artefact" not in full_artifact_call.lower():
                            full_artifact_call += "\n</artifact>"

                        self.completed_actions.append({"type": "artifact", "xml": full_artifact_call, "was_truncated": True})
            return

        if self._is_accumulating_context:
            self._tool_buffer += self._pending_buffer
            self._pending_buffer = ""
            self._try_complete_context_tag()
            if self._is_accumulating_context:
                self._is_accumulating_context = False
            return

        if self._pending_buffer:
            cleaned_buffer = re.sub(r'<done\s*/?>\s*$', '', self._pending_buffer, flags=re.IGNORECASE).strip()
            if cleaned_buffer:
                self.content += cleaned_buffer
                self._cb(cleaned_buffer)
            self._pending_buffer = ""
                
                 
             
    def was_done_detected(self) -> bool:
        return self._done_intercepted

    def get_clean_text(self) -> str:
        return self.content 
    
     
