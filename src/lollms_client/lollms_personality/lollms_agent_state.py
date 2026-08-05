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
    from lollms_client.lollms_types import MSG_TYPE
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
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.content = ""
        self.tool_trigger = False
        self.tool_json_data = ""
        self._done_detected = False
        self._action_dispatched = False

        self._is_accumulating_tool = False
        self._tool_buffer = ""
        self._pending_buffer = ""

        self._in_code_fence = False
        self._code_fence_buffer = ""
        self._code_fence_hold_buffer = ""
        self._in_inline_code = False

    def _cb(self, text: str, msg_type=None, meta: Optional[Dict] = None):
        if self.callback is None:
            return
        try:
            mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
            self.callback(text, mt, meta or {})
        except Exception:
            pass

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str) or not chunk:
            return True

        if self._action_dispatched:
            self._pending_buffer += chunk
            return True

        self._pending_buffer += chunk

        if not self._is_accumulating_tool and not self._in_code_fence and not self._in_inline_code:
            done_match = re.search(r'(?m)^\s*<done\s*/?>', self._pending_buffer, re.IGNORECASE)
            if done_match:
                self._done_detected = True
                self._pending_buffer = re.sub(r'(?m)^\s*<done\s*/?>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        if not self._is_accumulating_tool and not self._in_code_fence and not self._in_inline_code:
            proc_match = re.search(r'(?m)^\s*<processing', self._pending_buffer, re.IGNORECASE)
            if proc_match:
                self._pending_buffer = re.sub(r'(?m)^\s*<processing[^>]*>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        if not self._is_accumulating_tool:
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

        if not self._is_accumulating_tool and not self._in_code_fence:
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
            self._pending_buffer = ""
            if self._try_complete_tool():
                return False
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
                self._tool_buffer = self._pending_buffer[tag_start_idx:]
                self._pending_buffer = ""
                if self._try_complete_tool():
                    return False
                return True

        def _ends_with_partial_tag(buffer: str) -> int:
            tags_to_check = ["<tool", "<done"]
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

        partial_idx = _ends_with_partial_tag(self._pending_buffer)
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

    def _try_complete_tool(self) -> bool:
        close_match = re.search(r'</tool>\s*', self._tool_buffer, re.IGNORECASE)
        if not close_match:
            return False
        end_idx = close_match.start()
        end_len = len(close_match.group(0))
        full_tool_call = self._tool_buffer[:end_idx + end_len]
        json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE)
        json_body = re.sub(r'</tool>\s*$', '', json_body, flags=re.IGNORECASE).strip()
        self._is_accumulating_tool = False
        remaining = self._tool_buffer[end_idx + end_len:]
        self._tool_buffer = ""
        if remaining:
            self._pending_buffer = remaining + self._pending_buffer

        try:
            raw_data = json.loads(json_body)
        except json.JSONDecodeError:
            repaired = json_body
            while repaired.count('{') > repaired.count('}'):
                repaired += '}'
            while repaired.count('[') > repaired.count(']'):
                repaired += ']'
            try:
                raw_data = json.loads(repaired)
                json_body = repaired
            except json.JSONDecodeError:
                raw_data = None

        if isinstance(raw_data, dict):
            if "parameters" in raw_data and isinstance(raw_data["parameters"], dict):
                if not raw_data.get("name"):
                    params_dict = raw_data["parameters"]
                    nested_name = params_dict.get("tool_name") or params_dict.get("name")
                    if nested_name:
                        params_cleaned = {k: v for k, v in params_dict.items() if k not in ("tool_name", "name")}
                        normalized = {"name": nested_name, "parameters": params_cleaned}
                        self.tool_json_data = json.dumps(normalized)
                        return True
                self.tool_json_data = json_body
            else:
                tool_name = raw_data.get("name", "")
                params = {k: v for k, v in raw_data.items() if k != "name"}
                normalized = {"name": tool_name, "parameters": params}
                self.tool_json_data = json.dumps(normalized)
        else:
            self.tool_json_data = json_body

        self.tool_trigger = True
        self._action_dispatched = True
        return True

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
            if not self._try_complete_tool():
                full_tool_call = self._tool_buffer
                json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE).strip()
                self._is_accumulating_tool = False
                self._tool_buffer = ""
                try:
                    raw_data = json.loads(json_body)
                except Exception:
                    raw_data = None
                if isinstance(raw_data, dict):
                    if "parameters" in raw_data and isinstance(raw_data["parameters"], dict):
                        self.tool_json_data = json_body
                    else:
                        tool_name = raw_data.get("name", "")
                        params = {k: v for k, v in raw_data.items() if k != "name"}
                        normalized = {"name": tool_name, "parameters": params}
                        self.tool_json_data = json.dumps(normalized)
                else:
                    self.tool_json_data = json_body
                self.tool_trigger = True
                self._action_dispatched = True
            return

        if self._pending_buffer:
            self.content += self._pending_buffer
            self._cb(self._pending_buffer)
            self._pending_buffer = ""

    def was_done_detected(self) -> bool:
        return self._done_detected

    def was_action_dispatched(self) -> bool:
        return self._action_dispatched

    def get_tool_call_json(self) -> Optional[str]:
        return self.tool_json_data if self.tool_trigger else None

    def get_clean_text(self) -> str:
        return self.content