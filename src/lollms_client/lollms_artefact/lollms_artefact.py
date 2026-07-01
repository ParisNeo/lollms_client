# lollms_artefact/lollms_artefact.py
# ArtefactType, ArtefactVisibility, ArtefactStatus, and ArtefactManager

import re
import uuid
import os
import ast
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from ascii_colors import ASCIIColors, trace_exception

class ArtefactVisibility:
    """Visibility tiers for context optimization."""
    HIDDEN          = "hidden"           # Level 0: Invisible to LLM
    TREE_LOCKED     = "tree_locked"      # Level 1: In tree, cannot be unlocked
    TREE_UNLOCKABLE = "tree_unlockable"  # Level 2: In tree, can be unlocked by LLM
    METADATA        = "metadata"         # Level 3: Visible with schema/signatures only
    FULL            = "full"             # Level 4: Fully loaded in context


class ArtefactStatus:
    """Status states for tracking lifecycle of artefacts."""
    DRAFTING = "drafting"    # Actively being written or composed
    STABLE   = "stable"      # Completed, reviewed, and finalized
    REVISING = "revising"    # Modified, undergoing surgical patches
    ERROR    = "error"       # Failed compilation, broken code, or schema mismatch


class ArtefactType:
    """Registry for supported or custom artifact categories."""
    FILE          = "file"
    SEARCH_RESULT = "search_result"
    NOTE          = "note"
    SKILL         = "skill"
    CODE          = "code"
    DOCUMENT      = "document"
    IMAGE         = "image"
    PRESENTATION  = "presentation"
    DATA          = "data"
    TOOL          = "tool"

    ALL = {FILE, SEARCH_RESULT, NOTE, SKILL, CODE, DOCUMENT, IMAGE, PRESENTATION, DATA, TOOL}

    LABELS = {
        FILE:          "File",
        SEARCH_RESULT: "Search Result",
        NOTE:          "Note",
        SKILL:         "Skill",
        CODE:          "Code",
        DOCUMENT:      "Document",
        IMAGE:         "Image",
        PRESENTATION:  "Presentation",
        DATA:          "Data Interface",
        TOOL:          "Tool"
    }

    @classmethod
    def register_custom_type(cls, type_name: str, label: Optional[str] = None):
        name = type_name.lower().strip()
        cls.ALL.add(name)
        if label:
            cls.LABELS[name] = label
        elif name not in cls.LABELS:
            cls.LABELS[name] = name.capitalize()


_IMAGE_ID_SEP = "::"

def make_image_id(artefact_title: str, index: int) -> str:
    return f"{artefact_title}{_IMAGE_ID_SEP}{index}"


def parse_image_id(image_id: str) -> Optional[Tuple[str, int]]:
    if _IMAGE_ID_SEP not in image_id:
        return None
    title, _, idx_str = image_id.rpartition(_IMAGE_ID_SEP)
    try:
        return title, int(idx_str)
    except ValueError:
        return None


_ARTEFACT_IMAGE_TAG_RE = re.compile(
    r'<artefact_image\s+id=["\']([^"\']+)["\'](?:\s*/?>|>)',
    re.IGNORECASE,
)


def _extract_code_signatures(content: str, language: Optional[str]) -> str:
    lang = (language or "").lower()
    if lang == "python" or "def " in content or "class " in content:
        try:
            tree = ast.parse(content)
            sigs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            args = [arg.arg for arg in child.args.args]
                            methods.append(f"    def {child.name}({', '.join(args)}): ...")
                    base_classes = [ast.unparse(base) for base in node.bases]
                    base_str = f"({', '.join(base_classes)})" if base_classes else ""
                    sigs.append(f"class {node.name}{base_str}:\n" + "\n".join(methods))
                elif isinstance(node, ast.FunctionDef) and not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if node in p.body):
                    args = [arg.arg for arg in node.args.args]
                    sigs.append(f"def {node.name}({', '.join(args)}): ...")
            if sigs:
                return "\n\n".join(sigs)
        except Exception:
            pass

    signatures = []
    lines = content.splitlines()
    patterns = [
        r'^\s*(async\s+)?(function|fn|func|class|interface|struct)\s+([a-zA-Z0-9_]+)\b',
        r'^\s*(void|int|float|double|char|bool|auto|std::\w+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{?',
        r'^\s*(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(async\s*)?\([^)]*\)\s*=>'
    ]
    compiled = [re.compile(p) for p in patterns]

    for line in lines:
        line_strip = line.strip()
        if not line_strip or line_strip.startswith(("//", "#", "/*", "*")):
            continue
        for rx in compiled:
            if rx.match(line_strip):
                clean_line = re.sub(r'\s*\{\s*$', '', line_strip)
                signatures.append(clean_line + " { ... }")
                break

    if signatures:
        return "\n".join(signatures)
    return f"// [Code Signature Extraction Unavailable: File contains {len(lines)} lines]"


_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".cc", ".cxx",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".sh", ".bash",
    ".zsh", ".fish", ".bat", ".ps1", ".sql", ".r", ".m", ".lua", ".ex", ".exs",
    ".erl", ".hs", ".clj", ".scala", ".dart", ".zig", ".nim", ".v",
}


def _extract_file_metadata_and_signature(art: Dict[str, Any]) -> str:
    content = art.get("content", "")
    atype = art.get("type", "document")
    title = art["title"]
    lang = art.get("language")

    size_chars = len(content)
    line_count = len(content.splitlines())

    summary = [
        f"### [Partial Metadata] {title}",
        f"- Format Type: {atype.upper()} {f'({lang})' if lang else ''}",
        f"- Size: {size_chars:,} characters | {line_count:,} lines"
    ]

    if atype == ArtefactType.DATA:
        schema_lines = []
        for line in content.splitlines()[:20]:
            schema_lines.append(line)
        if len(content.splitlines()) > 20:
            schema_lines.append("\n  ... [remaining schema lines truncated]")
        summary.append("\n".join(schema_lines))
        return "\n".join(summary)

    if lang == "html" or atype == "presentation":
        soup_headings = re.findall(r'<h[1-3][^>]*>(.*?)</h[1-3]>|<section\s+class="slide[^"]*"[^>]*data-notes="([^"]*)"', content, re.I)
        if soup_headings:
            summary.append("#### Structure Outline:")
            for h in soup_headings[:15]:
                h_text = h[0] or f"Slide with notes: {h[1][:40]}..."
                summary.append(f"  • {re.sub(r'<[^>]+>', '', h_text).strip()}")
            return "\n".join(summary)

    if atype in (ArtefactType.CODE, ArtefactType.TOOL) or (lang and f".{lang}" in _CODE_EXTENSIONS):
        summary.append("#### Abstract Class & Function Signatures:")
        summary.append("```" + (lang or "python"))
        summary.append(_extract_code_signatures(content, lang))
        summary.append("```")
        return "\n".join(summary)

    preview = []
    for line in content.splitlines()[:6]:
        preview.append(line)
    summary.append("#### Context Preview:")
    summary.append("\n".join(preview) + "\n  ... [remaining document content truncated]")
    return "\n".join(summary)


def _find_closest_line(needle: str, haystack: str) -> str:
    needle_s = needle.strip()
    if not needle_s:
        return ''
    best_line  = ''
    best_score = -1.0
    for line in haystack.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        common = sum(a == b for a, b in zip(needle_s, line_s))
        score  = common / max(len(needle_s), len(line_s))
        if score > best_score:
            best_score = score
            best_line  = line
    return best_line


def _title_similarity(a: str, b: str) -> float:
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'\.[a-z0-9]{1,6}$', '', s)
        s = re.sub(r'[\s\-_/\\]+', ' ', s)
        return s

    na, nb = _norm(a), _norm(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 0.85

    def bigrams(s):
        return {s[i:i+2] for i in range(len(s) - 1)}

    bg_a, bg_b = bigrams(na), bigrams(nb)
    if not bg_a or not bg_b:
        common = sum(x == y for x, y in zip(na, nb))
        return common / max(len(na), len(nb))

    overlap = len(bg_a & bg_b)
    return (2.0 * overlap) / (len(bg_a) + len(bg_b))


def _find_best_title_match(
    candidate: str,
    existing_titles: List[str],
    threshold: float = 0.60,
) -> Optional[str]:
    best_title: Optional[str] = None
    best_score: float         = -1.0
    for title in existing_titles:
        score = _title_similarity(candidate, title)
        if score > best_score:
            best_score = score
            best_title = title
    if best_score >= threshold:
        return best_title
    return None


class ArtefactManager:
    """Manages the lifecycle of typed artefacts inside a LollmsDiscussion."""

    def __init__(self, discussion):
        object.__setattr__(self, '_discussion', discussion)

    def _get_lam_content(self, art: Dict[str, Any]) -> str:
        """Retrieves the logical metadata (.lam) content of an artifact if it exists on disk."""
        title = art.get("title", "untitled")
        atype = art.get("type", "document")
        language = art.get("language")
        file_ext = art.get("file_ext") or Path(title).suffix

        if getattr(self._discussion, "workspace_data_path", None):
            meta_dir = Path(self._discussion.artefacts_metadata_path)
        else:
            base_workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")
            disc_id = self._discussion.id
            disc_ws_dir = base_workspace_dir / disc_id
            meta_dir = disc_ws_dir / "artefacts_metadata"

        clean_title = title.replace("workspace/", "").replace("data_workspace/", "").replace("/", "_").replace("\\", "_")
        filename = self._get_filename_with_ext(clean_title, atype, language, file_ext)
        name_part, ext_part = os.path.splitext(filename) if '.' in filename else (filename, "")

        art_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, clean_title))
        lam_filename = f"{name_part}.lam"
        lam_path = meta_dir / art_id / lam_filename

        if lam_path.exists():
            try:
                return lam_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                pass
        return art.get("content", "").strip()

    def _get_all_raw(self) -> List[Dict]:
        metadata = self._discussion.metadata or {}
        raw = metadata.get("_artefacts", [])
        now = datetime.utcnow().isoformat()
        dirty = False
        migrated = []
        for a in raw:
            fixed = a.copy()
            if "id"               not in fixed: fixed["id"]               = str(uuid.uuid4()); dirty = True
            if "type"             not in fixed: fixed["type"]             = ArtefactType.DOCUMENT; dirty = True
            if "title"            not in fixed: fixed["title"]            = "untitled"; dirty = True
            if "content"          not in fixed: fixed["content"]          = ""; dirty = True
            if "images"           not in fixed: fixed["images"]           = []; dirty = True
            if "image_media_types" not in fixed: fixed["image_media_types"] = []; dirty = True
            if "audios"           not in fixed: fixed["audios"]           = []; dirty = True
            if "videos"           not in fixed: fixed["videos"]           = []; dirty = True
            if "zip"              not in fixed: fixed["zip"]              = None; dirty = True
            if "language"         not in fixed: fixed["language"]         = None; dirty = True
            if "url"              not in fixed: fixed["url"]              = None; dirty = True
            if "tags"             not in fixed: fixed["tags"]             = []; dirty = True

            if "visibility" not in fixed:
                legacy_active = fixed.get("active", False)
                fixed["visibility"] = ArtefactVisibility.FULL if legacy_active else ArtefactVisibility.HIDDEN
                dirty = True

            if "status" not in fixed:
                fixed["status"] = ArtefactStatus.STABLE
                dirty = True

            if "active"           not in fixed: fixed["active"]           = (fixed["visibility"] == ArtefactVisibility.FULL); dirty = True
            if "version"          not in fixed: fixed["version"]          = 1; dirty = True
            if "created_at"       not in fixed: fixed["created_at"]       = now; dirty = True
            if "updated_at"       not in fixed: fixed["updated_at"]       = now; dirty = True
            if "commit_message"   not in fixed: fixed["commit_message"]   = None; dirty = True
            if "version_tags"     not in fixed: fixed["version_tags"]     = []; dirty = True
            migrated.append(fixed)
        if dirty:
            new_meta = (self._discussion.metadata or {}).copy()
            new_meta["_artefacts"] = migrated
            self._discussion.metadata = new_meta
            self._discussion.commit()
        return migrated

    def _get_filename_with_ext(self, title: str, atype: str, language: Optional[str] = None, file_ext: Optional[str] = None) -> str:
        if "." in title and len(title.split(".")[-1]) <= 5:
            return title
        ext = file_ext or ""
        if not ext:
            if language:
                lang_map = {
                    "python": ".py", "javascript": ".js", "typescript": ".ts",
                    "html": ".html", "css": ".css", "markdown": ".md",
                    "latex": ".tex", "tex": ".tex", "sparql": ".rq", "sql": ".sql"
                }
                ext = lang_map.get(language.lower(), "")
            if not ext:
                type_map = {
                    "code": ".py", "document": ".md", "note": ".txt",
                    "skill": ".md", "data": ".csv", "presentation": ".html",
                    "tool": ".py"
                }
                ext = type_map.get(atype, ".txt")
        return f"{title}{ext}"

    def _sync_to_disk_workspace(self, title: str, content: str, version: int, atype: str, language: Optional[str] = None, file_ext: Optional[str] = None, physical_data: Optional[bytes] = None, logical_content: Optional[str] = None):
        try:
            if getattr(self._discussion, "workspace_data_path", None):
                ws_data_dir = Path(self._discussion.workspace_data_path)
                meta_dir = Path(self._discussion.artefacts_metadata_path)
            else:
                base_workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")
                disc_id = self._discussion.id
                disc_ws_dir = base_workspace_dir / disc_id
                ws_data_dir = disc_ws_dir / "workspace_data"
                meta_dir = disc_ws_dir / "artefacts_metadata"

            ws_data_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            clean_title = title.replace("workspace/", "").replace("data_workspace/", "").replace("/", "_").replace("\\", "_")
            filename = self._get_filename_with_ext(clean_title, atype, language, file_ext)
            name_part, ext_part = os.path.splitext(filename) if '.' in filename else (filename, "")

            art_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, clean_title))
            art_meta_dir = meta_dir / art_id
            art_meta_dir.mkdir(parents=True, exist_ok=True)

            primary_path = ws_data_dir / filename
            versioned_path = art_meta_dir / f"{name_part}_v{version}{ext_part}"

            lam_filename = f"{name_part}.lam"
            lam_path = art_meta_dir / lam_filename

            wrote_physical = False
            if physical_data is not None:
                try:
                    primary_path.write_bytes(physical_data)
                    versioned_path.write_bytes(physical_data)
                    wrote_physical = True
                except Exception as e:
                    trace_exception(e)

            if not wrote_physical and isinstance(content, str) and atype not in (ArtefactType.IMAGE, ArtefactType.DATA):
                try:
                    primary_path.write_text(content, encoding="utf-8", errors="ignore")
                    versioned_path.write_text(content, encoding="utf-8", errors="ignore")
                    wrote_physical = True
                except Exception as e:
                    trace_exception(e)

            if logical_content:
                try:
                    lam_path.write_text(logical_content, encoding="utf-8", errors="ignore")
                except Exception as e:
                    trace_exception(e)
            elif wrote_physical and atype in (ArtefactType.DATA, ArtefactType.IMAGE):
                minimal_lam = f"# Artefact Metadata: {filename}\n- **Type**: {atype}\n- **Version**: {version}\n- **Physical Path**: workspace_data/{filename}\n\n"
                try:
                    lam_path.write_text(minimal_lam, encoding="utf-8", errors="ignore")
                except Exception:
                    pass

        except Exception as e:
            ASCIIColors.warning(f"Failed to sync artifact '{title}' to disk: {e}")

    def _save_all(self, artefacts: List[Dict]):
        new_meta = (self._discussion.metadata or {}).copy()
        new_meta["_artefacts"] = artefacts
        self._discussion.metadata = new_meta
        self._discussion.commit()

    def _all_latest_titles(self) -> List[str]:
        seen: Dict[str, int] = {}
        for a in self._get_all_raw():
            t, v = a.get('title', ''), a.get('version', 1)
            if t not in seen or v > seen[t]:
                seen[t] = v
        return list(seen.keys())

    def _register_tool_artefact(self, art: Dict[str, Any]):
        """Registers or updates a tool artefact in the active LCP binding."""

        # 🛡️ SECURITY GATE: Check if dynamic tool registration is explicitly allowed for this session.
        # Defaults to False (disabled) if the attribute was never set by the chat loop.
        if not getattr(self._discussion, 'allow_dynamic_tools', False):
            ASCIIColors.warning(f"[ArtefactManager] 🛡️ Dynamic tool registration is DISABLED. Tool artefact '{art.get('title')}' was not executed.")
            return

        lc = getattr(self._discussion, 'lollmsClient', None)
        if not lc or not getattr(lc, 'tools', None):
            return

        lcp_binding = lc.tools
        if not hasattr(lcp_binding, 'register_tool_from_code'):
            return

        title = art.get("title", "untitled")
        content = art.get("content", "")

        # If it's a tool artefact, parse and register it
        if art.get("type") == ArtefactType.TOOL and content:
            try:
                # Extract code from markdown if present
                code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
                code = code_match.group(1) if code_match else content

                # Unregister old version first
                lcp_binding.unregister_tools_by_prefix(title)

                # Register new version
                lcp_binding.register_tool_from_code(title, code)
                ASCIIColors.success(f"[ArtefactManager] 🛠️ Registered tool artefact '{title}' as executable LCP tool")
            except Exception as e:
                ASCIIColors.warning(f"[ArtefactManager] Failed to register tool artefact '{title}': {e}")

    def _unregister_tool_artefact(self, title: str):
        """Removes a tool artefact from the active LCP binding."""
        lc = getattr(self._discussion, 'lollmsClient', None)
        if not lc or not getattr(lc, 'tools', None):
            return
            
        lcp_binding = lc.tools
        if hasattr(lcp_binding, 'unregister_tools_by_prefix'):
            lcp_binding.unregister_tools_by_prefix(title)

    def add(
        self,
        title:             str,
        artefact_type:     str = ArtefactType.DOCUMENT,
        content:           str = "",
        images:            Optional[List[str]] = None,
        image_media_types: Optional[List[str]] = None,
        audios:            Optional[List[str]] = None,
        videos:            Optional[List[str]] = None,
        zip_content:       Optional[str] = None,
        language:          Optional[str] = None,
        url:               Optional[str] = None,
        tags:              Optional[List[str]] = None,
        version:           int = 1,
        active:            Optional[bool] = None,
        visibility:        Optional[str] = None,
        commit_message:    Optional[str] = None,
        version_tags:      Optional[List[str]] = None,
        physical_data:     Optional[bytes] = None,
        logical_content:   Optional[str] = None,
        **extra_data
    ) -> Dict[str, Any]:
        file_ext = extra_data.get("file_ext") or Path(title).suffix
        file_ext_clean = (file_ext or "").lower()
        if file_ext_clean == ".sql" or title.lower().endswith(".sql"):
            artefact_type = ArtefactType.CODE
            language = "sql"
        elif file_ext_clean in (".py", ".js", ".ts", ".html", ".css", ".sh", ".bash") or any(title.lower().endswith(ext) for ext in (".py", ".js", ".ts", ".html", ".css", ".sh", ".bash")):
            artefact_type = ArtefactType.CODE
            if not language:
                language = file_ext_clean.replace(".", "")

        if artefact_type not in ArtefactType.ALL:
            raise ValueError(f"Unknown artefact type '{artefact_type}'.")
        artefacts = self._get_all_raw()
        artefacts = [a for a in artefacts if not (a.get('title') == title and a.get('version') == version)]
        for a in artefacts:
            if a.get('title') == title:
                a['active'] = False
                a['visibility'] = ArtefactVisibility.HIDDEN

        resolved_visibility = visibility
        if not resolved_visibility:
            if active is not None:
                resolved_visibility = ArtefactVisibility.FULL if active else ArtefactVisibility.HIDDEN
            else:
                if artefact_type == ArtefactType.DATA:
                    resolved_visibility = ArtefactVisibility.FULL
                else:
                    resolved_visibility = ArtefactVisibility.HIDDEN

        resolved_status = extra_data.pop("status", ArtefactStatus.STABLE)

        imgs  = images or []
        mtypes = image_media_types or []
        if len(mtypes) < len(imgs):
            mtypes = mtypes + ["image/jpeg"] * (len(imgs) - len(mtypes))

        now = datetime.utcnow().isoformat()
        token_count = self._discussion.lollmsClient.count_tokens(content) if (self._discussion and self._discussion.lollmsClient) else len(content) // 4
        new_artefact: Dict[str, Any] = {
            "id":               str(uuid.uuid4()),
            "title":            title,
            "type":             artefact_type,
            "version":          version,
            "content":          content,
            "token_count":      token_count,
            "images":           imgs,
            "image_media_types": mtypes,
            "audios":           audios or [],
            "videos":           videos or [],
            "zip":              zip_content,
            "language":         language,
            "url":              url,
            "tags":             tags or [],
            "visibility":       resolved_visibility,
            "status":           resolved_status,
            "active":           (resolved_visibility == ArtefactVisibility.FULL),
            "created_at":       now,
            "updated_at":       now,
            "commit_message":   commit_message,
            "version_tags":     version_tags or [],
            **extra_data,
        }
        artefacts.append(new_artefact)
        self._save_all(artefacts)

        self._sync_to_disk_workspace(
            title, content, version, artefact_type, language, extra_data.get("file_ext"),
            physical_data=physical_data,
            logical_content=logical_content
        )
        
        # 🛠️ DYNAMIC TOOL REGISTRATION
        if artefact_type == ArtefactType.TOOL:
            self._register_tool_artefact(new_artefact)
            
        return new_artefact

    def get(self, title: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        candidates = [a for a in self._get_all_raw() if a.get('title') == title]
        if not candidates:
            return None
        if version is not None:
            for a in candidates:
                curr_v = a.get('version')
                if str(curr_v) == str(version):
                    return a
                try:
                    if int(float(str(curr_v).split('.')[0])) == int(version):
                        return a
                except Exception:
                    pass
            return next((a for a in candidates if a.get('version') == version), None)

        def _safe_sort_key(a):
            try:
                v = a.get('version', 0)
                if isinstance(v, str):
                    v = int(float(v.split('.')[0]))
                return int(v)
            except Exception:
                return 0

        return max(candidates, key=_safe_sort_key)

    def get_by_id(self, artefact_id: str) -> Optional[Dict[str, Any]]:
        return next((a for a in self._get_all_raw() if a.get('id') == artefact_id), None)

    def list(self, artefact_type: Optional[str] = None, active_only: bool = False, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        result = self._get_all_raw()
        if artefact_type:
            result = [a for a in result if a.get('type') == artefact_type]
        if active_only:
            result = [a for a in result if a.get('active', False)]
        if tags:
            tag_set = set(tags)
            result = [a for a in result if tag_set.issubset(set(a.get('tags', [])))]
        return result

    def update(
        self,
        title:             str,
        new_content:       Optional[str] = None,
        new_type:          Optional[str] = None,
        new_images:        Optional[List[str]] = None,
        new_image_media_types: Optional[List[str]] = None,
        new_tags:          Optional[List[str]] = None,
        language:          Optional[str] = None,
        url:               Optional[str] = None,
        new_title:         Optional[str] = None,
        bump_version:      bool = True,
        active:            Optional[bool] = None,
        visibility:        Optional[str] = None,
        commit_message:    Optional[str] = None,
        version_tags:      Optional[List[str]] = None,
        version:           Optional[int] = None,
        **extra_data
    ) -> Optional[Dict[str, Any]]:
        latest = self.get(title)
        if latest is None:
            raise ValueError(f"Cannot update non-existent artefact '{title}'.")

        new_type = new_type or extra_data.pop("artefact_type", None)
        new_version = version if version is not None else ((latest.get('version', 1) + 1) if bump_version else latest.get('version', 1))

        resolved_visibility = visibility or latest.get("visibility")
        if not resolved_visibility:
            resolved_visibility = ArtefactVisibility.FULL if active else (latest.get("visibility") or ArtefactVisibility.FULL)
        if active is not None:
            resolved_visibility = ArtefactVisibility.FULL if active else ArtefactVisibility.HIDDEN

        target_title = new_title if new_title else title
        internal_keys = {
            "id", "title", "type", "version", "content", "images", "image_media_types",
            "audios", "videos", "zip", "language", "url", "tags", "active", "visibility",
            "created_at", "updated_at", "artefact_type", "commit_message", "version_tags"
        }
        merged_extra = {k: v for k, v in latest.items() if k not in internal_keys}
        merged_extra.update(extra_data)

        if new_title and new_title != title:
            artefacts = self._get_all_raw()
            for a in artefacts:
                if a.get('title') == title:
                    a['active'] = False
            self._save_all(artefacts)
            try:
                workspace_dir = Path("./data_workspace")
                try:
                    from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                    if awd is not None:
                        workspace_dir = awd
                except ImportError:
                    pass
                old_filename = self._get_filename_with_ext(title, latest.get('type'), latest.get('language'), latest.get('file_ext'))
                old_path = workspace_dir / "discussions" / self._discussion.id / old_filename
                if old_path.exists():
                    old_path.unlink()
            except Exception as e:
                trace_exception(e)

        use_content = new_content if new_content is not None else latest.get('content', '')
        if new_title and new_title != title:
            use_content = use_content.replace(f'id="{title}{_IMAGE_ID_SEP}', f'id="{new_title}{_IMAGE_ID_SEP}').replace(f"id='{title}{_IMAGE_ID_SEP}", f"id='{new_title}{_IMAGE_ID_SEP}")

        use_images = new_images if new_images is not None else latest.get('images', [])
        use_mtypes = new_image_media_types if new_image_media_types is not None else latest.get('image_media_types', [])

        result = self.add(
            title             = target_title,
            artefact_type     = new_type if new_type is not None else latest.get('type', ArtefactType.DOCUMENT),
            content           = use_content,
            images            = use_images,
            image_media_types = use_mtypes,
            audios            = latest.get('audios', []),
            videos            = latest.get('videos', []),
            zip_content       = latest.get('zip'),
            language          = language if language is not None else latest.get('language'),
            url               = url if url is not None else latest.get('url'),
            tags              = new_tags if new_tags is not None else latest.get('tags', []),
            version           = new_version,
            visibility        = resolved_visibility,
            commit_message    = commit_message if commit_message is not None else latest.get('commit_message'),
            version_tags      = version_tags if version_tags is not None else latest.get('version_tags', []),
            **merged_extra,
        )
        
        # 🛠️ DYNAMIC TOOL REGISTRATION (Update)
        if result and result.get("type") == ArtefactType.TOOL:
            self._register_tool_artefact(result)
            
        return result

    def revert(self, title: str, target_version: Union[int, str]) -> Dict[str, Any]:
        latest = self.get(title)
        if latest is None:
            raise ValueError(f"Artefact '{title}' not found.")

        if isinstance(target_version, str) and str(target_version).strip().lower() in ("last", "previous", "prev"):
            curr_v = latest.get('version', 1)
            if curr_v <= 1:
                raise ValueError(f"Cannot revert '{title}': no previous version exists.")
            target_version = curr_v - 1
        else:
            try:
                target_version = int(target_version)
            except ValueError:
                raise ValueError(f"Invalid target version: {target_version}")

        target = self.get(title, target_version)
        if target is None:
            raise ValueError(f"Version {target_version} of artefact '{title}' not found.")
        new_version = (latest.get('version', 1) + 1)
        extra_keys = {k: v for k, v in target.items() if k not in [
            "id", "title", "type", "version", "content", "images", "image_media_types",
            "audios", "videos", "zip", "language", "url", "tags", "active",
            "created_at", "updated_at"
        ]}
        return self.add(
            title             = title,
            artefact_type     = target.get('type', ArtefactType.DOCUMENT),
            content           = target.get('content', ''),
            images            = target.get('images', []),
            image_media_types = target.get('image_media_types', []),
            audios            = target.get('audios', []),
            videos            = target.get('videos', []),
            zip_content       = target.get('zip'),
            language          = target.get('language'),
            url               = target.get('url'),
            tags              = target.get('tags', []),
            version           = new_version,
            active            = True,
            **extra_keys,
        )

    def rename(self, old_title: str, new_title: str, new_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        artefacts = self._get_all_raw()
        found = False
        for a in artefacts:
            if a.get('title') == old_title:
                a['title'] = new_title
                a['updated_at'] = datetime.utcnow().isoformat()
                if new_type is not None:
                    a['type'] = new_type
                content = a.get('content', '')
                if content:
                    a['content'] = content.replace(f'id="{old_title}{_IMAGE_ID_SEP}', f'id="{new_title}{_IMAGE_ID_SEP}').replace(f"id='{old_title}{_IMAGE_ID_SEP}", f"id='{new_title}{_IMAGE_ID_SEP}'")
                found = True

        if found:
            self._save_all(artefacts)
            return self.get(new_title)
        return None

    def get_version_history(self, title: str) -> List[Dict[str, Any]]:
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        if not all_versions:
            return []

        def _safe_sort_key(a):
            try:
                v = a.get('version', 0)
                if isinstance(v, str):
                    v = int(float(v.split('.')[0]))
                return int(v)
            except Exception:
                return 0

        all_versions.sort(key=_safe_sort_key)
        active_version = None
        for a in all_versions:
            if a.get('active', False):
                active_version = a.get('version')
                break
        result = []
        for a in all_versions:
            content = a.get('content', '')
            result.append({
                "version":        a.get('version', 1),
                "created_at":     a.get('created_at', ''),
                "updated_at":     a.get('updated_at', ''),
                "content_preview": content[:200] + ("…" if len(content) > 200 else ""),
                "size_chars":     len(content),
                "image_count":    len(a.get('images', [])),
                "is_active":      a.get('version') == active_version,
            })
        return result

    def diff_versions(self, title: str, version_a: int, version_b: int) -> Dict[str, Any]:
        art_a = self.get(title, version_a)
        art_b = self.get(title, version_b)
        if art_a is None:
            raise ValueError(f"Version {version_a} of '{title}' not found.")
        if art_b is None:
            raise ValueError(f"Version {version_b} of '{title}' not found.")

        lines_a = art_a.get('content', '').splitlines()
        lines_b = art_b.get('content', '').splitlines()

        set_a = set(lines_a)
        set_b = set(lines_b)

        added = sorted([l for l in lines_b if l not in set_a])
        removed = sorted([l for l in lines_a if l not in set_b])
        common = sorted([l for l in lines_a if l in set_b])

        import difflib
        diff_text = "\n".join(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"{title} v{version_a}",
            tofile=f"{title} v{version_b}",
            lineterm="",
        ))

        return {
            "title":         title,
            "version_a":     version_a,
            "version_b":     version_b,
            "unified_diff":  diff_text,
            "added_lines":   len(added),
            "removed_lines": len(removed),
            "common_lines":  len(common),
            "added_content": "\n".join(added),
            "removed_content": "\n".join(removed),
        }

    def squash_versions(
        self,
        title: str,
        keep_versions: Optional[List[int]] = None,
        keep_last_n: Optional[int] = None,
        target_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        if not all_versions:
            raise ValueError(f"Artefact '{title}' not found.")

        all_versions.sort(key=lambda a: a.get('version', 0))

        active_version = None
        for a in all_versions:
            if a.get('active', False):
                active_version = a.get('version')
                break

        to_delete: List[int] = []
        to_preserve: List[int] = []
        new_baseline: Optional[int] = None
        space_reclaimed = 0

        if target_version is not None:
            target = self.get(title, target_version)
            if target is None:
                raise ValueError(f"Target version {target_version} of '{title}' not found.")

            squashed_messages = []
            for a in all_versions:
                v = a.get('version', 0)
                msg = a.get('commit_message')
                if v != target_version and msg:
                    squashed_messages.append(f"v{v}: {msg}")

            for a in all_versions:
                v = a.get('version', 0)
                if v != target_version:
                    to_delete.append(v)
                    space_reclaimed += len(a.get('content', ''))
                else:
                    to_preserve.append(1)

            artefacts = self._get_all_raw()
            new_list = []
            for a in artefacts:
                if a.get('title') == title:
                    v = a.get('version', 0)
                    if v == target_version:
                        a['version'] = 1
                        a['updated_at'] = datetime.utcnow().isoformat()
                        if squashed_messages:
                            orig_msg = a.get('commit_message') or "Baseline commit"
                            a['commit_message'] = f"{orig_msg}\n\nSquashed history:\n" + "\n".join(squashed_messages)
                        new_list.append(a)
                    elif v in to_delete:
                        continue
                    else:
                        new_list.append(a)
                else:
                    new_list.append(a)

            self._save_all(new_list)
            return {
                "success":      True,
                "deleted":      to_delete,
                "preserved":    [1],
                "new_baseline": 1,
                "space_reclaimed_estimate": space_reclaimed,
            }

        elif keep_versions is not None:
            keep_set = set(keep_versions)
            if active_version is not None and active_version not in keep_set:
                keep_set.add(active_version)
            for a in all_versions:
                v = a.get('version', 0)
                if v in keep_set:
                    to_preserve.append(v)
                else:
                    to_delete.append(v)
                    space_reclaimed += len(a.get('content', ''))

        elif keep_last_n is not None:
            if keep_last_n < 1:
                raise ValueError("keep_last_n must be >= 1")
            sorted_versions = [a.get('version', 0) for a in all_versions]
            preserved_set = set(sorted_versions[-keep_last_n:])
            if active_version is not None and active_version not in preserved_set:
                preserved_set.add(active_version)
                if len(preserved_set) > keep_last_n:
                    sorted_preserved = sorted(preserved_set)
                    preserved_set = set(sorted_preserved[-keep_last_n:])
            for a in all_versions:
                v = a.get('version', 0)
                if v in preserved_set:
                    to_preserve.append(v)
                else:
                    to_delete.append(v)
                    space_reclaimed += len(a.get('content', ''))
        else:
            raise ValueError("One of keep_versions, keep_last_n, or target_version must be specified.")

        if not to_preserve:
            raise ValueError("Squash would delete all versions.")

        if to_delete:
            squashed_messages = []
            for v_num in sorted(to_delete):
                matching_version = next((a for a in all_versions if a.get('version') == v_num), None)
                if matching_version and matching_version.get('commit_message'):
                    squashed_messages.append(f"v{v_num}: {matching_version['commit_message']}")

            artefacts = self._get_all_raw()
            receiver_version = min(to_preserve) if to_preserve else None

            new_list = []
            for a in artefacts:
                if a.get('title') == title:
                    v = a.get('version', 0)
                    if v in to_delete:
                        continue
                    if v == receiver_version and squashed_messages:
                        orig_msg = a.get('commit_message') or "Preserved baseline"
                        a['commit_message'] = f"{orig_msg}\n\nSquashed history:\n" + "\n".join(squashed_messages)
                new_list.append(a)
            self._save_all(new_list)

        return {
            "success":      True,
            "deleted":      to_delete,
            "preserved":    to_preserve,
            "new_baseline": new_baseline,
            "space_reclaimed_estimate": space_reclaimed,
        }

    def cleanup_old_versions(
        self,
        title: str,
        keep_count: int = 5,
        min_age_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.squash_versions(title, keep_last_n=keep_count)

    def remove(self, title: str, version: Optional[int] = None) -> int:
        self._cleanup_data_artefact_files(title, version)
        art = self.get(title, version)
        if art:
            try:
                workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")
                try:
                    from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                    if awd is not None:
                        workspace_dir = awd
                except ImportError:
                    pass
                del_filename = self._get_filename_with_ext(title, art.get('type'), art.get('language'), art.get('file_ext'))
                del_path = workspace_dir / self._discussion.id / del_filename
                if del_path.exists():
                    del_path.unlink()
            except Exception as e:
                ASCIIColors.warning(f"Failed to delete working file: {e}")

        artefacts = self._get_all_raw()
        initial = len(artefacts)
        if version is None:
            artefacts = [a for a in artefacts if a.get('title') != title]
        else:
            artefacts = [a for a in artefacts if not (a.get('title') == title and a.get('version') == version)]
        removed = initial - len(artefacts)
        if removed:
            self._save_all(artefacts)
            # 🛠️ DYNAMIC TOOL UNREGISTRATION
            if art and art.get("type") == ArtefactType.TOOL:
                self._unregister_tool_artefact(title)
        return removed

    def _cleanup_data_artefact_files(self, title: str, version: Optional[int] = None):
        try:
            art = self.get(title, version) if version else self.get(title)
            if not art or art.get('type') != 'data':
                return

            workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")
            try:
                from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                if awd is not None:
                    workspace_dir = awd
            except ImportError:
                pass

            if not workspace_dir.exists():
                return

            file_ext = art.get('file_ext', '.csv')
            files_to_delete = []

            if version is None:
                all_versions = [a for a in self._get_all_raw() if a.get('title') == title and a.get('type') == 'data']
                for v_art in all_versions:
                    v_num = v_art.get('version', 1)
                    versioned_file = workspace_dir / self._discussion.id / f"{title}_v{v_num}{file_ext}"
                    if versioned_file.exists():
                        files_to_delete.append(versioned_file)
                    if v_num == 1:
                        bundled_db = workspace_dir / self._discussion.id / f"{title}_consolidated.db"
                        if bundled_db.exists():
                            files_to_delete.append(bundled_db)
            else:
                if art.get('type') == 'data':
                    versioned_file = workspace_dir / self._discussion.id / f"{title}_v{version}{file_ext}"
                    if versioned_file.exists():
                        files_to_delete.append(versioned_file)
                    if version == 1:
                        bundled_db = workspace_dir / self._discussion.id / f"{title}_consolidated.db"
                        if bundled_db.exists():
                            files_to_delete.append(bundled_db)

            for file_path in set(files_to_delete):
                try:
                    resolved = file_path.resolve()
                    ws_resolved = workspace_dir.resolve()
                    if resolved.parts[:len(ws_resolved.parts)] != ws_resolved.parts:
                        continue
                    file_path.unlink()
                except Exception as e:
                    trace_exception(e)

        except Exception as e:
            ASCIIColors.warning(f"Error during cleanup for '{title}': {e}")

    def remove_by_id(self, artefact_id: str) -> bool:
        artefacts = self._get_all_raw()
        art_to_remove = next((a for a in artefacts if a.get('id') == artefact_id), None)
        new_list = [a for a in artefacts if a.get('id') != artefact_id]
        if len(new_list) < len(artefacts):
            self._save_all(new_list)
            # 🛠️ DYNAMIC TOOL UNREGISTRATION
            if art_to_remove and art_to_remove.get("type") == ArtefactType.TOOL:
                self._unregister_tool_artefact(art_to_remove.get("title", ""))
            return True
        return False

    def tag_version(self, title: str, version: int, tag_name: str) -> bool:
        tag_name = tag_name.strip().lower()
        if not tag_name:
            return False

        artefacts = self._get_all_raw()
        found = False
        for a in artefacts:
            if a.get('title') == title:
                tags = list(a.get('version_tags', []))
                if tag_name in tags:
                    tags.remove(tag_name)
                    a['version_tags'] = tags
                if a.get('version') == version:
                    tags.append(tag_name)
                    a['version_tags'] = tags
                    found = True

        if found:
            self._save_all(artefacts)
            return True
        return False

    def remove_tag(self, title: str, tag_name: str) -> bool:
        tag_name = tag_name.strip().lower()
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title:
                tags = list(a.get('version_tags', []))
                if tag_name in tags:
                    tags.remove(tag_name)
                    a['version_tags'] = tags
                    changed = True
        if changed:
            self._save_all(artefacts)
            return True
        return False

    def resolve_tag(self, title: str, tag_name: str) -> Optional[int]:
        tag_name = tag_name.strip().lower()
        for a in self._get_all_raw():
            if a.get('title') == title and tag_name in a.get('version_tags', []):
                return a.get('version')
        return None

    def get_log(self, title: str) -> List[Dict[str, Any]]:
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        all_versions.sort(key=lambda a: a.get('version', 0), reverse=True)

        log = []
        for a in all_versions:
            log.append({
                "commit_hash":    a.get("id"),
                "version":        a.get("version", 1),
                "author":         a.get("author", "AI Specialist"),
                "commit_message": a.get("commit_message") or f"Update '{title}' to version {a.get('version')}",
                "tags":           a.get("version_tags", []),
                "created_at":     a.get("created_at"),
                "size_chars":     len(a.get("content", "")),
                "is_active":      a.get("active", False)
            })
        return log

    def revert_to_tag(self, title: str, tag_name: str) -> Dict[str, Any]:
        version = self.resolve_tag(title, tag_name)
        if version is None:
            raise ValueError(f"Tag '{tag_name}' not found.")
        return self.revert(title, version)

    def activate(self, title: str, version: Optional[int] = None):
        artefacts = self._get_all_raw()
        changed = False
        if version is not None:
            for a in artefacts:
                if a.get('title') == title:
                    a['active'] = (a.get('version') == version)
                    changed = True
        else:
            latest_v = max([a.get('version', 1) for a in artefacts if a.get('title') == title], default=None)
            for a in artefacts:
                if a.get('title') == title:
                    a['active'] = (a.get('version') == latest_v)
                    changed = True
        if changed:
            self._save_all(artefacts)
            active_art = self.get(title, version)
            if active_art:
                self._sync_to_disk_workspace(
                    title=active_art["title"],
                    content=active_art["content"],
                    version=active_art["version"],
                    atype=active_art["type"],
                    language=active_art.get("language"),
                    file_ext=active_art.get("file_ext")
                )
                # 🛠️ DYNAMIC TOOL REGISTRATION (Activate)
                if active_art.get("type") == ArtefactType.TOOL:
                    self._register_tool_artefact(active_art)

    def sync_all_active_to_disk(self):
        active_arts = self.list(active_only=True)
        if not active_arts:
            return

        base_workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")
        try:
            from lollms_client.app.server import APP_WORKSPACE_DIR as awd
            if awd is not None:
                base_workspace_dir = awd
        except ImportError:
            pass

        disc_id = self._discussion.id
        workspace_dir = base_workspace_dir / disc_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = workspace_dir.resolve()

        synced_files = []
        for art in active_arts:
            if art.get("type") == "data":
                file_ext = art.get("file_ext", "")
                version = art.get("version", 1)
                title = art["title"]
                art_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, title.replace("workspace/", "").replace("data_workspace/", "").replace("/", "_").replace("\\", "_")))

                versioned_data = workspace_dir / "artefacts_metadata" / art_id / f"{title.replace('.csv','')}_v{version}{file_ext}"
                if versioned_data.exists():
                    dest = workspace_dir / "workspace_data" / f"{title}{file_ext}"
                    import shutil
                    shutil.copy(str(versioned_data), str(dest))
                    synced_files.append(str(dest.resolve()))
                else:
                    unversioned = workspace_dir / "workspace_data" / f"{title}{file_ext}"
                    if unversioned.exists():
                        synced_files.append(str(unversioned.resolve()))

            self._sync_to_disk_workspace(
                title=art["title"],
                content=art["content"],
                version=art["version"],
                atype=art["type"],
                language=art.get("language"),
                file_ext=art.get("file_ext")
            )

            filename = self._get_filename_with_ext(art["title"], art["type"], art.get("language"), art.get("file_ext"))
            expected_path = workspace_dir / filename
            if expected_path.exists():
                synced_files.append(str(expected_path.resolve()))

        return workspace_dir, synced_files

    def deactivate(self, title: str, version: Optional[int] = None):
        self._set_active(title, version, False)

    def toggle(self, title: str, version: Optional[int] = None) -> bool:
        artefact = self.get(title, version)
        if artefact is None:
            raise ValueError(f"Artefact '{title}' not found.")
        new_state = not artefact.get('active', False)
        self._set_active(title, version or artefact['version'], new_state)
        return new_state

    def _set_active(self, title: str, version: Optional[int], state: bool):
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title and (version is None or a.get('version') == version):
                a['visibility'] = ArtefactVisibility.FULL if state else ArtefactVisibility.HIDDEN
                a['active'] = state
                changed = True
        if changed:
            self._save_all(artefacts)
            if state:
                active_art = self.get(title, version)
                if active_art:
                    self._sync_to_disk_workspace(
                        title=active_art["title"],
                        content=active_art["content"],
                        version=active_art["version"],
                        atype=active_art["type"],
                        language=active_art.get("language"),
                        file_ext=active_art.get("file_ext")
                    )
                    # 🛠️ DYNAMIC TOOL REGISTRATION (_set_active)
                    if active_art.get("type") == ArtefactType.TOOL:
                        self._register_tool_artefact(active_art)

    def set_visibility(self, title: str, visibility: str, version: Optional[int] = None) -> bool:
        if visibility not in (ArtefactVisibility.HIDDEN, ArtefactVisibility.TREE_LOCKED, ArtefactVisibility.TREE_UNLOCKABLE, ArtefactVisibility.METADATA, ArtefactVisibility.FULL):
            raise ValueError(f"Invalid visibility tier: '{visibility}'")
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title and (version is None or a.get('version') == version):
                a['visibility'] = visibility
                a['active'] = (visibility == ArtefactVisibility.FULL)
                changed = True
        if changed:
            self._save_all(artefacts)
            return True
        return False

    def set_status(self, title: str, status: str, version: Optional[int] = None) -> bool:
        if status not in (ArtefactStatus.DRAFTING, ArtefactStatus.STABLE, ArtefactStatus.REVISING, ArtefactStatus.ERROR):
            raise ValueError(f"Invalid status: '{status}'")
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title and (version is None or a.get('version') == version):
                a['status'] = status
                a['updated_at'] = datetime.utcnow().isoformat()
                changed = True
        if changed:
            self._save_all(artefacts)
            return True
        return False

    def build_artefacts_context_zone(self) -> str:
        all_raw = self._get_all_raw()
        visible_artifacts = [
            a for a in all_raw 
            if a.get("visibility", ArtefactVisibility.HIDDEN) != ArtefactVisibility.HIDDEN 
            and not a.get("title", "").endswith("::images")
        ]

        if not visible_artifacts:
            return ""

        for a in visible_artifacts:
            if a.get("type") == "data":
                title = a.get("title", "UNNAMED")
                file_ext = a.get("file_ext", "")
                expected_filename = title if title.endswith(file_ext) else f"{title}{file_ext}"

        tree_lines = ["  workspace/"]
        root_node = {"files": [], "folders": {}}
        for a in visible_artifacts:
            display_title = a["title"]
            if a.get("type") == "data" and a.get("file_ext"):
                if not display_title.lower().endswith(a["file_ext"].lower()):
                    display_title = f"{display_title}{a['file_ext']}"

            parts = display_title.split("/")
            curr = root_node
            for i in range(len(parts) - 1):
                folder = parts[i]
                curr = curr["folders"].setdefault(folder, {"files": [], "folders": {}})
            curr["files"].append({**a, "display_title": display_title})

        def _traverse_tree_prompt(node, depth=1):
            lines = []
            indent = "  " * depth
            for f_name, f_node in sorted(node["folders"].items()):
                lines.append(f"{indent}├── {f_name}/")
                lines.extend(_traverse_tree_prompt(f_node, depth + 1))
            for a in sorted(node["files"], key=lambda x: x["display_title"]):
                v_tier = a.get("visibility", ArtefactVisibility.FULL)
                status = a.get("status", ArtefactStatus.STABLE).upper()
                marker = "[L]"
                if v_tier == ArtefactVisibility.FULL: marker = "[C]"
                elif v_tier == ArtefactVisibility.METADATA: marker = "[M]"
                elif v_tier == ArtefactVisibility.TREE_UNLOCKABLE: marker = "[U]"
                f_name = a["display_title"].split("/")[-1]
                lines.append(f"{indent}├── {f_name}  {marker}  ({status})")
            return lines

        tree_lines.extend(_traverse_tree_prompt(root_node))
        tree_text = "\n".join(tree_lines)

        full_visible_parts = []
        partial_visible_parts = []

        for item in visible_artifacts:
            v_tier = item.get("visibility", ArtefactVisibility.FULL)
            atype = item.get("type", ArtefactType.DOCUMENT)
            lang = item.get("language")
            fence = f"```{lang}" if lang else "```"

            display_title = item["title"]
            if atype == ArtefactType.DATA and item.get("file_ext"):
                if not display_title.lower().endswith(item["file_ext"].lower()):
                    display_title = f"{display_title}{item['file_ext']}"

            if v_tier == ArtefactVisibility.FULL:
                # FIX: For text-based artifacts (code, document), use the content field directly.
                # Only use the .lam logical twin for DATA artifacts (where content is a schema).
                if item.get("type") == ArtefactType.DATA:
                    content_text = self._get_lam_content(item).strip()
                else:
                    content_text = (item.get("content") or "").strip()

                if not content_text:
                    continue

                header = f"### [Full File: '{display_title}']"
                deactivated_contents = getattr(self._discussion, "deactivated_contents", set())

                if display_title in deactivated_contents:
                    seq_summaries = getattr(self._discussion, "sequential_summaries", {})
                    summary_text = seq_summaries.get(display_title, "Detailed summary not available.")
                    full_visible_parts.append(f"{header}\n[SEQUENTIAL COMPRESSED DATA DUE TO BUDGET CONSTRAINTS]:\n{summary_text}")
                else:
                    if atype == ArtefactType.DATA:
                        schema_desc = content_text
                        schema_desc = f"**FILENAME FOR TOOLS:** `{display_title}`\n\n{schema_desc}"
                        full_visible_parts.append(f"{header}\n{schema_desc}")
                    elif atype == ArtefactType.CODE or (lang and not _ARTEFACT_IMAGE_TAG_RE.search(content_text)):
                        full_visible_parts.append(f"{header}\n{fence}\n{content_text}\n```")
                    else:
                        full_visible_parts.append(f"{header}\n{content_text}")

            elif v_tier == ArtefactVisibility.METADATA:
                partial_desc = _extract_file_metadata_and_signature(item)
                if partial_desc:
                    partial_visible_parts.append(partial_desc)

        context_parts = [
            "## Workspace Directory Tree Index",
            "This index displays all files in your workspace directory tree with their context state markers:",
            "  - [C] Fully Loaded in Context (Verbatim text/code is read-ready below)",
            "  - [M] Signature / Metadata Only (Exposes schemas, layouts, or code signatures below)",
            "  - [U] Inactive/Unlockable (Excluded from context, but you can unlock/load it to [C] by calling <add_files_to_context>)",
            "  - [L] Locked in Tree (Excluded from context and cannot be unlocked)\n",
            tree_text
        ]

        if partial_visible_parts:
            context_parts.append("\n## Partial File Signatures & Metadata [M]\n" + "\n\n---\n\n".join(partial_visible_parts))
        if full_visible_parts:
            context_parts.append("\n## Fully Loaded File Contents [C]\n" + "\n\n---\n\n".join(full_visible_parts))

        return "\n\n".join(context_parts)

    def get_context_images(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        raw_list = self._get_all_raw()
        by_title = {a.get('title'): a for a in raw_list}

        for a in raw_list:
            if a.get('visibility', ArtefactVisibility.FULL) != ArtefactVisibility.FULL:
                continue
            imgs   = a.get('images') or []
            mtypes = a.get('image_media_types') or []
            for idx, img_b64 in enumerate(imgs):
                if not img_b64:
                    continue
                mtype = mtypes[idx] if idx < len(mtypes) else "image/jpeg"
                result.append({
                    "id":         make_image_id(a['title'], idx),
                    "data":       img_b64,
                    "media_type": mtype,
                    "title":      a['title'],
                    "index":      idx,
                    "active":     True,
                })

            comp_title = f"{a.get('title')}::images"
            if comp_title in by_title:
                comp_a = by_title[comp_title]
                if comp_a.get('visibility', ArtefactVisibility.FULL) == ArtefactVisibility.FULL:
                    comp_imgs = comp_a.get('images') or []
                    comp_mtypes = comp_a.get('image_media_types') or []
                    for idx, img_b64 in enumerate(comp_imgs):
                        if not img_b64:
                            continue
                        mtype = comp_mtypes[idx] if idx < len(comp_mtypes) else "image/jpeg"
                        result.append({
                            "id":         make_image_id(comp_title, idx),
                            "data":       img_b64,
                            "media_type": mtype,
                            "title":      comp_title,
                            "index":      idx,
                            "active":     True,
                        })
        return result

    def get_active_images(self) -> List[Dict[str, Any]]:
        result = []
        for a in self._get_all_raw():
            if a.get('active', False) and a.get('type') == ArtefactType.IMAGE:
                source_id = f"artifact:{a['title']} v{a['version']}"
                for img_b64 in (a.get('images') or []):
                    result.append({
                        "data":       img_b64,
                        "source":     source_id,
                        "active":     True,
                        "created_at": a.get('created_at', ''),
                    })
        return result

    @staticmethod
    def apply_aider_patch(original: str, patch_block: str) -> str:
        import re as _re

        SEARCH_RE  = _re.compile(r'^<{6,8}(?:\s*\w+)?\s*$',  _re.IGNORECASE)
        SEP_RE     = _re.compile(r'^={5,}\s*$')
        REPLACE_RE = _re.compile(r'^>{6,8}(?:\s*\w+)?\s*$', _re.IGNORECASE)

        patch_block = patch_block.replace('\r\n', '\n').replace('\r', '\n')
        lines = patch_block.split('\n')
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and not REPLACE_RE.match(lines[-1].rstrip()):
            lines.append('>>>>>>> REPLACE')
        patch_block = '\n'.join(lines)

        segments: List[Tuple[str, str]] = []
        raw_lines = patch_block.split('\n')
        i = 0
        while i < len(raw_lines):
            if SEARCH_RE.match(raw_lines[i].rstrip()):
                i += 1
                search_lines: List[str] = []
                while i < len(raw_lines):
                    if SEP_RE.match(raw_lines[i].rstrip()) or REPLACE_RE.match(raw_lines[i].rstrip()):
                        break
                    search_lines.append(raw_lines[i])
                    i += 1

                if i >= len(raw_lines):
                    raise ValueError(f"Malformed aider patch: SEARCH block has no ======= separator.")

                if REPLACE_RE.match(raw_lines[i].rstrip()):
                    segments.append(('\n'.join(search_lines), ''))
                    i += 1
                    continue

                i += 1
                replace_lines: List[str] = []
                while i < len(raw_lines):
                    if REPLACE_RE.match(raw_lines[i].rstrip()):
                        break
                    replace_lines.append(raw_lines[i])
                    i += 1
                if i < len(raw_lines) and REPLACE_RE.match(raw_lines[i].rstrip()):
                    i += 1

                segments.append(('\n'.join(search_lines), '\n'.join(replace_lines)))
            else:
                i += 1

        if not segments:
            raise ValueError("Malformed aider patch: no valid <<<<<<< SEARCH … >>>>>>> REPLACE block found.")

        # Helper functions
        def _get_indent(line: str) -> int:
            return len(line) - len(line.lstrip())

        def _reindent(r_text: str, target_indent: int) -> str:
            r_lines = r_text.splitlines()
            first_nonempty = next((l for l in r_lines if l.strip()), None)
            if first_nonempty is None:
                return r_text
            base   = _get_indent(first_nonempty)
            delta  = target_indent - base

            def shift(line: str) -> str:
                if not line.strip():
                    return ''
                return ' ' * max(0, _get_indent(line) + delta) + line.lstrip()

            return '\n'.join(shift(l) for l in r_lines)

        def _apply_at(res_lines: List[str], start: int, length: int, r_text: str) -> str:
            reindented = _reindent(r_text, _get_indent(res_lines[start]))
            return '\n'.join(res_lines[:start] + [reindented] + res_lines[start + length:])

        def _window_score(window: List[str], s_lines: List[str]) -> float:
            score = 0.0
            for w, s in zip(window, s_lines):
                if w == s:
                    score += 1.0
                elif w.rstrip() == s.rstrip():
                    score += 0.5
                elif w.strip() == s.strip():
                    score += 0.25
            return score

        def _strip_inline_comment(line: str) -> str:
            s = _re.sub(r'''(?x)(?<![\'\"\\])\s*\#[^\'\"]* $''', '', line)
            if s != line:
                return s.rstrip()
            return _re.sub(r'\s*//.*$', '', line).rstrip()

        def _comment_key(line: str) -> str:
            return _strip_inline_comment(line).rstrip()

        def _collapse_blanks(lines_in: List[str]) -> List[Tuple[int, str]]:
            out: List[Tuple[int, str]] = []
            prev_blank = False
            for idx, l in enumerate(lines_in):
                is_blank = not l.strip()
                if is_blank and prev_blank:
                    continue
                out.append((idx, l))
                prev_blank = is_blank
            return out

        result = original

        for seg_idx, (search_text, replace_text) in enumerate(segments):
            s_text = search_text.strip('\n')
            r_text = replace_text.strip('\n')

            res_lines = result.splitlines()
            s_lines   = s_text.splitlines()
            n_s       = len(s_lines)
            n_r       = len(res_lines)
            match_found = False

            label = f"seg {seg_idx + 1}/{len(segments)}"

            # Pass 1: Exact
            if s_text in result:
                result = result.replace(s_text, r_text, 1)
                continue

            # Pass 2: Trailing-space
            for i in range(n_r - n_s + 1):
                window = res_lines[i : i + n_s]
                if all(w.rstrip() == s.rstrip() for w, s in zip(window, s_lines)):
                    pre  = res_lines[:i]
                    post = res_lines[i + n_s:]
                    result = '\n'.join(pre + [r_text] + post)
                    match_found = True
                    break

            if match_found:
                continue

            # Pass 3: Indent
            best_i     = -1
            best_score = -1.0
            for i in range(n_r - n_s + 1):
                window = res_lines[i : i + n_s]
                if (
                    all(w.strip() == s.strip() for w, s in zip(window, s_lines))
                    and any(s.strip() for s in s_lines)
                ):
                    score = _window_score(window, s_lines)
                    if score > best_score:
                        best_score = score
                        best_i     = i

            if best_i >= 0:
                result = _apply_at(res_lines, best_i, n_s, r_text)
                match_found = True

            if match_found:
                continue

            # Pass C1: Comments
            s_keys = [_comment_key(l) for l in s_lines]
            best_i     = -1
            best_score = -1.0
            for i in range(n_r - n_s + 1):
                window      = res_lines[i : i + n_s]
                window_keys = [_comment_key(l) for l in window]
                if (
                    all(wk == sk for wk, sk in zip(window_keys, s_keys))
                    and any(sk for sk in s_keys)
                ):
                    score = _window_score(window, s_lines)
                    if score > best_score:
                        best_score = score
                        best_i     = i

            if best_i >= 0:
                result = _apply_at(res_lines, best_i, n_s, r_text)
                match_found = True

            if match_found:
                continue

            # Pass C2: Blanks
            collapsed_res = _collapse_blanks(res_lines)
            collapsed_s   = _collapse_blanks(s_lines)
            cs_len        = len(collapsed_s)

            best_i      = -1
            best_orig_start = -1
            best_orig_end   = -1
            best_score  = -1.0

            for i in range(len(collapsed_res) - cs_len + 1):
                window = collapsed_res[i : i + cs_len]
                if (
                    all(
                        w_line.rstrip() == s_line.rstrip()
                        for (_, w_line), (_, s_line) in zip(window, collapsed_s)
                    )
                    and any(s_line.strip() for _, s_line in collapsed_s)
                ):
                    w_lines = [l for _, l in window]
                    s_lines_c = [l for _, l in collapsed_s]
                    score = _window_score(w_lines, s_lines_c)
                    if score > best_score:
                        best_score      = score
                        best_i          = i
                        best_orig_start = window[0][0]
                        best_orig_end   = window[-1][0] + 1

            if best_i >= 0:
                result = _apply_at(res_lines, best_orig_start, best_orig_end - best_orig_start, r_text)
                match_found = True

            if match_found:
                continue

            # Pass D: Core delta
            s_lines_full = search_text.splitlines()
            r_lines_full = replace_text.splitlines()

            diff_indices = [
                idx for idx, (s, r) in enumerate(zip(s_lines_full, r_lines_full)) if s != r
            ]
            if not diff_indices and len(s_lines_full) != len(r_lines_full):
                diff_indices = list(range(len(s_lines_full)))

            if diff_indices:
                lo = max(0, min(diff_indices) - 1)
                hi = min(len(s_lines_full), max(diff_indices) + 2)
                core_search  = '\n'.join(s_lines_full[lo:hi])
                core_replace = '\n'.join(r_lines_full[lo:hi])

                if core_search and core_search in result:
                    result = result.replace(core_search, core_replace, 1)
                    match_found = True

            if match_found:
                continue

            first_line = s_lines[0] if s_lines else ""
            hint = _find_closest_line(first_line, result)
            raise ValueError(f"SEARCH text not found in artifact content.")

        return result

    def _apply_artefact_xml(
        self,
        text:          str,
        default_type:  str = ArtefactType.CODE,
        auto_activate: bool = True,
        replacements:  Optional[Dict[str, str]] = None,
        event_callback: Optional[Any] = None,
        already_processed_artifacts: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        affected: List[Dict] = []
        cleaned  = text

        artefact_pattern = re.compile(
            r'^[ \t]*<art[ei]fact\s([^>]*)>(.*?)</art[ei]fact>',
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        existing_titles = self._all_latest_titles()

        def handle_artefact(match: re.Match) -> str:
            nonlocal existing_titles

            attrs   = _parse_attrs(match.group(1))
            content = match.group(2)

            if replacements:
                for placeholder, original in replacements.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

            tag_title   = attrs.pop('name', attrs.pop('title', f'artifact_{uuid.uuid4().hex[:8]}'))
            new_name    = attrs.pop('rename', None)
            atype       = attrs.pop('type', default_type)
            language    = attrs.pop('language', None)
            version_str = attrs.pop('version', '1')
            version     = int(version_str) if version_str.isdigit() else 1
            is_ephemeral = attrs.pop('ephemeral', 'false').lower() in ('true', '1', 'yes')
            status      = attrs.pop('status', ArtefactStatus.STABLE)
            commit_message = attrs.pop('commit_message', None)
            version_tags   = attrs.pop('version_tags', None)

            if atype not in ArtefactType.ALL:
                if atype in ("csv", "tsv", "excel", "xlsx", "xls", "db", "sqlite"):
                    atype = ArtefactType.DATA
                elif atype in ("text", "txt", "markdown", "md"):
                    atype = ArtefactType.DOCUMENT
                else:
                    atype = default_type

            resolved_title = tag_title if tag_title in existing_titles else (
                _find_best_title_match(tag_title, existing_titles) or tag_title
            )

            if already_processed_artifacts and resolved_title in already_processed_artifacts:
                art = self.get(resolved_title)
                if art: affected.append(art)
                tag_name = "artifact" if "artifact" in match.group(0).lower() else "artefact"
                return f'<{tag_name} {match.group(1)}>\n[content stripped, refer to the artefact for details]\n</{tag_name}>'
            
            is_new = False
            if tag_title in existing_titles:
                resolved_title = tag_title
            else:
                fuzzy = _find_best_title_match(tag_title, existing_titles)
                if fuzzy:
                    resolved_title = fuzzy
                else:
                    resolved_title = tag_title
                    is_new = True

            _has_search = bool(
                re.search(r'<{6,8}\s*SEARCH', content, re.IGNORECASE)
            )

            result_artefact: Optional[Dict] = None

            if _has_search:
                existing = self.get(resolved_title)
                if existing is None:
                    result_artefact = self.add(
                        title=resolved_title, artefact_type=atype, content=content,
                        language=language, version=version, active=auto_activate,
                        commit_message=commit_message, version_tags=version_tags,
                        ephemeral=is_ephemeral, status=status,
                        **attrs
                    )
                    is_new = True
                else:
                    try:
                        patched = ArtefactManager.apply_aider_patch(
                            existing.get('content', ''), content
                        )
                        result_artefact = self.update(
                            title=resolved_title,
                            new_content=patched,
                            new_title=new_name,
                            language=language,
                            bump_version=True,
                            active=auto_activate,
                            commit_message=commit_message,
                            version_tags=version_tags,
                            ephemeral=is_ephemeral,
                            status=status,
                            **attrs
                        )
                    except ValueError as e:
                        result_artefact = None
            else:
                if is_new:
                    result_artefact = self.add(
                        title=resolved_title, artefact_type=atype,
                        content=content.strip(),
                        language=language, version=version, active=auto_activate,
                        commit_message=commit_message, version_tags=version_tags,
                        ephemeral=is_ephemeral, status=status,
                        **attrs
                    )
                else:
                    result_artefact = self.update(
                        title=resolved_title,
                        new_content=content.strip(),
                        new_title=new_name,
                        new_type=atype,
                        language=language,
                        bump_version=True,
                        active=auto_activate,
                        commit_message=commit_message,
                        version_tags=version_tags,
                        ephemeral=is_ephemeral,
                        status=status,
                        **attrs
                    )

            existing_titles = self._all_latest_titles()
            if result_artefact:
                affected.append(result_artefact)

            if event_callback and result_artefact:
                try:
                    event_callback(result_artefact, is_new)
                except Exception:
                    pass

            tag_name = "artifact" if "artifact" in match.group(0).lower() else "artefact"
            return f'<{tag_name} {match.group(1)}>\n[content stripped, refer to the artefact for details]\n</{tag_name}>'

        cleaned = artefact_pattern.sub(handle_artefact, cleaned)
        revert_pattern = re.compile(r'^[ \t]*<revert_art[ei]fact\s([^>]+)/?>', re.IGNORECASE | re.MULTILINE)

        def handle_revert(match: re.Match) -> str:
            attrs   = _parse_attrs(match.group(1))
            title   = attrs.get('name') or attrs.get('title')
            version = attrs.get('version')
            if title and version:
                resolved = title if title in existing_titles else (
                    _find_best_title_match(title, existing_titles) or title
                )
                try:
                    res_artefact = self.revert(resolved, version)
                    affected.append(res_artefact)
                    if event_callback:
                        try: event_callback(res_artefact, False)
                        except Exception: pass
                except ValueError:
                    pass
            return ''

        cleaned = revert_pattern.sub(handle_revert, cleaned)
        return cleaned.strip(), affected

    # Legacy Compatibility Shims
    def add_artefact(self, title, content="", images=None, audios=None, videos=None, zip_content=None, version=1, **extra_data) -> Dict:
        atype = extra_data.pop('artefact_type', ArtefactType.DOCUMENT)
        return self.add(title=title, artefact_type=atype, content=content, images=images, audios=audios, videos=videos, zip_content=zip_content, version=version, **extra_data)

    def get_artefact(self, title, version=None) -> Optional[Dict]:
        return self.get(title, version)

    def update_artefact(self, title, new_content, new_images=None, **extra_data) -> Dict:
        return self.update(title=title, new_content=new_content, new_images=new_images, **extra_data)

    def list_artefacts(self) -> List[Dict]:
        items = self.list()
        disc  = self._discussion
        for a in items:
            section_start  = f"--- Document: {a['title']} v{a['version']} ---"
            content_loaded = section_start in (disc.discussion_data_zone or "")
            source_id      = f"artifact:{a['title']} v{a['version']}"
            image_loaded   = any(img.get("source") == source_id for img in disc.get_discussion_images())
            a["is_loaded"] = content_loaded or image_loaded
        return items

    def get_version_history_artefact(self, title: str) -> List[Dict[str, Any]]:
        return self.get_version_history(title)

    def diff_versions_artefact(self, title: str, version_a: int, version_b: int) -> Dict[str, Any]:
        return self.diff_versions(title, version_a, version_b)

    def squash_versions_artefact(self, title: str, keep_versions: Optional[List[int]] = None, keep_last_n: Optional[int] = None, target_version: Optional[int] = None) -> Dict[str, Any]:
        return self.squash_versions(title, keep_versions, keep_last_n, target_version)

    def cleanup_old_versions_artefact(self, title: str, keep_count: int = 5, min_age_hours: Optional[float] = None) -> Dict[str, Any]:
        return self.cleanup_old_versions(title, keep_count, min_age_hours)

    def remove_artefact(self, title, version=None) -> int:
        return self.remove(title, version)

    def get_associated_images(self, title: str, version: Optional[int] = None) -> List[Dict[str, Any]]:
        images = []
        main_art = self.get(title, version)
        if main_art:
            imgs = main_art.get("images") or []
            mtypes = main_art.get("image_media_types") or []
            for idx, img_b64 in enumerate(imgs):
                mtype = mtypes[idx] if idx < len(mtypes) else "image/jpeg"
                images.append({"id": make_image_id(title, idx), "data": img_b64, "media_type": mtype, "title": title, "index": idx})

        comp_title = f"{title}::images"
        comp_art = self.get(comp_title)
        if comp_art:
            imgs = comp_art.get("images") or []
            mtypes = comp_art.get("image_media_types") or []
            for idx, img_b64 in enumerate(imgs):
                mtype = mtypes[idx] if idx < len(mtypes) else "image/jpeg"
                images.append({"id": make_image_id(comp_title, idx), "data": img_b64, "media_type": mtype, "title": comp_title, "index": idx})
        return images

    def export_artefact_bundle(self, title: str) -> Optional[Dict[str, Any]]:
        main_art = self.get(title)
        if not main_art:
            return None

        bundle = {
            "version": 1,
            "exported_at": datetime.utcnow().isoformat(),
            "main_artefact": {k: v for k, v in main_art.items() if k != "id"},
            "companion_artefacts": []
        }

        comp_title = f"{title}::images"
        comp_art = self.get(comp_title)
        if comp_art:
            bundle["companion_artefacts"].append({k: v for k, v in comp_art.items() if k != "id"})
        return bundle

    def import_artefact_bundle(self, bundle: Dict[str, Any], activate: bool = True) -> Optional[Dict[str, Any]]:
        if not isinstance(bundle, dict) or "main_artefact" not in bundle:
            raise ValueError("Invalid artefact bundle format")

        main_info = bundle["main_artefact"]
        main_art = self.add(
            title             = main_info["title"],
            artefact_type     = main_info.get("type", ArtefactType.DOCUMENT),
            content           = main_info.get("content", ""),
            images            = main_info.get("images"),
            image_media_types = main_info.get("image_media_types"),
            audios            = main_info.get("audios"),
            videos            = main_info.get("videos"),
            zip_content       = main_info.get("zip"),
            language          = main_info.get("language"),
            url               = main_info.get("url"),
            tags              = main_info.get("tags"),
            active            = activate
        )

        for comp_info in bundle.get("companion_artefacts", []):
            self.add(
                title             = comp_info["title"],
                artefact_type     = comp_info.get("type", ArtefactType.IMAGE),
                content           = comp_info.get("content", ""),
                images            = comp_info.get("images"),
                image_media_types = comp_info.get("image_media_types"),
                audios            = comp_info.get("audios"),
                videos            = comp_info.get("videos"),
                zip_content       = comp_info.get("zip"),
                language          = comp_info.get("language"),
                url               = comp_info.get("url"),
                tags              = comp_info.get("tags"),
                active            = activate
            )
        return main_art

    def export_artefact(self, title: str) -> Optional[Dict[str, Any]]:
        all_raw = self._get_all_raw()
        versions = [a for a in all_raw if a.get('title') == title]
        if not versions:
            return None

        def _safe_version_key(a):
            try:
                v = a.get('version', 1)
                if isinstance(v, str):
                    v = int(float(v.split('.')[0]))
                return int(v)
            except Exception:
                return 1

        sorted_versions = sorted(versions, key=_safe_version_key)
        latest = sorted_versions[-1]

        comp_versions = [a for a in all_raw if a.get('title') == f"{title}::images"]
        sorted_comp_versions = sorted(comp_versions, key=_safe_version_key)

        return {
            "version": 1,
            "title": title,
            "type": latest.get("type", ArtefactType.DOCUMENT),
            "versions": [{k: v for k, v in ver.items() if k != "id"} for ver in sorted_versions],
            "companion_images_versions": [{k: v for k, v in ver.items() if k != "id"} for ver in sorted_comp_versions],
            "exported_at": datetime.utcnow().isoformat()
        }

    def import_file(self, file_path: Union[str, Path], title: Optional[str] = None, active: bool = True) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if title is None:
            title = path.stem

        ext = path.suffix.lower()

        artefact_type = ArtefactType.DOCUMENT
        if ext in (".csv", ".tsv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3", ".sqlconn", ".ttl", ".rdf", ".parquet"):
            artefact_type = ArtefactType.DATA
        elif ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
            artefact_type = ArtefactType.CODE
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
            artefact_type = ArtefactType.IMAGE
        elif ext in (".md", ".txt", ".log", ".json", ".yaml", ".yml", ".xml"):
            artefact_type = ArtefactType.DOCUMENT

        try:
            physical_data = path.read_bytes()
        except Exception:
            physical_data = None

        content = ""
        if artefact_type == ArtefactType.DATA:
            try:
                from lollms_client.lollms_artefact.data_files import _parse_data_file
                schema_result = _parse_data_file(path, title, version=1)
                if len(schema_result) == 3:
                    content, _, _ = schema_result
                else:
                    content, _ = schema_result
            except Exception as e:
                content = f"# Data Interface: {title}\n"
        elif artefact_type == ArtefactType.IMAGE:
            content = f"### Image: `{title}{ext}`\n\n<artefact_image id=\"{title}{ext}::0\" />"
        elif artefact_type in (ArtefactType.CODE, ArtefactType.DOCUMENT):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                content = f"Failed to read content: {e}"

        art = self.add(
            title=title + ext,
            artefact_type=artefact_type,
            content=content,
            language=ext.replace(".", "") if artefact_type == ArtefactType.CODE else None,
            active=active,
            file_ext=ext,
            physical_data=physical_data
        )

        if self._discussion:
            self._discussion.commit()
        return art

    def import_artefact(self, artefact_data: Dict[str, Any], activate: bool = True) -> Optional[Dict[str, Any]]:
        if not isinstance(artefact_data, dict) or "versions" not in artefact_data:
            raise ValueError("Invalid multi-version exported artefact format")

        title = artefact_data.get("title")
        if not title:
            raise ValueError("Missing title in imported data")

        all_raw = self._get_all_raw()
        cleaned_raw = [a for a in all_raw if a.get('title') != title and a.get('title') != f"{title}::images"]

        imported_versions = []
        for ver_info in artefact_data["versions"]:
            new_ver = ver_info.copy()
            new_ver["id"] = str(uuid.uuid4())
            new_ver["active"] = False
            cleaned_raw.append(new_ver)
            imported_versions.append(new_ver)

        for comp_info in artefact_data.get("companion_images_versions", []):
            new_comp = comp_info.copy()
            new_comp["id"] = str(uuid.uuid4())
            new_comp["active"] = False
            cleaned_raw.append(new_comp)

        self._save_all(cleaned_raw)

        if imported_versions:
            latest_imported = imported_versions[-1]
            if activate:
                self.activate(title, latest_imported.get("version", 1))
                if artefact_data.get("companion_images_versions"):
                    self.activate(f"{title}::images")

        return self.get(title)
