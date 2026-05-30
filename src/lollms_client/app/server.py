#!/usr/bin/env python3
"""
server.py
=========
FastAPI backend for the Multimodal Document Viewer.
Supports local document ingestion, binary image streaming, bundle packaging, 
multi-source internet search/scraping, and streaming conversational chat.
"""
import re
import time
import sys
import os
import json
import tempfile
import requests
import queue
import threading
import asyncio
import inspect
import uuid
from pathlib import Path
from typing import Optional, Dict, Callable, List, Any
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
from lollms_client.lollms_discussion import ArtefactType

# Ensure correct workspace import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import yaml
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception
from datetime import datetime

app = FastAPI(title="Lollms Multimodal Document Viewer")

# Define paths
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Define application directories in user's home folder
APP_DIR = Path.home() / ".lollms_client_app"
APP_DIR.mkdir(parents=True, exist_ok=True)

APP_CONFIG_PATH = APP_DIR / "config.json"

def load_app_config():
    if APP_CONFIG_PATH.exists():
        try:
            with open(APP_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            ASCIIColors.warning(f"Failed to load app config: {e}")
    return {}

def save_app_config(cfg):
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        with open(APP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        ASCIIColors.error(f"Failed to save app config: {e}")


def _parse_soul_metadata(content: str) -> dict:
    import re
    meta = {}
    content_stripped = content.strip()
    m = re.search(r'^---\s*(.*?)\s*---', content_stripped, re.DOTALL)
    if m:
        yaml_str = m.group(1).strip()
        for line in yaml_str.splitlines():
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip().strip("'\"")
    return meta


# Global state mappings for active workspace allocation
CURRENT_WORKSPACE_DIR = None
APP_DATA_DIR = None
APP_WORKSPACE_DIR = None
db_mgr = None
discussion = None
client = None


def initialize_workspace_state(workspace_name: str):
    """Dynamically switch active project workspace databases, files, and clients in-process."""
    global CURRENT_WORKSPACE_DIR, APP_DATA_DIR, APP_WORKSPACE_DIR, db_mgr, discussion, client, LLM_MODEL_NAME, needs_configuration

    workspace_name_clean = "".join(c for c in workspace_name if c.isalnum() or c in ("-", "_")).strip().lower() or "default"

    CURRENT_WORKSPACE_DIR = APP_DIR / "workspaces" / workspace_name_clean
    APP_DATA_DIR = CURRENT_WORKSPACE_DIR / "data"
    APP_WORKSPACE_DIR = CURRENT_WORKSPACE_DIR / "data_workspace"

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    APP_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    db_path = APP_DATA_DIR / "multimodal_viewer.db"
    db_mgr = LollmsDataManager(f"sqlite:///{db_path}")

    cfg = load_app_config()
    cfg["active_workspace"] = workspace_name_clean
    save_app_config(cfg)

    active_personality_prompt = None
    if cfg and cfg.get("llm_binding_name") and cfg.get("llm_binding_config", {}).get("model_name"):
        try:
            LLM_MODEL_NAME = cfg["llm_binding_config"]["model_name"]
            client_kwargs = {
                "llm_binding_name": cfg["llm_binding_name"],
                "llm_binding_config": cfg["llm_binding_config"],
                "tools_binding_name": "lcp",
                "tools_binding_config": {
                    "tools_folders": [
                        str(APP_WORKSPACE_DIR.resolve()),
                        str(PROJECT_ROOT / "lollms_client" / "tools_bindings" / "lcp" / "default_tools")
                    ]
                }
            }
            if cfg.get("tti_binding_name"):
                client_kwargs["tti_binding_name"] = cfg["tti_binding_name"]
                client_kwargs["tti_binding_config"] = cfg.get("tti_binding_config", {})

            client = LollmsClient(**client_kwargs)
            ASCIIColors.green(f"⚡ [Workspace '{workspace_name_clean}'] Loaded saved configuration: {LLM_MODEL_NAME} via {cfg['llm_binding_name']}")
            needs_configuration = False

            p_name = cfg.get("personality_name")
            if p_name and "/" in p_name:
                category, persona = p_name.split("/", 1)
                soul_file = Path("./data_workspace") / "personalities" / category / persona / "SOUL.md"
                if soul_file.exists():
                    content = soul_file.read_text(encoding="utf-8", errors="ignore")
                    active_personality_prompt = re.sub(r'^---\s*(.*?)\s*---', '', content, flags=re.DOTALL).strip()
        except Exception as e:
            ASCIIColors.yellow(f"⚠️ [Workspace '{workspace_name_clean}'] Saved config failed to load ({e}). Falling back to mock.")
            needs_configuration = True
            client = DummyClient()
    else:
        ASCIIColors.yellow(f"⚠️ [Workspace '{workspace_name_clean}'] No saved configuration found. User must configure LLM.")
        needs_configuration = True
        client = DummyClient()

    # Load or create persistent discussion session
    if db_mgr.discussion_exists("viewer_session"):
        discussion = db_mgr.get_discussion(client, "viewer_session", autosave=True)
    else:
        discussion = LollmsDiscussion.create_new(
            lollms_client=client,
            db_manager=db_mgr,
            id="viewer_session",
            autosave=True
        )

    if active_personality_prompt:
        discussion.system_prompt = active_personality_prompt

    # Attach persistent multi-level memory manager so <mem_new> tags survive across turns
    mem_db_path = APP_DATA_DIR / "memories.db"
    mem_db_path.parent.mkdir(parents=True, exist_ok=True)
    discussion._init_memory(LollmsMemoryManager(f"sqlite:///{mem_db_path}"))

    # Update active discussion's context size budget
    if hasattr(client, "get_ctx_size"):
        try:
            val = client.get_ctx_size()
            discussion.max_context_size = val if (val and val > 1) else 4096
        except Exception:
            discussion.max_context_size = 4096
    else:
        discussion.max_context_size = 4096


# ── 🌐 New Models for Internet Ingestion ──
class WebSearchRequest(BaseModel):
    query: str
    provider: str = "duckduckgo"

class WikipediaSearchRequest(BaseModel):
    query: str

class WikipediaImportItem(BaseModel):
    title: str
    url: str

class WikipediaImportSelectedRequest(BaseModel):
    items: List[WikipediaImportItem]
    auto_load: bool = True

class ArxivSearchRequest(BaseModel):
    query: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    max_results: int = 5

class ArxivImportItem(BaseModel):
    id: str
    title: str
    mode: str = "abstract"

class ArxivImportSelectedRequest(BaseModel):
    items: List[ArxivImportItem]
    auto_load: bool = True

class GithubSearchRequest(BaseModel):
    query: str

class GithubImportRequest(BaseModel):
    url: str
    auto_load: bool = True

class StackOverflowSearchRequest(BaseModel):
    query: str

class StackOverflowImportRequest(BaseModel):
    url: str
    auto_load: bool = True

class YoutubeImportRequest(BaseModel):
    url: str
    language: str = "en"
    auto_load: bool = True


class GenerateImageForMessageRequest(BaseModel):
    prompt: Optional[str] = None


class DataQueryRequest(BaseModel):
    title: str
    code: str


class SaveToolRequest(BaseModel):
    title: str
    code: str
    commit_message: Optional[str] = "Save tool update"


class RefineToolRequest(BaseModel):
    code: str
    instruction: str
    docs: List[str] = Field(default_factory=list)


class RunToolInitRequest(BaseModel):
    code: str


class ParseToolFunctionsRequest(BaseModel):
    code: str


class ExecuteToolFunctionRequest(BaseModel):
    code: str
    function_name: str
    params: Dict[str, Any]


class CountToolTokensRequest(BaseModel):
    code: str
    docs: List[str]
    instruction: str


class ExportMemoriesRequest(BaseModel):
    scope: str
    format: str

class ExportToolsRequest(BaseModel):
    tools: List[str]
    format: str


class SelectPersonalityRequest(BaseModel):
    category: str
    persona: str


class DownloadZooRequest(BaseModel):
    category: str
    persona: str


class ProfileRequest(BaseModel):
    name: str


class ExecuteCommandRequest(BaseModel):
    modality: str  # "llm" or "tti"
    binding_name: str
    command_name: str
    parameters: Dict[str, Any]


class UpdateArtifactRequest(BaseModel):
    content: str
    commit_message: Optional[str] = "Manual update via Raw Editor"


class UpdateImageArtifactRequest(BaseModel):
    image_b64: str
    commit_message: Optional[str] = "Edit image via Photoshop UI"


class WorkspaceRequest(BaseModel):
    name: str

class EditMemoryRequest(BaseModel):
    content: Optional[str] = None
    importance: Optional[float] = None
    level: Optional[int] = None


class ImportMemoryRequest(BaseModel):
    content: str
    importance: Optional[float] = 0.75
    level: Optional[int] = 1
    tags: Optional[List[str]] = None

class BindingTestRequest(BaseModel):
    binding_name: str
    config: Dict[str, Any]

class ApplySettingsRequest(BaseModel):
    llm_binding_name: str
    llm_binding_config: Dict[str, Any]
    tti_binding_name: Optional[str] = None
    tti_binding_config: Optional[Dict[str, Any]] = None
    personality_name: Optional[str] = None


def check_ollama(host: str, model: str) -> bool:
    try:
        res = requests.get(f"{host}/api/tags", timeout=1.5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            return model in models or any(m.startswith(model) for m in models)
        return False
    except Exception:
        return False


class DummyClient:
    def __init__(self):
        self.debug = True
        self.llm = self
        self.model_name = "unknown"
        self.binding_name = "unknown"
    def count_tokens(self, text):
        return len(text) // 4
    def count_image_tokens(self, img):
        return 256
    def remove_thinking_blocks(self, text):
        return text
    def generate_text(self, prompt, **kwargs):
        return "Simulated text answer."
    def chat(self, discussion, **kwargs):
        callback = kwargs.get("streaming_callback")
        reply = (
            "Based on the loaded artifact, I can see that Page 11 introduces 'monolithic hell'. "
            "The key challenges are code complexity, slow deployment, and vertical scaling limits. "
            "Here is the page visual for your reference: <artefact_image id=\"invoice_visual::images::0\" />"
        )
        if callback:
            callback('<mem_tag id="099372fa" />', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            # Simulate a highly-fluid word-by-word streaming typing effect
            for word in reply.split(" "):
                callback(word + " ", MSG_TYPE.MSG_TYPE_CHUNK)
                time.sleep(0.04)
        return reply

needs_configuration = False
LLM_MODEL_NAME = "unknown"
tool_states = {}

# Dynamic Startup: Load active workspace from central config, defaulting to "default"
_cfg = load_app_config()
_active_ws = _cfg.get("active_workspace", "default")
initialize_workspace_state(_active_ws)

# Attach persistent multi-level memory manager so <mem_new> tags survive across turns
mem_db_path = APP_DATA_DIR / "memories.db"
mem_db_path.parent.mkdir(parents=True, exist_ok=True)
discussion._init_memory(LollmsMemoryManager(f"sqlite:///{mem_db_path}"))


@app.post("/api/import")
async def import_document(
    file: UploadFile = File(...),
    mode: str = Form("text_images"),
    title: Optional[str] = Form(None)
):
    """
    Uploads a file, writes it to a temporary path, and returns a StreamingResponse
    yielding real-time progress updates and the final output via SSE.
    """
    if mode not in ("text", "text_images", "text_embedded_images", "images_only", "ocr", "data"):
        raise HTTPException(status_code=400, detail="Invalid import mode selected.")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    art_title = title or tmp_path.stem

    async def event_generator():
        q = queue.Queue()

        def progress_callback(msg: str):
            q.put({"type": "progress", "message": msg})

        def run_import():
            try:
                progress_callback(f"Loading '{file.filename}' into memory...")
                result = discussion.import_file(
                    path=tmp_path,
                    mode=mode,
                    title=art_title,
                    activate=True,
                    progress_cb=progress_callback
                )
                progress_callback("Retrieving associated binary pages from database...")
                associated_images = discussion.artefacts.get_associated_images(art_title)
                
                # Yield final successful result
                q.put({
                    "type": "result",
                    "success": True,
                    "title": art_title,
                    "mode": mode,
                    "content": result["text_artefact"]["content"] if result["text_artefact"] else "",
                    "page_count": result["page_count"],
                    "image_count": len(associated_images),
                    "images": [
                        {
                            "id": img["id"],
                            "index": img["index"],
                            "media_type": img["media_type"]
                        }
                        for img in associated_images
                    ]
                })
            except Exception as e:
                ASCIIColors.error(f"Import thread failed: {e}")
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put(None)  # Sentinel to close generator loop

        thread = threading.Thread(target=run_import, daemon=True)
        thread.start()

        while True:
            while q.empty():
                await asyncio.sleep(0.05)
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

        # Clean up temporary file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                ASCIIColors.warning(f"Could not delete temp file: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/images/{title}/{index}")
async def get_binary_image(title: str, index: int, version: Optional[int] = None):
    """Locates and serves decoded base64 sub-images as binary stream."""
    images = discussion.artefacts.get_associated_images(title, version)
    for img in images:
        if img["index"] == index:
            import base64
            binary_data = base64.b64decode(img["data"])
            return Response(content=binary_data, media_type=img["media_type"])
            
    raise HTTPException(status_code=404, detail="Requested sub-image not found in bundle.")


@app.get("/api/workspace_files/{filename}")
async def get_workspace_file_endpoint(filename: str):
    """Dynamically serve active workspace data files for HTML widget rendering."""
    if not APP_WORKSPACE_DIR:
        raise HTTPException(status_code=400, detail="No active workspace directory.")
    
    file_path = APP_WORKSPACE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found in active workspace.")
        
    from fastapi.responses import FileResponse
    return FileResponse(str(file_path))

@app.get("/api/bundle/{title}")
async def download_bundle(title: str):
    """Packages the artifact and all of its sub-images into a self-contained portable bundle."""
    bundle = discussion.artefacts.export_artefact_bundle(title)
    if not bundle:
        raise HTTPException(status_code=404, detail="No artifact found matching that title.")
    return bundle


@app.post("/api/import_bundle")
async def import_bundle(file: UploadFile = File(...)):
    """Imports a JSON bundle file back into the active discussion."""
    try:
        content = await file.read()
        bundle_data = json.loads(content.decode("utf-8"))
        result = discussion.artefacts.import_artefact_bundle(bundle_data, activate=True)
        return {"success": True, "title": result["title"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid bundle package: {e}")


@app.get("/api/artifacts")
async def list_artifacts_endpoint():
    """Lists all artifacts in the discussion session with metadata, including active state."""
    artifacts = discussion.artefacts.list()
    # Group by title and keep only the latest version of each artifact for the list
    seen = {}
    for a in artifacts:
        t = a["title"]
        v = a.get("version", 1)
        if t not in seen or v > seen[t].get("version", 1):
            seen[t] = a

    latest_artifacts = list(seen.values())
    # Exclude ephemeral (temporary) artifacts from the user-facing sidebar list
    latest_artifacts = [a for a in latest_artifacts if not a.get("ephemeral")]

    return [
        {
            "title": a["title"],
            "type": a["type"],
            "version": a["version"],
            "language": a.get("language"),
            "size": len(a["content"]),
            "active": a.get("active", False),
            "author": a.get("author"),
            "category": a.get("category"),
            "description": a.get("description"),
            "created_at": a.get("created_at"),
            "read_only": a.get("read_only", False)
        }
        for a in latest_artifacts
    ]


@app.delete("/api/artifacts/{title}")
async def delete_artifact_endpoint(title: str):
    """Deletes an artifact and its companion images artifact completely."""
    try:
        removed_main = discussion.artefacts.remove(title)
        removed_comp = discussion.artefacts.remove(f"{title}::images")
        discussion.commit()
        return {"success": True, "removed_main": removed_main, "removed_comp": removed_comp}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/artifacts/{title}/toggle_read_only")
async def toggle_artifact_read_only_endpoint(title: str):
    """Toggles the read-only state of an artifact in place."""
    try:
        existing = discussion.artefacts.get(title)
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        new_state = not existing.get("read_only", False)
        discussion.artefacts.update(
            title=title,
            read_only=new_state,
            bump_version=False # update in place
        )
        discussion.commit()
        return {"success": True, "read_only": new_state}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/artifacts/{title}/versions/{version}")
async def delete_artifact_version_endpoint(title: str, version: int):
    """Deletes a specific version of an artifact from the database."""
    try:
        # Prevent deleting the last remaining version
        history = discussion.artefacts.get_version_history(title)
        if len(history) <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last remaining version of an artifact.")

        removed = discussion.artefacts.remove(title, version=version)
        discussion.commit()

        # If we deleted the currently active version, automatically activate the latest remaining version
        remaining = discussion.artefacts.get_version_history(title)
        if remaining:
            active_version_exists = any(h.get("is_active") for h in remaining)
            if not active_version_exists:
                latest_v = remaining[-1]["version"]
                discussion.artefacts.activate(title, latest_v)
                discussion.commit()

        return {"success": True, "removed_count": removed}
    except HTTPException:
        raise
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


class SquashRequest(BaseModel):
    keep_last_n: Optional[int] = None
    target_version: Optional[int] = None

@app.post("/api/artifacts/{title}/squash")
async def squash_artifact_versions_endpoint(title: str, payload: SquashRequest):
    """Squashes the version history of an artifact."""
    try:
        if payload.target_version is not None:
            res = discussion.artefacts.squash_versions(title, target_version=payload.target_version)
        elif payload.keep_last_n is not None:
            res = discussion.artefacts.squash_versions(title, keep_last_n=payload.keep_last_n)
        else:
            raise HTTPException(status_code=400, detail="Provide either 'target_version' or 'keep_last_n'.")

        discussion.commit()
        return {"success": True, "report": res}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/artifacts/{title}/toggle")
async def toggle_artifact_endpoint(title: str):
    """Toggles the active state of an artifact in the session context."""
    try:
        new_state = discussion.artefacts.toggle(title)
        discussion.commit()
        return {"success": True, "active": new_state}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/artifacts/{title}/images/{index}")
async def delete_artifact_image_endpoint(title: str, index: int, version: Optional[int] = None):
    """Deletes an individual image index from either the main or companion images artifact."""
    try:
        # Check both main and companion
        main_art = discussion.artefacts.get(title, version)
        comp_art = discussion.artefacts.get(f"{title}::images")

        target_art = None
        target_title = None
        target_index = None

        if comp_art and index < len(comp_art.get("images", [])):
            target_art = comp_art
            target_title = f"{title}::images"
            target_index = index
        elif main_art and index < len(main_art.get("images", [])):
            target_art = main_art
            target_title = title
            target_index = index

        if not target_art:
            raise HTTPException(status_code=404, detail="Image index not found in artifact.")

        # Remove image and media type from list
        imgs = list(target_art.get("images", []))
        mtypes = list(target_art.get("image_media_types", []))

        if 0 <= target_index < len(imgs):
            del imgs[target_index]
            if target_index < len(mtypes):
                del mtypes[target_index]

            # Update the artifact in database
            discussion.artefacts.update(
                title=target_title,
                new_images=imgs,
                new_image_media_types=mtypes,
                bump_version=False # Update in place
            )
            discussion.commit()
            return {"success": True}
        else:
            raise HTTPException(status_code=400, detail="Invalid target image index.")
    except HTTPException:
        raise
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories")
async def get_memories_endpoint():
    """Retrieves all stored memories from the persistent SQLite memory manager."""
    if not discussion.memory_manager:
        return []
    res = discussion.list_all_memories(page=1, page_size=100)
    return res.get("memories", [])

@app.delete("/api/memories/{memory_id}")
async def delete_memory_endpoint(memory_id: str):
    """Permanently deletes a memory."""
    if not discussion.memory_manager:
        raise HTTPException(status_code=400, detail="Memory manager not active.")
    try:
        ok = discussion.delete_memory(memory_id)
        return {"success": ok}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memories/{memory_id}/edit")
async def edit_memory_endpoint(memory_id: str, payload: EditMemoryRequest):
    """Edits a memory's content, importance, or level."""
    if not discussion.memory_manager:
        raise HTTPException(status_code=400, detail="Memory manager not active.")
    try:
        res = discussion.edit_memory(
            memory_id,
            content=payload.content,
            importance=payload.importance,
            level=payload.level,
        )
        if res is None:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return {"success": True, "memory": res}
    except HTTPException:
        raise
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memories/import")
async def import_memory_endpoint(payload: ImportMemoryRequest):
    """Manually imports a new memory."""
    if not discussion.memory_manager:
        raise HTTPException(status_code=400, detail="Memory manager not active.")
    try:
        res = discussion.add_memory(
            content=payload.content,
            importance=payload.importance,
            level=payload.level,
            tags=payload.tags,
        )
        if res is None:
            raise HTTPException(status_code=500, detail="Failed to create memory.")
        return {"success": True, "memory": res}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bindings/tti")
async def list_tti_bindings():
    """Discovers all TTI bindings by scanning the tti_bindings directory."""
    bindings = []
    bindings_root = PROJECT_ROOT / "src" / "lollms_client" / "tti_bindings"
    if bindings_root.exists():
        for d in sorted(bindings_root.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                desc_file = d / "description.yaml"
                data = {}
                if desc_file.exists():
                    try:
                        with open(desc_file, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f) or {}
                    except Exception:
                        pass

                unified_params = []
                for key in ("global_input_parameters", "model_input_parameters", "input_parameters", "server_parameters", "binding_parameters"):
                    if isinstance(data.get(key), list):
                        unified_params.extend(data[key])

                data["input_parameters"] = unified_params if unified_params else []
                data["binding_name"] = d.name
                data.setdefault("title", d.name.replace("_", " ").title())
                data.setdefault("description", f"A binding for {d.name}.")
                bindings.append(data)
    return {"success": True, "bindings": bindings}


@app.post("/api/bindings/tti/test")
async def test_tti_binding_endpoint(payload: BindingTestRequest):
    """Instantiates a TTI binding with user config and retrieves available models."""
    try:
        import importlib
        module_path = f"lollms_client.tti_bindings.{payload.binding_name}"
        module = importlib.import_module(module_path)
        binding_class = getattr(module, getattr(module, "BindingName", None), None)
        if not binding_class or not isinstance(binding_class, type) or inspect.isabstract(binding_class):
            return {"success": False, "error": f"Could not find a concrete Binding class in {payload.binding_name}"}

        instance = binding_class(**payload.config)
        models = []
        if hasattr(instance, "list_models") and callable(getattr(instance, "list_models")):
            models = instance.list_models()
        elif hasattr(instance, "get_zoo") and callable(getattr(instance, "get_zoo")):
            models = instance.get_zoo()

        if not isinstance(models, list):
            models = []
        return {"success": True, "models": models}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        return {"success": False, "error": str(e)}


@app.post("/api/bindings/tti/zoo")
async def get_tti_zoo_endpoint(payload: BindingTestRequest):
    """Retrieves the model zoo list for a specified TTI binding."""
    try:
        import importlib
        module_path = f"lollms_client.tti_bindings.{payload.binding_name}"
        module = importlib.import_module(module_path)
        binding_class = getattr(module, getattr(module, "BindingName", None), None)
        if not binding_class:
            return {"success": False, "error": "Binding class not found."}

        instance = binding_class(**payload.config)
        zoo = []
        if hasattr(instance, "get_zoo") and callable(getattr(instance, "get_zoo")):
            zoo = instance.get_zoo()
        return {"success": True, "zoo": zoo}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 📂 Profiles Management Endpoints ──

@app.get("/api/profiles")
async def list_profiles_endpoint():
    """Lists all saved configuration profiles under APP_DIR/profiles/"""
    profiles_dir = APP_DIR / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profiles = []
    for f in sorted(profiles_dir.glob("*.json")):
        profiles.append({"name": f.stem, "created_at": time.ctime(f.stat().st_ctime)})
    return {"success": True, "profiles": profiles}


@app.post("/api/profiles/save")
async def save_profile_endpoint(payload: ProfileRequest):
    """Saves the current configuration under a named profile."""
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Profile name cannot be empty.")

    profiles_dir = APP_DIR / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_app_config()
    profile_path = profiles_dir / f"{payload.name.strip().lower().replace(' ', '_')}.json"
    try:
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return {"success": True, "name": payload.name}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {e}")


@app.post("/api/profiles/load")
async def load_profile_endpoint(payload: ProfileRequest):
    """Hot-swaps the current active configuration with a selected profile."""
    global client, LLM_MODEL_NAME, needs_configuration
    profiles_dir = APP_DIR / "profiles"
    profile_path = profiles_dir / f"{payload.name.strip().lower().replace(' ', '_')}.json"

    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile not found.")

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_cfg = json.load(f)

        save_app_config(profile_cfg)

        # Re-initialize LollmsClient with loaded profile settings
        client_kwargs = {
            "llm_binding_name": profile_cfg.get("llm_binding_name"),
            "llm_binding_config": profile_cfg.get("llm_binding_config", {}),
            "tools_binding_name": "lcp",
            "tools_binding_config": {
                "tools_folders": [
                    str(Path("./data_workspace").resolve()),
                    str(PROJECT_ROOT / "lollms_client" / "tools_bindings" / "lcp" / "default_tools")
                ]
            }
        }
        if profile_cfg.get("tti_binding_name"):
            client_kwargs["tti_binding_name"] = profile_cfg["tti_binding_name"]
            client_kwargs["tti_binding_config"] = profile_cfg.get("tti_binding_config", {})

        client = LollmsClient(**client_kwargs)
        discussion.lollmsClient = client
        LLM_MODEL_NAME = profile_cfg.get("llm_binding_config", {}).get("model_name", "unknown")

        # Load associated personality system prompt if saved
        p_name = profile_cfg.get("personality_name")
        if p_name and "/" in p_name:
            category, persona = p_name.split("/", 1)
            soul_file = Path("./data_workspace") / "personalities" / category / persona / "SOUL.md"
            if soul_file.exists():
                content = soul_file.read_text(encoding="utf-8", errors="ignore")
                discussion.system_prompt = re.sub(r'^---\s*(.*?)\s*---', '', content, flags=re.DOTALL).strip()

        # Sync context size
        if hasattr(client, "get_ctx_size"):
            try:
                discussion.max_context_size = client.get_ctx_size()
            except Exception:
                discussion.max_context_size = 4096
        else:
            discussion.max_context_size = 4096

        needs_configuration = False
        return {
            "success": True,
            "system_prompt": discussion.system_prompt,
            "tools_imported": tools_imported,
            "skills_imported": skills_imported
        }
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/profiles/{name}")
async def delete_profile_endpoint(name: str):
    """Deletes a saved profile."""
    profiles_dir = APP_DIR / "profiles"
    profile_path = profiles_dir / f"{name.strip().lower().replace(' ', '_')}.json"
    if profile_path.exists():
        profile_path.unlink()
        return {"success": True}
    raise HTTPException(status_code=404, detail="Profile not found.")


@app.post("/api/bindings/execute_command")
async def execute_binding_command_endpoint(payload: ExecuteCommandRequest):
    """Dynamically instantiates a binding and executes one of its custom yaml commands."""
    try:
        import importlib

        # 1. Resolve and load the module based on modality
        if payload.modality == "llm":
            module_path = f"lollms_client.llm_bindings.{payload.binding_name}"
            # Read active config from app config
            cfg = load_app_config()
            active_config = cfg.get("llm_binding_config", {})
        elif payload.modality == "tti":
            module_path = f"lollms_client.tti_bindings.{payload.binding_name}"
            cfg = load_app_config()
            active_config = cfg.get("tti_binding_config", {})
        else:
            raise HTTPException(status_code=400, detail="Invalid modality specified.")

        module = importlib.import_module(module_path)
        binding_class = getattr(module, getattr(module, "BindingName", None), None)
        if not binding_class:
            raise HTTPException(status_code=404, detail="Binding class not found.")

        # 2. Instantiate binding with the combined configuration
        combined_config = {**active_config, **payload.parameters}
        instance = binding_class(**combined_config)

        # 3. Locate and execute the target method
        method = getattr(instance, payload.command_name, None)
        if not method or not callable(method):
            raise HTTPException(status_code=404, detail=f"Command '{payload.command_name}' is not implemented by this binding.")

        # Parse parameters dynamically to match method signature
        import inspect
        sig = inspect.signature(method)
        exec_params = {}

        # Stream progress updates using Server-Sent Events if supported by the command,
        # or execute synchronously for standard immediate commands.
        for k, v in payload.parameters.items():
            if k in sig.parameters:
                p_type = sig.parameters[k].annotation
                try:
                    if p_type == int: exec_params[k] = int(v)
                    elif p_type == float: exec_params[k] = float(v)
                    elif p_type == bool: exec_params[k] = v in (True, "true", "True", 1, "1")
                    else: exec_params[k] = v
                except Exception:
                    exec_params[k] = v

        # Check if progress callback is supported
        if "progress_callback" in sig.parameters:
            # For commands that download or run tasks asynchronously (like pull_model),
            # we run in a separate thread and stream progress via SSE, or run synchronously.
            # Here we run and capture the output dictionary.
            progress_history = []
            def prog_cb(status_dict):
                progress_history.append(status_dict)
            exec_params["progress_callback"] = prog_cb

            result = method(**exec_params)
            return {"success": True, "result": result, "progress_history": progress_history}
        else:
            result = method(**exec_params)
            return {"success": True, "result": result}

    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bindings/llm")
async def list_llm_bindings():
    """Discovers all LLM bindings by scanning the bindings directory."""
    bindings = []
    bindings_root = PROJECT_ROOT / "src" / "lollms_client" / "llm_bindings"
    if bindings_root.exists():
        for d in sorted(bindings_root.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                desc_file = d / "description.yaml"
                data = {}
                if desc_file.exists():
                    try:
                        with open(desc_file, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f) or {}
                    except Exception:
                        pass

                # Flatten global / model-specific / legacy parameter lists into one UI-facing list
                unified_params = []
                for key in ("global_input_parameters", "model_input_parameters", "input_parameters", "server_parameters", "binding_parameters"):
                    if isinstance(data.get(key), list):
                        unified_params.extend(data[key])

                if unified_params:
                    data["input_parameters"] = unified_params
                elif not data.get("input_parameters"):
                    fallback = LollmsLLMBindingManager._get_fallback_description(d.name)
                    data.setdefault("title", fallback.get("title"))
                    data.setdefault("description", fallback.get("description"))
                    data["input_parameters"] = fallback.get("input_parameters", [])

                data["binding_name"] = d.name
                bindings.append(data)
    return {"success": True, "bindings": bindings}


@app.post("/api/bindings/llm/test")
async def test_llm_binding_endpoint(payload: BindingTestRequest):
    """Instantiates a binding with user config and retrieves available models."""
    try:
        import importlib
        module_path = f"lollms_client.llm_bindings.{payload.binding_name}"
        module = importlib.import_module(module_path)

        # Discover the concrete binding class via the module's BindingName variable
        binding_class = getattr(module, getattr(module, "BindingName", None), None)
        if not binding_class or not isinstance(binding_class, type) or inspect.isabstract(binding_class):
            return {"success": False, "error": f"Could not find a concrete Binding class in {payload.binding_name}"}

        instance = binding_class(**payload.config)
        models = []
        if hasattr(instance, "list_models") and callable(getattr(instance, "list_models")):
            models = instance.list_models()
        elif hasattr(instance, "get_available_models") and callable(getattr(instance, "get_available_models")):
            models = instance.get_available_models()
        if not isinstance(models, list):
            models = []
        return {"success": True, "models": models}
    except Exception as e:
        trace_exception(e)
        return {"success": False, "error": str(e)}


@app.get("/api/settings")
async def get_settings_endpoint():
    """Retrieves the saved app configuration."""
    cfg = load_app_config()
    return {
        "success": True,
        "llm_binding_name": cfg.get("llm_binding_name", ""),
        "llm_binding_config": cfg.get("llm_binding_config", {}),
        "tti_binding_name": cfg.get("tti_binding_name", ""),
        "tti_binding_config": cfg.get("tti_binding_config", {}),
        "personality_name": cfg.get("personality_name", "")
    }


@app.post("/api/settings")
async def apply_settings_endpoint(payload: ApplySettingsRequest):
    """Applies new global client settings and reinitializes bindings."""
    global client, LLM_MODEL_NAME, needs_configuration
    try:
        # Resolve config parameters for LollmsClient
        kwargs = {
            "llm_binding_name": payload.llm_binding_name,
            "llm_binding_config": payload.llm_binding_config,
            "tools_binding_name": "lcp",
            "tools_binding_config": {
                "tools_folders": [
                    str(Path("./data_workspace").resolve()),
                    str(PROJECT_ROOT / "lollms_client" / "tools_bindings" / "lcp" / "default_tools")
                ]
            }
        }
        if payload.tti_binding_name:
            kwargs["tti_binding_name"] = payload.tti_binding_name
            kwargs["tti_binding_config"] = payload.tti_binding_config or {}

        client = LollmsClient(**kwargs)
        discussion.lollmsClient = client
        LLM_MODEL_NAME = payload.llm_binding_config.get("model_name", "unknown")

        # Update active discussion's context size budget
        if hasattr(client, "get_ctx_size"):
            try:
                val = client.get_ctx_size()
                discussion.max_context_size = val if (val and val > 1) else 4096
            except Exception:
                discussion.max_context_size = 4096
        else:
            discussion.max_context_size = 4096

        # Merge with existing configuration to prevent wiping out other fields like active_workspace
        current_config = load_app_config()
        current_config.update({
            "llm_binding_name": payload.llm_binding_name,
            "llm_binding_config": payload.llm_binding_config,
            "tti_binding_name": payload.tti_binding_name,
            "tti_binding_config": payload.tti_binding_config,
            "personality_name": payload.personality_name
        })
        save_app_config(current_config)
        needs_configuration = False
        return {"success": True, "model_name": LLM_MODEL_NAME}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        return {"success": False, "error": str(e)}


import shutil
import zipfile
import io

@app.get("/api/personalities")
async def list_personalities_endpoint():
    """Lists all installed personalities under ./data_workspace/personalities/"""
    personalities_dir = Path("./data_workspace") / "personalities"
    personalities_dir.mkdir(parents=True, exist_ok=True)

    result = []
    for cat_dir in sorted(personalities_dir.iterdir()):
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            for p_dir in sorted(cat_dir.iterdir()):
                if p_dir.is_dir() and not p_dir.name.startswith("."):
                    soul_file = p_dir / "SOUL.md"
                    soul_preview = ""
                    metadata = {}
                    if soul_file.exists():
                        try:
                            content = soul_file.read_text(encoding="utf-8", errors="ignore")
                            metadata = _parse_soul_metadata(content)
                            clean_content = re.sub(r'^---\s*(.*?)\s*---', '', content, flags=re.DOTALL).strip()
                            soul_preview = clean_content[:200] + ("..." if len(clean_content) > 200 else "")
                        except Exception:
                            pass

                    icon_exists = (p_dir / "assets" / "icon.png").exists()
                    icon_url = f"/api/personalities/{cat_dir.name}/{p_dir.name}/icon" if icon_exists else None

                    result.append({
                        "category": cat_dir.name,
                        "name": p_dir.name,
                        "title": metadata.get("name") or p_dir.name.replace("_", " ").title(),
                        "description": metadata.get("description", soul_preview),
                        "author": metadata.get("author", "Unknown"),
                        "version": metadata.get("version", "1.0.0"),
                        "icon_url": icon_url
                    })
    return {"success": True, "personalities": result}


@app.get("/api/personalities/{category}/{persona}/icon")
async def get_personality_icon(category: str, persona: str):
    """Serves the avatar icon for the specified personality."""
    icon_path = Path("./data_workspace") / "personalities" / category / persona / "assets" / "icon.png"
    if icon_path.exists():
        return Response(content=icon_path.read_bytes(), media_type="image/png")

    transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=transparent_png, media_type="image/png")


@app.post("/api/personalities/select")
async def select_personality_endpoint(payload: SelectPersonalityRequest):
    """Activates a personality, loading its SOUL.md system prompt, tools, and skills."""
    global client
    category = payload.category
    persona = payload.persona
    p_dir = Path("./data_workspace") / "personalities" / category / persona
    soul_file = p_dir / "SOUL.md"

    if not p_dir.exists():
        raise HTTPException(status_code=404, detail="Personality not found.")

    try:
        soul_content = ""
        metadata = {}
        if soul_file.exists():
            soul_content = soul_file.read_text(encoding="utf-8", errors="ignore")
            metadata = _parse_soul_metadata(soul_content)
            soul_content = re.sub(r'^---\s*(.*?)\s*---', '', soul_content, flags=re.DOTALL).strip()

        discussion.system_prompt = soul_content or f"You are {persona}, a helpful assistant."

        cfg = load_app_config()
        cfg["personality_name"] = f"{category}/{persona}"
        save_app_config(cfg)

        tools_dir = p_dir / "tools"
        tools_imported = []
        if tools_dir.exists() and tools_dir.is_dir():
            dest_tools_dir = Path("./data_workspace")
            dest_tools_dir.mkdir(parents=True, exist_ok=True)
            for py_file in tools_dir.glob("*.py"):
                shutil.copy(str(py_file), str(dest_tools_dir / py_file.name))
                tools_imported.append(py_file.name)

            lcp_binding = getattr(client, "tools", None)
            if lcp_binding and hasattr(lcp_binding, "_discover_local_tools"):
                lcp_binding._discover_local_tools()

        skills_dir = p_dir / "skills"
        skills_imported = []
        if skills_dir.exists() and skills_dir.is_dir():
            for skill_folder in skills_dir.iterdir():
                if skill_folder.is_dir():
                    skill_file = skill_folder / "SKILL.md"
                    if skill_file.exists():
                        try:
                            content = skill_file.read_text(encoding="utf-8", errors="ignore")
                            art = discussion._import_save_text(skill_folder.name, content, skill_file, activate=True)
                            skills_imported.append(art["title"])
                        except Exception as ex:
                            ASCIIColors.warning(f"Failed to load skill {skill_folder.name}: {ex}")
            discussion.commit()

        return {
            "success": True,
            "system_prompt": discussion.system_prompt,
            "tools_imported": tools_imported,
            "skills_imported": skills_imported
        }
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/personalities/import")
async def import_personality_endpoint(file: UploadFile = File(...)):
    """Imports a zipped personality into the data_workspace/personalities folder."""
    suffix = Path(file.filename).suffix
    if suffix != ".zip":
        raise HTTPException(status_code=400, detail="Only .zip archives are supported for personality import.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path) as z:
            soul_entry = next((name for name in z.namelist() if "SOUL.md" in name), None)
            if not soul_entry:
                soul_content = "---\ncategory: custom_personalities\nname: " + Path(file.filename).stem + "\n---\n# " + Path(file.filename).stem.replace("_", " ").title() + "\nA custom imported personality."
            else:
                with z.open(soul_entry) as sf:
                    soul_content = sf.read().decode("utf-8", errors="ignore")

            metadata = _parse_soul_metadata(soul_content)
            category = metadata.get("category", "custom_personalities").strip().lower()
            persona = metadata.get("name", Path(file.filename).stem).strip().lower()

            dest_dir = Path("./data_workspace") / "personalities" / category / persona
            dest_dir.mkdir(parents=True, exist_ok=True)

            prefix = ""
            for name in z.namelist():
                if name.endswith("SOUL.md") and name != "SOUL.md":
                    prefix = name[:-len("SOUL.md")]
                    break

            for name in z.namelist():
                if name.endswith("/") or not name.startswith(prefix):
                    continue
                rel_path = name[len(prefix):]
                if not rel_path:
                    continue
                target_file = dest_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, open(target_file, "wb") as target:
                    shutil.copyfileobj(source, target)

            if not (dest_dir / "SOUL.md").exists():
                (dest_dir / "SOUL.md").write_text(soul_content, encoding="utf-8")

        return {"success": True, "category": category, "persona": persona}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/api/personalities/{category}/{persona}/export")
async def export_personality_endpoint(category: str, persona: str):
    """Packages a local personality as a self-contained ZIP archive for download."""
    p_dir = Path("./data_workspace") / "personalities" / category / persona
    if not p_dir.exists():
        raise HTTPException(status_code=404, detail="Personality not found.")

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(str(p_dir)):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(p_dir)
                    z.write(str(file_path), str(rel_path))

        zip_buffer.seek(0)
        filename = f"{persona}_personality.zip"
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/personalities/download_zoo")
async def download_zoo_personality(payload: DownloadZooRequest):
    """Downloads and extracts a specific personality folder from the lollms_personalities_zoo repository."""
    category = payload.category
    persona = payload.persona
    try:
        url = "https://github.com/ParisNeo/lollms_personalities_zoo/archive/refs/heads/main.zip"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        zip_data = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_data) as z:
            prefix_in_zip = f"lollms_personalities_zoo-main/{category}/{persona}/"
            matched = [name for name in z.namelist() if name.startswith(prefix_in_zip)]

            if not matched:
                raise HTTPException(status_code=404, detail=f"Personality '{category}/{persona}' not found in the Zoo repository.")

            dest_dir = Path("./data_workspace") / "personalities" / category / persona
            dest_dir.mkdir(parents=True, exist_ok=True)

            for name in matched:
                rel_path = name[len(prefix_in_zip):]
                if not rel_path or rel_path.endswith("/"):
                    continue
                target_file = dest_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, open(target_file, "wb") as target:
                    shutil.copyfileobj(source, target)

            soul_file = dest_dir / "SOUL.md"
            if not soul_file.exists():
                config_file = dest_dir / "config.yaml"
                desc = "A downloaded zoo personality."
                if config_file.exists():
                    try:
                        cfg_data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                        desc = cfg_data.get("personality_description", desc)
                    except:
                        pass
                soul_content = f"---\ncategory: {category}\nname: {persona}\ndescription: {desc}\n---\n# {persona.replace('_', ' ').title()}\n{desc}"
                soul_file.write_text(soul_content, encoding="utf-8")

        return {"success": True, "category": category, "persona": persona}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat & Artifact Versioning Endpoints ──

@app.post("/api/web/search")
async def search_web_endpoint(payload: WebSearchRequest):
    results = discussion.search_web(payload.query, payload.provider)
    return {"success": True, "results": results}


@app.post("/api/wikipedia/search")
async def search_wikipedia_endpoint(payload: WikipediaSearchRequest):
    results = discussion.search_wikipedia(payload.query)
    return {"success": True, "results": results}


@app.post("/api/wikipedia/import")
async def import_wikipedia_endpoint(payload: WikipediaImportSelectedRequest):
    try:
        for item in payload.items:
            discussion.import_wikipedia(item.title, item.url, payload.auto_load)
        discussion.commit()
        return {"success": True}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arxiv/search")
async def search_arxiv_endpoint(payload: ArxivSearchRequest):
    results = discussion.search_arxiv(payload.query, payload.author, payload.year, payload.max_results)
    return {"success": True, "results": results}


@app.post("/api/arxiv/import")
async def import_arxiv_endpoint(payload: ArxivImportSelectedRequest):
    try:
        for item in payload.items:
            discussion.import_arxiv(item.id, item.mode, payload.auto_load)
        discussion.commit()
        return {"success": True}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/search")
async def search_github_endpoint(payload: GithubSearchRequest):
    results = discussion.search_github(payload.query)
    return {"success": True, "results": results}


@app.post("/api/github/import")
async def import_github_endpoint(payload: GithubImportRequest):
    art = discussion.import_github(payload.url, payload.auto_load)
    if not art:
        raise HTTPException(status_code=400, detail="Failed to import GitHub content.")
    return {"success": True, "title": art["title"]}


@app.post("/api/stackoverflow/search")
async def search_stackoverflow_endpoint(payload: StackOverflowSearchRequest):
    results = discussion.search_stackoverflow(payload.query)
    return {"success": True, "results": results}


@app.post("/api/stackoverflow/import")
async def import_stackoverflow_endpoint(payload: StackOverflowImportRequest):
    art = discussion.import_stackoverflow(payload.url, payload.auto_load)
    if not art:
        raise HTTPException(status_code=400, detail="Failed to import StackOverflow content.")
    return {"success": True, "title": art["title"]}


@app.post("/api/youtube/import")
async def import_youtube_endpoint(payload: YoutubeImportRequest):
    art = discussion.import_youtube_transcript(payload.url, payload.language, payload.auto_load)
    if not art:
        raise HTTPException(status_code=400, detail="Failed to import YouTube video transcript.")
    return {"success": True, "title": art["title"]}


# ── Chat & Artifact Versioning Endpoints ──

@app.post("/api/artifacts/{title}/update")
async def update_artifact_endpoint(title: str, payload: UpdateArtifactRequest):
    """Manually updates the content of any named artifact, incrementing its version."""
    try:
        # Retrieve the existing artifact to preserve its type and active state
        existing = discussion.artefacts.get(title)
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        res = discussion.artefacts.update(
            title=title,
            new_content=payload.content.strip(),
            bump_version=True,
            active=existing.get("active", True),
            commit_message=payload.commit_message
        )
        discussion.commit()
        return {"success": True, "version": res.get("version", 1)}
    except HTTPException:
        raise
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/artifacts/{title}/select_version")
async def select_artifact_version_endpoint(title: str, version: int):
    """Activates a specific version of an artifact, deactivating all other versions in the context."""
    try:
        discussion.artefacts.activate(title, version)
        discussion.commit()

        # Update active unversioned file on disk for data artifacts
        active = discussion.artefacts.get(title, version)
        if active and active.get("type") == "data":
            ext = active.get("file_ext", ".csv")
            workspace_dir = APP_WORKSPACE_DIR
            versioned_path = workspace_dir / f"{title}_v{version}{ext}"
            unversioned_path = workspace_dir / f"{title}{ext}"
            if versioned_path.exists():
                import shutil
                try:
                    shutil.copy(str(versioned_path), str(unversioned_path))
                except Exception as e:
                    ASCIIColors.warning(f"Failed to update active unversioned file on version select: {e}")

        return {"success": True, "active_version": version}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/artifacts/{title}/update_image")
async def update_image_artifact_endpoint(title: str, payload: UpdateImageArtifactRequest):
    """Updates the image data of a named image-type artifact, incrementing its version."""
    try:
        existing = discussion.artefacts.get(title)
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        b64_data = payload.image_b64.split(";base64,")[1] if ";base64," in payload.image_b64 else payload.image_b64

        # Update the image list (overwriting the first image)
        res = discussion.artefacts.update(
            title=title,
            new_images=[b64_data],
            bump_version=True,
            active=existing.get("active", True),
            commit_message=payload.commit_message
        )
        discussion.commit()
        return {"success": True, "version": res.get("version", 1)}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/artifacts/{title}")
async def get_artifact_version(title: str, version: Optional[int] = None):
    """Retrieves the content and metadata of a specific version of an artifact."""
    art = discussion.artefacts.get(title, version)
    if not art:
        raise HTTPException(status_code=404, detail="Artifact version not found.")
    return art


@app.get("/api/artifacts/{title}/history")
async def get_artifact_history(title: str):
    """Retrieves the version history log for a specific artifact."""
    history = discussion.artefacts.get_version_history(title)
    if not history:
        raise HTTPException(status_code=404, detail="No version history found for this artifact.")
    return history


@app.get("/api/data/{title}")
async def get_data_grid(title: str, version: Optional[int] = None):
    """Loads a specific versioned data artifact spreadsheet or SQLite DB."""
    active = discussion.artefacts.get(title, version)
    if not active or active.get("type") != "data":
        raise HTTPException(status_code=404, detail="Data artifact not found.")

    ext = active.get("file_ext", ".csv")
    current_version = active.get("version", 1)
    workspace_dir = APP_WORKSPACE_DIR
    file_path = workspace_dir / f"{title}_v{current_version}{ext}"
    if not file_path.exists():
        # Fallback scan: check other workspaces and the global data_workspace
        found_path = None
        scan_dirs = [Path("./data_workspace")]
        if APP_DIR and APP_DIR.exists():
            ws_dir = APP_DIR / "workspaces"
            if ws_dir.exists():
                for d in ws_dir.iterdir():
                    if d.is_dir():
                        scan_dirs.append(d / "data_workspace")
        for sd in scan_dirs:
            cand = sd / f"{title}_v{current_version}{ext}"
            if cand.exists():
                found_path = cand
                break
        if found_path:
            import shutil
            try:
                workspace_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(found_path), str(file_path))
                unversioned_path = workspace_dir / f"{title}{ext}"
                try:
                    shutil.copy(str(found_path), str(unversioned_path))
                except Exception:
                    pass
                ASCIIColors.success(f"✓ Recovered missing data file for grid: {file_path}")
            except Exception:
                pass

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Raw data file missing from workspace.")

    import pandas as pd
    try:
        if ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(str(file_path))
            result = {"type": "excel", "sheets": {}}
            for sheet in xl.sheet_names:
                df = pd.read_excel(str(file_path), sheet_name=sheet).head(100)
                df = df.replace({float('nan'): None, float('inf'): None, float('-inf'): None})
                result["sheets"][sheet] = {
                    "columns": list(df.columns),
                    "rows": df.to_dict(orient="records")
                }
        elif ext in (".db", ".sqlite", ".sqlite3"):
            import sqlite3
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            result = {"type": "sqlite", "sheets": {}}
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 100;", conn)
                df = df.replace({float('nan'): None, float('inf'): None, float('-inf'): None})
                result["sheets"][table] = {
                    "columns": list(df.columns),
                    "rows": df.to_dict(orient="records")
                }
            conn.close()
        else:
            sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
            df = pd.read_csv(str(file_path), sep=sep).head(100)
            df = df.replace({float('nan'): None, float('inf'): None, float('-inf'): None})
            result = {
                "type": "csv",
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records")
            }
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse dataset: {e}")


@app.post("/api/save_tool")
async def save_tool_endpoint(payload: SaveToolRequest):
    """Saves a tool to the local LCP directory and versions its artifact in SQLite."""
    try:
        # 1. Always save user-created tools to the workspace data folder
        tools_dir = APP_WORKSPACE_DIR
        tools_dir.mkdir(parents=True, exist_ok=True)

        lcp_binding = getattr(client, "tools", None)

        tool_name = payload.title.replace("_tool", "").strip().lower()
        py_file = Path(tools_dir) / f"{tool_name}.py"

        # Write flat file to disk
        py_file.write_text(payload.code, encoding="utf-8")

        # Reload LCP tools list so it is instantly execution-ready
        if lcp_binding and hasattr(lcp_binding, "_discover_local_tools"):
            lcp_binding._discover_local_tools()

        # 2. Update the session artifact version in SQLite
        art_content = (
            f"# Smart Tool: {tool_name}\n"
            f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
            f"### Python Implementation (`{tool_name}.py`)\n"
            f"```python\n{payload.code}\n```"
        )

        art = discussion.artefacts.get(f"{tool_name}_tool")
        if art is None:
            res_art = discussion.artefacts.add(
                title=f"{tool_name}_tool",
                artefact_type="tool",
                content=art_content,
                active=True,
                commit_message=payload.commit_message
            )
        else:
            res_art = discussion.artefacts.update(
                title=f"{tool_name}_tool",
                new_content=art_content,
                new_type="tool",
                active=True,
                commit_message=payload.commit_message
            )
        discussion.commit()
        return {"success": True, "title": f"{tool_name}_tool", "version": res_art.get("version", 1)}
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refine_tool")
async def refine_tool_endpoint(payload: RefineToolRequest):
    """Uses the LLM to securely edit and add features to the tool code based on instructions and reference docs."""
    doc_context = "\n\n".join(payload.docs) if payload.docs else "No reference documentation provided."
    prompt = (
        "You are an Expert Python LCP Tool Developer.\n"
        "Refine and update the following Python tool code according to the instruction.\n"
        "You MUST also study the provided reference documentation to ensure correct API usage, constraints, and architecture.\n\n"
        "=== REFERENCE DOCUMENTATION ===\n"
        f"{doc_context}\n"
        "=== END REFERENCE DOCUMENTATION ===\n\n"
        f"Instruction: \"{payload.instruction}\"\n\n"
        "CURRENT TOOL CODE:\n"
        "```python\n"
        f"{payload.code}\n"
        "```\n\n"
        "Requirements:\n"
        "1. Output ONLY valid, complete, and optimized Python code.\n"
        "2. Keep the function name starting with 'tool_' matching the original.\n"
        "3. Preserve all import libraries, docstring annotations, type-hints, and parameters unless the instruction asks to modify them.\n"
        "4. Do NOT wrap inside markdown fences, do NOT write explanations outside the code block."
    )
    try:
        raw_res = client.generate_text(
            prompt=prompt,
            temperature=0.2
        ).strip()

        # Clean any leaked markdown wraps
        clean_res = re.sub(r'^```python\s*|\s*```$', '', raw_res, flags=re.IGNORECASE).strip()
        return {"success": True, "code": clean_res}
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run_tool_init")
async def run_tool_init_endpoint(payload: RunToolInitRequest):
    """Runs the init_tool_library() function from the provided code and redirects stdout."""
    import sys
    import io
    local_vars = {}
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(payload.code, {}, local_vars)
        init_fn = local_vars.get("init_tool_library")
        if not init_fn:
            return {"success": True, "output": "No 'init_tool_library()' function found (not required)."}
        init_fn()
        return {"success": True, "output": redirected_output.getvalue() or "init_tool_library() executed successfully (no stdout)."}
    except Exception as e:
        return {"success": False, "error": str(e), "output": redirected_output.getvalue()}
    finally:
        sys.stdout = old_stdout


@app.post("/api/parse_tool_functions")
async def parse_tool_functions_endpoint(payload: ParseToolFunctionsRequest):
    """Uses Python's AST module to extract all callable functions, parameters, type annotations, and docstrings."""
    import ast
    try:
        tree = ast.parse(payload.code)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == "init_tool_library":
                    continue

                doc = ast.get_docstring(node) or "No description."
                params = []
                args_list = node.args.args
                defaults_list = node.args.defaults
                defaults_offset = len(args_list) - len(defaults_list) if defaults_list else len(args_list)

                for idx, arg in enumerate(args_list):
                    arg_name = arg.arg
                    if arg_name in ("lollms_client_instance", "client", "discussion_instance", "discussion"):
                        continue

                    arg_type = "str"
                    if arg.annotation:
                        anno = ast.unparse(arg.annotation).strip().lower()
                        if "int" in anno: arg_type = "int"
                        elif "float" in anno: arg_type = "float"
                        elif "bool" in anno: arg_type = "bool"
                        elif "dict" in anno: arg_type = "dict"
                        elif "list" in anno: arg_type = "list"

                    has_default = idx >= defaults_offset
                    default_val = None
                    if has_default and defaults_list:
                        try:
                            default_val = ast.literal_eval(defaults_list[idx - defaults_offset])
                        except:
                            default_val = ast.unparse(defaults_list[idx - defaults_offset]).strip("'\"")

                    params.append({
                        "name": arg_name,
                        "type": arg_type,
                        "required": not has_default,
                        "default": default_val
                    })

                functions.append({
                    "name": node.name,
                    "docstring": doc,
                    "parameters": params
                })
        return {"success": True, "functions": functions}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/execute_tool_function")
async def execute_tool_function_endpoint(payload: ExecuteToolFunctionRequest):
    """Executes a specific function from the provided Python code with custom inputs inside a redirected output sandbox."""
    import sys
    import io
    local_vars = {}
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(payload.code, {}, local_vars)
        fn = local_vars.get(payload.function_name)
        if not fn:
            raise ValueError(f"Function '{payload.function_name}' not found in the provided code.")

        import inspect
        sig = inspect.signature(fn)
        exec_params = {}
        if 'params' in sig.parameters:
            exec_params['params'] = payload.params
        elif 'args' in sig.parameters:
            exec_params['args'] = payload.params
        else:
            for k, v in payload.params.items():
                if k in sig.parameters:
                    p_type = sig.parameters[k].annotation
                    try:
                        if p_type == int: exec_params[k] = int(v)
                        elif p_type == float: exec_params[k] = float(v)
                        elif p_type == bool: exec_params[k] = v in (True, "true", "True", 1, "1")
                        else: exec_params[k] = v
                    except:
                        exec_params[k] = v

        if 'lollms_client_instance' in sig.parameters:
            exec_params['lollms_client_instance'] = client
        elif 'client' in sig.parameters:
            exec_params['client'] = client
        if 'discussion_instance' in sig.parameters:
            exec_params['discussion_instance'] = discussion
        elif 'discussion' in sig.parameters:
            exec_params['discussion'] = discussion

        result = fn(**exec_params)
        return {
            "success": True,
            "output": redirected_output.getvalue(),
            "result": result
        }
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        return {
            "success": False,
            "error": str(e),
            "output": redirected_output.getvalue()
        }
    finally:
        sys.stdout = old_stdout


class ExecuteSandboxRequest(BaseModel):
    code: str
    language: str = "python"


@app.post("/api/execute_sandbox")
async def execute_sandbox_endpoint(payload: ExecuteSandboxRequest):
    """Executes arbitrary Python code in a safe stdout-capturing sandbox, or previews HTML."""
    if payload.language == "python":
        import sys
        import io
        import os
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        # Ensure active unversioned copies exist for all data artifacts in the workspace
        if discussion and APP_WORKSPACE_DIR:
            try:
                for art in discussion.artefacts.list(active_only=True):
                    if art.get("type") == "data":
                        title = art["title"]
                        ext = art.get("file_ext", ".csv")
                        version = art.get("version", 1)
                        versioned_path = APP_WORKSPACE_DIR / f"{title}_v{version}{ext}"
                        unversioned_path = APP_WORKSPACE_DIR / f"{title}{ext}"
                        if versioned_path.exists() and not unversioned_path.exists():
                            import shutil
                            shutil.copy(str(versioned_path), str(unversioned_path))
            except Exception as ex:
                if client and client.debug:
                    trace_exception(ex)

        old_cwd = os.getcwd()
        try:
            if APP_WORKSPACE_DIR and APP_WORKSPACE_DIR.exists():
                os.chdir(str(APP_WORKSPACE_DIR))
            exec(payload.code, {}, {})
            return {"success": True, "output": redirected_output.getvalue()}
        except Exception as e:
            if client and client.debug:
                trace_exception(e)
            return {"success": False, "error": str(e), "output": redirected_output.getvalue()}
        finally:
            sys.stdout = old_stdout
            try:
                os.chdir(old_cwd)
            except Exception:
                pass
    else:
        return {"success": True, "output": ""}


@app.post("/api/count_tool_tokens")
async def count_tool_tokens_endpoint(payload: CountToolTokensRequest):
    """Calculates the exact token footprint of the Coder's prompt context."""
    try:
        doc_context = "\n\n".join(payload.docs) if payload.docs else "None"
        prompt_content = f"Instruction: {payload.instruction}\n\nContext Docs:\n{doc_context}\n\nCode:\n{payload.code}"
        tokens = client.count_tokens(prompt_content)
        return {"success": True, "tokens": tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools")
async def list_discovered_tools_endpoint():
    """Returns the list of all active/discovered local LCP tools with their schemas."""
    lcp_binding = getattr(client, "tools", None)
    if not lcp_binding:
        return []
    try:
        tools = lcp_binding.list_tools()
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", {}),
                "active": tool_states.get(t["name"], False)
            }
            for t in tools
        ]
    except Exception as e:
        trace_exception(e)
        return []


@app.post("/api/tools/{name}/toggle")
async def toggle_tool_endpoint(name: str):
    """Toggles the active state of a discovered LCP tool."""
    try:
        current = tool_states.get(name, True)
        tool_states[name] = not current
        return {"success": True, "active": not current}
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export_memories")
async def export_memories_endpoint(payload: ExportMemoriesRequest):
    """Exports and formats the persistent SQLite memories based on target scope and format."""
    if not discussion.memory_manager:
        raise HTTPException(status_code=400, detail="Memory manager is not active in this session.")
    try:
        lvl = None
        if payload.scope == "working": lvl = 1
        elif payload.scope == "deep": lvl = 2
        elif payload.scope == "archived": lvl = 3

        res = discussion.list_all_memories(level=lvl, page_size=1000)
        mems = res.get("memories", [])

        if payload.format == "csv":
            import csv
            import io
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=["id", "level", "importance", "content", "tags", "use_count"])
            w.writeheader()
            for m in mems:
                w.writerow({
                    "id": m["id"],
                    "level": m["level"],
                    "importance": m["importance"],
                    "content": m["content"],
                    "tags": m["tags"] or "",
                    "use_count": m["use_count"]
                })
            return Response(
                content=out.getvalue().encode("utf-8"),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=memories_export.csv"}
            )
        else:
            return Response(
                content=json.dumps(mems, indent=2, default=str),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=memories_export.json"}
            )
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export_tools")
async def export_tools_endpoint(payload: ExportToolsRequest):
    """Packages and downloads selected discovered local LCP tools as .py files in a .zip, or JSON bundle."""
    lcp_binding = getattr(client, "tools", None)
    if not lcp_binding:
        raise HTTPException(status_code=400, detail="LCP Tools binding is not active.")
    try:
        discovered = lcp_binding.list_tools()
        matched = [t for t in discovered if t["name"] in payload.tools]

        if payload.format == "json":
            return Response(
                content=json.dumps(matched, indent=2, default=str),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=tools_export.json"}
            )
        else:
            import zipfile
            import io
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for t in matched:
                    py_path_str = t.get("_python_file_path")
                    if py_path_str:
                        py_path = Path(py_path_str)
                        if py_path.exists():
                            zip_file.writestr(py_path.name, py_path.read_text(encoding="utf-8"))
            zip_buffer.seek(0)
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=tools_export.zip"}
            )
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query_data")
async def query_data_endpoint(payload: DataQueryRequest):
    """Executes a secure Python code block and versions results."""
    active = discussion.artefacts.get(payload.title)
    if not active or active.get("type") != "data":
        raise HTTPException(status_code=404, detail="Data artifact not found.")

    ext = active.get("file_ext", ".csv")
    current_version = active.get("version", 1)
    workspace_dir = APP_WORKSPACE_DIR
    file_path = workspace_dir / f"{payload.title}_v{current_version}{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Current version data file missing.")

    import pandas as pd
    import numpy as np
    import base64
    import io
    import sys
    import shutil
    import matplotlib
    matplotlib.use('Agg')  # Prevent GUI loop thread errors on servers
    import matplotlib.pyplot as plt

    local_vars = {
        "pd": pd,
        "plt": plt,
        "np": np,
        "Path": Path
    }

    sep = ","
    try:
        if ext in (".db", ".sqlite", ".sqlite3"):
            shutil.copy(str(source_file_path), str(new_file_path))
            import sqlite3
            conn = sqlite3.connect(str(new_file_path))
            local_vars["conn"] = conn
            local_vars["cursor"] = conn.cursor()
        elif ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(str(source_file_path))
            dfs = {sheet: pd.read_excel(str(source_file_path), sheet_name=sheet) for sheet in xl.sheet_names}
            local_vars["dfs"] = dfs
            if len(dfs) == 1:
                local_vars["df"] = list(dfs.values())[0]
        else:
            sep = ";" if ext == ".csv" and ";" in source_file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
            local_vars["df"] = pd.read_csv(str(source_file_path), sep=sep)
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    # Redirect stdout to capture prints
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    plot_b64 = None
    try:
        plt.clf()
        plt.close('all')

        # Execute code in local variables context
        exec(payload.code, {}, local_vars)
        
        # Check if any plot figure was generated
        fig_nums = plt.get_fignums()
        if fig_nums:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
        # If SQLite, commit and close
        if ext in (".db", ".sqlite", ".sqlite3") and "conn" in local_vars:
            local_vars["conn"].commit()
            local_vars["conn"].close()
        # If CSV/Excel, write DataFrame back to the new versioned file
        elif ext in (".xlsx", ".xls") and "dfs" in local_vars:
            with pd.ExcelWriter(new_file_path, engine="openpyxl") as writer:
                for sheet_name, sheet_df in local_vars["dfs"].items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        elif "df" in local_vars:
            local_vars["df"].to_csv(new_file_path, index=False, sep=sep)

        # Parse new schema and dynamically update/version the artifact
        from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
        new_schema, _ = _parse_data_file(new_file_path, payload.title, version=new_version, progress_cb=None)
        discussion.artefacts.update(
            title=payload.title,
            new_content=new_schema,
            new_type="data",
            active=True,
            file_ext=ext,
            version=new_version
        )
    except Exception as e:
        sys.stdout = old_stdout
        # Clean up any partial files if failed
        if new_file_path.exists() and ext not in (".db", ".sqlite", ".sqlite3"):
            new_file_path.unlink()
        if client and client.debug:
            trace_exception(e)
        return {"success": False, "error": str(e), "output": redirected_output.getvalue()}
    finally:
        sys.stdout = old_stdout

    return {
        "success": True,
        "output": redirected_output.getvalue(),
        "plot_b64": plot_b64,
        "new_version": new_version
    }


# ── Chat Request Payload ──
class ChatRequest(BaseModel):
    message: str
    regenerate: Optional[bool] = False
    images: Optional[List[str]] = None
    enable_memory: Optional[bool] = True
    enable_artefacts: Optional[bool] = True
    enable_in_message_status: Optional[bool] = True
    enable_presentations: Optional[bool] = True
    enable_books: Optional[bool] = True
    enable_skills: Optional[bool] = True
    enable_image_generation: Optional[bool] = True
    enable_image_editing: Optional[bool] = True
    enable_forms: Optional[bool] = True
    enable_inline_widgets: Optional[bool] = True
    enable_memory: Optional[bool] = True
    enable_artefacts: Optional[bool] = True
    enable_in_message_status: Optional[bool] = True
    enable_presentations: Optional[bool] = True
    enable_books: Optional[bool] = True
    enable_skills: Optional[bool] = True
    enable_image_generation: Optional[bool] = True
    enable_image_editing: Optional[bool] = True
    enable_forms: Optional[bool] = True


class ScanSkillsRequest(BaseModel):
    path: str


@app.post("/api/scan_skills")
async def scan_skills_directory(payload: ScanSkillsRequest):
    """Recursively scans a directory for Markdown skills and ingests them."""
    target_path = Path(payload.path)
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Provided path is not a valid directory.")

    imported_skills = []
    for md_file in target_path.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            art = discussion._import_save_text(md_file.stem, content, md_file, activate=True)
            if art.get("type") == "skill":
                imported_skills.append({
                    "title": art["title"],
                    "category": art.get("category"),
                    "version": art.get("version"),
                    "author": art.get("author")
                })
        except Exception as e:
            ASCIIColors.warning(f"Failed to scan/ingest skill file '{md_file.name}': {e}")

    return {"success": True, "count": len(imported_skills), "skills": imported_skills}


import re

def _execute_sequential_reading_subroutine(discussion, user_message: str, max_tokens: int, streaming_callback: Optional[Callable] = None) -> List[str]:
    """
    Sequentially reads active artifacts in chunks, extracting relevant information
    for the user_message using LLM calls. Saves a synthesized summary to the scratchpad
    and deactivates the source artifacts from the prompt context.
    """
    active_artifacts = discussion.artefacts.list(active_only=True)
    tokenizer = discussion.lollmsClient.count_tokens
    detokenizer = discussion.lollmsClient.detokenize
    
    extracted_knowledge = []
    deactivated_titles = []
    
    # 1. Start processing box in the UI
    if streaming_callback:
        try:
            streaming_callback("", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_open",
                "title": "Sequential Cognitive Scan"
            })
        except Exception as e:
            ASCIIColors.warning(f"Failed to open processing block: {e}")

    for art in active_artifacts:
        content = art.get("content", "").strip()
        if not content:
            continue
            
        art_tokens = tokenizer(content)
        
        # Target artifacts that are large or contribute to context budget overflow
        if art_tokens > 2000:
            msg = f"Large document detected: '{art['title']}' ({art_tokens:,} tokens). Initiating scan..."
            ASCIIColors.warning(f"[Sequential Reader] {msg}")
            if streaming_callback:
                try:
                    streaming_callback("", MSG_TYPE.MSG_TYPE_CHUNK, {
                        "type": "progress",
                        "message": msg
                    })
                except Exception:
                    pass
            
            if not hasattr(discussion, "deactivated_contents"):
                discussion.deactivated_contents = set()
            discussion.deactivated_contents.add(art["title"])
            deactivated_titles.append(art["title"])
            
            # Chunk into roughly 1000-token blocks
            tokens = discussion.lollmsClient.tokenize(content)
            chunk_size = 1000
            chunks = []
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunks.append(detokenizer(chunk_tokens))
            
            # Scan chunks sequentially
            art_knowledge = []
            for idx, chunk_text in enumerate(chunks):
                status_msg = f"Scanning chunk {idx+1}/{len(chunks)} of '{art['title']}'..."
                ASCIIColors.info(f"  ↳ {status_msg}")
                if streaming_callback:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_CHUNK, {
                            "type": "processing_status",
                            "status": status_msg
                        })
                    except Exception:
                        pass

                prompt = (
                    "You are a precise Information Extraction Subroutine.\n"
                    "Read this document fragment and extract only the specific facts, data points, or "
                    f"sections that are directly relevant to answering the query: '{user_message}'\n"
                    "Do not synthesize an answer yet—just pull the relevant raw information.\n"
                    "If the fragment contains nothing relevant, reply with exactly 'None'.\n\n"
                    f"--- FRAGMENT ---\n{chunk_text}\n\n"
                    "EXTRACTED FACTS:"
                )
                try:
                    extracted = discussion.lollmsClient.generate_text(
                        prompt=prompt,
                        temperature=0.1,
                        n_predict=300
                    ).strip()
                    if extracted and extracted.lower() != "none" and len(extracted) > 10:
                        extracted_knowledge.append(f"- **From '{art['title']}' (Part {idx+1})**:\n{extracted}")
                        art_knowledge.append(f"- **Part {idx+1}**:\n{extracted}")
                except Exception as ex:
                    ASCIIColors.warning(f"Failed to scan chunk {idx+1}: {ex}")

            if art_knowledge:
                if not hasattr(discussion, "sequential_summaries"):
                    discussion.sequential_summaries = {}
                discussion.sequential_summaries[art["title"]] = "\n\n".join(art_knowledge)

    # 2. Close processing box in the UI
    if streaming_callback:
        try:
            streaming_callback("", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_close"
            })
        except Exception as e:
            ASCIIColors.warning(f"Failed to close processing block: {e}")

    synthesis = ""
    if extracted_knowledge:
        synthesis = (
            "=== COGNITIVE SEQUENTIAL READING SYNTHESIS ===\n"
            "The original documents were too large for the context. "
            "The system ran a sequential scan and extracted these relevant facts:\n\n"
            + "\n\n".join(extracted_knowledge)
            + "\n=== END SYNTHESIS ==="
        )
    else:
        synthesis = "=== COGNITIVE SEQUENTIAL READING ===\nNo highly relevant matches found during the sequential scan."
        
    return deactivated_titles, synthesis


@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    """Streams conversational chat turns, integrating active memories and artifacts."""
    user_message = request.message
    if not request.regenerate and not user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    async def chat_event_generator():
        q = queue.Queue()

        def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None):
            q.put({
                "chunk": chunk,
                "msg_type": msg_type.name,
                "meta": meta or {}
            })
            return True

        def run_chat_thread():
            # Reset cancel event state prior to generation starting
            if client and getattr(client, "llm", None):
                try:
                    client.llm.reset_cancel()
                except Exception:
                    pass

            # Add the clean, unpolluted user message to the database first
            if not request.regenerate:
                user_msg = discussion.add_message(
                    sender=discussion.lollmsClient.user_name if (discussion.lollmsClient and hasattr(discussion.lollmsClient, "user_name")) else "user",
                    sender_type="user",
                    content=user_message,
                    images=request.images if request.images else []
                )
            else:
                user_msg = discussion.get_message(discussion.active_branch_id)

            status = discussion.get_context_status()
            current_tokens = status.get("current_tokens", 0)
            max_tokens = status.get("max_tokens", 8192)

            deactivated_titles = []
            synthesis = ""

            try:
                # 95% Context Limit Guard
                if current_tokens >= (0.95 * max_tokens):
                    ASCIIColors.warning(
                        f"[Context Guard] Total context ({current_tokens:,} tokens) exceeds 95% of maximum limit ({max_tokens:,} tokens). "
                        "Executing sequential cognitive reading subroutine..."
                    )
                    # Trigger sequential reading and capture which artifacts were deactivated
                    deactivated_titles, synthesis = _execute_sequential_reading_subroutine(discussion, user_message, max_tokens, streaming_callback)
                else:
                    # Clear scratchpad if using normal mode
                    discussion.scratchpad = ""

                # Temporarily swap the database content of the user message with the synthesis-prepended prompt
                # to let the LLM see the sequential read results during export, without polluting the stored history.
                original_content = user_msg.content if user_msg else user_message
                if synthesis and user_msg:
                    user_msg.content = f"{synthesis}\n\n---\n\nUser query: {original_content}"

                active_tools = {}
                if discussion.lollmsClient and getattr(discussion.lollmsClient, "tools", None):
                    # Gather discovered local LCP tools (respecting user activation toggle)
                    for t in discussion.lollmsClient.tools.list_tools():
                        if not tool_states.get(t["name"], True):
                            continue
                        # Wrap callable to execute through the LCP binding
                        active_tools[t["name"]] = {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": [
                                {
                                    "name": p_name,
                                    "type": p_info.get("type", "any"),
                                    "description": p_info.get("description", ""),
                                    "optional": p_name not in t["input_schema"].get("required", [])
                                }
                                for p_name, p_info in t["input_schema"].get("properties", {}).items()
                            ],
                            "callable": lambda tname=t["name"], **kw: discussion.lollmsClient.tools.execute_tool(
                                tname, kw, discussion.lollmsClient, discussion=discussion
                            ).get("output", {})
                        }

                try:
                    discussion.chat(
                        user_message=user_msg.content if user_msg else user_message,
                        streaming_callback=streaming_callback,
                        tools=active_tools if active_tools else None,
                        add_user_message=False,  # Already added cleanly above
                        enable_memory=request.enable_memory if request.enable_memory is not None else True,
                        enable_artefacts=request.enable_artefacts if request.enable_artefacts is not None else True,
                        enable_in_message_status=request.enable_in_message_status if request.enable_in_message_status is not None else True,
                        enable_presentations=request.enable_presentations if request.enable_presentations is not None else True,
                        enable_books=request.enable_books if request.enable_books is not None else True,
                        enable_skills=request.enable_skills if request.enable_skills is not None else True,
                        enable_image_generation=request.enable_image_generation if request.enable_image_generation is not None else True,
                        enable_image_editing=request.enable_image_editing if request.enable_image_editing is not None else True,
                        enable_forms=request.enable_forms if request.enable_forms is not None else True,
                        enable_inline_widgets=request.enable_inline_widgets if request.enable_inline_widgets is not None else True,
                        debug=True
                    )
                finally:
                    # Guarantee clean unpolluted content is restored to the database
                    if user_msg:
                        user_msg.content = original_content
                        discussion.commit()
            except Exception as e:
                ASCIIColors.error(f"Chat execution failed: {e}")
                if client and client.debug:
                    trace_exception(e)
                q.put({"error": str(e)})
            finally:
                # We preserve deactivated_contents and sequential_summaries on the active discussion
                # instance so that context_status queries and subsequent conversational turns
                # continue using the optimized summaries instead of reloading the full-size files.
                q.put(None)

        thread = threading.Thread(target=run_chat_thread, daemon=True)
        thread.start()

        while True:
            while q.empty():
                await asyncio.sleep(0.02)
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(chat_event_generator(), media_type="text/event-stream")


@app.get("/api/context_status")
async def get_context_status_endpoint():
    """Retrieves the detailed token breakdown and context status."""
    try:
        return discussion.get_context_status()
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/{title}")
async def export_artifact_endpoint(title: str, format: str, version: Optional[int] = None):
    """Exports any text or data artifact in various formats."""
    try:
        content, mime_type = discussion.export_artifact(title, format, version)

        # Build attachment filename
        art = discussion.artefacts.get(title, version)
        ext_map = {
            "markdown": "md", "html": "html", "pdf": "pdf",
            "docx": "docx", "pptx": "pptx", "csv": "csv", "excel": "xlsx"
        }
        ext = ext_map.get(format.lower().strip(), "bin")
        filename = f"{title}_v{art.get('version', 1)}.{ext}"

        return Response(
            content=content,
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


# ── 💬 Message Actions & History Endpoints ──

class EditMessageRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/discussions/viewer_session/messages/{message_id}/edit")
async def edit_session_message_endpoint(message_id: str, payload: EditMessageRequest):
    """Edits the content of an individual message in place."""
    try:
        msg = discussion.get_message(message_id)
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found.")

        msg.content = payload.content.strip()
        if payload.metadata is not None:
            msg.metadata = {**(msg.metadata or {}), **payload.metadata}
        discussion.touch()
        discussion.commit()
        return {"success": True, "content": msg.content}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/discussions/viewer_session/messages/{message_id}")
async def delete_session_message_endpoint(message_id: str, prune: bool = False):
    """Deletes a message from the tree (supports recursive subtree pruning)."""
    try:
        if prune:
            count = discussion.prune_branch(message_id)
            if count > 0:
                discussion.commit()
                return {"success": True, "pruned_count": count}
        else:
            ok = discussion.remove_message(message_id)
            if ok:
                discussion.commit()
                return {"success": True}
        raise HTTPException(status_code=404, detail="Message not found.")
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discussions/viewer_session/messages/{message_id}/edit")
async def edit_session_message_endpoint(message_id: str, payload: EditMessageRequest):
    """Edits the content of an individual message in place."""
    try:
        msg = discussion.get_message(message_id)
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found.")

        msg.content = payload.content.strip()
        discussion.touch()
        discussion.commit()
        return {"success": True, "content": msg.content}
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discussions/viewer_session/messages/{message_id}/regenerate")
async def regenerate_session_message_endpoint(message_id: str):
    """Triggers branch regeneration from the specified message node onward."""
    # We will trigger the regeneration stream in a separate thread.
    # To facilitate this, the frontend simply calls this POST, and we perform 
    # the branch rewinding. Then the frontend initiates a standard chat turn 
    # using the updated branch state! This is incredibly clean and robust.
    try:
        discussion._rebuild_message_index()
        if message_id not in discussion._message_index:
            raise HTTPException(status_code=404, detail="Message not found.")

        target_msg = discussion._message_index[message_id]

        if target_msg.sender_type == 'assistant':
            user_parent_id = target_msg.parent_id
            if not user_parent_id:
                raise HTTPException(status_code=400, detail="Cannot regenerate root assistant message.")

            # Switch active branch to the user parent, removing the bad assistant reply
            discussion.active_branch_id = user_parent_id
            discussion.remove_message(message_id)
        else:
            # If the user clicked regenerate on their own message, we rewind the branch tip to this user message
            discussion.active_branch_id = message_id

        discussion.touch()
        discussion.commit()

        # Get the user prompt to send back to the frontend for automatic resending
        user_prompt = discussion.get_message(discussion.active_branch_id).content
        return {"success": True, "prompt": user_prompt}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discussions/viewer_session/messages/{message_id}/generate_image")
async def generate_image_for_message_endpoint(message_id: str, payload: Optional[GenerateImageForMessageRequest] = None):
    """Retries or manually generates an image from <generate_image>, <edit_image>, or missing <artefact_image> tags in a message."""
    try:
        msg = discussion.get_message(message_id)
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found.")

        # Parse the <generate_image>, <edit_image>, or <artefact_image> tag from the message content
        match_gen = re.search(r'<generate_image\s*([^>]*)>(.*?)</generate_image>', msg.content, re.DOTALL | re.IGNORECASE)
        match_edit = re.search(r'<edit_image\s*([^>]*)>(.*?)</edit_image>', msg.content, re.DOTALL | re.IGNORECASE)
        match_art = re.search(r'<artefact_image\s+id=["\']([^"\']+)["\']\s*(?:\/>|>)', msg.content, re.IGNORECASE)

        if not match_gen and not match_edit and not match_art:
            raise HTTPException(status_code=400, detail="No image generation, editing, or artifact image tag found in this message.")

        is_edit = False
        if match_edit:
            is_edit = True
            match = match_edit
            attrs = {m.group(1): m.group(2) for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', match.group(1))}
            prompt = match.group(2).strip()
            art_title = attrs.get('name', attrs.get('title', f"edited_image_{uuid.uuid4().hex[:6]}"))
        elif match_gen:
            match = match_gen
            attrs = {m.group(1): m.group(2) for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', match.group(1))}
            prompt = match.group(2).strip()
            art_title = attrs.get('name', attrs.get('title', f"generated_image_{uuid.uuid4().hex[:6]}"))
        else:
            # Re-generation of a deleted / missing artifact image
            match = match_art
            image_id = match.group(1)
            parts = image_id.split("::")
            art_title = "::".join(parts[:-1])
            is_edit = "edit" in art_title.lower()
            prompt = payload.prompt.strip() if (payload and payload.prompt) else f"Creative illustration for {art_title}"
            attrs = {"name": art_title}

        # ── Update prompt if edited by user ──
        if payload and payload.prompt:
            new_prompt = payload.prompt.strip()
            if new_prompt and new_prompt != prompt:
                if not match_art:
                    # Replace inside the tag in message content
                    tag_start = match.group(0).split(prompt)[0]
                    tag_end = match.group(0).split(prompt)[-1]
                    new_tag = f"{tag_start}{new_prompt}{tag_end}"
                    msg.content = msg.content.replace(match.group(0), new_tag)
                prompt = new_prompt
                discussion.touch()
                discussion.commit()

        tti = getattr(client, 'tti', None)
        if tti is None:
            raise HTTPException(status_code=400, detail="TTI (Text-to-Image) engine is not active or configured. Please set a TTI binding in settings first.")

        if is_edit:
            # For editing, find the source image
            source_b64: Optional[str] = None
            artefact_name = attrs.get('name', '')
            if artefact_name:
                a = discussion.artefacts.get(artefact_name)
                if a and a.get('images'):
                    source_b64 = a['images'][-1]
            if source_b64 is None:
                active_imgs = msg.get_active_images()
                if active_imgs:
                    source_b64 = active_imgs[-1]
            if source_b64 is None:
                raise HTTPException(status_code=400, detail="Source image not found for editing.")

            img_bytes = tti.edit_image(image=source_b64, prompt=prompt)
            art_prefix = "edited_image"
            group_type = "edited"
        else:
            width = int(attrs.get('width', 1024) if attrs.get('width', '').isdigit() else 1024)
            height = int(attrs.get('height', 1024) if attrs.get('height', '').isdigit() else 1024)
            img_bytes = tti.generate_image(prompt=prompt, width=width, height=height)
            art_prefix = "generated_image"
            group_type = "generated"

        if not img_bytes:
            raise HTTPException(status_code=500, detail="TTI engine returned empty image data.")

        import base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Add to image pack
        msg.add_image_pack(
            images=[img_b64],
            group_type=group_type,
            active_by_default=True,
            title=attrs.get('name', f'img_{uuid.uuid4().hex[:6]}'),
            prompt=prompt,
        )

        # Ingest as persistent workspace artifact, supporting multiple versions
        art_title = attrs.get('name', attrs.get('title', f"{art_prefix}_{uuid.uuid4().hex[:6]}"))
        existing_art = discussion.artefacts.get(art_title)

        if existing_art is None:
            # First creation
            art = discussion.artefacts.add(
                title=art_title,
                artefact_type=ArtefactType.IMAGE,
                content=f"### {group_type.capitalize()} Image: '{prompt}'\n\n<artefact_image id=\"{art_title}::0\" />",
                images=[img_b64],
                image_media_types=["image/png"],
                active=True
            )
        else:
            # Dynamic update to create a new version of the generated image
            art = discussion.artefacts.update(
                title=art_title,
                new_content=f"### {group_type.capitalize()} Image (Version {existing_art.get('version', 1) + 1}): '{prompt}'\n\n<artefact_image id=\"{art_title}::{existing_art.get('version', 1)}\" />",
                new_images=existing_art.get("images", []) + [img_b64],
                new_image_media_types=existing_art.get("image_media_types", []) + ["image/png"],
                bump_version=True,
                active=True
            )

        # Retrieve the updated/newest tag so it points to the correct version's image index
        new_tag = f'\n<artefact_image id="{art_title}::{art.get("version", 1) - 1}" />\n'

        # Replace the tag with the correct artefact_image anchor pointing to the newly generated version
        # Find the tag in current message content (since we might have updated the prompt earlier in this function)
        fresh_match_gen = re.search(r'<generate_image\s*([^>]*)>(.*?)</generate_image>', msg.content, re.DOTALL | re.IGNORECASE)
        fresh_match_edit = re.search(r'<edit_image\s*([^>]*)>(.*?)</edit_image>', msg.content, re.DOTALL | re.IGNORECASE)
        fresh_match_art = re.search(r'<artefact_image\s+id=["\']([^"\']+)["\']\s*(?:\/>|>)', msg.content, re.IGNORECASE)
        current_match = fresh_match_edit if is_edit else (fresh_match_gen or fresh_match_art)

        if current_match:
            msg.content = msg.content.replace(current_match.group(0), new_tag)

        discussion.touch()
        discussion.commit()

        return {"success": True, "title": art_title, "content": msg.content}
    except HTTPException:
        raise
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status_endpoint():
    return {
        "status": "running",
        "model_name": LLM_MODEL_NAME,
        "is_mock": not isinstance(client, LollmsClient),
        "needs_configuration": needs_configuration
    }


@app.get("/api/models")
async def list_current_models():
    """Retrieves available models from the currently active LLM binding."""
    try:
        if isinstance(client, DummyClient):
            return {"success": True, "models": []}

        models = []
        llm = getattr(client, "llm", None)
        if llm:
            if hasattr(llm, "list_models") and callable(getattr(llm, "list_models")):
                models = llm.list_models()
            elif hasattr(llm, "get_available_models") and callable(getattr(llm, "get_available_models")):
                models = llm.get_available_models()

        if not isinstance(models, list):
            models = []

        return {"success": True, "models": models}
    except Exception as e:
        trace_exception(e)
        return {"success": False, "error": str(e)}


# ── 🗃️ Workspaces CRUD & Selection Endpoints ──

@app.get("/api/workspaces")
async def list_workspaces_endpoint():
    """Lists all created project workspaces."""
    workspaces_dir = APP_DIR / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_app_config()
    active_ws = cfg.get("active_workspace", "default")

    # Always ensure 'default' workspace folder exists
    (workspaces_dir / "default").mkdir(parents=True, exist_ok=True)

    result = []
    for d in sorted(workspaces_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            result.append({
                "name": d.name,
                "active": d.name == active_ws,
                "created_at": time.ctime(d.stat().st_ctime)
            })
    return {"success": True, "workspaces": result}


@app.post("/api/workspaces/create")
async def create_workspace_endpoint(payload: WorkspaceRequest):
    """Creates a new isolated workspace folder."""
    name_clean = "".join(c for c in payload.name if c.isalnum() or c in ("-", "_")).strip().lower()
    if not name_clean:
        raise HTTPException(status_code=400, detail="Invalid workspace name.")

    workspaces_dir = APP_DIR / "workspaces"
    new_ws_dir = workspaces_dir / name_clean
    if new_ws_dir.exists():
        raise HTTPException(status_code=400, detail="A workspace with that name already exists.")

    new_ws_dir.mkdir(parents=True, exist_ok=True)
    (new_ws_dir / "data").mkdir(parents=True, exist_ok=True)
    (new_ws_dir / "data_workspace").mkdir(parents=True, exist_ok=True)

    return {"success": True, "name": name_clean}


@app.post("/api/workspaces/select")
async def select_workspace_endpoint(payload: WorkspaceRequest):
    """Hot-swaps the current active workspace with another isolated folder."""
    name_clean = "".join(c for c in payload.name if c.isalnum() or c in ("-", "_")).strip().lower()
    workspaces_dir = APP_DIR / "workspaces"
    target_dir = workspaces_dir / name_clean

    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace folder not found on disk.")

    try:
        initialize_workspace_state(name_clean)
        return {"success": True, "active_workspace": name_clean}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


class SubmitFormRequest(BaseModel):
    answers: Dict[str, Any]


@app.post("/api/discussions/{discussion_id}/forms/{form_id}/submit")
async def submit_form_endpoint(discussion_id: str, form_id: str, payload: SubmitFormRequest):
    """Injects the user's form answers directly into the active conversation context."""
    try:
        ok = discussion.submit_form_response(form_id, payload.answers)
        if not ok:
            raise HTTPException(status_code=404, detail="Form not found or already submitted.")
        discussion.commit()
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workspaces/{name}")
async def delete_workspace_endpoint(name: str):
    """Deletes an isolated workspace folder."""
    name_clean = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip().lower()
    if name_clean == "default":
        raise HTTPException(status_code=400, detail="The 'default' workspace cannot be deleted.")

    cfg = load_app_config()
    if cfg.get("active_workspace") == name_clean:
        raise HTTPException(status_code=400, detail="Cannot delete the currently active workspace. Switch workspaces first.")

    workspaces_dir = APP_DIR / "workspaces"
    target_dir = workspaces_dir / name_clean
    if target_dir.exists():
        import shutil
        shutil.rmtree(str(target_dir))
        return {"success": True}
    raise HTTPException(status_code=404, detail="Workspace not found.")


@app.post("/api/settings/open_folder")
async def open_settings_folder_endpoint():
    """Opens the persistent application config directory in the OS file explorer."""
    import subprocess
    import platform
    try:
        path_str = str(APP_DIR.resolve())
        system = platform.system()
        if system == "Windows":
            os.startfile(path_str)
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", path_str])
        else:  # Linux
            subprocess.Popen(["xdg-open", path_str])
        return {"success": True}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/clear")
async def clear_chat_endpoint():
    """Clears the conversational history in the active workspace."""
    try:
        discussion._rebuild_message_index()
        for msg_id in list(discussion._message_index.keys()):
            discussion.remove_message(msg_id)
        discussion.active_branch_id = None
        discussion.touch()
        discussion.commit()
        return {"success": True}
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/cancel")
async def cancel_chat_generation_endpoint():
    """Signals the active LLM binding to cancel current generation."""
    if client and getattr(client, "llm", None):
        try:
            client.llm.cancel()
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"success": False, "error": "No active LLM binding to cancel."}


@app.get("/api/discussions/viewer_session/messages")
async def get_session_messages_endpoint():
    """Retrieves the full chronological list of messages in the active branch with sibling details."""
    try:
        discussion._rebuild_message_index()
        branch = discussion.get_branch(discussion.active_branch_id)
        serialized = []
        for m in branch:
            siblings = discussion.get_siblings(m.id)
            sibling_ids = [sib.id for sib in siblings]
            active_index = sibling_ids.index(m.id) if m.id in sibling_ids else 0

            serialized.append({
                "id": m.id,
                "sender": m.sender,
                "sender_type": m.sender_type,
                "content": m.content,
                "model_name": m.model_name or "unknown",
                "tokens": m.tokens or 0,
                "generation_speed": m.generation_speed or 0,
                "metadata": m.metadata or {},
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "siblings": sibling_ids,
                "active_sibling_index": active_index
            })
        return serialized
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/discussions/viewer_session/branches")
async def get_session_branches_endpoint():
    """Lists all leaves / branches in the current discussion."""
    try:
        branches = discussion.list_branches()
        return [b.to_dict() for b in branches]
    except Exception as e:
        if client and client.debug:
            trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


class SwitchBranchRequest(BaseModel):
    leaf_id: str


@app.post("/api/discussions/viewer_session/branches/switch")
async def switch_session_branch_endpoint(payload: SwitchBranchRequest):
    """Switches the active branch pointer to the selected leaf_id."""
    try:
        ok = discussion.switch_branch(payload.leaf_id)
        if ok:
            discussion.commit()
            return {"success": True, "active_branch_id": discussion.active_branch_id}
        raise HTTPException(status_code=404, detail="Selected branch leaf not found.")
    except HTTPException:
        raise
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


class ForkMessageRequest(BaseModel):
    initial_content: str


@app.post("/api/discussions/viewer_session/messages/{message_id}/fork")
async def fork_session_message_endpoint(message_id: str, payload: ForkMessageRequest):
    """Forks the conversation tree starting a new branch from message_id (supports root-level None fork)."""
    try:
        p_id = None if (message_id == "null" or not message_id) else message_id
        if p_id is None:
            new_msg = discussion.add_message(
                sender=discussion.lollmsClient.user_name if (discussion.lollmsClient and hasattr(discussion.lollmsClient, "user_name")) else "user",
                sender_type="user",
                content=payload.initial_content.strip(),
                parent_id=None
            )
        else:
            new_msg = discussion.fork_from(
                message_id=p_id,
                initial_content=payload.initial_content.strip()
            )
        discussion.commit()
        return {
            "success": True,
            "message_id": new_msg.id,
            "active_branch_id": discussion.active_branch_id
        }
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discussions/viewer_session/messages/{message_id}/siblings/cycle")
async def cycle_message_sibling_endpoint(message_id: str, direction: int = 1):
    """Cycles the active branch to the next/prev sibling of the specifically clicked message_id."""
    try:
        discussion._rebuild_message_index()
        siblings = discussion.get_siblings(message_id)
        if not siblings:
            raise HTTPException(status_code=404, detail="No siblings found for this message.")

        sibling_ids = [sib.id for sib in siblings]
        if message_id not in sibling_ids:
            raise HTTPException(status_code=404, detail="Message not found among siblings.")

        current_idx = sibling_ids.index(message_id)
        new_idx = (current_idx + direction) % len(siblings)
        new_sibling = siblings[new_idx]

        # Find the deepest leaf under the selected sibling to switch the branch cleanly
        new_leaf = discussion._find_deepest_leaf(new_sibling.id) or new_sibling.id
        discussion.active_branch_id = new_leaf
        discussion.touch()
        discussion.commit()

        return {
            "success": True,
            "message_id": new_sibling.id,
            "active_branch_id": discussion.active_branch_id
        }
    except HTTPException:
        raise
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    """Entry point for the application."""
    ASCIIColors.cyan("==================================================================")
    ASCIIColors.green("🎬 Starting Lollms Multimodal Document Viewer on http://localhost:9680")
    ASCIIColors.cyan("==================================================================")
    uvicorn.run(app, host="localhost", port=9680)


if __name__ == "__main__":
    main()
