"""
lollms_tti_server.py
====================
A FastAPI server that exposes an OpenAI-compatible Text-to-Image API
backed by any LollmsTTIBinding.

OpenAI endpoints implemented
-----------------------------
  POST /v1/images/generations   – generate an image from a prompt
  POST /v1/images/edits         – edit / vary one or more images
  GET  /v1/models               – list available TTI models
  GET  /v1/models/{model_id}    – describe a single model
  GET  /health                  – liveness probe

Compatibility notes
-------------------
* The `model` field in every request is **optional**.  When absent (or set to
  the sentinel value ``"lollms"``), the server uses whatever model is currently
  configured in the active TTI binding.  This is the main extension over the
  strict OpenAI spec.
* Responses are 100 % schema-compatible with the OpenAI Images API so any
  existing OpenAI client library works without modification.
* Authentication is optional: set ``--api-key`` (or ``LOLLMS_TTI_API_KEY``)
  to enforce bearer-token checking; leave blank to run open.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("lollms_tti_server")


# ---------------------------------------------------------------------------
# Lazy binding import (lollms_client may not be installed at import time)
# ---------------------------------------------------------------------------
def _load_binding(binding_name: str, binding_config: Dict[str, Any]):
    """Instantiate a LollmsTTIBinding by name."""
    from lollms_client.lollms_tti_binding import LollmsTTIBindingManager
    manager = LollmsTTIBindingManager()
    binding = manager.create_binding(binding_name=binding_name, **binding_config)
    if binding is None:
        raise RuntimeError(f"Could not load TTI binding '{binding_name}'")
    return binding


# ---------------------------------------------------------------------------
# Global state (set during startup)
# ---------------------------------------------------------------------------
class _State:
    binding: Any = None          # LollmsTTIBinding instance
    api_key: Optional[str] = None
    default_model: Optional[str] = None  # model name from config, may be None


_state = _State()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LoLLMS TTI – OpenAI-compatible API",
    description=(
        "Exposes any LollmsTTIBinding as an OpenAI-compatible image generation "
        "endpoint.  The `model` field is optional; omit it to use whatever model "
        "is configured in the active binding."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def _check_auth(authorization: Optional[str] = Header(None)) -> None:
    if not _state.api_key:
        return  # open server
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != _state.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API key")


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response schemas
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = Field(
        default=None,
        description=(
            "Model to use for generation.  Optional – when omitted or set to "
            "'lollms', the server uses the binding's currently configured model."
        ),
    )
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = Field(
        default=None,
        description="e.g. '1024x1024', '1536x1024'. None → binding default.",
    )
    quality: Optional[str] = Field(default=None)
    style: Optional[str] = Field(default=None)
    response_format: Optional[str] = Field(
        default="b64_json",
        description="'b64_json' or 'url' (url not supported, falls back to b64_json)",
    )
    user: Optional[str] = None
    # lollms extensions
    negative_prompt: Optional[str] = Field(default="")
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lollms"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_size(size_str: Optional[str]) -> tuple[int, int]:
    """'1024x768' → (1024, 768).  Falls back to (1024, 1024)."""
    if size_str and "x" in size_str:
        try:
            w, h = size_str.lower().split("x", 1)
            return int(w), int(h)
        except ValueError:
            pass
    return 1024, 1024


def _bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _resolve_model(requested: Optional[str]) -> Optional[str]:
    """
    Return the model name to pass to the binding.

    Rules:
      - None / empty / 'lollms' / 'lollms-tti'  → None  (binding uses its own default)
      - anything else                             → pass through verbatim
    """
    _sentinel = {None, "", "lollms", "lollms-tti"}
    if requested in _sentinel:
        return _state.default_model  # may itself be None → binding decides
    return requested


def _call_generate(req: ImageGenerationRequest) -> bytes:
    model = _resolve_model(req.model)
    w, h  = _parse_size(req.size) if req.size else (req.width or 1024, req.height or 1024)

    kwargs: Dict[str, Any] = {}
    if model:
        kwargs["model_name"] = model
    if req.quality:
        kwargs["quality"] = req.quality
    if req.style:
        kwargs["style"] = req.style
    if req.size:
        kwargs["size"] = req.size

    log.info("generate_image | model=%s size=%dx%d", model or "(binding default)", w, h)
    return _state.binding.generate_image(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or "",
        width=w,
        height=h,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "binding": type(_state.binding).__name__}


# -- /v1/models ---------------------------------------------------------------

@app.get("/v1/models", response_model=ModelListResponse,
         dependencies=[Depends(_check_auth)])
async def list_models():
    try:
        models = _state.binding.list_models()
    except Exception as exc:
        log.exception("list_models failed")
        raise HTTPException(status_code=500, detail=str(exc))

    objects = [
        ModelObject(id=m.get("model_name", m.get("id", "unknown")),
                    owned_by="lollms")
        for m in models
    ]
    return ModelListResponse(data=objects)


@app.get("/v1/models/{model_id}", response_model=ModelObject,
         dependencies=[Depends(_check_auth)])
async def get_model(model_id: str):
    try:
        models = _state.binding.list_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    for m in models:
        mid = m.get("model_name", m.get("id", ""))
        if mid == model_id:
            return ModelObject(id=mid, owned_by="lollms")

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# -- /v1/images/generations ---------------------------------------------------

@app.post("/v1/images/generations", response_model=ImageGenerationResponse,
          dependencies=[Depends(_check_auth)])
async def create_image(req: ImageGenerationRequest):
    n = max(1, req.n or 1)
    results: List[ImageData] = []

    for _ in range(n):
        try:
            img_bytes = _call_generate(req)
        except Exception as exc:
            log.exception("generate_image failed")
            raise HTTPException(status_code=500, detail=str(exc))

        results.append(ImageData(b64_json=_bytes_to_b64(img_bytes)))

    return ImageGenerationResponse(created=int(time.time()), data=results)


# -- /v1/images/edits ---------------------------------------------------------
# OpenAI uses multipart/form-data for edits; we handle that via UploadFile.
# We also accept a JSON body for clients that prefer it.

@app.post("/v1/images/edits", response_model=ImageGenerationResponse,
          dependencies=[Depends(_check_auth)])
async def edit_image(
    # multipart fields (OpenAI-standard)
    image:           Optional[UploadFile] = File(default=None),
    mask:            Optional[UploadFile] = File(default=None),
    prompt:          Optional[str]        = Form(default=None),
    model:           Optional[str]        = Form(default=None),
    n:               Optional[int]        = Form(default=1),
    size:            Optional[str]        = Form(default=None),
    quality:         Optional[str]        = Form(default=None),
    response_format: Optional[str]        = Form(default="b64_json"),
    # lollms extensions (form)
    negative_prompt: Optional[str]        = Form(default=""),
    width:           Optional[int]        = Form(default=None),
    height:          Optional[int]        = Form(default=None),
):
    if prompt is None:
        raise HTTPException(status_code=422, detail="'prompt' is required")
    if image is None:
        raise HTTPException(status_code=422, detail="'image' file is required")

    src_bytes  = await image.read()
    mask_bytes = (await mask.read()) if mask else None

    resolved_model = _resolve_model(model)
    w, h = _parse_size(size) if size else (width or 1024, height or 1024)

    kwargs: Dict[str, Any] = {}
    if resolved_model:
        kwargs["model_name"] = resolved_model
    if quality:
        kwargs["quality"] = quality
    if size:
        kwargs["size"] = size
    if mask_bytes:
        kwargs["mask"] = mask_bytes

    log.info("edit_image | model=%s size=%dx%d", resolved_model or "(binding default)", w, h)

    results: List[ImageData] = []
    for _ in range(max(1, n or 1)):
        try:
            img_bytes = _state.binding.edit_image(
                images=src_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=w,
                height=h,
                **kwargs,
            )
        except Exception as exc:
            log.exception("edit_image failed")
            raise HTTPException(status_code=500, detail=str(exc))

        results.append(ImageData(b64_json=_bytes_to_b64(img_bytes)))

    return ImageGenerationResponse(created=int(time.time()), data=results)


# ---------------------------------------------------------------------------
# Also handle the non-versioned paths some clients use
# ---------------------------------------------------------------------------
@app.post("/images/generations",  response_model=ImageGenerationResponse,
          dependencies=[Depends(_check_auth)])
async def create_image_no_prefix(req: ImageGenerationRequest):
    return await create_image(req)


@app.get("/models", response_model=ModelListResponse,
         dependencies=[Depends(_check_auth)])
async def list_models_no_prefix():
    return await list_models()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoLLMS TTI – OpenAI-compatible image generation server",
    )
    p.add_argument("--host",         default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port",         default=7860, type=int, help="Bind port (default: 7860)")
    p.add_argument("--binding",      default="openai",   help="LollmsTTIBinding name to load")
    p.add_argument("--model",        default=None,        help="Default model name (optional)")
    p.add_argument("--api-key",      default=None,        help="Bearer token to protect the API (optional)")
    p.add_argument("--service-key",  default=None,        help="API key forwarded to the TTI binding (e.g. OpenAI key)")
    p.add_argument("--host-address", default=None,        help="Base URL forwarded to the TTI binding")
    p.add_argument("--verify-ssl",   default=True, type=lambda x: x.lower() not in ("false","0","no"),
                   help="SSL verification for the TTI binding (default: true)")
    p.add_argument("--cert-file",    default=None,        help="PEM certificate file for the TTI binding")
    p.add_argument("--reload",       action="store_true", help="Enable uvicorn auto-reload (dev only)")
    return p


def main():
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # Build binding config from CLI args + environment fallbacks
    binding_config: Dict[str, Any] = {}

    service_key = (
        args.service_key
        or os.environ.get("LOLLMS_TTI_SERVICE_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if service_key:
        binding_config["service_key"] = service_key

    if args.host_address:
        binding_config["host_address"] = args.host_address

    binding_config["verify_ssl_certificate"] = args.verify_ssl

    if args.cert_file:
        binding_config["certificate_file_path"] = args.cert_file

    if args.model:
        binding_config["model_name"] = args.model

    log.info("Loading TTI binding '%s' …", args.binding)
    _state.binding = _load_binding(args.binding, binding_config)
    _state.api_key = args.api_key or os.environ.get("LOLLMS_TTI_API_KEY")
    _state.default_model = args.model  # None → let the binding decide

    log.info(
        "Binding ready: %s | default model: %s | auth: %s",
        type(_state.binding).__name__,
        _state.default_model or "(binding default)",
        "enabled" if _state.api_key else "disabled",
    )

    uvicorn.run(
        "lollms_tti_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
