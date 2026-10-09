"""MiniMax-H3 shared TTV server. One process, many binding instances.

Backends:
  comfyui   - self-installs ComfyUI + GGUF node + weights on first run, then drives it (<=24G tiers)
  diffusers - ModularPipeline + group offloading (>=40G tier)
Single GPU job at a time. Sync endpoints so /status stays responsive during install/generation.
"""
import argparse, base64, gc, io, json, os, tempfile, threading, time, uuid, wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import bootstrap
from tiers import Tier, clamp_params, detect_vram_gb, select_tier


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class LipsyncRequest(BaseModel):
    prompt: str = ""
    negative_prompt: Optional[str] = ""
    image_b64: str
    audio_b64: str
    image_ext: str = ".png"
    audio_ext: str = ".wav"
    params: Dict[str, Any] = Field(default_factory=dict)


def _wav_seconds(b: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(b)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def _wav_truncate(b: bytes, seconds: float) -> bytes:
    with wave.open(io.BytesIO(b)) as r:
        n = min(r.getnframes(), int(seconds * r.getframerate()))
        params, data = r.getparams(), r.readframes(n)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params); w.writeframes(data)
    return out.getvalue()


S: Dict[str, Any] = {"backend": None, "tier": None, "vram": 0.0, "busy": False, "progress": 0.0,
                     "msg": "idle", "phase": "starting", "error": None, "lock": threading.Lock()}


def _progress(frac: float, msg: str):
    S["progress"], S["msg"] = frac, msg
    print(f"[minimax_h3] {frac:.0%} {msg}", flush=True)


class Backend(ABC):
    def __init__(self, tier: Tier, cfg: Dict[str, Any]):
        self.tier, self.cfg = tier, cfg

    def prepare(self, progress): pass
    @abstractmethod
    def generate(self, req: GenerateRequest, p: Dict[str, Any], progress) -> bytes: ...
    def generate_lipsync(self, req, image: bytes, audio: bytes, p: Dict[str, Any], progress) -> bytes:
        raise NotImplementedError("lip-sync is not implemented for this backend")
    def unload(self): pass
    def loaded(self) -> bool: return False


class ComfyBackend(Backend):
    """Needs workflows/<workflow>.json = ComfyUI 'Export (API)' of a graph that works.
    String values equal to '{{key}}' are replaced: prompt, negative_prompt, width, height, num_frames,
    num_inference_steps, guidance_scale, seed, fps, unet_name, clip_name, vae_video_name, vae_audio_name."""

    def __init__(self, tier, cfg):
        super().__init__(tier, cfg)
        self.port = int(cfg.get("comfy_port", 8188))
        self.url = cfg.get("comfy_url", f"http://127.0.0.1:{self.port}").rstrip("/")
        self.root = Path(cfg["comfy_root"])
        self.bcfg = {**bootstrap.DEFAULTS, **{k: v for k, v in cfg.items() if k in bootstrap.DEFAULTS and v}}
        self.names: Dict[str, str] = {}
        self.proc = None

    def prepare(self, progress):
        comfy = bootstrap.ensure_comfy(self.root, self.bcfg, progress)
        self.names = bootstrap.ensure_models(comfy, self.tier, self.bcfg, cfg_token(self.cfg), progress)
        args = self.cfg.get("comfy_args") or self.bcfg["comfy_args"].get(self.tier.name, [])
        self.proc = bootstrap.start_comfy(self.root, comfy, self.url, self.port, args, progress)
        wf = self._wf_path("fl2va")
        if not wf.exists() or "_README" in wf.read_text(encoding="utf-8"):
            t = bootstrap.find_official_template(self.root)
            progress(1.0, f"READY but workflow missing: open ComfyUI at {self.url}, load the 'MiniMax H3 T2V' template"
                          f"{f' ({t})' if t else ''}, Export (API), save as {wf}")

    def generate_lipsync(self, req, image, audio, p, progress) -> bytes:
        """workflows/lipsync.json = API export of the official Ref2VA 'Talking Digital Human' graph.
        Extra placeholders: {{image_name}}, {{audio_name}}, {{duration}}; {{unet_name}} is the Ref2VA GGUF."""
        path = self._wf_path("lipsync")
        if not path.exists() or "_README" in path.read_text(encoding="utf-8"):
            raise FileNotFoundError(f"API-format lip-sync workflow not provided: {path}")
        comfy = self.root / "ComfyUI"
        progress(0.02, "ensuring Ref2VA weights")
        self.names["ref_unet_name"] = bootstrap.ensure_ref_unet(comfy, self.tier, self.bcfg, cfg_token(self.cfg), progress)
        inp = comfy / "input"; inp.mkdir(exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        img_name, aud_name = f"h3_lip_{tag}{req.image_ext}", f"h3_lip_{tag}{req.audio_ext}"
        (inp / img_name).write_bytes(image); (inp / aud_name).write_bytes(audio)
        try:
            values = {**p, **self.names, "unet_name": self.names["ref_unet_name"], "prompt": req.prompt,
                      "negative_prompt": req.negative_prompt or "", "image_name": img_name, "audio_name": aud_name}
            wf = self._fill(json.loads(path.read_text(encoding="utf-8")), values)
            return self._execute(wf, progress)
        finally:
            for n in (img_name, aud_name):
                (inp / n).unlink(missing_ok=True)

    def loaded(self): return bootstrap.comfy_alive(self.url)

    def _wf_path(self, workflow: str) -> Path:
        return Path(self.cfg.get("workflow_path") or Path(__file__).parent.parent / "workflows" / f"{workflow}.json")

    @staticmethod
    def _fill(node, values):
        if isinstance(node, dict):  return {k: ComfyBackend._fill(v, values) for k, v in node.items()}
        if isinstance(node, list):  return [ComfyBackend._fill(v, values) for v in node]
        if isinstance(node, str) and node.startswith("{{") and node.endswith("}}"):
            return values.get(node[2:-2], node)
        return node

    def _validate(self, wf: dict):
        info = requests.get(f"{self.url}/object_info", timeout=30).json()
        missing = sorted({n["class_type"] for n in wf.values() if isinstance(n, dict) and n.get("class_type") not in info})
        if missing:
            raise RuntimeError(f"ComfyUI lacks node types used by the workflow: {missing} (outdated ComfyUI or missing custom node)")

    def generate(self, req, p, progress) -> bytes:
        path = self._wf_path(p["workflow"])
        if not path.exists() or "_README" in path.read_text(encoding="utf-8"):
            raise FileNotFoundError(f"API-format workflow not provided: {path}")
        values = {**p, **self.names, "prompt": req.prompt, "negative_prompt": req.negative_prompt or ""}
        wf = self._fill(json.loads(path.read_text(encoding="utf-8")), values)
        return self._execute(wf, progress)

    def _execute(self, wf: dict, progress) -> bytes:
        self._validate(wf)
        progress(0.02, "queued in ComfyUI")
        r = requests.post(f"{self.url}/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex}, timeout=30)
        if not r.ok:
            raise RuntimeError(f"ComfyUI rejected workflow: {r.text[:800]}")
        pid = r.json()["prompt_id"]
        deadline = time.time() + float(self.cfg.get("timeout_s", 3600))
        while time.time() < deadline:
            h = requests.get(f"{self.url}/history/{pid}", timeout=10).json()
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution error: {json.dumps(st)[:800]}")
                for out in h[pid].get("outputs", {}).values():
                    for items in out.values():
                        for it in items if isinstance(items, list) else []:
                            if isinstance(it, dict) and str(it.get("filename", "")).lower().endswith((".mp4", ".webm", ".mkv")):
                                v = requests.get(f"{self.url}/view", params={"filename": it["filename"],
                                                 "subfolder": it.get("subfolder", ""), "type": it.get("type", "output")}, timeout=120)
                                v.raise_for_status()
                                return v.content
                raise RuntimeError("ComfyUI finished but no video file in outputs (check the workflow's save node)")
            progress(0.5, "ComfyUI running")
            time.sleep(1.5)
        raise TimeoutError("ComfyUI generation timed out")


def cfg_token(cfg): return cfg.get("hf_token") or os.environ.get("HF_TOKEN")


class DiffusersBackend(Backend):
    def __init__(self, tier, cfg):
        super().__init__(tier, cfg); self.pipe, self.workflow = None, None

    def loaded(self): return self.pipe is not None

    def generate_lipsync(self, req, image, audio, p, progress):
        raise NotImplementedError("lip-sync on the diffusers tier: ref2va pipeline argument names are unverified. "
                                  "Load ModularPipeline(workflow='ref2va') and read pipe.doc to wire image/audio inputs.")

    def unload(self):
        self.pipe, self.workflow = None, None
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

    def _load(self, workflow: str):
        if self.pipe is not None and self.workflow == workflow: return
        self.unload()
        import torch
        from diffusers import ModularPipeline
        pipe = ModularPipeline.from_pretrained(self.tier.artifact, workflow=workflow)
        pipe.load_components(torch_dtype=torch.bfloat16)   # per diffusers modular docs; verify on first run
        if self.tier.offload == "group":
            from diffusers.hooks import apply_group_offloading
            for name in ("transformer", "text_encoder"):
                comp = getattr(pipe, name, None)
                if comp is not None:
                    apply_group_offloading(comp, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                           offload_type="block_level", num_blocks_per_group=2)
        self.pipe, self.workflow = pipe, workflow

    def generate(self, req, p, progress) -> bytes:
        import torch
        from diffusers.utils import export_to_video, load_image
        self._load(p["workflow"])
        kw = dict(prompt=req.prompt, width=p["width"], height=p["height"], num_frames=p["num_frames"],
                  num_inference_steps=p["num_inference_steps"], guidance_scale=p["guidance_scale"])
        if p.get("seed", -1) != -1: kw["generator"] = torch.Generator("cpu").manual_seed(int(p["seed"]))
        if p.get("image"):      kw["image"] = load_image(p["image"])
        if p.get("last_image"): kw["last_image"] = load_image(p["last_image"])
        progress(0.1, "denoising")
        out = self.pipe(**kw)
        frames = out.frames[0]
        # TODO(first run): H3 also produces audio. Log dir(out) once and mux it with ffmpeg here.
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "out.mp4")
            export_to_video(frames, f, fps=int(p.get("fps", 24)))
            return Path(f).read_bytes()


app = FastAPI(title="MiniMax-H3 TTV Server")


@app.get("/status")
def status():
    t: Tier = S["tier"]
    return {"status": "running", "phase": S["phase"], "error": S["error"], "busy": S["busy"], "progress": S["progress"],
            "message": S["msg"], "vram_gb": round(S["vram"], 1), "tier": t.name, "backend": t.backend,
            "artifact": t.artifact, "max_frames": t.max_frames, "max_side": t.max_side,
            "model_loaded": S["phase"] == "ready" and S["backend"].loaded()}


@app.post("/unload")
def unload():
    S["backend"].unload(); return {"status": True}


@app.post("/generate_video")
def generate_video(req: GenerateRequest):
    if S["phase"] != "ready":
        raise HTTPException(status_code=503, detail=f"server not ready: {S['phase']} ({S['msg']}) {S['error'] or ''}")
    tier: Tier = S["tier"]
    p = {"workflow": "fl2va", "width": 768, "height": 512, "num_frames": 25, "num_inference_steps": 30,
         "guidance_scale": 5.0, "seed": -1, "fps": 24, **req.params}
    p["width"], p["height"], p["num_frames"], notes = clamp_params(tier, int(p["width"]), int(p["height"]), int(p["num_frames"]))
    if not S["lock"].acquire(blocking=False):
        raise HTTPException(status_code=409, detail="GPU busy with another generation")
    S["busy"] = True
    try:
        data = S["backend"].generate(req, p, _progress)
        return Response(content=data, media_type="video/mp4", headers={"X-Tier": tier.name, "X-Clamped": "; ".join(notes)})
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        S["busy"] = False; _progress(0.0, "idle"); S["lock"].release()


@app.post("/lipsync")
def lipsync(req: LipsyncRequest):
    if S["phase"] != "ready":
        raise HTTPException(status_code=503, detail=f"server not ready: {S['phase']} ({S['msg']}) {S['error'] or ''}")
    tier: Tier = S["tier"]
    if tier.backend != "comfyui":
        raise HTTPException(status_code=501, detail=f"lip-sync not implemented on tier {tier.name} ({tier.backend})")
    image, audio = base64.b64decode(req.image_b64), base64.b64decode(req.audio_b64)
    p = {"width": 768, "height": 512, "num_inference_steps": 30, "guidance_scale": 5.0, "seed": -1, "fps": 24, **req.params}
    fps, notes = int(p["fps"]), []
    secs = _wav_seconds(audio) if req.audio_ext.lower() == ".wav" else None
    nf = int(p.get("num_frames") or (int(secs * fps) + 1 if secs else 25))
    if not p.get("num_frames") and not secs:
        notes.append("audio length unknown (non-wav): pass num_frames; defaulted to 25")
    p["width"], p["height"], nf2, cl = clamp_params(tier, int(p["width"]), int(p["height"]), nf)
    notes += cl
    if nf2 < nf:
        keep = (nf2 - 1) / fps
        if secs:
            audio = _wav_truncate(audio, keep); notes.append(f"audio truncated to {keep:.1f}s to match video")
        else:
            notes.append("audio is longer than the video (tier frame cap)")
    p["num_frames"], p["duration"] = nf2, round((nf2 - 1) / fps, 2)
    if not S["lock"].acquire(blocking=False):
        raise HTTPException(status_code=409, detail="GPU busy with another generation")
    S["busy"] = True
    try:
        data = S["backend"].generate_lipsync(req, image, audio, p, _progress)
        return Response(content=data, media_type="video/mp4", headers={"X-Tier": tier.name, "X-Clamped": "; ".join(notes)})
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        S["busy"] = False; _progress(0.0, "idle"); S["lock"].release()


def _prepare_bg():
    try:
        S["phase"] = "installing"
        S["backend"].prepare(_progress)
        S["phase"], S["error"] = "ready", None
    except Exception as e:
        import traceback; traceback.print_exc()
        S["phase"], S["error"] = "failed", f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=9635)
    ap.add_argument("--models-path", default="./models"); ap.add_argument("--hf-token", default=None)
    ap.add_argument("--comfy-root", required=True)
    ap.add_argument("--config", default="{}")
    a = ap.parse_args()
    if a.hf_token: os.environ["HF_TOKEN"] = a.hf_token
    os.environ.setdefault("HF_HOME", str(Path(a.models_path) / "hf"))
    cfg = {**json.loads(a.config), "comfy_root": a.comfy_root, "hf_token": a.hf_token}
    S["vram"] = detect_vram_gb()
    S["tier"] = select_tier(S["vram"], cfg.get("tier"), max_frames=cfg.get("max_frames"), max_side=cfg.get("max_side"))
    S["backend"] = (ComfyBackend if S["tier"].backend == "comfyui" else DiffusersBackend)(S["tier"], cfg)
    print(f"[minimax_h3] VRAM {S['vram']:.1f} GiB -> {S['tier']}", flush=True)
    threading.Thread(target=_prepare_bg, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port)
