"""sd_cpp: TTV binding on top of stable-diffusion.cpp (sd-cli). MiniMax-H3 is the built-in preset;
any other sd.cpp video model can be downloaded from Hugging Face and registered as a profile.
No server, no ComfyUI: one sd-cli process per generation (frees VRAM between calls), GPU access
serialized across processes by a file lock."""
import math, shlex, subprocess, tempfile, threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:   # self-sufficiency: make sure the light Python deps exist
    import pipmaster as pm
    pm.ensure_packages(["requests", "filelock", "huggingface_hub", "imageio-ffmpeg"])
except Exception:
    pass

from ascii_colors import ASCIIColors
from filelock import FileLock
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, MediaInput

from . import hub, media, presets, runtime

BindingName = "SdCppTTVBinding"
VIDEO_EXT = (".mp4", ".avi", ".webm", ".mkv", ".mov")
H3_LADDER = [("Q2_K", "6.26 GiB"), ("Q3_K", "8.16 GiB"), ("Q4_K", "10.64 GiB"),
             ("Q5_0", "12.97 GiB"), ("Q6_K", "15.45 GiB"), ("Q8_0", "19.97 GiB")]
# UNVERIFIED: how Ref2VA wants an audio reference cited in the prompt. Override via lipsync_prompt_template.
LIPSYNC_TEMPLATE = ("Use the person from <Picture 1> as the main character, keeping their face and identity unchanged. "
                    "The character speaks the provided audio reference, with natural lip movements precisely "
                    "synchronized to the speech. {prompt}")


class SdCppTTVBinding(LollmsTTVBinding):
    def __init__(self, **kwargs):
        kw = dict(kwargs); debug = kw.pop("debug", False); kw.pop("binding_name", None)
        super().__init__(binding_name="sd_cpp", debug=debug, **kw)
        c = self.config
        sysroot = self.get_system_dir()
        self.root = self.resolve_system_path(c["root_path"]) if c.get("root_path") else sysroot / "sd_cpp"
        self.models_path = self.resolve_system_path(c["models_path"]) if c.get("models_path") else sysroot / "data" / "ttv_models" / "sd_cpp"
        self.root.mkdir(parents=True, exist_ok=True); self.models_path.mkdir(parents=True, exist_ok=True)
        self.hf_token = c.get("hf_token") or None
        self.auto_install = bool(c.get("auto_install", True))
        self.auto_download = bool(c.get("auto_download", True))
        self.timeout_s = float(c.get("timeout_s", 3600))
        self.h3_repo = c.get("h3_repo") or presets.GGUF_REPO
        self.profiles = hub.Profiles(self.models_path)
        self._cli_path: Optional[Path] = None
        self._gpu_lock = FileLock(str(self.root / "gpu.lock"), timeout=float(c.get("lock_timeout_s", 7200)))
        vram = presets.detect_vram_gb()
        self.tier = presets.select_tier(vram, c.get("tier"), int(c.get("max_frames") or 0), int(c.get("max_side") or 0))
        if vram == 0:
            self._log(f"no NVIDIA GPU detected, assuming tier {self.tier.name} (set 'tier' to override)")
        elif vram < presets.TIERS[-1].min_vram_gb:
            ASCIIColors.warning(f"[sd_cpp] {vram:.1f} GiB is below the smallest documented tier; trying {self.tier.name} anyway")

    def _log(self, msg: str): ASCIIColors.info(f"[sd_cpp] {msg}")

    # ── resource contract (nothing stays resident: one process per generation) ──
    def is_local(self) -> bool: return True
    def is_model_loaded(self, model_name=None) -> bool: return False
    def get_loaded_models(self) -> List[str]: return []
    def unload_model(self, model_name=None) -> bool: return True

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """Pre-flight: installs the runtime and downloads the weights now instead of at first generation."""
        try:
            self._cli()
            if (model_name or self.config.get("model_name") or "minimax_h3").lower() in presets.H3_NAMES:
                self._ensure_h3("fl2va")
            return True
        except Exception as e:
            self._last_error = str(e); return False

    def ps(self) -> List[Dict[str, Any]]:
        return [{"model_name": "minimax_h3", "is_loaded": False, "device": f"on-demand ({self.tier.name})", "vram_size": 0,
                 "gpu_usage_percent": 0.0, "cpu_usage_percent": 0.0, "ref_count": 0, "status": "idle"}]

    # ── runtime / files ──────────────────────────────────────────────────
    def _cli(self) -> Path:
        if self._cli_path is None:
            self._cli_path = runtime.ensure_sd_cli(self.root, self.config.get("sd_cli_path"), self.auto_install,
                                                   self.config.get("accelerator", "auto"), self._log)
        return self._cli_path

    def _ensure(self, repo: str, sub: str, local_pick: Callable, remote_pick: Callable) -> Path:
        d = self.models_path / sub
        local = local_pick([p.name for p in d.iterdir() if p.is_file()]) if d.exists() else None
        if local:
            return d / local
        if not self.auto_download:
            raise FileNotFoundError(f"missing file for {repo} in {d}; auto_download is off (use download_model)")
        files = hub.list_repo_files(repo, self.hf_token)
        hit = remote_pick(files)
        if not hit:
            raise RuntimeError(f"nothing suitable in {repo}. Candidates: {[f for f in files if f.endswith(('.gguf', '.safetensors'))][:40]}")
        return hub.download_file(repo, hit, d, self.hf_token, progress=self._log)

    def _ensure_h3(self, workflow: str, quant: Optional[str] = None) -> Dict[str, Path]:
        t = self.tier
        uq = (quant,) if quant else t.unet_quants
        return {
            "diffusion_model": self._ensure(self.h3_repo, "diffusion_models",
                                            lambda fs: presets.pick_diffusion(fs, workflow, uq[:1]),
                                            lambda fs: presets.pick_diffusion(fs, workflow, uq)),
            "llm": self._ensure(self.h3_repo, "text_encoders",
                                lambda fs: presets.pick_text_encoder(fs, t.te_quants[:1]),
                                lambda fs: presets.pick_text_encoder(fs, t.te_quants)),
            "vae": self._ensure(presets.VAE_REPO, "vae", lambda fs: hub.find_exact(fs, presets.VAE_VIDEO),
                                lambda fs: hub.find_exact(fs, presets.VAE_VIDEO)),
            "audio_vae": self._ensure(presets.VAE_REPO, "vae", lambda fs: hub.find_exact(fs, presets.VAE_AUDIO),
                                      lambda fs: hub.find_exact(fs, presets.VAE_AUDIO)),
        }

    # ── process execution ────────────────────────────────────────────────
    def _run(self, args: List[str], on_line: Optional[Callable[[str], None]] = None):
        cmd = [str(self._cli()), *[str(a) for a in args]]
        if self.debug: self._log("exec: " + " ".join(shlex.quote(a) for a in cmd))
        tail = deque(maxlen=40)
        with self._gpu_lock:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env=runtime.run_env(self._cli()), cwd=str(self._cli().parent))
            timer = threading.Timer(self.timeout_s, proc.kill); timer.start()
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    tail.append(line)
                    if on_line: on_line(line)
                    elif self.debug: print(line)
                rc = proc.wait()
            finally:
                timer.cancel()
        if rc != 0:
            raise RuntimeError(f"sd-cli failed (exit {rc}):\n" + "\n".join(tail))

    @staticmethod
    def _bytes_of(x: MediaInput, default_ext: str):
        if isinstance(x, (bytes, bytearray)):
            return bytes(x), default_ext
        p = Path(x).expanduser()
        return p.read_bytes(), (p.suffix.lower() or default_ext)

    def _stage(self, x: MediaInput, d: Path, stem: str, default_ext: str, wav: bool = False) -> str:
        """Returns a filesystem path sd-cli can read. Paths are used in place unless conversion is needed."""
        if not isinstance(x, (bytes, bytearray)) and not wav:
            p = Path(x).expanduser()
            if not p.exists():
                raise FileNotFoundError(p)
            return str(p)
        data, ext = self._bytes_of(x, default_ext)
        if wav:
            data, ext = media.to_wav_bytes(data, ext), ".wav"
        dst = d / f"{stem}{ext}"
        dst.write_bytes(data)
        return str(dst)

    # ── generation ───────────────────────────────────────────────────────
    def generate_video(self, prompt: str, **kwargs) -> bytes:
        model = kwargs.pop("model", None) or self.config.get("model_name") or "minimax_h3"
        return self._generate(prompt, model, **kwargs)

    def _generate(self, prompt: str, model: str, **kw) -> bytes:
        on_line = kw.pop("progress_callback", None)
        is_h3 = model.lower() in presets.H3_NAMES
        w, h = int(kw.pop("width", 864)), int(kw.pop("height", 480))
        nf = int(kw.pop("num_frames", 56))
        steps, cfg, seed = kw.pop("steps", None), float(kw.pop("cfg_scale", 1.0)), int(kw.pop("seed", -1))
        neg = kw.pop("negative_prompt", "")
        image, end_image = kw.pop("image", None), kw.pop("end_image", None)
        ref_images, ref_audios = kw.pop("ref_images", None) or [], kw.pop("ref_audios", None) or []
        ref_videos, ref_video_audios = kw.pop("ref_videos", None) or [], kw.pop("ref_video_audios", None) or []
        no_audio, quant = bool(kw.pop("no_audio", False)), kw.pop("quant", None)
        extra = kw.pop("extra_args", None) or self.config.get("extra_args") or []
        extra = shlex.split(extra) if isinstance(extra, str) else list(extra)
        has_ref = bool(ref_images or ref_audios or ref_videos)
        if has_ref and (image or end_image):
            raise ValueError("Ref2VA references cannot be combined with image/end_image (sd.cpp restriction)")

        notes: List[str] = []
        if is_h3:
            workflow = "ref2va" if has_ref else "fl2va"
            files = self._ensure_h3(workflow, quant)
            t = self.tier
            w, h, note = media.clamp_side(w, h, t.max_side)
            if note: notes.append(note)
            cap = media.floor_aligned(t.max_frames)
            if media.align_frames(nf) > cap:
                nf = cap; notes.append(f"num_frames clamped to {cap} (tier {t.name})")
            flags = (["--offload-to-cpu"] if t.offload else []) + (["--vae-on-cpu"] if t.vae_on_cpu else []) \
                + (["--backend", "te=cpu"] if t.te_cpu else []) + ["--diffusion-fa", "--rng", "cpu"]
            mode = "vid_gen"
            if no_audio: files.pop("audio_vae")
        else:
            prof = self.profiles.load().get(model)
            if not prof:
                raise ValueError(f"unknown model {model!r}. Known: {self.list_models()}")
            files = {k: Path(v) for k, v in prof["files"].items()}
            flags, mode = list(prof.get("args", [])), prof.get("mode", "vid_gen")
        for n in notes: ASCIIColors.warning(f"[sd_cpp] {n}")

        with tempfile.TemporaryDirectory(prefix="sdcpp_") as td:
            d = Path(td)
            args: List[str] = ["-M", mode]
            for role, flag in hub.Profiles.ROLE_FLAGS.items():
                if role in files: args += [flag, str(files[role])]
            args += ["-p", prompt, "-W", str(w), "-H", str(h), "--video-frames", str(nf), "--fps", "24",
                     "--cfg-scale", str(cfg), *flags]
            if neg: args += ["-n", neg]
            if steps: args += ["--steps", str(int(steps))]
            if seed >= 0: args += ["--seed", str(seed)]
            if image: args += ["-i", self._stage(image, d, "init", ".png")]
            if end_image: args += ["--end-img", self._stage(end_image, d, "end", ".png")]
            for i, r in enumerate(ref_images): args += ["-r", self._stage(r, d, f"ref{i}", ".png")]
            for i, a in enumerate(ref_audios): args += ["--ref-audio", self._stage(a, d, f"refaud{i}", ".wav", wav=True)]
            for v in ref_videos: args += ["--ref-video", str(v)]
            for i, a in enumerate(ref_video_audios): args += ["--ref-video-audio", self._stage(a, d, f"refvaud{i}", ".wav", wav=True)]
            args += [*extra, "-o", str(d / "out.mp4")]
            self._run(args, on_line)
            vids = sorted((p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT), key=lambda p: p.stat().st_mtime)
            if not vids:
                raise RuntimeError("sd-cli finished but wrote no video file")
            out = media.to_mp4(vids[-1])
            if out.suffix.lower() != ".mp4":
                ASCIIColors.warning(f"[sd_cpp] sd-cli wrote {out.suffix}; install ffmpeg (pip install imageio-ffmpeg) for mp4")
            return out.read_bytes()

    # ── lip sync (Ref2VA: portrait + audio reference) ────────────────────
    def supports_lipsync(self) -> bool:
        return (self.config.get("model_name") or "minimax_h3").lower() in presets.H3_NAMES

    def generate_lipsync_video(self, image: MediaInput, audio: MediaInput, prompt: str = "", **kwargs) -> bytes:
        if not self.supports_lipsync():
            raise NotImplementedError("lip-sync needs the minimax_h3 model")
        data, ext = self._bytes_of(audio, kwargs.pop("audio_ext", ".wav"))
        wav = media.to_wav_bytes(data, ext)
        secs, fps = media.wav_seconds(wav), 24
        nf = int(kwargs.pop("num_frames", 0) or (math.ceil(secs * fps) + 1 if secs else 56))
        cap = media.floor_aligned(self.tier.max_frames)
        if media.align_frames(nf) > cap:
            nf = cap
            if secs:
                wav = media.wav_truncate(wav, (nf - 1) / fps)
            ASCIIColors.warning(f"[sd_cpp] audio longer than the tier allows: video capped at {nf} frames, audio trimmed")
        template = kwargs.pop("lipsync_prompt_template", LIPSYNC_TEMPLATE)
        kwargs.pop("image_ext", None)
        return self._generate(template.format(prompt=prompt), "minimax_h3", ref_images=[image],
                              ref_audios=[wav], num_frames=nf, **kwargs)

    # ── model management / commands ──────────────────────────────────────
    def list_models(self, **kwargs) -> List[str]:
        return ["minimax_h3"] + sorted(self.profiles.load())

    def get_zoo(self) -> List[Dict[str, Any]]:
        return [{"name": f"MiniMax-H3 FL2VA pruned {q}", "description": f"unsloth GGUF ({q}); text encoder + VAEs fetched with it.",
                 "size": s, "type": "gguf", "link": f"minimax_h3:{q}"} for q, s in H3_LADDER]

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        zoo = self.get_zoo()
        if not 0 <= index < len(zoo):
            return {"status": False, "message": "Index out of bounds"}
        return self.install_model("minimax_h3", quant=zoo[index]["link"].split(":", 1)[1], workflow="fl2va")

    def install_runtime(self, force: bool = False, accelerator: Optional[str] = None) -> dict:
        try:
            if force or not self.config.get("sd_cli_path"):
                self._cli_path = runtime.install_runtime(self.root, accelerator or self.config.get("accelerator", "auto"), self._log) \
                    if force else self._cli()
            return {"status": True, "message": f"sd-cli: {self._cli_path}"}
        except Exception as e:
            self._last_error = str(e); return {"status": False, "message": str(e)}

    def list_repo_files(self, repo_id: str, pattern: Optional[str] = None, revision: Optional[str] = None) -> dict:
        try:
            fs = hub.list_repo_files(repo_id, self.hf_token, revision)
            if pattern: fs = hub.match_pattern(fs, pattern)
            return {"status": True, "files": fs, "message": f"{len(fs)} file(s)"}
        except Exception as e:
            return {"status": False, "files": [], "message": str(e)}

    def download_model(self, repo_id: str, filename: Optional[str] = None, pattern: Optional[str] = None,
                       subfolder: str = "other", revision: Optional[str] = None) -> dict:
        """Downloads from any HF repo into models_path/<subfolder>/. Refuses to pull a whole repo blindly."""
        if subfolder not in hub.SUBFOLDERS:
            return {"status": False, "message": f"subfolder must be one of {hub.SUBFOLDERS}"}
        try:
            files = hub.list_repo_files(repo_id, self.hf_token, revision)
            if filename:
                hit = hub.find_exact(files, filename)
                picked = [hit] if hit else []
            elif pattern:
                picked = hub.match_pattern(files, pattern)
            else:
                return {"status": False, "message": "give filename or pattern", "candidates": files[:100]}
            if not picked:
                return {"status": False, "message": "no matching file", "candidates": files[:100]}
            paths = [str(hub.download_file(repo_id, f, self.models_path / subfolder, self.hf_token, revision, self._log)) for f in picked]
            return {"status": True, "paths": paths, "message": f"{len(paths)} file(s) in {self.models_path / subfolder}"}
        except Exception as e:
            self._last_error = str(e); return {"status": False, "message": str(e)}

    def register_model(self, name: str, files: Dict[str, str], args: Optional[List[str]] = None, mode: str = "vid_gen") -> dict:
        """files: {role: path}, roles: diffusion_model, vae, audio_vae, llm, llm_vision. args: extra sd-cli flags."""
        try:
            self.profiles.register(name, files, args, mode)
            return {"status": True, "message": f"registered '{name}'"}
        except Exception as e:
            return {"status": False, "message": str(e)}

    def install_model(self, model_name: str, **kwargs) -> dict:
        """'minimax_h3' (tier-appropriate bundle; workflow=fl2va|ref2va|both, quant=...) or an HF
        'owner/repo' (+ filename=/pattern=) or 'owner/repo::filename'."""
        try:
            if model_name.lower() in presets.H3_NAMES:
                self._cli()
                wfs = ["fl2va", "ref2va"] if kwargs.get("workflow") == "both" else [kwargs.get("workflow", "fl2va")]
                got = {w: {k: str(v) for k, v in self._ensure_h3(w, kwargs.get("quant")).items()} for w in wfs}
                return {"status": True, "message": f"MiniMax-H3 ready (tier {self.tier.name})", "files": got}
            if "/" in model_name:
                repo, _, fn = model_name.partition("::")
                return self.download_model(repo, filename=fn or kwargs.get("filename"), pattern=kwargs.get("pattern"),
                                           subfolder=kwargs.get("subfolder", "other"), revision=kwargs.get("revision"))
            return {"status": model_name in self.profiles.load(), "message": f"unknown model {model_name!r}"}
        except Exception as e:
            self._last_error = str(e); return {"status": False, "message": str(e)}

    def pull_model(self, model_name: str, **kwargs) -> dict:
        return self.install_model(model_name, **kwargs)
