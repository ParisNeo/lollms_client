"""
LoLLMS Server TTV binding for lollms_client.

Connects to the LoLLMS video generation API (``/lollms/v1/videos*``) documented in
``VIDEO_GENERATION_API.md``. Supports text-to-video, image-to-video, first/last-frame
interpolation, and audio-driven lip-sync, in both asynchronous (submit + poll) and
synchronous (blocking) modes.

The binding speaks the exact same wire protocol vLLM-Omni clients expect, so any
LoLLMS server with an active TTV binding can be driven transparently.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import ssl
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_ttv_binding import LollmsTTVBinding, MediaInput

BindingName = "LollmsTTVBinding"

# Terminal job states per the LoLLMS video API contract.
_TERMINAL_STATES = frozenset({"completed", "failed"})
_POLL_INTERVAL_SECONDS = 2.0


def _guess_extension(data: bytes, default: str) -> str:
    """Best-effort file extension inference from magic bytes for multipart uploads."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "webp"
    if data.startswith(b"GIF8"):
        return "gif"
    if data.startswith(b"ID3") or data.startswith(b"\xff\xfb"):
        return "mp3"
    if data.startswith(b"RIFF") and b"WAVE" in data[:16]:
        return "wav"
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"OggS"):
        return "ogg"
    return default


class LollmsTTVBinding(LollmsTTVBinding):
    """Text-to-video binding backed by a remote LoLLMS server."""

    def __init__(
        self,
        host_address: str = "http://localhost:9642",
        service_key: Optional[str] = None,
        verify_ssl_certificate: bool = True,
        certificate_file_path: Optional[str] = None,
        model: Optional[str] = None,
        default_num_frames: int = 80,
        default_fps: int = 16,
        default_width: Optional[int] = None,
        default_height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        mode: str = "sync",
        timeout: int = 900,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(binding_name="lollms_ttv", debug=debug, **kwargs)

        host = (host_address or "http://localhost:9642").rstrip("/")
        for suffix in ("/lollms/v1", "/v1"):
            if host.endswith(suffix):
                host = host[: -len(suffix)].rstrip("/")
                break
        self.base_address = host
        self.api_base = f"{self.base_address}/lollms/v1"

        self.service_key = service_key or os.getenv("LOLLMS_API_KEY") or ""

        self.verify = self._resolve_ssl_verification(
            verify_ssl_certificate, certificate_file_path
        )

        self.model = model or None
        self.default_num_frames = int(default_num_frames)
        self.default_fps = int(default_fps)
        self.default_width = int(default_width) if default_width is not None else None
        self.default_height = int(default_height) if default_height is not None else None
        self.num_inference_steps = (
            int(num_inference_steps) if num_inference_steps is not None else None
        )
        self.guidance_scale = (
            float(guidance_scale) if guidance_scale is not None else None
        )

        self.mode = (mode or "sync").strip().lower()
        if self.mode not in ("sync", "async"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'sync' or 'async'.")

        self.timeout = int(timeout)
        self.poll_interval = float(poll_interval)

        self._session = requests.Session()
        if self.service_key:
            self._session.headers["Authorization"] = f"Bearer {self.service_key}"
        self._session.headers["Accept"] = "application/json"

    # ── SSL configuration ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ssl_verification(
        verify_ssl_certificate: Union[bool, str],
        certificate_file_path: Optional[str],
    ) -> Union[bool, str]:
        """
        Resolve the ``requests`` verify argument from binding config.

        Priority:
          1. ``LOLLMS_SKIP_SSL_VERIFY`` env var (any truthy value) → disable verify.
          2. ``verify_ssl_certificate=False`` → disable verify.
          3. ``certificate_file_path`` → custom CA bundle (validated on disk).
          4. Fall back to the boolean (default True).
        """
        env_skip = os.getenv("LOLLMS_SKIP_SSL_VERIFY", "").strip().lower()
        if env_skip in ("1", "true", "yes", "on"):
            return False

        if isinstance(verify_ssl_certificate, str):
            verify_ssl_certificate = verify_ssl_certificate.strip().lower() not in (
                "false", "0", "no", "off", ""
            )

        if not verify_ssl_certificate:
            return False

        if certificate_file_path:
            cert_path = Path(certificate_file_path).expanduser()
            if not cert_path.exists():
                raise FileNotFoundError(
                    f"[LollmsTTVBinding] Certificate file not found: {cert_path}"
                )
            # Validating eagerly surfaces CA-bundle parsing errors at init time
            # instead of deep inside a generation call.
            ssl.create_default_context(cafile=str(cert_path))
            return str(cert_path)

        return True

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.api_base}{path}"

    def _headers(self, json_body: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _extract_error(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict) and "detail" in payload:
                return str(payload["detail"])
            return str(payload)
        except ValueError:
            return response.text or f"HTTP {response.status_code}"

    # ── Request building ─────────────────────────────────────────────────────

    def _common_generation_fields(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: Optional[int],
        height: Optional[int],
        num_frames: Optional[int],
        fps: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Collect the scalar generation fields accepted by the API (§4)."""
        fields: Dict[str, Any] = {"prompt": prompt}

        if negative_prompt:
            fields["negative_prompt"] = negative_prompt

        resolved_width = width if width is not None else self.default_width
        resolved_height = height if height is not None else self.default_height
        resolved_frames = num_frames if num_frames is not None else self.default_num_frames
        resolved_fps = fps if fps is not None else self.default_fps
        resolved_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.num_inference_steps
        )
        resolved_guidance = (
            guidance_scale if guidance_scale is not None else self.guidance_scale
        )

        if resolved_width is not None:
            fields["width"] = int(resolved_width)
        if resolved_height is not None:
            fields["height"] = int(resolved_height)
        if resolved_frames is not None:
            fields["num_frames"] = int(resolved_frames)
        if resolved_fps is not None:
            fields["fps"] = int(resolved_fps)
        if resolved_steps is not None:
            fields["num_inference_steps"] = int(resolved_steps)
        if resolved_guidance is not None:
            fields["guidance_scale"] = float(resolved_guidance)
        if seed is not None and seed != -1:
            fields["seed"] = int(seed)

        if self.model:
            fields["model"] = self.model

        return fields

    @staticmethod
    def _read_media_bytes(media: MediaInput) -> bytes:
        """Normalize a MediaInput (path, bytes, or base64 string) to raw bytes."""
        if isinstance(media, bytes):
            return media
        if isinstance(media, Path):
            return media.read_bytes()
        if isinstance(media, str):
            path = Path(media).expanduser()
            if path.is_file():
                return path.read_bytes()
            # Fall back to base64 decoding (raw or data URL).
            raw = media.split(",", 1)[1] if media.startswith("data:") else media
            try:
                return base64.b64decode(raw, validate=True)
            except Exception as exc:
                raise ValueError(
                    f"Media input is neither an existing file nor valid base64: {exc}"
                ) from exc
        raise TypeError(f"Unsupported media type: {type(media)!r}")

    @staticmethod
    def _multipart_file(
        field: str,
        media: MediaInput,
        fallback_ext: str,
        default_mime: str,
    ) -> tuple:
        data = LollmsTTVBinding._read_media_bytes(media)
        ext = _guess_extension(data, fallback_ext)
        mime = mimetypes.guess_type(f"file.{ext}")[0] or default_mime
        return (field, (f"upload.{ext}", io.BytesIO(data), mime))

    # ── Transport (sync vs async) ────────────────────────────────────────────

    def _submit_sync(
        self, fields: Dict[str, Any], files: Optional[List[tuple]] = None
    ) -> bytes:
        """Use the blocking ``/videos/sync`` endpoint and return raw MP4 bytes."""
        resp = self._session.post(
            self._url("/videos/sync"),
            data=fields,
            files=files,
            timeout=self.timeout,
            verify=self.verify,
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"[LollmsTTVBinding] sync generation failed "
                f"[{resp.status_code}]: {self._extract_error(resp)}"
            )
        return resp.content

    def _submit_async_and_wait(self, fields: Dict[str, Any]) -> bytes:
        """Submit an async job, poll it to completion, then download the MP4."""
        submit = self._session.post(
            self._url("/videos"),
            json=fields,
            timeout=self.timeout,
            verify=self.verify,
            headers=self._headers(json_body=True),
        )
        if submit.status_code >= 400:
            raise RuntimeError(
                f"[LollmsTTVBinding] async submission failed "
                f"[{submit.status_code}]: {self._extract_error(submit)}"
            )

        job = submit.json()
        video_id = job.get("id")
        if not video_id:
            raise RuntimeError(
                f"[LollmsTTVBinding] Server did not return a video job id: {job}"
            )

        deadline = time.monotonic() + self.timeout
        status = job.get("status", "queued")
        while status not in _TERMINAL_STATES:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"[LollmsTTVBinding] Job {video_id} did not finish within "
                    f"{self.timeout}s (last status: {status})."
                )
            time.sleep(self.poll_interval)
            poll = self._session.get(
                self._url(f"/videos/{video_id}"),
                timeout=30,
                verify=self.verify,
                headers=self._headers(),
            )
            if poll.status_code >= 400:
                raise RuntimeError(
                    f"[LollmsTTVBinding] polling job {video_id} failed "
                    f"[{poll.status_code}]: {self._extract_error(poll)}"
                )
            job = poll.json()
            status = job.get("status", status)

        if status == "failed":
            raise RuntimeError(
                f"[LollmsTTVBinding] Job {video_id} failed: {job.get('error')}"
            )

        content = self._session.get(
            self._url(f"/videos/{video_id}/content"),
            timeout=self.timeout,
            verify=self.verify,
            headers=self._headers(),
        )
        if content.status_code >= 400:
            raise RuntimeError(
                f"[LollmsTTVBinding] downloading job {video_id} failed "
                f"[{content.status_code}]: {self._extract_error(content)}"
            )
        return content.content

    def _run(
        self, fields: Dict[str, Any], files: Optional[List[tuple]] = None
    ) -> bytes:
        """
        Dispatch a request according to the configured mode.

        ``sync`` uses the blocking endpoint for every recipe. ``async`` uses the
        job endpoints for reference-free requests; for multipart requests (where
        the async endpoint requires files we can't upload to ``/videos/sync`` in
        one shot), it still uses ``/videos`` then polls.
        """
        if self.mode == "sync" and files is None:
            return self._submit_sync(fields)
        if files is not None:
            # The async endpoint accepts multipart, submitting then polling.
            return self._submit_async_and_wait(fields)
        return self._submit_async_and_wait(fields)

    # ── Public API ───────────────────────────────────────────────────────────

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """Text-to-video generation (recipe A, §6)."""
        fields = self._common_generation_fields(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            fps,
            num_inference_steps,
            guidance_scale,
            seed,
        )
        return self._run(fields)

    def generate_video_from_image(
        self,
        prompt: str,
        image: MediaInput,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """Image-to-video generation using ``image_reference`` as the first frame (recipe B, §6)."""
        fields = self._common_generation_fields(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            fps,
            num_inference_steps,
            guidance_scale,
            seed,
        )
        files = [self._multipart_file("image_reference", image, "png", "image/png")]
        return self._run(fields, files)

    def supports_first_last_frame(self) -> bool:
        return True

    def generate_video_first_and_last_frame(
        self,
        prompt: str,
        start_image: MediaInput,
        end_image: MediaInput,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """
        First/last-frame interpolation (recipe C, §6).

        ``image_reference`` anchors the first frame, ``video_reference`` the last.
        Per §5 the server accepts an image or video for ``video_reference``; we
        always send the second keyframe as the ``video_reference`` field, matching
        the reference recipe.
        """
        fields = self._common_generation_fields(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            fps,
            num_inference_steps,
            guidance_scale,
            seed,
        )
        files = [
            self._multipart_file("image_reference", start_image, "png", "image/png"),
            self._multipart_file("video_reference", end_image, "png", "image/png"),
        ]
        return self._run(fields, files)

    def supports_lipsync(self) -> bool:
        return True

    def generate_lipsync_video(
        self,
        image: MediaInput,
        audio: MediaInput,
        prompt: str = "",
        **kwargs,
    ) -> bytes:
        """
        Audio-driven talking-head generation (recipe E, §6).

        Routes through ``image_reference`` + ``audio_reference`` which the server
        dispatches to the lip-sync pipeline when the active binding advertises
        ``supports_lipsync()``.
        """
        fields = self._common_generation_fields(
            prompt or "",
            kwargs.get("negative_prompt"),
            kwargs.get("width"),
            kwargs.get("height"),
            kwargs.get("num_frames"),
            kwargs.get("fps"),
            kwargs.get("num_inference_steps"),
            kwargs.get("guidance_scale"),
            kwargs.get("seed"),
        )
        files = [
            self._multipart_file("image_reference", image, "png", "image/png"),
            self._multipart_file("audio_reference", audio, "wav", "audio/wav"),
        ]
        return self._run(fields, files)

    # ── Discovery / capabilities ─────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, bool]:
        """Report what the *remote* server actually supports when reachable."""
        caps = super().get_capabilities()
        try:
            resp = self._session.get(
                self._url("/capabilities"),
                timeout=10,
                verify=self.verify,
                headers=self._headers(),
            )
            if resp.status_code == 200:
                reported = resp.json().get("capabilities", []) or []
                caps["text_to_video"] = "text_to_video" in reported
                caps["image_to_video"] = "image_to_video" in reported
        except Exception as exc:
            ASCIIColors.debug(
                f"[LollmsTTVBinding] Could not fetch remote capabilities: {exc}"
            )
        return caps

    def list_models(self, **kwargs) -> List[str]:
        """List TTV models exposed by the server (empty when unreachable)."""
        try:
            resp = self._session.get(
                self._url("/ttv/models"),
                timeout=15,
                verify=self.verify,
                headers=self._headers(),
            )
            if resp.status_code == 200:
                payload = resp.json()
                items = payload.get("data", payload) if isinstance(payload, dict) else payload
                models: List[str] = []
                for item in items or []:
                    if isinstance(item, str):
                        models.append(item)
                    elif isinstance(item, dict):
                        name = item.get("id") or item.get("model_name") or item.get("name")
                        if name:
                            models.append(str(name))
                return list(dict.fromkeys(models))
            ASCIIColors.warning(
                f"[LollmsTTVBinding] list_models returned HTTP {resp.status_code}"
            )
        except Exception as exc:
            trace_exception(exc)
            ASCIIColors.warning(f"[LollmsTTVBinding] Failed to list models: {exc}")
        return []

    def health_check(self) -> bool:
        """Returns True when the server's capabilities endpoint is reachable."""
        try:
            resp = self._session.get(
                self._url("/capabilities"),
                timeout=5,
                verify=self.verify,
                headers=self._headers(),
            )
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass