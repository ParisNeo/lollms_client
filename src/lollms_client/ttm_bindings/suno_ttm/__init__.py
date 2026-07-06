from __future__ import annotations

import os
import time
import uuid
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import unquote

from ascii_colors import trace_exception
from lollms_client.lollms_ttm_binding import LollmsTTMBinding

BindingName = "SunoTTMBinding"


class SunoTTMBinding(LollmsTTMBinding):
    """LOLLMS TTM binding for Suno AI using browser-session cookies.

    Supports:
      - generate_music: instrumental mode (prompt only)
      - generate_song: simple mode (prompt + auto lyrics)
      - generate_song_from_lyrics: custom mode (prompt + explicit lyrics)
    """

    def __init__(
        self,
        binding_name: str = "suno",
        debug: Optional[bool] = False,
        cookie: Optional[str] = None,
        cookie_env_var: str = "SUNO_COOKIE",
        output_dir: Optional[str] = None,
        auth_base_url: str = "https://auth.suno.com",
        api_base_url: str = "https://studio-api.prod.suno.com",
        request_timeout: int = 60,
        poll_interval: float = 5.0,
        max_poll_retries: int = 60,
        user_agent: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)
        self.cookie = cookie or os.environ.get(cookie_env_var, "")
        self.cookie_env_var = cookie_env_var
        self.output_dir = Path(output_dir or Path.cwd() / "suno_generations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auth_base_url = auth_base_url.rstrip("/")
        self.api_base_url = api_base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.max_poll_retries = max_poll_retries
        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._cookie_map = self._parse_cookie_string(self.cookie)
        self._jwt: Optional[str] = None
        self._session_id: Optional[str] = None
        self._device_id = self._extract_device_id(self._cookie_map)

    def _debug(self, message: str):
        if getattr(self, "debug", False):
            print(f"[SunoTTMBinding] {message}")

    @staticmethod
    def _parse_cookie_string(cookie_string: str) -> Dict[str, str]:
        cookies: Dict[str, str] = {}
        for part in cookie_string.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            cookies[key.strip()] = value.strip()
        return cookies

    @staticmethod
    def _extract_device_id(cookie_map: Dict[str, str]) -> str:
        raw = cookie_map.get("ajs_anonymous_id")
        if not raw:
            return str(uuid.uuid4())
        raw = unquote(raw).strip().strip('"')
        return raw or str(uuid.uuid4())

    def _require_cookie(self):
        if not self.cookie:
            raise ValueError(
                f"Missing Suno cookie. Pass cookie=... or define {self.cookie_env_var}."
            )
        if "__client" not in self._cookie_map:
            raise ValueError(
                "Invalid SUNO_COOKIE: missing __client field required for Clerk auth."
            )

    def _auth_headers(self) -> Dict[str, str]:
        self._require_cookie()
        return {
            "Authorization": self._cookie_map["__client"],
            "Cookie": self.cookie,
            "Accept": "application/json",
            "Origin": "https://suno.com",
            "Referer": "https://suno.com/",
        }

    def _api_headers(self) -> Dict[str, str]:
        if not self._jwt:
            self.keep_alive()
        return {
            "Authorization": f"Bearer {self._jwt}",
            "Cookie": self.cookie,
            "Device-Id": self._device_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": "https://suno.com",
            "Referer": "https://suno.com/",
        }

    def keep_alive(self) -> str:
        self._require_cookie()
        try:
            client_url = (
                f"{self.auth_base_url}/v1/client"
                "?__clerk_api_version=2025-11-10&_clerk_js_version=5.117.0"
            )
            r = self.session.get(
                client_url,
                headers=self._auth_headers(),
                timeout=self.request_timeout,
            )
            r.raise_for_status()
            client_info = r.json()
            self._session_id = client_info.get("response", {}).get("last_active_session_id") or client_info.get("last_active_session_id")
            if not self._session_id:
                raise RuntimeError("Failed to get session id from Clerk response")

            token_url = f"{self.auth_base_url}/v1/client/sessions/{self._session_id}/tokens"
            r = self.session.post(
                token_url,
                headers=self._auth_headers(),
                timeout=self.request_timeout,
            )
            r.raise_for_status()
            token_info = r.json()
            self._jwt = token_info.get("jwt") or token_info.get("response", {}).get("jwt")
            if not self._jwt:
                raise RuntimeError("Failed to get JWT from Clerk token response")
            self._debug("JWT refreshed successfully")
            return self._jwt
        except Exception as e:
            trace_exception(e)
            raise

    def _post_generate(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        make_instrumental: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "make_instrumental": make_instrumental,
            "wait_audio": kwargs.get("wait_audio", False),
        }

        if kwargs.get("model"):
            payload["mv"] = kwargs["model"]

        if lyrics and not make_instrumental:
            # Custom mode: explicit lyrics
            payload["lyrics"] = lyrics
            if kwargs.get("tags"):
                payload["tags"] = kwargs["tags"]
            if kwargs.get("title"):
                payload["title"] = kwargs["title"]
            if kwargs.get("negative_tags"):
                payload["negative_tags"] = kwargs["negative_tags"]
        else:
            # Simple mode or instrumental: no explicit lyrics
            if kwargs.get("tags"):
                payload["tags"] = kwargs["tags"]
            if kwargs.get("title"):
                payload["title"] = kwargs["title"]
            if kwargs.get("negative_tags"):
                payload["negative_tags"] = kwargs["negative_tags"]

        r = self.session.post(
            f"{self.api_base_url}/api/generate/v2/",
            json=payload,
            headers=self._api_headers(),
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "clips" in data:
            return data["clips"]
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise RuntimeError(f"Unexpected Suno generate response: {data}")

    def _fetch_clips(self, clip_ids: List[str]) -> List[Dict[str, Any]]:
        ids = ",".join(clip_ids)
        r = self.session.get(
            f"{self.api_base_url}/api/feed/?ids={ids}",
            headers=self._api_headers(),
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "clips" in data:
            return data["clips"]
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise RuntimeError(f"Unexpected Suno feed response: {data}")

    def _wait_for_audio(self, clip_ids: List[str]) -> List[Dict[str, Any]]:
        for _ in range(self.max_poll_retries):
            clips = self._fetch_clips(clip_ids)
            ready = [c for c in clips if c.get("audio_url")]
            if len(ready) >= 1:
                return clips
            time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Suno audio generation")

    def _download_first_audio(self, clips: List[Dict[str, Any]]) -> bytes:
        for clip in clips:
            audio_url = clip.get("audio_url")
            if not audio_url:
                continue
            r = self.session.get(audio_url, timeout=self.request_timeout)
            r.raise_for_status()
            return r.content
        raise RuntimeError("No audio_url found in generated Suno clips")

    def _execute_generation(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        make_instrumental: bool = False,
        **kwargs
    ) -> bytes:
        clips = self._post_generate(
            prompt,
            lyrics=lyrics,
            make_instrumental=make_instrumental,
            **kwargs
        )
        clip_ids = [c.get("id") for c in clips if c.get("id")]
        if not clip_ids:
            raise RuntimeError("Suno did not return clip ids")

        wait_audio = kwargs.get("wait_audio", True)
        if wait_audio:
            clips = self._wait_for_audio(clip_ids)

        audio_bytes = self._download_first_audio(clips)

        if kwargs.get("save_all", False):
            self._save_all(clips, kwargs.get("file_prefix", "suno_generation"))

        return audio_bytes

    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generate instrumental music from a text prompt.

        Extra kwargs:
            model: optional model/version string, forwarded as mv
            wait_audio: bool (default True inside binding)
            save_all: bool
            file_prefix: optional output file prefix
            tags/title/negative_tags: optional custom fields
        """
        return self._execute_generation(
            prompt,
            lyrics=None,
            make_instrumental=True,
            **kwargs
        )

    def generate_song(self, prompt: str, **kwargs) -> bytes:
        """
        Generate a song with auto-generated lyrics from a text prompt.

        Extra kwargs:
            model: optional model/version string, forwarded as mv
            wait_audio: bool (default True inside binding)
            save_all: bool
            file_prefix: optional output file prefix
            tags/title/negative_tags: optional custom fields
        """
        return self._execute_generation(
            prompt,
            lyrics=None,
            make_instrumental=False,
            **kwargs
        )

    def generate_song_from_lyrics(self, prompt: str, lyrics: str, **kwargs) -> bytes:
        """
        Generate a song with explicit lyrics from a text prompt and lyrics.

        Extra kwargs:
            model: optional model/version string, forwarded as mv
            wait_audio: bool (default True inside binding)
            save_all: bool
            file_prefix: optional output file prefix
            tags/title/negative_tags: optional custom fields
        """
        return self._execute_generation(
            prompt,
            lyrics=lyrics,
            make_instrumental=False,
            **kwargs
        )

    def _save_all(self, clips: List[Dict[str, Any]], prefix: str = "suno_generation") -> List[Path]:
        saved = []
        for idx, clip in enumerate(clips):
            audio_url = clip.get("audio_url")
            if not audio_url:
                continue
            r = self.session.get(audio_url, timeout=self.request_timeout)
            r.raise_for_status()
            path = self.output_dir / f"{prefix}_{idx}.mp3"
            path.write_bytes(r.content)
            saved.append(path)
        return saved

    def list_models(self, **kwargs) -> List[str]:
        """
        Unofficial binding: returns known model/version labels accepted by many community Suno wrappers.
        Actual availability depends on the current Suno backend.
        """
        return ["chirp-v3-0", "chirp-v3-5", "chirp-v4", "chirp-v4-5", "chirp-v5"]

    def get_zoo(self) -> List[Dict[str, Any]]:
        return []