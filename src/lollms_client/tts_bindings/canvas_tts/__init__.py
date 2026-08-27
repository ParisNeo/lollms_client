from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_tts_binding import LollmsTTSBinding

BindingName = "CanvasTTSBinding"

class CanvasTTSBinding(LollmsTTSBinding):
    """
    Binding for Canvas + TTS synchronized audio narration and browser video recording.
    Implements standard LollmsTTSBinding while providing Web Audio routing manifests
    and client-side recording orchestration templates.
    """

    def __init__(self, **kwargs):
        super().__init__(binding_name="canvas_tts", **kwargs)
        self.model_name = kwargs.get("model_name", "browser_speech_synthesis")
        self.audio_format = kwargs.get("audio_format", "webm")
        self.fps = kwargs.get("fps", 30)
        self.sample_rate = kwargs.get("sample_rate", 44100)
        self.default_voice = kwargs.get("voice", "default")

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """
        Generates audio payload descriptor or audio bytes.
        For browser-side Canvas + TTS orchestration, returns a serialized JSON manifest
        containing the synthesized speech parameters and Web Audio routing instructions.
        """
        selected_voice = voice or self.default_voice
        rate = kwargs.get("rate", 1.0)
        pitch = kwargs.get("pitch", 1.0)
        audio_url = kwargs.get("audio_url", None)

        manifest: Dict[str, Any] = {
            "text": text,
            "voice": selected_voice,
            "rate": rate,
            "pitch": pitch,
            "audio_format": self.audio_format,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "audio_url": audio_url,
            "mode": "external_tts" if audio_url else "browser_speech_synthesis"
        }

        try:
            return json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            ASCIIColors.error(f"CanvasTTSBinding: Error creating audio payload: {e}")
            trace_exception(e)
            return b""

    def list_voices(self, **kwargs) -> List[str]:
        """
        Returns supported voice identifiers and browser speech synthesis aliases.
        """
        return [
            "default",
            "browser_native_female",
            "browser_native_male",
            "browser_en_us_1",
            "browser_en_gb_1",
            "browser_fr_fr_1"
        ]

    def list_models(self, **kwargs) -> List[str]:
        """
        Returns available Canvas TTS integration models.
        """
        return [
            "browser_speech_synthesis",
            "external_api_web_audio",
            "canvas_stream_recorder"
        ]

    def get_recorder_script(self, canvas_selector: str = "canvas", audio_url: Optional[str] = None) -> str:
        """
        Generates production-grade client-side JavaScript to record HTML canvas visuals
        synchronized with Web Audio narration according to the Canvas + TTS skill guardrails.
        """
        escaped_selector = json.dumps(canvas_selector)
        escaped_url = json.dumps(audio_url or "")

        return f"""
(async function initCanvasTTSRecorder() {{
    const canvas = document.querySelector({escaped_selector});
    if (!canvas) throw new Error("Canvas element not found: " + {escaped_selector});

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const dest = audioCtx.createMediaStreamDestination();
    const audioUrl = {escaped_url};

    function pickSupportedMimeType() {{
        const options = [
            "video/webm;codecs=vp9,opus",
            "video/webm;codecs=vp8,opus",
            "video/webm"
        ];
        const found = options.find(t => MediaRecorder.isTypeSupported(t));
        if (!found) throw new Error("No supported video/webm MIME type in this browser");
        return found;
    }}

    async function recordWithExternalAudio(url) {{
        const audioEl = new Audio(url);
        audioEl.crossOrigin = "anonymous";
        const src = audioCtx.createMediaElementSource(audioEl);
        src.connect(dest);
        src.connect(audioCtx.destination);

        const canvasStream = canvas.captureStream({self.fps});
        const combined = new MediaStream([
            ...canvasStream.getVideoTracks(),
            ...dest.stream.getAudioTracks()
        ]);

        const recorder = new MediaRecorder(combined, {{
            mimeType: pickSupportedMimeType(),
            videoBitsPerSecond: 2500000,
            audioBitsPerSecond: 128000
        }});

        const chunks = [];
        recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
        recorder.onstop = () => {{
            if (!chunks.length) throw new Error("No data captured — recording failed");
            const blob = new Blob(chunks, {{ type: "video/webm" }});
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "canvas_tts_export.webm";
            link.click();
            setTimeout(() => URL.revokeObjectURL(link.href), 1000);
        }};

        recorder.start(1000);
        await audioEl.play();
        audioEl.onended = () => recorder.stop();
        return recorder;
    }}

    if (audioUrl) {{
        return await recordWithExternalAudio(audioUrl);
    }}
}})();
"""