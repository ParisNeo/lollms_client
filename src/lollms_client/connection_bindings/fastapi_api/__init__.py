"""
FastAPI HTTP API CONNECTION binding for lollms_client.

Exposes the LLM over HTTP (POST /chat, GET /health) via FastAPI + uvicorn,
served in-process on the configured host:port.
"""
from typing import List, Optional

from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, ConnectionSendResult

BindingName = "FastAPIAPIConnectionBinding"


class FastAPIAPIConnectionBinding(LollmsConnectionBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="fastapi_api", **kwargs)
        self.config = kwargs
        host = str(kwargs.get("host_address", "127.0.0.1:8000"))
        bare = host.replace("http://", "").replace("https://", "").split("/")[-1]
        if ":" in bare:
            h, p = bare.rsplit(":", 1)
            self.host = h or "127.0.0.1"
            self.port = int(p) if p.isdigit() else 8000
        else:
            self.host = "127.0.0.1"
            self.port = int(bare) if bare.isdigit() else 8000
        self.lollms_client = kwargs.get("lollms_client")
        self._running = False

    def list_channels(self) -> List[str]:
        return [f"http://{self.host}:{self.port}"]

    def send_message(self, content: str, channel_alias: Optional[str] = None, **kwargs) -> ConnectionSendResult:
        if not self._running:
            return ConnectionSendResult(success=False, message="FastAPI API is not running.", channel=channel_alias)
        return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="fastapi_api")

    async def start(self) -> None:
        try:
            import uvicorn
            from fastapi import FastAPI
            from pydantic import BaseModel
        except ImportError:
            ASCIIColors.red("fastapi/uvicorn not installed. Run: pip install 'fastapi' 'uvicorn[standard]'")
            return

        self._running = True
        app = FastAPI(title="Lollms Bot API")

        class ChatRequest(BaseModel):
            message: str

        class ChatResponse(BaseModel):
            reply: str

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/chat", response_model=ChatResponse)
        async def chat(req: ChatRequest):
            return ChatResponse(reply=self._reply(req.message or ""))

        ASCIIColors.info(f"🌐 FastAPI connection binding on http://{self.host}:{self.port}")
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.red(f"FastAPI binding error: {e}")
        finally:
            self._running = False

    def _reply(self, text: str) -> str:
        client = self.lollms_client
        if client is not None and getattr(client, "llm", None) is not None:
            try:
                out = client.llm.generate_text(prompt=text)
                if isinstance(out, dict):
                    return out.get("message", text)
                return out or text
            except Exception:
                return text
        return text