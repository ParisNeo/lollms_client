"""
connection_bindings — LOLLMS Connection Modality Bindings.

Hosts concrete connection binding implementations. Each sub-package exposes a
`LollmsConnectionBinding` subclass that the connection manager can instantiate
by name (e.g. `"telegram"`, `"discord"`, `"slack"`, `"fastapi_api"`,
`"generic_webhook"`, `"lollms"`, `"whatsapp"`).

Available bindings:
    - discord         : Discord bot connection.
    - fastapi_api     : In-process FastAPI HTTP API server.
    - generic_webhook : Generic HTTP webhook sender.
    - lollms          : lollms.com web gateway bridge.
    - slack           : Slack bot connection.
    - telegram        : Telegram bot connection.
    - whatsapp        : WhatsApp bridge (guarded stub).

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the connection manager.
"""