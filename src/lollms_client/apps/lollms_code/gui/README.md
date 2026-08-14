# lollms_code GUI

NiceGUI native-window front end for your `lollms_code` autonomous agent —
replaces the terminal REPL and the CLI setup wizard with a proper desktop app.

## Structure

```
main.py                 entry point — native window, routing, theming
app/config.py            AppConfig dataclass, replaces CodeAgentConfig, persisted to
                          ~/.lollms_client/lollms_code/gui_config.json
app/agent_bridge.py       LollmsClient / LollmsPersonality creation + threaded
                          streaming chat call — same logic as the CLI's
                          create_client()/create_coding_personality(), adapted
                          to push events onto a queue instead of printing
app/settings_page.py      in-app Settings UI — replaces run_wizard_and_save()
app/chat_page.py          chat/session UI — replaces run_interactive()
```

## One thing to fix before running

`app/agent_bridge.py` tries to import your real `CODING_SYSTEM_PROMPT` constant
from `lollms_code_cli`:

```python
from lollms_code_cli import CODING_SYSTEM_PROMPT
```

Point that import at wherever your existing CLI module actually lives (or
just paste the constant directly into `agent_bridge.py`) — it's a placeholder
so the GUI still boots even if the import fails.

## Run

```bash
pip install -r requirements.txt
python main.py
```

First launch drops you into Settings (this is the wizard replacement) since
`configured` starts `False`. Everything you'd have answered in the CLI
wizard — binding, model, host, API key, context size, shell autonomy,
sub-agents, memory, skills mode, paths — is a live form now, saved to JSON
and editable anytime via the gear icon.

## What maps to what (CLI → GUI)

| CLI concept | GUI equivalent |
|---|---|
| `run_wizard_and_save()` | Settings page, "Connection" + "Agent Behavior" tabs |
| `CodeAgentConfig` (argparse + env) | `AppConfig` (JSON-persisted, edited via form) |
| `StreamRenderer` (Rich panels in terminal) | `QueueStreamingCallback` → expandable cards in the transcript |
| `run_interactive()` REPL loop | `chat_page.py` — text input + streaming transcript |
| `/skills`, `/forget`, `/config` slash commands | Not yet wired — see "Next steps" |
| Native window | `ui.run(native=True, window_size=...)` (pywebview under the hood) |

## Customization already built in

- Dark/light mode toggle (header icon + Settings)
- Accent color (5 presets + custom color picker)
- Font family for the whole app
- Window size (applies next launch)
- Per-panel visibility toggles: tool calls, workspace changes, skills activity

## Next steps you'll likely want

- Wire `/skills`, `/forget`, `/files` as sidebar actions or a menu (the
  bridge functions they need — `personality.skills_manager`,
  `personality.wipe_all_memories()` — are already used by the CLI, just not
  yet exposed in `chat_page.py`).
- Multi-session tabs (currently one `ChatSession` per page load).
- Workspace file browser panel using `get_workspace_stats()` logic from the CLI.
- Standalone packaging: `pyinstaller --onefile --windowed --name lollms-code-gui main.py`
  once you've confirmed it runs cleanly against your real `lollms_client`.
