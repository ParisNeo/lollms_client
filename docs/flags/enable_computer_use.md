# `enable_computer_use`

```python
enable_computer_use: bool = False
```

**Category**: Tools & Security

## What it does

Mounts the desktop automation toolset: `tool_computer_desktop_info`,
`tool_computer_screenshot`, `tool_computer_click`, `tool_computer_move_cursor`,
`tool_computer_mouse_down`, `tool_computer_mouse_up`, `tool_computer_drag`,
`tool_computer_type`, `tool_computer_key`, `tool_computer_scroll`,
`tool_computer_wait`, `tool_computer_cursor_position`.

## What to expect — VISION GATE (hard structural gate)

The toolset is mounted **only if the active model has vision capability**
(`client.has_vision_capability()`, or the active binding's `vision_enabled`, or a
vision-capable child binding). With a blind model, the mount is **silently
skipped with a console warning** and the model truthfully reports it has no such
tools. First use auto-installs `pyautogui` / `pillow` via pipmaster — expect a
slower first call.

The capability override `allow_computer_use` (Optional[bool], also read from
`personality.capabilities`) ORs with this flag to request the mount.

## Recovering generated stuff (screenshots)

- Default file: `computer_use_screenshot.png` in the working directory —
  **overwritten on every recapture** (each screenshot shows the CURRENT screen
  state). Pass `file_name` to `tool_computer_screenshot` to keep a specific
  frame.
- Result keys: `screenshot_file`, `image_filename`, `context_files`,
  `image_b64`/`screenshot_b64` (also attached in-conversation so the vision model
  can verify), `screen_width`, `screen_height`, `cursor_x`, `cursor_y`.

## Usage

```python
result = discussion.chat(
    "Open the browser and take a screenshot",
    enable_computer_use=True,   # silently skipped if the model lacks vision
)
```

## Related flags

- `enable_vlm_query` — the fallback when vision is absent.
- `suppress_images` — withholds pixels from the model.