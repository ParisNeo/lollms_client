"""
test_computer_use.py
Validates the LoLLMS computer_use toolset in two phases.

  1. OFFLINE (no LLM): coordinate layer, captures, grounding parsers.
     WARNING: the mouse moves by itself during this phase.
  2. LIVE (real agent): a Tk board with 3 labeled buttons is opened; the
     agent is asked to click each one; every click is hit-tested.
     Needs a vision-capable model. Windows/Linux (Tk runs in a thread).

Usage:
  python test_computer_use.py                 # both phases
  python test_computer_use.py --offline-only
  python test_computer_use.py --live-only --trials 3

Env:
  COMPUTER_USE_PATH  override the path of computer_use.py
"""
import argparse
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

from ascii_colors import ASCIIColors

RESULTS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append(bool(cond))
    line = f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
    if cond:
        print(line)
    else:
        ASCIIColors.error(line)


# ───────────────────────────── Phase 1: offline ─────────────────────────────

def load_toolset():
    override = os.environ.get("COMPUTER_USE_PATH")
    if override:
        path = Path(override)
    else:
        import lollms_client
        path = (
            Path(lollms_client.__file__).parent.parent.parent
            / "src" / "lollms_client"/ "tools_bindings" / "lcp" / "default_tools"
            / "computer_use" / "computer_use.py"
        )
    if not path.is_file():
        raise FileNotFoundError(f"computer_use.py not found at {path} (set COMPUTER_USE_PATH)")
    spec = importlib.util.spec_from_file_location("computer_use_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offline_tests() -> None:
    print("\n=== Phase 1: offline (the mouse will move) ===")
    time.sleep(2)
    cu = load_toolset()

    backend, err = cu._resolve_backend()
    check("backend resolves", backend is not None, err or "")
    if backend is None:
        return

    mw, mh = backend.size()
    sw, sh = backend._raw.size()
    print(f"physical screen {sw}x{sh} -> model space {mw}x{mh}")
    check("model long edge <= limit", max(mw, mh) <= cu._MODEL_MAX_EDGE)
    check("aspect ratio preserved", abs(mw / mh - sw / sh) < 0.01)
    check("screenshot size == model space", backend.screenshot().size == (mw, mh),
          str(backend.screenshot().size))
    check("zone screenshot size", backend.screenshot(region=(100, 100, 300, 200)).size == (300, 200))

    _, aw, ah = cu._capture_annotated_b64(None, show_grid=True)
    check("annotated capture size == model space", (aw, ah) == (mw, mh), f"{aw}x{ah}")

    targets = [(mw // 2, mh // 2), (mw // 10, mh // 10), (mw * 9 // 10, mh * 9 // 10),
               (mw - 2, mh - 2), (5, 5)]
    for tx, ty in targets:
        res = cu.tool_computer_move_cursor(x=tx, y=ty)
        if not res.get("success"):
            check(f"move ({tx},{ty})", False, str(res.get("error")))
            continue
        px, py = backend.position()
        rx, ry = backend._raw.position()
        ex, ey = round(tx * sw / mw), round(ty * sh / mh)
        check(f"move ({tx},{ty}) model readback", max(abs(px - tx), abs(py - ty)) <= 2,
              f"got ({px},{py})")
        check(f"move ({tx},{ty}) physical ({ex},{ey})", max(abs(rx - ex), abs(ry - ey)) <= 2,
              f"got ({rx},{ry})")

    # Grounding parsers
    det = "<|ref|>x<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>"
    got = cu._extract_coordinates_with_scale(det, mw, mh)
    exp = (round(200 / 999 * mw), round(300 / 999 * mh))
    check("parser: DeepSeek <|det|> box center (0-999)",
          got is not None and max(abs(got[0] - exp[0]), abs(got[1] - exp[1])) <= 1,
          f"got {got}, expected {exp}")

    got = cu._extract_coordinates_with_scale('{"x": 640, "y": 360, "scale": "absolute"}', mw, mh)
    check("parser: JSON absolute", got == (640, 360), str(got))

    got = cu._extract_coordinates_with_scale('{"x": 500, "y": 500, "scale": "normalized"}', mw, mh)
    exp = (round(0.5 * mw), round(0.5 * mh))
    check("parser: JSON normalized 0-1000",
          got is not None and max(abs(got[0] - exp[0]), abs(got[1] - exp[1])) <= 1,
          f"got {got}, expected {exp}")

    got = cu._extract_coordinates_with_scale("click at (120, 340) please", mw, mh)
    check("parser: bare (x, y)", got == (120, 340), str(got))


# ───────────────────────────── Phase 2: live ────────────────────────────────

class TargetBoard:
    """Tk window with 3 labeled buttons; records where the agent clicks."""

    COLORS = {"ALPHA": "#d93025", "BRAVO": "#1e8e3e", "CHARLIE": "#1a73e8"}
    COLOR_NAMES = {"ALPHA": "red", "BRAVO": "green", "CHARLIE": "blue"}
    SPOTS = {"ALPHA": (0.10, 0.18), "BRAVO": (0.50, 0.52), "CHARLIE": (0.88, 0.85)}

    def __init__(self):
        self.clicks = []
        self.rects = {}
        self.sw = self.sh = 0
        self.root = None
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        if not self._ready.wait(10):
            raise RuntimeError("Tk board failed to start")

    def _run(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        self.root = root
        self.sw, self.sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = int(self.sw * 0.9), int(self.sh * 0.8)
        root.geometry(f"{w}x{h}+{(self.sw - w) // 2}+{(self.sh - h) // 2}")
        root.title("computer_use test board")
        root.attributes("-topmost", True)
        canvas = tk.Canvas(root, width=w, height=h, bg="white", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        bw, bh = 200, 90
        for label, (fx, fy) in self.SPOTS.items():
            cx, cy = int(w * fx), int(h * fy)
            rect = (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)
            self.rects[label] = rect
            canvas.create_rectangle(*rect, fill=self.COLORS[label], outline="black", width=3)
            canvas.create_text(cx, cy, text=label, fill="white", font=("Arial", 22, "bold"))
        canvas.bind("<Button-1>", self._on_click)
        self._ready.set()
        root.mainloop()

    def _on_click(self, ev) -> None:
        hit = next(
            (name for name, (x1, y1, x2, y2) in self.rects.items()
             if x1 <= ev.x <= x2 and y1 <= ev.y <= y2),
            None,
        )
        self.clicks.append({"hit": hit, "x": ev.x, "y": ev.y})

    def center(self, label: str):
        x1, y1, x2, y2 = self.rects[label]
        return (x1 + x2) // 2, (y1 + y2) // 2


def live_tests(trials: int) -> None:
    print("\n=== Phase 2: live agent ===")
    import pyautogui
    from lollms_client import MSG_TYPE
    from lollms_client.lollms_config_cli_env import get_client_from_env
    from lollms_client.lollms_discussion import LollmsDiscussion

    lc = get_client_from_env()
    workspace = Path(__file__).parent / "workspaces"
    workspace.mkdir(parents=True, exist_ok=True)

    def cb(chunk, msg_type, metadata):
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(chunk, end="", flush=True)
        return True

    board = TargetBoard()
    phys = tuple(pyautogui.size())
    if (board.sw, board.sh) != phys:
        ASCIIColors.warning(
            f"Tk sees {board.sw}x{board.sh} but pyautogui sees {phys[0]}x{phys[1]} "
            "(DPI mismatch): board hit-tests may be unreliable."
        )

    for label in board.rects:
        color = board.COLOR_NAMES[label]
        for trial in range(1, trials + 1):
            board.clicks.clear()
            ld = LollmsDiscussion(lc, workspace_path=workspace)
            prompt = (
                f"A window titled 'computer_use test board' is open on screen. "
                f"Click the {color} button labeled {label} exactly once. "
                f"Do not click anything else, then stop."
            )
            print(f"\n--- {label} (trial {trial}/{trials}) ---")
            try:
                ld.chat(prompt, streaming_callback=cb, enable_computer_use=True)
            except Exception as exc:
                check(f"live {label} #{trial}", False, f"chat error: {exc}")
                continue
            time.sleep(0.5)
            if not board.clicks:
                check(f"live {label} #{trial}", False, "no click registered")
                continue
            first = board.clicks[0]
            cx, cy = board.center(label)
            dist = ((first["x"] - cx) ** 2 + (first["y"] - cy) ** 2) ** 0.5
            extra = len(board.clicks) - 1
            check(
                f"live {label} #{trial}",
                first["hit"] == label,
                f"hit={first['hit']}, off-center={dist:.0f}px, extra clicks={extra}",
            )

    try:
        board.root.after(0, board.root.destroy)
    except Exception:
        pass


# ───────────────────────────────── main ─────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="computer_use toolset test")
    ap.add_argument("--offline-only", action="store_true")
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--trials", type=int, default=1, help="live trials per target")
    args = ap.parse_args()

    if not args.live_only:
        offline_tests()
    if not args.offline_only:
        live_tests(max(1, args.trials))

    passed, total = sum(RESULTS), len(RESULTS)
    print(f"\n=== {passed}/{total} checks passed ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()