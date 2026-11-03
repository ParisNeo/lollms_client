"""
scripts/verify_sandbox.py

Smoke-runner: asserts each exploit vector is blocked end-to-end via the public
tool entrypoints. Use it in CI before touching the sandbox kernel.

    python scripts/verify_sandbox.py

Exit codes: 0 = all sandboxed, 1 = at least one escape vector succeeded.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Prefer the installed `lollms_client` package; fall back to the repo's `src/`
# folder for dev checkouts where the package isn't pip-installed. Never use
# `from src.lollms_client...` — `src` is a valid but hijackable top-level name.
try:
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )
except ImportError:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )


EXPLOITS = [
    ("subprocess import", "import subprocess\nsubprocess.run(['whoami'])"),
    ("os.system call", "import os\nos.system('whoami')"),
    ("socket import", "import socket\nsocket.socket()"),
    ("ctypes import", "import ctypes\nctypes.CDLL(None)"),
    ("eval escape", "eval('1+1')"),
    ("exec escape", "exec('print(1)')"),
    ("dunder import", "__import__('os').system('whoami')"),
    ("importlib import", "import importlib\nimportlib.import_module('subprocess')"),
    ("path traversal", "open('../../../etc/passwd')"),
    ("requests", "import requests\nrequests.get('http://example.com')"),
]


def _check_data_query(code: str) -> bool:
    result = epdq.tool_execute_python_data_query(code=code, env=None)
    return result.get("success") is False


def _check_execute_python(code: str) -> bool:
    result = ep.tool_execute_python_code(code=code)
    return result.get("success") is False


def main() -> int:
    failures = []
    print("=" * 70)
    print("SANDBOX VERIFICATION")
    print("=" * 70)

    for label, code in EXPLOITS:
        dq_ok = _check_data_query(code)
        ep_ok = _check_execute_python(code)
        status = "✓" if (dq_ok and ep_ok) else "✗"
        print(f"[{status}] {label:<24} data_query={'blocked' if dq_ok else 'OPEN'} "
              f"execute_python={'blocked' if ep_ok else 'OPEN'}")
        if not (dq_ok and ep_ok):
            failures.append(label)

    print("=" * 70)
    if failures:
        print(f"❌ FAIL — {len(failures)} escape vector(s) remain open: {failures}")
        return 1
    print("✅ PASS — all escape vectors blocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())