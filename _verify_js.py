
import re
import pathlib
import shutil
import subprocess
import tempfile
import os

html_path = "src/lollms_client/apps/lollms_bot/mesh/gui/static/index.html"
html = pathlib.Path(html_path).read_text(encoding="utf-8")

scripts = re.findall(r"<script\b[^>]*>(.*?)</script\s*>", html, re.DOTALL | re.IGNORECASE)
print("script blocks:", len(scripts))

opens = len(re.findall(r"<script\b", html, re.IGNORECASE))
closes = len(re.findall(r"</script\s*>", html, re.IGNORECASE))
print("script tags: open=%d close=%d" % (opens, closes))
assert opens == closes, "UNBALANCED SCRIPT TAGS"

required = [
    "function renderGraph",
    "function renderAgents",
    "function openLiveSocket",
    "function fullRefresh",
    "async function refreshAgents",
    "function _agentActions",
    "function _capabilitiesInspector",
    "function openMcpEditor",
    "function saveMcpDescriptor",
    "function deleteMcpDescriptor",
    "function _updateMcpTransportFields",
    "function _kvToText",
    "function _textToKv",
]
missing = [m for m in required if m not in html]
if missing:
    print("MISSING:", missing)
else:
    print("all required declarations present")

print("modal-mcp present:", 'id="modal-mcp"' in html)
print("caps-panel delegate:", "closest('.caps-panel')" in html)

node = shutil.which("node")
print("node:", node)
if node:
    for i, js in enumerate(scripts):
        fd, path = tempfile.mkstemp(suffix=".js", text=True)
        os.close(fd)
        pathlib.Path(path).write_text(js, encoding="utf-8")
        r = subprocess.run([node, "--check", path], capture_output=True, text=True)
        print("block %d: rc=%d %s" % (i, r.returncode, ("len=%d" % len(js))))
        if r.returncode != 0:
            print(r.stderr[:3000])
        os.unlink(path)
