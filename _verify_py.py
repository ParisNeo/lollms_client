
import ast
import pathlib

FILES = [
    "src/lollms_client/apps/lollms_bot/mesh/agents.py",
    "src/lollms_client/apps/lollms_bot/mesh/gui/server.py",
    "src/lollms_client/apps/lollms_bot/mesh/mcp_store.py",
]

for path in FILES:
    p = pathlib.Path(path)
    if not p.is_file():
        print("MISSING FILE:", path)
        continue
    src = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
        print("OK parse:", path, "(%d lines)" % len(src.splitlines()))
    except SyntaxError as e:
        print("SYNTAX ERROR:", path, e)
        continue
    # Report functions / classes for a quick structural sanity check
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    print("  classes:", classes)
    print("  funcs:", len(funcs))

# Endpoint registration check in server.py
server_src = pathlib.Path("src/lollms_client/apps/lollms_bot/mesh/gui/server.py").read_text(encoding="utf-8")
endpoints = [
    '"/api/agents/{agent_id}/mcp"',
    '"/api/agents/{agent_id}/mcp/{server_id}"',
    '"/api/agents/{agent_id}/capabilities"',
]
for ep in endpoints:
    print("endpoint %s: %s" % (ep, "FOUND" if ep in server_src else "NOT FOUND"))

imports = [
    "from lollms_client.apps.lollms_bot.mesh.mcp_store import",
]
for imp in imports:
    print("import %s: %s" % (imp, "FOUND" if imp in server_src else "NOT FOUND"))

agents_src = pathlib.Path("src/lollms_client/apps/lollms_bot/mesh/agents.py").read_text(encoding="utf-8")
print("agents _secret-free summary used:", "_describe_handbag_capabilities" in agents_src)
