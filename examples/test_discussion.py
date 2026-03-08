# test_lollms_discussion.py
# Simple smoke test for the lollms_discussion package.
# Run from the folder that contains the lollms_discussion/ package directory.

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import (
    LollmsDiscussion,
    LollmsDataManager,
    LollmsMessage,
    ArtefactType,
    ArtefactManager,
)
# ---------------------------------------------------------------------------
# 1. Client
# ---------------------------------------------------------------------------
lc = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={
        "model_name": "Kimi-K 2.5",
        "service_key": "lollms_0vWhQYNV__q6GMMfZ4_Ib2Crw8wnPTAXDu2KWrsKvsulpptxzfiY",
    },
)
print("✅  LollmsClient created")

# ---------------------------------------------------------------------------
# 2. In-memory discussion (no DB required)
# ---------------------------------------------------------------------------
disc = LollmsDiscussion.create_new(lollms_client=lc)
print(f"✅  Discussion created  id={disc.id}")

# ---------------------------------------------------------------------------
# 3. Basic chat
# ---------------------------------------------------------------------------
result = disc.chat("Hello! What is 2 + 2?")
ai_msg: LollmsMessage = result["ai_message"]
print(f"✅  chat() returned     ai_message.content = {ai_msg.content!r}")

# ---------------------------------------------------------------------------
# 4. Artefact CRUD
# ---------------------------------------------------------------------------
disc.artefacts.add(
    title="hello.py",
    artefact_type=ArtefactType.CODE,
    content='print("Hello, World!")',
    language="python",
    active=True,
)
artefact = disc.artefacts.get("hello.py")
assert artefact is not None, "Artefact not found after add()"
assert artefact["content"] == 'print("Hello, World!")'
print(f"✅  Artefact created    title={artefact['title']}  type={artefact['type']}")

# Update it (bumps to v2)
disc.artefacts.update("hello.py", new_content='print("Hello, lollms!")')
v2 = disc.artefacts.get("hello.py")
assert v2["version"] == 2
assert v2["content"] == 'print("Hello, lollms!")'
print(f"✅  Artefact updated    version={v2['version']}")

# Context zone injection
zone = disc.artefacts.build_artefacts_context_zone()
assert "hello.py" in zone
print(f"✅  Context zone built  (len={len(zone)} chars)")

# ---------------------------------------------------------------------------
# 5. Chat with artefact instructions injected
# ---------------------------------------------------------------------------
result2 = disc.chat(
    "Write a Python function that adds two numbers and put it in an artefact.",
    auto_activate_artefacts=True,
)
print(f"✅  chat() with artefacts  artefacts affected: {[a['title'] for a in result2['artefacts']]}")

# ---------------------------------------------------------------------------
# 6. DB-backed discussion
# ---------------------------------------------------------------------------
db = LollmsDataManager("sqlite:///test_discussion.db")
disc_db = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db, autosave=True)
disc_db.chat("Briefly introduce yourself.")
disc_db.close()
print(f"✅  DB-backed discussion  id={disc_db.id}  saved to test_discussion.db")

# Reload and verify messages persisted
disc_reloaded = db.get_discussion(lc, disc_db.id)
messages = disc_reloaded.get_branch(disc_reloaded.active_branch_id)
assert len(messages) >= 2, "Expected at least user + assistant message after reload"
print(f"✅  Reloaded discussion   messages in branch: {len(messages)}")
disc_reloaded.close()

# ---------------------------------------------------------------------------
print("\n🎉  All tests passed.")