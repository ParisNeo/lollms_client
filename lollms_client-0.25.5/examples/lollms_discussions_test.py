# research_app_final.py

import os
import json
import shutil
from pathlib import Path
# Use the correct, specified import style
from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDataManager, LollmsDiscussion
from lollms_client.lollms_types import MSG_TYPE
from sqlalchemy import Column, String

# --- 1. Define Application-Specific Schema ---
# The developer can define their own fields for the database tables.
# This allows applications to store and query their own metadata.
class ResearchDiscussionMixin:
    # We want each discussion to have a 'project_name' that we can search for.
    project_name = Column(String(100), index=True, nullable=False)

class ResearchMessageMixin:
    # This mixin is empty for this example.
    pass

def setup_migration_dummies(folder: Path):
    """Creates a dummy JSON file to simulate an old, file-based project structure."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    # This data structure mimics what the old `to_dict` would have produced.
    discussion_data = {
        "id": "old_project_alpha",
        "metadata": {"project_name": "Project Alpha"},
        "system_prompt": "This is the system prompt for Alpha.",
        "created_at": "2023-01-01T12:00:00",
        "updated_at": "2023-01-01T12:05:00",
        "messages": [
            {"id": "msg1", "sender": "user", "sender_type":"user", "content": "What was the first finding?", "created_at": "2023-01-01T12:00:00"},
            {"id": "msg2", "sender": "assistant", "sender_type":"assistant", "content": "It was about quantum states.", "parent_id": "msg1", "created_at": "2023-01-01T12:05:00"}
        ]
    }
    with open(folder / "project_alpha.json", "w") as f:
        json.dump(discussion_data, f, indent=2)
    print(f"Created dummy migration file in '{folder}'.")

def main():
    # --- 2. Setup: Lollms Client is always needed ---
    print("--- LOLLMS Research Assistant (Final Version) ---")
    try:
        # Instantiate the real LollmsClient to connect to a running model service.
        # Ensure Ollama is running and has pulled the specified model.
        lc = LollmsClient("ollama", model_name="mistral-nemo:latest")
        print("LollmsClient connected successfully to Ollama.")
    except Exception as e:
        print(f"\nFATAL: Could not connect to LLM binding. Is Ollama running?\nError: {e}")
        return

    # --- DEMO 1: In-Memory Mode (Backward Compatibility) ---
    print("\n--- DEMO 1: In-Memory Discussion ---")
    
    # Create an in-memory discussion by NOT passing a db_manager.
    in_memory_discussion = LollmsDiscussion.create_new(lollms_client=lc)
    in_memory_discussion.system_prompt = "You are a helpful assistant."
    print("Created an in-memory discussion.")

    # Interact with it. The state is held entirely in the object.
    user_input_mem = "Can you remember that my favorite color is blue?"
    print(f"You > {user_input_mem}")
    print("AI  > ", end="", flush=True)
    def stream_to_console(token, msg_type=MSG_TYPE.MSG_TYPE_CHUNK):
        print(token, end="", flush=True)
        return True
    in_memory_discussion.chat(user_input_mem, streaming_callback=stream_to_console)
    print()

    # Save its state to a JSON file. This now works correctly.
    file_path = Path("./in_memory_save.json")
    with open(file_path, "w") as f:
        json.dump(in_memory_discussion.to_dict(), f, indent=2)
    print(f"\nIn-memory discussion saved to '{file_path}'.")
    os.remove(file_path)

    # --- DEMO 2: Database-Backed Mode with Migration ---
    print("\n--- DEMO 2: Database-Backed Mode ---")
    DB_PATH = "sqlite:///research_projects_final.db"
    ENCRYPTION_KEY = "a-secure-password-for-the-database"
    MIGRATION_FOLDER = Path("./old_discussions")
    
    try:
        # Initialize the LollmsDataManager with our schema and encryption key.
        db_manager = LollmsDataManager(
            db_path=DB_PATH,
            discussion_mixin=ResearchDiscussionMixin,
            message_mixin=ResearchMessageMixin,
            encryption_key=ENCRYPTION_KEY
        )
        print(f"Database setup complete. Encryption is ENABLED.")
    except Exception as e:
        print(f"\nFATAL: Could not initialize database. Error: {e}")
        return

    # Demonstrate the one-time migration from a folder of JSON files.
    setup_migration_dummies(MIGRATION_FOLDER)
    input("\nDummy migration files created. Press Enter to run the migration...")
    LollmsDiscussion.migrate(lollms_client=lc, db_manager=db_manager, folder_path=MIGRATION_FOLDER)
    
    session = db_manager.get_session()
    migrated_count = session.query(db_manager.DiscussionModel).count()
    print(f"Verification: Found {migrated_count} discussions in the database after migration.")
    session.close()

    # --- DEMO 3: Live Chat with a DB-Backed Discussion ---
    input("\nMigration complete. Press Enter to start a new, database-backed chat session...")
    
    # Create a new, database-backed discussion with our custom 'project_name'.
    discussion = LollmsDiscussion.create_new(
        lollms_client=lc,
        db_manager=db_manager,
        max_context_size=lc.default_ctx_size // 2,
        autosave=True,
        project_name="Project Gamma (Live)"
    )
    discussion.system_prompt = "You are a helpful assistant for Project Gamma."
    
    print(f"\n--- Live Chat for '{discussion.db_discussion.project_name}' ---")
    print("Type your message, or '/exit', '/export_openai', '/export_ollama' to quit.")
    
    while True:
        user_input = input("\nYou > ")
        if user_input.lower() == '/exit': break
        
        if user_input.lower().startswith('/export'):
            try:
                format_type = user_input.split('_')[1] + "_chat"
                exported_data = discussion.export(format_type)
                print(f"\n--- Exported for {format_type.split('_')[0].upper()} ---")
                print(json.dumps(exported_data, indent=2))
                print("-----------------------------------")
            except IndexError:
                print("Invalid export command. Use /export_openai or /export_ollama")
            continue

        print("AI > ", end="", flush=True)
        # The same streaming callback works seamlessly.
        discussion.chat(user_input, streaming_callback=stream_to_console)
        print()

    # --- Cleanup ---
    print("\n--- Demo complete. Cleaning up. ---")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if MIGRATION_FOLDER.exists():
        shutil.rmtree(MIGRATION_FOLDER)

if __name__ == "__main__":
    main()