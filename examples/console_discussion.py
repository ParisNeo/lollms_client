# research_app_final.py

import os
import json
import shutil
from pathlib import Path
# Use the correct, specified import style
from lollms_client import LollmsClient, LollmsDiscussion, MSG_TYPE, DatabaseManager
from ascii_colors import ASCIIColors
from sqlalchemy import Column, String

# --- 1. Define Application-Specific Schema ---
# We define our custom fields for the database tables.
# This allows applications to store and query their own metadata.
class ResearchDiscussionMixin:
    # We want each discussion to have a 'project_name' that we can search for.
    project_name = Column(String(100), index=True, nullable=False)

class ResearchMessageMixin:
    # This mixin is empty for this example.
    pass

def main():
    # --- 2. Setup: Lollms Client is always needed ---
    print("--- LOLLMS Research Assistant (Final Version) ---")
    try:
        # Instantiate the real LollmsClient to connect to a running model service.
        # Ensure Ollama is running and has pulled the specified model.
        lc = LollmsClient("ollama", model_name="qwen3:4b")
        print("LollmsClient connected successfully to Ollama.")
    except Exception as e:
        print(f"\nFATAL: Could not connect to LLM binding. Is Ollama running?\nError: {e}")
        return

    # --- 3. Setup Database Manager ---
    DB_PATH = "sqlite:///research_projects_final.db"
    ENCRYPTION_KEY = "a-secure-password-for-the-database"
    
    try:
        db_manager = DatabaseManager(
            db_path=DB_PATH,
            discussion_mixin=ResearchDiscussionMixin,
            message_mixin=ResearchMessageMixin,
            encryption_key=ENCRYPTION_KEY
        )
        print(f"Database setup complete. Encryption is ENABLED.")
    except Exception as e:
        print(f"\nFATAL: Could not initialize database. Error: {e}")
        return

    # --- 4. Main Application Loop ---
    # This loop demonstrates the new management features.
    discussion = None
    while True:
        print("\n--- Main Menu ---")
        if discussion:
            print(f"Current Project: '{discussion.metadata.get('project_name', discussion.id)}'")
            print("c. Chat in current project")
            print("r. Regenerate last AI response (create new branch)")
        print("l. List all projects")
        print("s. Search for a project")
        print("n. Start a new project")
        print("o. Open an existing project")
        print("d. Delete a project")
        print("e. Exit")
        
        choice = input("> ").lower()

        if choice == 'c' and discussion:
            chat_loop(discussion)
        elif choice == 'r' and discussion:
            regenerate_response(discussion)
        elif choice == 'l':
            list_all_projects(db_manager)
        elif choice == 's':
            search_for_project(db_manager)
        elif choice == 'n':
            discussion = start_new_project(lc, db_manager)
        elif choice == 'o':
            discussion = open_project(lc, db_manager)
        elif choice == 'd':
            delete_project(db_manager)
            if discussion and discussion.id not in [d['id'] for d in db_manager.list_discussions()]:
                discussion = None # Clear current discussion if it was deleted
        elif choice == 'e':
            break
        else:
            ASCIIColors.red("Invalid choice.")

    # --- Cleanup ---
    print("\n--- Demo complete. Cleaning up. ---")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def list_all_projects(db_manager: DatabaseManager):
    projects = db_manager.list_discussions()
    if not projects:
        ASCIIColors.yellow("No projects found.")
        return
    print("\n--- All Projects ---")
    for p in projects:
        print(f"- ID: {p['id']} | Name: {p.get('project_name', 'N/A')}")

def search_for_project(db_manager: DatabaseManager):
    term = input("Enter search term for project name: ")
    projects = db_manager.search_discussions(project_name=term)
    if not projects:
        ASCIIColors.yellow(f"No projects found matching '{term}'.")
        return
    print(f"\n--- Search Results for '{term}' ---")
    for p in projects:
        print(f"- ID: {p['id']} | Name: {p.get('project_name', 'N/A')}")

def start_new_project(lc: LollmsClient, db_manager: DatabaseManager) -> LollmsDiscussion:
    name = input("Enter new project name: ")
    if not name:
        ASCIIColors.red("Project name cannot be empty.")
        return None
    discussion = LollmsDiscussion.create_new(
        lollms_client=lc,
        db_manager=db_manager,
        autosave=True, # Recommended for interactive apps
        project_name=name
    )
    discussion.system_prompt = f"This is a research project about {name}."
    ASCIIColors.green(f"Project '{name}' created successfully.")
    return discussion

def open_project(lc: LollmsClient, db_manager: DatabaseManager) -> LollmsDiscussion:
    list_all_projects(db_manager)
    disc_id = input("Enter project ID to open: ")
    discussion = db_manager.get_discussion(lollms_client=lc, discussion_id=disc_id, autosave=True)
    if not discussion:
        ASCIIColors.red("Project not found.")
        return None
    ASCIIColors.green(f"Opened project '{discussion.metadata.get('project_name', discussion.id)}'.")
    return discussion

def delete_project(db_manager: DatabaseManager):
    list_all_projects(db_manager)
    disc_id = input("Enter project ID to DELETE: ")
    confirm = input(f"Are you sure you want to permanently delete project {disc_id}? (y/N): ")
    if confirm.lower() == 'y':
        db_manager.delete_discussion(disc_id)
        ASCIIColors.green("Project deleted.")
    else:
        ASCIIColors.yellow("Deletion cancelled.")

def chat_loop(discussion: LollmsDiscussion):
    """The interactive chat session for a given discussion."""
    print("\n--- Entering Chat ---")
    
    # Display the current branch history when entering the chat.
    current_branch = discussion.get_branch(discussion.active_branch_id)
    if current_branch:
        ASCIIColors.cyan("--- Current Conversation History ---")
        for msg in current_branch:
            sender = msg['sender']
            if sender == 'user':
                ASCIIColors.green(f"You: {msg['content']}")
            else:
                ASCIIColors.blue(f"AI: {msg['content']}")
        ASCIIColors.cyan("----------------------------------")

    print("Type your message, or /back, /toggle_thoughts")
    show_thoughts_flag = False
    
    def stream_to_console(token: str, msg_type: MSG_TYPE):
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(token, end="", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            ASCIIColors.magenta(token, end="", flush=True)
        return True

    while True:
        user_input = input("\nYou > ")
        if user_input.lower() == '/back': break
        
        if user_input.lower() == '/toggle_thoughts':
            show_thoughts_flag = not show_thoughts_flag
            ASCIIColors.yellow(f"\n[{'ON' if show_thoughts_flag else 'OFF'}] Thoughts are now displayed.")
            continue

        print("AI > ", end="", flush=True)
        discussion.chat(
            user_input, 
            show_thoughts=show_thoughts_flag,
            streaming_callback=stream_to_console
        )
        print()

def regenerate_response(discussion: LollmsDiscussion):
    """Demonstrates creating a new branch by regenerating."""
    try:
        ASCIIColors.yellow("\nRegenerating last AI response...")
        print("New AI > ", end="", flush=True)
        def stream_to_console(token: str, msg_type: MSG_TYPE):
            print(token, end="", flush=True)
            return True
        discussion.regenerate_branch(show_thoughts=True, streaming_callback=stream_to_console)
        print()
        ASCIIColors.green("New branch created.")
    except ValueError as e:
        ASCIIColors.red(f"Could not regenerate: {e}")

if __name__ == "__main__":
    main()