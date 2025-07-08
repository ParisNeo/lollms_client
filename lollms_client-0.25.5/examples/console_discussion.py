import os
import re
import yaml
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

# --- Mock RAG Backend (for demonstration purposes) ---
# In a real app, this would be a proper vector database (ChromaDB, FAISS, etc.)
MOCK_VECTOR_DB_PATH = Path("./rag_db")
MOCK_VECTOR_DB_PATH.mkdir(exist_ok=True)

def mock_vectorize_chunk(chunk_text: str, chunk_id: str):
    # Simulate vectorization by just saving the chunk text to a file.
    # A real implementation would convert chunk_text to a vector and store it.
    (MOCK_VECTOR_DB_PATH / f"{chunk_id}.json").write_text(json.dumps({
        "id": chunk_id,
        "text": chunk_text
    }, indent=2))

def mock_is_vectorized(chunk_id: str) -> bool:
    return (MOCK_VECTOR_DB_PATH / f"{chunk_id}.json").exists()

def mock_query_rag(user_query: str) -> str:
    # Simulate RAG by doing a simple keyword search across all chunk files.
    # A real implementation would do a vector similarity search.
    relevant_chunks = []
    query_words = set(user_query.lower().split())
    if not query_words:
        return ""
        
    for file in MOCK_VECTOR_DB_PATH.glob("*.json"):
        data = json.loads(file.read_text(encoding='utf-8'))
        if any(word in data["text"].lower() for word in query_words):
            relevant_chunks.append(data["text"])
    
    if not relevant_chunks:
        return ""

    return "\n---\n".join(relevant_chunks)

# --- Library Imports ---
# Assumes lollms_client.py, lollms_discussion.py, and lollms_personality.py are in the same directory or accessible in PYTHONPATH
from lollms_client import LollmsClient, MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_personality import LollmsPersonality
from ascii_colors import ASCIIColors
from sqlalchemy import Column, String
from sqlalchemy.exc import IntegrityError 

# --- Application-Specific Schema ---
class ResearchDiscussionMixin:
    project_name = Column(String(100), index=True, nullable=False, unique=True)

class ResearchMessageMixin:
    pass # No custom fields needed for this demo

# --- Personality Management ---
def load_personalities(personalities_path: Path) -> Dict[str, LollmsPersonality]:
    """Loads all personalities from a directory of YAML files."""
    personalities = {}
    if not personalities_path.is_dir():
        return {}
        
    for file_path in personalities_path.glob("*.yaml"):
        try:
            config = yaml.safe_load(file_path.read_text(encoding='utf-8'))
            
            script_content = None
            script_path = file_path.with_suffix(".py")
            if script_path.exists():
                script_content = script_path.read_text(encoding='utf-8')
                
            # Make data file paths relative to the personalities folder
            data_files = [personalities_path / f for f in config.get("data_files", [])]
                
            personality = LollmsPersonality(
                name=config.get("name", file_path.stem),
                author=config.get("author", "Unknown"),
                category=config.get("category", "General"),
                description=config.get("description", ""),
                system_prompt=config.get("system_prompt", "You are a helpful AI."),
                data_files=data_files,
                script=script_content,
                vectorize_chunk_callback=mock_vectorize_chunk,
                is_vectorized_callback=mock_is_vectorized,
                query_rag_callback=mock_query_rag
            )
            personalities[personality.personality_id] = personality
        except Exception as e:
            ASCIIColors.red(f"Failed to load personality from {file_path.name}: {e}")
            
    return personalities

def select_personality(personalities: Dict[str, LollmsPersonality]) -> Optional[LollmsPersonality]:
    """UI for selecting a personality."""
    if not personalities:
        ASCIIColors.yellow("No personalities found.")
        return None
        
    print("\n--- Select a Personality ---")
    sorted_p = sorted(personalities.values(), key=lambda p: (p.category, p.name))
    for i, p in enumerate(sorted_p):
        print(f"{i+1}. {p.category}/{p.name} (by {p.author})")
    print("0. Deselect Personality")
        
    while True:
        try:
            choice_str = input("> ")
            if not choice_str: return None
            choice = int(choice_str)
            if choice == 0:
                return None
            if 1 <= choice <= len(sorted_p):
                return sorted_p[choice - 1]
            else:
                ASCIIColors.red("Invalid number.")
        except ValueError:
            ASCIIColors.red("Please enter a number.")

# --- Main Application Logic ---
def main():
    print("--- LOLLMS Advanced Agentic Framework ---")
    try:
        lc = LollmsClient("ollama", model_name="qwen3:4b")
        print("LollmsClient connected successfully to Ollama.")
    except Exception as e:
        print(f"\nFATAL: Could not connect to LLM binding. Is the service running?\nError: {e}")
        return

    DB_PATH = "sqlite:///research_projects_final.db"
    ENCRYPTION_KEY = "a-very-secure-password-for-the-database"
    
    try:
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

    personalities_path = Path("./personalities")
    personalities = load_personalities(personalities_path)
    print(f"Loaded {len(personalities)} personalities.")

    discussion: Optional[LollmsDiscussion] = None
    personality: Optional[LollmsPersonality] = None

    while True:
        print("\n" + "="*20 + " Main Menu " + "="*20)
        if discussion:
            p_name = f" with '{personality.name}'" if personality else ""
            ASCIIColors.cyan(f"Current Project: '{discussion.project_name}'{p_name}")
            print("c. Chat in current project")
            print("r. Regenerate last AI response")
        print("l. List all projects")
        print("s. Search for a project")
        print("n. Start a new project")
        print("o. Open an existing project")
        print("d. Delete a project")
        print("p. Select a Personality")
        print("e. Exit")
        
        choice = input("> ").lower().strip()

        if choice == 'c' and discussion:
            chat_loop(discussion, personality)
        elif choice == 'r' and discussion:
            regenerate_response(discussion, personality)
        elif choice == 'l':
            list_all_projects(db_manager)
        elif choice == 's':
            search_for_project(db_manager)
        elif choice == 'n':
            new_discussion = start_new_project(lc, db_manager)
            if new_discussion: discussion = new_discussion
        elif choice == 'o':
            new_discussion = open_project(lc, db_manager)
            if new_discussion: discussion = new_discussion
        elif choice == 'd':
            delete_project(db_manager)
            if discussion and not db_manager.get_discussion(lc, discussion.id):
                discussion = None
        elif choice == 'p':
            personality = select_personality(personalities)
            if personality:
                ASCIIColors.green(f"Personality '{personality.name}' selected.")
            else:
                ASCIIColors.yellow("No personality selected.")
        elif choice == 'e':
            if discussion: discussion.close()
            break
        else:
            ASCIIColors.red("Invalid choice.")

    print("\n--- Demo complete. Database and RAG files are preserved. ---")

# --- UI Functions ---
def list_all_projects(db_manager: LollmsDataManager):
    projects = db_manager.list_discussions()
    if not projects:
        ASCIIColors.yellow("No projects found.")
        return
    print("\n--- All Projects ---")
    for p in projects:
        print(f"- Name: {p['project_name']:<30} | ID: {p['id']}")

def search_for_project(db_manager: LollmsDataManager):
    term = input("Enter search term for project name: ").strip()
    if not term: return
    projects = db_manager.search_discussions(project_name=term)
    if not projects:
        ASCIIColors.yellow(f"No projects found matching '{term}'.")
        return
    print(f"\n--- Search Results for '{term}' ---")
    for p in projects:
        print(f"- Name: {p['project_name']:<30} | ID: {p['id']}")

def start_new_project(lc: LollmsClient, db_manager: LollmsDataManager) -> Optional[LollmsDiscussion]:
    name = input("Enter new project name: ").strip()
    if not name:
        ASCIIColors.red("Project name cannot be empty.")
        return None
    try:
        discussion = LollmsDiscussion.create_new(
            lollms_client=lc, db_manager=db_manager,
            autosave=True, project_name=name
        )
        ASCIIColors.green(f"Project '{name}' created successfully.")
        return discussion
    except IntegrityError:
        ASCIIColors.red(f"Failed to create project. A project named '{name}' already exists.")
        return None
    except Exception as e:
        ASCIIColors.red(f"An unexpected error occurred while creating the project: {e}")
        return None

def open_project(lc: LollmsClient, db_manager: LollmsDataManager) -> Optional[LollmsDiscussion]:
    list_all_projects(db_manager)
    disc_id = input("Enter project ID to open: ").strip()
    if not disc_id: return None
    discussion = db_manager.get_discussion(lollms_client=lc, discussion_id=disc_id, autosave=True)
    if not discussion:
        ASCIIColors.red("Project not found."); return None
    ASCIIColors.green(f"Opened project '{discussion.project_name}'.")
    return discussion

def delete_project(db_manager: LollmsDataManager):
    list_all_projects(db_manager)
    disc_id = input("Enter project ID to DELETE: ").strip()
    if not disc_id: return
    confirm = input(f"Are you sure you want to permanently delete project {disc_id}? (y/N): ").lower()
    if confirm == 'y':
        db_manager.delete_discussion(disc_id)
        ASCIIColors.green("Project deleted.")
    else:
        ASCIIColors.yellow("Deletion cancelled.")

def display_branch_history(discussion: LollmsDiscussion):
    current_branch = discussion.get_branch(discussion.active_branch_id)
    if not current_branch: return
    ASCIIColors.cyan("\n--- Current Conversation History (Active Branch) ---")
    for msg in current_branch:
        if msg.sender_type == 'user':
            ASCIIColors.green(f"\nYou: {msg.content}")
        else:
            ASCIIColors.blue(f"\nAI: {msg.content}")
            speed_str = f"{msg.generation_speed:.1f} t/s" if msg.generation_speed is not None else "N/A"
            ASCIIColors.dim(f"    [Model: {msg.model_name}, Tokens: {msg.tokens}, Speed: {speed_str}]")
            if msg.thoughts:
                ASCIIColors.dark_gray(f"    [Thoughts: {msg.thoughts[:100]}...]")
            if msg.scratchpad:
                ASCIIColors.yellow(f"    [Scratchpad: {msg.scratchpad[:100]}...]")
    ASCIIColors.cyan("-----------------------------------------------------")

def display_message_tree(discussion: LollmsDiscussion):
    print("\n--- Project Message Tree ---")
    messages_by_id = {msg.id: msg for msg in discussion.messages}
    children_map = defaultdict(list)
    root_ids = []
    for msg in messages_by_id.values():
        if msg.parent_id and msg.parent_id in messages_by_id:
            children_map[msg.parent_id].append(msg.id)
        else:
            root_ids.append(msg.id)
    def print_node(msg_id, indent=""):
        msg = messages_by_id.get(msg_id)
        if not msg: return
        is_active = " (*)" if msg.id == discussion.active_branch_id else ""
        color = ASCIIColors.green if msg.sender_type == "user" else ASCIIColors.blue
        content_preview = re.sub(r'\s+', ' ', msg.content).strip()[:50] + "..."
        color(f"{indent}├─ {msg.id[-8:]}{is_active} ({msg.sender}): {content_preview}")
        for child_id in children_map.get(msg_id, []):
            print_node(child_id, indent + "   ")
    for root_id in root_ids:
        print_node(root_id)
    print("----------------------------")

def handle_config_command(discussion: LollmsDiscussion):
    while True:
        ASCIIColors.cyan("\n--- Thought Configuration ---")
        ASCIIColors.yellow(f"1. Show Thoughts during generation : {'ON' if discussion.show_thoughts else 'OFF'}")
        ASCIIColors.yellow(f"2. Include Thoughts in AI context  : {'ON' if discussion.include_thoughts_in_context else 'OFF'}")
        ASCIIColors.yellow(f"3. Thought Placeholder text      : '{discussion.thought_placeholder}'")
        print("Enter number to toggle, 3 to set text, or 'back'.")
        choice = input("> ").lower().strip()
        if choice == '1': discussion.show_thoughts = not discussion.show_thoughts
        elif choice == '2': discussion.include_thoughts_in_context = not discussion.include_thoughts_in_context
        elif choice == '3': discussion.thought_placeholder = input("Enter new placeholder text: ")
        elif choice == 'back': break
        else: ASCIIColors.red("Invalid choice.")

def handle_info_command(discussion: LollmsDiscussion):
    ASCIIColors.cyan("\n--- Discussion Info ---")
    rem_tokens = discussion.remaining_tokens
    if rem_tokens is not None:
        max_ctx = discussion.lollmsClient.binding.ctx_size
        ASCIIColors.yellow(f"Context Window: {rem_tokens} / {max_ctx} tokens remaining")
    else:
        ASCIIColors.yellow("Context Window: Max size not available from binding.")
    handle_config_command(discussion)

def chat_loop(discussion: LollmsDiscussion, personality: Optional[LollmsPersonality]):
    display_branch_history(discussion)
    
    print("\n--- Entering Chat ---")
    p_name = f" (with '{personality.name}')" if personality else ""
    ASCIIColors.cyan(f"Commands: /back, /tree, /switch <id>, /process, /history, /config, /info{p_name}")
    
    def stream_to_console(token: str, msg_type: MSG_TYPE):
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: print(token, end="", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK: ASCIIColors.magenta(token, end="", flush=True)
        return True

    while True:
        user_input = input("\nYou > ").strip()
        if not user_input: continue
        if user_input.lower() == '/back': break
        if user_input.lower() == '/history': display_branch_history(discussion); continue
        if user_input.lower() == '/tree': display_message_tree(discussion); continue
        if user_input.lower() == '/config': handle_config_command(discussion); continue
        if user_input.lower() == '/info': handle_info_command(discussion); continue

        if user_input.lower().startswith('/switch '):
            try:
                msg_id_part = user_input.split(' ', 1)[1]
                # Find the full message ID from the partial one
                full_id = next((mid for mid in discussion._message_index if mid.endswith(msg_id_part)), None)
                if not full_id: raise ValueError(f"No message found ending with '{msg_id_part}'")
                discussion.switch_to_branch(full_id)
                ASCIIColors.green(f"Switched to branch ending at message {full_id}.")
                display_branch_history(discussion)
            except IndexError: ASCIIColors.red("Usage: /switch <last_8_chars_of_id>")
            except ValueError as e: ASCIIColors.red(f"Error: {e}")
            continue

        if user_input.lower() == '/process':
            try:
                file_path_str = input("Enter path to text file: ").strip()
                chunk_size_str = input("Enter chunk size in characters [4096]: ").strip() or "4096"
                file_path = Path(file_path_str)
                if not file_path.exists():
                    ASCIIColors.red(f"File not found: {file_path}"); continue
                large_text = file_path.read_text(encoding='utf-8')
                ASCIIColors.yellow(f"Read {len(large_text)} characters from file.")
                user_prompt = input("What should I do with this text? > ").strip()
                if not user_prompt:
                    ASCIIColors.red("Prompt cannot be empty."); continue
                
                ASCIIColors.blue("AI is processing the document...")
                ai_message = discussion.process_and_summarize(large_text, user_prompt, chunk_size=int(chunk_size_str))
                ASCIIColors.blue(f"\nAI: {ai_message.content}")
                if ai_message.scratchpad:
                    ASCIIColors.yellow(f"    [AI's Scratchpad: {ai_message.scratchpad[:150]}...]")
            except Exception as e: ASCIIColors.red(f"An error occurred during processing: {e}")
            continue

        print("AI > ", end="", flush=True)
        discussion.chat(user_input, personality=personality, streaming_callback=stream_to_console)
        print()

def regenerate_response(discussion: LollmsDiscussion, personality: Optional[LollmsPersonality]):
    try:
        ASCIIColors.yellow("\nRegenerating last AI response (new branch will be created)...")
        print("New AI > ", end="", flush=True)
        def stream_to_console(token: str, msg_type: MSG_TYPE):
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: print(token, end="", flush=True)
            elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK: ASCIIColors.magenta(token, end="", flush=True)
            return True
        discussion.regenerate_branch(personality=personality, streaming_callback=stream_to_console)
        print()
        ASCIIColors.green(f"New branch created. Active message is now {discussion.active_branch_id}")
        ASCIIColors.cyan("Use '/tree' to see the branching structure.")
    except (ValueError, AttributeError) as e:
        ASCIIColors.red(f"Could not regenerate: {e}")

if __name__ == "__main__":
    # --- Create dummy personalities and data for first-time run ---
    personalities_folder = Path("./personalities")
    personalities_folder.mkdir(exist_ok=True)
    
    lollms_facts_file = personalities_folder / "lollms_facts.txt"
    if not lollms_facts_file.exists():
        lollms_facts_file.write_text(
            "LoLLMs is a project created by ParisNeo. It stands for Lord of Large Language Models. It aims to provide a unified interface for all LLMs. The client library allows for advanced discussion and agentic features."
        )

    lollms_expert_yaml = personalities_folder / "lollms_expert.yaml"
    if not lollms_expert_yaml.exists():
        lollms_expert_yaml.write_text("""
name: LoLLMs Expert
author: Manual
category: AI Tools
description: An expert on the LoLLMs project.
system_prompt: You are an expert on the LoLLMs project. Answer questions based on the provided information. Be concise.
data_files:
    - lollms_facts.txt
""")
    
    parrot_yaml = personalities_folder / "parrot.yaml"
    if not parrot_yaml.exists():
        parrot_yaml.write_text("""
name: Parrot
author: Manual
category: Fun
description: A personality that just repeats what you say.
system_prompt: You are a parrot. You must start every sentence with 'Squawk!'.
""")

    parrot_py = personalities_folder / "parrot.py"
    if not parrot_py.exists():
        parrot_py.write_text("""
def run(discussion, on_chunk_callback):
    # This script overrides the normal chat flow.
    user_message = discussion.get_branch(discussion.active_branch_id)[-1].content
    response = f"Squawk! {user_message}! Squawk!"
    if on_chunk_callback:
        # We need to simulate the message type for the callback
        from lollms_client import MSG_TYPE
        on_chunk_callback(response, MSG_TYPE.MSG_TYPE_CHUNK)
    return response # Return the full raw response
""")
    main()