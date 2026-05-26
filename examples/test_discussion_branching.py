#!/usr/bin/env python3
"""
test_discussion_branching.py
============================
A comprehensive integration test demonstrating first-class branch management:
1. Creating a linear conversation history.
2. Forking to create divergent conversational paths (leaves/branches).
3. Fetching and displaying the complete hierarchical tree.
4. Navigating branches and cycling through sibling replies.
5. Computing unified branch diffs (divergence analysis).
6. Merging divergent branches back together.
7. Deleting and pruning subtrees.
8. Verifying database persistence of branch trees and active pointers.

Requirements:
pip install lollms_client ascii_colors
"""

import sys
import tempfile
import json
from pathlib import Path

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from ascii_colors import ASCIIColors


def print_tree_node(node, depth=0, active_path=None):
    """Recursively pretty-prints a tree node."""
    indent = "  " * depth
    prefix = "└── " if depth > 0 else ""
    active_marker = "★ [ACTIVE]" if node.is_active_path else ""
    
    print(f"{indent}{prefix}[{node.message_id[:8]}] {node.sender.upper()}: {node.content_preview} {active_marker}")
    for child in node.children:
        print_tree_node(child, depth + 1)


def main():
    print("=" * 80)
    print("🔬 LOLLMS_CLIENT DISCUSSION BRANCHING SYSTEM: COMPREHENSIVE TEST")
    print("=" * 80)

    # ── 1. SETUP IN-MEMORY LOLLMS CLIENT & DISCUSSION ─────────────────────
    print("\n[Step 1] Initializing in-memory discussion...")
    client = LollmsClient(user_name="ParisNeo", ai_name="Lollms", debug=True)
    discussion = LollmsDiscussion(lollmsClient=client)
    print("✅ Discussion initialized.")

    # ── 2. CREATE A LINEAR HISTORY ────────────────────────────────────────
    print("\n[Step 2] Building initial linear conversation history...")
    msg_u1 = discussion.add_message(sender="user", content="Hi Lollms, what is the capital of France?")
    msg_a1 = discussion.add_message(sender="assistant", content="The capital of France is Paris.")
    msg_u2 = discussion.add_message(sender="user", content="Thanks! Can you tell me its population?")
    msg_a2 = discussion.add_message(sender="assistant", content="The population of Paris is approximately 2.1 million within the city limits.")

    print(f"  • Active branch tip: {discussion.active_branch_id[:8]}")
    print(f"  • Total messages in active branch: {len(discussion.get_branch(discussion.active_branch_id))}")

    # ── 3. FORK A NEW BRANCH ──────────────────────────────────────────────
    print("\n[Step 3] Forking a new branch from the first assistant reply...")
    # Forking from the first assistant reply (msg_a1)
    # This starts a divergent line where the user asks a completely different question.
    msg_fork_u = discussion.fork_from(
        message_id=msg_a1.id,
        label="Alternative Capital Discussion",
        initial_content="Actually, let's talk about Germany. What is its capital?",
        initial_sender="user",
        initial_sender_type="user"
    )
    msg_fork_a = discussion.add_message(sender="assistant", content="The capital of Germany is Berlin.")

    # ── 4. LIST BRANCHES AND PRINT HIERARCHICAL TREE ───────────────────────
    print("\n[Step 4] Querying branches and printing hierarchical tree...")
    branches = discussion.list_branches()
    print(f"  • Total branches found in tree: {len(branches)}")
    for idx, b in enumerate(branches, 1):
        active_status = " (ACTIVE)" if b.is_active else ""
        print(f"    Branch {idx}: Leaf [{b.leaf_id[:8]}]{active_status} | Label: {b.label} | Depth: {b.depth}")

    print("\n  Complete Conversational Tree Hierarchy:")
    print("-" * 60)
    roots = discussion.get_tree()
    for r in roots:
        print_tree_node(r)
    print("-" * 60)

    # Save leaf IDs for navigation
    leaf_germany = msg_fork_a.id
    leaf_france = msg_a2.id

    # ── 5. TEST BRANCH NAVIGATION & SIBLING CYCLING ───────────────────────
    print("\n[Step 5] Testing branch switching and sibling reply navigation...")
    
    # Switch back to the original France branch
    print(f"  • Current Active Tip: {discussion.active_branch_id[:8]}")
    print(f"  • Switching active branch back to original France branch tip [{leaf_france[:8]}]...")
    success = discussion.switch_branch(leaf_france)
    print(f"    - Success: {success} | New Active Tip: {discussion.active_branch_id[:8]}")
    assert discussion.active_branch_id == leaf_france, "Branch switch failed!"

    # Sibling navigation
    # Let's add an alternative reply to the same question (creating siblings)
    # We navigate to the parent of the France leaf and add an alternative
    parent_id = msg_a2.parent_id # msg_u2
    print(f"  • Adding an alternative assistant reply to the user message [{parent_id[:8]}]...")
    alternative_a2 = discussion.add_message(
        sender="assistant", 
        content="Paris has a population of about 2.14 million people in the city center.",
        parent_id=parent_id
    )
    
    # We now have two siblings for this turn. Let's switch between them!
    print(f"  • Current Active Reply: {discussion.active_branch_id[:8]} -> '{discussion.get_message(discussion.active_branch_id).content}'")
    print("  • Cycling to previous sibling reply...")
    prev_sibling = discussion.switch_to_sibling(direction=-1)
    print(f"    - Active after cycle: {discussion.active_branch_id[:8]} -> '{prev_sibling.content}'")
    
    print("  • Cycling to next sibling reply...")
    next_sibling = discussion.switch_to_sibling(direction=1)
    print(f"    - Active after cycle: {discussion.active_branch_id[:8]} -> '{next_sibling.content}'")

    # ── 6. COMPUTE BRANCH DIVERGENCE (DIFF) ───────────────────────────────
    print("\n[Step 6] Computing branch diff (divergence analysis)...")
    diff = discussion.branch_diff(leaf_france, leaf_germany)
    
    print(f"  • Common Ancestor Message ID: {diff['common_ancestor_id'][:8]}")
    print(f"    - Content: '{discussion.get_message(diff['common_ancestor_id']).content}'")
    print(f"  • Messages unique to France branch (Branch A): {[mid[:8] for mid in diff['only_in_a']]}")
    print(f"  • Messages unique to Germany branch (Branch B): {[mid[:8] for mid in diff['only_in_b']]}")
    print(f"  • Shared root messages: {[mid[:8] for mid in diff['shared']]}")

    # ── 7. MERGE BRANCHES ─────────────────────────────────────────────────
    print("\n[Step 7] Merging divergent Germany branch onto France branch...")
    # This copies unique messages from Germany branch, appends a separator, and joins them to France branch
    merged_tip = discussion.merge_branches(
        source_leaf_id=leaf_germany,
        target_leaf_id=leaf_france,
        separator_content="--- Context Merged: Alternate Geographical Topics ---"
    )
    
    print(f"  • Merged Tip Message ID: {merged_tip.id[:8]}")
    print("\n  Full Merged Branch Conversation Log:")
    print("-" * 60)
    merged_history = discussion.get_branch(merged_tip.id)
    for m in merged_history:
        print(f"  [{m.sender.upper()}]: {m.content}")
    print("-" * 60)

    # ── 8. DELETION AND SUBTREE PRUNING ───────────────────────────────────
    print("\n[Step 8] Testing branch leaf deletion and subtree pruning...")
    
    # Let's prune the alternate Germany branch from the tree completely
    print(f"  • Pruning alternate Germany branch subtree starting at [{msg_fork_u.id[:8]}]...")
    pruned_count = discussion.prune_branch(msg_fork_u.id)
    print(f"    - Total messages deleted: {pruned_count}")
    
    print("\n  Hierarchy after Pruning (Germany branch should be gone):")
    print("-" * 60)
    for r in discussion.get_tree():
        print_tree_node(r)
    print("-" * 60)

    # ── 9. PERSISTENCE CHECK ──────────────────────────────────────────────
    print("\n[Step 9] Testing persistent SQLite serialization of branch trees...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "persistent_branches.db"
        db_mgr = LollmsDataManager(f"sqlite:///{db_path}")
        
        try:
            # Create a persistent discussion and clone our structured branch there
            p_disc = LollmsDiscussion.create_new(
                lollms_client=client,
                db_manager=db_mgr,
                id="persistent_tree_test",
                autosave=True
            )
            
            # Re-build structured tree in DB
            r_u1 = p_disc.add_message(sender="user", content="Question A")
            r_a1 = p_disc.add_message(sender="assistant", content="Answer A")
            # Corrected parameter: fork_from uses initial_content to map back to add_message content
            r_fork = p_disc.fork_from(r_u1.id, initial_content="Divergent Question B")
            r_fork_a = p_disc.add_message(sender="assistant", content="Divergent Answer B")
            
            # Ensure we commit everything to SQLite
            p_disc.commit()
            p_disc.close()
            
            # Reload the discussion from SQLite
            reloaded_disc = db_mgr.get_discussion(client, "persistent_tree_test")
            reloaded_branches = reloaded_disc.list_branches()
            
            print(f"  • Saved DB paths correctly. Reloaded branches count: {len(reloaded_branches)}")
            assert len(reloaded_branches) == 2, "DB Restore failed to recover all branch leaves!"
            
            print("\n  Reloaded Database Tree Hierarchy:")
            print("-" * 60)
            for r in reloaded_disc.get_tree():
                print_tree_node(r)
            print("-" * 60)
            
            reloaded_disc.close()
        finally:
            # Dispose of the engine connection pool so Windows can release SQLite file locks under all execution flows
            db_mgr.engine.dispose()

    print("\n" + "=" * 80)
    print("🎉 ALL BRANCHING COMPLIANCE TESTS FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
