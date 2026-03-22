# lollms_discussion/_mixin_branch.py
# BranchMixin: first-class branch management for LollmsDiscussion.
#
# Concepts
# --------
# A discussion's messages form a *tree*, not a list.  Every message has an
# optional parent_id.  A *branch* is the path from a root message (parent=None)
# to a leaf (no children).  The *active branch* is the leaf whose ID is stored
# in discussion.active_branch_id.
#
# This mixin adds:
#
#   BranchInfo dataclass      — snapshot of one branch (id, label, depth, …)
#   MessageNode dataclass     — one node in the tree with child info
#
#   get_children(msg_id)      — direct children of a message
#   get_siblings(msg_id)      — messages that share the same parent
#   get_branch_info(leaf_id)  — BranchInfo for any leaf
#   list_branches()           — all branches in the discussion
#   get_tree()                — full tree as nested MessageNode dicts
#   get_message_branches(id)  — branches that pass *through* a given message
#
#   switch_branch(leaf_id)    — change active branch  (alias: switch_to_branch)
#   fork_from(msg_id, …)      — start a new branch from any message
#   delete_branch(leaf_id)    — remove a branch leaf (and childless ancestors)
#   prune_branch(msg_id)      — remove a message and all its descendants
#   merge_branches(…)         — concatenate two branches into a new one
#
# All methods are safe to call on both DB-backed and in-memory discussions.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors, trace_exception


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BranchInfo:
    """Lightweight descriptor for one branch (root-to-leaf path)."""
    leaf_id:       str                 # the leaf message's ID
    message_ids:   List[str]          # [root_id, …, leaf_id]
    depth:         int                 # number of messages in the branch
    label:         str                 # human-readable label (auto or custom)
    is_active:     bool               # True if this is the current active branch
    created_at:    Optional[datetime] = None  # leaf message's creation time
    last_sender:   str                = ""    # sender of the leaf message
    last_content_preview: str        = ""    # first 80 chars of leaf content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_id":             self.leaf_id,
            "message_ids":         self.message_ids,
            "depth":               self.depth,
            "label":               self.label,
            "is_active":           self.is_active,
            "created_at":          self.created_at.isoformat() if self.created_at else None,
            "last_sender":         self.last_sender,
            "last_content_preview": self.last_content_preview,
        }


@dataclass
class MessageNode:
    """One node in the message tree, with references to its children."""
    message_id:  str
    parent_id:   Optional[str]
    sender:      str
    sender_type: str
    content_preview: str            # first 120 chars
    created_at:  Optional[datetime]
    children:    List["MessageNode"] = field(default_factory=list)
    branch_count: int = 0           # number of branches passing through this node
    is_active_path: bool = False    # True if on current active branch

    def to_dict(self, recursive: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "message_id":       self.message_id,
            "parent_id":        self.parent_id,
            "sender":           self.sender,
            "sender_type":      self.sender_type,
            "content_preview":  self.content_preview,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "branch_count":     self.branch_count,
            "is_active_path":   self.is_active_path,
            "children":         [c.to_dict(recursive) for c in self.children] if recursive else [],
        }
        return d


# ---------------------------------------------------------------------------
# BranchMixin
# ---------------------------------------------------------------------------

class BranchMixin:
    """
    First-class branch management: discovery, navigation, forking, deletion,
    and merging.  Designed to sit in the MRO alongside CoreMixin.
    """

    # ================================================================ helpers

    def _children_map(self) -> Dict[str, List[str]]:
        """Returns {parent_id: [child_id, …]} for every message in the index."""
        self._rebuild_message_index()
        cm: Dict[str, List[str]] = {mid: [] for mid in self._message_index}
        for mid, msg in self._message_index.items():
            pid = msg.parent_id
            if pid and pid in cm:
                cm[pid].append(mid)
        return cm

    def _collect_leaves(self, cm: Dict[str, List[str]]) -> List[str]:
        """Return the IDs of all messages that have no children."""
        return [mid for mid, children in cm.items() if not children]

    def _path_to_root(self, leaf_id: str) -> List[str]:
        """
        Walks parent links from *leaf_id* up to the root.
        Returns the path as [root_id, …, leaf_id].
        """
        path: List[str] = []
        cur = leaf_id
        visited: set = set()
        while cur and cur in self._message_index:
            if cur in visited:
                break          # cycle guard
            visited.add(cur)
            path.append(cur)
            cur = self._message_index[cur].parent_id
        return list(reversed(path))

    # ================================================================ discovery

    def get_children(self, message_id: str) -> List[Any]:
        """
        Returns LollmsMessage objects that are direct children of *message_id*.
        An empty list means the message is a leaf (end of a branch).
        """
        from ._message import LollmsMessage
        cm = self._children_map()
        return [
            LollmsMessage(self, self._message_index[cid])
            for cid in cm.get(message_id, [])
            if cid in self._message_index
        ]

    def get_siblings(self, message_id: str) -> List[Any]:
        """
        Returns all messages that share the same parent as *message_id*,
        including *message_id* itself.  Useful for showing alternate replies.
        """
        from ._message import LollmsMessage
        if message_id not in self._message_index:
            return []
        parent_id = self._message_index[message_id].parent_id
        siblings = [
            LollmsMessage(self, msg)
            for mid, msg in self._message_index.items()
            if msg.parent_id == parent_id
        ]
        # Sort by creation time so the UI can display them in order
        siblings.sort(key=lambda m: m.created_at or datetime.min)
        return siblings

    def get_branch_info(self, leaf_id: str) -> Optional[BranchInfo]:
        """
        Returns a BranchInfo snapshot for the branch whose tip is *leaf_id*.
        Returns None if *leaf_id* is not in the message index.
        """
        if leaf_id not in self._message_index:
            return None
        path      = self._path_to_root(leaf_id)
        leaf_msg  = self._message_index[leaf_id]
        content   = (getattr(leaf_msg, 'content', '') or '').strip()
        preview   = content[:80] + ('…' if len(content) > 80 else '')
        sender    = getattr(leaf_msg, 'sender', '')
        created   = getattr(leaf_msg, 'created_at', None)

        # Auto-label: "Branch from <sender> @ depth <n>"
        label = f"Branch #{leaf_id[:6]} — {sender} (depth {len(path)})"
        meta  = getattr(leaf_msg, 'message_metadata', {}) or {}
        if meta.get('branch_label'):
            label = meta['branch_label']

        return BranchInfo(
            leaf_id              = leaf_id,
            message_ids          = path,
            depth                = len(path),
            label                = label,
            is_active            = (leaf_id == self.active_branch_id),
            created_at           = created,
            last_sender          = sender,
            last_content_preview = preview,
        )

    def list_branches(self) -> List[BranchInfo]:
        """
        Returns a BranchInfo for every leaf in the discussion tree, sorted by
        creation time of the leaf message (oldest first).
        """
        self._rebuild_message_index()
        cm     = self._children_map()
        leaves = self._collect_leaves(cm)
        infos  = [self.get_branch_info(lid) for lid in leaves]
        infos  = [b for b in infos if b is not None]
        infos.sort(key=lambda b: b.created_at or datetime.min)
        return infos

    def get_tree(self) -> List[MessageNode]:
        """
        Returns the full message tree as a forest of MessageNode objects
        (list of roots, each with a recursive ``children`` attribute).

        The ``branch_count`` field on each node tells how many branches pass
        through it.  ``is_active_path`` is True for nodes on the current
        active branch.
        """
        self._rebuild_message_index()
        cm = self._children_map()

        # Determine which IDs are on the active branch
        active_path: set = set()
        if self.active_branch_id:
            active_path = set(self._path_to_root(self.active_branch_id))

        # Count branches through each node (= number of leaves in subtree)
        leaves = set(self._collect_leaves(cm))

        def _count_leaves(mid: str) -> int:
            if mid in leaves:
                return 1
            return sum(_count_leaves(c) for c in cm.get(mid, []))

        def _make_node(mid: str) -> MessageNode:
            msg  = self._message_index[mid]
            cont = (getattr(msg, 'content', '') or '').strip()
            node = MessageNode(
                message_id      = mid,
                parent_id       = getattr(msg, 'parent_id', None),
                sender          = getattr(msg, 'sender', ''),
                sender_type     = getattr(msg, 'sender_type', ''),
                content_preview = cont[:120] + ('…' if len(cont) > 120 else ''),
                created_at      = getattr(msg, 'created_at', None),
                branch_count    = _count_leaves(mid),
                is_active_path  = mid in active_path,
            )
            child_ids = sorted(
                cm.get(mid, []),
                key=lambda c: getattr(self._message_index.get(c), 'created_at', datetime.min)
                              if self._message_index.get(c) else datetime.min,
            )
            node.children = [_make_node(c) for c in child_ids]
            return node

        roots = [
            mid for mid, msg in self._message_index.items()
            if not msg.parent_id or msg.parent_id not in self._message_index
        ]
        roots.sort(
            key=lambda mid: getattr(self._message_index[mid], 'created_at', datetime.min)
                            if self._message_index.get(mid) else datetime.min
        )
        return [_make_node(r) for r in roots]

    def get_message_branches(self, message_id: str) -> List[BranchInfo]:
        """
        Returns all branches (BranchInfo) that pass *through* the given
        message — i.e. branches whose path contains *message_id*.

        Useful for a UI that wants to show a "branch picker" when the user
        clicks on a message.
        """
        all_branches = self.list_branches()
        return [b for b in all_branches if message_id in b.message_ids]

    # ================================================================ navigation

    def switch_branch(self, leaf_id: str) -> bool:
        """
        Change the active branch to the one whose tip is *leaf_id*.
        Returns True on success, False if *leaf_id* is not found.

        This is a more explicit alias for the existing switch_to_branch(),
        adding a return value and a validity check.
        """
        self._rebuild_message_index()
        if leaf_id not in self._message_index:
            ASCIIColors.warning(f"[BranchMixin] switch_branch: '{leaf_id}' not found.")
            return False
        self.active_branch_id = leaf_id
        self.touch()
        return True

    def switch_to_sibling(self, direction: int = 1) -> Optional[Any]:
        """
        Move to the next (+1) or previous (-1) sibling of the current active
        branch tip's parent reply.  Useful for cycling through alternate AI
        replies to the same user message.

        Returns the new active LollmsMessage, or None if there is no sibling
        in that direction.
        """
        from ._message import LollmsMessage
        if not self.active_branch_id:
            return None
        siblings = self.get_siblings(self.active_branch_id)
        if len(siblings) <= 1:
            return None
        current_idx = next(
            (i for i, s in enumerate(siblings) if s.id == self.active_branch_id), None
        )
        if current_idx is None:
            return None
        new_idx = (current_idx + direction) % len(siblings)
        new_tip = siblings[new_idx]
        # Find the deepest leaf under this sibling
        new_leaf = self._find_deepest_leaf(new_tip.id) or new_tip.id
        self.active_branch_id = new_leaf
        self.touch()
        return LollmsMessage(self, self._message_index[new_leaf])

    # ================================================================ forking

    def fork_from(
        self,
        message_id: str,
        label: Optional[str] = None,
        initial_content: Optional[str] = None,
        initial_sender: str = "user",
        initial_sender_type: str = "user",
        **extra_msg_kwargs,
    ) -> Any:
        """
        Start a new branch from *message_id* by adding a new child message.

        Parameters
        ----------
        message_id : str
            The message from which to fork.  The new message's parent_id
            will be set to this ID.
        label : str | None
            Optional human-readable label stored in the new message's
            metadata as ``branch_label``.
        initial_content : str | None
            Content for the first message on the new branch.  Defaults to
            an empty string (the caller should populate it later).
        initial_sender / initial_sender_type : str
            Sender info for the fork message.
        **extra_msg_kwargs
            Forwarded to ``add_message``.

        Returns
        -------
        LollmsMessage
            The newly created fork message, which is now the active branch tip.
        """
        if message_id not in self._message_index:
            raise ValueError(f"fork_from: message '{message_id}' not found.")

        meta: Dict[str, Any] = extra_msg_kwargs.pop('metadata', {}) or {}
        if label:
            meta['branch_label'] = label
        meta.setdefault('forked_from', message_id)

        new_msg = self.add_message(
            sender=initial_sender,
            sender_type=initial_sender_type,
            content=initial_content or "",
            parent_id=message_id,
            metadata=meta,
            **extra_msg_kwargs,
        )
        ASCIIColors.info(
            f"[BranchMixin] Forked from '{message_id[:8]}…' → new branch tip '{new_msg.id[:8]}…'"
        )
        return new_msg

    # ================================================================ deletion

    def delete_branch(self, leaf_id: str, keep_ancestors: bool = True) -> int:
        """
        Delete the branch whose tip is *leaf_id*.

        Parameters
        ----------
        leaf_id : str
            The leaf message to start deletion from.
        keep_ancestors : bool
            If True (default), only remove the leaf.  If the leaf's parent
            becomes childless after removal, it too is removed — and so on
            recursively up the tree, stopping as soon as an ancestor still
            has other children or is a root.
            If False, delete the entire path from root to leaf (use with
            care: this removes messages shared with sibling branches).

        Returns
        -------
        int
            Number of messages removed.
        """
        self._rebuild_message_index()
        if leaf_id not in self._message_index:
            ASCIIColors.warning(f"[BranchMixin] delete_branch: '{leaf_id}' not found.")
            return 0

        cm = self._children_map()

        # Verify it is actually a leaf (no children)
        if cm.get(leaf_id):
            raise ValueError(
                f"'{leaf_id}' has children — use prune_branch() to remove a "
                "message and all its descendants."
            )

        removed: List[str] = []
        cur = leaf_id

        while cur:
            msg    = self._message_index.get(cur)
            if msg is None:
                break
            parent = getattr(msg, 'parent_id', None)

            # Remove this message
            self.remove_message(cur)
            removed.append(cur)

            if not keep_ancestors:
                # Walk all the way to root regardless
                cur = parent
                continue

            # Stop if the parent still has other children
            if parent and parent in self._message_index:
                remaining_children = [
                    c for c in self._children_map().get(parent, [])
                    if c != cur
                ]
                if remaining_children:
                    break
                # Parent is now childless — remove it too
                cur = parent
            else:
                break

        ASCIIColors.info(f"[BranchMixin] Deleted {len(removed)} message(s): {removed}")

        # Fix up active branch if it was on the deleted path
        if self.active_branch_id in removed:
            self._validate_and_set_active_branch()

        if self._is_db_backed:
            self.commit()

        return len(removed)

    def prune_branch(self, message_id: str) -> int:
        """
        Remove *message_id* AND all of its descendants from the discussion.
        The parent of *message_id* (if any) is preserved.

        Returns the number of messages removed.

        This is the operation to use when you want to cut an entire subtree
        (e.g. a bad AI reply and all follow-ups to it).
        """
        self._rebuild_message_index()
        if message_id not in self._message_index:
            ASCIIColors.warning(f"[BranchMixin] prune_branch: '{message_id}' not found.")
            return 0

        # Collect all descendants via BFS
        cm     = self._children_map()
        to_del: List[str] = []
        queue  = [message_id]
        qi     = 0
        while qi < len(queue):
            cur = queue[qi]; qi += 1
            to_del.append(cur)
            for child_id in cm.get(cur, []):
                if child_id not in to_del:
                    queue.append(child_id)

        for mid in to_del:
            self.remove_message(mid)

        ASCIIColors.info(f"[BranchMixin] Pruned {len(to_del)} message(s).")

        if self.active_branch_id in to_del:
            self._validate_and_set_active_branch()

        if self._is_db_backed:
            self.commit()

        return len(to_del)

    # ================================================================ merging

    def merge_branches(
        self,
        source_leaf_id: str,
        target_leaf_id: Optional[str] = None,
        separator_content: str = "--- merged from another branch ---",
        separator_sender: str = "system",
    ) -> Any:
        """
        Append all messages from the *source* branch (from the first message
        NOT shared with the *target* branch) onto the *target* branch,
        inserting an optional separator message.

        Parameters
        ----------
        source_leaf_id : str
            Leaf of the branch to copy messages FROM.
        target_leaf_id : str | None
            Leaf of the branch to append messages TO.
            Defaults to the current active branch.
        separator_content : str
            Content of a system separator message inserted between branches.
            Pass an empty string to skip the separator.
        separator_sender : str
            Sender name for the separator message.

        Returns
        -------
        LollmsMessage
            The new tip of the merged branch (last appended message).
        """
        from ._message import LollmsMessage

        target_leaf = target_leaf_id or self.active_branch_id
        if not target_leaf or target_leaf not in self._message_index:
            raise ValueError("merge_branches: target leaf not found.")
        if source_leaf_id not in self._message_index:
            raise ValueError("merge_branches: source leaf not found.")

        source_path = self._path_to_root(source_leaf_id)
        target_path = set(self._path_to_root(target_leaf))

        # Find messages in source NOT already in target
        unique_source = [mid for mid in source_path if mid not in target_path]
        if not unique_source:
            ASCIIColors.warning("[BranchMixin] merge_branches: branches are identical.")
            return LollmsMessage(self, self._message_index[target_leaf])

        current_parent = target_leaf

        # Separator
        if separator_content:
            sep = self.add_message(
                sender=separator_sender,
                sender_type="system",
                content=separator_content,
                parent_id=current_parent,
                metadata={"merge_separator": True},
            )
            current_parent = sep.id

        # Copy unique source messages in order
        last_msg = None
        for src_id in unique_source:
            src = self._message_index[src_id]
            new_msg = self.add_message(
                sender=getattr(src, 'sender', 'unknown'),
                sender_type=getattr(src, 'sender_type', 'user'),
                content=getattr(src, 'content', ''),
                parent_id=current_parent,
                images=list(getattr(src, 'images', []) or []),
                metadata={
                    **(getattr(src, 'message_metadata', {}) or {}),
                    "merged_from": src_id,
                },
            )
            current_parent = new_msg.id
            last_msg = new_msg

        ASCIIColors.info(
            f"[BranchMixin] Merged {len(unique_source)} message(s) from "
            f"'{source_leaf_id[:8]}…' onto '{target_leaf[:8]}…'."
        )
        return last_msg

    # ================================================================ labelling

    def set_branch_label(self, leaf_id: str, label: str) -> bool:
        """
        Attach a human-readable label to the branch identified by *leaf_id*.
        The label is stored in the leaf message's metadata and is returned
        by get_branch_info() / list_branches().

        Returns True on success.
        """
        if leaf_id not in self._message_index:
            return False
        msg  = self._message_index[leaf_id]
        meta = dict(getattr(msg, 'message_metadata', {}) or {})
        meta['branch_label'] = label
        msg.message_metadata = meta
        self.touch()
        if self._is_db_backed:
            self.commit()
        return True

    # ================================================================ utility

    def branch_diff(
        self,
        leaf_a: str,
        leaf_b: str,
    ) -> Dict[str, Any]:
        """
        Returns a dict describing the divergence between two branches:

        {
          "common_ancestor_id": str | None,
          "only_in_a":          [message_id, …],
          "only_in_b":          [message_id, …],
          "shared":             [message_id, …],
        }
        """
        path_a = self._path_to_root(leaf_a)
        path_b = self._path_to_root(leaf_b)
        set_a, set_b = set(path_a), set(path_b)

        shared   = [mid for mid in path_a if mid in set_b]
        only_a   = [mid for mid in path_a if mid not in set_b]
        only_b   = [mid for mid in path_b if mid not in set_a]
        ancestor = shared[-1] if shared else None

        return {
            "common_ancestor_id": ancestor,
            "only_in_a":          only_a,
            "only_in_b":          only_b,
            "shared":             shared,
        }