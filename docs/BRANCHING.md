# Branching in `LollmsDiscussion`

> How conversation state is stored, forked, navigated, and regenerated in
> `lollms_client`. This is the developer reference for the message DAG behind
> every discussion.

**TL;DR** — A discussion is a **directed acyclic graph (DAG) of messages**. A
*branch* is simply any root-to-leaf path in that graph, a *leaf* is a message
with no children, and the *active branch* is one pointer (`active_branch_id`)
stored on the discussion row. There is **no dedicated branch table** — branching
is emergent from each message's `parent_id`.

---

## Table of Contents

1. [Where the API lives](#1-where-the-api-lives)
2. [Conceptual model](#2-conceptual-model)
3. [The `LollmsMessage` entity](#3-the-lollmsmessage-entity)
4. [Appending & forking](#4-appending--forking)
5. [Core branch walking (`get_branch`)](#5-core-branch-walking-get_branch)
6. [API by mixin](#6-api-by-mixin)
7. [`regenerate_branch` semantics](#7-regenerate_branch-semantics)
8. [Persistence & reload](#8-persistence--reload)
9. [Usage examples](#9-usage-examples)
10. [DAG diagram](#10-dag-diagram)
11. [Cross-references & corrections](#11-cross-references--corrections)

---

## 1. Where the API lives

The `LollmsDiscussion` object is assembled from mixins. Branching logic is
spread across three of them:

| Mixin (file) | Responsibility |
| :--- | :--- |
| `CoreMixin` (`_mixin_core.py`) | Message CRUD, `get_branch`, `active_branch_id`, message index, branch validation |
| `BranchMixin` (`_mixin_branch.py`) | Branch discovery, navigation, forking, merging, deletion, diff, labelling |
| `UtilsMixin` (`_mixin_utils.py`) | `regenerate_branch`, `fix_orphan_messages`, legacy `switch_to_branch`, export/import |

> ⚠️ Common misconception: `regenerate_branch` is **not** in `BranchMixin` —
> it lives in `UtilsMixin`.

---

## 2. Conceptual model

### 2.1 A branch is a derived path, not a stored object

- There is **no `LollmsBranch` entity** and no branch table in `_db.py`.
  The only persistence entities are `DiscussionModel` and `MessageModel`.
- The in-memory index `self._message_index = {message_id: orm_object}` is the
  single source for branch computation. It is rebuilt on demand via
  `_rebuild_message_index()`.
- **A branch** = the chronological list of messages from a root ancestor
  (`parent_id is None`) down to a chosen leaf.
- **A leaf** = a message that has no children (no other message points to it as
  `parent_id`).
- **Forking** is achieved by adding a message with an explicit `parent_id` that
  is *not* the current active tip. The new message becomes a new leaf → a new
  branch tip.

### 2.2 The active branch pointer

- `active_branch_id` is a **column on the `DiscussionModel`** row (proxied
  through `CoreMixin.__getattr__`/`__setattr__`), **not** a key in
  `discussion_metadata`.
- It always points to a **leaf** (the newest message on the path the user/agent
  is currently on).
- On load, if the stored pointer is stale it is re-resolved:
  - missing id → `_find_deepest_leaf(None)` (most recent leaf overall),
  - id that has children → descend to the deepest leaf under it.

### 2.3 Orphaned messages

Deleting a message does **not** cascade to its descendants. Children that
pointed at the deleted id keep their `parent_id` but become unreachable from
`get_branch` — they are *orphans*. `fix_orphan_messages()` re-roots such
orphans onto the primary branch.

---

## 3. The `LollmsMessage` entity

`LollmsMessage` is a lightweight **proxy** over an ORM object
(`self._db_message`). Branching-relevant fields:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `id` | `str` | Unique message id (the node in the DAG). |
| `parent_id` | `Optional[str]` | **The DAG edge.** `None` for root messages. |
| `sender` | `str` | Display name (e.g. `"user"`, `"assistant"`). |
| `sender_type` | `str` | `user` / `assistant` / `system`. |
| `content` | `str` | Message body. |
| `images` | `List[str]` | Base64 / data-URL image payloads. |
| `active_images` | `List[Union[str, bool]]` | Per-image activation flags. |
| `metadata` | `Dict[str, Any]` | Maps to the `message_metadata` DB column (holds `branch_label`, `forked_from`, `merged_from`, …). |
| `created_at` | `datetime` | Used to order siblings and pick the deepest leaf. |

> There is **no `branch_id` field on a message.** Branch identity is emergent
> from `parent_id` links. Branch *labels* are stored in the **leaf's**
> `metadata["branch_label"]`.

---

## 4. Appending & forking

### 4.1 Appending to the active tip (automatic)

`CoreMixin.add_message(**kwargs)` does:

```python
parent_id = kwargs.get('parent_id', self.active_branch_id)   # default: active tip
...
self.active_branch_id = msg_id                                 # new message becomes the tip
```

So a plain `add_message` **always** extends the active branch.

### 4.2 Forking

Pass an explicit `parent_id` (or use `fork_from`):

```python
# Fork from an earlier assistant reply:
forked = discussion.add_message(
    sender="user",
    content="Let's try a different approach…",
    parent_id=some_assistant_message.id,   # explicit edge → new branch
)
# `forked.id` is now the active branch tip.
```

`BranchMixin.fork_from(...)` wraps this and also stamps the leaf's metadata:

```python
def fork_from(self, message_id, label=None, initial_content=None,
              initial_sender="user", initial_sender_type="user", **extra_msg_kwargs) -> LollmsMessage
```

- `metadata["branch_label"] = label` (when given)
- `metadata["forked_from"] = message_id`
- the new message becomes the active tip.

---

## 5. Core branch walking (`get_branch`)

`CoreMixin.get_branch(leaf_id)` walks **up** from `leaf_id` following
`parent_id` to the root, then reverses to return a **chronological root→leaf**
list of `LollmsMessage` objects.

Key guarantees:

- `visited` set guards against circular parent links.
- If `leaf_id` is not in the index, it falls back to the latest committed
  message (concurrency safety).
- `get_messages(branch_id=None)` is a convenience alias: it reads
  `active_branch_id` when `branch_id` is omitted.

```python
def get_messages(self, branch_id=None):
    leaf_id = branch_id if branch_id is not None else self.active_branch_id
    return self.get_branch(leaf_id)
```

---

## 6. API by mixin

### 6.1 `CoreMixin` (`_mixin_core.py`)

| Method | Signature | Notes |
| :--- | :--- | :--- |
| `get_branch` | `(leaf_id) -> List[LollmsMessage]` | Root→leaf walk, cycle-guarded. |
| `get_messages` | `(branch_id=None) -> List[LollmsMessage]` | Active branch when `branch_id` is None. |
| `get_message` | `(message_id) -> Optional[LollmsMessage]` | Single node lookup. |
| `get_all_messages_flat` | `() -> List[LollmsMessage]` | All nodes (unsorted). |
| `add_message` | `(**kwargs) -> LollmsMessage` | `parent_id` defaults to active tip; sets new tip. |
| `remove_message` / `delete_message` | `(message_id) -> bool` | Single-node delete (no cascade). Deferred to next `commit()`. |
| `active_branch_id` | property (via proxy) | Leaf pointer, persisted on the DB row. |
| `_validate_and_set_active_branch` | `() -> None` | Re-resolves a stale/child-bearing tip. |
| `_find_deepest_leaf` | `(start_id) -> Optional[str]` | Most-recent leaf reachable from `start_id` (or overall). |
| `create_new` / `from_messages` | classmethods | Factories; see §8. |

### 6.2 `BranchMixin` (`_mixin_branch.py`)

**Dataclasses**

```python
@dataclass
class BranchInfo:
    leaf_id: str
    message_ids: List[str]
    depth: int
    label: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_sender: str = ""
    last_content_preview: str = ""
    def to_dict(self) -> Dict[str, Any]: ...

@dataclass
class MessageNode:
    message_id: str
    parent_id: Optional[str]
    sender: str
    sender_type: str
    content_preview: str
    created_at: Optional[datetime]
    children: List["MessageNode"] = field(default_factory=list)
    branch_count: int = 0          # # of leaves in this subtree
    is_active_path: bool = False   # True if on the active branch
    def to_dict(self, recursive: bool = True) -> Dict[str, Any]: ...
```

**Discovery**

| Method | Signature |
| :--- | :--- |
| `list_branches` | `() -> List[BranchInfo]` — one per leaf, sorted by leaf `created_at`. |
| `get_branch_info` | `(leaf_id) -> Optional[BranchInfo]` |
| `get_tree` | `() -> List[MessageNode]` — full forest with `branch_count` / `is_active_path`. |
| `get_message_branches` | `(message_id) -> List[BranchInfo]` — branches passing through a message. |
| `get_children` | `(message_id) -> List[LollmsMessage]` |
| `get_siblings` | `(message_id) -> List[LollmsMessage]` — same parent, sorted by `created_at`. |

**Navigation**

| Method | Signature |
| :--- | :--- |
| `switch_branch` | `(leaf_id) -> bool` — validates id, sets `active_branch_id`, `touch()`. |
| `switch_to_sibling` | `(direction: int = 1) -> Optional[LollmsMessage]` — cycles among siblings of the tip, then descends to the deepest leaf. |

**Forking / mutation**

| Method | Signature |
| :--- | :--- |
| `fork_from` | `(message_id, label=None, initial_content=None, initial_sender="user", initial_sender_type="user", **kw) -> LollmsMessage` |
| `delete_branch` | `(leaf_id, keep_ancestors=True) -> int` — must be a leaf (raises if it has children → use `prune_branch`). |
| `prune_branch` | `(message_id) -> int` — BFS-removes the message **and** all descendants, keeps the parent. |
| `merge_branches` | `(source_leaf_id, target_leaf_id=None, separator_content="--- merged from another branch ---", separator_sender="system") -> LollmsMessage` |
| `set_branch_label` | `(leaf_id, label) -> bool` — stored in leaf `metadata["branch_label"]`. |
| `branch_diff` | `(leaf_a, leaf_b) -> Dict` → `{"common_ancestor_id", "only_in_a", "only_in_b", "shared"}` |

Internal helpers: `_children_map()`, `_collect_leaves(cm)`, `_path_to_root(leaf_id)`.

---

## 7. `regenerate_branch` semantics

`regenerate_branch` (in **`UtilsMixin`**) is the "re-run this AI reply"
operation. Exact prune-and-replay behaviour:

```python
def regenerate_branch(self, branch_tip_id=None, **kwargs):
    self._rebuild_message_index()
    target_id = branch_tip_id if branch_tip_id is not None else self.active_branch_id
    if not target_id or target_id not in self._message_index:
        raise ValueError("Regeneration failed: Target message ID not found.")

    target_msg = self._message_index[target_id]

    if target_msg.sender_type == 'assistant':
        # The user prompt that produced it is the replay origin.
        user_parent_id = target_msg.parent_id
        if user_parent_id is None or user_parent_id not in self._message_index:
            raise ValueError("Regeneration failed: No valid user parent.")
        user_msg_to_regenerate_from = self._message_index[user_parent_id]
        self.remove_message(target_id)               # delete the old AI reply
    elif target_msg.sender_type == 'user':
        user_msg_to_regenerate_from = target_msg     # no deletion
    else:
        raise ValueError(f"Unexpected sender type: '{target_msg.sender_type}'.")

    self.active_branch_id = user_msg_to_regenerate_from.id
    # Re-invoke generation with add_user_message=False so the same user prompt
    # is replayed and a fresh AI leaf is appended (a sibling of the deleted one).
    ...
```

**Step-by-step:**

1. Target = `branch_tip_id` or `active_branch_id`.
2. If the target is an **assistant** message:
   - its parent must be a valid **user** message,
   - the target (old AI reply) is deleted via `remove_message`,
   - the user message becomes the replay origin.
3. If the target is a **user** message: no deletion; it is the origin.
4. `active_branch_id` is set to the user message.
5. Generation is re-invoked with **`add_user_message=False`** and
   `branch_tip_id=<user message id>` (routed to
   `self.lollmsClient.chat(...)` if it exists and accepts `discussion`,
   otherwise to `self.chat(...)`), producing a **new sibling leaf**.

**Orphan nuance:** `remove_message` deletes **only** the single node. Any
descendants of the deleted message are **not** removed — they keep their
`parent_id` and simply become unreachable from `get_branch`. Use
`fix_orphan_messages()` (also `UtilsMixin`) to re-parent orphans onto a primary
root.

---

## 8. Persistence & reload

### 8.1 Creation

`CoreMixin.create_new(cls, lollms_client, db_manager=None, **kwargs)`:

- **DB-backed** (`db_manager` provided): builds a `DiscussionModel` row from
  kwargs **filtered to valid columns**, excluding
  `history_bearing_keys = {"messages", "active_branch_id", "pruning_summary",
  "pruning_point_id"}`. A UUID is generated if no `id`. An existing row with the
  same `id` is expunged and reused.
- **In-memory** (`db_manager=None`): `_create_in_memory_proxy(id=...)`
  initializes `active_branch_id = None`, `messages = []`. With
  `auto_create_db=True` (default), a SQLite DB is auto-created under the
  workspace.

### 8.2 `active_branch_id` on load

- Comes back from the DB row automatically via the `__getattr__` proxy.
- Any branch operation first calls `_rebuild_message_index()`;
  `_validate_and_set_active_branch()` then:
  - if the stored id is `None` or missing → `_find_deepest_leaf(None)`
    (most-recent leaf overall);
  - if the stored id **has children** → descend to the deepest leaf under it.

### 8.3 Deletions are deferred

`remove_message` adds the id to `_messages_to_delete_from_db`; the physical
DB row is removed at the next `commit()` (under `self._db_lock`), after which
the index is rebuilt.

---

## 9. Usage examples

### 9.1 Fork a branch and switch between them

```python
from lollms_client import LollmsClient, LollmsDiscussion

client = LollmsClient()
d = LollmsDiscussion.create_new(lollms_client=client)

d.add_message(sender="user", content="What is 2+2?")
d.add_message(sender="assistant", content="It's 4.")
d.add_message(sender="user", content="And 4*3?")
d.add_message(sender="assistant", content="12.")

# Fork from the first assistant answer:
first_ai = d.get_branch(d.active_branch_id)[1]   # "It's 4."
d.fork_from(first_ai.id, label="alt-history",
            initial_content="Actually, is it 5?")

# Inspect the available branches:
for b in d.list_branches():
    print(b.label, "active" if b.is_active else "", b.depth)

# Switch back to the main branch:
main_leaf = next(b.leaf_id for b in d.list_branches() if b.is_active is False)
d.switch_branch(main_leaf)
```

### 9.2 Regenerate the last AI reply

```python
# Replay the last user prompt to get a fresh AI answer (old one is removed):
d.regenerate_branch()
```

### 9.3 Merge and diff branches

```python
# Merge branch B into branch A:
d.merge_branches(source_leaf_id=leaf_b, target_leaf_id=leaf_a)

# Compare two branches:
diff = d.branch_diff(leaf_a, leaf_b)
print(diff["common_ancestor_id"], len(diff["only_in_a"]), len(diff["only_in_b"]))
```

### 9.4 Prune a whole subtree

```python
# Remove a branch tip AND all its descendants (keep the parent):
d.prune_branch(leaf_to_prune)
```

### 9.5 Build a discussion from a flat message list

```python
msgs = [
    {"sender": "user", "content": "Hi"},
    {"sender": "assistant", "content": "Hello!"},
]
d = LollmsDiscussion.from_messages(msgs, lollms_client=client)
# from_messages chains parent_id so the list becomes a single branch.
```

---

## 10. DAG diagram

```
                    root (parent_id = None)
                    │
                 [U1] user: "What is 2+2?"
                    │
                 [A1] assistant: "It's 4."
                    │
                 [U2] user: "And 4*3?"
                    │
                 [A2] assistant: "12."   ◄── active_branch_id (main branch)

                 (fork from [A1])
                    │
                 [U3] user: "Actually, is it 5?"
                    │
                 [A3] assistant: "No, 4." ◄── leaf of the 'alt-history' branch

Leaves: [A2], [A3]  →  2 branches
```

- The **main branch** is `root → U1 → A1 → U2 → A2`.
- The **alt-history branch** is `root → U1 → A1 → U3 → A3`.
- Both share the prefix up to `[A1]`; they diverge from there.

---

## 11. Cross-references & corrections

- See `src/lollms_client/lollms_discussion/README.md` §"Branching &
  Versioning" for the high-level overview. This document extends it with
  verified ground truth.
- **Correction 1** — The README implies a `BranchMixin` owns all branching
  operations. In reality `BranchMixin` owns *navigation/mutation*;
  `get_branch`, `add_message`, `remove_message`, `active_branch_id`, and branch
  validation live in `CoreMixin`, and `regenerate_branch` /
  `fix_orphan_messages` live in `UtilsMixin`.
- **Correction 2** — Branches are **derived** from `parent_id`; there is no
  dedicated branch table and no `branch_id` field on messages.