# `auto_activate_artefacts`

```python
auto_activate_artefacts: bool = True
```

**Category**: Artifacts

**Parent master switch**: `enable_artefacts`. When the master is `False`, this
flag is inert (no artifacts are created, so there is nothing to activate).

**Gate**: none. This flag acts on the context-hydration layer only; it does not
gate any capability.

## What it does

Controls whether newly created or patched artifacts become **active** — their
content is loaded into the model's context (Active Artifacts zone) — the moment
they are written, on the very next round.

This is the "the model can immediately see what it just wrote" behavior that
makes iterative editing (SEARCH/REPLACE hunks against a file the model itself
authored, follow-up reads to verify the write, follow-up patches after a
compiler error) work smoothly.

## What to expect

- `True` (default): after creation/patching, the model can read the artifact's
  content **without** an explicit `<unlock_file>`. The artifact enters
  `FULL` visibility and its content is injected into the round's
  Active Artifacts context block.
- `False`: artifacts are created with `TREE_UNLOCKABLE` visibility — the model
  must `<unlock_file>` them explicitly to read content. Use this when the
  workspace contains **larger artifacts than the model will actually edit**,
  so context budget is never wasted re-seeding the model's own output.

### Interaction with images (`active_images`)

`active_images` is a **message-level** field on `LollmsMessage` (a per-image
list of activation flags), not a `chat()` parameter — see `BRANCHING.md` §3.

Consequences:

- **Images obey their own visibility rules**, independent of
  `auto_activate_artefacts`. A freshly created image artifact defaults to
  `SINGLE_ROUND`: the model sees the pixels on the next round (to verify the
  generation), then the artifact is auto-demoted to `TREE_UNLOCKABLE` and the
  base64 payload is freed.
- Setting `auto_activate_artefacts=False` does **not** prevent image pixels
  from being loaded into context on the round immediately following creation
  — it only withholds **text** artefact content.
- To permanently keep an image in context, the model must emit `<unlock_file>`
  (which sets `FULL`) or pass `visibility="full"` on a subsequent write.
- `suppress_images=True` (negative polarity) is the correct way to withhold
  pixels — this flag never affects image pixels.

### Cost and latency

`auto_activate_artefacts=True` materially increases context size on successive
rounds as artifacts accumulate. With compaction disabled or a high
`context_compaction_threshold`, expect per-round prompt growth proportional to
the sum of active artifact sizes up to that threshold.

Guidelines:

- **Small workspaces** (a handful of small files, model actively editing them):
  leave at `True` — the immediacy is worth the tokens.
- **Large workspaces** (monorepos, generated assets, bulky `.docx`/`.xlsx`/
  `.pptx` artifacts the model is not currently editing): turn it off, or
  per-file `<lock_file>` the bulky ones after creation.
- **`compact_workspace_tree=True`** is a complementary tool — it prevents the
  workspace *tree* from being inlined; `auto_activate_artefacts` prevents
  individual *content* from being inlined. Use both for tight budgets.

### Per-turn vs. per-message granularity

`auto_activate_artefacts` is set **per `chat()` turn** and cannot be toggled
mid-turn. For finer-grained control:

- **Per-file, per-turn**: use `<lock_file>` / `<unlock_file>` tags within the
  conversation. `<lock_file title="big.md" />` immediately demotes that
  artifact to `TREE_LOCKED` regardless of `auto_activate_artefacts`.
- **Per-image**: use `visibility="full"` or nothing (default `SINGLE_ROUND`)
  in the `<generate_image>` / `<edit_image>` tag.
- **Global default per discussion**: pass the flag on every `chat()` call, or
  wrap `discussion.chat` in your host application with your preferred default.

## Usage

Default behavior — most chat and small-authoring workloads:

```python
# No flag needed; auto_activate_artefacts defaults to True.
result = discussion.chat("Write chapter 1 of my novel as a markdown artifact.")
# Author writes chapter1.md; on round 2 the model can read it back directly
# to continue, patch, or reply with excerpts.
```

Large workspace — protect the context budget:

```python
result = discussion.chat(
    """Review the workspace and fix any TODO comments you find. Workspace
    contains several large markdown files. Use <unlock_file> only on the ones
    you actually need to read.""",
    auto_activate_artefacts=False,        # model must <unlock_file> per target
    compact_workspace_tree=True,          # don't inline the recursive tree either
)
```

Mixed pattern (fine-grained per-file):

```python
result = discussion.chat(
    """Refactor `src/auth.py`. `src/legacy_parser.py` is huge — do NOT load it
    into context unless you need it for reference.""",
    auto_activate_artefacts=True,          # default: written files are active
    enable_code_execution=True,
)
# The model writes src/auth.py (auto-active). If it ever touches
# src/legacy_parser.py, it should <lock_file> it immediately afterwards to
# avoid burning tokens on subsequent rounds.
```

## Related flags

- `enable_artefacts` — master switch; `auto_activate_artefacts` is inert when
  the master is `False`.
- `compact_workspace_tree` — complementary context-budget mitigation for the
  workspace *tree*; use with `auto_activate_artefacts=False` on large projects.
- `enable_image_generation` / `enable_image_editing` — image artifacts obey
  their own `SINGLE_ROUND` default visibility, independent of this flag.
- `suppress_images` — the correct negative-polarity flag to withhold image
  pixels; `auto_activate_artefacts` never affects pixels.

## Recovering generated stuff

Recovery is identical either way — the flag only governs whether the model
**sees** content during the turn, not whether the artifact exists:

- `discussion.artefacts.get(title)` — retrieves the artifact (with content).
- `discussion.artefacts.get_version_history(title)` — every version.
- `discussion.artefacts.diff_versions(title, v1, v2)` — inline diffs.
- Turn result `"artefacts"` array — lists what was affected this turn.

For image artifacts, pixels live under `art["images"][0]` (base64 PNG) and the
model anchor is `<artefact_image id="<title>::0" />` in the assistant message
content — see `enable_image_generation` in the master reference.