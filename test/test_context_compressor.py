"""Unit tests for the pure-heuristic context compressor.

Verifies the guard requirements:
  1. Compression triggers only above the threshold.
  2. File unloading is recency-weighted, not biggest-first.
  3. The newest files and all explicitly-pinned files are preserved, BUT a
     lone oversized file remains evictable.
  4. Images are capped at <= 4, oldest dropped first.
  5. The file lever is BUDGETED so history compression shares the load, but
     the file lever finishes the job when history holds nothing compressible.
  6. `apply_plan` resolves a plan into exact (non-mutating) keep/drop sets.

Boundary discipline: the threshold guard uses a strict `>` comparison, so a
payload total EQUAL to `context_threshold_chars` does NOT trigger. Every
"should trigger" test keeps its total strictly ABOVE the threshold.
"""

try:
    from lollms_client.lollms_discussion._context_compressor import (
        ContextCompressor,
        HistoryMessage,
        ImageRef,
        PinnedFile,
        apply_plan,
    )
except ImportError:  # pragma: no cover - allows static analysis without the package
    ContextCompressor = HistoryMessage = ImageRef = PinnedFile = apply_plan = None  # type: ignore


def _mk_compressor(**over):
    base = dict(
        context_threshold_chars=10_000,
        target_chars=5_000,
        max_images=4,
        keep_recent_files=2,
        keep_recent_messages=3,
    )
    base.update(over)
    return ContextCompressor(**base)


def test_no_compression_under_threshold():
    c = _mk_compressor()
    plan = c.plan(
        pinned_files=[PinnedFile("a.py", 1_000)],
        images=[],
        history=[HistoryMessage(0, "user", 500)],
    )
    assert plan.triggered is False
    assert plan.actions == []


def test_triggers_above_threshold():
    c = _mk_compressor()
    plan = c.plan(
        pinned_files=[PinnedFile("big.py", 20_000, age_rounds=10)],
        images=[],
        history=[],
    )
    assert plan.triggered is True
    assert plan.projected_chars < plan.original_chars


def test_recency_weighted_not_biggest_first():
    """An old small file must be unloaded before a fresh large file.

    Total = 1_000 + 8_000 + 500 + 500 = 10_000. The threshold guard uses a
    strict `>` comparison, so the threshold must sit strictly BELOW the total
    (9_500) for compression to fire.
    """
    c = _mk_compressor(context_threshold_chars=9_500, target_chars=9_000)
    files = [
        PinnedFile("ancient_small.py", 1_000, age_rounds=20),   # old + small
        PinnedFile("fresh_huge.py", 8_000, age_rounds=0),       # new + huge
        PinnedFile("newest_a.py", 500, age_rounds=0),           # kept (newest)
        PinnedFile("newest_b.py", 500, age_rounds=0),           # kept (newest)
    ]
    plan = c.plan(pinned_files=files, images=[], history=[])
    unload = [a for a in plan.actions if a.kind == "unload_file"]
    assert unload, "expected an unload action"
    names = unload[0].detail["names"]
    # The ancient file outranks the fresh huge one.
    assert "ancient_small.py" in names
    assert names.index("ancient_small.py") < (
        names.index("fresh_huge.py") if "fresh_huge.py" in names else 999
    )


def test_never_unload_pinned_or_newest():
    c = _mk_compressor(target_chars=1_000)
    files = [
        PinnedFile("keep_pinned.py", 50_000, age_rounds=99, pinned=True),
        PinnedFile("old_big.py", 8_000, age_rounds=15),
        PinnedFile("fresh.py", 8_000, age_rounds=0),
    ]
    plan = c.plan(pinned_files=files, images=[], history=[])
    unload = [a for a in plan.actions if a.kind == "unload_file"]
    unloaded = set()
    for a in unload:
        unloaded.update(a.detail["names"])
    assert "keep_pinned.py" not in unloaded
    # keep_recent_files=2 → fresh.py is protected even though old_big gets cut.
    assert "fresh.py" not in unloaded


def test_single_oversized_file_still_compresses():
    """A lone oversized file must remain evictable.

    Regression guard: with keep_recent_files=2 and only ONE candidate, an
    unconditional recency guard would protect ALL candidates → zero actions →
    silent no-op. The compressor protects at most len(candidates)-1.
    """
    c = _mk_compressor(keep_recent_files=2, target_chars=5_000)
    plan = c.plan(
        pinned_files=[PinnedFile("solo.py", 30_000, age_rounds=0)],
        images=[],
        history=[],
    )
    unload = [a for a in plan.actions if a.kind == "unload_file"]
    assert unload, "a lone oversized file must be evictable despite the guard"
    assert "solo.py" in unload[0].detail["names"]


def test_image_cap_drops_oldest_first_and_caps_at_4():
    c = _mk_compressor(context_threshold_chars=1_000, target_chars=500, max_images=4)
    images = [ImageRef(f"img{i}", age_rounds=i * 3) for i in range(7)]  # 7 images
    plan = c.plan(pinned_files=[], images=images, history=[])
    cap = [a for a in plan.actions if a.kind == "drop_image"]
    assert cap, "expected an image-cap action"
    dropped = cap[0].detail["drop_names"]
    # 7 images, keep 4 newest ⇒ drop exactly the 3 oldest (img4, img5, img6).
    assert set(dropped) == {"img4", "img5", "img6"}
    for newest in ("img0", "img1", "img2", "img3"):
        assert newest not in dropped


def test_history_compression_preserves_tool_structure_and_recent():
    c = _mk_compressor(
        context_threshold_chars=2_000, target_chars=800, keep_recent_messages=3,
        max_images=4,
    )
    history = [
        HistoryMessage(0, "system", 300),      # preserved (system head)
        HistoryMessage(1, "user", 400),
        HistoryMessage(2, "assistant", 400),
        HistoryMessage(3, "tool", 900, is_tool_structure=True),  # barrier
        HistoryMessage(4, "assistant", 300),
        HistoryMessage(5, "user", 300),        # keep_recent tail start
        HistoryMessage(6, "assistant", 300),
        HistoryMessage(7, "user", 300),
    ]
    plan = c.plan(pinned_files=[], images=[], history=history)
    comp = [a for a in plan.actions if a.kind == "compress_span"]
    assert comp, "expected a history compression action"
    span = comp[0].detail
    # The span must not include the tool-structure message (index 3).
    assert span["end_index"] < 3
    assert span["start_index"] >= 1  # system head preserved


def test_returns_plan_when_nothing_to_touch():
    c = _mk_compressor(context_threshold_chars=100, target_chars=50)
    # Over threshold but everything is pinned/protected → no crash, plan valid.
    plan = c.plan(
        pinned_files=[PinnedFile("only.py", 10_000, pinned=True)],
        images=[],
        history=[],
    )
    assert plan.triggered is True
    # No unload action should target the pinned file.
    for a in plan.actions:
        if a.kind == "unload_file":
            assert "only.py" not in a.detail["names"]


def test_file_lever_is_budgeted_so_history_helps():
    """With both levers available, files must not absorb the whole overflow."""
    c = _mk_compressor(
        context_threshold_chars=10_000,
        target_chars=5_000,
        max_file_unload_ratio=0.5,   # files may reclaim at most half
    )
    files = [
        PinnedFile("stale_a.py", 6_000, age_rounds=30),
        PinnedFile("stale_b.py", 6_000, age_rounds=25),
        PinnedFile("fresh.py", 1_000, age_rounds=0),
        PinnedFile("fresh2.py", 1_000, age_rounds=0),
    ]
    history = [
        HistoryMessage(0, "system", 200),
        HistoryMessage(1, "user", 3_000),      # compressible
        HistoryMessage(2, "assistant", 3_000),  # compressible
        HistoryMessage(3, "user", 400),          # recent tail
        HistoryMessage(4, "assistant", 400),
        HistoryMessage(5, "user", 400),
    ]
    plan = c.plan(pinned_files=files, images=[], history=history)
    unload = [a for a in plan.actions if a.kind == "unload_file"]
    comp = [a for a in plan.actions if a.kind == "compress_span"]
    # Both levers must contribute when the file lever is budgeted.
    assert comp, "history must contribute when the file lever is budgeted"
    assert unload, "file lever should still reclaim its share"


def test_file_lever_completes_job_when_history_empty():
    """If nothing is compressible, the file lever may absorb the full overflow."""
    c = _mk_compressor(
        context_threshold_chars=10_000,
        target_chars=5_000,
        max_file_unload_ratio=0.1,   # absurdly small budget on purpose
    )
    files = [
        PinnedFile("stale.py", 12_000, age_rounds=40),
        PinnedFile("fresh.py", 500, age_rounds=0),
        PinnedFile("fresh2.py", 500, age_rounds=0),
    ]
    plan = c.plan(pinned_files=files, images=[], history=[])
    unload = [a for a in plan.actions if a.kind == "unload_file"]
    assert unload, "file lever must still run when history cannot help"
    assert "stale.py" in unload[0].detail["names"]


def test_apply_plan_resolves_concrete_sets():
    """apply_plan must return exact keep/drop sets without mutating inputs."""
    c = _mk_compressor(
        context_threshold_chars=10_000,
        target_chars=5_000,
        max_file_unload_ratio=1.0,   # let files carry everything here
        keep_recent_files=1,
    )
    files = [
        PinnedFile("old.py", 8_000, age_rounds=20),
        PinnedFile("keep.py", 1_000, age_rounds=0),
        PinnedFile("keep2.py", 1_000, age_rounds=0),
    ]
    images = [ImageRef(f"i{n}", age_rounds=n) for n in range(6)]
    files_before = list(files)
    images_before = list(images)

    plan = c.plan(pinned_files=files, images=images, history=[])
    applied = apply_plan(plan, files, images)

    # Inputs untouched.
    assert files == files_before
    assert images == images_before
    # Old file dropped; remaining kept files come from the original set.
    assert "old.py" in applied.dropped_file_names
    assert all(f.name in {orig.name for orig in files} for f in applied.kept_files)
    # Images capped to 4, oldest (i4, i5) dropped.
    assert len(applied.kept_images) == 4
    assert set(applied.dropped_image_names) == {"i4", "i5"}