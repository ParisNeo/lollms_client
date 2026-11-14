import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion._mixin_chat import _StreamState
from lollms_client.lollms_discussion._context_sanitizer import (
    ACTION_DROP,
    ACTION_KEEP,
    ACTION_SUMMARIZE_NAMES,
    ACTION_TRUNCATE_MIDDLE,
    ContextBudgetExceeded,
    ContextBudgetStore,
    ContextSegment,
    PRIORITY_ACTIVE_SKILLS,
    PRIORITY_CURRENT_USER_TURN,
    PRIORITY_LOADED_FILES,
    PRIORITY_SKILL_NAMES,
    PRIORITY_STABLE_PREFIX,
    PRIORITY_STARTUP_ARTIFACTS,
    PRIORITY_TOOL_DESCRIPTIONS,
    PRIORITY_USER_PROFILE,
    SEG_ID_ACTIVE_SKILLS,
    SEG_ID_USER_TURN,  # noqa: SIM222
    SEG_ID_LOADED_FILES,
    SEG_ID_SKILL_NAMES,
    SEG_ID_STABLE_PREFIX,
    SEG_ID_STARTUP_ARTIFACTS,
    SEG_ID_TOOL_DESCRIPTIONS,
    SEG_ID_USER_PROFILE,
    enforce_budget,
    render_budget_header,
    render_segment_tags,
)
from lollms_client.lollms_artefact import ArtefactType, ArtefactVisibility


class MockClient:
    """Mock client for testing artifact operations without an LLM."""

    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock"
        self.binding_name = "mock"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 0

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "ok"

    def reset_cancel(self):
        pass


class TestContextBudgetGuard(unittest.TestCase):
    """Validates the Context Budget Guard and Tool-Generated File Visibility Doctrine."""

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_budget_guard_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="budget_test",
            workspace_path=self.tmp_dir,
            autosave=True,
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ──────────────────────────────────────────────────────────────────────
    # Artifact-explosion guard (pre-existing coverage, preserved)
    # ──────────────────────────────────────────────────────────────────────

    def test_large_tool_generated_file_visibility(self):
        """A tool that generates a >100KB file must not occupy the context."""
        large_content = "A" * 250_000  # >100KB threshold, ~62,500 tokens

        file_size_kb = len(large_content.encode("utf-8")) / 1024
        is_large_file = file_size_kb > 100

        visibility = (
            ArtefactVisibility.TREE_UNLOCKABLE
            if is_large_file
            else ArtefactVisibility.FULL
        )
        active = not is_large_file

        self.discussion.artefacts.add(
            title="large_output.json",
            artefact_type=ArtefactType.DOCUMENT,
            content=large_content,
            active=active,
            visibility=visibility,
        )
        self.discussion.commit()

        retrieved = self.discussion.artefacts.get("large_output.json")
        self.assertIsNotNone(retrieved)
        self.assertEqual(
            retrieved.get("visibility"), ArtefactVisibility.TREE_UNLOCKABLE
        )
        self.assertFalse(retrieved.get("active"))
        expected_tokens = len(large_content) // 4
        self.assertEqual(retrieved.get("token_count", 0), expected_tokens)
        self.assertGreater(expected_tokens, 50_000)

    def test_unlock_file_blocked_for_large_file(self):
        """A >50k-token file stays TREE_UNLOCKABLE when unlocked."""
        large_content = "B" * 250_000  # ~62,500 tokens
        self.discussion.artefacts.add(
            title="huge_data.json",
            artefact_type=ArtefactType.DOCUMENT,
            content=large_content,
            active=False,
            visibility=ArtefactVisibility.TREE_UNLOCKABLE,
        )
        self.discussion.commit()

        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_1")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True,
        )
        ss.feed("\n<unlock_file>\nhuge_data.json\n</unlock_file>\n")
        ss.flush_remaining_buffer()

        retrieved = self.discussion.artefacts.get("huge_data.json")
        self.assertEqual(
            retrieved.get("visibility"), ArtefactVisibility.TREE_UNLOCKABLE
        )
        self.assertFalse(retrieved.get("active"))
        self.assertIn("BLOCKED", ai_msg.content)
        self.assertIn("too large for context", ai_msg.content)
        self.assertIn("huge_data.json", ai_msg.content)
        self.assertTrue(ss.context_unlock_requested)
        self.assertIn("huge_data.json", ss.context_unlocked_files)

    def test_unlock_file_allowed_for_small_file(self):
        """A small file unlocks cleanly into FULL visibility."""
        small_content = "C" * 1000
        self.discussion.artefacts.add(
            title="small_config.txt",
            artefact_type=ArtefactType.DOCUMENT,
            content=small_content,
            active=False,
            visibility=ArtefactVisibility.TREE_UNLOCKABLE,
        )
        self.discussion.commit()

        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_2")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True,
        )
        ss.feed("\n<unlock_file>\nsmall_config.txt\n</unlock_file>\n")
        ss.flush_remaining_buffer()

        retrieved = self.discussion.artefacts.get("small_config.txt")
        self.assertEqual(retrieved.get("visibility"), ArtefactVisibility.FULL)
        self.assertTrue(retrieved.get("active"))
        self.assertNotIn("BLOCKED", ai_msg.content)
        self.assertIn("✅ Unlocking", ai_msg.content)

    def test_context_status_counts_images(self):
        """Images loaded into the discussion feed `get_context_status`."""
        self.client.count_image_tokens = lambda img: 150
        self.discussion.add_discussion_image(
            "base64img" * 100, source="user", active=True
        )
        self.discussion.add_discussion_image(
            "base64img" * 100, source="user", active=True
        )
        self.discussion.add_message(
            sender="User", content="look at this", sender_type="user"
        )
        branch = self.discussion.get_branch(self.discussion.active_branch_id)
        branch[-1].images = [{"data": "base64img" * 100, "active": True}]
        self.discussion.commit()

        status = self.discussion.get_context_status()
        self.assertIn("images", status)
        self.assertEqual(status["images"]["discussion_images"]["count"], 2)
        self.assertGreater(status["images"]["discussion_images"]["tokens"], 0)
        self.assertGreater(status["images"]["total_image_tokens"], 0)
        mh = status["zones"].get("message_history", {})
        self.assertGreaterEqual(mh.get("breakdown", {}).get("image_tokens", 0), 0)

    def test_context_status_without_count_image_tokens(self):
        """A client without `count_image_tokens` must not crash the status."""
        self.client.count_image_tokens = None
        self.discussion.add_discussion_image(
            "x" * 1500, source="user", active=True
        )
        self.discussion.commit()
        status = self.discussion.get_context_status()
        self.assertIn("images", status)
        self.assertEqual(status["images"]["discussion_images"]["count"], 1)
        self.assertGreater(status["images"]["discussion_images"]["tokens"], 0)

    # ──────────────────────────────────────────────────────────────────────
    # Context Budget Guard: segment-level enforcement
    # ──────────────────────────────────────────────────────────────────────

    def _segments(self, *, tool_desc_chars: int = 200,
                  with_artifacts: bool = True) -> list:
        """Builds a realistic zero-state payload mirroring startup_context.log.

        Token model: `_estimate_tokens = len//4` (the deterministic fallback),
        so `chars = tokens * 4`.
        """
        segs = [
            ContextSegment(
                SEG_ID_STABLE_PREFIX, PRIORITY_STABLE_PREFIX,
                "system: " + "S" * 4_000, evictable=False,
            ),
            ContextSegment(
                SEG_ID_USER_TURN, PRIORITY_CURRENT_USER_TURN,
                "hello: " + "U" * 400, evictable=False,
            ),
            ContextSegment(
                SEG_ID_ACTIVE_SKILLS, PRIORITY_ACTIVE_SKILLS, "skill body " * 200,
            ),
            ContextSegment(
                SEG_ID_USER_PROFILE, PRIORITY_USER_PROFILE, "profile " * 100,
            ),
            ContextSegment(
                SEG_ID_LOADED_FILES, PRIORITY_LOADED_FILES, "loaded file " * 300,
            ),
            ContextSegment(
                SEG_ID_TOOL_DESCRIPTIONS, PRIORITY_TOOL_DESCRIPTIONS,
                "• **`tool_x(a: str)`**: description here\n" * 30
                + "T" * tool_desc_chars,
            ),
            ContextSegment(
                SEG_ID_SKILL_NAMES, PRIORITY_SKILL_NAMES,
                "`skill_a`, `skill_b`, `skill_c`",
            ),
        ]
        if with_artifacts:
            segs.append(
                ContextSegment(
                    SEG_ID_STARTUP_ARTIFACTS, PRIORITY_STARTUP_ARTIFACTS,
                    "debug dump " * 500,
                )
            )
        return segs

    def test_zero_state_within_budget(self):
        segs = self._segments()
        result = enforce_budget(segs, budget_tokens=8_000, reserve_output=1_000)

        self.assertIn(result.status, ("OK", "EVICTED"))
        self.assertLessEqual(result.total_tokens, 8_000 - 1_000)
        rendered = result.render()
        self.assertIn("system:", rendered)
        self.assertIn("hello:", rendered)

    def test_priority_eviction_order(self):
        """Lowest-priority segments are evicted first."""
        segs = self._segments()
        result = enforce_budget(segs, budget_tokens=2_000, reserve_output=0)
        by_id = {s.id: s for s in result.segments}

        self.assertIn(result.status, ("OK", "EVICTED"))
        # HARD invariants always survive byte-for-byte.
        self.assertEqual(by_id[SEG_ID_STABLE_PREFIX].content, segs[0].content)
        self.assertEqual(by_id[SEG_ID_USER_TURN].content, segs[1].content)

        # Startup artifacts (lowest priority) go first, active skills (higher
        # priority) go last among the evictable set.
        if result.status == "EVICTED":
            self.assertIn(
                by_id[SEG_ID_STARTUP_ARTIFACTS].action,
                (ACTION_DROP, ACTION_TRUNCATE_MIDDLE),
            )
            dropped = [
                s.id for s in result.segments
                if s.action == ACTION_DROP
            ]
            if SEG_ID_ACTIVE_SKILLS in dropped:
                self.assertIn(SEG_ID_STARTUP_ARTIFACTS, dropped)

    def test_idempotent_second_pass(self):
        """Re-running with the same inputs produces identical output."""
        segs = self._segments()
        first = enforce_budget(segs, budget_tokens=4_000, reserve_output=0)
        second = enforce_budget(segs, budget_tokens=4_000, reserve_output=0)

        self.assertEqual(first.status, second.status)
        self.assertEqual(first.total_tokens, second.total_tokens)
        self.assertEqual(first.render(), second.render())

    def test_db_roundtrip(self):
        """Per-segment rows persist and sum to the run's `total_tokens`."""
        segs = self._segments()
        result = enforce_budget(segs, budget_tokens=8_000, reserve_output=1_000)

        db_dir = Path(self.tmp_dir) / ".lollms_code"
        store = ContextBudgetStore(db_dir / "context_state.db")
        store.record(run_id="run_alpha", sanitized=result)

        import sqlite3
        conn = sqlite3.connect(str(db_dir / "context_state.db"))
        try:
            run_row = conn.execute(
                "SELECT budget, reserve, total_tokens, status FROM runs "
                "WHERE run_id = ?",
                ("run_alpha",),
            ).fetchone()
            self.assertIsNotNone(run_row)
            self.assertEqual(run_row[0], 8_000)
            self.assertEqual(run_row[1], 1_000)

            segment_sum = conn.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM segments "
                "WHERE run_id = ?",
                ("run_alpha",),
            ).fetchone()[0]
            self.assertEqual(segment_sum, run_row[2])

            seg_count = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE run_id = ?",
                ("run_alpha",),
            ).fetchone()[0]
            self.assertEqual(seg_count, len(result.segments))
        finally:
            conn.close()

    def test_db_recording_is_idempotent(self):
        """Recording the same run_id twice leaves exactly one row per segment."""
        segs = self._segments(with_artifacts=False)
        result = enforce_budget(segs, budget_tokens=8_000, reserve_output=0)
        store = ContextBudgetStore(Path(self.tmp_dir) / ".lollms_code" / "context_state.db")
        store.record(run_id="run_beta", sanitized=result)
        store.record(run_id="run_beta", sanitized=result)

        import sqlite3
        conn = sqlite3.connect(str(Path(self.tmp_dir) / ".lollms_code" / "context_state.db"))
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE run_id = ?",
                ("run_beta",),
            ).fetchone()[0]
            self.assertEqual(count, len(result.segments))
        finally:
            conn.close()

    def test_hard_fail_oversized_prefix(self):
        """A prefix that alone exceeds the budget raises, never truncates."""
        oversized = ContextSegment(
            SEG_ID_STABLE_PREFIX, PRIORITY_STABLE_PREFIX,
            "X" * (400 * 100),  # ≈10,000 tokens
            evictable=False,
        )
        user_turn = ContextSegment(
            SEG_ID_USER_TURN, PRIORITY_CURRENT_USER_TURN,
            "hi", evictable=False,
        )
        with self.assertRaises(ContextBudgetExceeded):
            enforce_budget([oversized, user_turn], budget_tokens=1_000, reserve_output=0)

    def test_tool_schema_demotion(self):
        """An oversized tool block becomes names + pointer, never disappears."""
        big_tool_block = "\n".join(
            f"• **`tool_{i}(a: str, b: int)`**: verbose description that will not fit"
            for i in range(200)
        )
        segs = [
            ContextSegment(SEG_ID_STABLE_PREFIX, PRIORITY_STABLE_PREFIX,
                           "prefix", evictable=False),
            ContextSegment(SEG_ID_USER_TURN, PRIORITY_CURRENT_USER_TURN,
                           "hi", evictable=False),
            ContextSegment(SEG_ID_TOOL_DESCRIPTIONS, PRIORITY_TOOL_DESCRIPTIONS,
                           big_tool_block),
        ]
        result = enforce_budget(segs, budget_tokens=1_500, reserve_output=0)
        tool_seg = next(s for s in result.segments if s.id == SEG_ID_TOOL_DESCRIPTIONS)

        self.assertEqual(tool_seg.action, ACTION_SUMMARIZE_NAMES)
        self.assertIn("tool_0", tool_seg.content)
        self.assertIn("tool_199", tool_seg.content)
        self.assertIn("tool_describe_tools", tool_seg.content)
        # Parameter schemas are gone.
        self.assertNotIn("(a: str, b: int)", tool_seg.content)

    def test_render_header_and_tags(self):
        """The log helpers emit the pinned `BUDGET:` + per-segment tag format."""
        segs = self._segments()
        result = enforce_budget(segs, budget_tokens=8_000, reserve_output=1_000)

        header = render_budget_header(result)
        self.assertTrue(header.startswith("BUDGET: total="))
        self.assertIn("/8000", header)
        self.assertIn("reserve=1000", header)
        self.assertIn("status=", header)

        tags = render_segment_tags(result)
        self.assertEqual(len(tags), len(result.segments))
        for tag, seg in zip(tags, result.segments):
            self.assertIn(f"tokens={seg.tokens}", tag)
            self.assertIn(f"priority={seg.priority}", tag)
            self.assertIn(f"action={seg.action}", tag)
            self.assertTrue(tag.endswith(seg.id))


if __name__ == "__main__":
    unittest.main()