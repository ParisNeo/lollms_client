import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion._context_sanitizer import (
    compress_artifacts_to_anchors,
    scrub_processing_blocks,
    sanitize_context_for_llm,
    build_anti_mimicry_directives
)

class TestContextDietProtocol(unittest.TestCase):
    """Tests for the Context Diet Protocol (Artifact compression and log scrubbing)."""

    def test_compress_full_artifact_to_anchor(self):
        """Verify that a large artifact block is replaced by a compact anchor."""
        text = (
            "Here is the code:\n"
            '<artifact name="main.py" type="code" language="python" version="2">\n'
            "import os\nprint('hello')\n"
            "</artifact>\n"
            "Let me know if you need changes."
        )
        sanitized = compress_artifacts_to_anchors(text)

        self.assertNotIn("import os", sanitized)
        self.assertNotIn("<artifact", sanitized)
        self.assertIn("[🔒SYSTEM_ARTIFACT_ANCHOR:main.py]", sanitized)
        self.assertIn("Here is the code:", sanitized)
        self.assertIn("Let me know if you need changes.", sanitized)

    def test_scrub_processing_blocks(self):
        """Verify that <processing> execution logs are removed."""
        text = (
            "Calling tool...\n"
            '<processing type="tool" title="Execution">\n'
            "* Running command...\n<!-- status:success -->\n"
            "</processing>\n"
            "Done."
        )
        sanitized = scrub_processing_blocks(text)
        
        self.assertNotIn("<processing", sanitized)
        self.assertNotIn("status:success", sanitized)
        self.assertNotIn("Running command", sanitized)
        self.assertIn("Calling tool...", sanitized)
        self.assertIn("Done.", sanitized)

    def test_scrub_orphaned_processing_block(self):
        """Verify that unclosed <processing> blocks are scrubbed to the end of string."""
        text = "Start\n<processing type=\"artefact\">\nBuilding..."
        sanitized = scrub_processing_blocks(text)
        self.assertNotIn("<processing", sanitized)
        self.assertNotIn("Building...", sanitized)
        self.assertIn("Start", sanitized)

    def test_full_sanitization_pipeline(self):
        """Verify the combined sanitize_context_for_llm function."""
        text = (
            '<artifact name="data.csv" type="data">a,b,c\n1,2,3</artifact>\n'
            '<processing type="tool">Running...</processing>\n'
            '<artefact_image id="img::0" />'
        )
        sanitized = sanitize_context_for_llm(text)

        self.assertIn("[🔒SYSTEM_ARTIFACT_ANCHOR:data.csv]", sanitized)
        self.assertNotIn("a,b,c", sanitized)
        self.assertNotIn("<processing", sanitized)
        self.assertNotIn("Running...", sanitized)
        self.assertNotIn("<artefact_image", sanitized)

    def test_anti_mimicry_directives_content(self):
        """Verify the anti-mimicry directives contain critical instructions."""
        directives = build_anti_mimicry_directives()

        self.assertIn("NEVER OUTPUT SYSTEM MARKERS", directives)
        self.assertIn("[🔒SYSTEM_ARTIFACT_ANCHOR:", directives)
        self.assertIn("[🔒SYSTEM_TOOL_EXECUTED:", directives)
        self.assertIn("USE REAL TAGS", directives)
        self.assertIn("<artifact name=", directives)

if __name__ == "__main__":
    unittest.main()
