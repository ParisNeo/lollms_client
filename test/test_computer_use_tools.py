"""
test_computer_use_tools.py
Verifies that the Computer Use desktop automation toolset:
1. Only mounts when allow_computer_use (or enable_computer_use) is True AND the running model supports vision.
2. Refuses to mount when allow_computer_use is False.
3. Refuses to mount when allow_computer_use is True but the running model is text-only (lacks vision).
4. Executes primitives safely with a mocked backend.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_personality.lollms_personality import LollmsPersonality, CapabilityFlags
import lollms_client.tools_bindings.lcp.default_tools.computer_use.computer_use as cu


class TestComputerUseGating(unittest.TestCase):

    def setUp(self):
        # Mock vision-capable client
        self.mock_vision_client = MagicMock()
        self.mock_vision_client.has_vision_capability.return_value = True
        self.mock_vision_client.llm.vision_enabled = True

        # Mock text-only client (no vision)
        self.mock_text_client = MagicMock()
        self.mock_text_client.has_vision_capability.return_value = False
        self.mock_text_client.llm.vision_enabled = False

    def test_computer_use_mounts_when_allowed_and_model_has_vision(self):
        """When allow_computer_use=True and model has vision, toolset must be discovered."""
        caps = CapabilityFlags(allow_computer_use=True)
        pers = LollmsPersonality(
            name="DesktopAgent",
            system_prompt="Test desktop agent",
            lollms_client=self.mock_vision_client,
            capabilities=caps,
        )

        tools = pers._discover_tools(
            allow_computer_use=True,
            enable_workspace_tools=False,
        )

        # Must contain computer use primitives
        self.assertIn("tool_computer_desktop_info", tools)
        self.assertIn("tool_computer_screenshot", tools)
        self.assertIn("tool_computer_click", tools)
        self.assertIn("tool_computer_move_cursor", tools)
        self.assertIn("tool_computer_mouse_down", tools)
        self.assertIn("tool_computer_mouse_up", tools)
        self.assertIn("tool_computer_drag", tools)
        self.assertIn("tool_computer_type", tools)
        self.assertIn("tool_computer_key", tools)
        self.assertIn("tool_computer_scroll", tools)
        self.assertIn("tool_computer_wait", tools)
        self.assertIn("tool_computer_cursor_position", tools)

    def test_computer_use_withheld_when_allow_computer_use_is_false(self):
        """When allow_computer_use=False, toolset must not be loaded even if model has vision."""
        caps = CapabilityFlags(allow_computer_use=False)
        pers = LollmsPersonality(
            name="DesktopAgent",
            system_prompt="Test desktop agent",
            lollms_client=self.mock_vision_client,
            capabilities=caps,
        )

        tools = pers._discover_tools(
            allow_computer_use=False,
            enable_computer_use=False,
            enable_workspace_tools=False,
        )

        self.assertNotIn("tool_computer_screenshot", tools)
        self.assertNotIn("tool_computer_click", tools)
        self.assertNotIn("tool_computer_desktop_info", tools)

    def test_computer_use_withheld_when_model_lacks_vision(self):
        """When allow_computer_use=True but model has NO vision, toolset must NOT be loaded."""
        caps = CapabilityFlags(allow_computer_use=True)
        pers = LollmsPersonality(
            name="DesktopAgent",
            system_prompt="Test desktop agent",
            lollms_client=self.mock_text_client,
            capabilities=caps,
        )

        tools = pers._discover_tools(
            allow_computer_use=True,
            enable_workspace_tools=False,
        )

        self.assertNotIn("tool_computer_screenshot", tools)
        self.assertNotIn("tool_computer_click", tools)
        self.assertNotIn("tool_computer_desktop_info", tools)


class TestComputerUsePrimitives(unittest.TestCase):

    def setUp(self):
        self.mock_pyautogui = MagicMock()
        self.mock_pyautogui.size.return_value = (1920, 1080)
        self.mock_pyautogui.position.return_value = (500, 300)

        # Mock PIL image for screenshot
        mock_pil = MagicMock()
        mock_pil.convert.return_value = mock_pil
        self.mock_pyautogui.screenshot.return_value = mock_pil

        # Inject into computer_use backend cache
        cu._BACKEND = self.mock_pyautogui
        cu._BACKEND_ERROR = None

    def tearDown(self):
        cu._BACKEND = None
        cu._BACKEND_ERROR = None

    def test_desktop_info(self):
        res = cu.tool_computer_desktop_info()
        self.assertTrue(res["success"])
        self.assertEqual(res["screen_width"], 1920)
        self.assertEqual(res["screen_height"], 1080)

    def test_click_coordinates(self):
        res = cu.tool_computer_click(x=100, y=200, button="left", clicks=1)
        self.assertTrue(res["success"])
        self.mock_pyautogui.click.assert_called_with(x=100, y=200, clicks=1, button="left")

    def test_mouse_down_and_up(self):
        res_down = cu.tool_computer_mouse_down(button="left", x=150, y=250)
        self.assertTrue(res_down["success"])
        self.mock_pyautogui.mouseDown.assert_called_with(x=150, y=250, button="left")

        res_up = cu.tool_computer_mouse_up(button="left")
        self.assertTrue(res_up["success"])
        self.mock_pyautogui.mouseUp.assert_called_with(button="left")

    def test_drag(self):
        res = cu.tool_computer_drag(from_x=100, from_y=100, to_x=400, to_y=500, duration_ms=200)
        self.assertTrue(res["success"])
        self.mock_pyautogui.moveTo.assert_called_with(100, 100)
        self.mock_pyautogui.dragTo.assert_called_with(400, 500, duration=0.2, button="left")

    def test_type_and_key(self):
        res_type = cu.tool_computer_type(text="Hello world")
        self.assertTrue(res_type["success"])
        self.mock_pyautogui.typewrite.assert_called()

        res_key = cu.tool_computer_key(key="ctrl+c")
        self.assertTrue(res_key["success"])
        self.mock_pyautogui.hotkey.assert_called_with("ctrl", "c")

    def test_wait(self):
        res_wait = cu.tool_computer_wait(duration_s=0.1)
        self.assertTrue(res_wait["success"])
        self.assertAlmostEqual(res_wait["duration_s"], 0.1)

    def test_cursor_position(self):
        res_pos = cu.tool_computer_cursor_position()
        self.assertTrue(res_pos["success"])
        self.assertEqual(res_pos["x"], 500)
        self.assertEqual(res_pos["y"], 300)


if __name__ == "__main__":
    unittest.main()