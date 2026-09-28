---
name: desktop_automation
title: Desktop UI Automation and Computer Use
category: desktop_automation
tags: [computer_use, automation, screenshot, gui, mouse, keyboard]
required_tools: [tool_computer_screenshot, tool_computer_click]
description: Vision-driven desktop automation loop for inspecting screens, locating UI elements, typing, and clicking.
visibility: loadable
---

# Desktop Automation Skill

## Operating Loop
1. **OBSERVE**: Call `tool_computer_screenshot` to see the current screen. Never act blind.
2. **LOCATE**: Identify coordinates from the visual image or use natural-language element descriptions.
3. **ACT**: Click (`tool_computer_click`), type (`tool_computer_type`), or press hotkeys (`tool_computer_key`). Click into input fields before typing.
4. **VERIFY**: Always capture a follow-up screenshot to confirm the action succeeded before proceeding to the next step.
5. **TERMINATE**: When the target state is reached, describe the outcome and emit `<done/>`.