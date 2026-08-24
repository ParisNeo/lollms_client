import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from lollms_client import LollmsClient
from lollms_client.lollms_personality import LollmsPersonality, CapabilityFlags
from lollms_client.lollms_types import MSG_TYPE


class LoopStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class TaskProfile:
    goal: str
    success_criteria: str
    allowed_tools: List[str] = field(default_factory=list)
    max_reasoning_steps: int = 50
    timeout_seconds: int = 300
    temperature: float = 0.3
    enable_code_execution: bool = False
    enable_sub_agents: bool = False
    enable_internet: bool = False
    enable_file_ops: bool = True
    workspace_path: Optional[str] = None
    system_prompt_override: Optional[str] = None

    def to_system_prompt(self) -> str:
        tools_str = ", ".join(self.allowed_tools) if self.allowed_tools else "No specific tools assigned. Rely on built-in workspace tools."
        prompt = (
            f"# TASK ASSIGNMENT\n"
            f"## Primary Goal\n{self.goal}\n\n"
            f"## Success Criteria\n{self.success_criteria}\n\n"
            f"## Allowed Tools\n{tools_str}\n\n"
            f"## Operating Constraints\n"
            f"- You are operating in a bounded agentic loop.\n"
            f"- Maximum reasoning steps: {self.max_reasoning_steps}.\n"
            f"- You must verify your progress against the Success Criteria after every action.\n"
            f"- If the Success Criteria are met, you MUST emit `<done/>` immediately.\n"
            f"- If you encounter a blocker that makes the goal impossible, emit `<done/>` and explain the blocker.\n"
            f"- Never ask the user for input; you are fully autonomous.\n"
        )
        if self.system_prompt_override:
            return self.system_prompt_override + "\n\n" + prompt
        return prompt

    def to_capabilities(self) -> CapabilityFlags:
        return CapabilityFlags(
            enable_code_execution=self.enable_code_execution,
            enable_sub_agents=self.enable_sub_agents,
            enable_workspace_tools=self.enable_file_ops,
            enable_skill_creation=True,
            enable_skill_loading=True,
            skills_mode="mixed",
            max_sub_agent_depth=2 if self.enable_sub_agents else 0,
            max_sub_agents_per_turn=3 if self.enable_sub_agents else 0,
        )


@dataclass
class LoopResult:
    status: LoopStatus
    response: str
    rounds: int
    tool_calls: List[Dict[str, Any]]
    workspace_changes: List[Dict[str, Any]]
    elapsed_time: float
    error: Optional[str] = None


class LollmsLoop:
    def __init__(
        self,
        client: LollmsClient,
        task_profile: TaskProfile,
        agent_name: str = "LoopAgent",
        skills_dir: Optional[str] = None,
    ):
        self.client = client
        self.profile = task_profile
        self.status = LoopStatus.PENDING
        self._cancel_flag = False
        self._start_time = 0.0

        self.agent = LollmsPersonality(
            name=agent_name,
            author="LollmsLoops",
            category="autonomous",
            description=f"Autonomous agent for goal: {task_profile.goal[:100]}",
            system_prompt=task_profile.to_system_prompt(),
            lollms_client=self.client,
            workspace_path=task_profile.workspace_path or "./lollms_loops_workspace",
            capabilities=task_profile.to_capabilities(),
            max_tokens_per_turn=8192,
            skills_dirs=[skills_dir] if skills_dir else None,
        )

    def _stream_callback(self, chunk: str, msg_type: MSG_TYPE, metadata: dict) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(chunk, end="", flush=True)
        if self._cancel_flag:
            return False
        return True

    def _check_budget(self, elapsed: float, rounds: int) -> Optional[LoopResult]:
        if self._cancel_flag:
            return LoopResult(
                status=LoopStatus.CANCELLED,
                response="Loop cancelled by user.",
                rounds=rounds,
                tool_calls=[],
                workspace_changes=[],
                elapsed_time=elapsed,
            )

        if rounds >= self.profile.max_reasoning_steps:
            return LoopResult(
                status=LoopStatus.BUDGET_EXHAUSTED,
                response=f"Exhausted maximum reasoning steps ({rounds}).",
                rounds=rounds,
                tool_calls=[],
                workspace_changes=[],
                elapsed_time=elapsed,
            )

        if elapsed >= self.profile.timeout_seconds:
            return LoopResult(
                status=LoopStatus.BUDGET_EXHAUSTED,
                response=f"Exhausted timeout budget ({elapsed:.1f}s).",
                rounds=rounds,
                tool_calls=[],
                workspace_changes=[],
                elapsed_time=elapsed,
            )

        return None

    def run(self) -> LoopResult:
        self.status = LoopStatus.RUNNING
        self._start_time = time.time()
        print(f"\n🚀 Starting Lollms Loop for: {self.profile.goal}")
        print(f"⏱️ Timeout: {self.profile.timeout_seconds}s | Max Steps: {self.profile.max_reasoning_steps}")
        print("=" * 60)

        try:
            tools_dict = {}
            if isinstance(self.profile.allowed_tools, list):
                for tool_name in self.profile.allowed_tools:
                    tools_dict[tool_name] = None

            result = self.agent.chat(
                prompt=f"Begin executing the assigned task. Goal: {self.profile.goal}",
                tools=tools_dict,
                max_reasoning_steps=self.profile.max_reasoning_steps,
                temperature=self.profile.temperature,
                streaming_callback=self._stream_callback,
                use_internal_history=True,
            )

            elapsed = time.time() - self._start_time
            print("\n" + "=" * 60)

            self.status = LoopStatus.SUCCESS
            return LoopResult(
                status=self.status,
                response=result.get("response", ""),
                rounds=result.get("rounds", 0),
                tool_calls=result.get("tool_calls", []),
                workspace_changes=result.get("workspace_changes", []),
                elapsed_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - self._start_time
            self.status = LoopStatus.FAILURE
            return LoopResult(
                status=self.status,
                response="",
                rounds=0,
                tool_calls=[],
                workspace_changes=[],
                elapsed_time=elapsed,
                error=str(e),
            )

    def cancel(self):
        self._cancel_flag = True
        self.agent.cancel_generation()