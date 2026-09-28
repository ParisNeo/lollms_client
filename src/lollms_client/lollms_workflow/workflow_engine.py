"""
workflow_engine.py — Deterministic execution harness for graph-based workflows.
Enforces hard constraints, dynamic model selection for subtasks, condition gating, and error recovery.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors, trace_exception

from .workflow_types import (
    GuardType,
    NodeType,
    Workflow,
    WorkflowContext,
    WorkflowEdge,
    WorkflowNode,
    WorkflowStatus,
)


class GuardViolationError(RuntimeError):
    """Raised when an enforced workflow guard invariant is broken."""
    pass


class WorkflowEngine:
    """
    Orchestrates the execution of a graph workflow around the LLM.
    Handles per-node model selection, variable interpolation, condition branches, and hard walls.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        personality: Optional[Any] = None,
        workspace_path: Optional[Union[str, Path]] = None,
        max_steps: int = 50,
        debug: bool = False,
    ):
        self.client = client
        self.personality = personality
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
        self.max_steps = max_steps
        self.debug = debug

    def _resolve_tool_callable(self, tool_name: str) -> Optional[Callable]:
        """Resolves a tool callable from the personality or client's tool binding."""
        if self.personality and hasattr(self.personality, "_discover_tools"):
            tools = self.personality._discover_tools(enable_workspace_tools=True, enable_shell=True)
            if tool_name in tools and "callable" in tools[tool_name]:
                return tools[tool_name]["callable"]

        if self.client and hasattr(self.client, "tools") and hasattr(self.client.tools, "execute_tool"):
            def _lcp_proxy(**kwargs):
                return self.client.tools.execute_tool(
                    tool_name,
                    kwargs,
                    discussion_instance=getattr(self.personality, "_artefact_proxy", None),
                    lollms_client_instance=self.client
                )
            return _lcp_proxy

        return None

    def _execute_with_model_override(
        self,
        model_override: Optional[str],
        execution_fn: Callable[[], Any]
    ) -> Any:
        """Executes a function under a selected model profile and safely restores the original model."""
        if not model_override or not self.client:
            return execution_fn()

        orig_model_alias = getattr(self.client, "_active_llm_alias", None)
        orig_model_name = getattr(getattr(self.client, "llm", None), "model_name", None)

        target = model_override.strip()
        switched = False

        try:
            if hasattr(self.client, "llm_model_profiles_registry") and target in self.client.llm_model_profiles_registry:
                switched = self.client.switch_model(target)
                if switched and self.debug:
                    ASCIIColors.info(f"[WorkflowEngine] Subtask model switched to profile '{target}'.")

            elif hasattr(self.client, "switch_active_model"):
                switched = self.client.switch_active_model(target)
                if switched and self.debug:
                    ASCIIColors.info(f"[WorkflowEngine] Subtask model switched to '{target}'.")

            return execution_fn()

        finally:
            if switched:
                if orig_model_alias and hasattr(self.client, "switch_model"):
                    self.client.switch_model(orig_model_alias)
                elif orig_model_name and hasattr(self.client, "switch_active_model"):
                    self.client.switch_active_model(orig_model_name)
                if self.debug:
                    ASCIIColors.info(f"[WorkflowEngine] Restored primary model to '{orig_model_alias or orig_model_name}'.")

    def evaluate_guard(self, node: WorkflowNode, context: WorkflowContext) -> Tuple[bool, str]:
        """Evaluates a Guard Wall against active state or workspace files."""
        cfg = node.config
        gtype_raw = cfg.get("guard_type", GuardType.FILE_EXISTS.value)
        try:
            gtype = GuardType(gtype_raw)
        except ValueError:
            gtype = GuardType.FILE_EXISTS

        target_raw = cfg.get("guard_target", "")
        target = context.render_template(target_raw).strip()
        pattern = context.render_template(cfg.get("guard_pattern", "")).strip()

        ws = context.workspace_path or self.workspace_path

        if gtype == GuardType.FILE_EXISTS:
            p = (ws / target).resolve() if not Path(target).is_absolute() else Path(target)
            passed = p.exists()
            return passed, f"File '{target}' exists: {passed}"

        elif gtype == GuardType.FILE_NON_EMPTY:
            p = (ws / target).resolve() if not Path(target).is_absolute() else Path(target)
            passed = p.exists() and p.is_file() and p.stat().st_size > 0
            size = p.stat().st_size if p.exists() else 0
            return passed, f"File '{target}' exists and has content ({size:,} bytes): {passed}"

        elif gtype == GuardType.REGEX_MATCH:
            val = str(context.get(target, target))
            passed = bool(re.search(pattern, val, re.MULTILINE | re.IGNORECASE))
            return passed, f"Regex '{pattern}' match against target: {passed}"

        elif gtype == GuardType.JSON_VALID:
            val = str(context.get(target, target))
            try:
                json.loads(val)
                return True, "JSON format is valid"
            except Exception as j_err:
                return False, f"Invalid JSON syntax: {j_err}"

        elif gtype == GuardType.PYTHON_EXPR:
            expr = pattern or target
            eval_env = {"state": context.state, "re": re, "Path": Path}
            try:
                result = bool(eval(expr, {}, eval_env))
                return result, f"Expression '{expr}' evaluated to {result}"
            except Exception as eval_err:
                return False, f"Expression evaluation failed: {eval_err}"

        elif gtype == GuardType.MAX_ERRORS:
            max_allowed = int(cfg.get("max_errors", 3))
            passed = context.error_count <= max_allowed
            return passed, f"Error count ({context.error_count}) <= {max_allowed}: {passed}"

        return True, "Guard passed"

    def execute_node(
        self,
        node: WorkflowNode,
        context: WorkflowContext,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> Any:
        """Executes a single workflow node according to its typed handler."""
        context.current_node_id = node.id
        node_start_time = time.time()
        model_override = node.config.get("model_name") or node.config.get("model")

        if event_callback:
            event_callback("node_start", node.id, {
                "name": node.name,
                "type": node.node_type.value,
                "model": model_override or "default",
            })

        if node.node_type == NodeType.GUARD:
            passed, explanation = self.evaluate_guard(node, context)
            elapsed = time.time() - node_start_time

            context.history.append({
                "node_id": node.id,
                "node_type": "guard",
                "name": node.name,
                "passed": passed,
                "explanation": explanation,
                "elapsed": elapsed,
            })

            if not passed:
                context.last_error = f"Guard '{node.name}' violated: {explanation}"
                context.error_count += 1
                if event_callback:
                    event_callback("guard_failed", node.id, {"explanation": explanation})
                if node.on_failure:
                    return {"passed": False, "target": node.on_failure, "explanation": explanation}
                raise GuardViolationError(f"Workflow Wall Hit at node '{node.name}': {explanation}")

            if event_callback:
                event_callback("guard_passed", node.id, {"explanation": explanation})
            return {"passed": True, "target": node.on_success, "explanation": explanation}

        elif node.node_type == NodeType.CONDITION:
            cond_expr = node.config.get("condition_expr", "")
            cond_key = node.config.get("condition_key", "")
            eval_env = {"state": context.state, "re": re}

            branch_result = None
            if cond_expr:
                try:
                    branch_result = eval(cond_expr, {}, eval_env)
                except Exception as ex:
                    branch_result = False
                    context.last_error = f"Condition eval error: {ex}"
            elif cond_key:
                branch_result = context.get(cond_key)

            out_var = node.config.get("output_variable")
            if out_var:
                context.set(out_var, branch_result)

            elapsed = time.time() - node_start_time
            context.history.append({
                "node_id": node.id,
                "node_type": "condition",
                "name": node.name,
                "result": branch_result,
                "elapsed": elapsed,
            })
            return branch_result

        elif node.node_type == NodeType.TOOL:
            tool_name = node.config.get("tool_name", "")
            raw_params = node.config.get("tool_params", {})
            params = {}
            for k, v in raw_params.items():
                if isinstance(v, str):
                    params[k] = context.render_template(v)
                else:
                    params[k] = v

            tool_fn = self._resolve_tool_callable(tool_name)
            if not tool_fn:
                raise RuntimeError(f"Tool '{tool_name}' not available for node '{node.name}'.")

            tool_result = tool_fn(**params)

            out_var = node.config.get("output_variable")
            if out_var:
                context.set(out_var, tool_result)

            elapsed = time.time() - node_start_time
            context.history.append({
                "node_id": node.id,
                "node_type": "tool",
                "name": node.name,
                "tool_name": tool_name,
                "params": params,
                "result": tool_result,
                "elapsed": elapsed,
            })
            return tool_result

        elif node.node_type == NodeType.LLM:
            raw_prompt = node.config.get("prompt", "")
            prompt = context.render_template(raw_prompt)
            system_prompt = context.render_template(node.config.get("system_prompt", ""))
            temperature = float(node.config.get("temperature", 0.3))
            n_predict = int(node.config.get("n_predict", 2048))

            def _call_llm():
                if not self.client:
                    raise RuntimeError("LollmsClient not connected to workflow engine.")
                return self.client.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    n_predict=n_predict,
                )

            llm_output = self._execute_with_model_override(model_override, _call_llm)
            clean_output = str(llm_output).strip() if llm_output else ""

            out_var = node.config.get("output_variable")
            if out_var:
                context.set(out_var, clean_output)

            elapsed = time.time() - node_start_time
            context.history.append({
                "node_id": node.id,
                "node_type": "llm",
                "name": node.name,
                "model": model_override or "default",
                "output_preview": clean_output[:200],
                "elapsed": elapsed,
            })
            return clean_output

        elif node.node_type == NodeType.AGENT:
            instruction = context.render_template(node.config.get("instruction", node.config.get("task", "")))
            conditioning = context.render_template(node.config.get("personality_conditioning", node.config.get("persona", "")))
            max_steps = int(node.config.get("max_steps", node.config.get("max_rounds", 6)))
            temperature = float(node.config.get("temperature", 0.3))

            def _call_agent():
                if hasattr(self.personality, "_sub_agent_spawner") and self.personality._sub_agent_spawner:
                    return self.personality._sub_agent_spawner.spawn(
                        instruction=instruction,
                        personality_conditioning=conditioning or None,
                        model_name=model_override,
                        temperature=temperature,
                        max_steps=max_steps,
                    )
                elif self.personality:
                    return self.personality.chat(
                        prompt=instruction,
                        max_reasoning_steps=max_steps,
                        temperature=temperature,
                        use_internal_history=False,
                    )
                raise RuntimeError("Personality not initialized for AGENT node.")

            agent_res = self._execute_with_model_override(model_override, _call_agent)
            out_text = ""
            if isinstance(agent_res, dict):
                out_text = agent_res.get("output") or agent_res.get("response") or ""
            else:
                out_text = str(agent_res)

            out_var = node.config.get("output_variable")
            if out_var:
                context.set(out_var, out_text)

            elapsed = time.time() - node_start_time
            context.history.append({
                "node_id": node.id,
                "node_type": "agent",
                "name": node.name,
                "model": model_override or "default",
                "result": out_text[:200],
                "elapsed": elapsed,
            })
            return agent_res

        elif node.node_type == NodeType.GATE:
            gate_prompt = context.render_template(node.config.get("prompt", f"Approve step '{node.name}'?"))
            context.status = WorkflowStatus.WAITING_APPROVAL
            context.pending_gate_prompt = gate_prompt
            if event_callback:
                event_callback("gate_reached", node.id, {"prompt": gate_prompt})
            return {"gate_pending": True, "prompt": gate_prompt}

        elif node.node_type == NodeType.TERMINAL:
            context.status = WorkflowStatus.COMPLETED
            summary = context.render_template(node.config.get("summary", "Workflow completed successfully."))
            out_var = node.config.get("output_variable")
            if out_var:
                context.set(out_var, summary)
            return summary

        return None

    def find_next_node_id(
        self,
        current_node: WorkflowNode,
        node_result: Any,
        workflow: Workflow,
        context: WorkflowContext,
    ) -> Optional[str]:
        """Resolves the next node ID in the graph based on edges, conditions, or defaults."""
        outgoing = workflow.get_outgoing_edges(current_node.id)

        if current_node.node_type == NodeType.CONDITION:
            for edge in outgoing:
                if edge.condition_value is not None:
                    if edge.condition_value == node_result or str(edge.condition_value).lower() == str(node_result).lower():
                        return edge.target_id
            default_edge = next((e for e in outgoing if e.condition_value is None), None)
            if default_edge:
                return default_edge.target_id
            return current_node.on_success

        if current_node.node_type == NodeType.GUARD:
            if isinstance(node_result, dict):
                if node_result.get("passed"):
                    return node_result.get("target") or (outgoing[0].target_id if outgoing else None)
                else:
                    return node_result.get("target") or current_node.on_failure

        if outgoing:
            return outgoing[0].target_id

        return current_node.on_success

    def run(
        self,
        workflow: Workflow,
        initial_state: Optional[Dict[str, Any]] = None,
        context: Optional[WorkflowContext] = None,
        start_node_id: Optional[str] = None,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> WorkflowContext:
        """
        Executes a workflow graph from start to completion.
        Harness guarantees enforcement of guard conditions and per-subtask model routing.
        """
        if context is None:
            context = WorkflowContext(
                state=dict(initial_state or {}),
                workspace_path=self.workspace_path,
                status=WorkflowStatus.RUNNING,
                active_model=getattr(getattr(self.client, "llm", None), "model_name", "unknown") if self.client else None,
            )
        else:
            context.status = WorkflowStatus.RUNNING
            context.last_error = None

        current_id = start_node_id or context.current_node_id or workflow.start_node_id
        if not current_id:
            context.status = WorkflowStatus.FAILED
            context.last_error = "Workflow has no start node."
            return context

        step_counter = 0

        if event_callback and not start_node_id:
            event_callback("workflow_start", workflow.id, {"name": workflow.name})

        while current_id and step_counter < self.max_steps:
            step_counter += 1
            node = workflow.nodes.get(current_id)
            if not node:
                context.status = WorkflowStatus.FAILED
                context.last_error = f"Node '{current_id}' not found in workflow."
                break

            try:
                node_result = self.execute_node(node, context, event_callback)

                if context.status == WorkflowStatus.WAITING_APPROVAL:
                    break

                if node.node_type == NodeType.TERMINAL:
                    context.status = WorkflowStatus.COMPLETED
                    break

                next_id = self.find_next_node_id(node, node_result, workflow, context)
                if not next_id:
                    context.status = WorkflowStatus.COMPLETED
                    break
                current_id = next_id

            except GuardViolationError as g_err:
                context.status = WorkflowStatus.GUARD_BLOCKED
                context.last_error = str(g_err)
                ASCIIColors.error(f"[WorkflowEngine] 🛑 {g_err}")
                break
            except Exception as ex:
                trace_exception(ex)
                context.status = WorkflowStatus.FAILED
                context.last_error = f"Exception at node '{node.name}': {ex}"
                break

        if step_counter >= self.max_steps:
            context.status = WorkflowStatus.FAILED
            context.last_error = f"Exceeded maximum workflow steps ({self.max_steps})."

        if event_callback and context.status != WorkflowStatus.WAITING_APPROVAL:
            event_callback("workflow_end", workflow.id, {
                "status": context.status.value,
                "steps": step_counter,
                "error": context.last_error,
            })

        return context

    def resume(
        self,
        workflow: Workflow,
        context: WorkflowContext,
        approved: bool = True,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> WorkflowContext:
        """Resumes a workflow paused at an approval gate node."""
        if context.status != WorkflowStatus.WAITING_APPROVAL:
            return context

        gate_node = workflow.nodes.get(context.current_node_id)
        if not gate_node:
            context.status = WorkflowStatus.FAILED
            context.last_error = f"Gate node '{context.current_node_id}' not found."
            return context

        if approved:
            next_id = self.find_next_node_id(gate_node, {"gate_approved": True, "approved": True}, workflow, context)
            if not next_id:
                next_id = gate_node.on_success
            context.status = WorkflowStatus.RUNNING
            context.pending_gate_prompt = None
            if event_callback:
                event_callback("gate_approved", gate_node.id, {"approved": True})
            return self.run(workflow, context=context, start_node_id=next_id, event_callback=event_callback)
        else:
            if event_callback:
                event_callback("gate_rejected", gate_node.id, {"approved": False})
            if gate_node.on_failure:
                context.status = WorkflowStatus.RUNNING
                context.pending_gate_prompt = None
                return self.run(workflow, context=context, start_node_id=gate_node.on_failure, event_callback=event_callback)
            else:
                context.status = WorkflowStatus.FAILED
                context.last_error = f"Operator rejected approval gate '{gate_node.name}'."
                context.pending_gate_prompt = None
                return context


# ── Project Workflow Persistence & Discovery ─────────────────────────────────

def get_project_workflows_dir(workspace_path: Union[str, Path]) -> Path:
    ws_dir = Path(workspace_path).resolve() / ".lollms_code" / "workflows"
    ws_dir.mkdir(parents=True, exist_ok=True)
    return ws_dir


def list_project_workflows(workspace_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Returns all project-saved workflows and built-in templates."""
    results: List[Dict[str, Any]] = []

    # 1. Project workflows saved in .lollms_code/workflows/
    wf_dir = get_project_workflows_dir(workspace_path)
    for f in sorted(wf_dir.glob("*.yaml")) + sorted(wf_dir.glob("*.yml")):
        try:
            content = f.read_text(encoding="utf-8")
            wf = Workflow.from_yaml(content)
            results.append({
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "source": "project",
                "file_path": str(f.resolve()),
                "workflow": wf,
                "nodes_count": len(wf.nodes),
            })
        except Exception as ex:
            ASCIIColors.warning(f"Could not load project workflow from {f.name}: {ex}")

    # 2. Built-in Templates
    templates = [
        create_file_organizer_workflow(),
        create_dual_model_review_workflow(),
    ]
    for tmpl in templates:
        results.append({
            "id": tmpl.id,
            "name": tmpl.name,
            "description": tmpl.description,
            "source": "template",
            "file_path": None,
            "workflow": tmpl,
            "nodes_count": len(tmpl.nodes),
        })

    return results


def save_project_workflow(workspace_path: Union[str, Path], workflow: Workflow) -> Path:
    """Saves a workflow to the project directory (.lollms_code/workflows/{id}.yaml)."""
    wf_dir = get_project_workflows_dir(workspace_path)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", workflow.id.strip().lower()) or "workflow"
    target_file = wf_dir / f"{safe_name}.yaml"
    target_file.write_text(workflow.to_yaml(), encoding="utf-8")
    return target_file


def delete_project_workflow(workspace_path: Union[str, Path], workflow_id: str) -> bool:
    """Deletes a custom project workflow file."""
    wf_dir = get_project_workflows_dir(workspace_path)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", workflow_id.strip().lower())
    for ext in (".yaml", ".yml", ".json"):
        candidate = wf_dir / f"{safe_name}{ext}"
        if candidate.exists():
            candidate.unlink()
            return True
    return False


def load_project_workflow(workspace_path: Union[str, Path], workflow_id_or_path: str) -> Optional[Workflow]:
    """Loads a workflow by ID or file path from project or templates."""
    p = Path(workflow_id_or_path)
    if p.exists() and p.is_file():
        try:
            return Workflow.from_yaml(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    for item in list_project_workflows(workspace_path):
        if item["id"] == workflow_id_or_path:
            return item["workflow"]
    return None


# ── Built-in Grounded Workflows ──────────────────────────────────────────────

def create_file_organizer_workflow() -> Workflow:
    """
    Constructs a deterministic, hard-gated workflow for file reorganization.
    Enforces that mapping.yaml MUST be created and verified before migration can execute.
    """
    wf = Workflow(
        id="file_organizer_guarded",
        name="Guarded File Organizer (4-Phase Wall)",
        description="Grounded 4-step file reorganization with plan verification wall.",
    )

    wf.add_node(WorkflowNode(
        id="step_plan",
        name="Generate Migration Plan",
        node_type=NodeType.LLM,
        config={
            "prompt": "Inspect the workspace files. Create a clean mapping.yaml file for up to 50 items using <artifact name=\"mapping.yaml\" type=\"document\">...",
            "system_prompt": "You are a file reorganization planner. Output ONLY the <artifact name=\"mapping.yaml\"> tag.",
            "output_variable": "plan_generation_output",
        },
    ))

    wf.add_node(WorkflowNode(
        id="guard_plan_exists",
        name="Guard: Verify mapping.yaml Exists",
        node_type=NodeType.GUARD,
        config={
            "guard_type": GuardType.FILE_NON_EMPTY.value,
            "guard_target": "mapping.yaml",
        },
        on_failure="step_plan",
    ))

    wf.add_node(WorkflowNode(
        id="gate_user_approval",
        name="Operator Approval Gate",
        node_type=NodeType.GATE,
        config={
            "prompt": "mapping.yaml has been verified on disk. Do you approve executing the migration?",
        },
    ))

    wf.add_node(WorkflowNode(
        id="step_execute_migration",
        name="Execute Migration via Tool",
        node_type=NodeType.TOOL,
        config={
            "tool_name": "tool_organize_files_from_plan",
            "tool_params": {
                "plan_file": "mapping.yaml",
                "move_files": True,
            },
            "output_variable": "migration_report",
        },
    ))

    wf.add_node(WorkflowNode(
        id="step_done",
        name="Finished Migration",
        node_type=NodeType.TERMINAL,
        config={
            "summary": "File migration finished successfully.\nReport:\n{{migration_report}}",
            "output_variable": "final_result",
        },
    ))

    wf.add_edge("step_plan", "guard_plan_exists")
    wf.add_edge("guard_plan_exists", "gate_user_approval")
    wf.add_edge("gate_user_approval", "step_execute_migration")
    wf.add_edge("step_execute_migration", "step_done")

    return wf


def create_dual_model_review_workflow(coder_model: str = "deep_coder", reviewer_model: str = "gpt4o") -> Workflow:
    """
    Constructs a dual-model engineering workflow:
    1. Coder Model implements the solution in an artifact.
    2. Guard Wall verifies tests pass.
    3. Reviewer Model (frontier/different model) audits the code for vulnerabilities.
    """
    wf = Workflow(
        id="dual_model_coder_review",
        name="Dual-Model Code & Audit Pipeline",
        description="Implements code using a specialist coder model, verifies execution, and audits using a reviewer model.",
    )

    wf.add_node(WorkflowNode(
        id="node_code",
        name="Implement Code",
        node_type=NodeType.AGENT,
        config={
            "instruction": "{{task_prompt}}",
            "model_name": coder_model,
            "max_steps": 6,
            "output_variable": "code_output",
        },
    ))

    wf.add_node(WorkflowNode(
        id="guard_syntax",
        name="Guard: Assert Implementation Exists",
        node_type=NodeType.GUARD,
        config={
            "guard_type": GuardType.PYTHON_EXPR.value,
            "guard_pattern": "len(state.get('code_output', '')) > 20",
        },
        on_failure="node_code",
    ))

    wf.add_node(WorkflowNode(
        id="node_review",
        name="Audit Code with Reviewer Model",
        node_type=NodeType.LLM,
        config={
            "model_name": reviewer_model,
            "prompt": "Review the following implementation for security, edge cases, and compliance:\n\n{{code_output}}\n\nProvide an executive audit summary.",
            "output_variable": "review_audit",
        },
    ))

    wf.add_node(WorkflowNode(
        id="node_finish",
        name="Final Result",
        node_type=NodeType.TERMINAL,
        config={
            "summary": "# Dual-Model Execution Completed\n\n## Implementation:\n{{code_output}}\n\n## Security Audit (Model: " + reviewer_model + "):\n{{review_audit}}",
            "output_variable": "final_summary",
        },
    ))

    wf.add_edge("node_code", "guard_syntax")
    wf.add_edge("guard_syntax", "node_review")
    wf.add_edge("node_review", "node_finish")

    return wf


__all__ = [
    "GuardViolationError",
    "WorkflowEngine",
    "create_file_organizer_workflow",
    "create_dual_model_review_workflow",
    "get_project_workflows_dir",
    "list_project_workflows",
    "save_project_workflow",
    "delete_project_workflow",
    "load_project_workflow",
]