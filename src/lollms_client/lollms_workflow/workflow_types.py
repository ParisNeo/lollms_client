"""
workflow_types.py — Type definitions and data schemas for graph-based workflows.
Enforces deterministic gating, per-node model selection, branching logic, and guardrails.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None


class NodeType(str, Enum):
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    GUARD = "guard"
    GATE = "gate"
    TERMINAL = "terminal"


class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    GUARD_BLOCKED = "guard_blocked"


class GuardType(str, Enum):
    FILE_EXISTS = "file_exists"
    FILE_NON_EMPTY = "file_non_empty"
    REGEX_MATCH = "regex_match"
    JSON_VALID = "json_valid"
    PYTHON_EXPR = "python_expr"
    MAX_ERRORS = "max_errors"


@dataclass
class WorkflowNode:
    """Represents a discrete executable step in a graph-based workflow."""
    id: str
    name: str
    node_type: NodeType
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    on_success: Optional[str] = None
    on_failure: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value if isinstance(self.node_type, NodeType) else str(self.node_type),
            "description": self.description,
            "config": self.config,
            "on_success": self.on_success,
            "on_failure": self.on_failure,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        raw_type = data.get("node_type", "llm").lower()
        try:
            ntype = NodeType(raw_type)
        except ValueError:
            ntype = NodeType.LLM
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            node_type=ntype,
            description=data.get("description", ""),
            config=data.get("config", {}),
            on_success=data.get("on_success"),
            on_failure=data.get("on_failure"),
        )


@dataclass
class WorkflowEdge:
    """Represents a directed transition between two workflow nodes."""
    source_id: str
    target_id: str
    condition_value: Optional[Any] = None
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "condition_value": self.condition_value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            condition_value=data.get("condition_value"),
            label=data.get("label", ""),
        )


@dataclass
class WorkflowContext:
    """Runtime execution state shared across nodes during a workflow run."""
    state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_node_id: Optional[str] = None
    workspace_path: Optional[Path] = None
    error_count: int = 0
    last_error: Optional[str] = None
    active_model: Optional[str] = None
    pending_gate_prompt: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def render_template(self, template: str) -> str:
        """Interpolates {{variable_name}} tokens from the active workflow state."""
        if not template or not isinstance(template, str):
            return ""

        def replacer(match: re.Match) -> str:
            var_name = match.group(1).strip()
            val = self.state.get(var_name, "")
            if isinstance(val, (dict, list)):
                return json.dumps(val, indent=2, ensure_ascii=False)
            return str(val)

        return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", replacer, template)


@dataclass
class Workflow:
    """Complete graph specification for an executable workflow."""
    id: str
    name: str
    description: str = ""
    start_node_id: str = ""
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: List[WorkflowEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: WorkflowNode) -> "Workflow":
        self.nodes[node.id] = node
        if not self.start_node_id:
            self.start_node_id = node.id
        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        condition_value: Optional[Any] = None,
        label: str = ""
    ) -> "Workflow":
        self.edges.append(WorkflowEdge(
            source_id=source_id,
            target_id=target_id,
            condition_value=condition_value,
            label=label or str(condition_value or "")
        ))
        return self

    def get_outgoing_edges(self, node_id: str) -> List[WorkflowEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "start_node_id": self.start_node_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        nodes = {}
        for nid, ndata in data.get("nodes", {}).items():
            if isinstance(ndata, dict):
                ndata.setdefault("id", nid)
                nodes[nid] = WorkflowNode.from_dict(ndata)

        edges = [WorkflowEdge.from_dict(edata) for edata in data.get("edges", [])]

        return cls(
            id=data.get("id", "workflow"),
            name=data.get("name", "Workflow"),
            description=data.get("description", ""),
            start_node_id=data.get("start_node_id", next(iter(nodes.keys())) if nodes else ""),
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {}),
        )

    def compute_auto_layout(
        self,
        start_x: int = 80,
        start_y: int = 140,
        spacing_x: int = 340,
        spacing_y: int = 220,
    ) -> Dict[str, Tuple[int, int]]:
        """Computes clean topological (x, y) coordinates for all nodes in the graph."""
        if not self.nodes:
            return {}

        depths: Dict[str, int] = {}
        visited: Set[str] = set()

        def _assign_depth(node_id: str, current_depth: int):
            if node_id in visited and depths.get(node_id, 0) >= current_depth:
                return
            visited.add(node_id)
            depths[node_id] = max(depths.get(node_id, 0), current_depth)

            outgoing = self.get_outgoing_edges(node_id)
            for edge in outgoing:
                _assign_depth(edge.target_id, current_depth + 1)

            node = self.nodes.get(node_id)
            if node and node.on_success:
                _assign_depth(node.on_success, current_depth + 1)

        start_id = self.start_node_id or next(iter(self.nodes.keys()))
        _assign_depth(start_id, 0)

        # Catch disconnected nodes
        for nid in self.nodes:
            if nid not in depths:
                depths[nid] = max(depths.values(), default=0) + 1

        # Group by depth column
        columns: Dict[int, List[str]] = {}
        for nid, d in depths.items():
            columns.setdefault(d, []).append(nid)

        positions: Dict[str, Tuple[int, int]] = {}
        for col_idx, col_nodes in columns.items():
            for row_idx, nid in enumerate(col_nodes):
                x = start_x + (col_idx * spacing_x)
                y = start_y + (row_idx * spacing_y)
                positions[nid] = (x, y)
                if nid in self.nodes:
                    self.nodes[nid].config.setdefault("ui_pos", {"x": x, "y": y})

        return positions

    def to_yaml(self) -> str:
        if not yaml:
            return json.dumps(self.to_dict(), indent=2)
        return yaml.dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "Workflow":
        if yaml:
            data = yaml.safe_load(yaml_content) or {}
        else:
            data = json.loads(yaml_content)
        return cls.from_dict(data)


__all__ = [
    "NodeType",
    "WorkflowStatus",
    "GuardType",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowContext",
    "Workflow",
]