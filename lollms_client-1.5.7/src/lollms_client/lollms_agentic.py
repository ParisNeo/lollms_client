import json
import re
import uuid
import base64
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class SubTask:
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict] = None
    confidence: float = 0.0
    tools_required: List[str] = field(default_factory=list)
    estimated_complexity: int = 1  # 1-5 scale

@dataclass
class ExecutionPlan:
    tasks: List[SubTask]
    total_estimated_steps: int
    execution_order: List[str]
    fallback_strategies: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class MemoryEntry:
    timestamp: float
    context: str
    action: str
    result: Dict
    confidence: float
    success: bool
    user_feedback: Optional[str] = None

@dataclass
class ToolPerformance:
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    total_calls: int = 0
    avg_response_time: float = 0.0
    last_used: float = 0.0
    failure_patterns: List[str] = field(default_factory=list)

class TaskPlanner:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def decompose_task(self, user_request: str, context: str = "") -> ExecutionPlan:
        """Break down complex requests into manageable subtasks"""
        decomposition_prompt = f"""
Analyze this user request and break it down into specific, actionable subtasks:

USER REQUEST: "{user_request}"
CONTEXT: {context}

Create a JSON plan with subtasks that are:
1. Specific and actionable
2. Have clear success criteria  
3. Include estimated complexity (1-5 scale)
4. List required tool types

Output format:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "specific action to take",
      "dependencies": ["task_id"],
      "estimated_complexity": 2,
      "tools_required": ["tool_type"]
    }}
  ],
  "execution_strategy": "sequential|parallel|hybrid"
}}
"""
        
        try:
            plan_data = self.llm_client.generate_structured_content(
                prompt=decomposition_prompt,
                schema={"tasks": "array", "execution_strategy": "string"},
                temperature=0.3
            )
            
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task = SubTask(
                    id=task_data.get("id", str(uuid.uuid4())),
                    description=task_data.get("description", ""),
                    dependencies=task_data.get("dependencies", []),
                    estimated_complexity=task_data.get("estimated_complexity", 1),
                    tools_required=task_data.get("tools_required", [])
                )
                tasks.append(task)
            
            execution_order = self._calculate_execution_order(tasks)
            total_steps = sum(task.estimated_complexity for task in tasks)
            
            return ExecutionPlan(
                tasks=tasks,
                total_estimated_steps=total_steps,
                execution_order=execution_order
            )
            
        except Exception as e:
            # Fallback: create single task
            single_task = SubTask(
                id="fallback_task",
                description=user_request,
                estimated_complexity=3
            )
            return ExecutionPlan(
                tasks=[single_task],
                total_estimated_steps=3,
                execution_order=["fallback_task"]
            )
    
    def _calculate_execution_order(self, tasks: List[SubTask]) -> List[str]:
        """Calculate optimal execution order based on dependencies"""
        task_map = {task.id: task for task in tasks}
        executed = set()
        order = []
        
        def can_execute(task_id: str) -> bool:
            task = task_map[task_id]
            return all(dep in executed for dep in task.dependencies)
        
        while len(order) < len(tasks):
            ready_tasks = [tid for tid in task_map.keys() 
                          if tid not in executed and can_execute(tid)]
            
            if not ready_tasks:
                # Handle circular dependencies - execute remaining tasks
                remaining = [tid for tid in task_map.keys() if tid not in executed]
                ready_tasks = remaining[:1] if remaining else []
            
            # Sort by complexity (simpler tasks first)
            ready_tasks.sort(key=lambda tid: task_map[tid].estimated_complexity)
            
            for task_id in ready_tasks:
                order.append(task_id)
                executed.add(task_id)
                
        return order

class MemoryManager:
    def __init__(self, max_entries: int = 1000):
        self.memory: List[MemoryEntry] = []
        self.max_entries = max_entries
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, float] = {}
        
    def add_memory(self, context: str, action: str, result: Dict, 
                   confidence: float, success: bool, user_feedback: str = None):
        """Add a new memory entry"""
        entry = MemoryEntry(
            timestamp=time.time(),
            context=context,
            action=action,
            result=result,
            confidence=confidence,
            success=success,
            user_feedback=user_feedback
        )
        
        self.memory.append(entry)
        
        # Prune old memories
        if len(self.memory) > self.max_entries:
            self.memory = self.memory[-self.max_entries:]
    
    def get_relevant_patterns(self, current_context: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant past experiences"""
        # Simple similarity scoring based on context overlap
        scored_memories = []
        current_words = set(current_context.lower().split())
        
        for memory in self.memory:
            memory_words = set(memory.context.lower().split())
            overlap = len(current_words & memory_words)
            if overlap > 0:
                score = overlap / max(len(current_words), len(memory_words))
                scored_memories.append((score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    def compress_scratchpad(self, scratchpad: str, current_goal: str, 
                           max_length: int = 8000) -> str:
        """Intelligently compress scratchpad while preserving key insights"""
        if len(scratchpad) <= max_length:
            return scratchpad
        
        # Extract key sections
        sections = re.split(r'\n### ', scratchpad)
        
        # Prioritize recent steps and successful outcomes
        important_sections = []
        for section in sections[-10:]:  # Keep last 10 sections
            if any(keyword in section.lower() for keyword in 
                   ['success', 'completed', 'found', 'generated', current_goal.lower()]):
                important_sections.append(section)
        
        # If still too long, summarize older sections
        if len('\n### '.join(important_sections)) > max_length:
            summary = f"### Previous Steps Summary\n- Completed {len(sections)-len(important_sections)} earlier steps\n- Working toward: {current_goal}\n"
            return summary + '\n### '.join(important_sections[-5:])
        
        return '\n### '.join(important_sections)
    
    def cache_result(self, key: str, value: Any, ttl: int = 300):
        """Cache expensive operation results"""
        self.cache[key] = value
        self.cache_ttl[key] = time.time() + ttl
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached result if still valid"""
        if key in self.cache:
            if time.time() < self.cache_ttl.get(key, 0):
                return self.cache[key]
            else:
                # Expired - remove
                del self.cache[key]
                if key in self.cache_ttl:
                    del self.cache_ttl[key]
        return None

class ToolPerformanceTracker:
    def __init__(self):
        self.tool_stats: Dict[str, ToolPerformance] = {}
        self.lock = threading.Lock()
    
    def record_tool_usage(self, tool_name: str, success: bool, 
                         confidence: float, response_time: float, 
                         error_msg: str = None):
        """Record tool usage statistics"""
        with self.lock:
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = ToolPerformance()
            
            stats = self.tool_stats[tool_name]
            stats.total_calls += 1
            stats.last_used = time.time()
            
            # Update success rate
            old_successes = stats.success_rate * (stats.total_calls - 1)
            new_successes = old_successes + (1 if success else 0)
            stats.success_rate = new_successes / stats.total_calls
            
            # Update average confidence
            old_conf_total = stats.avg_confidence * (stats.total_calls - 1)
            stats.avg_confidence = (old_conf_total + confidence) / stats.total_calls
            
            # Update response time
            old_time_total = stats.avg_response_time * (stats.total_calls - 1)
            stats.avg_response_time = (old_time_total + response_time) / stats.total_calls
            
            # Record failure patterns
            if not success and error_msg:
                stats.failure_patterns.append(error_msg[:100])
                # Keep only last 10 failure patterns
                stats.failure_patterns = stats.failure_patterns[-10:]
    
    def get_tool_reliability_score(self, tool_name: str) -> float:
        """Calculate overall tool reliability score (0-1)"""
        if tool_name not in self.tool_stats:
            return 0.5  # Neutral for unknown tools
        
        stats = self.tool_stats[tool_name]
        
        # Weighted combination of success rate and confidence
        reliability = (stats.success_rate * 0.7) + (stats.avg_confidence * 0.3)
        
        # Penalty for tools not used recently (older than 1 hour)
        if time.time() - stats.last_used > 3600:
            reliability *= 0.8
        
        return reliability
    
    def rank_tools_for_task(self, available_tools: List[str], 
                           task_description: str) -> List[Tuple[str, float]]:
        """Rank tools by suitability for a specific task"""
        tool_scores = []
        
        for tool_name in available_tools:
            base_score = self.get_tool_reliability_score(tool_name)
            
            # Simple keyword matching bonus
            task_lower = task_description.lower()
            if any(keyword in tool_name.lower() for keyword in 
                   ['search', 'research'] if 'find' in task_lower or 'search' in task_lower):
                base_score *= 1.2
            elif 'generate' in tool_name.lower() and 'create' in task_lower:
                base_score *= 1.2
                
            tool_scores.append((tool_name, min(base_score, 1.0)))
        
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores

class UncertaintyManager:
    @staticmethod
    def calculate_confidence(reasoning_step: str, tool_results: List[Dict], 
                           memory_patterns: List[MemoryEntry]) -> Tuple[float, ConfidenceLevel]:
        """Calculate confidence in current reasoning step"""
        base_confidence = 0.5
        
        # Boost confidence if similar patterns succeeded before
        if memory_patterns:
            successful_patterns = [m for m in memory_patterns if m.success]
            if successful_patterns:
                avg_success_confidence = sum(m.confidence for m in successful_patterns) / len(successful_patterns)
                base_confidence = (base_confidence + avg_success_confidence) / 2
        
        # Adjust based on tool result consistency
        if tool_results:
            success_results = [r for r in tool_results if r.get('status') == 'success']
            if success_results:
                base_confidence += 0.2
            
            # Check for consistent information across tools
            if len(tool_results) > 1:
                base_confidence += 0.1
        
        # Reasoning quality indicators
        if len(reasoning_step) > 50 and any(word in reasoning_step.lower() 
                                           for word in ['because', 'therefore', 'analysis', 'evidence']):
            base_confidence += 0.1
        
        confidence = max(0.0, min(1.0, base_confidence))
        
        # Map to confidence levels
        if confidence >= 0.8:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
            
        return confidence, level