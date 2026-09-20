---
title: "Asynchronous Multi-Instance Spawning and Coordination"
description: "Patterns and Python implementations for spawning multiple asynchronous worker instances, coordinating them via atomic file boards or local IPC sockets, and aggregating results."
category: "distributed_agentic_systems"
tags: [multi_agent, asynchronous, subprocess, socket_ipc, file_locking, coordination, parallel_execution]
visibility: loadable
modifiable: true
---

# Asynchronous Multi-Instance Spawning & Coordination

When a software engineering task requires processing dozens of files, running test suites in parallel, performing fuzz testing, or executing long-horizon tasks concurrently, running sequentially in a single turn wastes context and time.

This skill teaches how an orchestrating `lollms_code` agent can launch multiple independent background worker instances and coordinate them using **File-Based Task Boards** or **Local IPC Sockets**.

---

## 🏛️ Part 1: Coordination Architecture Patterns

```
                          ┌───────────────────────────┐
                          │  PRIMARY AGENT / LEADER   │
                          │   (Spawns & Coordinates)   │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     [Model A: File Task Board]                   [Model B: Socket IPC Hub]
  .lollms_code/coordinator/tasks/                   127.0.0.1:<port> TCP Server
   ├── pending/   (tasks queued)                    ├── Events (REGISTER, PROGRESS)
   ├── active/    (claimed via atomic lock)         ├── Heartbeats & Liveness
   └── completed/ (results & outputs)               └── Barrier Sync (wait_all)
         │           │           │                         │           │
         ▼           ▼           ▼                         ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐               ┌─────────┐ ┌─────────┐
    │Worker #1│ │Worker #2│ │Worker #3│               │Worker #1│ │Worker #2│
    └─────────┘ └─────────┘ └─────────┘               └─────────┘ └─────────┘
```

### Choosing the Right Model

| Feature | Model A: File-Based Task Board | Model B: Local Socket IPC Hub |
| :--- | :--- | :--- |
| **Best For** | File refactoring, parallel scripts, map-reduce tasks | Real-time streaming progress, dynamic barrier sync |
| **Transport** | Filesystem (`.json` + atomic lockfiles) | Local TCP socket (`127.0.0.1`) with JSON-lines |
| **Persistence** | Survives crashes; resumable on restart | In-memory during process lifecycle |
| **Dependencies** | Standard Library (`os`, `json`, `time`, `pathlib`) | Standard Library (`socket`, `threading`, `json`) |

---

## 📂 Part 2: Model A — File-Based Task Board Coordinator

In this model, the coordinator writes discrete task files into a directory queue. Workers atomically claim tasks, execute them in background processes, and write their results back.

### Complete File Coordinator (`coordinator_file.py`)

```python
"""
coordinator_file.py
Zero-dependency, cross-platform file task board for multi-worker coordination.
"""

import os
import sys
import json
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

WORKSPACE_DIR = Path(".").resolve()
COORDINATOR_DIR = WORKSPACE_DIR / ".lollms_code" / "coordinator"
PENDING_DIR = COORDINATOR_DIR / "pending"
ACTIVE_DIR = COORDINATOR_DIR / "active"
COMPLETED_DIR = COORDINATOR_DIR / "completed"
FAILED_DIR = COORDINATOR_DIR / "failed"


def init_task_board():
    """Initializes the task board directory structure."""
    for d in (PENDING_DIR, ACTIVE_DIR, COMPLETED_DIR, FAILED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def create_task(task_type: str, payload: Dict[str, Any], task_id: Optional[str] = None) -> str:
    """Enqueues a new atomic task."""
    init_task_board()
    t_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
    task_data = {
        "task_id": t_id,
        "task_type": task_type,
        "payload": payload,
        "created_at": time.time(),
        "status": "pending"
    }
    
    # Write atomically via temp file
    temp_file = COORDINATOR_DIR / f"{t_id}.tmp"
    target_file = PENDING_DIR / f"{t_id}.json"
    
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2)
    os.replace(temp_file, target_file)
    return t_id


def claim_task(worker_id: str) -> Optional[Dict[str, Any]]:
    """
    Atomically claims the next pending task using filesystem rename.
    Guarantees no two workers can claim the same task.
    """
    init_task_board()
    pending_files = sorted(PENDING_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    
    for p_file in pending_files:
        target_active = ACTIVE_DIR / p_file.name
        try:
            # Atomic move: only the winning worker succeeds
            os.replace(str(p_file), str(target_active))
            
            # Read and append claim metadata
            with open(target_active, "r", encoding="utf-8") as f:
                task = json.load(f)
                
            task["status"] = "in_progress"
            task["worker_id"] = worker_id
            task["claimed_at"] = time.time()
            task["pid"] = os.getpid()
            
            with open(target_active, "w", encoding="utf-8") as f:
                json.dump(task, f, indent=2)
                
            return task
        except (OSError, FileNotFoundError):
            # Another worker claimed it first; proceed to next
            continue
            
    return None


def complete_task(task_id: str, result: Dict[str, Any], success: bool = True):
    """Marks a task as completed or failed and writes its output."""
    active_file = ACTIVE_DIR / f"{task_id}.json"
    dest_dir = COMPLETED_DIR if success else FAILED_DIR
    target_file = dest_dir / f"{task_id}.json"
    
    if not active_file.exists():
        return
        
    try:
        with open(active_file, "r", encoding="utf-8") as f:
            task = json.load(f)
            
        task["status"] = "completed" if success else "failed"
        task["completed_at"] = time.time()
        task["result"] = result
        
        temp_file = COORDINATOR_DIR / f"{task_id}_done.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)
            
        os.replace(temp_file, target_file)
        if active_file.exists():
            active_file.unlink()
    except Exception as e:
        print(f"[Coordinator] Error completing task {task_id}: {e}", file=sys.stderr)


def get_board_status() -> Dict[str, Any]:
    """Returns a summary of all task states on the board."""
    init_task_board()
    pending = [p.stem for p in PENDING_DIR.glob("*.json")]
    active = [p.stem for p in ACTIVE_DIR.glob("*.json")]
    completed = [p.stem for p in COMPLETED_DIR.glob("*.json")]
    failed = [p.stem for p in FAILED_DIR.glob("*.json")]
    
    return {
        "pending_count": len(pending),
        "active_count": len(active),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "is_finished": len(pending) == 0 and len(active) == 0,
        "tasks": {
            "pending": pending,
            "active": active,
            "completed": completed,
            "failed": failed
        }
    }


def spawn_workers(worker_script: str, num_workers: int = 3) -> List[subprocess.Popen]:
    """Launches N asynchronous worker processes running the specified script."""
    processes = []
    py_exec = sys.executable
    script_path = str(WORKSPACE_DIR / worker_script)
    
    for idx in range(1, num_workers + 1):
        worker_id = f"worker_{idx}"
        cmd = [py_exec, script_path, "--worker-id", worker_id]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(WORKSPACE_DIR)
        )
        processes.append(proc)
        print(f"🚀 [Supervisor] Spawned {worker_id} (PID: {proc.pid})")
        
    return processes


def wait_for_completion(timeout: float = 120.0, poll_interval: float = 1.0) -> Dict[str, Any]:
    """Polls the board until all tasks are completed or timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_board_status()
        print(f"⏳ [Supervisor] Progress: {status['completed_count']} done, {status['active_count']} active, {status['pending_count']} pending")
        
        if status["is_finished"]:
            print("✅ [Supervisor] All tasks completed successfully.")
            return status
            
        time.sleep(poll_interval)
        
    raise TimeoutError(f"Tasks timed out after {timeout} seconds.")
```

### Generic Worker Script (`worker_task_board.py`)

```python
"""
worker_task_board.py
Autonomous worker pulling tasks from the file task board.
"""

import sys
import time
import argparse
from coordinator_file import claim_task, complete_task

def execute_payload(task_type: str, payload: dict) -> dict:
    """Execute the domain logic based on task_type."""
    if task_type == "test_run":
        target = payload.get("test_file")
        # Run tests or computations
        time.sleep(1.0)  # Simulating work
        return {"file": target, "passed": 12, "failed": 0}
        
    elif task_type == "code_transform":
        target = payload.get("file_path")
        # Perform file transformation
        return {"file": target, "status": "refactored"}
        
    return {"status": "unknown_task_type"}


def run_worker_loop(worker_id: str):
    print(f"[{worker_id}] Starting worker loop...")
    idle_rounds = 0
    
    while True:
        task = claim_task(worker_id)
        if task is None:
            idle_rounds += 1
            if idle_rounds >= 5:
                print(f"[{worker_id}] Queue empty for 5 cycles. Exiting cleanly.")
                break
            time.sleep(0.5)
            continue
            
        idle_rounds = 0
        task_id = task["task_id"]
        task_type = task["task_type"]
        payload = task["payload"]
        
        print(f"[{worker_id}] Claimed {task_id} ({task_type})")
        try:
            result = execute_payload(task_type, payload)
            complete_task(task_id, result=result, success=True)
            print(f"[{worker_id}] Completed {task_id}")
        except Exception as e:
            complete_task(task_id, result={"error": str(e)}, success=False)
            print(f"[{worker_id}] Failed {task_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", default="worker_default")
    args = parser.parse_args()
    run_worker_loop(args.worker_id)
```

---

## 🔌 Part 3: Model B — Local IPC Socket Coordinator

When real-time telemetry, low overhead, and dynamic backpressure are needed, use a local socket coordinator on `127.0.0.1`.

### Complete Socket Coordinator Server (`coordinator_socket.py`)

```python
"""
coordinator_socket.py
Lightweight localhost TCP coordinator using a JSON-lines protocol.
"""

import sys
import json
import time
import socket
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


class SocketCoordinator:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, port))
        self.port = self.server_sock.getsockname()[1]
        
        self.running = True
        self.lock = threading.Lock()
        
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        self.server_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.server_thread.start()
        print(f"📡 [Coordinator] Server listening on {self.host}:{self.port}")

    def add_task(self, task_id: str, payload: Dict[str, Any]):
        with self.lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "payload": payload,
                "status": "pending",
                "assigned_to": None
            }

    def _accept_loop(self):
        self.server_sock.listen(10)
        while self.running:
            try:
                conn, _ = self.server_sock.accept()
                threading.Thread(target=self._client_handler, args=(conn,), daemon=True).start()
            except Exception:
                break

    def _client_handler(self, conn: socket.socket):
        worker_id = None
        buffer = ""
        with conn:
            while self.running:
                data = conn.recv(4096).decode("utf-8")
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                        response = self._process_message(msg)
                        if response:
                            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                    except json.JSONDecodeError:
                        continue

    def _process_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        mtype = msg.get("type")
        worker_id = msg.get("worker_id", "unknown")

        with self.lock:
            if mtype == "REGISTER":
                self.workers[worker_id] = {"last_seen": time.time(), "status": "idle"}
                return {"type": "ACK", "message": "Registered"}

            elif mtype == "REQUEST_TASK":
                self.workers[worker_id] = {"last_seen": time.time(), "status": "requesting"}
                # Find first pending task
                for t_id, task in self.tasks.items():
                    if task["status"] == "pending":
                        task["status"] = "in_progress"
                        task["assigned_to"] = worker_id
                        return {
                            "type": "ASSIGN_TASK",
                            "task_id": t_id,
                            "payload": task["payload"]
                        }
                return {"type": "NO_TASKS"}

            elif mtype == "REPORT_RESULT":
                t_id = msg.get("task_id")
                result = msg.get("result", {})
                success = msg.get("success", True)
                
                if t_id in self.tasks:
                    self.tasks[t_id]["status"] = "completed" if success else "failed"
                    self.results[t_id] = result
                    
                return {"type": "ACK", "message": "Result received"}

        return {"type": "ERROR", "message": "Unknown command"}

    def wait_all_done(self, timeout: float = 60.0) -> Dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                total = len(self.tasks)
                completed = sum(1 for t in self.tasks.values() if t["status"] in ("completed", "failed"))
                if total > 0 and completed == total:
                    return self.results
            time.sleep(0.5)
        raise TimeoutError("Socket coordinator wait_all_done timed out.")

    def close(self):
        self.running = False
        try:
            self.server_sock.close()
        except Exception:
            pass
```

### Socket Worker Script (`worker_socket.py`)

```python
"""
worker_socket.py
Asynchronous client communicating with the SocketCoordinator over TCP.
"""

import sys
import json
import time
import socket
import argparse

def send_msg(sock: socket.socket, msg: dict) -> dict:
    raw = json.dumps(msg) + "\n"
    sock.sendall(raw.encode("utf-8"))
    
    buffer = ""
    while "\n" not in buffer:
        chunk = sock.recv(4096).decode("utf-8")
        if not chunk:
            raise ConnectionError("Server disconnected.")
        buffer += chunk
        
    line = buffer.split("\n", 1)[0]
    return json.loads(line)

def run_worker(port: int, worker_id: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    
    # Register
    send_msg(sock, {"type": "REGISTER", "worker_id": worker_id})
    print(f"[{worker_id}] Connected and registered.")

    while True:
        resp = send_msg(sock, {"type": "REQUEST_TASK", "worker_id": worker_id})
        
        if resp.get("type") == "NO_TASKS":
            time.sleep(0.5)
            # Recheck or exit if drained
            resp2 = send_msg(sock, {"type": "REQUEST_TASK", "worker_id": worker_id})
            if resp2.get("type") == "NO_TASKS":
                print(f"[{worker_id}] No more tasks. Shutting down.")
                break
            resp = resp2

        if resp.get("type") == "ASSIGN_TASK":
            task_id = resp["task_id"]
            payload = resp["payload"]
            print(f"[{worker_id}] Executing {task_id}...")
            
            # Domain processing
            time.sleep(0.8)
            result = {"output": f"Processed {payload.get('item')}", "code": 0}
            
            send_msg(sock, {
                "type": "REPORT_RESULT",
                "worker_id": worker_id,
                "task_id": task_id,
                "result": result,
                "success": True
            })
            
    sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-id", default="socket_worker_1")
    args = parser.parse_args()
    run_worker(args.port, args.worker_id)
```

---

## ⚡ Part 4: Step-by-Step Agent Workflow

When asked to run tasks in parallel or across multiple asynchronous instances:

1. **Step 1 — Create the Coordinator & Worker Files**:
   Emit `<artifact name="coordinator_file.py" type="code">` and `<artifact name="worker_task_board.py" type="code">`.
2. **Step 2 — Populate the Tasks Queue**:
   Execute a Python snippet using `tool_execute_python_code` that calls `create_task()` for each item.
3. **Step 3 — Spawn Background Workers**:
   Call `spawn_workers("worker_task_board.py", num_workers=4)`.
4. **Step 4 — Wait and Synchronize**:
   Call `wait_for_completion(timeout=120.0)` to aggregate all result JSONs from `completed/`.
5. **Step 5 — Cleanup**:
   Clean up temporary coordination files and present the aggregated summary to the user.