<analysis>
1.  **Understanding**: The user wants comprehensive documentation on creating a Text-to-Image (TTI) binding for the Lollms system. The documentation must cover specific architectural requirements: handling multiple processes (multiprocessing), resource management (releasing resources), client-server architecture for heavy models (like Diffusers), and direct API integration for external services (like OpenRouter). It also needs to explain the `description.yaml` file, custom commands, and providing examples.

2.  **Problem Hypothesis**: The current documentation in the project (based on the file tree) might be missing a specific, detailed guide on building TTI bindings with these advanced patterns. The user provided existing code (`diffusers` and `open_router`) as the "source of truth" for how these bindings should be structured. I need to synthesize this code into a clear guide.

3.  **Solution Plan**:
    *   Create a new markdown file `docs/md/building_tti_bindings.md`.
    *   **Structure**:
        *   **Concept**: Explain what a TTI binding is in Lollms.
        *   **File Structure**: Required files (`__init__.py`, `description.yaml`, optional `server/`).
        *   **Configuration**: detailed breakdown of `description.yaml` (metadata, params, commands).
        *   **Coding the Binding**:
            *   Base class `LollmsTTIBinding`.
            *   Essential methods (`generate_image`, `edit_image`).
            *   Helper methods (`process_image` for watermarking).
        *   **Architecture Patterns**:
            *   **Type 1: External API (Stateless)** - Use OpenRouter code as the example. Simple HTTP calls.
            *   **Type 2: Local Resource (Stateful/Client-Server)** - Use Diffusers code as the example. Explain the necessity of a dedicated server process to handle GPU contention across multiple Lollms workers (UI, API, etc.).
        *   **Deep Dive: Client-Server Pattern**:
            *   **Server**: FastAPI, Queue system for serializing GPU requests, `pipmaster` for dependencies.
            *   **Client**: Process management, FileLocks for safe startup, proxying requests.
            *   **Resource Release**: Shutdown endpoints, `__del__`.
        *   **UI Integration**: How `input_parameters` map to the Settings UI.

4.  **Implementation**: I will generate the full markdown content based on the analysis of the provided `diffusers` and `open_router` code.
</analysis>

File: docs/md/building_tti_bindings.md
```markdown
# Building Lollms TTI Bindings

This guide explains how to create Text-to-Image (TTI) bindings for Lollms. A binding acts as a bridge between the Lollms core and an image generation provider, whether it's an external API (like DALL-E, OpenRouter) or a local model running on GPU (like Stable Diffusion).

## 1. Directory Structure

All TTI bindings reside in `src/lollms_client/tti_bindings/`. A binding is a folder containing at least two files:

```text
tti_bindings/
└─ my_binding/
   ├─ __init__.py          # The Python logic (Class definition)
   ├─ description.yaml     # Configuration, UI settings, and metadata
   └─ server/              # (Optional) Standalone server code for local models
      ├─ main.py
      └─ requirements.txt
```

---

## 2. Configuration: `description.yaml`

This file defines how your binding appears in the settings UI and declares what configuration parameters it needs.

```yaml
title: "My TTI Binding"
author: "Your Name"
version: 1.0.0
binding_name: "my_binding"  # Must match folder name
description: |
  Description of what this binding does.
  Supports MarkDown.

# Parameters shown in the Lollms Web UI Settings tab
input_parameters:
  - name: "api_key"
    type: "str"
    description: "API Key for the service"
    mandatory: true
    is_secret: true         # Hides input in UI
    default: ""

  - name: "resolution"
    type: "str"
    description: "Image generation resolution"
    default: "1024x1024"
    options: ["512x512", "1024x1024"]

  - name: "gpu_id"
    type: "int"
    description: "GPU ID for local models"
    default: 0

# Custom commands actionable via buttons in the UI
commands:
  - name: install_dependencies
    title: "Install Dependencies"
    description: "Installs required python packages."
    parameters: []
    output:
      - name: status
        type: bool
      - name: message
        type: str
```

---

## 3. The Python Binding Class (`__init__.py`)

Your binding must inherit from `LollmsTTIBinding`.

### Basic Skeleton

```python
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import ASCIIColors

# The class name must match the module name or be imported as such
BindingName = "MyTTIBinding"

class MyTTIBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        # kwargs contains keys from your description.yaml
        self.api_key = kwargs.get("api_key", "")
    
    def get_settings(self, **kwargs):
        """Return current config to update UI state"""
        return self.config

    def set_settings(self, settings, **kwargs):
        """Handle settings updates from UI"""
        self.config.update(settings)
        # Update internal state if necessary
        return True

    def list_models(self):
        """Return list of available models"""
        return [{"model_name": "v1", "display_name": "Version 1"}]

    def generate_image(self, prompt, negative_prompt="", width=512, height=512, **kwargs):
        """
        The core generation logic.
        Must return raw image bytes (PNG/JPG).
        """
        # Logic here...
        image_bytes = b"..." 
        
        # ALWAYS use self.process_image before returning
        # This handles watermarking and metadata injection
        return self.process_image(image_bytes, **kwargs)

    def edit_image(self, images, prompt, **kwargs):
        """Optional: Implement if supporting Image-to-Image"""
        pass
```

---

## 4. Architecture Patterns

There are two main ways to build a binding depending on resource usage.

### Type A: External API (Stateless)
**Use Case:** OpenRouter, OpenAI, Leonardo AI.

Since these rely on HTTP requests to a remote server, the binding instance is lightweight. Lollms processes (server, workers) can simply instantiate the class and make requests.

**Key Requirements:**
1.  **Authentication**: Use `is_secret: true` in `description.yaml` for keys.
2.  **Concurrency**: APIs handle concurrency remotely. No special locking needed locally.

**Example Implementation Snippet:**
```python
import requests
import base64

def generate_image(self, prompt, **kwargs):
    headers = {"Authorization": f"Bearer {self.api_key}"}
    payload = {"prompt": prompt}
    response = requests.post("https://api.provider.com/generate", json=payload, headers=headers)
    
    # Process response
    url = response.json()['url']
    img_data = requests.get(url).content
    return self.process_image(img_data, **kwargs)
```

---

### Type B: Local Resource (Client-Server Architecture)
**Use Case:** Stable Diffusion (Diffusers), Flux, Local GGUF models.

**The Problem:**
1.  **Multiprocessing:** Lollms runs in multiple processes (Server process, SocketIO workers, etc.). You cannot load a 10GB GPU model into RAM in *every* process.
2.  **GPU Contention:** Multiple requests cannot access the GPU VRAM simultaneously without crashing or queuing.
3.  **Dependencies:** Heavy libraries (`torch`, `diffusers`) conflict with Lollms core dependencies.

**The Solution:**
Implement a **Client-Server** architecture.
*   **Server**: A dedicated Python process (running FastAPI) that holds the model in VRAM. It has its own Virtual Environment (venv).
*   **Client**: The `LollmsTTIBinding` class acts as a proxy, sending HTTP requests to the local server.

#### 1. The Server (`server/main.py`)
Use `FastAPI` and `uvicorn`. Implement a queue to serialize generation requests.

```python
# server/main.py
from fastapi import FastAPI
import uvicorn
import queue
import threading

app = FastAPI()
q = queue.Queue()

# Worker thread to process generation sequentially
def worker():
    while True:
        future, prompt = q.get()
        # ... Run heavy GPU generation ...
        result = run_stable_diffusion(prompt)
        future.set_result(result)
        q.task_done()

threading.Thread(target=worker, daemon=True).start()

@app.post("/generate_image")
def generate(request: Request):
    # Add to queue, wait for result
    future = Future()
    q.put((future, request.prompt))
    return future.result()
```

#### 2. The Client (`__init__.py`)
The client must manage the server lifecycle: start it if missing, restart it if crashed, and handle dependencies.

**Key Features to Implement:**

1.  **Dependency Management (Pipmaster):**
    Use `pipmaster` to install the server's requirements into a *separate* venv to avoid conflicts.
    ```python
    import pipmaster as pm
    venv_path = Path("./venv/my_binding_venv")
    pm = pm.PackageManager(venv_path=str(venv_path))
    pm.install("fastapi uvicorn torch diffusers")
    ```

2.  **Process Safety (FileLock):**
    When Lollms starts, multiple workers might try to launch your server simultaneously. Use `filelock` to ensure only one instance starts.
    ```python
    from filelock import FileLock
    
    def start_server(self):
        with FileLock("server.lock"):
            if not self.is_server_running():
                # subprocess.Popen(...)
    ```

3.  **Resource Release:**
    Implement `shutdown` methods.
    ```python
    def shutdown(self):
        requests.post(f"{self.base_url}/shutdown")
    ```

4.  **Auto-Discovery:**
    Implement `list_models` by querying the server, which in turn scans the disk.

#### Example Client Implementation (Simplified):

```python
class MyLocalBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        self.base_url = "http://localhost:9632"
        self.ensure_server_running()

    def ensure_server_running(self):
        if not self.check_connection():
            self.start_server_process()

    def start_server_process(self):
        # Use pipmaster to ensure venv exists
        # Use subprocess to launch server/main.py inside that venv
        pass

    def generate_image(self, prompt, **kwargs):
        # Proxy to local server
        response = requests.post(f"{self.base_url}/generate", json={"prompt": prompt})
        return self.process_image(response.content, **kwargs)
```

---

## 5. Best Practices

1.  **Process Images**: Always pass generated bytes through `self.process_image(bytes, **kwargs)`. This ensures:
    *   Metadata injection (Generation params, Author).
    *   Visible watermarking (if configured in Lollms).
    *   Consistent format conversion.

2.  **Progress Reporting**:
    When downloading models, use a callback if available or log using `ASCIIColors`.

3.  **Logging**:
    Use `ascii_colors` for standardized logging.
    ```python
    from ascii_colors import ASCIIColors
    ASCIIColors.info("Starting generation...")
    ASCIIColors.error("Model failed to load.")
    ```

4.  **Handling Multi-Process Requests**:
    If using the Client-Server model, the `queue` in your server is crucial. It ensures that if 3 users request images simultaneously, the GPU processes them one by one (Serial) while the HTTP requests wait (Async).

5.  **Shutdown**:
    Implement a `__del__` or explicit `shutdown` command to kill the background server process when Lollms closes, preventing orphan python processes consuming VRAM.
```