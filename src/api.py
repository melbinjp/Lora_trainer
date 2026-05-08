from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import os
import sys
import uuid
import threading
import json
import base64

# Disable telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OPENVINO_DISABLE_TELEMETRY"] = "1"

app = FastAPI(title="LoRA Trainer & Inference API", version="1.1.0")

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Mount public directory
app.mount("/static", StaticFiles(directory="public"), name="static")

# Lock to ensure only one heavy inference process runs at a time per node to prevent OOM
inference_lock = threading.Lock()

class TaskState:
    def __init__(self):
        self.active_process = None
        self.logs = []
        self.images = [] # Store as base64
        self.status = "idle" # idle, running, completed, error
        self.lock = threading.Lock()
        self.task_id = None
        self.num_images = 0
        self.request_data = {}

    def start(self, cmd, task_id, num_images, request_data):
        with self.lock:
            if self.active_process and self.active_process.poll() is None:
                return False
            self.logs = []
            self.images = []
            self.task_id = task_id
            self.num_images = num_images
            self.request_data = request_data
            self.status = "running"
            
            self.active_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            threading.Thread(target=self._drain_logs, daemon=True).start()
            return True

    def _drain_logs(self):
        try:
            for line in iter(self.active_process.stdout.readline, ''):
                if line:
                    log_line = line.strip()
                    with self.lock:
                        self.logs.append(log_line)
                        if "[SAVED]" in log_line:
                            try:
                                path = log_line.split("[SAVED]")[-1].strip()
                                if os.path.exists(path):
                                    with open(path, "rb") as f:
                                        b64 = base64.b64encode(f.read()).decode("utf-8")
                                        self.images.append(b64)
                            except Exception as e:
                                self.logs.append(f"Error encoding image: {e}")
            
            exit_code = self.active_process.wait()
            with self.lock:
                if exit_code == 0:
                    self.status = "completed"
                else:
                    self.status = "error"
                    self.logs.append(f"Process failed with exit code {exit_code}")
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.logs.append(f"Internal error draining logs: {e}")

    def get_info(self):
        with self.lock:
            return {
                "status": self.status,
                "task_id": self.task_id,
                "num_images": self.num_images,
                "logs_count": len(self.logs),
                "images_count": len(self.images),
                "request": self.request_data
            }

task_state = TaskState()

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "OpenVINO/FLUX.1-schnell-int4-ov"
    hardware_target: str = "STABLE_HYBRID"
    token: str = None
    steps: int = None
    guidance: float = None
    num_images: int = 1
    width: int = 512
    height: int = 512
    seed: int = -1
    
class EnhanceRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return FileResponse("public/index.html")

@app.get("/health")
def health_check():
    """Endpoint for orchestrators to check if this node is alive and its lock status."""
    return {"status": "alive", "busy": inference_lock.locked()}

@app.get("/models_status")
def get_models_status():
    def normalize(name):
        name = name.lower().replace("flux.1", "flux").replace("flux1", "flux")
        return name.replace(".", "").replace("-", "")
        
    def get_status(repo_id):
        repo_name = repo_id.split("/")[-1].lower()
        if not os.path.exists("models"): return False
        norm_repo = normalize(repo_name)
        for folder in os.listdir("models"):
            if normalize(folder) == norm_repo:
                path = os.path.join("models", folder)
                ov_bin = os.path.join(path, "transformer", "openvino_model.bin")
                diff_bin = os.path.join(path, "transformer", "diffusion_pytorch_model.safetensors")
                if os.path.exists(ov_bin) or os.path.exists(diff_bin):
                    return True
        return False
        
    return {
        "schnell": get_status("OpenVINO/FLUX.1-schnell-int4-ov"),
        "sdxl_turbo": get_status("OpenVINO/stable-diffusion-xl-turbo-int4-ov"),
        "sd15_lcm": get_status("OpenVINO/stable-diffusion-v1.5-lcm-int8-ov")
    }

@app.post("/enhance_prompt")
def enhance_prompt(req: EnhanceRequest):
    from engine.prompt_llm import enhance_prompt_locally_stream
    def stream_generator():
        try:
            for token in enhance_prompt_locally_stream(req.prompt):
                yield json.dumps({"token": token}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.get("/get_current_task")
def get_current_task():
    return task_state.get_info()

@app.post("/generate_stream")
def generate_image_stream(req: GenerateRequest):
    """
    Streams logs and results. If a task is already running, it joins it.
    If not, it starts a new one.
    """
    def event_stream():
        # Check if already running
        info = task_state.get_info()
        
        if info["status"] != "running":
            # Start new task
            task_id = str(uuid.uuid4())
            output_img = os.path.join("outputs", f"api_generated_{task_id}.png")
            intel_script = os.path.join(os.path.dirname(__file__), "engine", "intel_inference.py")
            
            cmd = [sys.executable, intel_script, req.hardware_target, 
                 "--prompt", req.prompt, 
                 "--output", output_img,
                 "--model", req.model]
            if req.token:
                cmd.extend(["--token", req.token])
            if req.steps is not None:
                cmd.extend(["--steps", str(req.steps)])
            if req.guidance is not None:
                cmd.extend(["--guidance", str(req.guidance)])
            if req.num_images > 1:
                cmd.extend(["--num_images", str(req.num_images)])
            if req.width:
                cmd.extend(["--width", str(req.width)])
            if req.height:
                cmd.extend(["--height", str(req.height)])
            if req.seed is not None:
                cmd.extend(["--seed", str(req.seed)])
            
            # Request management data for the UI to reconstruct state
            request_data = req.model_dump()
            
            success = task_state.start(cmd, task_id, req.num_images, request_data)
            if not success:
                yield json.dumps({"type": "error", "data": "A task is already starting..."}) + "\n"
                return

        # Stream from the beginning of the current task
        last_log_idx = 0
        last_img_idx = 0
        
        import time
        while True:
            with task_state.lock:
                # Send new logs
                new_logs = task_state.logs[last_log_idx:]
                for log in new_logs:
                    yield json.dumps({"type": "log", "data": log}) + "\n"
                last_log_idx = len(task_state.logs)
                
                # Send new images
                new_images = task_state.images[last_img_idx:]
                for img_b64 in new_images:
                    yield json.dumps({"type": "result", "data": img_b64}) + "\n"
                last_img_idx = len(task_state.images)
                
                status = task_state.status
                
            if status in ["completed", "error"]:
                break
                
            time.sleep(0.5) # Poll internal state for the stream
            
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@app.post("/cancel_task")
def cancel_task():
    with task_state.lock:
        if task_state.active_process and task_state.active_process.poll() is None:
            task_state.active_process.terminate()
            task_state.status = "idle"
            task_state.logs.append("--- Task Cancelled by User ---")
            return {"status": "cancelled"}
    return {"status": "none running"}

@app.post("/generate")
def generate_image(req: GenerateRequest):
    """
    Programmatic endpoint for generating images. 
    This allows distributed systems or scripts to request image generation.
    Queues/executes the request using the optimized Intel engine sequentially.
    """
    task_id = str(uuid.uuid4())
    output_img = os.path.join("outputs", f"api_generated_{task_id}.png")
    intel_script = os.path.join(os.path.dirname(__file__), "engine", "intel_inference.py")
    
    # Acquire lock to prevent multiple models loading into RAM simultaneously on one node
    acquired = inference_lock.acquire(timeout=600) # Wait up to 10 minutes in queue
    if not acquired:
        raise HTTPException(status_code=503, detail="Node is too busy, timeout waiting in queue.")
        
    try:
        process = subprocess.run(
            [sys.executable, intel_script, req.hardware_target, 
             "--prompt", req.prompt, 
             "--output", output_img,
             "--model", req.model],
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0 and os.path.exists(output_img):
            return FileResponse(output_img, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail=f"Generation failed: {process.stderr or process.stdout}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        inference_lock.release()
        # Cleanup the file after sending (BackgroundTasks could be better, but simple is fine for now)

if __name__ == "__main__":
    import uvicorn
    # Using 127.0.0.1 and port 8080 for better compatibility on Windows
    print("\n--- Vanilla UI Server Starting ---")
    print("If http://0.0.0.0:8000 is unreachable, use: http://localhost:8080")
    print("----------------------------------\n")
    uvicorn.run(app, host="127.0.0.1", port=8080)
