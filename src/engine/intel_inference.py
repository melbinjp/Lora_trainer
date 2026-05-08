import openvino_genai as ov_genai
import openvino as ov
from huggingface_hub import snapshot_download
from PIL import Image
import os
import sys
import psutil
import shutil
import ctypes
import time
import logging

# Disable telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OPENVINO_DISABLE_TELEMETRY"] = "1"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.helpers import aggressive_flush
try:
    from src.engine.prompt_llm import enhance_prompt_locally
except ImportError:
    enhance_prompt_locally = None

# --- CONFIGURATION ---
DEFAULT_MODEL_ID = "OpenVINO/FLUX.1-schnell-int4-ov"
OUTPUT_FILE = os.path.join("outputs", "flower_tortoise_final.png")
LOG_FILE = os.path.join("outputs", "inference_debug.log")

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_and_print(msg):
    print(msg)
    logging.info(msg)

def get_model_path(model_id):
    repo_name = model_id.split("/")[-1]
    
    def normalize(name):
        name = name.lower().replace("flux.1", "flux").replace("flux1", "flux")
        return name.replace(".", "").replace("-", "")
        
    norm_repo = normalize(repo_name)
    
    if os.path.exists("models"):
        for folder in os.listdir("models"):
            if normalize(folder) == norm_repo:
                return os.path.abspath(os.path.join("models", folder))
                
    # Fallback to exact name if not found
    return os.path.abspath(os.path.join("models", repo_name))

def check_memory(threshold_gb=4.0): 
    """Returns True if enough memory is available, False otherwise."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    log_and_print(f"[MEMORY] Available: {available_gb:.2f} GB / Total: {mem.total / (1024**3):.2f} GB")
    
    if available_gb < threshold_gb:
        log_and_print(f"--- MEMORY ALERT ---")
        log_and_print(f"Required: {threshold_gb}GB | Found: {available_gb:.2f}GB")
        log_and_print("Safety check failed. The model might crash your PC.")
        log_and_print("-----------------------")
        return False
    return True

def download_model(model_id, hf_token=None):
    """Handles the 11GB model download."""
    log_and_print(f"[STEP 1] Checking/Downloading weights ({model_id})...")
    
    model_path = get_model_path(model_id)
    
    transformer_bin = os.path.join(model_path, "transformer", "openvino_model.bin")
    if os.path.exists(model_path) and not os.path.exists(transformer_bin):
        log_and_print("[WARNING] Weights missing. Cleaning corrupted folder...")
        shutil.rmtree(model_path, ignore_errors=True)
    
    if not os.path.exists(model_path) or not os.path.exists(transformer_bin):
        log_and_print(f"[INFO] Starting 11GB download of {model_id}. This is safe for your RAM.")
        try:
            snapshot_download(repo_id=model_id, local_dir=model_path, token=hf_token)
            log_and_print("[SUCCESS] Download complete.")
        except Exception as e:
            log_and_print(f"[FATAL] Download failed: {e}")
            sys.exit(1)
    else:
        log_and_print("[INFO] Model weights already present.")

def progress_callback(step: int, num_steps: int, latent: object) -> bool:
    """Prints progress during the denoising steps."""
    percent = ((step + 1) / num_steps) * 100
    log_and_print(f"[PROGRESS] Step {step + 1}/{num_steps} ({percent:.1f}%)")
    sys.stdout.flush()
    return False

def run_inference(mode="AUTO", threshold=4.0, prompt=None, output_file=None, model_id=DEFAULT_MODEL_ID, steps=None, guidance=None, enhance=False, width=512, height=512, seed=-1):
    """Handles the memory-intensive image generation with resource limiting."""
    log_and_print(f"[STEP 2] Preparing Inference (Mode: {mode})...")
    
    if enhance and enhance_prompt_locally:
        log_and_print(f"[INFO] Enhancing prompt locally...")
        try:
            enhanced = enhance_prompt_locally(prompt)
            log_and_print(f"[ENHANCED PROMPT] {enhanced}")
            prompt = enhanced
        except Exception as e:
            log_and_print(f"[WARNING] Prompt enhancement failed: {e}. Using original prompt.")

    log_and_print(f"[INFO] Model: {model_id}")
    
    model_path = get_model_path(model_id)
    
    aggressive_flush()
    
    if not check_memory(threshold_gb=threshold):
        log_and_print("[STOP] Aborting inference to prevent system crash.")
        return

    upper_args = [a.upper() for a in sys.argv]
    num_images = 1
    if "--NUM_IMAGES" in upper_args:
        num_images = int(sys.argv[upper_args.index("--NUM_IMAGES") + 1])

    try:
        log_and_print(f"[INIT] Initializing OpenVINO Core and Pipeline...")
        core = ov.Core()
        
        # Enable caching to reduce compilation spikes
        cache_dir = "ov_cache"
        os.makedirs(cache_dir, exist_ok=True)
        core.set_property({"CACHE_DIR": cache_dir})
        
        # Limit CPU threads to avoid locking the UI/terminal
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": 6})
        
        # Load Pipeline
        pipe = ov_genai.Text2ImagePipeline(model_path)
        
        # Standardize shapes (Mandatory for NPU, recommended for GPU/CPU efficiency)
        log_and_print(f"[INIT] Reshaping pipeline for static dimensions ({width}x{height}, batch=1, guidance=0.0)...")
        pipe.reshape(num_images_per_prompt=1, height=height, width=width, guidance_scale=0.0)
        
        available_devices = core.available_devices
        
        compiled = False
        if mode == "ULTRA_EFF":
            log_and_print("[INIT] Compiling ULTRA_EFF: Text(NPU) + Denoise(CPU) + VAE(GPU)")
            try:
                pipe.compile(
                    text_encode_device="NPU" if "NPU" in available_devices else "CPU",
                    denoise_device="CPU",
                    vae_device="GPU" if "GPU" in available_devices else "CPU"
                )
                compiled = True
            except Exception as ex:
                log_and_print(f"[WARNING] NPU compilation failed: {ex}. Falling back to STABLE_HYBRID.")
                pipe = ov_genai.Text2ImagePipeline(model_path)
                pipe.reshape(num_images_per_prompt=1, height=height, width=width, guidance_scale=0.0)
                mode = "STABLE_HYBRID" 
        
        if not compiled:
            if mode == "STABLE_HYBRID" or mode == "HYBRID":
                log_and_print("[INIT] Compiling STABLE_HYBRID: Text(CPU) + Denoise(CPU) + VAE(GPU)")
                pipe.compile(
                    text_encode_device="CPU",
                    denoise_device="CPU",
                    vae_device="GPU" if "GPU" in available_devices else "CPU"
                )
            elif mode == "AUTO":
                log_and_print("[INIT] Compiling with AUTO mode...")
                pipe.compile("AUTO")
            else:
                log_and_print(f"[INIT] Compiling for target: {mode}")
                pipe.compile(mode)
        
        log_and_print("[INIT] Compilation finished.")
        
        if not prompt:
            prompt = (
                "A hyper-realistic photograph of a beautiful vibrant exotic flower "
                "blooming directly from the weathered shell of a massive ancient tortoise. "
                "The tortoise is wading through a tranquil crystal-clear pond. "
                "8k resolution, macro shot, cinematic morning light."
            )
        
        log_and_print(f"[INFERENCE] Rendering images (Count: {num_images})...")
        log_and_print(f"[PROMPT] {prompt}")
        
        for i in range(num_images):
            log_and_print(f"[START_IMAGE] {i}")
            # Use provided seed or generate a random base for the batch
            import random
            if seed == -1:
                base_seed = random.randint(0, 2**31 - 1)
            else:
                base_seed = seed
            
            current_seed = base_seed + i
            log_and_print(f"[SEED] {current_seed}")
            
            image_tensor = pipe.generate(
                prompt, 
                width=width, 
                height=height, 
                num_inference_steps=steps if steps is not None else (4 if "schnell" in model_id.lower() else 20),
                guidance_scale=guidance if guidance is not None else (0.0 if "schnell" in model_id.lower() else 3.5),
                seed=current_seed,
                callback=progress_callback
            )
            
            log_and_print(f"[SUCCESS_IMAGE] {i}")
            image = Image.fromarray(image_tensor.data[0])
            
            # Handle filename indexing
            final_output = output_file if output_file else OUTPUT_FILE
            if num_images > 1:
                base, ext = os.path.splitext(final_output)
                indexed_output = f"{base}_{i}{ext}"
            else:
                indexed_output = final_output
                
            image.save(indexed_output)
            log_and_print(f"[SAVED] {os.path.abspath(indexed_output)}")
        log_and_print(f"--- SUCCESS ---")
        log_and_print(f"Image saved to: {os.path.abspath(final_output)}")
        
    except Exception as e:
        log_and_print(f"[FATAL] Inference failed: {str(e)}")
        logging.exception("Detailed Traceback:")
        raise e # Re-raise to trigger exit code 1

def main():
    log_and_print("--- Intel Ultra Stable Inference ---")
    
    # Defaults
    mode = "STABLE_HYBRID"
    threshold = 4.0
    prompt = None
    output = None
    model_id = DEFAULT_MODEL_ID
    hf_token = None
    steps = None
    guidance = None
    enhance = False
    width = 512
    height = 512
    
    # Simple argument handling
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i].upper()
        if arg in ["GPU", "NPU", "HYBRID", "CPU", "AUTO", "STABLE_HYBRID", "ULTRA_EFF"]:
            mode = arg
        elif arg == "--PROMPT" and i + 1 < len(args):
            prompt = args[i+1]
            i += 1
        elif arg == "--OUTPUT" and i + 1 < len(args):
            output = args[i+1]
            i += 1
        elif arg == "--MODEL" and i + 1 < len(args):
            model_id = args[i+1]
            i += 1
        elif arg == "--TOKEN" and i + 1 < len(args):
            hf_token = args[i+1]
            i += 1
        elif arg == "--STEPS" and i + 1 < len(args):
            try:
                steps = int(args[i+1])
            except:
                pass
            i += 1
        elif arg == "--GUIDANCE" and i + 1 < len(args):
            try:
                guidance = float(args[i+1])
            except:
                pass
            i += 1
        elif arg == "--BYPASS-SAFETY":
            threshold = 1.0
        elif arg == "--ENHANCE":
            enhance = True
        elif arg == "--NUM_IMAGES" and i + 1 < len(args):
            i += 1
        elif arg == "--SEED" and i + 1 < len(args):
            try:
                seed = int(args[i+1])
            except:
                seed = -1
            i += 1
        elif arg == "--WIDTH" and i + 1 < len(args):
            try:
                width = int(args[i+1])
            except:
                pass
            i += 1
        elif arg == "--HEIGHT" and i + 1 < len(args):
            try:
                height = int(args[i+1])
            except:
                pass
            i += 1
        i += 1

    try:
        download_model(model_id, hf_token)
        run_inference(mode, threshold, prompt, output, model_id, steps, guidance, enhance, width, height, seed)
    except Exception as e:
        log_and_print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
