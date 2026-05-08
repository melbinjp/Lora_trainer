import os
import sys

# --- AGGRESSIVE TELEMETRY BLOCKING ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OPENVINO_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["DISABLE_TELEMETRY"] = "1"

# Monkey-patch Hugging Face Hub telemetry to be impossible
try:
    import huggingface_hub.utils._telemetry
    huggingface_hub.utils._telemetry.send_telemetry = lambda *args, **kwargs: None
except:
    pass

# Monkey-patch Streamlit telemetry
try:
    import streamlit.runtime.metrics_util
    streamlit.runtime.metrics_util.track_user_tracker = lambda *args, **kwargs: None
except:
    pass

import logging
import traceback
import psutil
import platform
import streamlit as st

LOG_FILE = os.path.join("outputs", "lora_ui.log")

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        force=True
    )

def clear_log_file():
    with open(LOG_FILE, "w") as f:
        f.truncate(0)

def download_log_file():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Log File", f, file_name=LOG_FILE)

def log_info(msg):
    logging.info(msg)
    st.info(msg)

def log_warning(msg):
    logging.warning(msg)
    st.warning(msg)

def log_error(msg):
    logging.error(msg)
    st.error(msg)

def log_exception(msg):
    logging.exception(msg)
    st.error(msg + "\n" + traceback.format_exc())

def aggressive_flush():
    """Forces all accessible processes to release their working set memory."""
    logging.info("[CLEANER] Performing aggressive RAM flush...")
    count = 0
    import psutil
    import ctypes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            handle = ctypes.windll.kernel32.OpenProcess(0x0108, False, proc.info['pid'])
            if handle:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
                ctypes.windll.kernel32.CloseHandle(handle)
                count += 1
        except:
            pass
    logging.info(f"[CLEANER] Flushed {count} processes.")

def get_system_resources():
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    vram_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        elif hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available():
            vram_gb = None
    except Exception:
        pass
    return ram_gb, vram_gb

def detect_model_type(model_id, hf_token=None):
    """
    Detect model type using config/class inspection if possible, fallback to string heuristic.
    Returns: 'diffusion', 'transformer', or 'unknown'
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, token=hf_token)
        # Check tags in model card
        tags = info.tags if hasattr(info, 'tags') else []
        if any(t in tags for t in ["stable-diffusion", "diffusers", "sdxl", "unet"]):
            return "diffusion"
        if any(t in tags for t in ["causal-lm", "text-generation", "transformers", "bert", "gpt", "llama", "bloom", "t5", "roberta", "deberta"]):
            return "transformer"
        # Try to load config and check class
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, use_auth_token=hf_token)
            if hasattr(config, 'architectures'):
                archs = [a.lower() for a in config.architectures or []]
                if any(x in archs for x in ["unet", "diffusion"]):
                    return "diffusion"
                if any(x in archs for x in ["causallm", "gpt", "bert", "llama", "bloom", "t5", "roberta", "deberta"]):
                    return "transformer"
        except Exception:
            pass
    except Exception:
        pass
    # Fallback to string heuristic
    model_id_lower = model_id.lower() if model_id else ""
    if any(x in model_id_lower for x in ["stable-diffusion", "sdxl", "sd-turbo", "unet"]):
        return "diffusion"
    if any(x in model_id_lower for x in ["bert", "gpt", "llama", "bloom", "t5", "roberta", "deberta", "gpt2", "gpt-neo", "gpt-j", "gpt3", "gpt4", "opt", "falcon", "mistral"]):
        return "transformer"
    return "unknown"
