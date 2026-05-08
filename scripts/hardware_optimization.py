#!/usr/bin/env python3
"""
hardware_optimization.py

Surgical hardware detection and optimization for LoRA Trainer.
Ensures only necessary dependencies are installed for the detected hardware.
Supported: Intel Core Ultra (iGPU/NPU), NVIDIA (CUDA), Apple Silicon (MPS), CPU.
"""
import sys
import os
import platform
import subprocess
import shutil

def run(cmd):
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        return False
    return True

def is_intel_ultra():
    """Heuristic to detect Intel Core Ultra (Meteor Lake/Lunar Lake)."""
    if platform.system() != "Windows":
        return False # Add Linux detection if needed
    try:
        # Check for NPU in device manager or via OpenVINO if already present
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "NPU" in devices or ("GPU" in devices and "Intel" in core.get_property("GPU", "FULL_DEVICE_NAME")):
            return True
    except:
        # Fallback to system info
        try:
            output = subprocess.check_output("wmic cpu get name", shell=True).decode()
            if "Ultra" in output or "Intel" in output:
                return True
        except:
            pass
    return False

def detect_hardware():
    if shutil.which("nvidia-smi"):
        return "cuda"
    if platform.system() == "Darwin":
        return "mps"
    if is_intel_ultra():
        return "intel"
    return "cpu"

def install_torch(hardware):
    print(f"[INFO] Optimizing Torch for: {hardware.upper()}")
    base = "pip install torch torchvision torchaudio"
    if hardware == "cuda":
        cmd = f"{base} --index-url https://download.pytorch.org/whl/cu121"
    elif hardware == "intel":
        # Intel Core Ultra often benefits from stock torch or intel-extension-for-pytorch
        # For our OpenVINO workflow, stock torch is fine as OpenVINO handles the optimization
        cmd = f"{base}"
    else:
        cmd = f"{base}"
    
    run(cmd)

def install_hardware_extras(hardware):
    if hardware == "intel":
        print("[INFO] Installing Intel-specific optimizations (OpenVINO GenAI)...")
        intel_req = os.path.join(os.path.dirname(__file__), "requirements_intel.txt")
        if os.path.exists(intel_req):
            run(f"pip install -r {intel_req}")
        else:
            run("pip install openvino>=2026.1.0 openvino-genai>=2026.1.0.0 openvino-tokenizers>=2026.1.0.0")
    elif hardware == "cuda":
        print("[INFO] Installing CUDA-specific optimizations (xformers)...")
        run("pip install xformers")

def main():
    hardware = detect_hardware()
    print(f"--- Hardware Optimization Strategy: {hardware.upper()} ---")
    
    if "--install" in sys.argv:
        install_torch(hardware)
        install_hardware_extras(hardware)
        print(f"[SUCCESS] Hardware-specific optimization for {hardware.upper()} complete.")
    else:
        print("[INFO] This script will install optimal torch and hardware-specific extras.")
        print("[INFO] Run with --install to apply optimizations.")

if __name__ == "__main__":
    main()
