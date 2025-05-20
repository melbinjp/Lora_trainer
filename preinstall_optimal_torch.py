#!/usr/bin/env python3
"""
preinstall_optimal_torch.py

Detects the best available hardware (CUDA, ROCm, MPS, CPU) and prints the recommended pip install command for torch/accelerator.
Optionally, can auto-install torch for you (use --install flag).
Safe to run before requirements.txt or other setup scripts.

Usage:
  python preinstall_optimal_torch.py           # Just print the recommended command
  python preinstall_optimal_torch.py --install # Print and run the install command

Works in Colab, Windows, Linux, Mac. Handles Colab, CUDA, ROCm, MPS, CPU.
"""
import sys
import os
import platform
import subprocess
import shutil

# --- Helper: Run shell command ---
def run(cmd):
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)

# --- Detect Colab ---
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# --- Detect hardware ---
def detect_hardware():
    # CUDA
    if shutil.which("nvidia-smi"):
        return "cuda"
    # ROCm (AMD)
    if shutil.which("rocminfo") or os.path.exists("/opt/rocm"):
        return "rocm"
    # Apple MPS
    if platform.system() == "Darwin":
        try:
            import torch
            if hasattr(torch, "has_mps") and torch.has_mps:
                return "mps"
        except ImportError:
            return "mps"  # Assume MPS available on Apple Silicon
    return "cpu"

# --- Recommend torch install command ---
def get_torch_command(hardware, py_version):
    base = f"pip install torch torchvision torchaudio"
    if hardware == "cuda":
        # Use latest stable CUDA (12.1 as of May 2025)
        return f"{base} --index-url https://download.pytorch.org/whl/cu121"
    elif hardware == "rocm":
        # ROCm 5.7 is latest as of May 2025
        return f"{base} --index-url https://download.pytorch.org/whl/rocm5.7"
    elif hardware == "mps":
        return base  # MPS is included in Mac wheels
    else:
        return base  # CPU only

# --- Main logic ---
def main():
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    hardware = detect_hardware()
    print(f"[INFO] Detected Python: {py_version}")
    print(f"[INFO] Detected hardware: {hardware.upper()}")
    cmd = get_torch_command(hardware, py_version)
    print(f"[RECOMMEND] To install optimal torch/accelerator, run:")
    print(f"  {cmd}")
    if "--install" in sys.argv:
        print("[INFO] Installing optimal torch/accelerator...")
        run(cmd)
        print("[SUCCESS] torch/accelerator installed.")
    else:
        print("[INFO] To auto-install, run: python preinstall_optimal_torch.py --install")

if __name__ == "__main__":
    main()
