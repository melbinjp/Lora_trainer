#!/bin/bash
# Smart setup script for Linux/macOS/Colab
# Installs only missing packages, skips those already present (esp. in Colab)
# Detects device and suggests best practices for performance

set -e

# Detect if running in Google Colab
IS_COLAB=0
python -c "import google.colab" 2>/dev/null && IS_COLAB=1

if [ $IS_COLAB -eq 1 ]; then
    echo "[INFO] Running in Google Colab."
    PIP_INSTALL="pip install --upgrade --quiet"
else
    echo "[INFO] Not running in Colab."
    PIP_INSTALL="pip install --upgrade"
fi

# Function to check if a Python package is installed
is_installed() {
    python -c "import $1" 2>/dev/null
}

# Install requirements, skipping those already present
while read req; do
    pkg=$(echo $req | cut -d'=' -f1 | cut -d'>' -f1 | tr '-' '_')
    if [ -z "$pkg" ]; then continue; fi
    echo "[CHECK] $pkg..."
    if python -c "import $pkg" 2>/dev/null; then
        echo "[SKIP] $pkg already installed."
    else
        echo "[INSTALL] $req..."
        $PIP_INSTALL "$req" || {
            echo "[ERROR] Failed to install $req. Try upgrading pip or check your Python version."
            echo "[SUGGESTION] Try: python -m pip install --upgrade pip"
        }
    fi
done < requirements.txt

# Special: clip-interrogator (sometimes needs --no-deps)
if ! python -c "import clip_interrogator" 2>/dev/null; then
    $PIP_INSTALL "clip-interrogator" || $PIP_INSTALL "clip-interrogator --no-deps"
fi

# Suggest device-specific optimizations
PYTHON_DEVICE_CHECK="import torch; print(torch.cuda.is_available(), getattr(torch, 'has_mps', False))"
read -r HAS_CUDA HAS_MPS <<< $(python -c "$PYTHON_DEVICE_CHECK")
if [ "$HAS_CUDA" = "True" ]; then
    echo "[INFO] CUDA GPU detected. For best performance, ensure you have the correct torch version for CUDA."
    echo "[SUGGESTION] See: https://pytorch.org/get-started/locally/"
elif [ "$HAS_MPS" = "True" ]; then
    echo "[INFO] Apple MPS detected. For best performance, use torch>=1.12 on Mac."
fi

echo "[SUCCESS] Setup complete! To run:"
echo "  streamlit run test.py"
