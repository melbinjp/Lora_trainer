#!/bin/bash
# Surgical setup script for LoRA Streamlit UI
# Detects hardware and only installs necessary dependencies.

set -e

echo "[STEP 1] Optimizing for hardware..."
python3 scripts/hardware_optimization.py --install

echo "[STEP 2] Installing core dependencies..."
pip install -r requirements.txt

# Special: clip-interrogator
if ! python3 -c "import clip_interrogator" 2>/dev/null; then
    echo "[INSTALL] clip-interrogator..."
    pip install --upgrade clip-interrogator --no-deps
fi

echo "[SUCCESS] Setup complete! To run:"
echo "  streamlit run src/app.py"
