@echo off
REM Surgical setup script for LoRA Streamlit UI
REM Detects hardware and only installs necessary dependencies.

setlocal enabledelayedexpansion

echo [STEP 1] Optimizing for hardware...
python scripts/hardware_optimization.py --install

echo [STEP 2] Installing core dependencies...
pip install -r requirements.txt

REM Special: clip-interrogator
python -c "import clip_interrogator" 2>NUL
if errorlevel 1 (
    echo [INSTALL] clip-interrogator...
    pip install --upgrade clip-interrogator --no-deps
)

echo [SUCCESS] Setup complete! To run:
echo   streamlit run src/app.py
