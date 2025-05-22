@echo off
REM Windows setup script for Streamlit LoRA UI
REM Checks for each package before installing, suggests fixes on error

setlocal enabledelayedexpansion

for /f "usebackq tokens=*" %%a in (requirements.txt) do (
    set "line=%%a"
    if not "!line!"=="" (
        for /f "tokens=1 delims==> " %%b in ("!line!") do (
            set "pkg=%%b"
            echo [CHECK] !pkg!...
            python -c "import !pkg!" 2>NUL
            if errorlevel 1 (
                echo [INSTALL] %%a...
                pip install --upgrade %%a || (
                    echo [ERROR] Failed to install %%a. Try upgrading pip or check your Python version.
                    echo [SUGGESTION] Try: python -m pip install --upgrade pip
                )
            ) else (
                echo [SKIP] !pkg! already installed.
            )
        )
    )
)

REM Special: clip-interrogator
python -c "import clip_interrogator" 2>NUL
if errorlevel 1 (
    pip install --upgrade clip-interrogator || pip install --upgrade clip-interrogator --no-deps
)

REM Suggest device-specific optimizations
python -c "import torch; print('[INFO] CUDA:', torch.cuda.is_available()); print('[INFO] MPS:', getattr(torch.backends.mps, 'is_built', lambda: False)())"

ECHO [SUCCESS] Setup complete! To run:
ECHO   streamlit run test.py
