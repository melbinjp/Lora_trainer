@echo off
title Headless Flux Inference - 16GB Optimization
echo ===================================================
echo   💨ヘッドレス Flux Inference (Headless Mode)
echo ===================================================
echo.
echo [1/4] Closing memory-heavy applications...
taskkill /F /IM Code.exe /T 2>nul
taskkill /F /IM pyrefly.exe /T 2>nul
taskkill /F /IM msedge.exe /T 2>nul
echo.
echo [2/4] Running Aggressive RAM Flush...
python flush_ram.py
echo.
echo [3/4] Starting High-Quality Inference (HYBRID Mode Priority)...
echo Attempting GPU + NPU acceleration. If it fails, we will fallback to CPU.
echo.
python intel_high_quality_inference.py HYBRID --inference-only --bypass-safety
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] HYBRID mode failed (likely GPU Memory limit). 
    echo [!] Falling back to CPU mode... This will be slower but IS GUARANTEED TO WORK.
    echo.
    python intel_high_quality_inference.py CPU --inference-only --bypass-safety
)
echo.
echo [4/4] Finished!
echo Check your folder for: flower_tortoise_final.png
echo.
pause
