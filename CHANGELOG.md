# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-05-06

### Added
- **Intel Core Ultra Optimization:** Introduced `STABLE_HYBRID` mode for stable Flux.1 inference on 16GB RAM systems using OpenVINO (iGPU acceleration + CPU stability).
- **Professional Directory Structure:** Refactored project into `src/`, `scripts/`, `tests/`, `docs/`, and `outputs/`.
- **Surgical Hardware Detection:** Added `scripts/hardware_optimization.py` to automatically detect and install only necessary dependencies for Intel, NVIDIA, and Mac systems.
- **Improved UI Transparency:** The Streamlit UI now displays real-time local availability status for all supported models.
- **Enhanced Logging:** Consolidated all logs into `outputs/` with granular debugging info.
- **Static Shape Reshaping:** Implemented static shape enforcement for OpenVINO models to improve performance and stability.
- **NPU Fallback Logic:** Added automated fallback from NPU/GPU to stable modes if compilation fails.

### Changed
- Moved main Streamlit entry point to `src/app.py`.
- Renamed and modularized inference engines into `src/engine/`.
- Updated all setup scripts and notebooks to reflect the new modular architecture.
- Optimized default inference resolution to 512x512 for consistent performance on 16GB systems.

### Fixed
- Resolved `CL_INVALID_EVENT` and shader compilation crashes on Intel GPUs.
- Fixed `ImportError` in unit tests caused by incorrect file path references.
- Corrected Streamlit session state issues during auto-captioning.
- Removed all "junk" files and redundant scripts from the root directory.

### Removed
- Removed large/unused files from root to improve repository cleanliness.
- Deleted all `.bak` and temporary log files.
