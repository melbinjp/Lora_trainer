# LoRA Streamlit UI - Professional AI Training & Inference

A modular, high-performance toolkit for LoRA training (via Google Colab/Cloud) and optimized Intel-native inference (via OpenVINO).

## 📂 Project Structure

- `src/app.py`: Main Streamlit UI entry point.
- `src/engine/`: Core inference engines (Standard & Intel-optimized).
- `src/utils/`: Shared helper functions and memory management.
- `notebooks/`: Jupyter notebooks for quickstart and custom training.
- `scripts/`: Setup, installation, and utility scripts.
- `docs/`: Guides and walkthroughs.
- `tests/`: Unit and integration testing suite.
- `data/`: Local storage for training datasets.
- `outputs/`: Default directory for generated images and logs.

## 🚀 Quick Start (Local)

1. **Clone & Setup:**
   ```powershell
   git clone https://github.com/melbinjp/Lora_trainer
   cd Lora_trainer
   # Windows
   .\scripts\setup.bat
   # Linux/macOS
   bash scripts/setup.sh
   ```

2. **Run the App:**
   ```powershell
   streamlit run src/app.py
   ```

## ☁️ Google Colab (Cloud Training)

1. Open `colab_quickstart.ipynb` in Google Colab.
2. Run the cells to launch the UI and begin training.
3. The UI will be accessible via a secure `trycloudflare.com` tunnel.

## 🖥️ Intel-Native Optimized Inference

This project includes specialized support for **Intel Core Ultra** systems (Meteor Lake/Lunar Lake) using OpenVINO.

- **Stable Hybrid Mode:** Automatically partitions the Flux.1 model between CPU and iGPU to ensure stability on 16GB RAM systems.
- **Real-Time UI Progress:** Streamlit UI now displays live progress bars and status updates during inference.
- **Dynamic Prompts:** Generate images directly from the UI or via CLI:
  ```powershell
  python src/engine/intel_inference.py STABLE_HYBRID --prompt "your prompt" --output outputs/result.png
  ```

## 🌐 API & Distributed Processing

A lightweight API is included to allow programmatic access to the inference engine. This is ideal for distributed networks, batch processing, or integrating with other applications.

1. **Start the API Server:**
   ```powershell
   python src/api.py
   ```
2. **Send a Generation Request (example using curl or Postman):**
   ```bash
   curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "A futuristic city", "model": "OpenVINO/FLUX.1-schnell-int4-ov", "hardware_target": "STABLE_HYBRID"}' --output result.png
   ```

## 🛠️ Features & Tools

### Dataset Engineering
- **Auto-Captioning:** Uses BLIP to automatically generate captions for your training images.
- **Export:** Packages your dataset into an optimized zip file ready for cloud training.

### Memory Management
- **Aggressive RAM Flush:** Includes utilities to clear system memory before intensive inference runs.
- **Resource Detection:** Automatically detects available RAM and VRAM to suggest optimal settings.

### Cloud Integration
- **Hugging Face:** Seamlessly download/upload models using HF Hub integration.
- **Cloud Storage:** Mount Google Drive, S3, or OneDrive for persistent model storage.

## 🧪 Testing

Run the test suite to ensure everything is configured correctly:
```powershell
pytest tests/unit/test_logic.py
```

## 📜 Documentation

For more detailed information, see:
- [Walkthrough Guide](docs/walkthrough.md)
- [Setup Instructions](README.md)

---
*Created with ❤️ for AI researchers and enthusiasts.*
