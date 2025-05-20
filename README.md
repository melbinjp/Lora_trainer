# LoRA Streamlit UI - Setup & Run Instructions

## Local (Windows, Linux, macOS)

1. Clone this repository:
   ```powershell
   git clone https://github.com/melbinjp/Lora_trainer
   cd Lora_trainer
   ```
2. Run the setup script:
   - On Linux/macOS:
     ```sh
     bash setup.sh
     ```
   - On Windows:
     ```bat
     setup.bat
     ```
3. Start the app:
   ```sh
   streamlit run test.py
   ```

## Google Colab (One-click)

1. Open `colab_quickstart.ipynb` in Colab.
2. Run all cells. The UI will be available via a public link (look for the `trycloudflare.com` URL in the output).
3. (Optional) For persistent tunnels, see the Cloudflared Auth section below.

### Hugging Face Token (for private/custom models)
- Go to https://huggingface.co/settings/tokens
- Click "New token" (read access is enough)
- Copy the token and paste it into the Hugging Face Token field in the Streamlit sidebar (left side of the UI).

### Public Link via Cloudflared (Colab)
- The Colab notebook automatically installs and runs Cloudflared to create a public link.
- The public URL will be printed in the Colab output (look for `https://...trycloudflare.com`).
- No account or login is required for basic use.

#### (Optional) Persistent Cloudflared Tunnel
- For advanced users who want a persistent, custom subdomain:
  1. Sign up at https://dash.cloudflare.com/ and add a site (free tier is fine).
  2. Follow the Cloudflared docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/
  3. Authenticate with `cloudflared tunnel login` (not needed for basic Colab use).

## Cloud VM (AWS, GCP, etc.)

1. Clone your repo and enter the `Lora_trainer` folder.
2. Run `bash setup.sh` (Linux) or `setup.bat` (Windows).
3. Start Streamlit as above.

---

- The setup scripts will skip already-installed packages (especially in Colab) to save time.
- If you encounter errors, try re-running the setup script or manually installing the missing package.
- For custom ports or public sharing, see Streamlit docs or add `--server.port=XXXX` to the run command.

## Troubleshooting & Best Practices

### Common Issues

- **Package install errors**: If a package fails to install, try upgrading pip:
  ```sh
  python -m pip install --upgrade pip
  ```
  Then re-run the setup script.
- **torch/transformers version mismatch**: For best performance, install the correct torch version for your device (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).
- **Colab: Out of memory**: Restart the runtime and try again, or reduce batch size in advanced settings.
- **Windows: Permission errors**: Run your terminal as Administrator.
- **Apple Silicon (M1/M2/M3)**: Use torch>=1.12 for MPS support.

### Best Practices

- Use a virtual environment (venv, conda) to avoid dependency conflicts.
- For production, pin package versions in `requirements.txt`.
- Always check device selection in the UI; training is much faster on GPU/NPU/MPS.
- If you encounter errors, check the Streamlit logs and error messages for suggestions.
- For cloud/Colab, avoid reinstalling packages unless necessary to save time.

---
