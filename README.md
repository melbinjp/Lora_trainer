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

## Cloud Storage Mounting (Optional, for Custom Models)

If you want to use models from Google Drive, OneDrive, AWS S3, Azure, or GCP, run the provided script **before** starting the Streamlit UI:

```sh
python mount_cloud_storage.py
```
- The script will let you choose which cloud storage(s) to mount (multiple allowed, or none for default use).
- For Google Drive (Colab), it will run the mount code. For other providers, it will guide you to use rclone.
- After mounting, start the Streamlit app and enter the path to your model in the UI.
- If you skip this step, you can still use Hugging Face or local upload as usual.

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

## Using Custom Models from Hugging Face

You can use your own models (captioning, tagging, base, or LoRA) from Hugging Face, including private repos. This is useful for persistent storage and sharing.

### Uploading Local Models to Hugging Face
1. [Create a Hugging Face account](https://huggingface.co/join) if you don't have one.
2. [Create a new repository](https://huggingface.co/new) (choose 'Model').
3. Upload your model files (or folder) via the web UI, or use the `huggingface_hub` Python library:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path="/path/to/your/model_dir",
       repo_id="your-username/your-model-repo",
       token="YOUR_HF_TOKEN"
   )
   ```
   - For private repos, set the repo visibility to 'private' when creating the repo.
   - You can also upload zipped folders.

### Using Repo IDs in the UI/Notebook
- In the Streamlit UI, enter the Hugging Face repo ID (e.g., `your-username/your-model-repo`) in the custom model field.
- For private repos, paste your Hugging Face token in the sidebar field.
- The app will automatically use your token for authentication.

---

## Using Cloud Storage for Models (Google Drive, OneDrive, AWS S3, Azure, GCP)

You can load models from cloud storage for persistent and large-scale use. Supported sources:
- **Google Drive**
- **OneDrive**
- **AWS S3**
- **Azure Blob Storage**
- **Google Cloud Storage (GCS)**

### Mounting Cloud Storage
- Run `python mount_cloud_storage.py` and follow the prompts to mount your storage.
- For Google Drive (Colab), the script will run the mount code for you.
- For OneDrive/S3/Azure/GCP, the script will prompt for your rclone remote and mount point.
- After mounting, enter the path to your model in the Streamlit UI (e.g., `/content/drive/MyDrive/my_model_dir`).

### Google Drive (Colab & Local)
- **Colab:**
  1. In Colab, run:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
  2. Your Drive files will be available at `/content/drive/MyDrive/`.
  3. In the Streamlit UI, enter the path to your model (e.g., `/content/drive/MyDrive/my_model_dir`).
- **Local:**
  1. Install [Google Drive for Desktop](https://www.google.com/drive/download/).
  2. Your Drive will be mounted as a local folder (e.g., `G:/My Drive/`).
  3. Enter the path in the UI.

### OneDrive (Colab & Local)
- **Colab:**
  1. Use [rclone](https://rclone.org/onedrive/) to mount OneDrive (see rclone docs).
  2. Enter the mounted path in the UI.
- **Local:**
  1. Install OneDrive client and sync your files.
  2. Enter the local path in the UI.

### AWS S3, Azure Blob, GCP Storage
- Use [rclone](https://rclone.org/) to mount these cloud buckets as local folders in Colab or on your machine.
- Example (Colab):
  ```python
  !pip install rclone
  !rclone config  # Follow prompts to set up your remote
  !rclone mount remote:bucket /content/bucket &
  ```
- Enter the mounted path (e.g., `/content/bucket/my_model_dir`) in the UI.

### Authentication
- For cloud storage, follow the provider's authentication steps (OAuth, access keys, etc.).
- For Hugging Face, paste your token in the sidebar for private repos.

---

## Persistent Model Storage & Reuse

- **After uploading a model to Hugging Face:**
  - The UI will display the repo URL (e.g., `https://huggingface.co/your-username/your-model-repo`).
  - To reuse the model, enter the repo ID (`your-username/your-model-repo`) in the custom model field in the UI or notebook.
- **For cloud storage:**
  - Enter the full path to your model directory in the UI (e.g., `/content/drive/MyDrive/my_model_dir`).
  - The app will use the model from that location.

---

## UI Options for Model Selection/Loading

- **Hugging Face:** Enter a repo ID for any model (public or private, with token).
- **Local Upload:** Upload a `.zip` of your model folder in the advanced UI section.
- **Cloud Storage:** Enter the path to your mounted cloud storage folder in the custom model field.
- The app will prioritize local uploads, then cloud/local paths, then Hugging Face.

For more details, see the advanced UI section in the app and the comments in `test.py`.
