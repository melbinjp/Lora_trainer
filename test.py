import streamlit as st
import platform
import psutil
import shutil
import requests
from PIL import Image
import io
from huggingface_hub import snapshot_download
from transformers import BlipProcessor, BlipForConditionalGeneration
from clip_interrogator import Config, Interrogator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from peft import LoraConfig
import threading
import time
from huggingface_hub import hf_hub_download
import os
import random
from datetime import datetime
from huggingface_hub import HfApi
import logging
import traceback
import subprocess # Added for train_lora

# --- Logging setup ---
LOG_FILE = "lora_ui.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

def clear_log_file():
    with open(LOG_FILE, "w") as f:
        f.truncate(0)

def download_log_file():
    with open(LOG_FILE, "rb") as f:
        st.download_button("Download Log File", f, file_name=LOG_FILE)

# Helper to log and show in Streamlit
def log_info(msg):
    logging.info(msg)
    # st.info(msg) # Reduce streamlit clutter for verbose logs

def log_warning(msg):
    logging.warning(msg)
    st.warning(msg)

def log_error(msg):
    logging.error(msg)
    st.error(msg)

def log_exception(msg):
    logging.exception(msg)
    st.error(msg + "\n" + traceback.format_exc())

# Try to import torch for GPU info, if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Default settings and descriptions ---
DEFAULT_LORA_METHOD = {
    "name": "kohya-ss LoRA",
    "desc": "Efficient and widely used LoRA training for diffusion models.",
    "compatible_base": ["Stable Diffusion v1.5", "SDXL", "Stable Diffusion 2.x"],
    "default": True
}
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# --- Caching for Model Files Only (not full model objects) ---
@st.cache_resource(show_spinner=False)
def get_blip_model_files(hf_token, blip_model_id, local_path=None):
    if local_path and os.path.exists(local_path):
        processor = BlipProcessor.from_pretrained(local_path)
        model = BlipForConditionalGeneration.from_pretrained(local_path)
    else:
        processor = BlipProcessor.from_pretrained(
            blip_model_id,
            use_auth_token=hf_token
        )
        model = BlipForConditionalGeneration.from_pretrained(
            blip_model_id,
            use_auth_token=hf_token
        )
    return processor, model

@st.cache_resource(show_spinner=False)
def get_clip_interrogator_files(clip_model_id, local_path=None):
    if local_path and os.path.exists(local_path):
        ci = Interrogator(Config(clip_model_name=local_path))
    else:
        ci = Interrogator(Config(clip_model_name=clip_model_id))
    return ci

# --- Updated Caption/Tag Generation Functions with Progress Bar ---
def generate_captions_for_images(images, hf_token, blip_model_id, local_blip_path=None):
    log_info(f"Loading BLIP model for captioning: {blip_model_id if not local_blip_path else local_blip_path}")
    progress_bar = st.progress(0)
    processor, model = get_blip_model_files(hf_token, blip_model_id, local_blip_path)
    progress_bar.progress(50)
    time.sleep(0.2)
    progress_bar.progress(100)
    captions = []
    for img_file in images:
        image = Image.open(img_file).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    progress_bar.empty()
    return captions

def generate_tags_for_images(images, clip_model_id, local_clip_path=None):
    log_info(f"Loading CLIP Interrogator for tagging: {clip_model_id if not local_clip_path else local_clip_path}")
    progress_bar = st.progress(0)
    ci = get_clip_interrogator_files(clip_model_id, local_clip_path)
    progress_bar.progress(50)
    time.sleep(0.2)
    progress_bar.progress(100)
    tags = []
    for img_file in images:
        image = Image.open(img_file).convert('RGB')
        tag = ci.interrogate(image)
        tags.append(tag)
    progress_bar.empty()
    return tags

def search_hf_models(query, hf_token):
    """
    Search Hugging Face Hub for models matching the query.
    Returns a list of dicts with modelId, name, description, etc.
    """
    api_url = f"https://huggingface.co/api/models?search={query}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    resp = requests.get(api_url, headers=headers)
    if resp.status_code == 200:
        models = resp.json()
        results = []
        for m in models[:10]:  # Limit to 10 results for UI clarity
            results.append({
                "modelId": m.get("modelId", ""),
                "name": m.get("modelId", ""),
                "desc": m.get("pipeline_tag", ""),
                "tags": ", ".join(m.get("tags", []))
            })
        return results
    else:
        return []

def detect_model_type(model_id, hf_token=None):
    """
    Detect model type using config/class inspection if possible, fallback to string heuristic.
    Returns: 'diffusion', 'transformer', or 'unknown'
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, token=hf_token)
        # Check tags in model card
        tags = info.tags if hasattr(info, 'tags') else []
        if any(t in tags for t in ["stable-diffusion", "diffusers", "sdxl", "unet"]):
            return "diffusion"
        if any(t in tags for t in ["causal-lm", "text-generation", "transformers", "bert", "gpt", "llama", "bloom", "t5", "roberta", "deberta"]):
            return "transformer"
        # Try to load config and check class
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, use_auth_token=hf_token)
            if hasattr(config, 'architectures'):
                archs = [a.lower() for a in config.architectures or []]
                if any(x in archs for x in ["unet", "diffusion"]):
                    return "diffusion"
                if any(x in archs for x in ["causallm", "gpt", "bert", "llama", "bloom", "t5", "roberta", "deberta"]):
                    return "transformer"
        except Exception:
            pass
    except Exception:
        pass
    # Fallback to string heuristic
    model_id_lower = model_id.lower() if model_id else ""
    if any(x in model_id_lower for x in ["stable-diffusion", "sdxl", "sd-turbo", "unet"]):
        return "diffusion"
    if any(x in model_id_lower for x in ["bert", "gpt", "llama", "bloom", "t5", "roberta", "deberta", "gpt2", "gpt-neo", "gpt-j", "gpt3", "gpt4", "opt", "falcon", "mistral"]):
        return "transformer"
    return "unknown"

# --- Placeholder for get_best_device_and_precision ---
# This function should be defined elsewhere or imported.
# For now, a basic placeholder to avoid NameError:
def get_best_device_and_precision():
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda", torch.float16, "CUDA device detected.", torch.__version__
        elif hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available():
             # MPS currently has limitations, often safer to default to float32 or check compatibility
            return "mps", torch.float32, "MPS device detected (Apple Silicon).", torch.__version__ 
    return "cpu", torch.float32, "CPU device detected.", "N/A" if not TORCH_AVAILABLE else torch.__version__


with st.expander("4. Advanced Training Settings", expanded=False):
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=4)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%e")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    optimizer = st.selectbox("Optimizer", ["AdamW", "SGD"], index=0) # Add more if Kohya supports easily
    resolution = st.text_input("Resolution (e.g., 512,512 or 768,768)", value="512,512")
    lora_rank = st.number_input("LoRA Rank (network_dim)", min_value=1, max_value=256, value=8) # Increased default and max
    lora_alpha = st.number_input("LoRA Alpha (network_alpha)", min_value=1, max_value=256, value=lora_rank) # Default alpha to rank
    save_every_n_epochs = st.number_input("Save every N epochs", min_value=1, max_value=epochs, value=1)
    lr_scheduler = st.selectbox("LR Scheduler", ["cosine_with_restarts", "linear", "constant", "cosine"], index=0)
    gradient_checkpointing = st.checkbox("Enable Gradient Checkpointing (reduces VRAM)", value=False)
    noise_offset = st.number_input("Noise Offset (recommended 0.1 for SDXL, 0 for SD1.5/2.x, optional)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    min_snr_gamma = st.number_input("Min SNR Gamma (e.g., 5, optional, for improved training stability)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)


    # --- New: Expose LoRA adapter name, adapter weights, and allow multiple adapters ---
    # These are more advanced Diffusers/PEFT concepts, may not directly map to Kohya sd-scripts easily
    # For Kohya, output_name is the primary way to name the LoRA.
    # Multiple adapters are not typically trained simultaneously in one Kohya command in the same way.
    # Retaining for potential future use or if a diffusers backend is added.
    lora_adapter_name_ui = st.text_input("LoRA Adapter Name (for PEFT/Diffusers, usually 'default')", value="default")
    # enable_multi_adapter = st.checkbox("Enable Multiple Adapters (PEFT/Diffusers)", value=False)
    # adapter_names = [lora_adapter_name_ui]
    # adapter_weights = [1.0]
    # if enable_multi_adapter:
    #     adapter_names = st.text_area("Adapter Names (comma-separated)", value=lora_adapter_name_ui).split(",")
    #     adapter_weights = st.text_area("Adapter Weights (comma-separated, same order)", value=",".join(["1.0"]*len(adapter_names))).split(",")
    #     adapter_names = [n.strip() for n in adapter_names if n.strip()]
    #     adapter_weights = [float(w.strip()) for w in adapter_weights if w.strip()]
    
    adv_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "optimizer": optimizer,
        "resolution": resolution,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "save_every_n_epochs": save_every_n_epochs,
        "lr_scheduler": lr_scheduler,
        "gradient_checkpointing": gradient_checkpointing,
        "noise_offset": noise_offset if noise_offset > 0 else None, # Pass None if 0
        "min_snr_gamma": min_snr_gamma if min_snr_gamma > 0 else None, # Pass None if 0
        # "lora_adapter_name": lora_adapter_name_ui, # For PEFT
        # "adapter_names": adapter_names, # For PEFT
        # "adapter_weights": adapter_weights, # For PEFT
        # "enable_multi_adapter": enable_multi_adapter # For PEFT
    }

# --- Sidebar: Hugging Face Token (Optional for default model) ---
colab_hf_token = None
try:
    # Try Colab secret storage first
    try:
        from google.colab import userdata
        colab_hf_token = userdata.get('HF_TOKEN')
        if colab_hf_token:
            log_info('Hugging Face token extracted from Colab secret storage (userdata).')
    except Exception:
        pass
    # Fallback to environment variable
    if not colab_hf_token:
        colab_hf_token = os.environ.get('HF_TOKEN', None)
        if colab_hf_token:
            log_info('Hugging Face token extracted from environment successfully (HF_TOKEN).')
    if not colab_hf_token:
        log_info('No Hugging Face token found in Colab secrets or environment. You can enter it below.')
except Exception as e:
    log_exception(f'Error checking for Hugging Face token: {e}')

hf_token_global = st.sidebar.text_input(
    "Hugging Face Token",
    type="password",
    key="hf_token_global",
    value=colab_hf_token if colab_hf_token else ""
)

# --- Log management UI ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Log Management")
if st.sidebar.button("Clear Log File"):
    clear_log_file()
    st.sidebar.success("Log file cleared.")
download_log_file()

# --- Optional: Email Log for Support ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Email Log for Support (Optional)")
support_email = st.sidebar.text_input("Support Email Address (optional)", value="", key="support_email")
if support_email:
    st.sidebar.markdown("You can send the log file to support for debugging. SMTP setup required.")
    smtp_host = st.sidebar.text_input("SMTP Host", value=os.environ.get("SMTP_HOST", ""), key="smtp_host")
    smtp_port = st.sidebar.number_input("SMTP Port", min_value=1, max_value=65535, value=int(os.environ.get("SMTP_PORT", 587)), key="smtp_port")
    smtp_user = st.sidebar.text_input("SMTP Username", value=os.environ.get("SMTP_USER", ""), key="smtp_user")
    smtp_pass = st.sidebar.text_input("SMTP Password", type="password", value=os.environ.get("SMTP_PASS", ""), key="smtp_pass")
    sender_email = st.sidebar.text_input("Sender Email", value=os.environ.get("SENDER_EMAIL", ""), key="sender_email")
    if st.sidebar.button("Send Log File via Email"):
        try:
            import smtplib
            from email.message import EmailMessage
            with open(LOG_FILE, "rb") as f:
                log_data = f.read()
            msg = EmailMessage()
            msg["Subject"] = "LoRA UI Log File"
            msg["From"] = sender_email or smtp_user
            msg["To"] = support_email
            msg.set_content("Attached is the LoRA UI log file for support/debugging.")
            msg.add_attachment(log_data, maintype="application", subtype="octet-stream", filename="lora_ui.log")
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            st.sidebar.success(f"Log file sent to {support_email}.")
            log_info(f"Log file sent to {support_email} via email.")
        except Exception as e:
            log_exception(f"Failed to send log file via email: {e}\nCheck SMTP settings and network connectivity.")
            st.sidebar.error("Failed to send email. See logs for details.")

# --- System Resource Detection ---
def get_system_resources():
    # import psutil # Already imported globally
    # import platform # Already imported globally
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    vram_gb = None
    try:
        # import torch # Already imported globally
        if TORCH_AVAILABLE and torch.cuda.is_available():
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        # MPS VRAM is not easily/reliably queryable in the same way
    except Exception: # Broad exception to catch any issues during VRAM detection
        pass # vram_gb remains None
    return ram_gb, vram_gb

ram_gb, vram_gb = get_system_resources()
st.sidebar.markdown(f"**Detected RAM:** {ram_gb} GB")
if vram_gb:
    st.sidebar.markdown(f"**Detected VRAM:** {vram_gb} GB")
else:
    st.sidebar.markdown("**VRAM:** Not detected or N/A (e.g., CPU, MPS)")

manual_ram = st.sidebar.number_input("Override RAM (GB, optional)", min_value=1.0, value=ram_gb, step=0.5)
manual_vram = st.sidebar.number_input("Override VRAM (GB, optional)", min_value=0.0, value=vram_gb or 0.0, step=0.5)

# --- Model Selection with Resource Check ---
def get_default_model_for_resources(ram, vram):
    # Example: You can expand this logic for more models
    if vram and vram >= 8: # Assuming 8GB VRAM is decent for SD 1.5
        return {
            "modelId": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion v1.5",
            "desc": "Popular base model for image generation and LoRA training.",
            "compatible_lora": ["kohya-ss", "diffusers"],
            "default": True
        }
    else: # Fallback for lower VRAM or CPU/MPS
        return {
            "modelId": "stabilityai/sd-turbo", # SD Turbo is very efficient
            "name": "SD Turbo (Efficient)",
            "desc": "Very efficient, fast, and low-memory model for image generation and LoRA training.",
            "compatible_lora": ["kohya-ss", "diffusers"], # Assuming Kohya scripts can handle it
            "default": True
        }

DEFAULT_BASE_MODEL = get_default_model_for_resources(manual_ram, manual_vram)

# --- Main UI ---
st.title("LoRA Training UI (Easy & Advanced Modes)")

# --- Step 1: Upload Images ---
st.subheader("1. Upload Images")
images = st.file_uploader("Upload images for LoRA training", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if images:
    # Always show images in the UI
    st.write("### Uploaded Images")
    # --- Auto-caption/tag logic ---
    auto_captions = st.session_state.get('auto_captions', None)
    captions_ready = auto_captions is not None and len(auto_captions) == len(images)
    if caption_mode == "Automatic (Recommended)": # This button is for on-demand captioning
        log_info(f"Using {'BLIP' if caption_type == 'Caption (Sentence)' else 'CLIP Interrogator'} model for automatic {caption_type.lower()}.")
        if st.button("Auto Caption/Tag All Images Now"): # Changed button label for clarity
            with st.spinner("Downloading captioning/tagging model and generating captions/tags for all images..."):
                try:
                    if caption_type == "Caption (Sentence)":
                        gen_captions = generate_captions_for_images(
                            images,
                            hf_token_global,
                            cap_hf_id if cap_source == "Hugging Face (repo ID)" else None,
                            cap_local_file if cap_source == "Local Upload (.zip)" else None
                        )
                    else: # Tag (Keywords)
                        gen_captions = generate_tags_for_images(
                            images,
                            tag_hf_id if tag_source == "Hugging Face (repo ID)" else None,
                            tag_local_file if tag_source == "Local Upload (.zip)" else None
                        )
                    # Only set session state if all captions are generated successfully
                    if gen_captions and len(gen_captions) == len(images):
                        for idx, cap in enumerate(gen_captions):
                            st.session_state[f"caption_{idx}"] = cap
                        st.session_state['auto_captions'] = gen_captions
                        # Rerun with compatibility
                        if hasattr(st, "rerun"): st.rerun()
                        else: st.experimental_rerun()
                    else:
                        log_error("Auto-captioning/tagging did not return captions for all images. Please try again.")
                except Exception as e:
                    # Clear any partial/failed results
                    for idx in range(len(images)): st.session_state.pop(f"caption_{idx}", None)
                    st.session_state.pop('auto_captions', None)
                    log_exception(f"Auto-captioning/tagging failed: {e}")
    # Set captions from auto_captions if available and not already set
    if caption_mode == "Automatic (Recommended)" and captions_ready:
        for idx, cap in enumerate(auto_captions):
            if st.session_state.get(f"caption_{idx}") is None: # Only fill if not manually set
                st.session_state[f"caption_{idx}"] = cap
    # Render text areas for captions/tags
    for idx, img in enumerate(images):
        with st.container():
            st.image(img, width=200, caption=img.name)
            st.text_area(
                f"{'Caption' if caption_type == 'Caption (Sentence)' else 'Tags'} for {img.name}",
                value=st.session_state.get(f"caption_{idx}", ""),
                key=f"caption_{idx}" # Key ensures widget state is preserved
            )

# --- Step 2: Captioning (Default: Automatic) ---
st.subheader("2. Captioning")
caption_mode = st.radio("How do you want to provide captions/tags?", ("Automatic (Recommended)", "Manual"), index=0, key="caption_mode_radio")
caption_type = st.radio("Captioning/Tagging Mode", ["Caption (Sentence)", "Tags (Keywords)"], index=0, key="caption_type_radio")
st.checkbox("Always auto-caption/tag before training if not already done", key="force_auto_caption_before_train", value=True)


# --- Model Source Selection (Unified Dropdowns) ---
def model_source_dropdown(label, default_hf_id, local_key, cloud_key_prefix):
    source = st.selectbox(
        f"{label} Model Source",
        ["Hugging Face (repo ID)", "Local Upload (.zip)", "Cloud Storage (Drive/S3/etc.)"],
        key=f"{cloud_key_prefix}_source"
    )
    model_id = default_hf_id
    local_file = None # For .zip uploads
    cloud_path = None   # For cloud paths
    cloud_provider = None
    cloud_auth = None

    if source == "Hugging Face (repo ID)":
        model_id = st.text_input(f"Hugging Face repo ID for {label} model", value=default_hf_id, key=f"{cloud_key_prefix}_hf_id")
    elif source == "Local Upload (.zip)":
        # This uploader is for a ZIP file containing the model.
        # The extraction and path handling needs to be done where the model is loaded.
        local_file = st.file_uploader(f"Upload local {label} model folder as .zip", type=["zip"], key=f"{local_key}_zip")
        if local_file:
             # For local ZIP, the 'path' becomes the name of the uploaded file (needs extraction later)
            cloud_path = local_file.name 
    elif source == "Cloud Storage (Drive/S3/etc.)":
        cloud_provider = st.selectbox(f"Cloud Provider for {label} model", ["Google Drive", "OneDrive", "AWS S3", "Azure Blob", "GCP Storage"], key=f"{cloud_key_prefix}_provider")
        # Dynamic instructions based on provider
        mount_instructions = {
            "Google Drive": "To use Google Drive, run the mounting script or, in Colab, run:\n\n```python\nfrom google.colab import drive\ndrive.mount('/content/drive')\n```\nThen enter the path to your model (e.g., `/content/drive/MyDrive/my_model_dir`).",
            "OneDrive": "To use OneDrive, run a mounting script or use `rclone` to mount your OneDrive remote. Then enter the mount path (e.g., `/content/onedrive/my_model_dir`). See README for details.",
            "AWS S3": "To use AWS S3, run a mounting script or use `rclone` to mount your S3 bucket. Then enter the mount path (e.g., `/content/s3bucket/my_model_dir`). See README for details. You may need to configure AWS credentials.",
            "Azure Blob": "To use Azure Blob, run a mounting script or use `rclone` to mount your Azure remote. Then enter the mount path (e.g., `/content/azure/my_model_dir`). See README for details.",
            "GCP Storage": "To use Google Cloud Storage, run a mounting script or use `rclone` to mount your GCS bucket. Then enter the mount path (e.g., `/content/gcs/my_model_dir`). See README for details."
        }
        if cloud_provider:
            st.info(mount_instructions[cloud_provider])
        cloud_path = st.text_input(f"Path to {label} model in cloud storage (mounted path)", key=f"{cloud_key_prefix}_path")
        if cloud_provider in ["AWS S3", "Azure Blob", "GCP Storage"]: # Providers that might need explicit auth string
            cloud_auth = st.text_area(f"Auth/config for {cloud_provider} (e.g., rclone config string, optional)", key=f"{cloud_key_prefix}_auth")
    
    return source, model_id, local_file, cloud_provider, cloud_path, cloud_auth


with st.expander("Advanced Model Selection", expanded=False):
    st.markdown("#### Captioning Model (BLIP)")
    cap_source, cap_hf_id, cap_local_file, cap_cloud_provider, cap_cloud_path, cap_cloud_auth = model_source_dropdown(
        "Captioning (BLIP)", DEFAULT_CAPTION_MODEL, "local_blip", "cap_blip"
    )
    st.markdown("#### Tagging Model (CLIP Interrogator)")
    tag_source, tag_hf_id, tag_local_file, tag_cloud_provider, tag_cloud_path, tag_cloud_auth = model_source_dropdown(
        "Tagging (CLIP)", "ViT-L-14/openai", "local_clip", "tag_clip" # Default CLIP model for Interrogator
    )
    st.markdown("#### Base Model (for LoRA Training)")
    base_source, base_hf_id, base_local_file, base_cloud_provider, base_cloud_path, base_cloud_auth = model_source_dropdown(
        "Base", DEFAULT_BASE_MODEL["modelId"], "local_base", "base"
    )
    st.markdown("#### LoRA Model (optional, for reuse/further training - Not fully implemented)")
    lora_source, lora_hf_id, lora_local_file, lora_cloud_provider, lora_cloud_path, lora_cloud_auth = model_source_dropdown(
        "LoRA", "", "local_lora", "lora" # No default LoRA
    )
    
    # --- Highly Advanced (show with checkbox, not expander) ---
    show_highly_advanced = st.checkbox("Show Highly Advanced Options (e.g., custom training code)", value=False)
    if show_highly_advanced:
        default_code = (
            "# Example: Custom training config for Kohya_ss as Python dict\n"
            "# These are passed as additional command line arguments.\n"
            "# See Kohya_ss sd-scripts documentation for all options.\n"
            "custom_config = {\n"
            "    # 'min_snr_gamma': 5, # Example: for --min_snr_gamma=5\n"
            "    # 'network_train_unet_only': True, # Example: for --network_train_unet_only\n"
            "    # 'optimizer_args': \"betas=.9,.99\", # Example: for --optimizer_args betas=.9,.99 (must be str)\n"
            "    # 'unet_lr': 1e-5, # Example for setting unet learning rate\n"
            "    # 'text_encoder_lr': 5e-6, # Example for setting text encoder learning rate\n"
            "}\n"
        )
        custom_code = st.text_area("Edit training config/code (Python dict 'custom_config')", value=default_code, height=200)

# --- Step 3: Base Model & LoRA Method (Default, with search/advanced) ---
st.subheader("3. Base Model & LoRA Method")
# For now, only Kohya_ss is directly supported for training. Diffusers is conceptual.
# selected_lora_method_name = st.selectbox("Select LoRA Training Method", [DEFAULT_LORA_METHOD["name"], "Diffusers LoRA (Conceptual)"])
selected_lora_method_name = DEFAULT_LORA_METHOD["name"] # Hardcode to Kohya for now

with st.expander("Show/Change Model (Advanced)", expanded=False):
    st.markdown(f"**Current Base Model (for LoRA):** `{base_hf_id if base_source == 'Hugging Face (repo ID)' else (base_cloud_path or 'N/A')}`")
    st.markdown(f"**Current LoRA Method:** `{selected_lora_method_name}`")
    
    # --- Model search ---
    search_query = st.text_input("Search for other base models (Hugging Face)")
    if st.button("Search Models"):
        results = search_hf_models(search_query, hf_token_global)
        if results:
            for r in results:
                st.markdown(f"**{r['name']}**  \n_{r['desc']}_  \nTags: {r['tags']}")
                # Example: Add a warning if model is large or incompatible
                detected_type = detect_model_type(r['modelId'], hf_token_global)
                if detected_type != "diffusion":
                     log_warning(f"Warning: {r['name']} appears to be a '{detected_type}' model. Base models for LoRA should typically be diffusion models.")
                if vram_gb and ('xl' in r['name'].lower() or 'sdxl' in r['name'].lower()) and vram_gb < 12 : # Basic VRAM check for XL
                    log_warning(f"Warning: {r['name']} (XL) may require more VRAM than detected ({vram_gb} GB). Training might be slow or fail.")
        else:
            log_warning("No models found or error searching Hugging Face.")
    
    custom_model_id_input = st.text_input("Or manually enter a Hugging Face model ID for Base Model", value=base_hf_id)
    if custom_model_id_input != base_hf_id: # If user changes it
        base_hf_id = custom_model_id_input
        base_source = "Hugging Face (repo ID)" # Assume HF if manually entered here
        log_info(f"Base model changed to: {base_hf_id} (assumed Hugging Face repo ID)")
        # Optionally, re-validate model type and VRAM requirements here


# --- Step 6: LoRA Metadata ---
st.subheader("6. LoRA Metadata (for Kohya_ss)")
# 'lora_model_name' is used as 'output_name' for Kohya_ss, which becomes the .safetensors filename.
lora_model_name = st.text_input("LoRA Model Name (output filename, e.g., 'my_character_lora')", value="my_lora_model")
# 'lora_activation_keyword' is for prompting, not directly used by Kohya for saving, but essential for use.
lora_activation_keyword = st.text_input("Activation Keyword (trigger word, e.g., 'mychar_style')", value="my_keyword")
# For Hugging Face Hub upload
hf_repo_name = st.text_input("Hugging Face Repo Name (e.g., 'lora-my-character', will be created if new)", value=f"lora-{lora_model_name.lower().replace('_', '-')}")


# --- train_lora function defined here to ensure it's available before the button ---
def train_lora(images, captions, base_model_id_resolved, 
               lora_model_name_resolved, lora_activation_keyword_resolved, 
               output_dir_resolved, hf_token_resolved, device_resolved, 
               adv_config_resolved=None, custom_code_resolved=None, precision_resolved=None):
    """
    Trains a LoRA model using an external script like Kohya_ss's train_network.py,
    enhanced with accelerate integration and CPU optimization.
    """
    logging.info(f"Starting LoRA training for {lora_model_name_resolved} based on {base_model_id_resolved}...")
    logging.info(f"Output directory: {output_dir_resolved}, Device: {device_resolved}, Precision: {precision_resolved}")
    if adv_config_resolved is None: adv_config_resolved = {}
    logging.info(f"Advanced config: {adv_config_resolved}")
    if custom_code_resolved: logging.info(f"Custom code provided: {custom_code_resolved[:200]}...")

    # --- Check for accelerate ---
    accelerate_path = shutil.which("accelerate")
    if not accelerate_path:
        log_error("`accelerate` command not found. Please ensure Diffusers, PyTorch and Accelerate are installed (`pip install diffusers transformers accelerate torch`) and `accelerate` is in your system's PATH.")
        return "error_accelerate_not_found"
    else:
        logging.info(f"Found accelerate at: {accelerate_path}")
        log_info("Using 'accelerate launch'. If you have multiple GPUs or encounter device issues, ensure your Accelerate configuration is set up correctly. You can run 'accelerate config' in your terminal to configure it.")

    # --- Locate the training script (Kohya's train_network.py) ---
    script_paths_to_check = [
        os.path.abspath("./sd-scripts/train_network.py"),
        os.path.abspath("./kohya_ss/train_network.py"),
    ]
    train_script_path = None
    for path_to_check in script_paths_to_check:
        if os.path.exists(path_to_check) and os.access(path_to_check, os.X_OK):
            train_script_path = path_to_check
            logging.info(f"Found executable Kohya training script at: {train_script_path}")
            break
    if not train_script_path:
        # Fallback to checking PATH if not in common local dirs
        path_in_system_path = shutil.which("train_network.py")
        if path_in_system_path and os.access(path_in_system_path, os.X_OK):
            train_script_path = os.path.abspath(path_in_system_path)
            logging.info(f"Found executable Kohya training script in PATH: {train_script_path}")
        else:
            log_error(
                "Kohya's train_network.py script not found or not executable. "
                f"Checked locations: {', '.join(script_paths_to_check)}, and system PATH. "
                "Please ensure it's correctly placed and executable."
            )
            return "error_script_not_found"

    # --- Prepare dataset ---
    run_specific_data_dir_name = f"{lora_model_name_resolved}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    prepared_data_dir = os.path.abspath(os.path.join("temp_train_data", run_specific_data_dir_name))
    if os.path.exists(prepared_data_dir): shutil.rmtree(prepared_data_dir)
    os.makedirs(prepared_data_dir, exist_ok=True)
    logging.info(f"Preparing training data in: {prepared_data_dir}")
    for i, (img_file, caption_text) in enumerate(zip(images, captions)):
        try:
            original_filename_stem = os.path.splitext(img_file.name)[0]
            safe_stem = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in original_filename_stem)
            image_extension = os.path.splitext(img_file.name)[1].lower()
            base_savename = f"img_{i}_{safe_stem}"[:50]
            image_savename = f"{base_savename}{image_extension}"
            caption_savename = f"{base_savename}.txt"
            image_path = os.path.join(prepared_data_dir, image_savename)
            caption_path = os.path.join(prepared_data_dir, caption_savename)
            with open(image_path, "wb") as f: f.write(img_file.getbuffer())
            with open(caption_path, "w", encoding="utf-8") as f: f.write(caption_text)
        except Exception as e:
            log_exception(f"Error preparing image/caption {img_file.name}: {e}")
            shutil.rmtree(prepared_data_dir)
            return "error_unexpected"
    logging.info("Training data preparation complete.")

    # --- Construct the accelerate launch command and Kohya script arguments ---
    abs_output_dir = os.path.abspath(output_dir_resolved)
    os.makedirs(abs_output_dir, exist_ok=True)
    kohya_logs_dir = os.path.abspath("./lora_logs")
    os.makedirs(kohya_logs_dir, exist_ok=True)

    # Translate PyTorch dtype to string for Kohya script
    precision_arg = "no"
    if precision_resolved == torch.float16: precision_arg = "fp16"
    elif precision_resolved == torch.bfloat16: precision_arg = "bf16"

    # Base accelerate command
    accelerate_cmd_list = [accelerate_path, "launch"]

    # Add CPU specific accelerate args
    if device_resolved == "cpu":
        num_cpu_threads = max(1, os.cpu_count() // 2) # Use half of CPU cores, min 1
        accelerate_cmd_list.append(f"--num_cpu_threads_per_process={num_cpu_threads}")
        logging.info(f"CPU training detected. Using --num_cpu_threads_per_process={num_cpu_threads} for accelerate.")
        # Note: ensure accelerate config for CPU is appropriate (e.g., `accelerate config` -> CPU only, no deepspeed etc.)

    # Kohya's train_network.py arguments
    kohya_args_list = [
        train_script_path,
        f"--pretrained_model_name_or_path={base_model_id_resolved}",
        f"--train_data_dir={prepared_data_dir}",
        f"--output_dir={abs_output_dir}",
        f"--output_name={lora_model_name_resolved}",
        f"--resolution={adv_config_resolved.get('resolution', '512,512')}",
        f"--train_batch_size={str(adv_config_resolved.get('batch_size', 1))}",
        f"--learning_rate={str(adv_config_resolved.get('learning_rate', 1e-4))}",
        f"--max_train_epochs={str(adv_config_resolved.get('epochs', 10))}",
        f"--network_dim={str(adv_config_resolved.get('lora_rank', 8))}",
        f"--network_alpha={str(adv_config_resolved.get('lora_alpha', adv_config_resolved.get('lora_rank', 8)))}",
        "--network_module=networks.lora",
        f"--mixed_precision={precision_arg}", # This is for Kohya, accelerate handles its own mixed precision if configured
        f"--save_precision={precision_arg}", # Kohya also has a save_precision matching mixed_precision usually
        "--save_model_as=safetensors",
        f"--save_every_n_epochs={str(adv_config_resolved.get('save_every_n_epochs', 1))}",
        f"--logging_dir={kohya_logs_dir}",
        f"--log_prefix={lora_model_name_resolved}",
        f"--lr_scheduler={adv_config_resolved.get('lr_scheduler', 'cosine_with_restarts')}",
    ]

    if hf_token_resolved: kohya_args_list.append(f"--hf_token={hf_token_resolved}")
    
    optimizer_choice = adv_config_resolved.get('optimizer', 'AdamW').lower()
    if optimizer_choice == 'adamw': kohya_args_list.append("--optimizer_type=AdamW") # Kohya supports various, e.g., AdamW8bit
    elif optimizer_choice == 'sgd': kohya_args_list.append("--optimizer_type=SGD")
    
    if adv_config_resolved.get('gradient_checkpointing', False): kohya_args_list.append("--gradient_checkpointing")
    if adv_config_resolved.get('noise_offset') is not None: kohya_args_list.append(f"--noise_offset={str(adv_config_resolved.get('noise_offset'))}")
    if adv_config_resolved.get('min_snr_gamma') is not None: kohya_args_list.append(f"--min_snr_gamma={str(adv_config_resolved.get('min_snr_gamma'))}")

    if custom_code_resolved:
        try:
            local_scope = {}
            exec(custom_code_resolved, {"torch": torch, "os": os}, local_scope)
            custom_params_dict = local_scope.get('custom_config', {})
            if isinstance(custom_params_dict, dict):
                for key, value in custom_params_dict.items():
                    if isinstance(value, bool):
                        if value: kohya_args_list.append(f"--{key}")
                    elif value is not None: kohya_args_list.append(f"--{key}={str(value)}")
                logging.info(f"Applied custom config parameters to Kohya script: {custom_params_dict}")
            else: logging.warning("'custom_config' in custom_code_resolved was not a dictionary.")
        except Exception as e: logging.warning(f"Could not parse/apply custom_code_resolved: {e}\nCode: {custom_code_resolved}")

    full_command = accelerate_cmd_list + kohya_args_list
    logging.info(f"Executing full training command: {' '.join(full_command)}")
    st.info(f"Starting LoRA training for '{lora_model_name_resolved}' via accelerate. This may take a while...")

    try:
        process = subprocess.run(full_command, capture_output=True, text=True, env=os.environ.copy(), check=False)
        if process.stdout: logging.info(f"Accelerate/Kohya STDOUT:\n{process.stdout}")
        else: logging.info("Accelerate/Kohya STDOUT: (empty)")
        if process.stderr:
            if process.returncode != 0: logging.error(f"Accelerate/Kohya STDERR:\n{process.stderr}")
            else: logging.info(f"Accelerate/Kohya STDERR (might contain progress):\n{process.stderr}")
        else: logging.info("Accelerate/Kohya STDERR: (empty)")

        if process.returncode == 0:
            expected_model_path = os.path.join(abs_output_dir, f"{lora_model_name_resolved}.safetensors")
            if os.path.exists(expected_model_path):
                logging.info(f"Training script completed. Model: {expected_model_path}")
                st.success(f"Training for '{lora_model_name_resolved}' completed successfully!")
                st.session_state['last_base_model_id_trained'] = base_model_id_resolved # Save the base model ID used
                return "trained"
            else:
                log_error(f"Script success (ret 0), but model not found: {expected_model_path}")
                return "error_training_failed"
        else:
            error_summary = f"Training script for '{lora_model_name_resolved}' failed (accelerate launch). Return code: {process.returncode}."
            logging.error(error_summary)
            st.error(f"{error_summary} See lora_ui.log. STDERR snippet: {process.stderr[:500] if process.stderr else 'N/A'}")
            return "error_training_failed"
    except FileNotFoundError: # Should primarily catch issues with accelerate_path if not caught by initial check
        log_error(f"Critical error: `accelerate` or script {train_script_path} not found during subprocess execution.")
        return "error_accelerate_not_found" # Or a more general script error if accelerate was found
    except Exception as e:
        log_exception(f"Unexpected error during LoRA training subprocess: {e}")
        return "error_unexpected"
    finally:
        if os.path.exists(prepared_data_dir):
            try:
                shutil.rmtree(prepared_data_dir)
                logging.info(f"Cleaned temp data: {prepared_data_dir}")
            except Exception as e: logging.warning(f"Failed to clean temp data {prepared_data_dir}: {e}")


# --- Step 7: Start LoRA Training ---
# Ensure custom_code is always defined from the UI element
custom_code_from_ui = custom_code if 'custom_code' in locals() and show_highly_advanced else ""

if st.button("Start LoRA Training"):
    if not images:
        log_error("No images uploaded. Please upload images for LoRA training.") # Uses st.error
        st.stop()

    # --- Modified Auto-Captioning Logic ---
    # Check if auto-captioning should be forced or if mode is "Automatic"
    # force_auto_caption is the new checkbox, caption_mode is the radio button
    should_auto_caption_now = st.session_state.get("force_auto_caption_before_train", True) or \
                              st.session_state.get("caption_mode_radio") == "Automatic (Recommended)"

    if should_auto_caption_now:
        auto_captions_session = st.session_state.get('auto_captions', None)
        # Check if captions are already there for all images (either from previous auto or manual entry)
        all_captions_populated_in_ui = True
        for idx in range(len(images)):
            if not st.session_state.get(f"caption_{idx}", "").strip():
                all_captions_populated_in_ui = False
                break
        
        # Trigger auto-captioning if:
        # 1. auto_captions_session is not complete OR
        # 2. individual caption fields in the UI are not all populated.
        # This ensures that if "force_auto_caption_before_train" is true, it will fill missing captions.
        if not (auto_captions_session and len(auto_captions_session) == len(images) and all_captions_populated_in_ui) :
            log_info("Auto-captioning triggered before training (either forced or mode is Automatic and captions missing/incomplete).")
            with st.spinner("Auto-generating captions/tags..."):
                try:
                    # Determine caption type from the UI radio button
                    current_caption_type = st.session_state.get("caption_type_radio", "Caption (Sentence)")
                    if current_caption_type == "Caption (Sentence)":
                        generated_captions = generate_captions_for_images(
                            images, hf_token_global,
                            cap_hf_id if cap_source == "Hugging Face (repo ID)" else None,
                            cap_local_file if cap_source == "Local Upload (.zip)" else None
                        )
                    else: # Tags
                        generated_captions = generate_tags_for_images(
                            images, hf_token_global,
                            tag_hf_id if tag_source == "Hugging Face (repo ID)" else None,
                            tag_local_file if tag_source == "Local Upload (.zip)" else None
                        )
                    
                    if generated_captions and len(generated_captions) == len(images):
                        st.session_state['auto_captions'] = generated_captions # Store the full list
                        for idx, cap_text in enumerate(generated_captions): # Populate individual text areas
                            st.session_state[f"caption_{idx}"] = cap_text # Update UI fields
                        log_info("Auto-captions generated/updated. Re-running to update UI and proceed with training.")
                        if hasattr(st, "rerun"): st.rerun()
                        else: st.experimental_rerun() 
                    else:
                        log_error("Auto-captioning/tagging failed to return all captions. Please try manual captioning or check model & logs.")
                        st.stop() 
                except Exception as e:
                    st.session_state.pop('auto_captions', None) # Clear potentially incomplete results
                    log_exception(f"Auto-captioning/tagging failed during pre-training process: {e}")
                    st.stop() 

    # Validate all images have captions/tags AFTER potential auto-captioning
    final_captions_for_training = []
    missing_captions_indices = []
    for idx in range(len(images)):
        caption_text = st.session_state.get(f"caption_{idx}", "").strip()
        if not caption_text:
            missing_captions_indices.append(images[idx].name)
        else:
            final_captions_for_training.append(caption_text)

    if missing_captions_indices:
        log_error(f"Please provide captions for all images. Missing for: {', '.join(missing_captions_indices)}")
    else:
        current_output_dir = f"lora_output/{lora_model_name}" 
        log_info("All validations passed. Starting LoRA training process...")
        
        actual_base_model_id_for_training = base_hf_id 
        if base_source == "Local Upload (.zip)" and base_local_file:
            # This needs to be a path that Kohya script can access.
            # For simplicity, we'll assume it's extracted to a known relative path.
            # In a real app, you'd handle extraction and path management more robustly.
            actual_base_model_id_for_training = os.path.abspath(f"./extracted_models/{base_local_file.name.replace('.zip','')}")
            log_info(f"Using local base model (assumed extracted to): {actual_base_model_id_for_training}")
            # Placeholder: You would need to ensure the model is actually extracted to this path.
            # os.makedirs(os.path.dirname(actual_base_model_id_for_training), exist_ok=True) 
        elif base_source == "Cloud Storage (Drive/S3/etc.)" and base_cloud_path:
            actual_base_model_id_for_training = base_cloud_path # This path must be accessible by the training script
            log_info(f"Using cloud base model path: {actual_base_model_id_for_training}")

        device_selected, precision_selected, device_msg, torch_version_detected = get_best_device_and_precision()
        log_info(f"{device_msg} (torch version: {torch_version_detected})")
        if device_selected == "cpu":
            log_warning("No supported accelerator (GPU/MPS) detected. Training on CPU will be very slow.")

        with st.spinner(f"Training LoRA model '{lora_model_name}'... This may take a significant amount of time. Monitor logs for progress."):
            try:
                # Reset model_ready state before starting new training
                st.session_state['model_ready'] = False
                st.session_state['trained_model_dir'] = None
                st.session_state['last_base_model_id_trained'] = None # Clear previous one
                st.session_state['uploaded_repo_url'] = None


                training_result = train_lora(
                    images=images,
                    captions=final_captions_for_training,
                    base_model_id_resolved=actual_base_model_id_for_training,
                    lora_model_name_resolved=lora_model_name, 
                    lora_activation_keyword_resolved=lora_activation_keyword, 
                    output_dir_resolved=current_output_dir,
                    hf_token_resolved=hf_token_global, 
                    device_resolved=device_selected,
                    adv_config_resolved=adv_config, 
                    custom_code_resolved=custom_code_from_ui, 
                    precision_resolved=precision_selected
                )

                if training_result == "trained":
                    st.session_state['trained_model_dir'] = current_output_dir
                    st.session_state['model_ready'] = True 
                    st.session_state['last_base_model_id_trained'] = actual_base_model_id_for_training # Save for generation
                    
                    if hf_token_global and hf_repo_name: 
                        with st.spinner(f"Uploading model to Hugging Face Hub: {hf_repo_name} (private)..."):
                            try:
                                api = HfApi()
                                user_info = api.whoami(token=hf_token_global)
                                hf_username = user_info.get('name', user_info.get('user')) 
                                if not hf_username: raise ValueError("Could not determine Hugging Face username.")
                                
                                final_repo_id = f"{hf_username}/{hf_repo_name}"
                                log_info(f"Attempting to create/upload to repo: {final_repo_id}")
                                
                                api.create_repo(repo_id=final_repo_id, exist_ok=True, token=hf_token_global, private=True) 
                                api.upload_folder(
                                    folder_path=current_output_dir,
                                    repo_id=final_repo_id,
                                    token=hf_token_global,
                                    commit_message=f"Add {lora_model_name} LoRA model files"
                                )
                                repo_url = f"https://huggingface.co/{final_repo_id}"
                                st.session_state['uploaded_repo_url'] = repo_url
                                log_info(f"Model successfully uploaded to Hugging Face (private): {repo_url}")
                                st.markdown(f"### [View Model on Hugging Face]({repo_url})")
                            except Exception as e_upload:
                                log_exception(f"Automatic upload to Hugging Face failed: {e_upload}")
                    else:
                        log_warning("Skipping Hugging Face upload: Token or Repo Name not provided.")
                else:
                    log_error(f"LoRA training failed with status: {training_result}. Check UI and logs for details.")
                    # Ensure model_ready is false if training didn't succeed (already handled by reset above)
            except Exception as e_train_call:
                log_exception(f"LoRA training process encountered an unexpected error: {e_train_call}")
                st.session_state['model_ready'] = False # Ensure this on any exception during the training call too
                st.session_state['trained_model_dir'] = None
                st.session_state['last_base_model_id_trained'] = None


# --- Always show image generation UI after training if model is ready ---
if st.session_state.get('model_ready', False):
    st.subheader("8. Model Download & Upload")
    model_dir_trained = st.session_state.get('trained_model_dir', 'lora_output/default_model') 
    repo_url_trained = st.session_state.get('uploaded_repo_url', None)

    if repo_url_trained:
        st.markdown(f"**[View/Access Model on Hugging Face]({repo_url_trained})**")
    
    zip_display_name = f"{os.path.basename(model_dir_trained)}_lora_model.zip"
    zip_path_to_create = f"{model_dir_trained}.zip" 
    
    if not os.path.exists(zip_path_to_create): 
        try:
            shutil.make_archive(model_dir_trained, 'zip', root_dir=os.path.dirname(model_dir_trained), base_dir=os.path.basename(model_dir_trained))
            log_info(f"Created zip for download: {zip_path_to_create}")
        except Exception as e_zip:
            log_exception(f"Failed to create zip archive for download: {e_zip}")
            zip_path_to_create = None 

    if zip_path_to_create and os.path.exists(zip_path_to_create):
        with open(zip_path_to_create, "rb") as fp:
            st.download_button(
                label="Download Trained LoRA Model (.zip)",
                data=fp,
                file_name=zip_display_name, 
                mime="application/zip"
            )
    else:
        st.warning("Could not prepare model for download (zipping failed or model directory issue).")

    st.info(f"You can also find the model files on the server at: `{os.path.abspath(model_dir_trained)}`")


    # --- Step 9: Generate Images with Trained Model (Prompt Queue) ---
    st.subheader("9. Generate Images with Trained Model")
    
    def prepend_lora_keyword(prompt, keyword):
        prompt = prompt.strip()
        if keyword and keyword.strip() and keyword.strip() not in prompt:
            return f"{keyword.strip()}, {prompt}"
        return prompt

    current_lora_keyword_for_prompting = lora_activation_keyword 
    if 'last_lora_keyword_for_prompts' not in st.session_state or \
       st.session_state.get('last_lora_keyword_for_prompts') != current_lora_keyword_for_prompting:
        default_prompts_list = [
            "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood under a stormy night sky, artstation, masterful, gtx",
            "photo of a dog wearing a wizard hat",
            "concept art of a futuristic city with flying cars, detailed, science fiction"
        ]
        st.session_state['queued_prompts'] = [prepend_lora_keyword(p, current_lora_keyword_for_prompting) for p in default_prompts_list]
        st.session_state['last_lora_keyword_for_prompts'] = current_lora_keyword_for_prompting
    
    if 'queued_num_images_per_prompt' not in st.session_state:
        st.session_state['queued_num_images_per_prompt'] = 1

    log_info(f"Image generation prompt queue initialized. LoRA keyword: '{current_lora_keyword_for_prompting}'")
    
    for i in range(len(st.session_state['queued_prompts'])): 
        user_prompt = st.text_input(f"Prompt {i+1}", value=st.session_state['queued_prompts'][i], key=f"queued_prompt_input_{i}")
        st.session_state['queued_prompts'][i] = prepend_lora_keyword(user_prompt, current_lora_keyword_for_prompting)

    col1_prompt_mgmt, col2_prompt_mgmt = st.columns(2)
    with col1_prompt_mgmt:
        if st.button("Add Another Prompt"):
            st.session_state['queued_prompts'].append(prepend_lora_keyword("Another prompt", current_lora_keyword_for_prompting))
            if hasattr(st, "rerun"): st.rerun()
            else: st.experimental_rerun()
    with col2_prompt_mgmt:
        if len(st.session_state['queued_prompts']) > 0 : 
            if st.button("Remove Last Prompt", key="remove_last_prompt_btn"):
                if st.session_state['queued_prompts']: 
                    st.session_state['queued_prompts'].pop()
                    if hasattr(st, "rerun"): st.rerun()
                    else: st.experimental_rerun()
    
    num_images_to_gen = st.slider("Number of images to generate per prompt", 1, 4, st.session_state.get('queued_num_images_per_prompt', 1), key="num_images_slider")
    st.session_state['queued_num_images_per_prompt'] = num_images_to_gen


    if st.button("Generate Images with Trained LoRA", key="generate_images_main_button"):
        # trained_model_dir is already available from the top of the "if model_ready" block
        # lora_model_name is available from the UI input field
        if not model_dir_trained: # Check if trained_model_dir is valid
            log_error("Trained model directory not found in session state. Cannot generate images.")
            st.error("Trained model directory is missing. Please ensure training was successful.")
        else:
            with st.spinner("Loading model and generating images... This may take time."):
                try:
                    gen_device, gen_precision, _, _ = get_best_device_and_precision()
                    
                    # 1. Load Base Model using last_base_model_id_trained
                    base_model_id_for_gen = st.session_state.get('last_base_model_id_trained')
                    if not base_model_id_for_gen:
                        log_error("Base model ID used for training not found in session state. Cannot generate images.")
                        st.error("Critical Error: Base model ID from training is missing. Cannot proceed with image generation.")
                        st.stop() # Stop execution for this button press

                    log_info(f"Loading base model pipeline: {base_model_id_for_gen}")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        base_model_id_for_gen, 
                        torch_dtype=gen_precision,
                        use_auth_token=hf_token_global if hf_token_global else None
                    )
                    pipe = pipe.to(gen_device)
                    log_info(f"Base model {base_model_id_for_gen} loaded successfully onto {gen_device}.")

                    # 2. Load LoRA Weights
                    # model_dir_trained is the directory path like "lora_output/my_lora_model"
                    # lora_model_name is the filename like "my_lora_model"
                    # The LoRA file itself is typically model_dir_trained + "/" + lora_model_name + ".safetensors"
                    lora_weights_path = os.path.join(model_dir_trained, f"{lora_model_name}.safetensors")
                    
                    if not os.path.exists(lora_weights_path):
                        log_error(f"LoRA weights file not found at: {lora_weights_path}. Attempting to load from directory {model_dir_trained} instead.")
                        # Fallback: try loading from the directory, diffusers might find it.
                        # This also covers cases where the file might be named differently but is the only LoRA there.
                        lora_weights_path = model_dir_trained 
                        if not os.path.exists(lora_weights_path): # Double check directory exists
                             log_error(f"LoRA weights directory also not found: {lora_weights_path}. Cannot load LoRA.")
                             st.error(f"Error: LoRA weights not found at expected path or directory: {lora_weights_path}")
                             st.stop()


                    log_info(f"Attempting to load LoRA weights from: {lora_weights_path}")
                    pipe.load_lora_weights(lora_weights_path) 
                    log_info(f"Successfully loaded LoRA weights into pipeline from {lora_weights_path}.")
                    
                    st.markdown("---")
                    st.markdown("### Generated Images:")
                    for current_prompt in st.session_state['queued_prompts']:
                        if not current_prompt.strip():
                            log_warning("Skipping empty prompt.")
                            continue
                        
                        st.markdown(f"**Generating for prompt:** `{current_prompt}`")
                        generated_images_list = pipe(
                            [current_prompt] * num_images_to_gen, 
                            num_inference_steps=30 
                        ).images
                        
                        cols = st.columns(num_images_to_gen if num_images_to_gen <= 4 else 4) 
                        for i, img_pil in enumerate(generated_images_list):
                            cols[i % len(cols)].image(img_pil, caption=f"Image {i+1}")
                        st.markdown("---") 
                    log_info("Image generation queue processed.")
                except Exception as e_generate:
                    log_exception(f"Image generation failed: {e_generate}")
                    st.error(f"An error occurred during image generation: {e_generate}")

# --- UI: Download logs ---
with st.sidebar.expander("Logs & Diagnostics", expanded=False):
    st.markdown("**Download logs for support or debugging:**")
    try:
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Log File (lora_ui.log)", f, file_name="lora_ui.log", mime="text/plain")
    except Exception: 
        log_info("Log file not available for download yet. It will be created once operations begin.")

[end of test.py]
