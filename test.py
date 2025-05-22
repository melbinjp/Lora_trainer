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
    st.info(msg)

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

with st.expander("4. Advanced Training Settings", expanded=False):
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=4)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%e")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    optimizer = st.selectbox("Optimizer", ["AdamW", "SGD"], index=0)
    lora_rank = st.number_input("LoRA Rank (smaller = less memory, default 2)", min_value=1, max_value=128, value=2)
    lora_alpha = st.number_input("LoRA Alpha (default 2)", min_value=1, max_value=128, value=2)
    # --- New: Expose LoRA adapter name, adapter weights, and allow multiple adapters ---
    lora_adapter_name = st.text_input("LoRA Adapter Name (default: 'lora')", value="lora")
    enable_multi_adapter = st.checkbox("Enable Multiple Adapters (set_adapters)", value=False)
    adapter_names = [lora_adapter_name]
    adapter_weights = [1.0]
    if enable_multi_adapter:
        adapter_names = st.text_area("Adapter Names (comma-separated)", value=lora_adapter_name).split(",")
        adapter_weights = st.text_area("Adapter Weights (comma-separated, same order)", value=",".join(["1.0"]*len(adapter_names))).split(",")
        adapter_names = [n.strip() for n in adapter_names if n.strip()]
        adapter_weights = [float(w.strip()) for w in adapter_weights if w.strip()]
    adv_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "optimizer": optimizer,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_adapter_name": lora_adapter_name,
        "adapter_names": adapter_names,
        "adapter_weights": adapter_weights,
        "enable_multi_adapter": enable_multi_adapter
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
    import psutil
    import platform
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    vram_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        elif hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available():
            vram_gb = None  # Apple MPS: VRAM not easily available
    except Exception:
        pass
    return ram_gb, vram_gb

ram_gb, vram_gb = get_system_resources()
st.sidebar.markdown(f"**Detected RAM:** {ram_gb} GB")
if vram_gb:
    st.sidebar.markdown(f"**Detected VRAM:** {vram_gb} GB")
else:
    st.sidebar.markdown("**Detected VRAM:** Not detected or not available")

manual_ram = st.sidebar.number_input("Override RAM (GB, optional)", min_value=1.0, value=ram_gb, step=0.5)
manual_vram = st.sidebar.number_input("Override VRAM (GB, optional)", min_value=0.0, value=vram_gb or 0.0, step=0.5)

# --- Model Selection with Resource Check ---
def get_default_model_for_resources(ram, vram):
    # Example: You can expand this logic for more models
    if vram and vram >= 8:
        return {
            "modelId": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion v1.5",
            "desc": "Popular base model for image generation and LoRA training.",
            "compatible_lora": ["kohya-ss", "diffusers"],
            "default": True
        }
    else:
        return {
            "modelId": "stabilityai/sd-turbo",
            "name": "SD Turbo (Efficient)",
            "desc": "Very efficient, fast, and low-memory model for image generation and LoRA training.",
            "compatible_lora": ["kohya-ss", "diffusers"],
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
    if caption_mode == "Automatic (Recommended)":
        log_info(f"Using {'BLIP' if caption_type == 'Caption (Sentence)' else 'CLIP Interrogator'} model for automatic {caption_type.lower()}.")
        if st.button("Auto Caption/Tag All Images"):
            with st.spinner("Downloading captioning/tagging model and generating captions/tags for all images..."):
                try:
                    if caption_type == "Caption (Sentence)":
                        gen_captions = generate_captions_for_images(
                            images,
                            hf_token_global,
                            cap_hf_id if cap_source == "Hugging Face (repo ID)" else None,
                            cap_local_file if cap_source == "Local Upload (.zip)" else None
                        )
                    else:
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
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        log_error("Auto-captioning/tagging did not return captions for all images. Please try again.")
                except Exception as e:
                    # Clear any partial/failed results
                    for idx in range(len(images)):
                        st.session_state.pop(f"caption_{idx}", None)
                    st.session_state.pop('auto_captions', None)
                    log_exception(f"Auto-captioning/tagging failed: {e}")
    # Set captions from auto_captions if available and not already set
    if caption_mode == "Automatic (Recommended)" and captions_ready:
        for idx, cap in enumerate(auto_captions):
            if st.session_state.get(f"caption_{idx}") is None:
                st.session_state[f"caption_{idx}"] = cap
    # Render text areas for captions/tags
    for idx, img in enumerate(images):
        with st.container():
            st.image(img, width=200, caption=img.name)
            st.text_area(
                f"{'Caption' if caption_type == 'Caption (Sentence)' else 'Tags'} for {img.name}",
                value=st.session_state.get(f"caption_{idx}", ""),
                key=f"caption_{idx}"
            )

# --- Step 2: Captioning (Default: Automatic) ---
st.subheader("2. Captioning")
caption_mode = st.radio("How do you want to provide captions/tags?", ("Automatic (Recommended)", "Manual"), index=0)
caption_type = st.radio("Captioning/Tagging Mode", ["Caption (Sentence)", "Tags (Keywords)"], index=0)

# --- Model Source Selection (Unified Dropdowns) ---
def model_source_dropdown(label, default_hf_id, local_key, cloud_key_prefix):
    source = st.selectbox(
        f"{label} Model Source",
        ["Hugging Face (repo ID)", "Local Upload (.zip)", "Cloud Storage (Drive/S3/etc.)"],
        key=f"{cloud_key_prefix}_source"
    )
    model_id = default_hf_id
    local_file = None
    cloud_provider = None
    cloud_path = None
    cloud_auth = None
    if source == "Hugging Face (repo ID)":
        model_id = st.text_input(f"Hugging Face repo ID for {label} model", value=default_hf_id, key=f"{cloud_key_prefix}_hf_id")
    elif source == "Local Upload (.zip)":
        local_file = st.file_uploader(f"Upload local {label} model folder as .zip", type=["zip"], key=f"{local_key}_zip")
    elif source == "Cloud Storage (Drive/S3/etc.)":
        cloud_provider = st.selectbox(f"Cloud Provider for {label} model", ["Google Drive", "OneDrive", "AWS S3", "Azure Blob", "GCP Storage"], key=f"{cloud_key_prefix}_provider")
        # Show mounting instructions dynamically
        if cloud_provider == "Google Drive":
            log_info("To use Google Drive, run the mounting script or, in Colab, run:\n\nfrom google.colab import drive\ndrive.mount('/content/drive')\n\nThen enter the path to your model (e.g., /content/drive/MyDrive/my_model_dir).")
        elif cloud_provider == "OneDrive":
            log_info("To use OneDrive, run the mounting script or use rclone to mount your OneDrive remote. Then enter the mount path (e.g., /content/onedrive/my_model_dir). See README for details.")
        elif cloud_provider == "AWS S3":
            log_info("To use AWS S3, run the mounting script or use rclone to mount your S3 bucket. Then enter the mount path (e.g., /content/s3bucket/my_model_dir). See README for details.")
        elif cloud_provider == "Azure Blob":
            log_info("To use Azure Blob, run the mounting script or use rclone to mount your Azure remote. Then enter the mount path (e.g., /content/azure/my_model_dir). See README for details.")
        elif cloud_provider == "GCP Storage":
            log_info("To use Google Cloud Storage, run the mounting script or use rclone to mount your GCS bucket. Then enter the mount path (e.g., /content/gcs/my_model_dir). See README for details.")
        cloud_path = st.text_input(f"Path to {label} model in cloud storage (mounted path)", key=f"{cloud_key_prefix}_path")
        if cloud_provider in ["AWS S3", "Azure Blob", "GCP Storage"]:
            cloud_auth = st.text_area(f"Auth/config for {cloud_provider} (if needed)", key=f"{cloud_key_prefix}_auth")
    return source, model_id, local_file, cloud_provider, cloud_path, cloud_auth

with st.expander("Advanced Model Selection", expanded=False):
    st.markdown("#### Captioning Model")
    cap_source, cap_hf_id, cap_local_file, cap_cloud_provider, cap_cloud_path, cap_cloud_auth = model_source_dropdown(
        "Captioning (BLIP)", "Salesforce/blip-image-captioning-base", "local_blip", "cap_blip"
    )
    st.markdown("#### Tagging Model")
    tag_source, tag_hf_id, tag_local_file, tag_cloud_provider, tag_cloud_path, tag_cloud_auth = model_source_dropdown(
        "Tagging (CLIP)", "ViT-L-14/openai", "local_clip", "tag_clip"
    )
    st.markdown("#### Base Model")
    base_source, base_hf_id, base_local_file, base_cloud_provider, base_cloud_path, base_cloud_auth = model_source_dropdown(
        "Base", DEFAULT_BASE_MODEL["modelId"], "local_base", "base"
    )
    st.markdown("#### LoRA Model (optional, for reuse)")
    lora_source, lora_hf_id, lora_local_file, lora_cloud_provider, lora_cloud_path, lora_cloud_auth = model_source_dropdown(
        "LoRA", "", "local_lora", "lora"
    )
    # --- Highly Advanced (show with checkbox, not expander) ---
    show_highly_advanced = st.checkbox("Show Highly Advanced Options", value=False)
    if show_highly_advanced:
        default_code = (
            "# Example: Custom training config as Python dict\n"
            "custom_config = {\n"
            "    'batch_size': 4,\n"
            "    'learning_rate': 1e-4,\n"
            "    'epochs': 10,\n"
            "    'optimizer': 'AdamW'\n"
            "}\n"
        )
        custom_code = st.text_area("Edit training config/code (Python)", value=default_code, height=180)

# --- Step 3: Base Model & LoRA Method (Default, with search/advanced) ---
st.subheader("3. Base Model & LoRA Method")
with st.expander("Show/Change Model (Advanced)", expanded=False):
    st.markdown(f"**Default Base Model:** `{DEFAULT_BASE_MODEL['name']}`")
    st.markdown(f"_{DEFAULT_BASE_MODEL['desc']}_")
    st.markdown(f"**Default LoRA Method:** `{DEFAULT_LORA_METHOD['name']}`")
    st.markdown(f"_{DEFAULT_LORA_METHOD['desc']}_")
    st.markdown("Compatible LoRA methods: " + ", ".join(DEFAULT_BASE_MODEL['compatible_lora']))
    st.markdown("Compatible base models: " + ", ".join(DEFAULT_LORA_METHOD['compatible_base']))
    # --- Model search ---
    search_query = st.text_input("Search for other base models (Hugging Face/Spaces)")
    if st.button("Search Models"):
        results = search_hf_models(search_query, hf_token_global)
        if results:
            for r in results:
                st.markdown(f"**{r['name']}**  \n_{r['desc']}_  \nTags: {r['tags']}")
                # Example: Add a warning if model is large
                if vram_gb and 'xl' in r['name'].lower() and vram_gb < 12:
                    log_warning(f"Warning: {r['name']} may require more VRAM than detected ({vram_gb} GB). Training or inference may fail or be very slow.")
        else:
            log_warning("No models found or error searching Hugging Face.")
    # --- Manual model selection ---
    custom_model_id = st.text_input("Or enter a custom model ID (Hugging Face)")
    if custom_model_id:
        # Example: Check for large models
        if vram_gb and ('xl' in custom_model_id.lower() or 'sdxl' in custom_model_id.lower()) and vram_gb < 12:
            log_warning(f"Warning: {custom_model_id} may require more VRAM than detected ({vram_gb} GB). Training or inference may fail or be very slow.")

# --- Step 6: LoRA Metadata ---
st.subheader("6. LoRA Metadata")
lora_model_name = st.text_input("LoRA Model Name (required)", value="my_lora")
lora_activation_keyword = st.text_input("Activation Keyword (required)", value="my_lora_keyword")
hf_repo_name = st.text_input("Hugging Face Repo Name (required, will be created if not exists)", value=f"lora-{lora_model_name}")

# --- Step 7: Start LoRA Training ---
# Ensure custom_code is always defined
if 'custom_code' not in locals():
    custom_code = ""
if st.button("Start LoRA Training"):
    # Auto-generate captions/tags if in automatic mode and not already generated
    if caption_mode == "Automatic (Recommended)":
        auto_captions = st.session_state.get('auto_captions', None)
        captions_ready = auto_captions is not None and len(auto_captions) == len(images)
        if not captions_ready:
            log_info("Auto-generating captions/tags for all images...")
            try:
                if caption_type == "Caption (Sentence)":
                    gen_captions = generate_captions_for_images(
                        images,
                        hf_token_global,
                        cap_hf_id if cap_source == "Hugging Face (repo ID)" else None,
                        cap_local_file if cap_source == "Local Upload (.zip)" else None
                    )
                else:
                    gen_captions = generate_tags_for_images(
                        images,
                        tag_hf_id if tag_source == "Hugging Face (repo ID)" else None,
                        tag_local_file if tag_source == "Local Upload (.zip)" else None
                    )
                # Only update auto_captions and rerun, let widget pick up new value
                if gen_captions and len(gen_captions) == len(images):
                    st.session_state['auto_captions'] = gen_captions
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                else:
                    log_error("Auto-captioning/tagging did not return captions for all images. Please try again.")
            except Exception as e:
                st.session_state.pop('auto_captions', None)
                log_exception(f"Auto-captioning/tagging failed: {e}")
                st.stop()
    # Validate all images have captions/tags
    missing = [idx for idx in range(len(images)) if not st.session_state.get(f"caption_{idx}", "").strip()]
    if missing:
        log_error(f"Please provide a caption/tag for all images. Missing for: {', '.join(str(images[i].name) for i in missing)}")
    else:
        captions = [st.session_state.get(f"caption_{idx}", "") for idx in range(len(images))]
        output_dir = f"lora_output/{lora_model_name}"
        log_info("Training LoRA... This may take a while.")
        # --- Improved Device selection ---
        device, precision, device_msg, torch_version = get_best_device_and_precision()
        log_info(f"{device_msg} (torch version: {torch_version})")
        if device == "cpu":
            log_warning("No supported accelerator detected. Training will be slow. For best performance, use a machine with a supported GPU, Apple Silicon (MPS), or AMD ROCm. See https://pytorch.org/get-started/locally/ for install instructions.")
        try:
            result = train_lora(
                images,
                captions,
                base_hf_id if base_source == "Hugging Face (repo ID)" else None,
                lora_model_name,
                lora_activation_keyword,
                output_dir,
                hf_token_global,
                device,
                adv_config=adv_config,
                custom_code=custom_code,
                precision=precision
            )
            if result == "trained":
                log_info(f"LoRA training complete! Weights saved to {output_dir}")
                st.session_state['trained_model_dir'] = output_dir
                st.session_state['model_ready'] = True
                # --- Automatically upload to Hugging Face ---
                with st.spinner("Uploading model to Hugging Face Hub (private)..."):
                    try:
                        from huggingface_hub import HfApi, whoami
                        api = HfApi()
                        # Get username from token
                        user_info = api.whoami(token=hf_token_global)
                        hf_username = user_info['name'] if 'name' in user_info else user_info.get('user', 'user')
                        repo_id = f"{hf_username}/{hf_repo_name}"
                        api.create_repo(name=hf_repo_name, exist_ok=True, token=hf_token_global, private=True)
                        api.upload_folder(
                            folder_path=output_dir,
                            repo_id=repo_id,
                            token=hf_token_global
                        )
                        repo_url = f"https://huggingface.co/{repo_id}"
                        st.session_state['uploaded_repo_url'] = repo_url
                        log_info(f"Model uploaded to Hugging Face (private): {repo_url}")
                        st.markdown(f"[View on Hugging Face]({repo_url})")
                    except Exception as e:
                        log_exception(f"Automatic upload to Hugging Face failed: {e}")
            else:
                log_error("LoRA training failed. See above for details.")
        except Exception as e:
            log_exception(f"LoRA training encountered an unexpected error: {e}")

# --- Always show image generation UI after training ---
if st.session_state.get('model_ready', False):
    st.subheader("8. Model Download & Upload")
    model_dir = st.session_state['trained_model_dir']
    repo_url = st.session_state.get('uploaded_repo_url', None)
    if repo_url:
        log_info(f"Model is available on Hugging Face (private): {repo_url}")
        st.markdown(f"[View on Hugging Face]({repo_url})")
    # Download option
    zip_path = f"{model_dir}.zip"
    if not os.path.exists(zip_path):
        shutil.make_archive(model_dir, 'zip', model_dir)
    with open(zip_path, "rb") as f:
        st.download_button("Download Trained Model (.zip)", f, file_name=os.path.basename(zip_path))
    log_info(f"Model files are saved in: {model_dir} (Colab: /content/Lora_trainer/{model_dir})")
    # --- Step 9: Generate Images with Trained Model (Prompt Queue) ---
    st.subheader("9. Generate Images with Trained Model")
    def prepend_lora_keyword(prompt, keyword):
        prompt = prompt.strip()
        if keyword and keyword not in prompt:
            return f"{keyword}, {prompt}"
        return prompt
    if 'last_lora_keyword' not in st.session_state or st.session_state.get('last_lora_keyword') != lora_activation_keyword:
        default_prompts = [
            "A futuristic cityscape at sunset, ultra detailed, trending on artstation",
            "A portrait of a cat wearing sunglasses, digital art, vibrant colors",
            "A fantasy landscape with mountains and rivers, epic lighting"
        ]
        st.session_state['queued_prompts'] = [prepend_lora_keyword(p, lora_activation_keyword) for p in default_prompts]
        st.session_state['last_lora_keyword'] = lora_activation_keyword
    if 'queued_num_images' not in st.session_state:
        st.session_state['queued_num_images'] = 1
    # Prompt queue UI
    log_info("You can queue prompts and settings below. Images will be generated automatically as soon as training completes. The LoRA activation keyword will be included in each prompt.")
    for i, prompt in enumerate(st.session_state['queued_prompts']):
        user_prompt = st.text_input(f"Prompt {i+1}", value=prompt, key=f"queued_prompt_{i}")
        st.session_state['queued_prompts'][i] = prepend_lora_keyword(user_prompt, lora_activation_keyword)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Prompt"):
            st.session_state['queued_prompts'].append(prepend_lora_keyword("", lora_activation_keyword))
    with col2:
        if len(st.session_state['queued_prompts']) > 1:
            remove_idx = st.number_input("Remove prompt #", min_value=1, max_value=len(st.session_state['queued_prompts']), value=1, step=1, key="remove_prompt_idx")
            if st.button("Remove Prompt"):
                st.session_state['queued_prompts'].pop(remove_idx-1)
    num_images = st.slider("Number of images to generate per prompt", 1, 4, key="queued_num_images")
    if st.button("Generate Images with Trained Model"):
        with st.spinner("Loading model and generating images for queued prompts..."):
            try:
                device, precision, device_msg, torch_version = get_best_device_and_precision()
                pipe = StableDiffusionPipeline.from_pretrained(
                    st.session_state['trained_model_dir'],
                    torch_dtype=precision
                )
                pipe = pipe.to(device)
                for prompt in st.session_state['queued_prompts']:
                    prompt_with_keyword = prepend_lora_keyword(prompt, lora_activation_keyword)
                    st.markdown(f"**Prompt:** {prompt_with_keyword}")
                    images = pipe([prompt_with_keyword]*st.session_state['queued_num_images']).images
                    for img in images:
                        st.image(img)
            except Exception as e:
                log_exception(f"Image generation failed: {e}")

# --- UI: Download logs ---
with st.sidebar.expander("Logs & Diagnostics", expanded=False):
    st.markdown("**Download logs for support or debugging:**")
    try:
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Log File (lora_ui.log)", f, file_name="lora_ui.log")
    except Exception:
        log_info("No log file yet. Logs will be available after running the workflow.")
