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
    st.info(f"Loading BLIP model for captioning: {blip_model_id if not local_blip_path else local_blip_path}")
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
    st.info(f"Loading CLIP Interrogator for tagging: {clip_model_id if not local_clip_path else local_clip_path}")
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
    adv_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "optimizer": optimizer,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha
    }

def train_lora(images, captions, base_model_id, lora_model_name, lora_activation_keyword, output_dir, hf_token, device, adv_config=None, custom_code=None, precision=None):
    """
    Robust LoRA training for diffusion/image models using diffusers native LoRA support.
    - No PEFT/transformer code is used.
    - Uses diffusers' built-in LoRA hooks for image models.
    - Handles dataset, error handling, config, and cleanup.
    """
    import torch
    import os
    import shutil
    import tempfile
    import gc
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from accelerate import Accelerator
    import time

    # --- Model type check ---
    model_type = detect_model_type(base_model_id, hf_token)
    if model_type != "diffusion":
        st.error("Only image/diffusion models are supported for LoRA training. Please select a compatible model (e.g., Stable Diffusion, SDXL). Transformer/text models are not supported.")
        return None

    # --- Prepare output dir ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare dataset ---
    class ImageCaptionDataset(Dataset):
        def __init__(self, images, captions, image_size=512):
            self.images = images
            self.captions = captions
            self.image_size = image_size
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img_path = self.images[idx]
            caption = self.captions[idx]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            image = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0
            return image, caption

    # --- Save uploaded images to temp files if needed ---
    temp_dir = tempfile.mkdtemp()
    img_paths = []
    for i, img in enumerate(images):
        if hasattr(img, 'read'):
            img_path = os.path.join(temp_dir, f"temp_img_{i}.png")
            with open(img_path, "wb") as f:
                f.write(img.read())
            img_paths.append(img_path)
        else:
            img_paths.append(img)

    # --- Training config ---
    config = {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'epochs': 10,
        'image_size': 512,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'save_steps': 0,  # Only save at end
        'resume_from_checkpoint': False,
        'mixed_precision': 'fp16' if precision == torch.float16 else 'no',
    }
    if adv_config:
        config.update(adv_config)
    if custom_code:
        try:
            exec(custom_code, {}, {'custom_config': config})
        except Exception as e:
            st.warning(f"Custom config code error: {e}")

    # --- Accelerator setup ---
    accelerator = Accelerator(mixed_precision=config['mixed_precision'])
    device = accelerator.device

    # --- Load pipeline and enable LoRA ---
    st.info(f"Loading base model: {base_model_id}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            use_auth_token=hf_token,
            torch_dtype=precision if precision is not None else (torch.float16 if device.type in ["cuda", "npu"] else torch.float32),
            safety_checker=None,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        )
        pipe = pipe.to(device)
        # Enable LoRA training (diffusers >=0.20) with user-specified rank/alpha
        lora_rank = adv_config["lora_rank"] if adv_config and "lora_rank" in adv_config else 2
        lora_alpha = adv_config["lora_alpha"] if adv_config and "lora_alpha" in adv_config else 2
        pipe.enable_lora(r=lora_rank, alpha=lora_alpha)
    except Exception as e:
        st.error(f"Failed to load base model: {e}")
        shutil.rmtree(temp_dir)
        return None

    # --- Prepare dataset and dataloader ---
    dataset = ImageCaptionDataset(img_paths, captions, image_size=config['image_size'])

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=config['learning_rate'])

    # --- Training loop with OOM handling and auto batch size reduction ---
    st.info(f"Starting LoRA training for {config['epochs']} epochs...")
    global_step = 0
    pipe_dtype = pipe.unet.dtype if hasattr(pipe.unet, 'dtype') else (torch.float16 if device.type in ["cuda", "npu"] else torch.float32)
    batch_size = config['batch_size']
    min_batch_size = 1
    epoch = 0
    while epoch < config['epochs']:
        running_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        step = 0
        oom_in_epoch = False
        for step, (images_batch, captions_batch) in enumerate(dataloader):
            with accelerator.accumulate(pipe.unet):
                images_batch = images_batch.to(device=device, dtype=pipe_dtype)
                prompts = list(captions_batch)
                try:
                    latents = pipe.vae.encode(images_batch).latent_dist.sample().to(device=device, dtype=pipe_dtype)
                    latents = latents * pipe.vae.config.scaling_factor
                    noise = torch.randn_like(latents, device=device, dtype=pipe_dtype)
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (images_batch.shape[0],), device=device).long()
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    input_ids = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
                    encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                    model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    target = noise
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.warning(f"CUDA out of memory at batch size {batch_size}. Reducing batch size and retrying...")
                        torch.cuda.empty_cache()
                        oom_in_epoch = True
                        break
                    else:
                        st.error(f"Training step failed: {e}")
                        continue
                except Exception as e:
                    st.error(f"Training step failed: {e}")
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), config['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                global_step += 1
                if step % 2 == 0:
                    st.info(f"Epoch {epoch+1}/{config['epochs']}, Step {step+1}, Loss: {loss.item():.4f}")
        if oom_in_epoch:
            if batch_size > min_batch_size:
                batch_size = max(min_batch_size, batch_size // 2)
                st.warning(f"Retrying epoch {epoch+1} with reduced batch size: {batch_size}")
                torch.cuda.empty_cache()
                continue  # retry this epoch
            else:
                st.error("CUDA out of memory even at batch size 1. Cannot continue training. Try reducing image size or using a smaller model.")
                shutil.rmtree(temp_dir)
                return None
        avg_loss = running_loss / (step+1) if (step+1) > 0 else float('nan')
        st.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        epoch += 1
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Check LoRA layers are present ---
    if not hasattr(pipe, 'lora_layers') or not pipe.lora_layers:
        st.error("No LoRA layers found in the pipeline. Training did not modify any LoRA weights.")
        shutil.rmtree(temp_dir)
        return None

    # --- Save LoRA weights using diffusers API ---
    st.info("Saving LoRA weights...")
    try:
        pipe.save_lora_weights(output_dir)
        st.success(f"LoRA training complete! Weights saved to {output_dir}")
    except Exception as e:
        st.error(f"Failed to save LoRA weights: {e}")
        shutil.rmtree(temp_dir)
        return None

    # --- Clean up temp files ---
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return "trained"

def upload_model_to_hf(model_dir, hf_token, base_name, user_name):
    """
    Uploads the model to Hugging Face Hub with a unique name (timestamp-based).
    Returns the repo URL if successful.
    """
    api = HfApi()
    unique_name = f"{base_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(1000,9999)}"
    repo_id = f"{user_name}/{unique_name}"
    api.create_repo(name=unique_name, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        token=hf_token
    )
    return f"https://huggingface.co/{repo_id}"

# --- Device selection: NPU/GPU/CPU/MPS/ROCm ---
def get_best_device_and_precision():
    try:
        import torch
        device = "cpu"
        precision = torch.float32
        device_msg = "CPU detected. Training will be slow."
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            device = "cuda"
            precision = torch.float16
            device_msg = "CUDA GPU detected. Using float16 for best performance."
        elif hasattr(torch, 'has_mps') and torch.has_mps and torch.mps.is_available():
            device = "mps"
            precision = torch.float16
            device_msg = "Apple MPS detected. Using float16 for best performance."
        elif hasattr(torch, 'has_hip') and torch.has_hip and torch.has_hip():
            device = "cuda"  # ROCm is seen as 'cuda' in torch
            precision = torch.float16
            device_msg = "AMD ROCm detected. Using float16 for best performance."
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
            precision = torch.float16
            device_msg = "NPU detected. Using float16 for best performance."
        return device, precision, device_msg, torch.__version__
    except Exception as e:
        return "cpu", torch.float32, f"Device detection failed: {e}. Defaulting to CPU.", "unknown"

# --- Sidebar: Hugging Face Token (Optional for default model) ---
import os
colab_hf_token = os.environ.get('HF_TOKEN', None)
if colab_hf_token:
    st.sidebar.info('Hugging Face token loaded from Colab secret (HF_TOKEN). You can override below if needed.')
hf_token_global = st.sidebar.text_input(
    "Hugging Face Token",
    type="password",
    key="hf_token_global",
    value=colab_hf_token if colab_hf_token else ""
)

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
            st.info("To use Google Drive, run the mounting script or, in Colab, run:\n\nfrom google.colab import drive\ndrive.mount('/content/drive')\n\nThen enter the path to your model (e.g., /content/drive/MyDrive/my_model_dir).")
        elif cloud_provider == "OneDrive":
            st.info("To use OneDrive, run the mounting script or use rclone to mount your OneDrive remote. Then enter the mount path (e.g., /content/onedrive/my_model_dir). See README for details.")
        elif cloud_provider == "AWS S3":
            st.info("To use AWS S3, run the mounting script or use rclone to mount your S3 bucket. Then enter the mount path (e.g., /content/s3bucket/my_model_dir). See README for details.")
        elif cloud_provider == "Azure Blob":
            st.info("To use Azure Blob, run the mounting script or use rclone to mount your Azure remote. Then enter the mount path (e.g., /content/azure/my_model_dir). See README for details.")
        elif cloud_provider == "GCP Storage":
            st.info("To use Google Cloud Storage, run the mounting script or use rclone to mount your GCS bucket. Then enter the mount path (e.g., /content/gcs/my_model_dir). See README for details.")
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

if images:
    # Always show images in the UI
    st.write("### Uploaded Images")
    # --- Auto-caption/tag logic ---
    auto_captions = st.session_state.get('auto_captions', None)
    captions_ready = auto_captions is not None and len(auto_captions) == len(images)
    if caption_mode == "Automatic (Recommended)":
        st.info(f"Using {'BLIP' if caption_type == 'Caption (Sentence)' else 'CLIP Interrogator'} model for automatic {caption_type.lower()}.")
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
                        st.error("Auto-captioning/tagging did not return captions for all images. Please try again.")
                except Exception as e:
                    # Clear any partial/failed results
                    for idx in range(len(images)):
                        st.session_state.pop(f"caption_{idx}", None)
                    st.session_state.pop('auto_captions', None)
                    st.error(f"Auto-captioning/tagging failed: {e}")
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
                    st.warning(f"Warning: {r['name']} may require more VRAM than detected ({vram_gb} GB). Training or inference may fail or be very slow.")
        else:
            st.warning("No models found or error searching Hugging Face.")
    # --- Manual model selection ---
    custom_model_id = st.text_input("Or enter a custom model ID (Hugging Face)")
    if custom_model_id:
        # Example: Check for large models
        if vram_gb and ('xl' in custom_model_id.lower() or 'sdxl' in custom_model_id.lower()) and vram_gb < 12:
            st.warning(f"Warning: {custom_model_id} may require more VRAM than detected ({vram_gb} GB). Training or inference may fail or be very slow.")

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
            st.info("Auto-generating captions/tags for all images...")
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
                    st.error("Auto-captioning/tagging did not return captions for all images. Please try again.")
            except Exception as e:
                st.session_state.pop('auto_captions', None)
                st.error(f"Auto-captioning/tagging failed: {e}")
                st.stop()
    # Validate all images have captions/tags
    missing = [idx for idx in range(len(images)) if not st.session_state.get(f"caption_{idx}", "").strip()]
    if missing:
        st.error(f"Please provide a caption/tag for all images. Missing for: {', '.join(str(images[i].name) for i in missing)}")
    else:
        captions = [st.session_state.get(f"caption_{idx}", "") for idx in range(len(images))]
        output_dir = f"lora_output/{lora_model_name}"
        st.info("Training LoRA... This may take a while.")
        # --- Improved Device selection ---
        device, precision, device_msg, torch_version = get_best_device_and_precision()
        st.info(f"{device_msg} (torch version: {torch_version})")
        if device == "cpu":
            st.warning("No supported accelerator detected. Training will be slow. For best performance, use a machine with a supported GPU, Apple Silicon (MPS), or AMD ROCm. See https://pytorch.org/get-started/locally/ for install instructions.")
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
            st.success(f"LoRA training complete! Weights saved to {output_dir}")
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
                    st.success(f"Model uploaded to Hugging Face (private): {repo_url}")
                    st.markdown(f"[View on Hugging Face]({repo_url})")
                except Exception as e:
                    st.error(f"Automatic upload to Hugging Face failed: {e}")
        else:
            st.error("LoRA training failed. See above for details.")
# --- Always show image generation UI after training ---
if st.session_state.get('model_ready', False):
    st.subheader("8. Model Download & Upload")
    model_dir = st.session_state['trained_model_dir']
    repo_url = st.session_state.get('uploaded_repo_url', None)
    if repo_url:
        st.info(f"Model is available on Hugging Face (private): {repo_url}")
        st.markdown(f"[View on Hugging Face]({repo_url})")
    # Download option
    zip_path = f"{model_dir}.zip"
    if not os.path.exists(zip_path):
        shutil.make_archive(model_dir, 'zip', model_dir)
    with open(zip_path, "rb") as f:
        st.download_button("Download Trained Model (.zip)", f, file_name=os.path.basename(zip_path))
    st.info(f"Model files are saved in: {model_dir} (Colab: /content/Lora_trainer/{model_dir})")
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
    st.info("You can queue prompts and settings below. Images will be generated automatically as soon as training completes. The LoRA activation keyword will be included in each prompt.")
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
                st.error(f"Image generation failed: {e}")
