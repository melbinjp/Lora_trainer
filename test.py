import streamlit as st
import platform
import psutil
import shutil
import requests
from PIL import Image
import io
from huggingface_hub import snapshot_download, HfApi
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTextModel, AutoTokenizer
from clip_interrogator import Config, Interrogator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig
import threading
import time
from huggingface_hub import hf_hub_download
import os
import random
from datetime import datetime
import logging
import traceback
from datasets import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- Logging setup ---
LOG_FILE = "lora_ui.log"
logging.basicConfig(filename=LOG_FILE, filemode="a", format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def clear_log_file():
    with open(LOG_FILE, "w") as f: f.truncate(0)

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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log_error("PyTorch is not installed. Please run the setup script.")

# --- Default settings ---
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# --- Model/Resource Functions ---
@st.cache_resource(show_spinner=False)
def get_blip_model_files(hf_token, blip_model_id):
    processor = BlipProcessor.from_pretrained(blip_model_id, use_auth_token=hf_token)
    model = BlipForConditionalGeneration.from_pretrained(blip_model_id, use_auth_token=hf_token)
    return processor, model

def generate_captions_for_images(images, hf_token, blip_model_id):
    processor, model = get_blip_model_files(hf_token, blip_model_id)
    captions = []
    for img_file in images:
        image = Image.open(img_file).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        captions.append(processor.decode(out[0], skip_special_tokens=True))
    return captions

def get_system_resources():
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    vram_gb = None
    if TORCH_AVAILABLE and torch.cuda.is_available():
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    return ram_gb, vram_gb

def get_default_model_for_resources(ram, vram):
    if vram and vram >= 8:
        return {"modelId": "runwayml/stable-diffusion-v1-5", "name": "Stable Diffusion v1.5"}
    else:
        return {"modelId": "stabilityai/sd-turbo", "name": "SD Turbo (Efficient)"}

# --- Image Generation Function ---
@st.cache_resource(show_spinner=False)
def get_lora_pipeline(_base_model_id, _lora_output_dir, _hf_token):
    pipe = StableDiffusionPipeline.from_pretrained(_base_model_id, use_auth_token=_hf_token)
    pipe.unet.load_attn_procs(_lora_output_dir)
    pipe.to("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    return pipe

def generate_image_with_lora(base_model_id, lora_output_dir, prompt, hf_token):
    log_info("Loading pipeline with trained LoRA...")
    pipeline = get_lora_pipeline(base_model_id, lora_output_dir, hf_token)
    log_info(f"Generating image for prompt: {prompt}")
    with st.spinner("Generating image..."):
        image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    return image

# --- Core LoRA Training Function ---
def train_lora(images, captions, base_model_id, output_dir, lora_rank=4, learning_rate=1e-4, num_train_epochs=15, train_batch_size=1, hf_token=None):
    # ... (Full implementation from before)
    pass

# --- Automated Training Orchestrator ---
def run_automated_training(images, lora_name, hf_token):
    # ... (Full implementation from before)
    pass

# --- UI Rendering Functions ---
def render_simple_mode(hf_token):
    st.header("Simple Mode: Train in 3 Easy Steps")
    st.markdown("Upload images of your subject, give your model a name (optional), and click train!")
    images = st.file_uploader("1. Upload Your Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="simple_images")
    if images:
        cols = st.columns(5)
        for i, img in enumerate(images):
            cols[i % 5].image(img, use_column_width=True, caption=f"Image {i+1}")
    lora_name = st.text_input("2. Name Your Model (Optional)", key="simple_lora_name", placeholder="my-awesome-lora")
    if st.button("3. ✨ Start Training ✨", key="simple_start_training", type="primary", disabled=not images, use_container_width=True):
        run_automated_training(images, lora_name, hf_token)

    if st.session_state.get('simple_model_trained', False):
        st.success(f"Training complete! Your LoRA model '{st.session_state['simple_model_name']}' is ready.")
        st.balloons()
        if 'hf_repo_url' in st.session_state:
            st.markdown(f"**✅ Uploaded to Hugging Face:** [{st.session_state.hf_repo_url}]({st.session_state.hf_repo_url})")

        zip_path = f"{st.session_state['simple_output_dir']}.zip"
        if not os.path.exists(zip_path):
            shutil.make_archive(st.session_state['simple_output_dir'], 'zip', st.session_state['simple_output_dir'])
        with open(zip_path, "rb") as f:
            st.download_button("Download Model (.zip)", f, file_name=os.path.basename(zip_path))

        st.subheader("Generate Images")
        prompt = st.text_input("Enter a prompt...", key="simple_prompt", placeholder=f"A photo of {st.session_state['simple_model_name']} style dog")
        if st.button("Generate Image", key="simple_generate"):
            ram, vram = get_system_resources()
            base_model_id = get_default_model_for_resources(ram, vram)["modelId"]
            lora_output_dir = st.session_state['simple_output_dir']
            generated_image = generate_image_with_lora(base_model_id, lora_output_dir, prompt, hf_token)
            st.image(generated_image, caption=f"Generated image for: {prompt}")

def render_advanced_mode(hf_token, base_model, vram):
    st.header("Advanced Mode")
    st.warning("Advanced mode is a placeholder in this version.")
    # In a full implementation, this would contain the detailed UI from the original file,
    # and its "Start Training" button would call `train_lora` with the user-selected parameters.
    # The post-training UI would call `generate_image_with_lora`.

# --- Main App Execution ---
st.title("LoRA Training UI")
# ... (Sidebar and mode selection logic from before)

if st.session_state.mode == "Simple":
    render_simple_mode(hf_token_global)
else:
    render_advanced_mode(hf_token_global, DEFAULT_BASE_MODEL, vram_gb)
