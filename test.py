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
from peft import LoraConfig, get_peft_model
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
DEFAULT_BASE_MODEL = {
    "modelId": "runwayml/stable-diffusion-v1-5",
    "name": "Stable Diffusion v1.5",
    "desc": "Popular base model for image generation and LoRA training.",
    "compatible_lora": ["kohya-ss", "diffusers"],
    "default": True
}
DEFAULT_LORA_METHOD = {
    "name": "kohya-ss LoRA",
    "desc": "Efficient and widely used LoRA training for diffusion models.",
    "compatible_base": ["Stable Diffusion v1.5", "SDXL", "Stable Diffusion 2.x"],
    "default": True
}
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# --- Progress bar utility for model downloads ---
def download_with_progress(model_id, description, download_fn, *args, **kwargs):
    """
    Download a model with a Streamlit progress bar and details.
    model_id: str, description: str, download_fn: callable, *args: args for download_fn
    Returns the downloaded object.
    """
    st.info(f"Downloading model: {model_id}\n\n{description}")
    progress_bar = st.progress(0)
    # Simulate progress for demonstration; replace with real progress if possible
    for percent in range(0, 100, 10):
        progress_bar.progress(percent)
        time.sleep(0.1)
    # Actual download
    obj = download_fn(*args, **kwargs)
    progress_bar.progress(100)
    st.success(f"Downloaded: {model_id}")
    return obj

def generate_captions_for_images(images, model_name, hf_token):
    """
    Generate captions for a list of images using BLIP.
    Returns a list of captions (one per image).
    """
    processor = download_with_progress(
        model_name,
        "BLIP Processor for image captioning.",
        BlipProcessor.from_pretrained,
        model_name,
        use_auth_token=hf_token
    )
    model = download_with_progress(
        model_name,
        "BLIP Model for image captioning.",
        BlipForConditionalGeneration.from_pretrained,
        model_name,
        use_auth_token=hf_token
    )
    captions = []
    for img_file in images:
        image = Image.open(img_file).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def generate_tags_for_images(images, hf_token):
    """
    Generate tags for a list of images using CLIP Interrogator.
    Returns a list of tags (one per image).
    """
    ci = download_with_progress(
        "ViT-L-14/openai",
        "CLIP Interrogator (ViT-L-14) for image tagging.",
        lambda: Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    )
    tags = []
    for img_file in images:
        image = Image.open(img_file).convert('RGB')
        tag = ci.interrogate(image)
        tags.append(tag)
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

def train_lora(images, captions, base_model_id, lora_model_name, lora_activation_keyword, output_dir, hf_token, device, adv_config=None, custom_code=None, precision=None):
    pipe = download_with_progress(
        base_model_id,
        "Stable Diffusion base model for LoRA training.",
        StableDiffusionPipeline.from_pretrained,
        base_model_id,
        use_auth_token=hf_token,
        torch_dtype=precision if precision is not None else (torch.float16 if device in ["cuda", "npu"] else torch.float32),
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    )
    pipe = pipe.to(device)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["attn1", "attn2"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    # Prepare dataset (images, captions)
    # Use adv_config for batch size, learning rate, epochs, optimizer
    if adv_config:
        batch_size = adv_config.get("batch_size", 4)
        learning_rate = adv_config.get("learning_rate", 1e-4)
        epochs = adv_config.get("epochs", 10)
        optimizer = adv_config.get("optimizer", "AdamW")
    else:
        batch_size, learning_rate, epochs, optimizer = 4, 1e-4, 10, "AdamW"
    # Optionally execute custom_code (highly advanced)
    if custom_code and custom_code.strip():
        try:
            exec(custom_code, globals())
        except Exception as e:
            st.warning(f"Custom config/code error: {e}")
    # ... (implement a PyTorch dataset for your images/captions, use batch_size, learning_rate, epochs, optimizer)
    # Train LoRA (implement a training loop or use diffusers' LoRATrainer)
    # Save LoRA weights
    pipe.unet.save_pretrained(output_dir)
    return output_dir

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

# --- Sidebar: Hugging Face Token ---
st.sidebar.header("Hugging Face Token (Required for Model Downloads)")
hf_token_global = st.sidebar.text_input("Hugging Face Token", type="password", key="hf_token_global")

# --- Main UI ---
st.title("LoRA Training UI (Easy & Advanced Modes)")

# --- Step 1: Upload Images ---
st.subheader("1. Upload Images")
images = st.file_uploader("Upload images for LoRA training", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Step 2: Captioning (Default: Automatic) ---
st.subheader("2. Captioning")
caption_mode = st.radio("How do you want to provide captions/tags?", ("Automatic (Recommended)", "Manual"), index=0)
caption_type = st.radio("Captioning/Tagging Mode", ["Caption (Sentence)", "Tags (Keywords)"], index=0)

if images:
    # Always show images in the UI
    st.write("### Uploaded Images")
    for idx, img in enumerate(images):
        with st.container():
            st.image(img, width=200, caption=img.name)
            st.text_area(
                f"{'Caption' if caption_type == 'Caption (Sentence)' else 'Tags'} for {img.name}",
                value=st.session_state.get(f"caption_{idx}", ""),
                key=f"caption_{idx}"
            )
    if caption_mode == "Automatic (Recommended)":
        st.info(f"Using BLIP model: `{DEFAULT_CAPTION_MODEL}` for automatic captioning.")
        if st.button("Auto Caption/Tag All Images"):
            with st.spinner("Downloading captioning/tagging model and generating captions/tags for all images..."):
                try:
                    if caption_type == "Caption (Sentence)":
                        gen_captions = generate_captions_for_images(
                            images,
                            DEFAULT_CAPTION_MODEL,
                            hf_token_global
                        )
                    else:
                        gen_captions = generate_tags_for_images(images, hf_token_global)
                    for idx, cap in enumerate(gen_captions):
                        st.session_state[f"caption_{idx}"] = cap
                    st.success("Captions/Tags generated! You can edit them below.")
                except Exception as e:
                    st.error(f"Auto-captioning/tagging failed: {e}")

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
        else:
            st.warning("No models found or error searching Hugging Face.")
    # --- LoRA method selection (future) ---
    # ... (dropdown for LoRA methods, if needed) ...

# --- Advanced Training Settings ---
with st.expander("Advanced Training Settings", expanded=False):
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=4)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%e")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    optimizer = st.selectbox("Optimizer", ["AdamW", "SGD"], index=0)
    adv_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "optimizer": optimizer
    }

# --- Highly Advanced: Edit Training Config/Code ---
default_code = (
    "# Example: Custom training config as Python dict\n"
    "custom_config = {\n"
    "    'batch_size': 4,\n"
    "    'learning_rate': 1e-4,\n"
    "    'epochs': 10,\n"
    "    'optimizer': 'AdamW'\n"
    "}\n"
)
with st.expander("Highly Advanced: Edit Training Config/Code", expanded=False):
    custom_code = st.text_area("Edit training config/code (Python)", value=default_code, height=180)

# --- Step 4: LoRA Metadata ---
st.subheader("4. LoRA Metadata")
lora_model_name = st.text_input("LoRA Model Name (required)", value="my_lora")
lora_activation_keyword = st.text_input("Activation Keyword (required)", value="my_lora_keyword")

# --- Step 5: Start Training ---
if st.button("Start LoRA Training"):
    # Auto-generate captions/tags if in automatic mode and not already generated
    if caption_mode == "Automatic (Recommended)":
        st.info("Auto-generating captions/tags for all images...")
        try:
            if caption_type == "Caption (Sentence)":
                gen_captions = generate_captions_for_images(
                    images,
                    DEFAULT_CAPTION_MODEL,
                    hf_token_global
                )
            else:
                gen_captions = generate_tags_for_images(images, hf_token_global)
            for idx, cap in enumerate(gen_captions):
                st.session_state[f"caption_{idx}"] = cap
            st.success("Captions/Tags generated!")
        except Exception as e:
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
        train_lora(
            images,
            captions,
            DEFAULT_BASE_MODEL["modelId"],
            lora_model_name,
            lora_activation_keyword,
            output_dir,
            hf_token_global,
            device,
            adv_config=adv_config,
            custom_code=custom_code,
            precision=precision
        )
        st.success(f"LoRA training complete! Weights saved to {output_dir}")
        st.session_state['trained_model_dir'] = output_dir
        st.session_state['model_ready'] = True

# --- Step 6: Model Download & Upload Options ---
if st.session_state.get('model_ready', False):
    st.subheader("6. Model Download & Upload")
    model_dir = st.session_state['trained_model_dir']
    download_toggle = st.checkbox("Download model after training", value=True)
    upload_toggle = st.checkbox("Upload model to Hugging Face Hub", value=False)
    hf_username = st.text_input("Your Hugging Face username (for upload)", value="", key="hf_username")
    if download_toggle:
        # Zip the model directory for download
        zip_path = f"{model_dir}.zip"
        if not os.path.exists(zip_path):
            shutil.make_archive(model_dir, 'zip', model_dir)
        with open(zip_path, "rb") as f:
            st.download_button("Download Trained Model (.zip)", f, file_name=os.path.basename(zip_path))
        st.info(f"Model files are saved in: {model_dir} (Colab: /content/Lora_trainer/{model_dir})")
    else:
        st.info(f"Model files are saved in: {model_dir} (Colab: /content/Lora_trainer/{model_dir})")
    if upload_toggle and st.button("Upload Model to Hugging Face"):
        if not hf_token_global or not hf_username:
            st.warning("Please provide your Hugging Face token and username.")
        else:
            with st.spinner("Uploading model to Hugging Face Hub..."):
                try:
                    repo_url = upload_model_to_hf(model_dir, hf_token_global, lora_model_name, hf_username)
                    st.success(f"Model uploaded to Hugging Face: {repo_url}")
                    st.markdown(f"[View on Hugging Face]({repo_url})")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

# --- Step 7: Generate Images with Trained Model ---
if st.session_state.get('model_ready', False):
    st.subheader("7. Generate Images with Trained Model")
    st.info("You can now generate images using your newly trained LoRA model. Edit the prompts below as needed.")
    sample_prompts = [
        "A futuristic cityscape at sunset, ultra detailed, trending on artstation",
        "A portrait of a cat wearing sunglasses, digital art, vibrant colors",
        "A fantasy landscape with mountains and rivers, epic lighting"
    ]
    prompts = [st.text_input(f"Prompt {i+1}", value=sample_prompts[i]) for i in range(len(sample_prompts))]
    num_images = st.slider("Number of images to generate per prompt", 1, 4, 1)
    if st.button("Generate Images"):
        with st.spinner("Loading model and generating images..."):
            try:
                device, precision, device_msg, torch_version = get_best_device_and_precision()
                pipe = StableDiffusionPipeline.from_pretrained(
                    st.session_state['trained_model_dir'],
                    torch_dtype=precision
                )
                pipe = pipe.to(device)
                for prompt in prompts:
                    st.markdown(f"**Prompt:** {prompt}")
                    images = pipe([prompt]*num_images).images
                    for img in images:
                        st.image(img)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
