import streamlit as st
import os
import shutil
from PIL import Image
import time

st.set_page_config(page_title="Personal LoRA Trainer UI", layout="wide")

st.title("🎨 High-Fidelity Personal LoRA Workflow")
st.markdown("Upload your images, auto-caption them, and export your optimized dataset for Flux.1 training.")

# --- Sidebar: Hugging Face Token ---
st.sidebar.header("Hugging Face Token")
st.sidebar.info("Required if you want to use advanced Auto-Captioning models from the Hub.")
hf_token_global = st.sidebar.text_input("Hugging Face Token", type="password", key="hf_token_global")

# --- Default Hyperparameters ---
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_LR = 4e-4
DEFAULT_STEPS_PER_IMAGE = 100

dataset_name = st.text_input("Dataset Name (e.g., 'ohwx_man')", value="my_subject")
trigger_word = st.text_input("Trigger Word (e.g., 'ohwx')", value="my_trigger")

images = st.file_uploader("Upload 15-30 high-quality images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Helper function for auto-captioning
def auto_caption_images(uploaded_images, trigger):
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        st.error("Missing transformers library. Please run: pip install transformers torch torchvision")
        return []

    st.info("Downloading/Loading BLIP model... This might take a minute the first time.")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    captions = []
    progress_bar = st.progress(0)
    for i, img_file in enumerate(uploaded_images):
        raw_image = Image.open(img_file).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        # Prepend the trigger word
        captions.append(f"a photo of {trigger}, {caption}")
        progress_bar.progress((i + 1) / len(uploaded_images))
    return captions

if images:
    st.write("### Uploaded Images & Captions")
    
    # Initialize session state for captions
    if 'captions' not in st.session_state:
        st.session_state.captions = {}
        for img in images:
            st.session_state.captions[img.name] = f"a photo of {trigger_word}, "

    caption_mode = st.radio("Captioning Mode", ["Automatic (AI-Generated)", "Manual"], index=0)
    
    if caption_mode == "Automatic (AI-Generated)":
        if st.button("✨ Auto-Caption All Images"):
            with st.spinner("Generating captions using AI..."):
                try:
                    generated = auto_caption_images(images, trigger_word)
                    if generated:
                        for idx, img in enumerate(images):
                            st.session_state.captions[img.name] = generated[idx]
                        st.success("Auto-captioning complete!")
                except Exception as e:
                    st.error(f"Error during auto-captioning: {e}")

    # Display images and captions
    cols = st.columns(3)
    for idx, img_file in enumerate(images):
        with cols[idx % 3]:
            img = Image.open(img_file)
            st.image(img, use_column_width=True)
            st.session_state.captions[img_file.name] = st.text_area(
                f"Caption for {img_file.name}", 
                value=st.session_state.captions.get(img_file.name, ""), 
                key=f"cap_{idx}"
            )

    # --- Advanced Settings Expander ---
    with st.expander("⚙️ Advanced Training Settings (Hidden for Simplicity)"):
        st.markdown("These settings will be embedded in your exported dataset to guide your Colab training.")
        lora_rank = st.number_input("LoRA Rank", min_value=4, max_value=128, value=DEFAULT_LORA_RANK)
        lora_alpha = st.number_input("LoRA Alpha", min_value=4, max_value=128, value=DEFAULT_LORA_ALPHA)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=DEFAULT_LR, format="%e")
        total_steps = st.number_input("Total Training Steps", min_value=100, max_value=10000, value=max(1500, len(images)*DEFAULT_STEPS_PER_IMAGE))

    # --- Export ---
    st.markdown("---")
    if st.button("🚀 Prepare & Export Dataset", type="primary"):
        export_dir = f"dataset_{dataset_name}"
        os.makedirs(export_dir, exist_ok=True)
        
        for img_file in images:
            img = Image.open(img_file)
            base_name = os.path.splitext(img_file.name)[0]
            img.save(os.path.join(export_dir, img_file.name))
            with open(os.path.join(export_dir, f"{base_name}.txt"), "w") as f:
                f.write(st.session_state.captions.get(img_file.name, ""))
        
        # Save advanced config choices to a txt file so they know what to put in colab
        with open(os.path.join(export_dir, "training_hyperparameters.txt"), "w") as f:
            f.write(f"Rank: {lora_rank}\nAlpha: {lora_alpha}\nLR: {learning_rate}\nSteps: {total_steps}")
            
        zip_name = f"{export_dir}.zip"
        shutil.make_archive(export_dir, 'zip', export_dir)
        
        with open(zip_name, "rb") as f:
            st.download_button(f"📥 Download {zip_name} (Upload to Colab)", f, file_name=zip_name)
        st.success(f"Dataset exported! You can now use the `colab_quickstart.ipynb` to train.")

# --- Local Inference Guide ---
with st.expander("🖥️ Local Inference Guide (Intel Ultra 5)"):
    st.warning("⚠️ Full-precision Flux.1 inference will exceed your 16GB system RAM and crash. You MUST use GGUF quantization.")
    st.markdown("""
    ### Post-Processing (Realism Enhancement):
    To remove the "AI Sheen" and achieve maximum fidelity, apply these structural modulations during generation:
    *   **FreeU Parameters**: `b1: 1.2`, `b2: 1.4`, `s1: 0.8`, `s2: 0.4`
    *   **FaceDetailer**: Perform a pass at `0.35` denoising strength.
    """)
