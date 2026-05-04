import streamlit as st
import os
import shutil
import glob
from PIL import Image

st.set_page_config(page_title="Personal LoRA Trainer UI", layout="wide")

# --- Default Hyperparameters (Optimal for Flux.1 on T4 / Intel Ultra 5 Inference) ---
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_LR = 4e-4
DEFAULT_STEPS_PER_IMAGE = 100

st.title("🎨 High-Fidelity Personal LoRA Workflow")
st.markdown("""
This streamlined UI helps you prepare your dataset for **Flux.1 [dev]** training and provides instructions for running memory-efficient inference locally using **GGUF** and **OpenVINO**.
""")

# --- Step 1: Dataset Preparation ---
st.header("1. Dataset Engineering")
st.markdown("""
For best realism and likeness coherence:
*   **Optimal Image Count**: 20 - 25 images.
*   **Diversity**: Include at least 30% profile or angled shots.
*   **Resolution**: Training will occur at 1024x1024.
""")

dataset_name = st.text_input("Dataset Name (e.g., 'ohwx_man')", value="my_subject")
trigger_word = st.text_input("Trigger Word (e.g., 'ohwx')", value="my_trigger")

images = st.file_uploader("Upload 15-30 high-quality images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if images:
    if len(images) < 15:
        st.warning(f"You only uploaded {len(images)} images. Consider uploading at least 15 for better facial consistency.")
    elif len(images) > 30:
        st.warning(f"You uploaded {len(images)} images. This may cause overfitting or very long training times.")
    else:
        st.success(f"{len(images)} images uploaded. Good dataset size!")
    
    st.subheader("Captions")
    st.info(f"Using a consistent caption format is crucial. Ensure your trigger word '{trigger_word}' is clearly separated from the environment.")
    
    # Store captions
    if 'captions' not in st.session_state:
        st.session_state.captions = {}
        
    cols = st.columns(3)
    for idx, img_file in enumerate(images):
        with cols[idx % 3]:
            img = Image.open(img_file)
            st.image(img, use_column_width=True)
            default_cap = f"a photo of {trigger_word}, "
            st.session_state.captions[img_file.name] = st.text_input(f"Caption for {img_file.name}", value=st.session_state.captions.get(img_file.name, default_cap), key=f"cap_{idx}")

    # Dataset Export
    if st.button("Export Dataset for Colab"):
        export_dir = f"dataset_{dataset_name}"
        os.makedirs(export_dir, exist_ok=True)
        
        for img_file in images:
            img = Image.open(img_file)
            base_name = os.path.splitext(img_file.name)[0]
            # Save image
            img.save(os.path.join(export_dir, img_file.name))
            # Save txt caption
            with open(os.path.join(export_dir, f"{base_name}.txt"), "w") as f:
                f.write(st.session_state.captions.get(img_file.name, ""))
        
        zip_name = f"{export_dir}.zip"
        shutil.make_archive(export_dir, 'zip', export_dir)
        
        with open(zip_name, "rb") as f:
            st.download_button(f"Download {zip_name} (Upload to Colab)", f, file_name=zip_name)
        st.success(f"Dataset exported! You can now use the `colab_quickstart.ipynb` to train via AI Toolkit.")

# --- Step 2: Training Parameters Guide ---
st.header("2. Training Strategy (Colab)")
st.markdown(f"""
Your training will be executed via the **Google Colab MCP Server** or by manually uploading the dataset to the `colab_quickstart.ipynb` notebook.
The notebook utilizes the **AI Toolkit (Ostris)** optimized for 16GB VRAM (T4).

**Optimal Hyperparameters for {dataset_name}:**
*   **Rank / Alpha**: {DEFAULT_LORA_RANK} / {DEFAULT_LORA_ALPHA}
*   **Learning Rate**: {DEFAULT_LR}
*   **Steps**: {max(1500, len(images)*DEFAULT_STEPS_PER_IMAGE if images else 2000)} (Approx 100 steps per image)
*   **Optimizer**: AdamW (8-bit)
""")

# --- Step 3: Local Inference Guide ---
st.header("3. Local Inference (Intel Ultra 5)")
st.warning("⚠️ Full-precision Flux.1 inference will exceed your 16GB system RAM and crash. You MUST use GGUF quantization.")

st.markdown("""
### Recommended Workflow for your Hardware:
1.  **UI**: Use **Forge** or **ComfyUI**.
2.  **Model Format**: Download a **GGUF Q4_0** or **Q3_K_S** version of Flux.1 [dev] (~6.8GB).
3.  **Text Encoder**: Use a quantized FP8 or GGUF version of the T5-XXL text encoder and force it to load on CPU.
4.  **Hardware Acceleration**: Install the `Optimum-Intel` extension (or `ComfyUI-OpenVINO`) to offload math to the Arc iGPU/NPU.

### Post-Processing (Realism Enhancement):
To remove the "AI Sheen" and achieve maximum fidelity, apply these structural modulations during generation:
*   **FreeU Parameters**: `b1: 1.2`, `b2: 1.4`, `s1: 0.8`, `s2: 0.4`
*   **FaceDetailer**: Perform a pass at `0.35` denoising strength using an SDXL/Flux face detector to lock in the likeness.
""")
