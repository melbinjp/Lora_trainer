import streamlit as st
import os
import shutil
from PIL import Image
import time
from huggingface_hub import hf_hub_download

import sys
from dotenv import load_dotenv

# Disable all telemetry before loading libraries
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OPENVINO_DISABLE_TELEMETRY"] = "1"

# Load environment variables from .env file
load_dotenv()

# Add src to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
try:
    from engine.inference_engine import load_flux_pipeline, run_inference
    from utils.helpers import get_system_resources, setup_logging, log_info
    setup_logging()
except ImportError:
    pass

st.set_page_config(page_title="Personal LoRA Trainer UI", layout="wide")

st.title("🎨 High-Fidelity Personal LoRA Workflow")

# --- Sidebar: System Resources ---
ram_gb, vram_gb = get_system_resources()
st.sidebar.header("System Resources")
st.sidebar.markdown(f"**Detected RAM:** {ram_gb} GB")
if vram_gb:
    st.sidebar.markdown(f"**Detected VRAM:** {vram_gb} GB")
else:
    st.sidebar.markdown("**Detected VRAM:** iGPU/None")

st.sidebar.markdown("---")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset Engineering", "☁️ Cloud Training (Colab)", "🖥️ Local Inference", "🌐 Distributed Batch"])

# --- Tab 1: Dataset Engineering ---
with tab1:
    st.header("Step 1: Dataset Engineering")
    st.markdown("Upload your images, auto-caption them, and export your optimized dataset for Flux.1 training.")

    # --- Sidebar: Hugging Face Token (Moved to Sidebar for global access) ---
    st.sidebar.header("Hugging Face Settings")
    hf_token_global = st.sidebar.text_input("Hugging Face Token", type="password", key="hf_token_global", value=os.environ.get("HF_TOKEN", ""))
    st.sidebar.info("Required for downloading models and using private LoRAs.")

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
            captions.append(f"a photo of {trigger}, {caption}")
            progress_bar.progress((i + 1) / len(uploaded_images))
        return captions

    if images:
        st.write("### Uploaded Images & Captions")
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

        cols = st.columns(3)
        for idx, img_file in enumerate(images):
            with cols[idx % 3]:
                img = Image.open(img_file)
                st.image(img, use_container_width=True)
                st.session_state.captions[img_file.name] = st.text_area(
                    f"Caption for {img_file.name}", 
                    value=st.session_state.captions.get(img_file.name, ""), 
                    key=f"cap_{idx}"
                )

        with st.expander("⚙️ Advanced Training Settings"):
            lora_rank = st.number_input("LoRA Rank", min_value=4, max_value=128, value=DEFAULT_LORA_RANK)
            lora_alpha = st.number_input("LoRA Alpha", min_value=4, max_value=128, value=DEFAULT_LORA_ALPHA)
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=DEFAULT_LR, format="%e")
            total_steps = st.number_input("Total Training Steps", min_value=100, max_value=10000, value=max(1500, len(images)*DEFAULT_STEPS_PER_IMAGE))

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
            
            with open(os.path.join(export_dir, "training_hyperparameters.txt"), "w") as f:
                f.write(f"Rank: {lora_rank}\nAlpha: {lora_alpha}\nLR: {learning_rate}\nSteps: {total_steps}")
            
            shutil.make_archive(export_dir, 'zip', export_dir)
            zip_name = f"{export_dir}.zip"
            with open(zip_name, "rb") as f:
                st.download_button(f"📥 Download {zip_name}", f, file_name=zip_name)
            st.success(f"Dataset exported!")

# --- Tab 2: Cloud Training ---
with tab2:
    st.header("Step 2: Cloud Training (Google Colab)")
    st.markdown("""
    1. Upload the exported `.zip` file to your Google Drive.
    2. Open the `colab_quickstart.ipynb` notebook.
    3. Run the cells to start the **Ostris AI Toolkit**.
    4. Once training is complete, download your trained `pytorch_lora_weights.safetensors` file.
    """)
    if st.button("Download Colab Quickstart Notebook"):
        notebook_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "colab_quickstart.ipynb")
        with open(notebook_path, "rb") as f:
            st.download_button("📥 Download Notebook", f, file_name="colab_quickstart.ipynb")

# --- Helper: Model Status ---
def get_model_status(repo_id):
    # Determine the local folder name from repo ID
    repo_name = repo_id.split("/")[-1].lower()
    
    if not os.path.exists("models"):
        return "⏳ DOWNLOAD REQUIRED"
        
    # Standardize comparison: remove dots, dashes, and handle flux.1 vs flux
    def normalize(name):
        name = name.lower().replace("flux.1", "flux").replace("flux1", "flux")
        return name.replace(".", "").replace("-", "")
        
    norm_repo = normalize(repo_name)
    
    for folder in os.listdir("models"):
        if normalize(folder) == norm_repo:
            path = os.path.join("models", folder)
            # A model is "READY" if the transformer weights are present
            # check for both openvino and diffusers structures
            ov_bin = os.path.join(path, "transformer", "openvino_model.bin")
            diff_bin = os.path.join(path, "transformer", "diffusion_pytorch_model.safetensors")
            if os.path.exists(ov_bin) or os.path.exists(diff_bin):
                return "✅ READY (Local)"
            
    return "⏳ DOWNLOAD REQUIRED"

# --- Tab 3: Local Inference ---
with tab3:
    st.header("Step 3: Local Inference")
    st.info("Run inference locally optimized for your detected hardware.")

    col_inf_1, col_inf_2 = st.columns([1, 1])

    with col_inf_1:
        st.subheader("Configuration")
        
        hardware_target = st.radio("Hardware Target", ["Standard (CPU/Offload)", "Intel Optimized (OpenVINO iGPU/NPU)"], index=1)
        
        # Dynamic model mapping based on hardware
        if hardware_target == "Intel Optimized (OpenVINO iGPU/NPU)":
            model_map = {
                "FLUX.1 Schnell": "OpenVINO/FLUX.1-schnell-int4-ov",
                "FLUX.1 Dev": "OpenVINO/FLUX.1-dev-int4-ov"
            }
        else:
            model_map = {
                "FLUX.1 Schnell": "black-forest-labs/FLUX.1-schnell",
                "FLUX.1 Dev": "black-forest-labs/FLUX.1-dev"
            }

        display_options = []
        for name, repo_id in model_map.items():
            status = get_model_status(repo_id)
            display_options.append(f"{name} ({status})")
            
        selected_display = st.selectbox("Base Model", display_options, index=0)
        # Extract pure name to get repo ID
        pure_name = selected_display.split(" (")[0]
        selected_model = model_map[pure_name]
        
        lora_file = st.file_uploader("Upload your trained LoRA (.safetensors)", type=["safetensors"])
        
        # Display Model Info
        st.markdown("---")
        st.markdown("**Model Details:**")
        if "schnell" in selected_model.lower():
            st.caption("FLUX.1 Schnell: Fast, 4-step generation. Good for rapid prototyping.")
        else:
            st.caption("FLUX.1 Dev: High quality, 20+ step generation. Requires Hugging Face Token.")
            
        if lora_file:
            lora_size_mb = round(len(lora_file.getvalue()) / (1024 * 1024), 2)
            st.caption(f"Uploaded LoRA Size: {lora_size_mb} MB")
            if lora_size_mb > 500:
                st.warning("Large LoRA detected. This may increase loading time.")
        
        if st.button("🎁 Download Example LoRA (Realism)"):
            with st.spinner("Downloading example LoRA from Hugging Face..."):
                try:
                    example_lora_path = hf_hub_download(
                        repo_id="Shakker-Labs/Flux.1-Dev-LoRA-Realism-v2",
                        filename="pytorch_lora_weights.safetensors",
                        token=hf_token_global
                    )
                    st.session_state.example_lora = example_lora_path
                    st.success(f"Example LoRA downloaded to: {example_lora_path}")
                except Exception as e:
                    st.error(f"Download failed: {e}. Please ensure your HF token is correct.")

    with col_inf_2:
        st.subheader("Inference Settings")
        
        prompt_input = st.text_area("Prompt", value="a turtle with a flower", height=100)
        use_local_llm = st.checkbox("✨ Auto-Enhance Prompt with Local AI (100% Offline, runs locally)")
        
        inf_steps = st.slider("Steps", 1, 20, 4 if "schnell" in selected_model.lower() else 20)
        guidance = st.slider("Guidance Scale", 0.0, 10.0, 3.5 if "dev" in selected_model.lower() else 0.0)
        num_images = st.number_input("Number of Images to Generate", min_value=1, max_value=20, value=1)
        
        if st.button("🚀 Generate Image(s)", type="primary"):
            final_prompt = prompt_input
            if use_local_llm:
                with st.spinner("🤖 Local AI is enhancing your prompt (Offline)..."):
                    try:
                        from engine.prompt_llm import enhance_prompt_locally
                        final_prompt = enhance_prompt_locally(prompt_input)
                        st.success("Prompt successfully enhanced locally!")
                        st.info(f"**Enhanced Prompt:** {final_prompt}")
                    except Exception as e:
                        st.error(f"Local LLM failed: {e}")
                        final_prompt = prompt_input
                        
            model_status = get_model_status(selected_model)
            if not hf_token_global and "dev" in selected_model.lower() and "DOWNLOAD REQUIRED" in model_status:
                st.error("Hugging Face Token is required to download FLUX.1-dev.")
            elif hardware_target == "Intel Optimized (OpenVINO iGPU/NPU)":
                st.info(f"Targeting {selected_model} in stable mode. Generating {num_images} image(s)...")
                
                for i in range(num_images):
                    st.write(f"### Image {i+1} of {num_images}")
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        import subprocess
                        import re
                        output_img = f"outputs/generated_output_{i}.png"
                        intel_script = os.path.join(os.path.dirname(__file__), "engine", "intel_inference.py")
                        
                        cmd = [sys.executable, intel_script, "STABLE_HYBRID", 
                               "--prompt", final_prompt, 
                               "--output", output_img,
                               "--model", selected_model,
                               "--steps", str(inf_steps),
                               "--guidance", str(guidance)]
                               
                        if hf_token_global:
                            cmd.extend(["--token", hf_token_global])
                        
                        process = subprocess.Popen(
                            cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1
                        )
                        
                        for line in iter(process.stdout.readline, ''):
                            if not line:
                                break
                            line = line.strip()
                            
                            if "[PROGRESS]" in line:
                                match = re.search(r'\((\d+\.\d+)%\)', line)
                                if match:
                                    pct = float(match.group(1)) / 100.0
                                    progress_bar.progress(min(pct, 1.0))
                                    progress_text.text(f"Denoising (clarifying & removing noise): {line.split('[PROGRESS]')[-1].strip()}")
                            elif "[INIT]" in line:
                                progress_text.text(f"Initialization: {line.split('[INIT]')[-1].strip()}")
                            elif "[MEMORY]" in line:
                                progress_text.text(f"Memory Check: {line.split('[MEMORY]')[-1].strip()}")
                            elif "[STEP 1]" in line or "[STEP 2]" in line:
                                progress_text.text(line.strip())
                                
                        process.stdout.close()
                        process.wait()
                        
                        if process.returncode == 0:
                            progress_bar.progress(1.0)
                            progress_text.text("Generation Complete!")
                            st.image(output_img)
                        else:
                            st.error(f"Generation failed with exit code {process.returncode}. See outputs/inference_debug.log for details.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info(f"Loading {selected_model} and generating {num_images} image(s)...")
                try:
                    current_lora = None
                    if lora_file:
                        with open("temp_lora.safetensors", "wb") as f:
                            f.write(lora_file.getbuffer())
                        current_lora = "temp_lora.safetensors"
                    elif 'example_lora' in st.session_state:
                        current_lora = st.session_state.example_lora
                    
                    with st.spinner("Loading model weights into RAM/VRAM..."):
                        pipe = load_flux_pipeline(selected_model, current_lora, hf_token_global)
                        
                    for i in range(num_images):
                        with st.spinner(f"Generating Image {i+1} of {num_images} (this will block the UI)..."):
                            image = run_inference(pipe, final_prompt, inf_steps, guidance)
                            st.image(image, caption=f"Generated Image {i+1}", use_container_width=True)
                except Exception as e:
                    st.error(f"Inference failed: {e}")

# --- Tab 4: Distributed Batch Processing ---
with tab4:
    st.header("Step 4: Distributed / Batch Processing")
    st.info("Process multiple prompts concurrently by distributing the workload across available network nodes.")
    
    col_batch_1, col_batch_2 = st.columns([1, 1])
    
    with col_batch_1:
        st.subheader("Node Configuration")
        nodes_input = st.text_area("Worker Nodes (URLs, comma-separated)", value="http://localhost:8000")
        st.caption("Ensure the API server (`python src/api.py`) is running on each node.")
        
        batch_model_options = {
            "FLUX.1 Schnell": "OpenVINO/FLUX.1-schnell-int4-ov",
            "FLUX.1 Dev": "OpenVINO/FLUX.1-dev-int4-ov"
        }
        batch_model_display = st.selectbox("Model to Use Across Nodes", list(batch_model_options.keys()), index=0)
        batch_model_id = batch_model_options[batch_model_display]
        batch_hardware = st.radio("Hardware Target (For all nodes)", ["STABLE_HYBRID", "CPU", "GPU"], index=0)
        batch_steps = st.slider("Batch Steps", 1, 20, 4 if "schnell" in batch_model_id.lower() else 20)
        batch_guidance = st.slider("Batch Guidance Scale", 0.0, 10.0, 3.5 if "dev" in batch_model_id.lower() else 0.0)
        
    with col_batch_2:
        st.subheader("Batch Prompts")
        batch_prompts_input = st.text_area("Prompts (one per line)", value="A futuristic city in the clouds, 8k\nA cyberpunk street market at night, cinematic\nA peaceful forest with glowing mushrooms")
        
        if st.button("🌐 Generate Batch", type="primary"):
            prompts = [p.strip() for p in batch_prompts_input.split('\n') if p.strip()]
            nodes = [n.strip() for n in nodes_input.split(',') if n.strip()]
            
            if not prompts or not nodes:
                st.error("Please provide at least one prompt and one node URL.")
            else:
                try:
                    from engine.distributed_orchestrator import run_distributed_batch
                    import io
                    import queue
                    import threading
                    import re
                    import time
                    
                    st.info(f"Distributing {len(prompts)} tasks across {len(nodes)} node(s)...")
                    
                    # Create UI elements for each prompt
                    prompt_ui = {}
                    for p in prompts:
                        c = st.container()
                        c.markdown(f"**Prompt:** {p}")
                        prompt_ui[p] = {
                            "bar": c.progress(0),
                            "status": c.empty(),
                            "img": c.empty()
                        }
                    
                    update_queue = queue.Queue()
                    
                    def status_cb(prompt, log_line):
                        update_queue.put(("status", prompt, log_line))
                        
                    def result_cb(prompt, img_bytes):
                        update_queue.put(("done", prompt, img_bytes))

                    def bg_task():
                        run_distributed_batch(prompts, nodes, batch_model_id, batch_hardware, hf_token_global, batch_steps, batch_guidance, result_callback=result_cb, status_callback=status_cb)
                        
                    thread = threading.Thread(target=bg_task)
                    # Add Streamlit context to the thread if needed, but queue approach makes it safe
                    # from streamlit.runtime.scriptrunner import add_script_run_ctx
                    # add_script_run_ctx(thread)
                    thread.start()
                    
                    completed = 0
                    while completed < len(prompts):
                        while not update_queue.empty():
                            msg_type, prompt, data = update_queue.get()
                            ui = prompt_ui[prompt]
                            
                            if msg_type == "status":
                                line = data
                                if "[PROGRESS]" in line:
                                    match = re.search(r'\((\d+\.\d+)%\)', line)
                                    if match:
                                        pct = float(match.group(1)) / 100.0
                                        ui["bar"].progress(min(pct, 1.0))
                                        ui["status"].text(f"Denoising (clarifying & removing noise): {line.split('[PROGRESS]')[-1].strip()}")
                                elif "[INIT]" in line:
                                    ui["status"].text(f"Initialization: {line.split('[INIT]')[-1].strip()}")
                                elif "[MEMORY]" in line:
                                    ui["status"].text(f"Memory Check: {line.split('[MEMORY]')[-1].strip()}")
                                elif "[ERROR]" in line:
                                    ui["status"].error(line)
                            elif msg_type == "done":
                                completed += 1
                                ui["bar"].progress(1.0)
                                if data:
                                    ui["status"].success("Complete!")
                                    ui["img"].image(io.BytesIO(data), use_container_width=True)
                                else:
                                    ui["status"].error("Failed to generate image.")
                        time.sleep(0.1) # Prevent high CPU usage while waiting
                        
                    thread.join()
                    st.success("All batch tasks complete!")
                        
                except Exception as e:
                    st.error(f"Distributed batch failed: {e}")

    with st.expander("📖 How Distributed Processing Works"):
        st.markdown("""
        **Overview:**
        This feature turns your local application into an Orchestrator. It takes your list of prompts and dispatches them as concurrent HTTP requests to the Worker Nodes you specify.
        
        **Setup a Worker Node:**
        1. On any machine (or your local machine), start the API server:
           `python src/api.py`
        2. Note the IP address and port (e.g., `http://192.168.1.100:8000`).
        
        **Execution:**
        The orchestrator uses a ThreadPool to dispatch tasks. It limits concurrent requests to match the number of active nodes, ensuring each 16GB node only processes one heavy Flux model at a time to prevent Out-Of-Memory (OOM) crashes.
        """)
