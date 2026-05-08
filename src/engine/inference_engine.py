import torch
from diffusers import FluxPipeline
import os
import sys
import logging

# Add src to path for relative imports if run as script
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.helpers import setup_logging, log_info, log_error

def load_flux_pipeline(model_id="black-forest-labs/FLUX.1-schnell", lora_path=None, hf_token=None):
    """
    Loads the Flux pipeline with CPU offloading for 16GB RAM support.
    """
    setup_logging()
    log_info(f"Loading standard pipeline for model: {model_id}...")
    
    try:
        # Use bfloat16 for efficiency on modern CPUs/GPUs
        dtype = torch.bfloat16
        
        pipe = FluxPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            token=hf_token,
            low_cpu_mem_usage=True
        )
        
        # Critical for 16GB RAM: Offload model parts to CPU when not in use
        log_info("Enabling model CPU offload for 16GB RAM optimization.")
        pipe.enable_model_cpu_offload()
        
        if lora_path and os.path.exists(lora_path):
            log_info(f"Loading LoRA weights from {lora_path}...")
            pipe.load_lora_weights(lora_path)
        
        return pipe
    except Exception as e:
        log_error(f"Failed to load Flux pipeline: {e}")
        raise

def run_inference(pipe, prompt, steps=4, guidance_scale=3.5, width=512, height=512):
    """
    Runs inference and returns the PIL image.
    """
    log_info(f"Running standard inference for prompt: {prompt}")
    try:
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        return image
    except Exception as e:
        log_error(f"Standard inference failed: {e}")
        raise
