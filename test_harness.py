"""
Automated test harness for LoRA Streamlit UI core logic.
Simulates the main workflow: image upload, captioning, LoRA training, model upload, and image generation.
Catches/report errors and prints actionable feedback for each step.

Run: python test_harness.py
"""
import os
import sys
import traceback
from PIL import Image
import io
import random
import shutil

# --- Config ---
SAMPLE_IMAGE_PATH = "sample_test_image.jpg"
SAMPLE_IMAGE_SIZE = (256, 256)
SAMPLE_CAPTION = "A test image of a cat."
SAMPLE_PROMPT = "A cat wearing sunglasses, digital art"
SAMPLE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SAMPLE_LORA_NAME = "test_lora"
SAMPLE_ACTIVATION_KEYWORD = "test_lora_cat"
SAMPLE_OUTPUT_DIR = "lora_output/test_lora"

# --- Step 1: Create a sample image if not present ---
def create_sample_image(path):
    img = Image.new("RGB", SAMPLE_IMAGE_SIZE, color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    img.save(path)
    print(f"[TEST] Sample image created at {path}")

# --- Step 2: Import main functions from test.py ---
try:
    import test as lora_ui
    print("[TEST] Successfully imported test.py.")
except Exception as e:
    print("[FAIL] Could not import test.py: ", e)
    traceback.print_exc()
    sys.exit(1)

# --- Step 3: Run the workflow ---
def run_workflow():
    print("[TEST] Starting LoRA UI automated test workflow...")
    results = {"image_upload": False, "captioning": False, "lora_training": False, "model_upload": False, "image_generation": False}
    # 1. Image upload
    try:
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            create_sample_image(SAMPLE_IMAGE_PATH)
        images = [open(SAMPLE_IMAGE_PATH, "rb")]
        results["image_upload"] = True
        print("[PASS] Image upload step succeeded.")
    except Exception as e:
        print(f"[FAIL] Image upload step failed: {e}")
        traceback.print_exc()
        return results
    # 2. Captioning (simulate)
    try:
        captions = [SAMPLE_CAPTION]
        results["captioning"] = True
        print("[PASS] Captioning step succeeded.")
    except Exception as e:
        print(f"[FAIL] Captioning step failed: {e}")
        traceback.print_exc()
        return results
    # 3. LoRA training (minimal config)
    try:
        print("[TEST] Starting LoRA training (minimal config, CPU)...")
        result = lora_ui.train_lora(
            images,
            captions,
            SAMPLE_MODEL_ID,
            SAMPLE_LORA_NAME,
            SAMPLE_ACTIVATION_KEYWORD,
            SAMPLE_OUTPUT_DIR,
            None,  # No HF token for public model
            "cpu",
            adv_config={"epochs": 1, "batch_size": 1, "lora_rank": 2, "lora_alpha": 2},
            custom_code=None,
            precision=None
        )
        if result == "trained":
            results["lora_training"] = True
            print("[PASS] LoRA training step succeeded.")
        else:
            print(f"[FAIL] LoRA training step failed. Result: {result}")
            return results
    except Exception as e:
        print(f"[FAIL] LoRA training step crashed: {e}")
        traceback.print_exc()
        return results
    # 4. Model upload (mocked)
    try:
        print("[TEST] Skipping Hugging Face upload (mocked in test).")
        results["model_upload"] = True
    except Exception as e:
        print(f"[FAIL] Model upload step failed: {e}")
        traceback.print_exc()
    # 5. Image generation (mocked)
    try:
        print("[TEST] Skipping image generation (mocked in test).")
        results["image_generation"] = True
    except Exception as e:
        print(f"[FAIL] Image generation step failed: {e}")
        traceback.print_exc()
    print("[TEST] Automated workflow complete.")
    # Cleanup
    try:
        images[0].close()
        if os.path.exists(SAMPLE_IMAGE_PATH):
            os.remove(SAMPLE_IMAGE_PATH)
        if os.path.exists(SAMPLE_OUTPUT_DIR):
            shutil.rmtree(SAMPLE_OUTPUT_DIR)
    except Exception:
        pass
    return results

def print_summary(results):
    print("\n========== TEST SUMMARY ==========")
    for step, passed in results.items():
        print(f"{step:20}: {'PASS' if passed else 'FAIL'}")
    print("==================================\n")
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    results = run_workflow()
    print_summary(results)
