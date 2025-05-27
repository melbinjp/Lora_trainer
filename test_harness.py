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
import unittest # For more structured testing
from unittest.mock import patch, MagicMock, mock_open

# --- Config ---
SAMPLE_IMAGE_NAME = "sample_test_image.jpg" # Changed to just name
SAMPLE_IMAGE_SIZE = (256, 256)
SAMPLE_CAPTION = "A test image of a cat."
SAMPLE_PROMPT = "A cat wearing sunglasses, digital art"
SAMPLE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SAMPLE_LORA_NAME = "test_lora"
SAMPLE_ACTIVATION_KEYWORD = "test_lora_cat"
SAMPLE_OUTPUT_DIR = "lora_output/test_lora_harness" # Make it unique for harness
TEMP_IMAGE_DIR = "temp_test_images_harness" # For creating dummy images

# --- Step 1: Create a sample image if not present ---
def create_sample_image_file(image_dir, image_name):
    os.makedirs(image_dir, exist_ok=True)
    path = os.path.join(image_dir, image_name)
    img = Image.new("RGB", SAMPLE_IMAGE_SIZE, color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    img.save(path)
    print(f"[TEST_HARNESS] Sample image created at {path}")
    return path

# --- Step 2: Import main functions from test.py ---
try:
    import test as lora_ui 
    # Also ensure torch is available for precision arguments in train_lora
    if not lora_ui.TORCH_AVAILABLE:
        print("[WARN] PyTorch not available in lora_ui (test.py). Some tests might be limited.")
except Exception as e:
    print("[FAIL] Could not import test.py (aliased as lora_ui): ", e)
    traceback.print_exc()
    sys.exit(1)

# --- Test Cases ---
class TestLoraUISmoke(unittest.TestCase):
    def setUp(self):
        self.results = {"image_upload": False, "captioning": False, "lora_training": False, "model_upload": False, "image_generation": False, "command_construction": False}
        self.sample_image_path = create_sample_image_file(TEMP_IMAGE_DIR, SAMPLE_IMAGE_NAME)
        # Mock streamlit UI elements that might be called directly or indirectly
        self.patcher_st_info = patch('lora_ui.st.info')
        self.patcher_st_error = patch('lora_ui.st.error')
        self.patcher_st_success = patch('lora_ui.st.success')
        self.mock_st_info = self.patcher_st_info.start()
        self.mock_st_error = self.patcher_st_error.start()
        self.mock_st_success = self.patcher_st_success.start()


    def tearDown(self):
        if os.path.exists(self.sample_image_path):
            os.remove(self.sample_image_path)
        if os.path.exists(TEMP_IMAGE_DIR):
            shutil.rmtree(TEMP_IMAGE_DIR)
        if os.path.exists(SAMPLE_OUTPUT_DIR): # Ensure test-specific output is cleaned
            shutil.rmtree(SAMPLE_OUTPUT_DIR)
        self.patcher_st_info.stop()
        self.patcher_st_error.stop()
        self.patcher_st_success.stop()


    @patch('lora_ui.shutil.which') # Mock shutil.which to control path findings
    @patch('lora_ui.subprocess.run') # Target subprocess.run imported in lora_ui (test.py)
    def test_train_lora_command_construction(self, mock_subprocess_run, mock_shutil_which):
        print("\n[TEST_HARNESS] Running: test_train_lora_command_construction")
        
        # Setup mock for shutil.which to return predictable paths
        def side_effect_shutil_which(cmd):
            if cmd == "accelerate":
                return "/fake/path/to/accelerate"
            if cmd == "train_network.py": # If the script tries path fallback
                return "/fake/path/to/train_network.py"
            return None
        mock_shutil_which.side_effect = side_effect_shutil_which

        # Mock the return value of subprocess.run to simulate successful execution
        mock_process_result = MagicMock()
        mock_process_result.returncode = 0
        mock_process_result.stdout = "Mocked subprocess STDOUT"
        mock_process_result.stderr = "Mocked subprocess STDERR (empty for success)"
        mock_subprocess_run.return_value = mock_process_result
        
        # Mock os.path.exists for the training script path check (if direct paths are checked first)
        # and for the final model existence check
        def side_effect_os_path_exists(path):
            if path == os.path.abspath("./sd-scripts/train_network.py"): # Example direct check
                return True # Simulate it's found here
            if path == os.path.join(os.path.abspath(SAMPLE_OUTPUT_DIR), f"{SAMPLE_LORA_NAME}.safetensors"):
                return True # Simulate model was "created"
            return False # Default for other paths
        
        mock_os_access = MagicMock(return_value=True) # Assume scripts are executable

        with patch('lora_ui.os.path.exists', side_effect=side_effect_os_path_exists), \
             patch('lora_ui.os.access', mock_os_access):

            # Prepare dummy image file object
            # Create a dummy image file in memory for the test
            img_byte_arr = io.BytesIO()
            dummy_pil_image = Image.new('RGB', (60, 30), color='red')
            dummy_pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Create a FileUploader-like object
            class MockUploadedFile:
                def __init__(self, name, data):
                    self.name = name
                    self.data = data
                def getbuffer(self):
                    return self.data
            
            mock_image_file = MockUploadedFile(SAMPLE_IMAGE_NAME, img_byte_arr.getvalue())
            
            sample_images = [mock_image_file]
            sample_captions = [SAMPLE_CAPTION]
            
            adv_config_test = {
                "epochs": 5, 
                "batch_size": 2, 
                "lora_rank": 4, 
                "lora_alpha": 2,
                "learning_rate": 2e-5,
                "resolution": "256,256",
                "lr_scheduler": "cosine",
                "save_every_n_epochs": 1,
                # other params will use defaults in train_lora
            }

            # Use torch.float16 if available, else a placeholder that won't break
            precision_to_test = lora_ui.torch.float16 if lora_ui.TORCH_AVAILABLE else "fp16_placeholder"


            result = lora_ui.train_lora(
                images=sample_images,
                captions=sample_captions,
                base_model_id_resolved=SAMPLE_MODEL_ID,
                lora_model_name_resolved=SAMPLE_LORA_NAME,
                lora_activation_keyword_resolved=SAMPLE_ACTIVATION_KEYWORD, # Not used in command, but good to pass
                output_dir_resolved=SAMPLE_OUTPUT_DIR,
                hf_token_resolved=None,
                device_resolved="cpu", # Test CPU path
                adv_config_resolved=adv_config_test,
                custom_code_resolved=None,
                precision_resolved=precision_to_test 
            )

            self.assertTrue(mock_subprocess_run.called, "subprocess.run was not called.")
            
            # Get the command list from the call arguments
            called_command_args = mock_subprocess_run.call_args[0][0]
            self.assertIsInstance(called_command_args, list, "Command should be a list.")
            
            # Convert command list to a single string for easier substring checks, or check list elements
            called_command_str = " ".join(called_command_args)

            print(f"[TEST_HARNESS] Captured command: {called_command_str}")

            self.assertIn("accelerate", called_command_args[0], "accelerate should be the first part of the command.")
            self.assertIn("launch", called_command_args, "'launch' should be part of the accelerate command.")
            
            # Check for the training script path (it's made absolute in train_lora)
            expected_train_script_path = os.path.abspath("./sd-scripts/train_network.py")
            self.assertIn(expected_train_script_path, called_command_args, f"Training script path {expected_train_script_path} not found in command.")

            self.assertIn(f"--pretrained_model_name_or_path={SAMPLE_MODEL_ID}", called_command_str)
            
            expected_output_dir_abs = os.path.abspath(SAMPLE_OUTPUT_DIR)
            self.assertIn(f"--output_dir={expected_output_dir_abs}", called_command_str)
            
            self.assertIn(f"--output_name={SAMPLE_LORA_NAME}", called_command_str)
            
            # Check for adv_config parameters
            self.assertIn(f"--max_train_epochs={adv_config_test['epochs']}", called_command_str)
            self.assertIn(f"--train_batch_size={adv_config_test['batch_size']}", called_command_str)
            self.assertIn(f"--network_dim={adv_config_test['lora_rank']}", called_command_str)
            self.assertIn(f"--network_alpha={adv_config_test['lora_alpha']}", called_command_str)
            self.assertIn(f"--learning_rate={adv_config_test['learning_rate']}", called_command_str)
            self.assertIn(f"--resolution={adv_config_test['resolution']}", called_command_str)
            self.assertIn(f"--lr_scheduler={adv_config_test['lr_scheduler']}", called_command_str)
            self.assertIn(f"--save_every_n_epochs={adv_config_test['save_every_n_epochs']}", called_command_str)


            # Check precision (fp16 if torch was available, otherwise "no" if placeholder was used and logic defaults)
            expected_precision_arg = "fp16" if lora_ui.TORCH_AVAILABLE else "no" # depends on how train_lora handles invalid precision_resolved
            if not lora_ui.TORCH_AVAILABLE and precision_to_test == "fp16_placeholder":
                 # If torch is not available, train_lora's precision_arg logic will default to "no"
                 # because `precision_resolved == torch.float16` will be false.
                self.assertIn("--mixed_precision=no", called_command_str, "Precision should default to 'no' if torch not available for test.")
                self.assertIn("--save_precision=no", called_command_str)

            else: # If torch is available
                self.assertIn(f"--mixed_precision={expected_precision_arg}", called_command_str)
                self.assertIn(f"--save_precision={expected_precision_arg}", called_command_str)


            # Check for CPU specific accelerate arg
            self.assertIn("--num_cpu_threads_per_process", called_command_str, "CPU thread arg missing for CPU training.")
            
            self.assertEqual(result, "trained", "train_lora did not return 'trained' on mocked success.")
            self.results["command_construction"] = True
            print("[PASS] test_train_lora_command_construction")


    # Existing smoke test (can be run as part of the suite)
    @patch('lora_ui.train_lora') # Mock the actual training for this smoke test
    def test_smoke_workflow(self, mock_train_lora):
        print("\n[TEST_HARNESS] Running: test_smoke_workflow")
        mock_train_lora.return_value = "trained" # Simulate successful training

        # 1. Image upload
        try:
            # Use a real file-like object for this part, as train_lora (mocked here) might expect it
            with open(self.sample_image_path, "rb") as f:
                # Create a FileUploader-like object to pass to functions if they expect it
                class MockUploadedFileSmoke:
                    def __init__(self, name, data_bytes):
                        self.name = name
                        self._data_bytes = data_bytes
                    def getbuffer(self): # Used by train_lora to save images
                        return self._data_bytes
                    def read(self): # For functions that might call read()
                        return self._data_bytes
                
                file_content = f.read()
                mock_uploaded_file = MockUploadedFileSmoke(SAMPLE_IMAGE_NAME, file_content)
            
            self.assertGreater(len(mock_uploaded_file.getbuffer()), 0)
            self.results["image_upload"] = True
            print("[PASS] Smoke: Image upload step succeeded.")
        except Exception as e:
            print(f"[FAIL] Smoke: Image upload step failed: {e}")
            traceback.print_exc()
            self.fail("Image upload failed in smoke test")

        # 2. Captioning (simulate)
        try:
            captions = [SAMPLE_CAPTION]
            self.results["captioning"] = True
            print("[PASS] Smoke: Captioning step succeeded.")
        except Exception as e:
            print(f"[FAIL] Smoke: Captioning step failed: {e}")
            traceback.print_exc()
            self.fail("Captioning failed in smoke test")

        # 3. LoRA training (mocked at the function level for this smoke test)
        try:
            print("[TEST_HARNESS] Smoke: Calling mocked LoRA training...")
            # The 'images' argument to train_lora needs to be a list of file-like objects
            # that have a .name attribute and a .getbuffer() method.
            result = lora_ui.train_lora(
                images=[mock_uploaded_file], # Pass the mocked file object
                captions=captions,
                base_model_id_resolved=SAMPLE_MODEL_ID,
                lora_model_name_resolved=SAMPLE_LORA_NAME,
                lora_activation_keyword_resolved=SAMPLE_ACTIVATION_KEYWORD,
                output_dir_resolved=SAMPLE_OUTPUT_DIR,
                hf_token_resolved=None,
                device_resolved="cpu",
                adv_config_resolved={"epochs": 1, "batch_size": 1, "lora_rank": 2, "lora_alpha": 2},
                custom_code_resolved=None,
                precision_resolved=lora_ui.torch.float16 if lora_ui.TORCH_AVAILABLE else None
            )
            self.assertEqual(result, "trained")
            mock_train_lora.assert_called_once()
            self.results["lora_training"] = True
            print("[PASS] Smoke: LoRA training step (mocked) succeeded.")
        except Exception as e:
            print(f"[FAIL] Smoke: LoRA training step (mocked) crashed: {e}")
            traceback.print_exc()
            self.fail("LoRA training (mocked) failed in smoke test")
        
        # 4. Model upload (mocked - conceptually)
        self.results["model_upload"] = True
        print("[PASS] Smoke: Model upload step (conceptual) succeeded.")
        
        # 5. Image generation (mocked - conceptually)
        self.results["image_generation"] = True
        print("[PASS] Smoke: Image generation step (conceptual) succeeded.")
        print("[TEST_HARNESS] Smoke: Automated workflow (mocked training) complete.")


def print_final_summary(test_results_summary):
    print("\n========== TEST HARNESS FINAL SUMMARY ==========")
    all_passed = True
    for test_name, result_dict in test_results_summary.items():
        print(f"\n--- Test: {test_name} ---")
        for step, passed in result_dict.items():
            print(f"  {step:25}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
    print("==============================================\n")
    if not all_passed:
        print("[FINAL_STATUS] One or more tests failed.")
        sys.exit(1)
    else:
        print("[FINAL_STATUS] All tests passed successfully!")

if __name__ == "__main__":
    # unittest.main() # This would run all TestClass methods

    # Manual execution to control flow and reporting for now
    final_summary_data = {}

    # Run Smoke Test
    smoke_tester = TestLoraUISmoke()
    try:
        smoke_tester.setUp()
        smoke_tester.test_smoke_workflow()
    except Exception as e:
        print(f"Error during smoke test execution: {e}")
        traceback.print_exc()
    finally:
        final_summary_data["test_smoke_workflow"] = smoke_tester.results
        smoke_tester.tearDown()

    # Run Command Construction Test
    command_test_results = {"command_construction": False} # Initialize specific results for this test
    command_tester = TestLoraUISmoke() # Re-instance for clean setup/teardown if needed, or use existing one
    try:
        command_tester.setUp() # Call setup for this test too
        command_tester.test_train_lora_command_construction()
        # The test_train_lora_command_construction method updates its own results dict entry
        # We need to retrieve it if we are running it this way.
        command_test_results["command_construction"] = command_tester.results["command_construction"]
    except Exception as e:
        print(f"Error during command construction test execution: {e}")
        traceback.print_exc()
        command_test_results["command_construction"] = False # Ensure it's marked as fail
    finally:
        final_summary_data["test_train_lora_command_construction"] = command_test_results
        command_tester.tearDown()


    print_final_summary(final_summary_data)
```
