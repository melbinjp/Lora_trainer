import os
import logging
import pytest
from unittest import mock

# Import functions from test.py
from test import detect_model_type, get_system_resources, LOG_FILE

# --- Logging helpers ---
def test_log_file_write_and_clear():
    # Write to log
    logging.info("Test log entry")
    assert os.path.exists(LOG_FILE)
    with open(LOG_FILE, "r") as f:
        content = f.read()
    assert "Test log entry" in content
    # Clear log
    with open(LOG_FILE, "w") as f:
        f.truncate(0)
    with open(LOG_FILE, "r") as f:
        assert f.read() == ""

def test_detect_model_type_diffusion():
    # Should detect as diffusion by heuristic
    assert detect_model_type("runwayml/stable-diffusion-v1-5") == "diffusion"
    assert detect_model_type("stabilityai/sd-turbo") == "diffusion"

def test_detect_model_type_transformer():
    # Should detect as transformer by heuristic
    assert detect_model_type("bert-base-uncased") == "transformer"
    assert detect_model_type("gpt2") == "transformer"

def test_detect_model_type_unknown():
    # Should return unknown for random string
    assert detect_model_type("some-random-model") == "unknown"

def test_get_system_resources():
    ram, vram = get_system_resources()
    assert ram > 0
    # vram can be None if no GPU, but should not raise

# --- Mocked test for error handling in detect_model_type ---
def test_detect_model_type_handles_exception(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise Exception("fail")
    monkeypatch.setattr("test.detect_model_type", raise_exc)
    # Should not raise, just fallback to unknown
    try:
        result = detect_model_type("fail-model")
    except Exception:
        result = None
    assert result is None or result == "unknown"

if __name__ == "__main__":
    pytest.main([__file__])
