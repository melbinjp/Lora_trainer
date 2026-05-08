import torch
import logging
import gc
import sys
from threading import Thread

def enhance_prompt_locally_stream(user_prompt):
    """
    Loads a tiny local LLM, yields the enhanced prompt tokens in real-time, 
    and completely unloads it to ensure RAM/VRAM is 100% free for the main Flux generation.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        
        logging.info(f"Loading local LLM ({model_id}) for prompt streaming...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", 
            dtype=torch.bfloat16
        )
        
        sys_prompt = (
            "You are an expert AI image prompt engineer. Expand the user's idea into a highly detailed comprehensive cinematic prompt.\n"
            "CRITICAL RULES:\n"
            "1. You MUST enclose the main subject in {curly brackets}.\n"
            "2. You MUST enclose the background/setting in [square brackets].\n"
            "3. You MUST enclose lighting and camera details in (parentheses).\n"
            "4. Reply ONLY with the exact prompt text. No conversational filler.\n\n"
            "Example Input: a cat\n"
            "Example Output: {A majestic fluffy cat with glowing eyes}, [sitting on a neon-lit cyberpunk rooftop], (cinematic lighting, 8k resolution, macro photography), ultra detailed."
        )
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Enhance this idea: {user_prompt}"}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=400, # Increased from 150 to prevent cut-off
            do_sample=True,
            temperature=0.7
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            if new_text:
                yield new_text
            
        thread.join()
        
        # Aggressive memory cleanup before returning
        logging.info("Unloading local LLM and freeing memory...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logging.error(f"Local LLM streaming failed: {e}")
        yield user_prompt + ", highly detailed, cinematic lighting, 8k resolution"

def enhance_prompt_locally(user_prompt):
    """
    Non-streaming version of the prompt enhancer.
    """
    full_prompt = ""
    for chunk in enhance_prompt_locally_stream(user_prompt):
        full_prompt += chunk
    return full_prompt

