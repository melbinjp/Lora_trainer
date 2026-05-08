import requests
import concurrent.futures
import time
import logging
import json
import base64

def check_node_health(url):
    """Checks if a node is alive and returns its status."""
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def process_prompt_on_node(node_url, prompt, model, hardware_target, token=None, steps=None, guidance=None, status_callback=None):
    """Sends a generation stream request to a specific node."""
    endpoint = f"{node_url.rstrip('/')}/generate_stream"
    payload = {
        "prompt": prompt,
        "model": model,
        "hardware_target": hardware_target
    }
    if token:
        payload["token"] = token
    if steps is not None:
        payload["steps"] = steps
    if guidance is not None:
        payload["guidance"] = guidance
        
    try:
        # Long timeout because image generation takes time (e.g., 5-10 minutes)
        response = requests.post(endpoint, json=payload, stream=True, timeout=600)
        if response.status_code != 200:
            logging.error(f"Node {node_url} failed with status {response.status_code}: {response.text}")
            return None
            
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if data["type"] == "log":
                        if status_callback:
                            status_callback(prompt, data["data"])
                    elif data["type"] == "result":
                        return base64.b64decode(data["data"]) # Returns image bytes
                    elif data["type"] == "error":
                        logging.error(f"Error from node {node_url}: {data['data']}")
                        if status_callback:
                            status_callback(prompt, f"[ERROR] {data['data']}")
                        return None
                except json.JSONDecodeError:
                    pass
    except requests.exceptions.RequestException as e:
        logging.error(f"Node {node_url} connection error: {e}")
        return None
    return None

def run_distributed_batch(prompts, nodes, model, hardware_target, token=None, steps=None, guidance=None, result_callback=None, status_callback=None):
    """
    Orchestrates a batch of prompts across multiple nodes.
    Calls callbacks as tasks progress and complete.
    """
    # Verify nodes
    active_nodes = []
    for node in nodes:
        health = check_node_health(node)
        if health:
            active_nodes.append(node)
            logging.info(f"Node {node} is active. Busy: {health.get('busy')}")
        else:
            logging.warning(f"Node {node} is unresponsive and will be skipped.")

    if not active_nodes:
        raise ValueError("No active nodes available for distributed processing.")

    results = []

    # We use a ThreadPoolExecutor to handle concurrent requests to the nodes.
    # The max_workers is set to the number of active nodes so we don't overload them,
    # as each node can only process one image at a time efficiently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_nodes)) as executor:
        # A dictionary to map the future to the prompt it's processing
        future_to_prompt = {}
        
        # Simple round-robin dispatch
        for i, prompt in enumerate(prompts):
            node = active_nodes[i % len(active_nodes)]
            future = executor.submit(process_prompt_on_node, node, prompt, model, hardware_target, token, steps, guidance, status_callback)
            future_to_prompt[future] = prompt
            
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                image_bytes = future.result()
                if result_callback:
                    result_callback(prompt, image_bytes)
                
                if image_bytes:
                    results.append((prompt, image_bytes))
            except Exception as exc:
                logging.error(f"{prompt} generated an exception: {exc}")
                if status_callback:
                    status_callback(prompt, f"[ERROR] Exception: {exc}")
                if result_callback:
                    result_callback(prompt, None)
                    
    return results
