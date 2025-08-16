# app.py
# Main Flask web app for CharMind_LLM.
# Provides a professional UI for chat, RLHF data labeling, and model management.

import os
import json
import logging
import sys

# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

from threading import Lock
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, Response

import subprocess
from core.utils.config import (
    CONTEXT_LENGTH, DEVICE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    TOKENIZER_PATH, PRETRAIN_CHECKPOINT_DIR, FINETUNE_CHECKPOINT_DIR,
    RLHF_CHECKPOINT_DIR, EOS_TOKEN,
    PROCESSED_PREFERENCE_FILE, LOGS_DIR, PRETRAIN_LOGS_DIR, FINETUNE_LOGS_DIR, RLHF_LOGS_DIR, REWARD_MODEL_DIR, TOKENIZER_PATH, BASE_MODEL_CHECKPOINT, SAVED_MODELS_GLOBAL_DIR
)
from core.utils.model import BPETransformer
from core.utils.model_io import load_model as load_model_from_dir
from tokenizer.tokenizer import load_tokenizer
from core.utils.memory import MemoryManager
from core.utils.utils import prepare_model_input
import re # Add this import at the top of the file if not already present

# --- App Initialization ---
app = Flask(__name__, template_folder='ui', static_folder='ui')
logging.basicConfig(level=logging.INFO)

# --- Globals ---
model_lock = Lock()
training_processes_lock = Lock()
tokenizer = load_tokenizer(TOKENIZER_PATH)
model: Optional[BPETransformer] = None
active_model_dir: Optional[str] = None
memory_manager: Optional[MemoryManager] = None
# Dictionary to keep track of running training subprocesses
training_processes: Dict[str, Dict[str, Any]] = {} # Changed to Dict[str, Dict[str, Any]]

# --- Model and System Management ---

def get_managed_models() -> List[Dict[str, Any]]:
    """
    Retrieves a list of managed models (latest 2 + best) for each training type.
    Expected structure: { "id": full_path, "name": display_name, "type": model_type }
    """
    managed_models = []
    model_type_dirs = {
        "Pre-trained": PRETRAIN_CHECKPOINT_DIR,
        "Fine-tuned": FINETUNE_CHECKPOINT_DIR,
        "RLHF": RLHF_CHECKPOINT_DIR,
        "Reward Model": REWARD_MODEL_DIR # Added for reward model visibility
    }

    for model_type, dir_path in model_type_dirs.items():
        if not os.path.exists(dir_path):
            continue

        all_run_names = set()
        # First, find all unique run names in this directory
        for d_name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, d_name)
            if os.path.isdir(full_path):
                # Example: "run_12345_epoch_10" or "rlhf_12345_ppo_epoch_5"
                if '_epoch_' in d_name:
                    parts = d_name.split('_epoch_')
                elif '_ppo_epoch_' in d_name:
                    parts = d_name.split('_ppo_epoch_')
                else:
                    parts = [d_name]
                run_name_part = parts[0]
                if run_name_part: # Ensure it's not empty
                    all_run_names.add(run_name_part)

        for run_name in all_run_names:
            run_checkpoints = []
            best_model_for_run_path = None
            best_metric_for_run = float('-inf') if model_type == "RLHF" else float('inf')

            # Collect all checkpoints for this run_name and find the best one
            for d_name in os.listdir(dir_path):
                full_path = os.path.join(dir_path, d_name)
                if os.path.isdir(full_path) and d_name.startswith(run_name) and ('_epoch_' in d_name or '_ppo_epoch_' in d_name):
                    try:
                        # Extract epoch number for sorting
                        if '_epoch_' in d_name:
                            epoch_num_str = d_name.split('_epoch_')[1]
                        elif '_ppo_epoch_' in d_name:
                            epoch_num_str = d_name.split('_ppo_epoch_')[1]
                        else:
                            continue
                        epoch_num = int(epoch_num_str)
                        run_checkpoints.append((epoch_num, full_path))

                        # Check for best_model_info.json
                        best_info_path = os.path.join(full_path, 'best_model_info.json')
                        if os.path.exists(best_info_path):
                            with open(best_info_path, 'r') as f:
                                best_info = json.load(f)
                            if best_info.get('is_best'):
                                # Use appropriate metric for comparison
                                current_metric = best_info.get('val_loss') or best_info.get('reward_score')
                                if current_metric is not None:
                                    if (model_type == "RLHF" and current_metric > best_metric_for_run) or \
                                       (model_type != "RLHF" and current_metric < best_metric_for_run):
                                        best_metric_for_run = current_metric
                                        best_model_for_run_path = full_path
                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        app.logger.warning(f"Skipping malformed checkpoint dir {full_path} or info file: {e}")
                        continue

            # Sort by epoch number to get the latest
            run_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Collect paths for the latest 2 and the best model for this run
            current_run_models = set()
            # Add latest 2
            for _, path in run_checkpoints[:2]:
                current_run_models.add(path)
            # Add the best model for this run
            if best_model_for_run_path:
                current_run_models.add(best_model_for_run_path)

            # Add to the main managed_models list
            for model_path in current_run_models:
                display_name = os.path.basename(model_path)
                # Append "(Best)" to the name if it's the best model
                if model_path == best_model_for_run_path:
                    display_name += " (Best)"
                
                # Check if this exact model (id) is already added to avoid duplicates if best == latest
                if not any(m['id'] == model_path for m in managed_models):
                    managed_models.append({
                        'id': model_path,
                        'name': f"[{model_type}] {display_name}",
                        'type': model_type
                    })

    # Sort the final list for consistent display, e.g., by type then name
    managed_models.sort(key=lambda x: (x['type'], x['name']))
    return managed_models

def load_active_model(model_dir: str):
    """
    Loads a model checkpoint and initializes the memory manager.
    This function is thread-safe.
    """
    global model, active_model_dir, memory_manager
    with model_lock:
        app.logger.info(f"Attempting to load model from: {model_dir}")
        try:
            model = load_model_from_dir(model_dir, device=DEVICE)
            model.eval()
            # The active_model_dir is now the full path for clarity
            active_model_dir = model_dir
            memory_manager = MemoryManager(model, tokenizer, DEVICE)
            app.logger.info(f"Successfully loaded model and initialized memory from {active_model_dir}")
        except Exception as e:
            app.logger.error(f"Failed to load model from {model_dir}: {e}", exc_info=True)
            model = None # Ensure model is None if loading fails
            active_model_dir = None
            memory_manager = None
            raise

# --- Initial Load at Startup ---
try:
    managed_models_at_startup = get_managed_models()
    if managed_models_at_startup:
        # Load the first managed model found as the default active model
        default_model_to_load = managed_models_at_startup[0]['id']
        load_active_model(default_model_to_load)
    else:
        app.logger.warning("No managed checkpoints found at startup. Please train or select a model.")
except Exception as e:
    app.logger.critical(f"FATAL: Could not load model at startup. Error: {e}", exc_info=True)

# --- Middleware ---
@app.after_request
def add_cors_headers(response: Response) -> Response:
    """Adds CORS headers to all responses."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# --- API Endpoints ---

@app.route("/")
def index() -> str:
    """Serves the main index.html file."""
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status() -> Response:
    """Returns the current status of the backend."""
    return jsonify({
        "model_loaded": model is not None,
        # Return the full path to be consistent with the available_models endpoint
        "active_model": active_model_dir
    })

@app.route("/api/available_models", methods=["GET"])
def available_models() -> Response:
    """Returns a list of available model checkpoints from all directories."""
    models_list = get_managed_models()
    return jsonify(models_list)

@app.route("/api/set_model", methods=["POST"])
def set_model() -> Response:
    """Sets the active model for the chat session using its full path."""
    data = request.get_json()
    model_full_path = data.get('model_id') # The ID is now the full path
    if not model_full_path:
        return jsonify({'status': 'error', 'message': 'Invalid model ID (full path expected)'}), 400

    # Basic security check: ensure the path is within the checkpoints base directory
    if not os.path.abspath(model_full_path).startswith(os.path.abspath(SAVED_MODELS_GLOBAL_DIR)):
        return jsonify({'status': 'error', 'message': 'Invalid model path'}), 403

    if not os.path.isdir(model_full_path):
        return jsonify({'status': 'error', 'message': 'Model directory not found'}), 404

    try:
        load_active_model(model_full_path)
        display_name = os.path.basename(model_full_path)
        return jsonify({'status': 'ok', 'message': f'Model set to {display_name}'})
    except Exception as e:
        app.logger.error(f"Error setting model to {model_full_path}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat() -> Response:
    """Handles the main chat interaction with streaming response."""
    data = request.get_json()
    history: List[Dict[str, str]] = data.get('history', [])
    
    if not history or history[-1].get('role') != 'user':
        return Response("No valid user input found.", status=400)
    if model is None or memory_manager is None:
        return Response("Model not initialized.", status=500)

    user_input = history[-1]['content']
    
    try:
        system_context = memory_manager.retrieve_context_for_prompt(user_input, history[:-1])

        def generate_stream():
            """Generator for streaming response tokens."""
            generated_ids = []
            prev_text = ""
            
            # Prepare the initial input tensor.
            input_tensor = prepare_model_input(
                prompt=user_input,
                history=history[:-1],
                system_prompt=system_context,
                tokenizer=tokenizer,
                context_length=CONTEXT_LENGTH,
                device=DEVICE
            )
            # This is a placeholder for the actual KV cache from the model
            past_key_values = None

            for _ in range(MAX_NEW_TOKENS):
                with torch.no_grad():
                    logits, _, past_key_values, _ = model(input_tensor, use_cache=True, past_key_values=past_key_values)
                    logits = logits[:, -1, :] / TEMPERATURE
                    
                    if TOP_P is not None and 0.0 < TOP_P < 1.0:
                        # Top-p (nucleus) sampling
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > TOP_P
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)

                if next_token_id.item() == tokenizer.token_to_id(EOS_TOKEN):
                    break

                # Append new token and prepare input for next iteration
                input_tensor = next_token_id

                generated_ids.append(next_token_id.item())
                full_text = tokenizer.decode(generated_ids)
                delta = full_text[len(prev_text):]
                if delta:
                    app.logger.info(f"Generated delta: '{delta}'")
                    yield delta
                    prev_text = full_text

            # After generation, commit the full turn to memory
            if prev_text:
                final_history = history + [{"role": "assistant", "content": prev_text}]
                memory_manager.commit_to_memory(final_history)

        return Response(generate_stream(), mimetype='text/plain')
    except Exception as e:
        app.logger.error(f"Error in /api/chat endpoint: {e}", exc_info=True)
        return Response('Error generating reply.', status=500)

def _generate_full_response(prompt_text: str, temp_override: Optional[float] = None, top_p_override: Optional[float] = None) -> str:
    """Helper to generate a complete, non-streamed response."""
    with model_lock:
        if model is None or memory_manager is None:
            raise RuntimeError("Model or memory manager not initialized.")

        system_context = memory_manager.retrieve_context_for_prompt(prompt_text, [])
        input_tensor = prepare_model_input(
            prompt=prompt_text, tokenizer=tokenizer, context_length=CONTEXT_LENGTH,
            device=DEVICE, system_prompt=system_context
        )

        try:
            output_tensor = model.generate(
                input_tensor,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temp_override or TEMPERATURE,
                top_p=top_p_override or TOP_P,
                eos_token_id=tokenizer.token_to_id(EOS_TOKEN)
            )
            output_ids = output_tensor[0, input_tensor.size(1):].tolist()
            return tokenizer.decode(output_ids).strip()
        except Exception as e:
            app.logger.error(f"Error in _generate_full_response: {e}", exc_info=True)
            return "I'm sorry, I couldn't generate a response."

@app.route("/api/rlhf_generate", methods=["POST"])
def rlhf_generate() -> Response:
    """Generates two distinct responses for RLHF rating."""
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'status': 'error', 'message': 'Prompt is missing'}), 400
    
    try:
        response1 = _generate_full_response(prompt, temp_override=TEMPERATURE)
        response2 = _generate_full_response(prompt, temp_override=TEMPERATURE * 1.5)

        if response1 == response2:
            response2 = _generate_full_response(prompt, temp_override=TEMPERATURE * 2.0, top_p_override=0.95)
        if response1 == response2:
            response2 = "Here's an alternative perspective: " + response1

        return jsonify({'response1': response1, 'response2': response2})
    except Exception as e:
        app.logger.error(f"Error in RLHF generate endpoint: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/api/save_feedback", methods=["POST"])
def save_feedback() -> Response:
    """Saves user preference feedback for RLHF."""
    data = request.get_json()
    if not data or 'prompt' not in data or 'chosen' not in data or 'rejected' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid feedback format'}), 400

    feedback_entry = {
        "prompt": data['prompt'],
        "chosen": data['chosen'],
        "rejected": data['rejected']
    }

    try:
        # Append to the JSONL file
        with open(PROCESSED_PREFERENCE_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        return jsonify({'status': 'ok', 'message': 'Feedback saved successfully'})
    except IOError as e:
        app.logger.error(f"Error saving feedback to {PROCESSED_PREFERENCE_FILE}: {e}")
        return jsonify({'status': 'error', 'message': 'Could not save feedback'}), 500

# --- Training Management Endpoints ---

@app.route("/api/start_training", methods=["POST"])
def start_training():
    """Starts a new training job in the background."""
    data = request.get_json()
    mode = data.get('mode')
    run_name = data.get('run_name')
    base_model_path = data.get('base_model_path') # For RLHF

    if not mode or not run_name:
        return jsonify({"status": "error", "message": "Mode and run_name are required."}), 400

    with training_processes_lock:
        if run_name in training_processes and training_processes[run_name]['process'].poll() is None:
            return jsonify({"status": "error", "message": f"A job with name '{run_name}' is already running."}), 409

    command = [sys.executable] # Use the same python interpreter

    if mode == 'pretrain':
        command.extend(['-m', 'core.pretrain.train', '--mode', mode, '--run_name', run_name])
        log_dir_for_mode = PRETRAIN_LOGS_DIR
    elif mode == 'finetune':
        command.extend(['-m', 'core.pretrain.train', '--mode', mode, '--run_name', run_name])
        log_dir_for_mode = FINETUNE_LOGS_DIR
        if base_model_path: # Pass base_model_path for fine-tuning
            command.extend(['--base_model_path', base_model_path])
    elif mode == 'rlhf':
        if not base_model_path:
            return jsonify({"status": "error", "message": "Base model path is required for RLHF."}), 400
        command.extend(['-m', 'core.rlhf.train_rlhf', '--run_name', run_name, '--base_model_path', base_model_path])
        log_dir_for_mode = RLHF_LOGS_DIR
    else:
        return jsonify({"status": "error", "message": f"Invalid mode: {mode}"}), 400

    try:
        # Use the mode-specific log directory
        log_path = os.path.join(log_dir_for_mode, f"training_{run_name}.log")
        os.makedirs(log_dir_for_mode, exist_ok=True) # Ensure the log directory exists
        log_file = open(log_path, 'w')

        # Start the subprocess
        proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

        with training_processes_lock:
            training_processes[run_name] = {'process': proc, 'mode': mode} # Store mode as well

        app.logger.info(f"Started training job '{run_name}' with command: {' '.join(command)}")
        return jsonify({"status": "ok", "message": f"Training job '{run_name}' started. Log at {log_path}"})

    except Exception as e:
        app.logger.error(f"Failed to start training job '{run_name}': {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/training_status", methods=["GET"])
def training_status():
    """Returns the status of all training jobs, including progress percentage."""
    statuses = {}
    with training_processes_lock:
        for name, proc_info in training_processes.items(): # Changed to proc_info to store more data
            proc = proc_info['process']
            mode_of_run = proc_info['mode']

            status = "Running"
            return_code = proc.poll()
            
            # Initialize progress and completion message, and latest metrics
            progress_percent = 0
            completion_message = ""
            latest_train_loss = None
            latest_val_loss = None
            latest_avg_reward = None

            # Determine the correct log directory based on mode_of_run
            if mode_of_run == 'pretrain':
                current_logs_dir = PRETRAIN_LOGS_DIR
            elif mode_of_run == 'finetune':
                current_logs_dir = FINETUNE_LOGS_DIR
            elif mode_of_run == 'rlhf':
                current_logs_dir = RLHF_LOGS_DIR
            else:
                current_logs_dir = PRETRAIN_LOGS_DIR # Fallback, though should not happen with validation
            
            log_path = os.path.join(current_logs_dir, f"training_{name}.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()

                        # --- Parse Epoch Progress ---
                        # This regex now also attempts to capture batch progress
                        # Example: Epoch 1/10 [pretrain]: Batch 100/200
                        # Example: Epoch Summary: Epoch 1/10, Train Loss: 0.5, Val Loss: 0.6
                        # Example: PPO Epoch Summary: Epoch 1/4, Avg Reward: 0.7
                        
                        # Find the latest epoch/batch progress
                        batch_progress_matches = list(re.finditer(r"Epoch (\d+)/(\d+) \[\w+\]: Batch (\d+)/(\d+)", log_content))
                        epoch_summary_matches = list(re.finditer(r"Epoch Summary: Epoch (\d+)/(\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)", log_content))
                        ppo_epoch_summary_matches = list(re.finditer(r"PPO Epoch Summary: Epoch (\d+)/(\d+), Avg Reward: ([\d.]+)", log_content))

                        current_epoch = 0
                        total_epochs = 0
                        current_batch = 0
                        total_batches = 0

                        if batch_progress_matches:
                            latest_batch_match = batch_progress_matches[-1]
                            current_epoch = int(latest_batch_match.group(1))
                            total_epochs = int(latest_batch_match.group(2))
                            current_batch = int(latest_batch_match.group(3))
                            total_batches = int(latest_batch_match.group(4))
                            if total_epochs > 0 and total_batches > 0:
                                # Calculate overall progress based on epoch and batch
                                progress_per_epoch = 100 / total_epochs
                                progress_within_epoch = (current_batch / total_batches) * progress_per_epoch
                                progress_percent = int(((current_epoch - 1) * progress_per_epoch) + progress_within_epoch)
                                progress_percent = min(100, max(0, progress_percent))
                                
                        elif epoch_summary_matches: # Fallback to epoch-level if no batch data (older logs or different format)
                            latest_epoch_summary = epoch_summary_matches[-1]
                            current_epoch = int(latest_epoch_summary.group(1))
                            total_epochs = int(latest_epoch_summary.group(2))
                            latest_train_loss = float(latest_epoch_summary.group(3))
                            latest_val_loss = float(latest_epoch_summary.group(4))
                            if total_epochs > 0:
                                progress_percent = int((current_epoch / total_epochs) * 100)
                                progress_percent = min(100, max(0, progress_percent))

                        if ppo_epoch_summary_matches:
                            latest_ppo_summary = ppo_epoch_summary_matches[-1]
                            # No need for current_epoch/total_epochs for RLHF PPO, as progress is usually per PPO_EPOCHS
                            # However, we extract them for consistency if available
                            ppo_current_epoch = int(latest_ppo_summary.group(1))
                            ppo_total_epochs = int(latest_ppo_summary.group(2))
                            latest_avg_reward = float(latest_ppo_summary.group(3))
                            # If RLHF, percentage is based on PPO_EPOCHS
                            if ppo_total_epochs > 0:
                                progress_percent = int((ppo_current_epoch / ppo_total_epochs) * 100)
                                progress_percent = min(100, max(0, progress_percent))

                except Exception as e:
                    app.logger.error(f"Could not parse log file {log_path}: {e}")

            if return_code is not None:
                if return_code == 0:
                    status = "Completed"
                    completion_message = "Model training completed successfully!"
                    progress_percent = 100 # Ensure 100% when completed
                else:
                    status = f"Failed (Code: {return_code})"
                    completion_message = f"Model training failed with exit code {return_code}."
            
            statuses[name] = {
                "status": status,
                "return_code": return_code,
                "progress_percent": progress_percent,
                "completion_message": completion_message if status == "Completed" else "",
                "latest_train_loss": latest_train_loss,
                "latest_val_loss": latest_val_loss,
                "latest_avg_reward": latest_avg_reward,
            }
    return jsonify(statuses)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
