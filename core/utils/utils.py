# src/utils.py
# Utility functions for the CharMind_LLM project.
# Consolidates common, reusable functionality used across multiple modules.

import os
import json
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
from .config import EOS_TOKEN

def ensure_directory_exists(directory_path: str):
    """Ensures a directory exists, creating it if necessary."""
    os.makedirs(directory_path, exist_ok=True)

def load_json_file(file_path: str, default: Any = None) -> Any:
    """Loads a JSON file with error handling, returning a default value on failure."""
    if not os.path.exists(file_path):
        return default if default is not None else []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return default if default is not None else []

def save_json_file(file_path: str, data: Any) -> bool:
    """Saves data to a JSON file with error handling."""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def get_device() -> str:
    """Gets the appropriate torch device, preferring CUDA if available."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def format_model_size(num_params: int) -> str:
    """Formats a number of parameters into a human-readable string (e.g., 1.2B, 250M)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the total number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clean_text(text: str) -> str:
    """Cleans and normalizes a string by trimming and unifying whitespace."""
    if not text:
        return ""
    return ' '.join(text.split()).strip()

def prepare_model_input(
    prompt: str,
    tokenizer: Any,
    context_length: int,
    device: str,
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """
    Prepares a tokenized and correctly truncated input tensor for the model.

    This function builds a full prompt from system instructions, conversation history,
    and the latest user message. It then tokenizes and truncates the sequence from the
    left to ensure it fits within the model's context window.

    Args:
        prompt: The latest user prompt.
        tokenizer: The tokenizer instance with an `.encode()` method.
        context_length: The maximum sequence length for the model.
        device: The torch device ('cuda' or 'cpu') for the tensor.
        history: A list of past conversation turns.
        system_prompt: An optional system-level instruction.

    Returns:
        A torch.Tensor of shape (1, T) where T <= context_length.
    """
    full_prompt_parts = []
    if system_prompt:
        full_prompt_parts.append(f"system: {system_prompt}")

    if history:
        history_str = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        full_prompt_parts.append(history_str)

    full_prompt_parts.append(f"user: {prompt}")
    full_prompt = "\n".join(full_prompt_parts)

    # Tokenize and truncate from the left
    token_ids = tokenizer.encode(full_prompt).ids
    if len(token_ids) > context_length:
        token_ids = token_ids[-context_length:]

    # Ensure token_ids is not empty; if so, use EOS_TOKEN
    if not token_ids:
        token_ids = [tokenizer.encode(EOS_TOKEN).ids[0]] # Encode EOS_TOKEN and take its first ID

    print(f"[DEBUG] prepare_model_input: token_ids before tensor creation: {token_ids}")
    return torch.tensor([token_ids], dtype=torch.long, device=device)
