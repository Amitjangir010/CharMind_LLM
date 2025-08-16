# ======================================================================================
# model_io.py
# Handles robust model saving and loading, with special handling for LoRA weights.
# ======================================================================================

import torch
import os
import json

from .model import BPETransformer
from .lora import LoraLayer, patch_model_with_lora
from .config import (
    VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, D_FF, CONTEXT_LENGTH, DROPOUT,
    LORA_R, LORA_ALPHA, CHECKPOINTS_BASE_DIR
)

def save_model(model, save_directory, is_lora):
    """
    Saves a model to a specified directory with special handling for LoRA.

    This function implements a robust saving strategy:
    - If LoRA was used for training (`is_lora=True`), it saves only the
      trainable LoRA weights (`lora_weights.pt`) and the LoRA configuration
      (`lora_config.json`). This is highly efficient as the large base model
      weights are not duplicated.
    - If LoRA was not used, it saves the entire model's state dictionary
      to `full_model.pt`.

    Args:
        model (nn.Module): The model instance to save.
        save_directory (str): The path to the directory where model files
            will be stored. The directory will be created if it doesn't exist.
        is_lora (bool): A flag indicating if the model was trained with LoRA.
            This determines the saving strategy.
    """
    os.makedirs(save_directory, exist_ok=True)

    if is_lora:
        print(f"Saving LoRA model to {save_directory}...")
        # Save only the trainable LoRA parameters
        lora_weights = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
        torch.save(lora_weights, os.path.join(save_directory, 'lora_weights.pt'))

        # Save the LoRA configuration
        lora_config = {'r': LORA_R, 'alpha': LORA_ALPHA}
        with open(os.path.join(save_directory, 'lora_config.json'), 'w') as f:
            json.dump(lora_config, f)

        print("LoRA weights and config saved successfully.")
    else:
        print(f"Saving full model to {save_directory}...")
        torch.save(model.state_dict(), os.path.join(save_directory, 'full_model.pt'))
        print("Full model saved successfully.")


def load_model(load_directory, device):
    """
    Loads a model from a directory, automatically detecting if it's a full model
    or a LoRA-adapted model.

    This function first initializes a base `BPETransformer` architecture. It then
    checks the specified directory for model files:
    - If `lora_weights.pt` and `lora_config.json` are found, it assumes a
      LoRA model. It loads the base model weights, patches the model with LoRA
      layers based on the saved config, and then loads the LoRA weights into
      the patched model.
    - If `full_model.pt` is found, it loads the state dict directly into the
      base architecture.
    - If neither is found, it raises an error.

    Args:
        load_directory (str): The directory containing the model files.
        device: The torch device ('cpu' or 'cuda') to load the model onto.

    Returns:
        torch.nn.Module: The fully loaded and configured model, ready for inference.
    """
    lora_weights_path = os.path.join(load_directory, 'lora_weights.pt')
    lora_config_path = os.path.join(load_directory, 'lora_config.json')
    full_model_path = os.path.join(load_directory, 'full_model.pt')

    # Initialize the base model architecture
    model = BPETransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD,
        d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT
    ).to(device)

    if os.path.exists(lora_weights_path) and os.path.exists(lora_config_path):
        print(f"Found LoRA weights at {load_directory}. Loading LoRA model...")
        # Load the base model weights (assuming a base checkpoint exists)
        # In a real system, you'd specify the base model path. Here we assume
        # the user starts with a pre-trained `bpe_best.pt`.
        base_ckpt_path = os.path.join(CHECKPOINTS_BASE_DIR, 'pretrain', 'epoch_50', 'full_model.pt')
        if os.path.exists(base_ckpt_path):
            model.load_state_dict(torch.load(base_ckpt_path, map_location=device))

        # Patch the model with LoRA layers using the saved config
        with open(lora_config_path, 'r') as f:
            lora_config = json.load(f)
        patch_model_with_lora(model, rank=lora_config['r'], alpha=lora_config['alpha'])

        # Load the saved LoRA weights
        # strict=False allows loading a partial state_dict
        model.load_state_dict(torch.load(lora_weights_path, map_location=device), strict=False)
        print("LoRA model loaded successfully.")

    elif os.path.exists(full_model_path):
        print(f"Found full model weights at {load_directory}. Loading full model...")
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        print("Full model loaded successfully.")

    else:
        raise FileNotFoundError(f"No valid model checkpoint or LoRA weights found in {load_directory}")

    return model
