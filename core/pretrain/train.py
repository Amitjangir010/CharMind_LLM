# ======================================================================================
# train.py
# Main training script, now refactored for DDP and best-checkpoint-saving.
# This script should be launched via `launch.py`.
# ======================================================================================

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split, Subset
from tqdm import tqdm
import json
import argparse
import time
import shutil # Added for checkpoint management
import logging # Added for logging
from typing import Optional # Added for Optional type hint

# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust path for core/pretrain
sys.path.append(PROJECT_ROOT)

from core.utils.dataset import BPEDataset
from core.utils.model import BPETransformer
from tokenizer.tokenizer import load_tokenizer
from core.utils.model_io import save_model
from core.utils.config import (
    CONTEXT_LENGTH, BATCH_SIZE, LEARNING_RATE, EPOCHS, D_MODEL, N_LAYER, N_HEAD, D_FF,
    DROPOUT, PROCESSED_TRAINING_FILE, TOKENIZER_PATH, PRETRAIN_CHECKPOINT_DIR,
    FINETUNE_CHECKPOINT_DIR, VOCAB_SIZE, USE_LORA, LORA_R, LORA_ALPHA, PRETRAIN_LOGS_DIR, FINETUNE_LOGS_DIR
)

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def validate_model(model, val_loader, device):
    """Calculates the validation loss for the model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = model(xb, yb)[1]
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main(rank, world_size, mode: str, run_name: str, base_model_path: Optional[str] = None):
    """Main training function to be run by each process."""
    is_ddp = world_size > 1
    if is_ddp:
        setup(rank, world_size)

    device = rank if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Determine checkpoint and logs directory based on mode
    if mode == 'pretrain':
        checkpoint_dir = PRETRAIN_CHECKPOINT_DIR
        logs_dir = PRETRAIN_LOGS_DIR
        use_lora = False # Pre-training should not use LoRA
    elif mode == 'finetune':
        checkpoint_dir = FINETUNE_CHECKPOINT_DIR
        logs_dir = FINETUNE_LOGS_DIR
        use_lora = USE_LORA # Use LoRA based on config for fine-tuning
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'pretrain' or 'finetune'.")

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # --- Data Loading (Rank 0 only) ---
    full_dataset = None
    if rank == 0:
        texts = []
        if os.path.exists(PROCESSED_TRAINING_FILE):
            with open(PROCESSED_TRAINING_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        texts.append(json.loads(line)['text'])
                    except (json.JSONDecodeError, KeyError):
                        continue
        if not texts:
            print(f"No data in {PROCESSED_TRAINING_FILE}. Run data/prepare_data.py. Exiting.")
            if is_ddp: cleanup()
            return
        full_dataset = BPEDataset(texts, tokenizer, CONTEXT_LENGTH)
        val_split = int(len(full_dataset) * 0.1)
        if val_split == 0 and len(full_dataset) > 0:
            val_split = 1
        train_split = len(full_dataset) - val_split
        train_indices, val_indices = random_split(range(len(full_dataset)), [train_split, val_split])
        # Save dataset indices to the mode-specific logs directory
        os.makedirs(logs_dir, exist_ok=True) # Ensure logs directory exists
        with open(os.path.join(logs_dir, 'dataset_indices.json'), 'w') as f:
            json.dump({'train': list(train_indices.indices), 'val': list(val_indices.indices)}, f)

    if is_ddp:
        dist.barrier()

    # --- Dataset and DataLoader for all ranks ---
    # Load dataset indices from the mode-specific logs directory
    with open(os.path.join(logs_dir, 'dataset_indices.json'), 'r') as f:
        indices = json.load(f)
    # Re-create dataset object on all ranks if not already loaded
    if full_dataset is None:
        # This part is tricky in DDP. For simplicity, we assume the dataset is small enough
        # that re-loading the texts on each rank is acceptable.
        texts = []
        with open(PROCESSED_TRAINING_FILE, 'r', encoding='utf-8') as f:
            for line in f: texts.append(json.loads(line)['text'])
        full_dataset = BPEDataset(texts, tokenizer, CONTEXT_LENGTH)

    train_dataset = Subset(full_dataset, indices['train'])
    val_dataset = Subset(full_dataset, indices['val'])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=val_sampler)

    # --- Model Initialization and Loading ---
    model = BPETransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT).to(device)
    if base_model_path:
        try:
            # Assuming base_model_path is a directory containing 'full_model.pt'
            model_load_path = os.path.join(base_model_path, 'full_model.pt')
            model.load_state_dict(torch.load(model_load_path, map_location=device))
            logging.info(f"Loaded base model from {model_load_path} for fine-tuning.")
        except Exception as e:
            logging.error(f"Failed to load base model from {base_model_path}: {e}")
            if rank == 0: print(f"Error: Failed to load base model for fine-tuning: {e}")
            if is_ddp: cleanup()
            return

    if use_lora:
        from .lora import patch_model_with_lora # Lazy import
        patch_model_with_lora(model, rank=LORA_R, alpha=LORA_ALPHA)
    if is_ddp:
        model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_model_path = None # To store the path of the best model

    for epoch in range(EPOCHS):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{mode}]", disable=(rank != 0))
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = model(xb, yb)[1]
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({"loss": f"{total_train_loss / (pbar.n + 1):.4f}"})
                # Log batch progress more frequently
                logging.info(f"Epoch {epoch+1}/{EPOCHS} [{mode}]: Batch {pbar.n + 1}/{len(train_loader)}")

        # --- Validation and Checkpointing (Rank 0 only) ---
        if rank == 0:
            val_loss = validate_model(model.module if is_ddp else model, val_loader, device)
            avg_train_loss = total_train_loss / len(train_loader)
            # Log epoch summary in a parsable format
            logging.info(f"Epoch Summary: Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Always save the current epoch's checkpoint
            model_to_save = model.module if is_ddp else model
            current_save_dir_name = f"{run_name}_epoch_{epoch+1}"
            current_save_directory = os.path.join(checkpoint_dir, current_save_dir_name)
            save_model(model_to_save, current_save_directory, is_lora=use_lora)
            print(f"Saved current model checkpoint to {current_save_directory} (Val Loss: {val_loss:.4f})")

            # Update best model if current one is better
            if val_loss < best_val_loss:
                # If there was a previous best model for this run_name, clear its 'is_best' flag
                if best_model_path and os.path.exists(best_model_path):
                    old_best_info_path = os.path.join(best_model_path, 'best_model_info.json')
                    if os.path.exists(old_best_info_path):
                        os.remove(old_best_info_path) # Remove old best flag
                        logging.info(f"Cleared old best model flag from: {old_best_info_path}")

                best_val_loss = val_loss
                best_model_path = current_save_directory
                
                # Save best model info
                best_info_path = os.path.join(best_model_path, 'best_model_info.json')
                with open(best_info_path, 'w') as f:
                    json.dump({"is_best": True, "val_loss": best_val_loss}, f)
                logging.info(f"New best model found and marked at {best_model_path} with Val Loss: {best_val_loss:.4f}")
            
            # Manage checkpoints: Keep latest 2 + the best one
            # Get all checkpoints for the current run_name
            run_checkpoints = []
            for d_name in os.listdir(checkpoint_dir):
                full_path = os.path.join(checkpoint_dir, d_name)
                if os.path.isdir(full_path) and d_name.startswith(f'{run_name}_epoch_'):
                    try:
                        # Extract epoch number for sorting
                        epoch_num = int(d_name.split('_epoch_')[1])
                        run_checkpoints.append((epoch_num, full_path))
                    except ValueError:
                        # Ignore directories that don't match the expected naming convention
                        continue

            # Sort by epoch number to get the latest
            run_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Identify checkpoints to keep
            keep_paths = set()
            # Keep the latest 2
            for _, path in run_checkpoints[:2]:
                keep_paths.add(path)
            # Always keep the overall best model (if it exists)
            if best_model_path:
                keep_paths.add(best_model_path)
            
            # Remove old checkpoints
            for _, path in run_checkpoints:
                if path not in keep_paths:
                    shutil.rmtree(path)
                    logging.info(f"Removed old checkpoint: {path}")

    # --- Cleanup ---
    if rank == 0:
        print(f"Training run '{run_name}' complete.")
        # Remove dataset_indices.json from the mode-specific logs directory
        if os.path.exists(os.path.join(logs_dir, 'dataset_indices.json')):
            os.remove(os.path.join(logs_dir, 'dataset_indices.json'))
            print("Cleaned up dataset_indices.json")
    if is_ddp:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main training script for BPE Transformer.")
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune'], help="Training mode: 'pretrain' or 'finetune'.")
    parser.add_argument('--run_name', type=str, default=f"run_{int(time.time())}", help="A name for the training run, used for checkpoint folder.")
    parser.add_argument('--base_model_path', type=str, default=None, help="Path to a base model checkpoint directory for fine-tuning.")
    args = parser.parse_args()

    # This script is intended to be launched by a process manager like torchrun
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    main(rank, world_size, args.mode, args.run_name, args.base_model_path)
