# src/rlhf/train_reward_model.py
# Training script for the Reward Model.

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
import sys
from tqdm import tqdm

# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust path for core/rlhf
sys.path.append(PROJECT_ROOT)

from tokenizer.tokenizer import load_tokenizer
from .reward_model import RewardModel
from ..utils.config import (
    CONTEXT_LENGTH, DEVICE, TOKENIZER_PATH, VOCAB_SIZE,
    REWARD_MODEL_DIR, RLHF_DATA_DIR, PRETRAIN_CHECKPOINT_DIR, REWARD_MODEL_EPOCHS, REWARD_MODEL_LR, REWARD_MODEL_BATCH_SIZE, BASE_MODEL_CHECKPOINT
)

# --- RLHF-Specific Config ---
# Paths are now centrally managed in config.py
# Removed hardcoded paths
REWARD_MODEL_SAVE_PATH = os.path.join(REWARD_MODEL_DIR, 'reward_model.pt')
PREFERENCE_FILE_PATH = os.path.join(RLHF_DATA_DIR, "Preference_data.json") # Using the correct data path
# BASE_MODEL_CHECKPOINT is now imported from config

# Performance optimizations
TORCH_COMPILE = False
use_amp = DEVICE == 'cuda'


class PreferenceDataset(Dataset):
    """Dataset for preference pairs (chosen, rejected)."""
    def __init__(self, json_path, tokenizer, max_length):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Preference data file not found: {json_path}. Run generate_preference_data.py first.")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        # Configure padding on the tokenizer
        self.tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        # Combine prompt and response
        chosen_text = f"Prompt: {prompt}\nResponse: {chosen}"
        rejected_text = f"Prompt: {prompt}\nResponse: {rejected}"

        # Tokenize (padding and truncation are handled by the tokenizer settings)
        chosen_tokens = self.tokenizer.encode(chosen_text).ids
        rejected_tokens = self.tokenizer.encode(rejected_text).ids

        return {
            'chosen_ids': torch.tensor(chosen_tokens, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_tokens, dtype=torch.long)
        }


def train_reward_model():
    """Main training function for the reward model."""
    print("--- Starting Reward Model Training ---")

    # Use the path from config
    os.makedirs(REWARD_MODEL_DIR, exist_ok=True) # Ensure the directory exists

    # --- Load Tokenizer and Dataset ---
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    dataset = PreferenceDataset(PREFERENCE_FILE_PATH, tokenizer, CONTEXT_LENGTH)

    if len(dataset) == 0:
        print("No preference data found to train the reward model. Please collect some ratings in the UI first.")
        return

    dataloader = DataLoader(dataset, batch_size=REWARD_MODEL_BATCH_SIZE, shuffle=True)

    # --- Initialize Model ---
    model = RewardModel(base_model_ckpt_path=BASE_MODEL_CHECKPOINT).to(DEVICE)

    if TORCH_COMPILE and hasattr(torch, 'compile'):
        print("Compiling the reward model with torch.compile()...")
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=REWARD_MODEL_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Training reward model for {REWARD_MODEL_EPOCHS} epochs on {len(dataset)} preference pairs...")

    # --- Training Loop ---
    for epoch in range(REWARD_MODEL_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{REWARD_MODEL_EPOCHS}")

        for batch in pbar:
            chosen_ids = batch['chosen_ids'].to(DEVICE)
            rejected_ids = batch['rejected_ids'].to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                chosen_scores = model(chosen_ids)
                rejected_scores = model(rejected_ids)
                loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{REWARD_MODEL_EPOCHS} | Average Loss: {avg_loss:.4f}")

    # --- Save Model ---
    print(f"Training complete. Saving reward model to: {REWARD_MODEL_SAVE_PATH}")
    # If using torch.compile, it's better to save the state_dict of the original model
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(model_to_save.state_dict(), REWARD_MODEL_SAVE_PATH)
    print("--- Reward Model Training Finished ---")


if __name__ == '__main__':
    train_reward_model()