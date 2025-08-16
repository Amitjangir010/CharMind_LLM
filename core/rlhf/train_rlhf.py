# ======================================================================================
# train_rlhf.py
#
# A from-scratch implementation of Proximal Policy Optimization (PPO) for
# Reinforcement Learning from Human Feedback (RLHF). This version is optimized
# for performance with a batched update step.
# ======================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import os
import sys
# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust path for core/rlhf
sys.path.append(PROJECT_ROOT)
import json
import argparse
import time
import logging
import shutil

from tokenizer.tokenizer import load_tokenizer
from ..utils.model import BPETransformer
from .reward_model import RewardModel
from ..utils.model_io import save_model
from ..utils.config import (
    CONTEXT_LENGTH, DEVICE, TOKENIZER_PATH, VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, D_FF, DROPOUT,
    REWARD_MODEL_DIR, PPO_EPOCHS, PPO_LEARNING_RATE, PPO_CLIP_EPSILON, PROCESSED_PREFERENCE_FILE,
    RLHF_CHECKPOINT_DIR, PAD_TOKEN, PRETRAIN_CHECKPOINT_DIR, BASE_MODEL_CHECKPOINT # Added BASE_MODEL_CHECKPOINT
)

# --- Configuration ---
# These are now mostly from config.py
PPO_BATCH_SIZE = 4
GAMMA = 0.99
LAMBDA = 0.95
KL_COEFF = 0.2
PPO_UPDATE_EPOCHS = 4 # Inner loop epochs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = BPETransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT)
        self.critic_head = nn.Linear(D_MODEL, 1)
    def forward(self, input_ids):
        logits, _, _, hidden_states = self.actor(input_ids, return_hidden_states=True)
        value = self.critic_head(hidden_states) # Get value for each token's hidden state
        return logits, value.squeeze(-1)

# --- PPO Experience Dataset ---
class PPOExperienceDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences
    def __len__(self):
        return len(self.experiences)
    def __getitem__(self, idx):
        return self.experiences[idx]


def main(run_name: str, base_model_path: str):
    """Main RLHF training function."""
    logging.info(f"--- Starting PPO RLHF Training Run: {run_name} ---")
    logging.info(f"--- Using base model from: {base_model_path} ---")

    # --- 1. Load Models and Tokenizer ---
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN)

    actor_critic = ActorCritic().to(DEVICE)
    ref_model = BPETransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT).to(DEVICE)
    reward_model = RewardModel(base_model_path).to(DEVICE)
    # Use REWARD_MODEL_DIR from config, which now points to core/rlhf/saved_models/reward_model/
    reward_model_ckpt = os.path.join(REWARD_MODEL_DIR, 'reward_model.pt')

    try:
        # Base model can be from pretrain or finetune, so construct path based on its location
        # Assuming base_model_path is already the full path to a checkpoint directory (e.g., core/pretrain/saved_models/run_xxx_epoch_yy)
        base_model_file_path = os.path.join(base_model_path, 'full_model.pt')
        actor_critic.actor.load_state_dict(torch.load(base_model_file_path, map_location=DEVICE))
        ref_model.load_state_dict(torch.load(base_model_file_path, map_location=DEVICE))
        
        # Load reward model
        reward_model.load_state_dict(torch.load(reward_model_ckpt, map_location=DEVICE))
    except FileNotFoundError as e:
        logging.error(f"Error: A required model file was not found. {e}")
        logging.error("Please ensure you have a trained base model and reward model.")
        return

    ref_model.eval()
    for param in ref_model.parameters(): param.requires_grad = False
    reward_model.eval()
    for param in reward_model.parameters(): param.requires_grad = False

    # --- 2. Prepare Dataset of Prompts ---
    prompts = []
    if os.path.exists(PROCESSED_PREFERENCE_FILE):
        with open(PROCESSED_PREFERENCE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prompts.append(json.loads(line)['prompt'])
                except (json.JSONDecodeError, KeyError):
                    logging.warning(f"Skipping malformed line in {PROCESSED_PREFERENCE_FILE}: {line.strip()}")
                    continue
    if not prompts:
        logging.error(f"No prompts in {PROCESSED_PREFERENCE_FILE}. Run data/prepare_data.py. Exiting.")
        return
    prompts = sorted(list(set(prompts)))
    prompt_tensors = [torch.tensor(tokenizer.encode(p).ids, dtype=torch.long) for p in prompts]

    # --- 3. PPO Training Loop ---
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=PPO_LEARNING_RATE)

    best_reward_score = float('-inf') # For tracking the best model based on reward
    best_model_path = None

    for ppo_epoch in range(PPO_EPOCHS):
        logging.info(f"\n--- PPO Epoch {ppo_epoch + 1}/{PPO_EPOCHS} ---")

        actor_critic.eval()
        experiences = []
        epoch_rewards = [] # To track rewards for this epoch

        for prompt_tensor in tqdm(prompt_tensors, desc="Generating Trajectories"):
            query_tensor_unbatched = prompt_tensor.to(DEVICE)
            query_for_gen = query_tensor_unbatched.unsqueeze(0)

            response_tokens, log_probs, values = [], [], []
            with torch.no_grad():
                for _ in range(100): # Max generation length
                    logits, value_sequence = actor_critic(query_for_gen)
                    values.append(value_sequence[:, -1].squeeze())
                    prob_dist = torch.softmax(logits[:, -1, :], dim=-1)
                    action = torch.multinomial(prob_dist, 1)
                    log_prob = torch.log(prob_dist.gather(1, action)).squeeze()
                    response_tokens.append(action.item())
                    log_probs.append(log_prob)
                    query_for_gen = torch.cat([query_for_gen, action], dim=1)
                    if action.item() == tokenizer.token_to_id("[EOS]"): break

            with torch.no_grad():
                response_tensor = torch.tensor(response_tokens, dtype=torch.long, device=DEVICE)
                log_probs = torch.stack(log_probs)
                values = torch.stack(values)
                full_sequence = torch.cat([query_tensor_unbatched, response_tensor]).unsqueeze(0)
                ref_logits, _, _, _ = ref_model(full_sequence)
                kl_penalty = log_probs - torch.log_softmax(ref_logits, dim=-1)[0, len(query_tensor_unbatched)-1:-1].gather(1, response_tensor.view(-1, 1)).squeeze()
                final_reward_score = reward_model(full_sequence).squeeze()
                rewards = -KL_COEFF * kl_penalty
                rewards[-1] += final_reward_score
                
                epoch_rewards.append(final_reward_score.item()) # Store final reward for this trajectory

                advantages = torch.zeros_like(rewards)
                last_advantage = 0
                for t in reversed(range(len(rewards))):
                    next_value = values[t+1] if t < len(rewards) - 1 else 0
                    delta = rewards[t] + GAMMA * next_value - values[t]
                    last_advantage = delta + GAMMA * LAMBDA * last_advantage
                    advantages[t] = last_advantage
                returns = advantages + values
                experiences.append({
                    "query": query_tensor_unbatched, "response": response_tensor,
                    "log_probs": log_probs, "advantages": advantages, "returns": returns
                })

        actor_critic.train()
        for _ in tqdm(range(PPO_UPDATE_EPOCHS), desc="PPO Inner Update Epochs"):
            for exp in experiences:
                query, response, old_log_probs, advantages, returns = exp.values()
                full_sequence = torch.cat([query, response]).unsqueeze(0)
                logits, value_sequence = actor_critic(full_sequence)
                values = value_sequence[0, len(query)-1:-1]
                prob_dist = torch.softmax(logits[0, len(query)-1:-1], dim=-1)
                new_log_probs = prob_dist.gather(1, response.view(-1, 1)).squeeze()
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # After inner PPO update loops, calculate average reward for the epoch
        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        # Log epoch summary in a parsable format
        logging.info(f"PPO Epoch Summary: Epoch {ppo_epoch + 1}/{PPO_EPOCHS}, Avg Reward: {avg_epoch_reward:.4f}")

        # Save checkpoint periodically and manage them
        current_save_dir_name = f"{run_name}_ppo_epoch_{ppo_epoch+1}"
        # The checkpoint will be saved inside core/rlhf/saved_models/rlhf_model/
        current_save_directory = os.path.join(RLHF_CHECKPOINT_DIR, current_save_dir_name)
        save_model(actor_critic.actor, current_save_directory, is_lora=False)
        logging.info(f"Saved current RLHF model checkpoint to {current_save_directory} (Avg Reward: {avg_epoch_reward:.4f})")

        # Update best model if current one is better
        if avg_epoch_reward > best_reward_score: # Maximize reward
            best_reward_score = avg_epoch_reward
            best_model_path = current_save_directory
            logging.info(f"New best RLHF model found at {best_model_path} with Avg Reward: {best_reward_score:.4f}")

        # Manage checkpoints: Keep latest 2 + the best one for this run_name
        run_checkpoints = []
        # Search within the specific RLHF model save directory
        for d_name in os.listdir(RLHF_CHECKPOINT_DIR):
            full_path = os.path.join(RLHF_CHECKPOINT_DIR, d_name)
            if os.path.isdir(full_path) and d_name.startswith(f'{run_name}_ppo_epoch_'):
                try:
                    # Extract epoch number for sorting
                    epoch_num = int(d_name.split('_ppo_epoch_')[1])
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
        # Always keep the overall best model
        if best_model_path:
            keep_paths.add(best_model_path)
        
        # Remove old checkpoints
        for _, path in run_checkpoints:
            if path not in keep_paths:
                shutil.rmtree(path)
                logging.info(f"Removed old RLHF checkpoint: {path}")

    logging.info("\n--- PPO Training Finished ---")
    # The final saving part is now handled by the per-epoch saving and cleanup.
    # No need for this block:
    # final_model_dir = os.path.join(RLHF_CHECKPOINT_DIR, run_name)
    # save_model(actor_critic.actor, final_model_dir, is_lora=False)
    # logging.info(f"Saved final aligned actor model to {final_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF Training using PPO.")
    parser.add_argument('--run_name', type=str, default=f"rlhf_{int(time.time())}", help="A name for the training run.")
    parser.add_argument('--base_model_path', type=str, required=True, help="Path to the pre-trained or fine-tuned model checkpoint directory to start from.")
    args = parser.parse_args()
    main(args.run_name, args.base_model_path)
