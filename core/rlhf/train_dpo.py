# ======================================================================================
# train_dpo.py
#
# A from-scratch implementation of Direct Preference Optimization (DPO) for
# aligning a language model with human preferences. This version is optimized
# for performance with batched operations.
#
# ======================================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import json

from ../tokenizer import load_tokenizer
from ../model import BPETransformer
from config import (
    CONTEXT_LENGTH, DEVICE, TOKENIZER_PATH, CHECKPOINTS_BASE_DIR, VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, D_FF, DROPOUT,
    PREFERENCE_DATA_FILE, RLHF_CHECKPOINT_DIR, PAD_TOKEN
)

# --- DPO Configuration ---
DPO_EPOCHS = 3
DPO_LR = 5e-6
DPO_BETA = 0.1
DPO_BATCH_SIZE = 4 # Can be increased now with batching

# --- Batched Preference Dataset & Collation ---
class PreferenceDataset(Dataset):
    # ... (previous implementation is fine, no changes needed here) ...
    def __init__(self, file_path, tokenizer, context_length):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preference data file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = context_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        chosen_full = f"User: {prompt}\nLLM: {chosen}"
        rejected_full = f"User: {prompt}\nLLM: {rejected}"
        chosen_tokens = self.tokenizer.encode(chosen_full).ids
        rejected_tokens = self.tokenizer.encode(rejected_full).ids
        if len(chosen_tokens) > self.max_len: chosen_tokens = chosen_tokens[-self.max_len:]
        if len(rejected_tokens) > self.max_len: rejected_tokens = rejected_tokens[-self.max_len:]
        return {"chosen_ids": torch.tensor(chosen_tokens, dtype=torch.long), "rejected_ids": torch.tensor(rejected_tokens, dtype=torch.long)}

def create_dpo_collate_fn(pad_token_id):
    """Creates a collate function for padding DPO batches."""
    def collate_fn(batch):
        chosen_ids = [item['chosen_ids'] for item in batch]
        rejected_ids = [item['rejected_ids'] for item in batch]

        chosen_padded = pad_sequence(chosen_ids, batch_first=True, padding_value=pad_token_id)
        rejected_padded = pad_sequence(rejected_ids, batch_first=True, padding_value=pad_token_id)

        # Create attention masks
        chosen_mask = (chosen_padded != pad_token_id).long()
        rejected_mask = (rejected_padded != pad_token_id).long()

        return {
            "chosen_ids": chosen_padded, "chosen_mask": chosen_mask,
            "rejected_ids": rejected_padded, "rejected_mask": rejected_mask
        }
    return collate_fn

# --- DPO Loss Function (From Scratch) ---
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    pi_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_log_ratios - ref_log_ratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss

def get_batch_log_probs(model, input_ids, attention_mask):
    """
    Calculates the log-probabilities for a batch of sequences, ignoring padding.
    """
    labels = input_ids[:, 1:].clone()
    labels[input_ids[:, 1:] == tokenizer.token_to_id(PAD_TOKEN)] = -100 # Ignore pad tokens in loss

    logits, _, _ = model(input_ids[:, :-1])

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log-probs for the tokens in the sequences
    per_token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Zero out the log-probs for padding tokens
    per_token_log_probs = per_token_log_probs * attention_mask[:, 1:]

    # Sum the log-probs for each sequence in the batch
    return per_token_log_probs.sum(dim=1)

# --- Main Training ---
def main():
    print("--- Starting From-Scratch DPO Training (Batched) ---")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN)

    policy_model = BPETransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT).to(DEVICE)
    base_model_ckpt = os.path.join(CHECKPOINTS_BASE_DIR, 'pretrain', 'bpe_best.pt')
    policy_model.load_state_dict(torch.load(base_model_ckpt, map_location=DEVICE))
    print(f"Loaded policy model from {base_model_ckpt}")

    ref_model = BPETransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT).to(DEVICE)
    ref_model.load_state_dict(torch.load(base_model_ckpt, map_location=DEVICE))
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Created frozen reference model.")

    dataset = PreferenceDataset(PREFERENCE_DATA_FILE, tokenizer, CONTEXT_LENGTH)
    collate_fn = create_dpo_collate_fn(pad_token_id)
    dataloader = DataLoader(dataset, batch_size=DPO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=DPO_LR)

    print(f"Starting DPO training for {DPO_EPOCHS} epochs...")
    for epoch in range(DPO_EPOCHS):
        policy_model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch + 1}/{DPO_EPOCHS}")

        for batch in pbar:
            optimizer.zero_grad()

            # Move batch to device
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)

            # Get log-probabilities from the policy model
            policy_chosen_logps = get_batch_log_probs(policy_model, chosen_ids, chosen_mask)
            policy_rejected_logps = get_batch_log_probs(policy_model, rejected_ids, rejected_mask)

            # Get log-probabilities from the reference model
            with torch.no_grad():
                ref_chosen_logps = get_batch_log_probs(ref_model, chosen_ids, chosen_mask)
                ref_rejected_logps = get_batch_log_probs(ref_model, rejected_ids, rejected_mask)

            loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=DPO_BETA)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

        print(f"Epoch {epoch + 1} average loss: {total_loss / len(dataloader):.4f}")

    print("\n--- DPO Training Finished ---")
    final_model_path = os.path.join(RLHF_CHECKPOINT_DIR, 'dpo_bpe_model.pt')
    torch.save(policy_model.state_dict(), final_model_path)
    print(f"Saved final DPO-aligned model to {final_model_path}")

if __name__ == "__main__":
    main()
