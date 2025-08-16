# src/rlhf/reward_model.py
# Defines the RewardModel class for the BPE-based RLHF pipeline.

import torch
import torch.nn as nn
import os
from tokenizer.tokenizer import load_tokenizer
from ..utils.model import BPETransformer
from ..utils.config import (
    D_MODEL, N_LAYER, N_HEAD, D_FF, DROPOUT, DEVICE, CONTEXT_LENGTH, VOCAB_SIZE
)

class RewardModel(nn.Module):
    """
    The RewardModel class. It uses a pre-trained transformer model as a base
    and adds a linear head to output a scalar reward value.
    """
    def __init__(self, base_model_ckpt_path):
        super().__init__()

        # Initialize the base transformer model.
        # It's crucial that the architecture matches the pre-trained model.
        self.base_model = BPETransformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            d_ff=D_FF,
            context_length=CONTEXT_LENGTH,
            dropout=DROPOUT,
            device=DEVICE
        )

        # Load the weights from the pre-trained base model checkpoint
        print(f"Loading base model weights from: {base_model_ckpt_path}")
        self.base_model.load_state_dict(torch.load(os.path.join(base_model_ckpt_path, 'full_model.pt'), map_location=DEVICE))

        # The reward head is a single linear layer that maps the model's
        # hidden state to a single scalar value (the reward).
        self.reward_head = nn.Linear(D_MODEL, 1)

    def forward(self, input_ids):
        """
        Forward pass for the reward model.

        Args:
            input_ids (torch.Tensor): A tensor of token IDs of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of scalar reward values of shape (batch_size,).
        """
        # Get the hidden states from the base model by passing the new flag.
        _, _, _, hidden_states = self.base_model(input_ids, return_hidden_states=True)

        # We use the hidden state of the last token in the sequence as the
        # representation of the entire sequence.
        last_hidden_state = hidden_states[:, -1, :]

        # Pass the last hidden state through the reward head to get the scalar reward.
        reward = self.reward_head(last_hidden_state)

        # Squeeze to remove the last dimension, resulting in a (batch_size,) tensor.
        return reward.squeeze(-1)

if __name__ == '__main__':
    # Example of how to instantiate the reward model.
    # This assumes you have a pre-trained base model checkpoint.

    # NOTE: This is a placeholder path. You need a real checkpoint from `train.py`.
    EXAMPLE_CHECKPOINT_PATH = 'saved_models/checkpoints/bpe/bpe_best.pt'

    if not os.path.exists(EXAMPLE_CHECKPOINT_PATH):
        print("Warning: Example checkpoint not found. This script will not be able to run.")
        print(f"Please make sure a checkpoint exists at: {EXAMPLE_CHECKPOINT_PATH}")
    else:
        print("Instantiating RewardModel with example configuration...")
        model = RewardModel(base_model_ckpt_path=EXAMPLE_CHECKPOINT_PATH).to(DEVICE)
        print("RewardModel instantiated successfully.")

        # Create a dummy input tensor
        dummy_input = torch.randint(0, VOCAB_SIZE, (4, CONTEXT_LENGTH), device=DEVICE)

        # Test the forward pass
        rewards = model(dummy_input)
        print(f"Forward pass successful. Output shape: {rewards.shape}")
        print(f"Example rewards: {rewards.detach().cpu().numpy()}")