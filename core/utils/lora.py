# ======================================================================================
# lora.py
# From-scratch implementation of a LoRA (Low-Rank Adaptation) layer.
# ======================================================================================

import torch
import torch.nn as nn
import math

class LoraLayer(nn.Module):
    """
    Implements a from-scratch Low-Rank Adaptation (LoRA) layer.

    This layer wraps a standard `torch.nn.Linear` layer and adds a parallel,
    trainable low-rank path. The original weights of the linear layer are frozen,
    and only the low-rank matrices (A and B) are updated during fine-tuning.
    This significantly reduces the number of trainable parameters.

    The forward pass computes `h = Wx + (B*A)x * (alpha/r)`, where W is the frozen
    original weight matrix and A and B are the trainable low-rank matrices.
    """
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        """
        Initializes the LoraLayer.

        Args:
            original_layer (nn.Linear): The linear layer from the pre-trained
                model that will be adapted.
            rank (int): The rank 'r' of the low-rank decomposition. This determines
                the size of the trainable matrices and the expressiveness of the
                adaptation. A lower rank means fewer parameters.
            alpha (int): A scaling factor for the LoRA path. It's a hyperparameter
                that balances the influence of the LoRA path. A common practice
                is to set alpha equal to the rank.
        """
        super().__init__()
        self.linear = original_layer

        # Freeze the original layer's weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        self.rank = rank
        self.alpha = alpha

        # Create the low-rank matrices A and B
        self.lor_A = nn.Linear(in_features, rank, bias=False).to(self.linear.weight.device)
        self.lor_B = nn.Linear(rank, out_features, bias=False).to(self.linear.weight.device)

        # Initialize lor-A with Kaiming uniform distribution and lor-B with zeros
        nn.init.kaiming_uniform_(self.lor_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lor_B.weight)

        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the LoRA layer.

        The output is the sum of the original linear layer's output and the scaled
        output of the LoRA path.
        """
        # Original path (frozen)
        original_output = self.linear(x)

        # LoRA path (trainable)
        a_output = self.lor_A(x)
        if a_output.size(-1) != self.rank:
            print(f"Shape mismatch in LoRA: A output shape {a_output.shape}, expected rank {self.rank}")
            return original_output
        lora_output = self.lor_B(a_output) * self.scaling

        return original_output + lora_output

    def __repr__(self):
        return (f"{self.__class__.__name__}(rank={self.rank}, alpha={self.alpha}, "
                f"original_layer={self.linear})")


def patch_model_with_lora(model: nn.Module, rank: int, alpha: int) -> int:
    """
    Recursively traverses a model and replaces all `torch.nn.Linear` layers
    with the custom `LoraLayer`.

    This function modifies the model in-place, meaning it directly changes the
    layers of the passed model object. It's a convenient way to apply LoRA
    to an entire transformer architecture without manually specifying which
    layers to adapt.

    Args:
        model (nn.Module): The model to be patched with LoRA layers.
        rank (int): The rank 'r' to be used for all LoRA layers.
        alpha (int): The alpha scaling factor to be used for all LoRA layers.

    Returns:
        int: The total number of layers that were replaced.

    Example:
        >>> model = MyTransformer()
        >>> patched_count = patch_model_with_lora(model, rank=8, alpha=16)
        >>> print(f"Patched {patched_count} layers.")
    """
    patched_layers_count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the linear layer with our LoraLayer
            setattr(model, name, LoraLayer(module, rank, alpha))
            patched_layers_count += 1
        else:
            # Recurse into submodules and add the count of patched layers
            patched_layers_count += patch_model_with_lora(module, rank, alpha)

    return patched_layers_count
