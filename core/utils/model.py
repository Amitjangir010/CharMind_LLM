# src/model.py
# Professional, optimized Transformer model with strict configuration and KV cache.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Implements rotary embeddings for positional information, which is more
    effective than traditional positional embeddings for transformers.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the cosine and sine components for RoPE."""
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary positional embedding to query and key tensors."""
    cos = cos.squeeze(1).squeeze(0)[position_ids]
    sin = sin.squeeze(1).squeeze(0)[position_ids]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with RoPE and KV caching.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float, context_length: int, device: Optional[str] = None):
        super().__init__()
        self.d_model, self.n_head, self.head_dim = d_model, n_head, d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        # context_length is no longer passed here directly to RotaryEmbedding
        # RotaryEmbedding will use its own max_seq_len initialized from BPETransformer's context_length
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=context_length, device=device)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None, # Added
        position_ids: Optional[torch.LongTensor] = None, # Added
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Use the passed position_ids directly, no need to re-calculate past_len here
        if position_ids is None:
            # Fallback for when position_ids are not explicitly provided (e.g., in initial self-test)
            past_len = layer_past[0].shape[-2] if layer_past is not None else 0
            total_seq_len = past_len + T
            position_ids = torch.arange(past_len, total_seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Apply rotary embeddings using the provided position_ids
        # The seq_len for rotary_emb should be derived from the position_ids or total sequence length.
        # The current implementation of rotary_emb requires seq_len, which should be max(position_ids) + 1
        # For consistent behavior, let's derive it from the total length of k/v after concat if past_key_values exist.
        # If past_key_values is None, it's just T.
        current_total_len = T + (layer_past[0].shape[-2] if layer_past is not None else 0)
        cos, sin = self.rotary_emb(v, seq_len=current_total_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = (k, v) if use_cache else None

        # Apply attention mask if provided (for padding etc.)
        # Note: ScaledDotProductAttention in PyTorch handles masks internally when passed.
        # If attention_mask is provided, it should be of shape (B, 1, T_q, T_k)
        attn_mask = None
        if attention_mask is not None:
            # Expand to (B, N_head, T, T_total) for broadcasting
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(1) # B, 1, 1, T
            # We need a mask of shape (B, N_head, T_q, T_k) where T_q is current_seq_len (T) and T_k is total_seq_len (current_total_len)
            # The attention mask typically comes as (batch_size, sequence_length)
            # It needs to be expanded to be compatible with attention scores
            # This might require custom masking logic if not using nn.MultiheadAttention directly.
            # For now, let's assume attention_mask is compatible or needs to be created from input_ids length
            # For simple causal masking with padding, you might need to combine with current_total_len
            # For now, let's keep it simple and assume the mask is correctly formed for total length.
            
            # A common way to handle causal and padding mask together:
            # causal_mask = torch.triu(torch.ones(T, current_total_len, dtype=torch.bool, device=x.device), diagonal=1)
            # attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_head, T, -1) & ~causal_mask
            pass # This requires more complex mask handling, leaving for now as is_causal handles causal.
            

        # For decoding, is_causal should be False if we're only passing one token (for the next step)
        # and we are using past_key_values. Otherwise, use T > 1 to determine causality for initial pass.
        is_causal_attention = T > 1 or (layer_past is None and T == 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal_attention)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, present

class FeedForward(nn.Module):
    """A simple feed-forward network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False), nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """A single Transformer block, combining attention and feed-forward layers."""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float, context_length: int, device: Optional[str] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout, context_length, device=device)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x_ln = self.ln1(x)
        # Pass position_ids and attention_mask to MultiHeadAttention
        attn_output, present = self.attn(
            x_ln,
            layer_past=layer_past,
            use_cache=use_cache,
            attention_mask=attention_mask, # Pass attention_mask
            position_ids=position_ids,      # Pass position_ids
        )
        h = residual + attn_output
        h_ln = self.ln2(h)
        ff_output = self.ff(h_ln)
        output = h + ff_output
        return output, present

class BPETransformer(nn.Module):
    """
    Byte-Pair Encoding Transformer model.
    A full transformer model with embedding, multiple transformer blocks, and a final
    language model head. Supports KV caching for efficient generation.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, d_ff: int, context_length: int, dropout: float, device: Optional[str] = None):
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout, context_length, device) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_hidden_states: bool = False, # Added this argument
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[torch.Tensor]]:

        batch_size, current_sequence_length = input_ids.shape

        # 1. Determine past sequence length from the KV cache
        if past_key_values is not None:
            past_sequence_length = past_key_values[0][0].shape[2] # Shape: (batch_size, num_heads, seq_len, head_dim)
        else:
            past_sequence_length = 0

        # 2. Correctly compute position_ids if not provided
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_sequence_length,
                past_sequence_length + current_sequence_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, current_sequence_length)

        hidden_states = self.token_emb(input_ids)

        presents = [] if use_cache else None

        # If past_key_values is None, initialize it for the loop
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Pass the correctly calculated position_ids to each block
            hidden_states, present = block(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
                attention_mask=attention_mask, # Pass attention_mask
                position_ids=position_ids,      # Pass position_ids
            )
            if use_cache:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states) # Apply final layer norm

        logits = self.lm_head(hidden_states)
        loss = None
        if targets is not None:
            # Reshape logits and targets for cross_entropy
            # (batch_size * sequence_length, vocab_size) and (batch_size * sequence_length)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, presents, hidden_states # Added hidden_states to return

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, eos_token_id: int, temperature: float = 1.0, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generates a sequence of tokens autoregressively.
        Uses KV caching for efficient generation.
        """
        self.eval()
        past_key_values = None

        # Initialize position_ids for the first token
        # This will be updated inside the loop for subsequent tokens
        position_ids = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)

        for i in range(max_new_tokens):
            # If the current sequence length plus the new token would exceed context_length, stop.
            # This prevents position_ids from going out of bounds in RotaryEmbedding.
            if idx.size(1) >= self.context_length:
                break

            # If we have a KV cache, we only need to process the last token.
            # Otherwise, use the full sequence (truncated if necessary for the first pass).
            if past_key_values:
                input_ids = idx[:, -1:]
                # Update position_ids for the new token (it's the last one)
                # Its position is the current total length of the sequence, but clamped to context_length
                current_position = idx.size(1) - 1
                # If total sequence length exceeds context_length, we use a sliding window approach for RoPE
                # The position of the new token relative to the *start of the current window*
                clamped_position = min(current_position, self.context_length - 1)
                position_ids = torch.tensor([[clamped_position]], dtype=torch.long, device=idx.device)
            else:
                # For the first token, ensure input_ids doesn't exceed context_length
                # position_ids are already initialized for the first full sequence, clamped if necessary.
                input_ids = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
                # When prefilling, position_ids are from 0 to current_sequence_length - 1 (clamped)
                position_ids = torch.arange(
                    0, 
                    input_ids.size(1), 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0)

            logits, _, past_key_values, _ = self(input_ids, use_cache=True, past_key_values=past_key_values, position_ids=position_ids)
            logits = logits[:, -1, :] / temperature

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            # If EOS token is generated, stop
            if idx_next.item() == eos_token_id:
                break

        self.train()
        return idx

if __name__ == "__main__":
    # This block is for self-testing the model file independently.
    # In actual training, all values will come from .config.py.
    print("--- Running Model Self-Test ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # All parameters must be defined here for the test
    test_config = {
        'vocab_size': 50,
        'd_model': 128,
        'n_layer': 4,
        'n_head': 4,
        'd_ff': 128 * 4,
        'context_length': 256,
        'dropout': 0.1,
        'device': device
    }

    model = BPETransformer(**test_config).to(device)

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Test forward pass
    x = torch.randint(0, test_config['vocab_size'], (4, 128), device=device)
    logits, loss, _, _ = model(x, targets=x)
    print("Forward pass successful. Loss:", loss.item())

    # Test generation
    generated = model.generate(x[:,:1], max_new_tokens=50, eos_token_id=5)
    print("Generation successful. Shape:", generated.shape)
    print("--- Self-Test Complete ---")
