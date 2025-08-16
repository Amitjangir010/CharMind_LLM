# src/memory.py
# A hierarchical, long-term memory system for the language model.

import torch
from torch import nn
from datetime import datetime
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class MediumMemory:
    """
    Manages summaries of recent conversations for medium-term recall.
    This is a simplified implementation focusing on recency.
    """
    def __init__(self):
        self.summaries: List[Dict[str, Any]] = []

    def add_summary(self, summary_text: str):
        """Adds a new conversation summary."""
        self.summaries.append({"timestamp": datetime.now().isoformat(), "summary": summary_text})

    def get_relevant_summaries(self, query_text: str) -> Optional[str]:
        """
        Retrieves the most recent summary.
        A more advanced implementation would use semantic search.
        """
        if not self.summaries:
            return None
        return self.summaries[-1]['summary']

class SlowMemory:
    """
    Manages the long-term knowledge base using a vector store for semantic search.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, device: str):
        self.vectors: List[torch.Tensor] = []
        self.texts: List[str] = []
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_embedding(self, text_chunk: str) -> torch.Tensor:
        """Generates a semantic embedding for a given text chunk."""
        input_ids = self.tokenizer.encode(text_chunk).ids

        # Determine the correct embedding layer and vocab size
        if hasattr(self.model, 'actor'): # RLHF ActorCritic model
            embedding_layer = self.model.actor.token_emb
        else: # Base BPETransformer model
            embedding_layer = self.model.token_emb

        num_embeddings = embedding_layer.num_embeddings
        tokens = [t for t in input_ids if t < num_embeddings]

        if not tokens:
            d_model = self.model.actor.d_model if hasattr(self.model, 'actor') else self.model.d_model
            return torch.zeros(1, d_model, device=self.device)

        # Generate embeddings in evaluation mode
        is_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            token_tensor = torch.tensor(tokens, device=self.device)
            token_embeddings = embedding_layer(token_tensor).mean(dim=0)

        if is_training:
            self.model.train()

        return token_embeddings.unsqueeze(0)

    def add_chunk(self, text_chunk: str):
        """Adds a text chunk and its embedding to the vector store."""
        embedding = self._get_embedding(text_chunk)
        self.vectors.append(embedding)
        self.texts.append(text_chunk)

    def search(self, query_text: str, k: int = 3) -> List[str]:
        """Performs k-NN search using cosine similarity."""
        if not self.vectors:
            return []

        query_embedding = self._get_embedding(query_text)
        # Ensure all vectors are on the same device and stacked correctly
        all_vectors = torch.stack(self.vectors).squeeze(1).to(self.device)

        # Normalize vectors for cosine similarity calculation
        query_norm = F.normalize(query_embedding, p=2, dim=1)
        db_norms = F.normalize(all_vectors, p=2, dim=1)

        similarities = torch.matmul(db_norms, query_norm.transpose(0, 1)).squeeze()

        # Handle cases with fewer vectors than k
        num_vectors = len(self.vectors)
        k = min(k, num_vectors)

        # Ensure similarities is at least 1-dimensional for torch.topk
        # If similarities is 0-d (scalar), unsqueeze it to make it 1-d
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        top_k_scores, top_k_indices = torch.topk(similarities, k=k)

        # Ensure top_k_indices is always a 1D tensor for iteration
        if top_k_indices.dim() == 0:
            top_k_indices = top_k_indices.unsqueeze(0)

        return [self.texts[i] for i in top_k_indices]

class MemoryManager:
    """
    Orchestrates the memory tiers to provide context for the LLM.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, device: str):
        self.medium_memory = MediumMemory()
        self.slow_memory = SlowMemory(model, tokenizer, device)

    def retrieve_context_for_prompt(self, prompt_text: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Retrieves relevant context from all memory tiers for a given prompt.
        """
        # 1. Get long-term context from Slow Memory
        slow_context_list = self.slow_memory.search(prompt_text)
        # 2. Get medium-term context from Medium Memory
        medium_context_str = self.medium_memory.get_relevant_summaries(prompt_text)

        # 3. Combine all contexts into a single string
        context_parts = []
        if slow_context_list:
            context_parts.append(f"[Long-Term Memory: {' | '.join(slow_context_list)}]")
        if medium_context_str:
            context_parts.append(f"[Medium-Term Memory: {medium_context_str}]")

        return " ".join(context_parts)

    def commit_to_memory(self, chat_history: List[Dict[str, str]]):
        """
        Processes a completed conversation and commits it to memory.
        """
        conversation_text = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])

        # 1. Add a summary to medium-term memory
        summary = f"Summary of conversation: {conversation_text[:150]}..."
        self.medium_memory.add_summary(summary)

        # 2. Chunk and add the full conversation to the slow memory vector store
        chunks = conversation_text.split('\n')
        for chunk in chunks:
            if chunk.strip() and len(chunk.split()) > 4: # Basic filtering
                self.slow_memory.add_chunk(chunk)

        print("Committed conversation to medium and slow memory.")
