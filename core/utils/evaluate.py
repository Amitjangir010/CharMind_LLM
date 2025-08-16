# ======================================================================================
# evaluate.py
# From-scratch evaluation suite for the CharMind LLM.
# Calculates perplexity and performs a simple QA task.
# ======================================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import json
import math

from tokenizer.tokenizer import load_tokenizer
from .model import BPETransformer
from .dataset import BPEDataset
from .config import (
    CONTEXT_LENGTH, DEVICE, TOKENIZER_PATH, CHECKPOINTS_BASE_DIR, VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, D_FF, DROPOUT
)

def calculate_perplexity(model, dataloader):
    """
    Calculates the perplexity of a language model on a given dataset.

    Perplexity is a standard metric for evaluating language models. It measures
    how well a probability distribution predicts a sample. A lower perplexity
    indicates that the model is less "surprised" by the test data and thus
    has a better understanding of the language. It is calculated as the
    exponentiation of the average cross-entropy loss.

    Args:
        model (nn.Module): The transformer model to evaluate.
        dataloader (DataLoader): The DataLoader containing the test data.

    Returns:
        float: The calculated perplexity score.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    print("Calculating perplexity...")
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="Perplexity Calculation"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, loss = model(xb, targets=yb)

            # The loss is already the average cross-entropy for the batch
            total_loss += loss.item() * yb.numel()
            total_tokens += yb.numel()

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def evaluate_qa(model, tokenizer, qa_pairs):
    """
    Performs a simple Question-Answering (QA) evaluation.

    This function tests the model's ability to perform a basic reasoning and
    generation task. It feeds a question to the model and checks if the
    generated answer contains the expected keywords. This provides a more
    task-oriented measure of performance than statistical metrics alone.

    Args:
        model (nn.Module): The transformer model to evaluate.
        tokenizer: The tokenizer instance.
        qa_pairs (list of dict): A list of dictionaries, where each dict has
            a "question" and "answer" key.

    Returns:
        float: The accuracy of the model on the QA task, as a percentage.
    """
    model.eval()
    correct = 0
    total = len(qa_pairs)

    print("\nPerforming Question-Answering evaluation...")
    for item in tqdm(qa_pairs, desc="QA Evaluation"):
        question = item["question"]
        expected_answer = item["answer"]

        prompt = f"User: {question}\nLLM:"
        input_ids = tokenizer.encode(prompt).ids
        context_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            output_tensor = model.generate(
                context_tensor,
                max_new_tokens=50, # Keep it short for QA
                eos_token_id=tokenizer.token_to_id("[EOS]")
            )

        output_ids = output_tensor[0, len(input_ids):].tolist()
        generated_answer = tokenizer.decode(output_ids)

        # Simple check for keyword presence
        if expected_answer.lower() in generated_answer.lower():
            correct += 1

        print(f"Q: {question}")
        print(f"A: {generated_answer.strip()}")
        print("-" * 20)

    accuracy = (correct / total) * 100
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluation suite for CharMind LLM.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINTS_BASE_DIR, 'pretrain', 'bpe_best.pt'),
        help="Path to the model checkpoint to evaluate."
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # --- 1. Load Model and Tokenizer ---
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = BPETransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD,
        d_ff=D_FF, context_length=CONTEXT_LENGTH, dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    print(f"Loaded model from {args.checkpoint}")

    # --- 2. Prepare Datasets ---
    # For perplexity, we'll use a dummy dataset for demonstration.
    # In a real scenario, this would be a held-out test set.
    dummy_text_data = [
        "This is a test sentence for calculating perplexity.",
        "A good language model should not be surprised by common phrases.",
        "The quick brown fox jumps over the lazy dog."
    ]
    perplexity_dataset = BPEDataset(dummy_text_data, tokenizer, CONTEXT_LENGTH)
    perplexity_loader = DataLoader(perplexity_dataset, batch_size=4)

    # For QA, we'll use a small, hardcoded set of questions.
    qa_pairs = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
        {"question": "What is the powerhouse of the cell?", "answer": "mitochondria"}
    ]

    # --- 3. Run Evaluations ---
    perplexity = calculate_perplexity(model, perplexity_loader)
    qa_accuracy = evaluate_qa(model, tokenizer, qa_pairs)

    # --- 4. Print Report ---
    print("\n--- Evaluation Report ---")
    print(f"Model: {args.checkpoint}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"QA Accuracy: {qa_accuracy:.2f}%")
    print("-------------------------")

if __name__ == "__main__":
    main()
