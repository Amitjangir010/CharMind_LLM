# src/dataset.py
# Corrected Dataset class for the BPE-based model.
# Now correctly handles each conversational pair as an independent unit.

import torch
from torch.utils.data import Dataset
import os
from tokenizer.tokenizer import load_tokenizer

class BPEDataset(Dataset):
    def __init__(self, texts, tokenizer, context_length):
        """
        Initializes the dataset for a BPE tokenizer.
        Args:
            texts (list of str): A list of text strings, where each string is a complete conversational turn.
            tokenizer (tokenizers.Tokenizer): The BPE tokenizer instance.
            context_length (int): The context length for the model.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.data = []

        print("Processing texts into training samples...")
        # Har text entry (User/LLM pair) ko alag se process karein
        for text in texts:
            if not text:
                continue

            # Har pair ko alag se encode karein
            encoded_text = self.tokenizer.encode(text).ids

            # Sirf ussi pair ke andar se input/target ke jode (pairs) banayein
            # Handle texts shorter than context_length by padding
            if len(encoded_text) <= self.context_length:
                # Pad the sequence if it's shorter than context_length
                pad_token_id = self.tokenizer.token_to_id("[PAD]")
                if pad_token_id is None:
                    raise ValueError("PAD token not found in tokenizer vocabulary. Please train tokenizer with [PAD] special token.")
                # Pad the sequence if it's shorter than context_length
                padded_encoded_text = encoded_text + [pad_token_id] * (self.context_length - len(encoded_text))
                # Ensure x and y are always context_length long
                x = torch.tensor(padded_encoded_text[:self.context_length], dtype=torch.long)
                y = torch.tensor(padded_encoded_text[1:self.context_length+1] + [pad_token_id], dtype=torch.long) # Shifted by one for target
                self.data.append((x, y))
            else:
                # For longer texts, create multiple samples
                for i in range(len(encoded_text) - self.context_length + 1):
                    x = torch.tensor(encoded_text[i : i + self.context_length], dtype=torch.long)
                    # y should be the next token for each token in x
                    # If it's the last sequence, pad y to context_length
                    if i + self.context_length + 1 > len(encoded_text):
                        y = torch.tensor(encoded_text[i + 1 : i + self.context_length + 1] + [pad_token_id] * (i + self.context_length + 1 - len(encoded_text)), dtype=torch.long)
                    else:
                        y = torch.tensor(encoded_text[i + 1 : i + self.context_length + 1], dtype=torch.long)
                    self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # Example of how to use the corrected BPEDataset
    from .config import TOKENIZER_PATH

    print("--- Testing Corrected BPEDataset ---")

    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: BPE Tokenizer not found at {TOKENIZER_PATH}")
        print("Please train the tokenizer first by running `python -m src.train_tokenizer`")
    else:
        # 1. Load the BPE tokenizer
        bpe_tokenizer = load_tokenizer(TOKENIZER_PATH)
        print(f"Loaded BPE tokenizer with vocab size: {bpe_tokenizer.get_vocab_size()}")

        # 2. Define sample data in the correct format
        sample_texts = [
            "User: Hello, how are you?\nLLM: I am doing great, thanks for asking![EOS]",
            "User: What is a Large Language Model?\nLLM: A Large Language Model is a type of AI trained on vast amounts of text data to understand and generate human-like language.[EOS]"
        ]

        # 3. Create the dataset instance
        dataset = BPEDataset(texts=sample_texts, tokenizer=bpe_tokenizer, context_length=16)

        # 4. Print dataset info
        print(f"Dataset created. Number of samples: {len(dataset)}")
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"\nSample 0:")
            print(f"  Input tensor (x): {x}")
            print(f"  Target tensor (y): {y}")

            # Decode for verification
            decoded_x = bpe_tokenizer.decode(x.tolist())
            decoded_y = bpe_tokenizer.decode(y.tolist())

            print(f"\nDecoded Input:  '{decoded_x}'")
            print(f"Decoded Target: '{decoded_y}'")
            
            # Check a later sample to ensure it doesn't cross the boundary
            if len(dataset) > 20:
                 x2, y2 = dataset[20]
                 decoded_x2 = bpe_tokenizer.decode(x2.tolist())
                 print(f"\nDecoded Input from a later sample: '{decoded_x2}'")
                 # You will notice this sample is still from the first sentence, not mixed with the second.
        else:
            print("Dataset is empty. The total text length might be less than the context length.")
