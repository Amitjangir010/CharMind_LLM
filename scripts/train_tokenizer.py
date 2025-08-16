# src/train_tokenizer.py
# This script trains a professional Byte-Pair Encoding (BPE) tokenizer on your data corpus.
# It now AUTOMATICALLY updates the VOCAB_SIZE in your config.py file.

import os
import json
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
# Removed glob import as it's no longer used
# Removed sys import as we are hardcoding paths now
# Removed core.utils.config import as we are hardcoding paths now

# --- Configuration ---
VOCAB_SIZE = 32000 # Ek acchi, professional vocabulary size

# Hardcoded paths relative to the script location for robustness
# These paths are relative to 'scripts/train_tokenizer.py'
OUTPUT_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'tokenizer', 'bpe_tokenizer.json')
DATA_FILE_FOR_TOKENIZER = os.path.join(os.path.dirname(__file__), '..', 'core', 'pretrain', 'data', 'processed_training_corpus.jsonl')
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'core', 'utils', 'config.py') # Path to config for VOCAB_SIZE update

def update_config_file(new_vocab_size):
    """
    Yeh function config.py file ko automatically update karta hai.
    """
    print(f"\nAttempting to automatically update config.py...")
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        updated_lines = []
        found = False
        for line in lines:
            if line.strip().startswith("VOCAB_SIZE"):
                updated_lines.append(f"VOCAB_SIZE = {new_vocab_size} # Automatically updated by train_tokenizer.py\n")
                found = True
            else:
                updated_lines.append(line)
        
        if not found:
            print(f"Warning: 'VOCAB_SIZE' not found in {CONFIG_FILE_PATH}. Please update it manually.")
            return

        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        print(f"✅ Success! config.py has been updated with VOCAB_SIZE = {new_vocab_size}")

    except Exception as e:
        print(f"❌ Error updating config.py automatically: {e}")
        print(f"Please update VOCAB_SIZE to {new_vocab_size} in config.py manually.")

def extract_text_from_json_files(file_paths):
    """
    Yeh function JSON files se sirf 'text' field ki value nikaalta hai.
    """
    all_texts = []
    print("Extracting clean text from JSON files...")
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            all_texts.append(item['text'])
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not read or parse {file_path}. Skipping. Error: {e}")
    return all_texts

def train_bpe_tokenizer():
    """
    Trains and saves a BPE tokenizer.
    """
    print("Starting BPE tokenizer training...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # === YAHAN PAR BADLAAV KIYA GAYA HAI ===
    # Humne User:, LLM:, aur newline character (\n) ko bhi special tokens bana diya hai.
    special_tokens = [
        "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]",
        "User:", "LLM:", "\n"
    ]
    
    print(f"Training with special tokens: {special_tokens}")
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    # Directly load text from processed_training_corpus.jsonl
    # Note: The tokenizer expects a list of strings, one string per document/text example.
    # processed_training_corpus.jsonl is a JSONL file, so we need to read each line as a separate document.
    
    if not os.path.exists(DATA_FILE_FOR_TOKENIZER):
        print(f"Error: Processed data file not found at {DATA_FILE_FOR_TOKENIZER}. Cannot train tokenizer. Please run prepare_data.py first.")
        return

    all_texts = []
    try:
        with open(DATA_FILE_FOR_TOKENIZER, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_entry = json.loads(line)
                    if 'text' in data_entry and data_entry['text'].strip():
                        all_texts.append(data_entry['text'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line in {DATA_FILE_FOR_TOKENIZER}: {line.strip()}. Error: {e}")
                    continue
    except Exception as e:
        print(f"Error reading {DATA_FILE_FOR_TOKENIZER}. Error: {e}")
        return

    if not all_texts:
        print("Error: No text could be extracted from the data file. Nothing to train on.")
        return

    print(f"\nTraining tokenizer on {len(all_texts)} extracted texts with a target vocab size of {VOCAB_SIZE}...")
    tokenizer.train_from_iterator(all_texts, trainer=trainer)
    print("Training complete.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(OUTPUT_TOKENIZER_PATH)
    print(f"\nTokenizer successfully trained and saved to: {OUTPUT_TOKENIZER_PATH}")

    final_vocab_size = tokenizer.get_vocab_size()
    print(f"\nFinal vocabulary size: {final_vocab_size}")
    
    update_config_file(final_vocab_size)

    vocab = tokenizer.get_vocab()
    for token in ["User:", "LLM:", "[EOS]", "\n"]:
        if token in vocab:
            print(f"✅ Success! '{token}' token is in the vocabulary with ID: {vocab[token]}")
        else:
            print(f"❌ Error! '{token}' token was not added to the vocabulary.")


if __name__ == "__main__":
    train_bpe_tokenizer()
