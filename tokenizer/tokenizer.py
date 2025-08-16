# src/tokenizer.py
# This module handles loading the trained BPE tokenizer.

import os
from tokenizers import Tokenizer
# config.py se EOS_TOKEN import karne ke liye (jab aap config.py bana lenge)
# import config 

def load_tokenizer(tokenizer_path):
    """
    Loads a tokenizer from a file.

    Args:
        tokenizer_path (str): The path to the tokenizer JSON file.

    Returns:
        A Tokenizer instance.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file not found at {tokenizer_path}. "
            "Please run `python -m src.train_tokenizer` to train and create the tokenizer first."
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

if __name__ == "__main__":
    # Example of how to load and use the tokenizer
    
    # Maan lete hain ki aapne config.py mein EOS_TOKEN define kiya hai
    # Agar abhi nahi kiya hai, to is line ko comment kar dein aur neeche [EOS] hardcode kar lein
    # EOS_TOKEN = config.EOS_TOKEN
    EOS_TOKEN = "[EOS]" # Abhi ke liye yahan likh dete hain

    # This path should match the output of the train_tokenizer.py script
    TOKENIZER_FILE = os.path.join(os.path.dirname(__file__), '../saved_models/tokenizer/bpe_tokenizer.json')

    print(f"Attempting to load tokenizer from: {TOKENIZER_FILE}")

    try:
        # Load the tokenizer
        my_tokenizer = load_tokenizer(TOKENIZER_FILE)

        print("\nTokenizer loaded successfully!")
        print(f"Vocabulary size: {my_tokenizer.get_vocab_size()}")

        # === NAYA CHECK ===
        # Ab hum check karenge ki EOS token vocabulary mein hai ya nahi
        vocab = my_tokenizer.get_vocab()
        if EOS_TOKEN in vocab:
            print(f"✅ Success: '{EOS_TOKEN}' token found in vocabulary with ID: {vocab[EOS_TOKEN]}")
        else:
            print(f"⚠️ Warning: '{EOS_TOKEN}' token NOT found in vocabulary.")
            print("   You need to train the tokenizer with this special token first.")
        # === CHECK KHATAM ===


        # Example usage
        text = f"This is an example sentence.{EOS_TOKEN}" # EOS token ke saath text
        print(f"\nOriginal text: '{text}'")

        # Encode
        encoded = my_tokenizer.encode(text)
        print(f"Encoded IDs: {encoded.ids}")
        print(f"Encoded Tokens: {encoded.tokens}")

        # Decode
        # `skip_special_tokens=False` taaki humein special tokens dikhein
        decoded_text = my_tokenizer.decode(encoded.ids, skip_special_tokens=False)
        print(f"Decoded text (with special tokens): '{decoded_text}'")

        # `skip_special_tokens=True` special tokens ko hata dega
        decoded_text_skipped = my_tokenizer.decode(encoded.ids, skip_special_tokens=True)
        print(f"Decoded text (without special tokens): '{decoded_text_skipped}'")


    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure you have run the tokenizer training script first:")
        print("`python -m src.train_tokenizer`")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")