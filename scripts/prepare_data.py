# data/prepare_data.py
# This script consolidates and processes all raw data sources into
# clean, final datasets for training, fine-tuning, and RLHF.

import os
import json
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
import sys

# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from core.utils.config import (
    PRETRAIN_DATA_DIR, RLHF_DATA_DIR,
    PROCESSED_TRAINING_FILE, PROCESSED_PREFERENCE_FILE
)

# Define paths for raw data and final output
# Removed RAW_DATA_DIR, FINETUNE_DIR, FEEDBACK_DIR, CORPUS_DIR as they are now in config
# Replaced with imports from config.py

# Output files are also now directly from config.py
TRAINING_OUTPUT_FILE = PROCESSED_TRAINING_FILE
PREFERENCE_OUTPUT_FILE = PROCESSED_PREFERENCE_FILE

def load_json_file(path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file and returns its content, handles errors."""
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {path}: {e}")
        return []

def save_jsonl_file(data: List[Dict[str, Any]], path: str):
    """Saves a list of dictionaries to a JSONL file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Successfully saved {len(data)} records to {path}")
    except IOError as e:
        print(f"Error writing to {path}: {e}")

def process_general_training_data():
    """
    Consolidates data for general-purpose training and fine-tuning.
    Sources:
    - combined_memory.json: This is the main source now.
    """
    print("--- Processing General Training Data ---")
    
    # Directly use combined_memory.json
    combined_memory_path = os.path.join(PRETRAIN_DATA_DIR, 'combined_memory.json')
    combined_data = load_json_file(combined_memory_path)
    
    consolidated_texts = []
    
    for item in combined_data:
        if isinstance(item, dict) and 'text' in item and item['text'].strip():
            consolidated_texts.append({'text': item['text'].strip()})

    if consolidated_texts:
        save_jsonl_file(consolidated_texts, TRAINING_OUTPUT_FILE)
    else:
        print("No general training data was processed.")
        
    return [combined_memory_path] # Return only the file that was actually processed


def process_preference_data():
    """
    Consolidates data into a preference format for RLHF (DPO).
    The final format is: {"prompt": str, "chosen": str, "rejected": str}
    Sources:
    - Rating_feedback.json: Directly use this for preference pairs.
    """
    print("\n--- Processing Preference Data for RLHF ---")
    
    # Directly use Rating_feedback.json
    rating_path = os.path.join(RLHF_DATA_DIR, 'Rating_feedback.json')
    rating_feedback = load_json_file(rating_path)
    
    final_preferences = []
    
    prompt_to_replies = defaultdict(list)
    for entry in rating_feedback:
        if 'prompt' in entry and 'reply' in entry and 'rating' in entry:
            prompt_to_replies[entry['prompt']].append({'reply': entry['reply'], 'rating': entry['rating']})

    for prompt, replies in prompt_to_replies.items():
        sorted_replies = sorted(replies, key=lambda x: x['rating'], reverse=True)
        if len(sorted_replies) >= 2 and sorted_replies[0]['rating'] > sorted_replies[-1]['rating']:
            chosen = sorted_replies[0]['reply']
            rejected = sorted_replies[-1]['reply']
            final_preferences.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })

    if final_preferences:
        # Remove duplicates
        df = pd.DataFrame(final_preferences)
        df.drop_duplicates(subset=['prompt', 'chosen', 'rejected'], inplace=True)
        unique_preferences = df.to_dict('records')
        save_jsonl_file(unique_preferences, PREFERENCE_OUTPUT_FILE)
    else:
        print("No preference data was processed.")
        
    return [rating_path] # Return only the file that was actually processed

def cleanup_old_files(files_to_remove: List[str]):
    """Deletes the old raw data files after processing."""
    print("\n--- Cleaning up old data files ---")
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed old file: {f}")
            except OSError as e:
                print(f"Error removing file {f}: {e}")
        else:
            print(f"File not found, skipping cleanup: {f}")


if __name__ == "__main__":
    print("Starting data consolidation process...")

    # Ensure output directories exist - using config paths
    os.makedirs(PRETRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(RLHF_DATA_DIR, exist_ok=True)

    # Process both types of data
    general_files = process_general_training_data()
    preference_files = process_preference_data()

    # Cleanup old files
    cleanup_old_files(general_files + preference_files)

    print("\nData consolidation finished.")
    print(f"General training data saved to: {TRAINING_OUTPUT_FILE}")
    print(f"Preference data for RLHF saved to: {PREFERENCE_OUTPUT_FILE}")
