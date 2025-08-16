# generate_preference_data.py
# Generate Preference_data.json from Rating_feedback.json for RLHF reward model

import json
import os
from collections import defaultdict
from core.utils.config import RLHF_DATA_DIR # Import the data directory from config

RATING_FILE = os.path.join(RLHF_DATA_DIR, 'Rating_feedback.json')
PREF_FILE = os.path.join(RLHF_DATA_DIR, 'Preference_data.json')

# Load all feedback
with open(RATING_FILE, 'r', encoding='utf-8') as f:
    feedback = json.load(f)

# Group replies by prompt
prompt_to_replies = defaultdict(list)
for entry in feedback:
    prompt = entry['prompt']
    reply = entry['reply']
    rating = entry['rating']
    prompt_to_replies[prompt].append({'reply': reply, 'rating': rating})

preference_pairs = []
for prompt, replies in prompt_to_replies.items():
    # Sort replies by rating (descending)
    sorted_replies = sorted(replies, key=lambda x: x['rating'], reverse=True)
    # Pair highest-rated with lowest-rated (if at least 2 unique ratings)
    if len(sorted_replies) >= 2 and sorted_replies[0]['rating'] > sorted_replies[-1]['rating']:
        chosen = sorted_replies[0]['reply']
        rejected = sorted_replies[-1]['reply']
        preference_pairs.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })

# Save preference pairs
with open(PREF_FILE, 'w', encoding='utf-8') as f:
    json.dump(preference_pairs, f, ensure_ascii=False, indent=2)

print(f"Generated {len(preference_pairs)} preference pairs in {PREF_FILE}") 