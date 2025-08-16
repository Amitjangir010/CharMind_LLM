# ======================================================================================
# config.py
# Central configuration file for CharMind_LLM project.
# All project-wide parameters are managed from here.
# ======================================================================================

import os
import torch

# Determine the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --------------------------------------------------------------------------------------
# --- Path and Directory Settings ---
# --------------------------------------------------------------------------------------
# Base directories for each training type
PRETRAIN_ROOT_DIR = os.path.join(PROJECT_ROOT, 'core', 'pretrain')
FINETUNE_ROOT_DIR = os.path.join(PROJECT_ROOT, 'core', 'finetune')
RLHF_ROOT_DIR = os.path.join(PROJECT_ROOT, 'core', 'rlhf')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
SAVED_MODELS_GLOBAL_DIR = os.path.join(PROJECT_ROOT, 'saved_models') # For tokenizer
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs') # Global logs directory
CHECKPOINTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'saved_models') # Base directory for all checkpoints

# Data directories
PRETRAIN_DATA_DIR = os.path.join(PRETRAIN_ROOT_DIR, 'data')
FINETUNE_DATA_DIR = os.path.join(FINETUNE_ROOT_DIR, 'data')
RLHF_DATA_DIR = os.path.join(RLHF_ROOT_DIR, 'data')

# Saved Models directories (checkpoints)
PRETRAIN_CHECKPOINT_DIR = os.path.join(PRETRAIN_ROOT_DIR, 'saved_models')
FINETUNE_CHECKPOINT_DIR = os.path.join(FINETUNE_ROOT_DIR, 'saved_models')
RLHF_CHECKPOINT_DIR = os.path.join(RLHF_ROOT_DIR, 'saved_models', 'rlhf_model') 

# Logs directories
PRETRAIN_LOGS_DIR = os.path.join(PRETRAIN_ROOT_DIR, 'logs')
FINETUNE_LOGS_DIR = os.path.join(FINETUNE_ROOT_DIR, 'logs')
RLHF_LOGS_DIR = os.path.join(RLHF_ROOT_DIR, 'logs')

# Specific file paths
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'tokenizer', 'bpe_tokenizer.json')
PROCESSED_TRAINING_FILE = os.path.join(PRETRAIN_DATA_DIR, "processed_training_corpus.jsonl")
PROCESSED_PREFERENCE_FILE = os.path.join(RLHF_DATA_DIR, "processed_preference_data.jsonl")
REWARD_MODEL_DIR = os.path.join(RLHF_ROOT_DIR, 'saved_models', 'reward_model')

# Ensure all directories exist
os.makedirs(PRETRAIN_DATA_DIR, exist_ok=True)
os.makedirs(FINETUNE_DATA_DIR, exist_ok=True)
os.makedirs(RLHF_DATA_DIR, exist_ok=True)
os.makedirs(PRETRAIN_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINETUNE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(RLHF_ROOT_DIR, 'saved_models', 'rlhf_model'), exist_ok=True)
os.makedirs(PRETRAIN_LOGS_DIR, exist_ok=True)
os.makedirs(FINETUNE_LOGS_DIR, exist_ok=True)
os.makedirs(RLHF_LOGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True) # Ensure the global logs directory exists
os.makedirs(os.path.join(SAVED_MODELS_GLOBAL_DIR, 'tokenizer'), exist_ok=True)
os.makedirs(os.path.join(RLHF_ROOT_DIR, 'saved_models', 'reward_model'), exist_ok=True)

# --- Special Tokens ---
EOS_TOKEN = "[EOS]"


# --------------------------------------------------------------------------------------
# --- Hardware & Performance Settings ---
# --------------------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() and os.environ.get('USE_CPU') != '1' else 'cpu'
TORCH_COMPILE = True  # Use PyTorch 2.0 compiler for a significant speedup.
GRADIENT_ACCUMULATION_STEPS = 4  # Increase effective batch size. Set to 1 to disable.


# --------------------------------------------------------------------------------------
# --- Base Model Architecture Settings ---
# --------------------------------------------------------------------------------------
VOCAB_SIZE = 117 # Automatically updated by train_tokenizer.py
D_MODEL = 128
N_LAYER = 8
N_HEAD = 8
D_FF = 512
CONTEXT_LENGTH = 2048
DROPOUT = 0.10


# --------------------------------------------------------------------------------------
# --- LoRA / QLoRA Settings (for PEFT from Scratch) ---
# --------------------------------------------------------------------------------------
USE_LORA = True # Set to True to enable LoRA for fine-tuning

# LORA_R: The rank of the low-rank matrices. A smaller 'r' means fewer parameters.
LORA_R = 8
# LORA_ALPHA: The scaling factor for the LoRA matrices.
LORA_ALPHA = 16
# LORA_DROPOUT: Dropout probability for LoRA layers.
LORA_DROPOUT = 0.05
# USE_QLORA: Placeholder for enabling from-scratch QLoRA logic.
USE_QLORA = False


# --------------------------------------------------------------------------------------
# --- Training Settings ---
# --------------------------------------------------------------------------------------
EPOCHS = 2 # Reduced for quick testing
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


# --------------------------------------------------------------------------------------
# --- Generation (Inference) Settings ---
# --------------------------------------------------------------------------------------
TEMPERATURE = 0.8
TOP_P = 0.92
MAX_NEW_TOKENS = 256


# --------------------------------------------------------------------------------------
# --- RLHF and PPO Settings ---
# --------------------------------------------------------------------------------------
# These will be used for the from-scratch PPO implementation.
REWARD_MODEL_EPOCHS = 5
PPO_EPOCHS = 4
PPO_CLIP_EPSILON = 0.2  # Epsilon for clipping in PPO loss
PPO_LEARNING_RATE = 1e-5
REWARD_MODEL_LR = 1e-5
REWARD_MODEL_BATCH_SIZE = 4
BASE_MODEL_CHECKPOINT = os.path.join(PRETRAIN_CHECKPOINT_DIR, '1_epoch_50') # Default base model for RLHF


# --------------------------------------------------------------------------------------
# --- Tokenizer and UI Settings ---
# --------------------------------------------------------------------------------------

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "[EOS]"
# Placeholder for a future UI-specific setting
UI_THEME = "dark"