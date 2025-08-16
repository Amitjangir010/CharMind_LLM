# ======================================================================================
# launch.py
#
# A launcher script for multi-GPU training using `torch.distributed`.
# This script is the designated entry point for initiating any training process.
#
# It automatically detects the number of available GPUs and spawns a separate
# process for each one, handling the setup for distributed training. If only one
# or zero GPUs are found, it gracefully falls back to a single-process mode.
#
# Usage:
#   python launch.py
#
# ======================================================================================

import torch
import torch.multiprocessing as mp
import os
import sys

# Add project root to path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pretrain.train import main as train_main

def main():
    """
    Main function to set up and launch the distributed training processes.
    """
    # Determine the number of GPUs available
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")

    if world_size > 1:
        print("Starting distributed training...")
        # Use torch.multiprocessing.spawn to launch 'world_size' processes,
        # each running the 'train_main' function.
        # The 'rank' of the process (from 0 to world_size-1) is passed as the first argument.
        mp.spawn(train_main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    elif world_size == 1:
        print("Only one GPU found. Running in single-GPU mode.")
        # If only one GPU, just call the main training function directly
        train_main(rank=0, world_size=1)
    else:
        print("No GPUs found. Running on CPU.")
        # If no GPUs, run in single-process CPU mode
        train_main(rank=0, world_size=1)


if __name__ == "__main__":
    # Note: It's crucial to set the start method to 'spawn' for CUDA compatibility
    # especially on systems other than Linux.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    main()
