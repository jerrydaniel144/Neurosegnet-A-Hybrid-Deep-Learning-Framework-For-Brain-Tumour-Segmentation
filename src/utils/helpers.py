import os
import torch
import numpy as np
import random
import time
from pathlib import Path


def set_seed(seed=42):
    """
    Set all random seeds for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint safely (state_dict only).

    Args:
        model (torch.nn.Module): Model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epoch (int): Current epoch.
        loss (float): Current loss value.
        path (str): Path to save the checkpoint.
    """
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path)
    print(f" Checkpoint saved: {path}")


def get_timestamp():
    """
    Get formatted current time for logging and filenames.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def log_model_size(model):
    """
    Print total number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model has {total_params:,} trainable parameters.")


def count_class_voxels(mask_tensor, class_labels):
    """
    Count number of voxels per class in a 3D segmentation label tensor.

    Args:
        mask_tensor (torch.Tensor): Tensor of shape [B, H, W, D] or [B, C, H, W, D].
        class_labels (list[int]): List of class IDs to count.

    Returns:
        dict: Mapping from class label to voxel count.
    """
    if mask_tensor.ndim == 5:
        # Convert one-hot to argmax if needed
        mask_tensor = torch.argmax(mask_tensor, dim=1)

    counts = {}
    for cls in class_labels:
        counts[cls] = torch.sum(mask_tensor == cls).item()
    return counts