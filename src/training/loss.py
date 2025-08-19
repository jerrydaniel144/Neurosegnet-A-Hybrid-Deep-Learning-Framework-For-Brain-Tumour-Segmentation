import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-5, dice_weight=1.0, focal_weight=1.0, use_soft_labels=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.use_soft_labels = use_soft_labels
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Resize targets to match prediction
        if inputs.shape != targets.shape:
            targets = F.interpolate(targets, size=inputs.shape[2:], mode='trilinear', align_corners=False)

        # Optional hard binarization (only if soft labels are disabled)
        if not self.use_soft_labels:
            targets = (targets > 0.5).float()

        # Focal Loss: uses BCE base
        bce_loss = self.bce_loss_fn(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # Dice Loss
        probs = torch.sigmoid(inputs)
        probs = probs.clamp(min=1e-4, max=1.0 - 1e-4)  # Prevents NaNs
        dims = tuple(range(2, inputs.ndim))  # Supports both 2D and 3D

        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
