import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score
)

def evaluate(model, val_loader, loss_fn, device, config=None):
    model.eval()
    val_loss = 0.0

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean_batch")
    sigmoid = Activations(sigmoid=True)

    precision_vals = []
    recall_vals = []
    f1_vals = []
    specificity_vals = []
    accuracy_vals = []

    # Optional smoothing (for soft labels)
    smooth = 1.0
    use_soft_labels = False
    if config:
        smooth = config.get("loss", {}).get("smooth_labels", 1.0)
        use_soft_labels = config.get("use_soft_labels", False)

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            if use_soft_labels:
                labels = labels * smooth + (1.0 - labels) * (1.0 - smooth)
            else:
                labels = (labels > 0.5).float()

            outputs = sigmoid(model(inputs))
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            # Binarize predictions
            preds_bin = (outputs > 0.5).float()
            labels_bin = (labels > 0.5).float()

            # Resize labels if needed
            if preds_bin.shape != labels_bin.shape:
                labels_bin = torch.nn.functional.interpolate(
                    labels_bin.float(), size=preds_bin.shape[2:], mode="nearest"
                )

            # Skip if empty
            if torch.sum(preds_bin) == 0 or torch.sum(labels_bin) == 0:
                continue

            # MONAI Metrics
            dice_metric(preds_bin, labels_bin)
            hd95_metric(preds_bin, labels_bin)

            # Sklearn metrics per case
            for p, l in zip(preds_bin, labels_bin):
                p_np = p.cpu().numpy().astype(int).ravel()
                l_np = l.cpu().numpy().astype(int).ravel()

                accuracy_vals.append(accuracy_score(l_np, p_np))
                precision_vals.append(precision_score(l_np, p_np, zero_division=0))
                recall_vals.append(recall_score(l_np, p_np, zero_division=0))
                f1_vals.append(f1_score(l_np, p_np, zero_division=0))

                cm = confusion_matrix(l_np, p_np, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificity_vals.append(spec)

    metrics = {
        "val_loss": val_loss / len(val_loader),
        "dice": dice_metric.aggregate().item(),
        "hd95": hd95_metric.aggregate().item(),
        "precision": np.mean(precision_vals),
        "recall": np.mean(recall_vals),
        "specificity": np.mean(specificity_vals),
        "f1_score": np.mean(f1_vals),
        "accuracy": np.mean(accuracy_vals),
    }

    dice_metric.reset()
    hd95_metric.reset()
    return metrics
