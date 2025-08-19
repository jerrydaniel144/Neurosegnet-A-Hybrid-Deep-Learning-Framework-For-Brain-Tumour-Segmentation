import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import json
import sys
import pandas as pd

# Add project root to import local modules
sys.path.append("C:/Users/user/NeuroSegNet/")
from src.training.loss import DiceFocalLoss
from src.training.validation import evaluate

def train(model, train_loader, val_loader, optimizer, scheduler, config, device):
    torch.manual_seed(config["seed"])
    model = model.to(device)

    # Optimizer setup
    opt_cfg = config["optimizer"]
    opt_type = opt_cfg.get("type", "AdamW").lower()
    
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    elif opt_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")

    # Scheduler setup 
    sch_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sch_cfg.get("mode", "min"),
        factor=sch_cfg.get("factor", 0.5),
        patience=sch_cfg.get("patience", 3),
        min_lr=sch_cfg.get("min_lr", 1e-6),
        verbose=True
    )

    # Loss Function 
    loss_cfg = config.get("loss", {})
    loss_fn = DiceFocalLoss(
        alpha=loss_cfg.get("focal_alpha", 0.8),
        gamma=loss_cfg.get("focal_gamma", 2.0),
        smooth=loss_cfg.get("smooth_labels", 1e-5),
        dice_weight=loss_cfg.get("lambda_dice", 1.0),
        focal_weight=loss_cfg.get("lambda_focal", 1.0),
        use_soft_labels=config.get("use_soft_labels", True),
    )

    # Checkpointing 
    checkpoint_path = Path(config.get("checkpoint_path", "best_model.pth"))
    best_loss_file = checkpoint_path.with_suffix(".val_loss.json")
    metrics_log_file = checkpoint_path.with_suffix(".metrics.csv")
    best_val_loss = float("inf")
    all_metrics = []

    if checkpoint_path.exists() and best_loss_file.exists():
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        with open(best_loss_file, "r") as f:
            best_val_loss = float(json.load(f)["best_val_loss"])
        print(f"Resumed from: {checkpoint_path}")
        print(f"Starting from previous best validation loss: {best_val_loss:.4f}")
    else:
        print("Training from scratch.")

    patience = config["early_stopping"]["patience"]
    monitor_metric = config["early_stopping"]["monitor"]
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]")

        for batch in loop:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            if torch.isnan(loss):
                print("Skipping NaN loss batch")
                continue  # Skip the batch

            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=float(loss.detach()))

        # Validation
        metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = metrics["val_loss"]
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        metrics["epoch"] = epoch + 1
        all_metrics.append(metrics)

        if val_loss < best_val_loss:
            print(f"{monitor_metric} improved. Saving best model.")
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(checkpoint_path))
            with open(best_loss_file, "w") as f:
                json.dump({"best_val_loss": best_val_loss}, f)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        scheduler.step(val_loss)

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    pd.DataFrame(all_metrics).to_csv(metrics_log_file, index=False)
    print(f"Saved training metrics to {metrics_log_file}")
