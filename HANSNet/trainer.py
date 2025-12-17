"""
trainer.py - Training utilities for HANS-Net liver segmentation

This module provides:
    - dice_loss: Dice coefficient loss function
    - combined_loss: BCE + Dice combined loss
    - train_one_epoch: Single epoch training loop
    - validate_one_epoch: Single epoch validation loop
    - train_pipeline: Complete training workflow
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm

from dataloader import build_dataloaders
from model import HANSNet


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_checkpoint(model, optimizer, epoch: int, path: str) -> None:
    """
    Save model + optimizer state + epoch number to a checkpoint file.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch number
        path: Path to save checkpoint file
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    device: str = "cuda",
) -> int:
    """
    Load model (and optionally optimizer) state from a checkpoint file.
    
    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        The stored epoch number (or 0 if not present)
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state"])
    
    # Load optimizer state if provided and available
    if optimizer is not None and "optim_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])
    
    # Return stored epoch or 0
    return checkpoint.get("epoch", 0)


# =============================================================================
# Loss Functions
# =============================================================================

def dice_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute Dice loss from logits.
    
    Args:
        pred_logits: [B, 1, H, W] raw logits BEFORE sigmoid
        target_mask: [B, 1, H, W] binary mask {0, 1}
        
    Returns:
        Scalar Dice loss averaged over batch
    """
    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred_logits)
    
    # Flatten spatial dimensions for each sample
    pred_flat = pred_probs.view(pred_probs.size(0), -1)
    target_flat = target_mask.view(target_mask.size(0), -1)
    
    # Compute Dice coefficient
    numerator = 2.0 * (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-8
    
    dice = 1.0 - numerator / denominator
    
    # Average over batch
    return dice.mean()


def combined_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Combined BCE + Dice loss.
    
    Args:
        pred_logits: [B, 1, H, W] raw logits BEFORE sigmoid
        target_mask: [B, 1, H, W] binary mask {0, 1}
        
    Returns:
        Scalar combined loss
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
    dice = dice_loss(pred_logits, target_mask)
    return bce + dice


# =============================================================================
# Training & Validation Loops
# =============================================================================

def train_one_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: HANSNet model
        loader: Training DataLoader
        optimizer: Optimizer instance
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    for imgs_3, mask_center in tqdm(loader, desc="Training", leave=False):
        # Move data to device
        imgs_3 = imgs_3.to(device)
        mask_center = mask_center.to(device)
        
        # Forward pass
        pred = model(imgs_3)
        
        # Compute loss
        loss = combined_loss(pred, mask_center)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


def validate_one_epoch(model, loader, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: HANSNet model
        loader: Validation DataLoader
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
        for imgs_3, mask_center in tqdm(loader, desc="Validation", leave=False):
            # Move data to device
            imgs_3 = imgs_3.to(device)
            mask_center = mask_center.to(device)
            
            # Forward pass
            pred = model(imgs_3)
            
            # Compute loss
            loss = combined_loss(pred, mask_center)
            epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


# =============================================================================
# Training Pipeline
# =============================================================================

def train_pipeline(
    dataset_root: str,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    resume_path: str | None = None,
):
    """
    Complete training pipeline for HANS-Net with checkpointing support.
    
    Args:
        dataset_root: Path to dataset root folder
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for DataLoaders
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        resume_path: Path to checkpoint file to resume from (optional)
        
    Returns:
        Trained HANSNet model
    """
    # Initialize model
    model = HANSNet().to(device)
    
    # Build DataLoaders
    train_loader, val_loader = build_dataloaders(
        dataset_root,
        batch_size=batch_size,
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_path is not None and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        start_epoch = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch += 1  # Start from next epoch
        print(f"Resuming from epoch {start_epoch}")
    
    # Initialize best validation loss tracker
    best_val_loss = float("inf")
    
    # Training loop
    print(f"Starting training for {epochs} epochs (from epoch {start_epoch + 1})...")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print("-" * 50)
    
    for epoch in range(start_epoch, epochs):
        # Store best_val_loss before this epoch for comparison
        best_val_loss_before = best_val_loss
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        
        # Always save "last" checkpoint
        last_ckpt_path = os.path.join(checkpoint_dir, "last.pt")
        save_checkpoint(model, optimizer, epoch, last_ckpt_path)
        print(f"  [checkpoint] last  -> {last_ckpt_path}")
        
        # Save "best" checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(model, optimizer, epoch, best_ckpt_path)
            print(f"  [checkpoint] best  -> {best_ckpt_path} (val_loss improved)")
    
    print("-" * 50)
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    
    return model


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    """Command-line interface for training HANSNet."""
    parser = argparse.ArgumentParser(description="Train HANSNet on LiTS-style slices")
    parser.add_argument("--data-root", type=str, default="Dataset", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Resolve dataset paths in the same way as build_dataloaders
    img_root = os.path.join(args.data_root, "train_images", "train_images")
    mask_root = os.path.join(args.data_root, "train_masks")
    
    # If dataset folders are missing, print warning and exit gracefully
    if not (os.path.isdir(img_root) and os.path.isdir(mask_root)):
        print(f"[WARN] Dataset folders not found under: {args.data_root}")
        print(f"       Expected: {img_root} and {mask_root}")
        print("       Skipping training. Download or mount the dataset first.")
        return
    
    # Run training pipeline
    model = train_pipeline(
        dataset_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
    )
    
    print("Training finished.")


if __name__ == "__main__":
    main()
