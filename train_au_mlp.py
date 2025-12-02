#!/usr/bin/env python3
"""
Train Neural Network AU Predictor to Replace SVMs

Trains an MLP to predict 17 AU intensities from HOG + geometric features,
using the current SVR predictions as training targets (knowledge distillation).

Usage:
    python train_au_mlp.py --data-dir training_data --epochs 100
    python train_au_mlp.py --data-h5 training_data.h5 --epochs 100  # HDF5 mode

Architecture:
    Input: 4702 features (4464 HOG + 238 geometric)
    → Dense(1024) → BatchNorm → ReLU → Dropout(0.3)
    → Dense(512) → BatchNorm → ReLU → Dropout(0.3)
    → Dense(256) → BatchNorm → ReLU → Dropout(0.2)
    → Dense(128) → BatchNorm → ReLU
    → Dense(17) → Output (17 AU intensities)

Features:
    - Mixed precision (FP16) training for faster GPU training
    - TensorBoard logging for training visualization
    - ONNX export for production deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import h5py
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# Enable cuDNN autotuning for best performance
torch.backends.cudnn.benchmark = True


class AUDataset(Dataset):
    """Dataset for AU prediction training."""

    def __init__(self, hog_features, geom_features, au_targets):
        """
        Args:
            hog_features: (N, 4464) HOG features
            geom_features: (N, 238) geometric features
            au_targets: (N, 17) AU intensity targets
        """
        # Concatenate features
        self.features = np.concatenate([hog_features, geom_features], axis=1)
        self.targets = au_targets

        # Convert to torch tensors
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.FloatTensor(self.targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class AUMLP(nn.Module):
    """MLP for AU intensity prediction."""

    def __init__(self, input_dim=4702, num_aus=17):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: 4702 → 1024
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 4: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Output: 128 → 17
            nn.Linear(128, num_aus)
        )

    def forward(self, x):
        return self.network(x)


def load_training_data(data_dir: Path) -> tuple:
    """
    Load all training data from HDF5 files (legacy directory format).

    Returns:
        hog_features: (N, 4464)
        geom_features: (N, 238)
        au_targets: (N, 17)
        video_ids: (N,) - for train/val split by video
    """
    au_dir = data_dir / 'au_features'
    if not au_dir.exists():
        raise FileNotFoundError(f"AU features directory not found: {au_dir}")

    all_hog = []
    all_geom = []
    all_aus = []
    all_video_ids = []

    h5_files = sorted(au_dir.glob('*.h5'))
    print(f"Found {len(h5_files)} video files")

    for video_idx, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as f:
            hog = f['hog_features'][:]
            geom = f['geom_features'][:]
            aus = f['au_predictions'][:]

            all_hog.append(hog)
            all_geom.append(geom)
            all_aus.append(aus)
            all_video_ids.extend([video_idx] * len(hog))

            print(f"  {h5_file.stem}: {len(hog)} samples")

    hog_features = np.concatenate(all_hog, axis=0)
    geom_features = np.concatenate(all_geom, axis=0)
    au_targets = np.concatenate(all_aus, axis=0)
    video_ids = np.array(all_video_ids)

    print(f"\nTotal samples: {len(hog_features)}")
    print(f"HOG shape: {hog_features.shape}")
    print(f"Geom shape: {geom_features.shape}")
    print(f"AU shape: {au_targets.shape}")

    return hog_features, geom_features, au_targets, video_ids


def load_training_data_h5(h5_path: Path) -> tuple:
    """
    Load training data from unified HDF5 file (Big Red 200 format).

    Returns:
        hog_features: (N, 4464)
        geom_features: (N, 238) - extracted from local_params
        au_targets: (N, 17)
        video_ids: (N,) - for train/val split by video
    """
    print(f"Loading training data from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        hog_features = f['hog_features'][:]
        local_params = f['local_params'][:]  # (N, 34) PDM parameters
        au_targets = f['au_intensities'][:]

        # Get video names for split
        if 'metadata' in f and 'video_names' in f['metadata']:
            video_names = f['metadata/video_names'][:]
            if isinstance(video_names[0], bytes):
                video_names = [v.decode('utf-8') for v in video_names]

            # Map video names to IDs
            unique_videos = list(set(video_names))
            video_name_to_id = {name: idx for idx, name in enumerate(unique_videos)}
            video_ids = np.array([video_name_to_id[v] for v in video_names])
        else:
            # Fall back to sequential IDs
            video_ids = np.zeros(len(hog_features), dtype=np.int32)

    # Extract geometric features from local params
    # The full geometric feature vector is 238 dimensions
    # For compatibility, we'll compute it from local_params
    try:
        import sys
        sys.path.insert(0, 'pyfaceau')
        from pyfaceau.features.pdm import PDMParser
        pdm_parser = PDMParser("pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")

        print("Extracting geometric features from local params...")
        geom_features = np.array([
            pdm_parser.extract_geometric_features(lp)
            for lp in local_params
        ], dtype=np.float32)
    except Exception as e:
        print(f"Warning: Could not extract geometric features: {e}")
        print("Using zeros for geometric features (training will still work)")
        geom_features = np.zeros((len(hog_features), 238), dtype=np.float32)

    print(f"\nTotal samples: {len(hog_features)}")
    print(f"HOG shape: {hog_features.shape}")
    print(f"Geom shape: {geom_features.shape}")
    print(f"AU shape: {au_targets.shape}")
    print(f"Unique videos: {len(np.unique(video_ids))}")

    return hog_features, geom_features, au_targets, video_ids


def export_to_onnx(model, output_path: str, input_dim: int = 4702):
    """
    Export trained model to ONNX format.

    Args:
        model: Trained AUMLP model
        output_path: Path to save ONNX model
        input_dim: Input feature dimension
    """
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['features'],
        output_names=['au_intensities'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'au_intensities': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True,
    )

    print(f"ONNX model exported to: {output_path}")


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0

    for features, targets in dataloader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(features)
            loss = criterion(outputs, targets)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(features)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate per-AU correlation
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    correlations = []
    for i in range(preds.shape[1]):
        corr = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)

    mean_corr = np.mean(correlations)
    mean_loss = total_loss / len(dataloader)

    return mean_loss, mean_corr, correlations


def main():
    parser = argparse.ArgumentParser(description="Train AU MLP")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Directory containing training data (legacy format)")
    parser.add_argument("--data-h5", type=str, default=None,
                       help="Path to unified HDF5 training data (Big Red 200 format)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Validation split ratio")
    parser.add_argument("--output", type=str, default="models/au_mlp.pt",
                       help="Output model path")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Enable TensorBoard logging")
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export to ONNX after training")
    args = parser.parse_args()

    print("=" * 80)
    print("AU MLP TRAINING")
    print("=" * 80)

    # TensorBoard setup
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = f"runs/au_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            writer = SummaryWriter(log_dir=log_dir)
            print(f"\nTensorBoard: {log_dir}")
        except ImportError:
            print("Warning: TensorBoard not available, skipping logging")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("\nLoading training data...")
    if args.data_h5:
        hog_features, geom_features, au_targets, video_ids = load_training_data_h5(Path(args.data_h5))
    elif args.data_dir:
        hog_features, geom_features, au_targets, video_ids = load_training_data(Path(args.data_dir))
    else:
        # Default to data-dir
        hog_features, geom_features, au_targets, video_ids = load_training_data(Path("training_data"))

    # Split by video (not by frame) to avoid data leakage
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=args.val_split, random_state=42
    )

    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)

    print(f"\nTrain videos: {len(train_videos)}, samples: {train_mask.sum()}")
    print(f"Val videos: {len(val_videos)}, samples: {val_mask.sum()}")

    # Create datasets
    train_dataset = AUDataset(
        hog_features[train_mask],
        geom_features[train_mask],
        au_targets[train_mask]
    )
    val_dataset = AUDataset(
        hog_features[val_mask],
        geom_features[val_mask],
        au_targets[val_mask]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    input_dim = hog_features.shape[1] + geom_features.shape[1]
    num_aus = au_targets.shape[1]
    model = AUMLP(input_dim=input_dim, num_aus=num_aus).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Loss, optimizer, scaler for mixed precision
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    scaler = GradScaler()  # For mixed precision training

    # Training loop
    print("\nTraining with mixed precision (FP16)...")
    best_val_corr = 0
    best_epoch = 0

    au_names = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
                'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
                'AU25', 'AU26', 'AU45']

    for epoch in range(args.epochs):
        start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate
        val_loss, val_corr, au_corrs = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        elapsed = time.time() - start

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_corr={val_corr:.4f}, time={elapsed:.1f}s")

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Correlation/val_mean', val_corr, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            for au_name, corr in zip(au_names, au_corrs):
                writer.add_scalar(f'Correlation/{au_name}', corr, epoch)

        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch + 1

            # Save model
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_corr': val_corr,
                'val_loss': val_loss,
                'au_correlations': dict(zip(au_names, au_corrs)),
                'input_dim': input_dim,
                'num_aus': num_aus
            }, output_path)

            print(f"  → Saved best model (corr={val_corr:.4f})")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation correlation: {best_val_corr:.4f}")
    print(f"Model saved to: {args.output}")

    # Load and print per-AU results
    checkpoint = torch.load(args.output)
    print("\nPer-AU Correlations:")
    for au_name, corr in checkpoint['au_correlations'].items():
        status = "✓" if corr > 0.9 else "○" if corr > 0.8 else "✗"
        print(f"  {status} {au_name}: {corr:.4f}")

    # Close TensorBoard
    if writer:
        writer.close()
        print(f"\nTensorBoard logs saved to: runs/")

    # Export to ONNX
    if args.export_onnx:
        onnx_path = Path(args.output).with_suffix('.onnx')
        model.load_state_dict(checkpoint['model_state_dict'])
        export_to_onnx(model, str(onnx_path), input_dim=input_dim)


if __name__ == "__main__":
    main()
