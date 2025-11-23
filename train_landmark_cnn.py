#!/usr/bin/env python3
"""
Train Neural Network Landmark Detector to Replace CLNF

Trains a CNN to predict 68 facial landmarks from face crops,
using CLNF landmarks as training targets (knowledge distillation).

Usage:
    python train_landmark_cnn.py --data-dir training_data --epochs 100

Architecture:
    Input: 256×256×3 face crop
    → MobileNetV2 backbone (pretrained)
    → Global Average Pooling → 1280
    → Dense(512) → BatchNorm → ReLU → Dropout(0.3)
    → Dense(256) → BatchNorm → ReLU → Dropout(0.2)
    → Dense(136) → Output (68 landmarks × 2 coordinates)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import h5py
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import time

# Enable cuDNN autotuning for best performance
torch.backends.cudnn.benchmark = True


class LandmarkDataset(Dataset):
    """Dataset for landmark prediction training."""

    def __init__(self, face_crops, landmarks, augment=False):
        """
        Args:
            face_crops: (N, 256, 256, 3) face crops as uint8
            landmarks: (N, 68, 2) landmark coordinates normalized to [0, 1]
            augment: Enable data augmentation
        """
        self.face_crops = face_crops
        self.landmarks = landmarks
        self.augment = augment

        # Normalization for pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Augmentation transforms
        if augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )

    def __len__(self):
        return len(self.face_crops)

    def __getitem__(self, idx):
        # Get image and landmarks
        image = self.face_crops[idx]  # (256, 256, 3) uint8
        landmarks = self.landmarks[idx].copy()  # (68, 2)

        # Convert to tensor and normalize
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0  # (3, 256, 256)

        # Apply augmentation
        if self.augment:
            # Color jitter
            image = self.color_jitter(image)

            # Random horizontal flip (with landmark mirroring)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, [2])  # Flip width
                landmarks[:, 0] = 1.0 - landmarks[:, 0]  # Mirror x coordinates

                # Swap left/right landmark pairs
                # Jaw: 0-16 (symmetric around 8)
                for i in range(8):
                    landmarks[[i, 16-i]] = landmarks[[16-i, i]]
                # Eyebrows: 17-21 ↔ 22-26
                landmarks[[17,18,19,20,21, 22,23,24,25,26]] = \
                    landmarks[[26,25,24,23,22, 21,20,19,18,17]]
                # Nose: 31-35 (symmetric around 33)
                landmarks[[31,32, 34,35]] = landmarks[[35,34, 32,31]]
                # Eyes: 36-41 ↔ 42-47
                landmarks[[36,37,38,39,40,41, 42,43,44,45,46,47]] = \
                    landmarks[[45,44,43,42,47,46, 39,38,37,36,41,40]]
                # Mouth outer: 48-54 ↔ 54-60 (symmetric around 51, 57)
                landmarks[[48,49,50, 52,53,54, 55,56, 58,59]] = \
                    landmarks[[54,53,52, 50,49,48, 59,58, 56,55]]
                # Mouth inner: 60-64 ↔ 64-67
                landmarks[[60,61,62, 63,64, 65,66,67]] = \
                    landmarks[[64,63,62, 61,60, 67,66,65]]

            # Small rotation (±10 degrees)
            if torch.rand(1) < 0.3:
                angle = (torch.rand(1) * 20 - 10).item()  # -10 to +10 degrees
                # Rotate image
                image = transforms.functional.rotate(image, angle)
                # Rotate landmarks around center
                cx, cy = 0.5, 0.5
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                x = landmarks[:, 0] - cx
                y = landmarks[:, 1] - cy
                landmarks[:, 0] = x * cos_a - y * sin_a + cx
                landmarks[:, 1] = x * sin_a + y * cos_a + cy

        # Normalize image for pretrained model
        image = self.normalize(image)

        # Flatten landmarks for output
        landmarks_flat = torch.FloatTensor(landmarks.flatten())  # (136,)

        return image, landmarks_flat


class LandmarkCNN(nn.Module):
    """CNN for facial landmark prediction using MobileNetV2 backbone."""

    def __init__(self, num_landmarks=68, pretrained=True):
        super().__init__()

        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # Remove classifier
        self.backbone = mobilenet.features  # Output: (batch, 1280, 8, 8)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return x


def load_training_data(data_dir: Path) -> tuple:
    """
    Load all landmark training data from HDF5 files.

    Returns:
        face_crops: (N, 256, 256, 3) uint8
        landmarks: (N, 68, 2) normalized to [0, 1]
        video_ids: (N,) for train/val split
    """
    lm_dir = data_dir / 'landmarks'
    if not lm_dir.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {lm_dir}")

    all_crops = []
    all_landmarks = []
    all_video_ids = []

    h5_files = sorted(lm_dir.glob('*.h5'))
    print(f"Found {len(h5_files)} video files")

    for video_idx, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as f:
            crops = f['face_crops'][:]  # (N, 256, 256, 3)
            landmarks = f['landmarks'][:]  # (N, 68, 2)

            # Normalize landmarks to [0, 1]
            landmarks = landmarks / 256.0

            all_crops.append(crops)
            all_landmarks.append(landmarks)
            all_video_ids.extend([video_idx] * len(crops))

            print(f"  {h5_file.stem}: {len(crops)} samples")

    face_crops = np.concatenate(all_crops, axis=0)
    landmarks = np.concatenate(all_landmarks, axis=0)
    video_ids = np.array(all_video_ids)

    print(f"\nTotal samples: {len(face_crops)}")
    print(f"Face crops shape: {face_crops.shape}")
    print(f"Landmarks shape: {landmarks.shape}")

    return face_crops, landmarks, video_ids


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0

    for images, landmarks in dataloader:
        images = images.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, landmarks)

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
        for images, landmarks in dataloader:
            images = images.to(device, non_blocking=True)
            landmarks = landmarks.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, landmarks)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(landmarks.cpu().numpy())

    # Calculate mean error in pixels (assuming 256x256 image)
    preds = np.concatenate(all_preds, axis=0).reshape(-1, 68, 2) * 256
    targets = np.concatenate(all_targets, axis=0).reshape(-1, 68, 2) * 256

    errors = np.sqrt(((preds - targets) ** 2).sum(axis=2))  # (N, 68)
    mean_error = errors.mean()
    max_error = errors.max(axis=1).mean()  # Mean of per-sample max errors

    mean_loss = total_loss / len(dataloader)

    return mean_loss, mean_error, max_error


def main():
    parser = argparse.ArgumentParser(description="Train Landmark CNN")
    parser.add_argument("--data-dir", type=str, default="training_data",
                       help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Validation split ratio")
    parser.add_argument("--output", type=str, default="models/landmark_cnn.pt",
                       help="Output model path")
    parser.add_argument("--no-augment", action="store_true",
                       help="Disable data augmentation")
    args = parser.parse_args()

    print("=" * 80)
    print("LANDMARK CNN TRAINING")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("\nLoading training data...")
    data_dir = Path(args.data_dir)
    face_crops, landmarks, video_ids = load_training_data(data_dir)

    # Split by video
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=args.val_split, random_state=42
    )

    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)

    print(f"\nTrain videos: {len(train_videos)}, samples: {train_mask.sum()}")
    print(f"Val videos: {len(val_videos)}, samples: {val_mask.sum()}")

    # Create datasets
    train_dataset = LandmarkDataset(
        face_crops[train_mask],
        landmarks[train_mask],
        augment=not args.no_augment
    )
    val_dataset = LandmarkDataset(
        face_crops[val_mask],
        landmarks[val_mask],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    model = LandmarkCNN(num_landmarks=68, pretrained=True).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss, optimizer, scaler for mixed precision
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    scaler = GradScaler()  # For mixed precision training

    # Training loop
    print("\nTraining with mixed precision (FP16)...")
    best_val_error = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate
        val_loss, val_error, val_max_error = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        elapsed = time.time() - start

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"val_error={val_error:.2f}px, max={val_max_error:.2f}px, "
              f"time={elapsed:.1f}s")

        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            best_epoch = epoch + 1

            # Save model
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_error,
                'val_max_error': val_max_error,
                'val_loss': val_loss
            }, output_path)

            print(f"  → Saved best model (error={val_error:.2f}px)")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation error: {best_val_error:.2f} pixels")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
