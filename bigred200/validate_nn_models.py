#!/usr/bin/env python3
"""
Validate trained NN models against ground truth and original pipeline.

Checks:
1. Landmark accuracy (correlation, MAE vs ground truth)
2. AU accuracy (correlation, MAE vs ground truth)
3. Speed benchmark (FPS)

Usage:
    python validate_nn_models.py --data training_data_merged.h5 \
                                 --landmark-model models/landmark_pose/checkpoint_best.pt \
                                 --au-model models/au_prediction/checkpoint_best.pt
"""
import argparse
import time
import numpy as np
import h5py
import torch
from pathlib import Path


AU_NAMES = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]


def load_validation_data(h5_path: str, n_samples: int = 1000):
    """Load random validation samples from HDF5."""
    with h5py.File(h5_path, 'r') as f:
        total = f.attrs['num_samples']
        indices = np.random.choice(total, min(n_samples, total), replace=False)
        indices = np.sort(indices)

        images = f['images'][indices]
        landmarks = f['landmarks'][indices]
        global_params = f['global_params'][indices]
        local_params = f['local_params'][indices]
        au_intensities = f['au_intensities'][indices]

    return {
        'images': images,
        'landmarks': landmarks,
        'global_params': global_params,
        'local_params': local_params,
        'au_intensities': au_intensities,
    }


def validate_landmark_model(checkpoint_path: str, data: dict, device: torch.device):
    """Validate landmark model accuracy and speed."""
    from pyfaceau.nn.landmark_pose_net import UnifiedLandmarkPoseNet

    print("\n" + "=" * 60)
    print("LANDMARK MODEL VALIDATION")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UnifiedLandmarkPoseNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"Training val loss: {checkpoint.get('best_val_loss', '?')}")

    images = data['images']
    gt_landmarks = data['landmarks']
    gt_global = data['global_params']
    gt_local = data['local_params']

    n_samples = len(images)
    print(f"\nValidating on {n_samples} samples...")

    # Predictions
    pred_landmarks = []
    pred_global = []
    pred_local = []

    # Speed benchmark
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, n_samples, 32):
            batch = images[i:i+32]

            # Preprocess
            batch = batch.astype(np.float32) / 255.0
            batch = np.transpose(batch, (0, 3, 1, 2))
            batch = torch.from_numpy(batch).to(device)

            # Forward
            lm, gp, lp = model(batch)

            pred_landmarks.append(lm.cpu().numpy())
            pred_global.append(gp.cpu().numpy())
            pred_local.append(lp.cpu().numpy())

    elapsed = time.time() - start_time
    fps = n_samples / elapsed

    pred_landmarks = np.concatenate(pred_landmarks)
    pred_global = np.concatenate(pred_global)
    pred_local = np.concatenate(pred_local)

    # Landmark accuracy
    lm_diff = pred_landmarks - gt_landmarks
    lm_error = np.sqrt(lm_diff[:, :, 0]**2 + lm_diff[:, :, 1]**2)
    lm_mae = lm_error.mean()
    lm_std = lm_error.std()

    # Per-landmark correlation
    lm_corrs = []
    for i in range(68):
        for j in range(2):
            corr = np.corrcoef(pred_landmarks[:, i, j], gt_landmarks[:, i, j])[0, 1]
            if not np.isnan(corr):
                lm_corrs.append(corr)
    lm_corr_mean = np.mean(lm_corrs)

    # Global params accuracy
    gp_mae = np.abs(pred_global - gt_global).mean()
    gp_corrs = []
    for i in range(6):
        corr = np.corrcoef(pred_global[:, i], gt_global[:, i])[0, 1]
        if not np.isnan(corr):
            gp_corrs.append(corr)
    gp_corr_mean = np.mean(gp_corrs) if gp_corrs else 0

    print(f"\nResults:")
    print(f"  Landmark MAE: {lm_mae:.3f} px (target: <2.0)")
    print(f"  Landmark Correlation: {lm_corr_mean:.4f} (target: ≥0.99)")
    print(f"  Global Params MAE: {gp_mae:.4f}")
    print(f"  Global Params Correlation: {gp_corr_mean:.4f}")
    print(f"  Speed: {fps:.1f} FPS")

    # Pass/fail
    lm_pass = lm_mae < 2.0 and lm_corr_mean >= 0.99
    print(f"\n  Status: {'PASS ✓' if lm_pass else 'FAIL ✗'}")

    return {
        'lm_mae': lm_mae,
        'lm_corr': lm_corr_mean,
        'gp_mae': gp_mae,
        'gp_corr': gp_corr_mean,
        'fps': fps,
        'passed': lm_pass,
    }


def validate_au_model(checkpoint_path: str, data: dict, device: torch.device):
    """Validate AU model accuracy and speed."""
    from pyfaceau.nn.au_prediction_net import AUPredictionNet

    print("\n" + "=" * 60)
    print("AU MODEL VALIDATION")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AUPredictionNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"Training val loss: {checkpoint.get('best_val_loss', '?')}")

    images = data['images']
    gt_aus = data['au_intensities']

    n_samples = len(images)
    print(f"\nValidating on {n_samples} samples...")

    # Predictions
    pred_aus = []

    # Speed benchmark
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, n_samples, 32):
            batch = images[i:i+32]

            # Preprocess
            batch = batch.astype(np.float32) / 255.0
            batch = np.transpose(batch, (0, 3, 1, 2))
            batch = torch.from_numpy(batch).to(device)

            # Forward
            au_pred = model(batch)
            pred_aus.append(au_pred.cpu().numpy())

    elapsed = time.time() - start_time
    fps = n_samples / elapsed

    pred_aus = np.concatenate(pred_aus)

    # Per-AU metrics
    print(f"\nPer-AU Results:")
    print(f"{'AU':<10} {'Corr':>10} {'MAE':>10} {'Status':>10}")
    print("-" * 45)

    au_corrs = []
    au_maes = []
    au_passed = 0

    for i, au_name in enumerate(AU_NAMES):
        pred = pred_aus[:, i]
        gt = gt_aus[:, i]

        if pred.std() > 0.001 and gt.std() > 0.001:
            corr = np.corrcoef(pred, gt)[0, 1]
        else:
            corr = np.nan

        mae = np.abs(pred - gt).mean()

        if not np.isnan(corr):
            au_corrs.append(corr)
        au_maes.append(mae)

        passed = corr >= 0.95 if not np.isnan(corr) else False
        if passed:
            au_passed += 1

        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
        status = "✓" if passed else "✗"
        print(f"{au_name:<10} {corr_str:>10} {mae:>10.3f} {status:>10}")

    mean_corr = np.mean(au_corrs) if au_corrs else 0
    mean_mae = np.mean(au_maes)

    print("-" * 45)
    print(f"{'Mean':<10} {mean_corr:>10.3f} {mean_mae:>10.3f}")
    print(f"\nAUs passing (≥0.95): {au_passed}/{len(AU_NAMES)}")
    print(f"Speed: {fps:.1f} FPS")

    # Pass/fail
    au_pass = mean_corr >= 0.95 and au_passed >= 15
    print(f"\nStatus: {'PASS ✓' if au_pass else 'FAIL ✗'}")

    return {
        'au_corr_mean': mean_corr,
        'au_mae_mean': mean_mae,
        'au_passed': au_passed,
        'fps': fps,
        'passed': au_pass,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate NN models')
    parser.add_argument('--data', type=str, default='training_data_merged.h5',
                        help='Path to validation data')
    parser.add_argument('--landmark-model', type=str,
                        default='models/landmark_pose/checkpoint_best.pt',
                        help='Path to landmark model')
    parser.add_argument('--au-model', type=str,
                        default='models/au_prediction/checkpoint_best.pt',
                        help='Path to AU model')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of validation samples')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, mps, cpu, auto)')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("NEURAL NETWORK MODEL VALIDATION")
    print("=" * 60)
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading validation data from {args.data}...")
    data = load_validation_data(args.data, args.n_samples)
    print(f"Loaded {len(data['images'])} samples")

    results = {}

    # Validate landmark model
    if Path(args.landmark_model).exists():
        results['landmark'] = validate_landmark_model(
            args.landmark_model, data, device
        )
    else:
        print(f"\nLandmark model not found: {args.landmark_model}")

    # Validate AU model
    if Path(args.au_model).exists():
        results['au'] = validate_au_model(
            args.au_model, data, device
        )
    else:
        print(f"\nAU model not found: {args.au_model}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'landmark' in results:
        lm = results['landmark']
        print(f"Landmark Model: {'PASS' if lm['passed'] else 'FAIL'}")
        print(f"  Correlation: {lm['lm_corr']:.4f}, MAE: {lm['lm_mae']:.3f} px")

    if 'au' in results:
        au = results['au']
        print(f"AU Model: {'PASS' if au['passed'] else 'FAIL'}")
        print(f"  Correlation: {au['au_corr_mean']:.4f}, {au['au_passed']}/17 AUs passing")

    # Combined speed
    if 'landmark' in results and 'au' in results:
        # Sequential inference speed
        combined_fps = 1 / (1/results['landmark']['fps'] + 1/results['au']['fps'])
        print(f"\nCombined Speed: ~{combined_fps:.0f} FPS (sequential)")
        print("  (Actual pipeline will be faster with batching)")


if __name__ == '__main__':
    main()
