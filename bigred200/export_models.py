#!/usr/bin/env python3
"""
Export trained models to ONNX and CoreML formats.

Usage:
    python export_models.py --landmark-checkpoint models/landmark_pose/checkpoint_best.pt \
                            --au-checkpoint models/au_prediction/checkpoint_best.pt \
                            --output-dir models/exported
"""
import argparse
import os
import sys
from pathlib import Path

import torch


def export_landmark_model(checkpoint_path: str, output_dir: str):
    """Export LandmarkPoseNet to ONNX and CoreML."""
    from pyfaceau.nn.landmark_pose_net import (
        UnifiedLandmarkPoseNet,
        export_to_onnx,
        export_to_coreml,
    )

    print(f"\n{'='*60}")
    print("Exporting LandmarkPoseNet")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = UnifiedLandmarkPoseNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'landmark_pose.onnx')
    print(f"\nExporting to ONNX: {onnx_path}")
    export_to_onnx(model, onnx_path)
    print(f"  Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    # Export to CoreML
    try:
        coreml_path = os.path.join(output_dir, 'landmark_pose.mlpackage')
        print(f"\nExporting to CoreML: {coreml_path}")
        export_to_coreml(model, coreml_path)
        print(f"  Export successful")
    except Exception as e:
        print(f"  CoreML export failed (may need macOS): {e}")

    return onnx_path


def export_au_model(checkpoint_path: str, output_dir: str):
    """Export AUPredictionNet to ONNX and CoreML."""
    from pyfaceau.nn.au_prediction_net import (
        AUPredictionNet,
        export_au_to_onnx,
        export_au_to_coreml,
    )

    print(f"\n{'='*60}")
    print("Exporting AUPredictionNet")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = AUPredictionNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'au_prediction.onnx')
    print(f"\nExporting to ONNX: {onnx_path}")
    export_au_to_onnx(model, onnx_path)
    print(f"  Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    # Export to CoreML
    try:
        coreml_path = os.path.join(output_dir, 'au_prediction.mlpackage')
        print(f"\nExporting to CoreML: {coreml_path}")
        export_au_to_coreml(model, coreml_path)
        print(f"  Export successful")
    except Exception as e:
        print(f"  CoreML export failed (may need macOS): {e}")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export trained models')
    parser.add_argument('--landmark-checkpoint', type=str,
                        default='models/landmark_pose/checkpoint_best.pt',
                        help='Path to landmark model checkpoint')
    parser.add_argument('--au-checkpoint', type=str,
                        default='models/au_prediction/checkpoint_best.pt',
                        help='Path to AU model checkpoint')
    parser.add_argument('--output-dir', type=str, default='models/exported',
                        help='Output directory for exported models')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MODEL EXPORT")
    print("=" * 60)

    # Export landmark model
    if os.path.exists(args.landmark_checkpoint):
        export_landmark_model(args.landmark_checkpoint, args.output_dir)
    else:
        print(f"\nLandmark checkpoint not found: {args.landmark_checkpoint}")

    # Export AU model
    if os.path.exists(args.au_checkpoint):
        export_au_model(args.au_checkpoint, args.output_dir)
    else:
        print(f"\nAU checkpoint not found: {args.au_checkpoint}")

    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files:")
    for f in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            print(f"  {f}: {os.path.getsize(fpath) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
