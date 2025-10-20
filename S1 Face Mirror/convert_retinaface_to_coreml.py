#!/usr/bin/env python3
"""
Convert RetinaFace Model to CoreML with Optimal ANE Settings

This script converts the RetinaFace face detection model to CoreML format
with settings optimized for Apple Neural Engine execution.

RetinaFace specifics:
- Input: Variable size images (but we'll use fixed sizes for optimization)
- Output: Face detections (boxes, scores, landmarks)
- Architecture: 100% ANE-compatible (Conv2d, BatchNorm2d, LeakyReLU)

Expected improvements over baseline (81.3% ANE coverage):
- Target: 90-95% ANE coverage
- Speedup: 1.7-2x (167ms → 80-100ms)
"""

import torch
import sys
from pathlib import Path


def load_retinaface_model(weights_path: str):
    """
    Load RetinaFace model from PyTorch checkpoint

    Args:
        weights_path: Path to Alignment_RetinaFace.pth

    Returns:
        model: RetinaFace model ready for tracing
    """
    print("Loading RetinaFace model...")

    from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
    from openface.Pytorch_Retinaface.detect import load_model
    from openface.Pytorch_Retinaface.data import cfg_mnet

    # Create model
    cfg = cfg_mnet
    model = RetinaFace(cfg=cfg, phase='test')

    # Load checkpoint
    print(f"Loading checkpoint: {weights_path}")
    model = load_model(model, weights_path, load_to_cpu=True)

    # Set to eval mode
    model.eval()

    print("✓ Model loaded successfully")
    print()

    return model, cfg


def trace_model(model, input_size=(640, 640)):
    """
    Trace the model with torch.jit for CoreML conversion

    Args:
        model: RetinaFace model
        input_size: Input image size (height, width)

    Returns:
        traced_model: JIT-traced model
    """
    print(f"Tracing model with input size {input_size}...")

    # Create example input (typical video frame size)
    example_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    print("✓ Model traced successfully")
    print()

    return traced_model


def convert_to_coreml(traced_model, output_path: str, input_size=(640, 640)):
    """
    Convert traced PyTorch model to CoreML with optimal ANE settings

    Args:
        traced_model: JIT-traced model
        output_path: Path to save .mlpackage
        input_size: Input image size (height, width)

    Returns:
        coreml_model: Converted CoreML model
    """
    print("Converting to CoreML with optimal ANE settings...")
    print()

    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed")
        print("Install with: pip install coremltools")
        print()
        print("Note: Use Python 3.10 if you encounter 'BlobWriter not loaded' errors")
        return None

    # Print coremltools version
    print(f"coremltools version: {ct.__version__}")
    print()

    print("Conversion settings:")
    print("  Format:         MLProgram (iOS 15+)")
    print("  Precision:      FLOAT16")
    print("  Compute Units:  CPU_AND_NE (prefer Neural Engine)")
    print(f"  Input Shape:    Fixed {input_size}")
    print("  Memory Layout:  Channels-last (NHWC)")
    print()

    import time
    start_time = time.time()

    # Convert with optimal settings
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input_image",
            shape=(1, 3, input_size[0], input_size[1]),
            color_layout=ct.colorlayout.RGB,
            scale=1.0,  # No normalization (done in preprocessing)
        )],
        convert_to="mlprogram",  # MLProgram format (better than NeuralNetwork)
        compute_precision=ct.precision.FLOAT16,  # FP16 for ANE optimization
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Prefer ANE over GPU
        minimum_deployment_target=ct.target.iOS17,
    )

    elapsed = time.time() - start_time

    print(f"✓ Conversion completed in {elapsed:.1f}s")
    print()

    # Save model
    print(f"Saving CoreML model to: {output_path}")
    coreml_model.save(output_path)

    # Get file size
    import os
    def get_dir_size(path):
        total = 0
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
        return total

    size_bytes = get_dir_size(output_path)
    size_mb = size_bytes / 1024 / 1024

    print(f"✓ Model saved ({size_mb:.1f} MB)")
    print()

    return coreml_model


def print_next_steps(output_path: str):
    """Print next steps for analysis and integration"""

    print("=" * 80)
    print("CONVERSION COMPLETE - NEXT STEPS")
    print("=" * 80)
    print()

    print("✅ CoreML model created successfully!")
    print()
    print(f"📍 Location: {output_path}")
    print()

    print("📊 STEP 1: Analyze with Xcode")
    print("-" * 80)
    print()
    print("Open the model in Xcode to check ANE coverage:")
    print()
    print(f"  open '{output_path}'")
    print()
    print("Then in Xcode:")
    print("  1. Click 'Performance' tab")
    print("  2. Look for 'Neural Engine' operations count")
    print()
    print("Expected results:")
    print("  • Current baseline: 81.3% ANE coverage")
    print("  • Target: 90-95% ANE coverage")
    print("  • Optimistic: 95%+ ANE coverage")
    print()
    print("Note: LeakyReLU activations (38 layers) should run on ANE")
    print()

    print("📊 NOTE: RetinaFace Optimization Impact")
    print("-" * 80)
    print()
    print("⚠️  With Option 3 (skip face detection on mirrored videos),")
    print("   RetinaFace is NO LONGER USED during AU extraction!")
    print()
    print("This optimization will help:")
    print("  • Face mirroring step (initial face detection)")
    print("  • Processing non-mirrored videos")
    print("  • Future features requiring face detection")
    print()
    print("Current AU extraction: ~167ms saved by skipping RetinaFace entirely")
    print("                       (Option 3 architectural fix)")
    print()

    print("=" * 80)


def main():
    """Main conversion entry point"""

    print()
    print("=" * 80)
    print("RetinaFace Model → CoreML Conversion (Optimized for ANE)")
    print("=" * 80)
    print()

    # Paths
    script_dir = Path(__file__).parent
    weights_path = script_dir / 'weights' / 'Alignment_RetinaFace.pth'
    output_path = script_dir / 'weights' / 'retinaface_optimized.mlpackage'

    # Check weights exist
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print("Please ensure Alignment_RetinaFace.pth is in the weights/ directory")
        return 1

    # Load model
    model, cfg = load_retinaface_model(str(weights_path))

    # Trace model
    # Use 640x640 as typical video frame size (can be adjusted)
    traced_model = trace_model(model, input_size=(640, 640))

    # Convert to CoreML
    coreml_model = convert_to_coreml(traced_model, str(output_path), input_size=(640, 640))

    if coreml_model is None:
        return 1

    # Print next steps
    print_next_steps(str(output_path))

    return 0


if __name__ == '__main__':
    sys.exit(main())
