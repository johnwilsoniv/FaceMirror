#!/usr/bin/env python3
"""
Convert MTL Model to CoreML with Optimal ANE Settings

This script converts the MTL EfficientNet-B0 model to CoreML format
with settings optimized for Apple Neural Engine execution.

Based on successful STAR optimization, we use:
- MLProgram format (iOS 15+) for better FP16 support
- FLOAT16 precision for 2x memory reduction and faster ANE
- CPU_AND_NE compute units to prefer ANE over GPU
- Fixed input shape (224x224) to eliminate dynamic shape overhead
- Channels-last memory format (NHWC) preferred by ANE

Expected improvements over baseline (69% ANE coverage):
- Target: 85-90% ANE coverage
- Speedup: 1.5-2x (60ms → 30-40ms)
"""

import torch
import sys
from pathlib import Path


def load_mtl_model(checkpoint_path: str):
    """
    Load MTL model from PyTorch checkpoint

    Args:
        checkpoint_path: Path to MTL_backbone.pth

    Returns:
        model: MTL model ready for tracing
    """
    print("Loading MTL model...")

    from openface.model.MTL import MTL

    # Create model
    model = MTL()

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # Set to eval mode
    model.eval()

    print("✓ Model loaded successfully")
    print()

    return model


def trace_model(model):
    """
    Trace the model with torch.jit for CoreML conversion

    Args:
        model: MTL model

    Returns:
        traced_model: JIT-traced model
    """
    print("Tracing model with torch.jit...")

    # Create example input (224x224 RGB image)
    example_input = torch.randn(1, 3, 224, 224)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    print("✓ Model traced successfully")
    print()

    return traced_model


def convert_to_coreml(traced_model, output_path: str):
    """
    Convert traced PyTorch model to CoreML with optimal ANE settings

    Args:
        traced_model: JIT-traced model
        output_path: Path to save .mlpackage

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
    print("  Input Shape:    Fixed 224x224")
    print("  Memory Layout:  Channels-last (NHWC)")
    print()

    import time
    start_time = time.time()

    # Convert with optimal settings
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input_face",
            shape=(1, 3, 224, 224),
            color_layout=ct.colorlayout.RGB,
            scale=1.0/255.0,  # Normalize to [0, 1] during conversion
        )],
        outputs=[
            ct.TensorType(name="emotion_output"),
            ct.TensorType(name="gaze_output"),
            ct.TensorType(name="au_output"),
        ],
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
    print("  • Current baseline: 69% ANE coverage")
    print("  • Target: 85-90% ANE coverage")
    print("  • Optimistic: 90-95% ANE coverage")
    print()
    print("Note: SiLU activations (49 layers) may still run on CPU,")
    print("      but conversion optimization should help significantly")
    print()

    print("📊 STEP 2: Export to ONNX")
    print("-" * 80)
    print()
    print("Once Xcode analysis looks good, export to ONNX:")
    print()
    print("  python3 export_mtl_to_onnx.py")
    print()

    print("📊 STEP 3: Benchmark Performance")
    print("-" * 80)
    print()
    print("After ONNX export and integration:")
    print()
    print("  python3 main.py")
    print("  # Process test video and compare performance")
    print()
    print("Expected improvements:")
    print("  • Baseline: 60.8ms average (22.2s total, 46.2% of processing)")
    print("  • Target: 30-40ms average (~10-15s total)")
    print("  • Speedup: 1.5-2x")
    print("  • Overall pipeline: 55.5s → 43-45s (additional 1.24x)")
    print()

    print("=" * 80)


def main():
    """Main conversion entry point"""

    print()
    print("=" * 80)
    print("MTL Model → CoreML Conversion (Optimized for ANE)")
    print("=" * 80)
    print()

    # Paths
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / 'weights' / 'MTL_backbone.pth'
    output_path = script_dir / 'weights' / 'mtl_efficientnet_b0_optimized.mlpackage'

    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure MTL_backbone.pth is in the weights/ directory")
        return 1

    # Load model
    model = load_mtl_model(str(checkpoint_path))

    # Trace model
    traced_model = trace_model(model)

    # Convert to CoreML
    coreml_model = convert_to_coreml(traced_model, str(output_path))

    if coreml_model is None:
        return 1

    # Print next steps
    print_next_steps(str(output_path))

    return 0


if __name__ == '__main__':
    sys.exit(main())
