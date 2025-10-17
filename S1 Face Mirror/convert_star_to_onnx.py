#!/usr/bin/env python3
"""
Convert OpenFace 3.0 STAR landmark model to ONNX format for Apple Silicon optimization.

This script exports the STAR (Stacked Hourglass) model to ONNX format, which can then
be accelerated using ONNX Runtime with CoreML execution provider on Apple Silicon.

Expected speedup: 10-20x (from 1.8s to 90-180ms per frame)
"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Import OpenFace 3.0 STAR components
from openface.STAR.lib.utility import get_config, get_net
from openface.STAR.conf.alignment import Alignment as AlignmentConfig


def load_star_model(model_path, device='cpu'):
    """
    Load the STAR landmark detection model from checkpoint

    Args:
        model_path: Path to Landmark_98.pkl file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: Loaded PyTorch model in eval mode
        config: Model configuration
    """
    print(f"Loading STAR model from: {model_path}")

    # Create minimal config for alignment model
    args = argparse.Namespace(config_name='alignment', device_id=-1 if device == 'cpu' else 0)
    config = get_config(args)

    # Build network architecture
    net = get_net(config)

    # Load checkpoint
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path)

    # Load state dict
    net.load_state_dict(checkpoint['net'])
    net.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Architecture: {config.net}")
    print(f"  Landmarks: {config.classes_num}")
    print(f"  Input size: {config.width}x{config.height}")

    return net, config


def export_to_onnx(model, output_path, input_size=256, opset_version=14):
    """
    Export STAR model to ONNX format

    Args:
        model: PyTorch STAR model
        output_path: Path to save ONNX model
        input_size: Input image size (default: 256)
        opset_version: ONNX opset version (14 recommended for CoreML)
    """
    print(f"\nExporting model to ONNX...")
    print(f"  Opset version: {opset_version}")
    print(f"  Input size: {input_size}x{input_size}")

    # Create dummy input (batch_size=1, channels=3, height=256, width=256)
    # Input range: [-1, 1] (as per STAR preprocessing)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Set model to eval mode
    model.eval()

    # Test forward pass to ensure model works
    print("  Testing forward pass...")
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  ✓ Forward pass successful")

        # STAR model returns (output, heatmap, landmarks)
        # We want the landmarks (last element)
        if isinstance(test_output, tuple):
            landmarks = test_output[-1]  # Get landmarks from tuple
            print(f"  Output format: tuple with {len(test_output)} elements")
            print(f"  Landmarks shape: {landmarks.shape}")
        else:
            landmarks = test_output
            print(f"  Landmarks shape: {landmarks.shape}")

    # Export to ONNX
    print(f"  Exporting to: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['output', 'heatmap', 'landmarks'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'output': {0: 'batch_size'},
            'heatmap': {0: 'batch_size'},
            'landmarks': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ ONNX export complete!")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {file_size_mb:.1f} MB")


def verify_onnx_model(onnx_path, input_size=256):
    """
    Verify the exported ONNX model works correctly

    Args:
        onnx_path: Path to ONNX model
        input_size: Input image size
    """
    print(f"\nVerifying ONNX model...")

    try:
        import onnx
        import onnxruntime as ort

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)

        # Check model validity
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid")

        # Test inference with ONNX Runtime
        print("  Testing ONNX Runtime inference...")

        # CPU execution provider for verification
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Create test input
        test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # Run inference
        outputs = session.run(None, {'input_image': test_input})

        print(f"  ✓ ONNX Runtime inference successful")
        print(f"  Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"    Output {i} shape: {output.shape}")

        # Extract landmarks (last output)
        landmarks = outputs[-1]
        print(f"  ✓ Landmarks extracted: shape {landmarks.shape}")

        return True

    except ImportError as e:
        print(f"  ⚠ Cannot verify: {e}")
        print(f"  Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    """Main conversion script"""
    parser = argparse.ArgumentParser(description='Convert STAR model to ONNX')
    parser.add_argument('--model', type=str,
                       default='weights/Landmark_98.pkl',
                       help='Path to STAR model checkpoint (default: weights/Landmark_98.pkl)')
    parser.add_argument('--output', type=str,
                       default='weights/star_landmark_98_coreml.onnx',
                       help='Output ONNX model path (default: weights/star_landmark_98_coreml.onnx)')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version (default: 14 for CoreML compatibility)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX model after export')

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    output_path = script_dir / args.output

    # Check model exists
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print(f"Please ensure the STAR model is available at the specified path.")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAR MODEL TO ONNX CONVERTER")
    print("="*60)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print("="*60)

    # Load model
    model, config = load_star_model(str(model_path), device='cpu')

    # Export to ONNX
    export_to_onnx(model, str(output_path), input_size=config.width, opset_version=args.opset)

    # Verify if requested
    if args.verify:
        success = verify_onnx_model(str(output_path), input_size=config.width)
        if not success:
            print("\nNote: Verification failed, but ONNX file may still be valid.")
            print("Try running with ONNX Runtime CoreML provider separately.")

    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Install ONNX Runtime with CoreML support:")
    print(f"   pip install onnxruntime")
    print(f"")
    print(f"2. Use the optimized detector:")
    print(f"   from onnx_star_detector import ONNXStarDetector")
    print(f"   detector = ONNXStarDetector('{output_path}')")
    print(f"")
    print(f"Expected speedup: 10-20x (1800ms → 90-180ms per frame)")
    print("="*60)


if __name__ == '__main__':
    main()
