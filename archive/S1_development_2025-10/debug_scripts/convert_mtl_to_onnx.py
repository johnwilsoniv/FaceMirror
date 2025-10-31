#!/usr/bin/env python3
"""
Convert OpenFace 3.0 MTL (Multi-Task Learning) model to ONNX format for Apple Silicon optimization.

The MTL model uses EfficientNet-B0 backbone with three task heads:
- Emotion classification (8 classes)
- Gaze regression (2 values: yaw/pitch)
- AU regression (8 AUs with GNN-based Head)

Expected speedup: 3-5x (from ~50-100ms to ~15-30ms per face)
"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Import MTL model
from openface.model.MTL import MTL


def load_mtl_model(model_path, device='cpu'):
    """
    Load the MTL model from checkpoint

    Args:
        model_path: Path to MTL_backbone.pth file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: Loaded PyTorch model in eval mode
    """
    print(f"Loading MTL model from: {model_path}")

    # Build MTL model (EfficientNet-B0 + 3 task heads)
    model = MTL()

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if device != 'cpu':
        model = model.to(device)

    print(f"✓ Model loaded successfully")
    print(f"  Backbone: EfficientNet-B0 (tf_efficientnet_b0_ns)")
    print(f"  Input size: 224x224 RGB")
    print(f"  Outputs: emotion (8), gaze (2), AU (8)")

    return model


def export_to_onnx(model, output_path, input_size=224, opset_version=14):
    """
    Export MTL model to ONNX format

    Args:
        model: PyTorch MTL model
        output_path: Path to save ONNX model
        input_size: Input image size (default: 224x224)
        opset_version: ONNX opset version (14 recommended for CoreML)
    """
    print(f"\nExporting model to ONNX...")
    print(f"  Opset version: {opset_version}")
    print(f"  Input size: {input_size}x{input_size}")

    # Create dummy input (batch_size=1, channels=3, height, width)
    # MTL expects normalized RGB images (ImageNet stats)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Set model to eval mode
    model.eval()

    # Test forward pass to ensure model works
    print("  Testing forward pass...")
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  ✓ Forward pass successful")

        # MTL model returns (emotion_output, gaze_output, au_output)
        if isinstance(test_output, tuple) and len(test_output) == 3:
            emotion, gaze, au = test_output
            print(f"  Output shapes:")
            print(f"    emotion: {emotion.shape} (8 classes)")
            print(f"    gaze: {gaze.shape} (yaw, pitch)")
            print(f"    au: {au.shape} (8 AUs)")
        else:
            print(f"  Output format: {type(test_output)}")

    # Export to ONNX
    print(f"  Exporting to: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_face'],
        output_names=['emotion', 'gaze', 'au'],
        dynamic_axes={
            'input_face': {0: 'batch_size'},
            'emotion': {0: 'batch_size'},
            'gaze': {0: 'batch_size'},
            'au': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ ONNX export complete!")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {file_size_mb:.1f} MB")


def verify_onnx_model(onnx_path, input_size=224):
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

        # Try CoreML provider first, then CPU
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)

        # Check which provider is active
        active_providers = session.get_providers()
        if 'CoreMLExecutionProvider' in active_providers:
            print("  ✓ CoreML execution provider available")
        else:
            print("  ✓ Using CPU execution provider")

        # Create test input
        test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # Run inference
        outputs = session.run(None, {'input_face': test_input})

        print(f"  ✓ ONNX Runtime inference successful")
        print(f"  Number of outputs: {len(outputs)}")
        output_names = ['emotion', 'gaze', 'au']
        for i, output in enumerate(outputs):
            name = output_names[i] if i < len(output_names) else f'output_{i}'
            print(f"    {name}: shape {output.shape}")

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
    parser = argparse.ArgumentParser(description='Convert MTL model to ONNX')
    parser.add_argument('--model', type=str,
                       default='weights/MTL_backbone.pth',
                       help='Path to MTL model checkpoint (default: weights/MTL_backbone.pth)')
    parser.add_argument('--output', type=str,
                       default='weights/mtl_efficientnet_b0_coreml.onnx',
                       help='Output ONNX model path (default: weights/mtl_efficientnet_b0_coreml.onnx)')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input size (default: 224 for 224x224)')
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
        print(f"Please ensure the MTL model is available at the specified path.")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MTL MODEL TO ONNX CONVERTER")
    print("="*60)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print("="*60)

    # Load model
    model = load_mtl_model(str(model_path), device='cpu')

    # Export to ONNX
    export_to_onnx(model, str(output_path),
                   input_size=args.input_size,
                   opset_version=args.opset)

    # Verify if requested
    if args.verify:
        success = verify_onnx_model(str(output_path), input_size=args.input_size)
        if not success:
            print("\nNote: Verification failed, but ONNX file may still be valid.")
            print("Try running with ONNX Runtime CoreML provider separately.")

    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. The ONNX model is ready to use")
    print(f"2. Use the optimized detector:")
    print(f"   from onnx_mtl_detector import ONNXMultitaskPredictor")
    print(f"   predictor = ONNXMultitaskPredictor('{output_path}')")
    print(f"")
    print(f"Expected speedup: 3-5x (from ~50-100ms to ~15-30ms per face)")
    print("="*60)


if __name__ == '__main__':
    main()
