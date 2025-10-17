#!/usr/bin/env python3
"""
Convert OpenFace 3.0 RetinaFace model to ONNX format for Apple Silicon optimization.

This script exports the RetinaFace-MobileNet-0.25 model to ONNX format, which can then
be accelerated using ONNX Runtime with CoreML execution provider on Apple Silicon.

Expected speedup: 5-10x (from ~191ms to ~20-40ms per detection)
"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Import RetinaFace components
from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
from openface.Pytorch_Retinaface.data import cfg_mnet
from openface.Pytorch_Retinaface.detect import load_model


def load_retinaface_model(model_path, device='cpu'):
    """
    Load the RetinaFace model from checkpoint

    Args:
        model_path: Path to Alignment_RetinaFace.pth file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: Loaded PyTorch model in eval mode
        cfg: Model configuration
    """
    print(f"Loading RetinaFace model from: {model_path}")

    # Use MobileNet 0.25 configuration (same as OpenFace 3.0)
    cfg = cfg_mnet

    # Build RetinaFace model
    model = RetinaFace(cfg=cfg, phase='test')

    # Load weights
    model = load_model(model, model_path, load_to_cpu=(device == 'cpu'))
    model.eval()

    if device != 'cpu':
        model = model.to(device)

    print(f"✓ Model loaded successfully")
    print(f"  Configuration: MobileNet-0.25")
    print(f"  Input size: Variable (supports any resolution)")

    return model, cfg


def export_to_onnx(model, output_path, input_size=(640, 640), opset_version=12):
    """
    Export RetinaFace model to ONNX format

    Args:
        model: PyTorch RetinaFace model
        output_path: Path to save ONNX model
        input_size: Input image size (height, width) - default 640x640
        opset_version: ONNX opset version (12 recommended for compatibility)
    """
    print(f"\nExporting model to ONNX...")
    print(f"  Opset version: {opset_version}")
    print(f"  Input size: {input_size[1]}x{input_size[0]}")

    # Create dummy input (batch_size=1, channels=3, height, width)
    # RetinaFace expects BGR images with mean subtraction applied
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Set model to eval mode
    model.eval()

    # Test forward pass to ensure model works
    print("  Testing forward pass...")
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  ✓ Forward pass successful")

        # RetinaFace model returns (loc, conf, landms)
        # loc: bounding box predictions relative to anchors
        # conf: classification scores (background vs face)
        # landms: 5-point facial landmarks
        if isinstance(test_output, tuple) and len(test_output) == 3:
            loc, conf, landms = test_output
            print(f"  Output shapes:")
            print(f"    loc (boxes): {loc.shape}")
            print(f"    conf (scores): {conf.shape}")
            print(f"    landms (landmarks): {landms.shape}")
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
        input_names=['input'],
        output_names=['loc', 'conf', 'landms'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'loc': {0: 'batch_size'},
            'conf': {0: 'batch_size'},
            'landms': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ ONNX export complete!")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {file_size_mb:.1f} MB")


def verify_onnx_model(onnx_path, input_size=(640, 640)):
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
        test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)

        # Run inference
        outputs = session.run(None, {'input': test_input})

        print(f"  ✓ ONNX Runtime inference successful")
        print(f"  Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            output_names = ['loc', 'conf', 'landms']
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
    parser = argparse.ArgumentParser(description='Convert RetinaFace model to ONNX')
    parser.add_argument('--model', type=str,
                       default='weights/Alignment_RetinaFace.pth',
                       help='Path to RetinaFace model checkpoint (default: weights/Alignment_RetinaFace.pth)')
    parser.add_argument('--output', type=str,
                       default='weights/retinaface_mobilenet025_coreml.onnx',
                       help='Output ONNX model path (default: weights/retinaface_mobilenet025_coreml.onnx)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='Input size (height width) for export (default: 640 640)')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version (default: 12 for broad compatibility)')
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
        print(f"Please ensure the RetinaFace model is available at the specified path.")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("RETINAFACE MODEL TO ONNX CONVERTER")
    print("="*60)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print("="*60)

    # Load model
    model, cfg = load_retinaface_model(str(model_path), device='cpu')

    # Export to ONNX
    export_to_onnx(model, str(output_path),
                   input_size=tuple(args.input_size),
                   opset_version=args.opset)

    # Verify if requested
    if args.verify:
        success = verify_onnx_model(str(output_path), input_size=tuple(args.input_size))
        if not success:
            print("\nNote: Verification failed, but ONNX file may still be valid.")
            print("Try running with ONNX Runtime CoreML provider separately.")

    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. The ONNX model is ready to use")
    print(f"2. Use the optimized detector:")
    print(f"   from onnx_retinaface_detector import ONNXRetinaFaceDetector")
    print(f"   detector = ONNXRetinaFaceDetector('{output_path}')")
    print(f"")
    print(f"Expected speedup: 5-10x (from ~191ms to ~20-40ms per detection)")
    print("="*60)


if __name__ == '__main__':
    main()
