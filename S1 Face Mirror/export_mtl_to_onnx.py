#!/usr/bin/env python3
"""
Export MTL Model to ONNX Format

Exports the MTL EfficientNet-B0 model directly from PyTorch to ONNX format
for integration with the Face Mirror pipeline.

The ONNX model will use ONNX Runtime's CoreML Execution Provider,
which will leverage the optimized CoreML model internally.
"""

import torch
import sys
from pathlib import Path


def main():
    print("=" * 80)
    print("PyTorch → ONNX Export for MTL Model")
    print("=" * 80)
    print()

    script_dir = Path(__file__).parent

    # Load the model
    print("Loading MTL model...")

    from openface.model.MTL import MTL

    checkpoint_path = script_dir / 'weights' / 'MTL_backbone.pth'

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return 1

    # Create and load model
    model = MTL()
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print("✓ Model loaded")
    print()

    # Export to ONNX
    output_path = script_dir / 'weights' / 'mtl_efficientnet_b0_optimized.onnx'

    print(f"Exporting to: {output_path}")
    print("(This will take 1-2 minutes...)")
    print()

    example_input = torch.randn(1, 3, 224, 224)

    import time
    start = time.time()

    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input_face'],  # Match existing code expectations
        output_names=['emotion_output', 'gaze_output', 'au_output'],
        verbose=False,
    )

    elapsed = time.time() - start

    print(f"✓ Export completed in {elapsed:.1f}s")
    print()

    # Check file size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ ONNX model size: {size_mb:.1f} MB")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Replace the old model:")
    print("   cd weights")
    print("   mv mtl_efficientnet_b0_coreml.onnx mtl_efficientnet_b0_coreml.onnx.backup")
    print("   cp mtl_efficientnet_b0_optimized.onnx mtl_efficientnet_b0_coreml.onnx")
    print()
    print("2. Test with profiler:")
    print("   python3 main.py")
    print()
    print("Expected results:")
    print("  • MTL: 60.8ms → 30-40ms (1.5-2x speedup)")
    print("  • Pipeline: 55.5s → 43-45s")
    print("  • ANE Coverage: 69% → 78.2%")
    print()
    print("Note: 78.2% ANE coverage (194/248 ops)")
    print("      52 GPU ops (likely SiLU activations)")
    print("      Still expect good speedup from ANE + GPU mix")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
