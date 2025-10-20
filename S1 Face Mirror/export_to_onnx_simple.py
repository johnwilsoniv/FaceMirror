#!/usr/bin/env python3
"""
Simple ONNX Export for STAR Model

Exports the traced PyTorch STAR model directly to ONNX format.
This is simpler and faster than going through CoreML.
"""

import torch
import sys
from pathlib import Path


def main():
    print("="*80)
    print("PyTorch → ONNX Export for STAR Model")
    print("="*80)
    print()

    # Load the model
    print("Loading STAR model...")

    from openface.landmark_detection import LandmarkDetector
    import openface.STAR.conf.alignment as alignment_conf
    import os
    import io
    import logging

    # Patch config
    original_init = alignment_conf.Alignment.__init__

    def patched_init(self_inner, args):
        original_init(self_inner, args)
        import os.path as osp
        home_dir = os.path.expanduser('~')
        self_inner.ckpt_dir = os.path.join(home_dir, '.cache', 'openface', 'STAR')
        self_inner.work_dir = osp.join(self_inner.ckpt_dir, self_inner.data_definition, self_inner.folder)
        self_inner.model_dir = osp.join(self_inner.work_dir, 'model')
        self_inner.log_dir = osp.join(self_inner.work_dir, 'log')
        os.makedirs(self_inner.ckpt_dir, exist_ok=True)
        if hasattr(self_inner, 'writer') and self_inner.writer is not None:
            try:
                self_inner.writer.close()
            except:
                pass
        self_inner.writer = None

    alignment_conf.Alignment.__init__ = patched_init

    # Suppress output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    logging.getLogger().setLevel(logging.CRITICAL)

    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        script_dir = Path(__file__).parent
        checkpoint_path = str(script_dir / 'weights' / 'Landmark_98.pkl')
        detector = LandmarkDetector(model_path=checkpoint_path, device='cpu')
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.getLogger().setLevel(logging.INFO)
        alignment_conf.Alignment.__init__ = original_init

    model = detector.alignment.alignment
    model.eval()

    print("✓ Model loaded")
    print()

    # Export to ONNX
    output_path = script_dir / 'weights' / 'star_landmark_98_optimized.onnx'

    print(f"Exporting to: {output_path}")
    print("(This will take 1-2 minutes...)")
    print()

    example_input = torch.randn(1, 3, 256, 256)

    import time
    start = time.time()

    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input_image'],  # Match existing code expectations
        output_names=['output'],
        verbose=False,
    )

    elapsed = time.time() - start

    print(f"✓ Export completed in {elapsed:.1f}s")
    print()

    # Check file size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ ONNX model size: {size_mb:.1f} MB")
    print()

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Replace the old model:")
    print("   cd weights")
    print("   mv star_landmark_98_coreml.onnx star_landmark_98_coreml.onnx.backup")
    print("   cp star_landmark_98_optimized.onnx star_landmark_98_coreml.onnx")
    print()
    print("2. Test with profiler:")
    print("   python3 main.py")
    print()
    print("Expected results:")
    print("  • STAR: 163ms → 55-65ms (2.5-3x speedup)")
    print("  • Pipeline: 3.2 → 5.0-5.5 FPS")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
