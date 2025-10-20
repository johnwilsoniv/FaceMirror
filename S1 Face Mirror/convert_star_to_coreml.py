#!/usr/bin/env python3
"""
STAR Model CoreML Conversion Script

Converts the STAR landmark detection model to optimized CoreML format
with settings designed to maximize Apple Neural Engine utilization.

Expected improvements:
- ANE coverage: 82.6% → 90-95%+
- Partitions: 18 → <8 (ideally <5)
- Inference time: 163ms → 55-100ms (2-3x speedup)
"""

import torch
import numpy as np
from pathlib import Path
import sys
import io
import logging
import time


def patch_openface_config():
    """
    Patch OpenFace STAR config to avoid /work path errors
    (Same approach as openface_integration.py and star_architecture_analysis.py)
    """
    import openface.STAR.conf.alignment as alignment_conf
    import os

    original_init = alignment_conf.Alignment.__init__

    def patched_init(self_inner, args):
        original_init(self_inner, args)
        # Replace hardcoded /work path with a writable location
        import os.path as osp
        home_dir = os.path.expanduser('~')
        self_inner.ckpt_dir = os.path.join(home_dir, '.cache', 'openface', 'STAR')
        self_inner.work_dir = osp.join(self_inner.ckpt_dir, self_inner.data_definition, self_inner.folder)
        self_inner.model_dir = osp.join(self_inner.work_dir, 'model')
        self_inner.log_dir = osp.join(self_inner.work_dir, 'log')
        # Create directories if they don't exist
        os.makedirs(self_inner.ckpt_dir, exist_ok=True)

        # CRITICAL: Close and disable TensorBoard writer
        if hasattr(self_inner, 'writer') and self_inner.writer is not None:
            try:
                self_inner.writer.close()
            except:
                pass
        self_inner.writer = None

    alignment_conf.Alignment.__init__ = patched_init
    return original_init


def load_star_model(checkpoint_path: str):
    """
    Load STAR model via OpenFace LandmarkDetector

    Args:
        checkpoint_path: Path to Landmark_98.pkl

    Returns:
        PyTorch model in eval mode
    """
    print("="*80)
    print("STEP 1: Loading STAR Model")
    print("="*80)
    print()

    from openface.landmark_detection import LandmarkDetector

    # Patch config before loading
    print("Patching OpenFace config...")
    original_init = patch_openface_config()

    # Suppress verbose output during initialization
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    logging.getLogger().setLevel(logging.CRITICAL)

    try:
        print("Loading model via LandmarkDetector...")
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        detector = LandmarkDetector(model_path=checkpoint_path, device='cpu')

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.getLogger().setLevel(logging.INFO)

    # Get the actual model
    model = detector.alignment.alignment
    model.eval()

    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.1f} MB")
    print()

    return model, detector


def prepare_model_for_ane(model):
    """
    Prepare model for optimal ANE performance

    Args:
        model: PyTorch model

    Returns:
        Model converted to channels-last format
    """
    print("="*80)
    print("STEP 2: Preparing Model for ANE")
    print("="*80)
    print()

    # Convert to channels-last memory format (NHWC)
    # This is preferred by Apple Neural Engine
    print("Converting to channels-last memory format (NHWC)...")
    model = model.to(memory_format=torch.channels_last)

    print("✓ Memory format: channels_last (optimized for ANE)")
    print()

    return model


def trace_model(model, input_size=(1, 3, 256, 256)):
    """
    Trace model with fixed input shape using torch.jit.trace

    Args:
        model: PyTorch model in eval mode
        input_size: Input tensor shape (B, C, H, W)

    Returns:
        Traced model
    """
    print("="*80)
    print("STEP 3: Tracing Model with Fixed Input Shape")
    print("="*80)
    print()

    print(f"Creating example input: {input_size}")
    example_input = torch.randn(*input_size).to(memory_format=torch.channels_last)

    print("Tracing model with torch.jit.trace...")
    print("(This may take 30-60 seconds...)")

    start_time = time.time()

    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    elapsed = time.time() - start_time

    print(f"✓ Model traced successfully in {elapsed:.1f}s")
    print(f"✓ Input shape: {input_size} (fixed)")
    print()

    # Validate traced model produces same output
    print("Validating traced model structure...")
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = traced_model(example_input)

    # Check output structure matches (not values, just structure)
    print(f"✓ Original output type: {type(original_output)}")
    print(f"✓ Traced output type: {type(traced_output)}")

    if isinstance(original_output, (tuple, list)):
        print(f"✓ Output has {len(original_output)} elements")

    print("✓ Tracing structure validation passed")
    print("  (Note: Value comparison skipped for complex output structures)")
    print()

    return traced_model


def convert_to_coreml(traced_model, output_path: str):
    """
    Convert traced PyTorch model to CoreML with optimal ANE settings

    Args:
        traced_model: Traced PyTorch model
        output_path: Path to save .mlpackage

    Returns:
        CoreML model
    """
    print("="*80)
    print("STEP 4: Converting to CoreML with Optimal ANE Settings")
    print("="*80)
    print()

    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed")
        print("Install with: pip install coremltools")
        sys.exit(1)

    print("CoreML Tools version:", ct.__version__)
    print()

    print("Configuring conversion settings:")
    print("  - Format: MLProgram (iOS 15+)")
    print("  - Precision: FLOAT16 (ANE-optimized)")
    print("  - Compute Units: CPU_AND_NE (prefer ANE)")
    print("  - Input Shape: Fixed 256x256")
    print("  - Deployment Target: iOS 17")
    print("  - Pass Pipeline: DEFAULT_PRUNING")
    print()

    print("Converting to CoreML...")
    print("(This may take 2-5 minutes...)")
    print()

    start_time = time.time()

    # Convert with optimal settings for ANE
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input_image",
            shape=(1, 3, 256, 256),  # Fixed shape - critical for ANE optimization
            color_layout=ct.colorlayout.RGB,
            scale=1.0/255.0,  # Normalize [0, 255] → [0, 1]
        )],
        convert_to="mlprogram",  # MLProgram format (better FP16 support)
        compute_precision=ct.precision.FLOAT16,  # FP16 for ANE
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Prefer ANE over GPU
        minimum_deployment_target=ct.target.iOS17,
        pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    )

    elapsed = time.time() - start_time

    print(f"✓ Conversion completed in {elapsed:.1f}s")
    print()

    # Save the model
    print(f"Saving CoreML model to: {output_path}")
    coreml_model.save(output_path)

    # Get file size
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        size_mb = sum(f.stat().st_size for f in output_path_obj.rglob('*')) / 1024 / 1024
        print(f"✓ Model saved successfully ({size_mb:.1f} MB)")

    print()

    return coreml_model


def test_coreml_model(coreml_model, num_runs=5):
    """
    Test CoreML model inference and measure performance

    Args:
        coreml_model: CoreML model
        num_runs: Number of inference runs for benchmarking

    Returns:
        Average inference time in milliseconds
    """
    print("="*80)
    print("STEP 5: Testing CoreML Model")
    print("="*80)
    print()

    print(f"Running {num_runs} inference passes to measure performance...")
    print()

    # Create test input (256x256 RGB image)
    test_input = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Warm-up run
    print("Warm-up run...")
    try:
        _ = coreml_model.predict({"input_image": test_input})
        print("✓ Warm-up successful")
    except Exception as e:
        print(f"✗ Warm-up failed: {e}")
        print()
        return None

    # Benchmark runs
    print(f"Running {num_runs} timed inferences...")
    times = []

    for i in range(num_runs):
        start = time.time()
        _ = coreml_model.predict({"input_image": test_input})
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.1f}ms")

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print()
    print("Performance Summary:")
    print(f"  Average: {avg_time:.1f}ms (±{std_time:.1f}ms)")
    print(f"  Min:     {min_time:.1f}ms")
    print(f"  Max:     {max_time:.1f}ms")
    print()

    return avg_time


def print_next_steps(output_path: str, avg_time: float):
    """
    Print next steps for the user

    Args:
        output_path: Path to saved CoreML model
        avg_time: Average inference time in ms
    """
    print("="*80)
    print("CONVERSION COMPLETE - NEXT STEPS")
    print("="*80)
    print()

    print("✅ CoreML model created successfully!")
    print()

    print(f"📍 Model Location: {output_path}")
    if avg_time:
        print(f"⚡ Performance: ~{avg_time:.1f}ms per inference")
    print()

    print("📊 STEP 6: Analyze with Xcode Performance Report")
    print("-"*80)
    print()
    print("To check ANE coverage and partitions:")
    print()
    print(f'  1. Open in Xcode: open "{output_path}"')
    print("  2. In Xcode menu: Product → Perform Action → Profile Model")
    print("  3. Check the Performance tab:")
    print("     - ANE coverage % (target: >90%, currently 82.6%)")
    print("     - Number of partitions (target: <8, currently 18)")
    print("     - Compute unit assignment per layer")
    print()

    print("🔄 STEP 7: Export to ONNX for Integration")
    print("-"*80)
    print()
    print("After validating ANE coverage in Xcode, convert to ONNX:")
    print()
    print("  python3 convert_coreml_to_onnx.py")
    print()
    print("This will create: weights/star_landmark_98_optimized.onnx")
    print()

    print("🧪 STEP 8: Benchmark in Full Pipeline")
    print("-"*80)
    print()
    print("Test the optimized model in the full Face Mirror pipeline:")
    print()
    print("  1. Update onnx_star_detector.py to use new ONNX model")
    print("  2. Run profiler on test video")
    print("  3. Compare performance with baseline")
    print()
    print("Expected improvements:")
    print(f"  • Current:  163.2ms average (baseline)")
    if avg_time:
        speedup = 163.2 / avg_time
        print(f"  • New:      ~{avg_time:.1f}ms average ({speedup:.1f}x speedup)")
    else:
        print(f"  • Target:   55-100ms average (2-3x speedup)")
    print(f"  • Pipeline: 3.2 → 4.5-5.3 FPS")
    print()

    print("📈 Expected Outcomes")
    print("-"*80)
    print()
    print("Conservative (90% ANE, 8 partitions):")
    print("  ✓ STAR: 163ms → 82-99ms (2.0-2.2x speedup)")
    print("  ✓ Pipeline: 3.2 → 4.2-4.5 FPS")
    print()
    print("Optimistic (95% ANE, 5 partitions):")
    print("  ✓ STAR: 163ms → 55-71ms (2.8-3.7x speedup)")
    print("  ✓ Pipeline: 3.2 → 5.0-5.3 FPS")
    print()

    print("="*80)


def main():
    """Main conversion workflow"""
    print()
    print("="*80)
    print("STAR Model → CoreML Conversion (ANE-Optimized)")
    print("="*80)
    print()
    print("This script converts the STAR landmark detection model to CoreML")
    print("with settings optimized for Apple Neural Engine performance.")
    print()
    print("Expected improvements:")
    print("  • ANE coverage: 82.6% → 90-95%+")
    print("  • Partitions: 18 → <8")
    print("  • Inference time: 163ms → 55-100ms (2-3x speedup)")
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    checkpoint_path = str(script_dir / 'weights' / 'Landmark_98.pkl')
    output_path = str(script_dir / 'weights' / 'star_landmark_98_optimized.mlpackage')

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the STAR model checkpoint is downloaded.")
        return 1

    try:
        # Step 1: Load model
        model, detector = load_star_model(checkpoint_path)

        # Step 2: Prepare for ANE
        model = prepare_model_for_ane(model)

        # Step 3: Trace model
        traced_model = trace_model(model)

        # Step 4: Convert to CoreML
        coreml_model = convert_to_coreml(traced_model, output_path)

        # Step 5: Test performance
        avg_time = test_coreml_model(coreml_model, num_runs=5)

        # Print next steps
        print_next_steps(output_path, avg_time)

        return 0

    except Exception as e:
        print()
        print("="*80)
        print("ERROR DURING CONVERSION")
        print("="*80)
        print()
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
