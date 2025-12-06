"""
Local Test Script for HPC Tools

Tests the diagnostic and ablation tools on a small subset of frames
before deploying to Big Red 200.

Run from project root:
    python bigred200/test_hpc_tools_local.py
"""

import sys
import os

# Add paths
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')
sys.path.insert(0, 'pyfaceau')
sys.path.insert(0, 'pyfhog')

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import time


def test_diagnostic_tool(max_frames: int = 5):
    """Test the diagnostic data collection tool."""
    print("\n" + "=" * 60)
    print("TEST 1: Diagnostic Tool")
    print("=" * 60)

    from bigred200.diagnostic.data_structures import FrameDiagnostic, LandmarkDiagnostic
    from bigred200.diagnostic.instrumented_optimizer import InstrumentedNURLMSOptimizer
    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser

    # Load C++ reference
    cpp_csv = pd.read_csv("validation_output_0942/IMG_0942.csv")
    cpp_landmarks = np.zeros((len(cpp_csv), 68, 2))
    for i in range(68):
        cpp_landmarks[:, i, 0] = cpp_csv[f'x_{i}'].values
        cpp_landmarks[:, i, 1] = cpp_csv[f'y_{i}'].values

    print(f"Loaded C++ reference: {len(cpp_csv)} frames")

    # Initialize pipeline with instrumented optimizer
    print("Initializing pipeline with instrumented optimizer...")
    detector = MTCNN()

    # Create CLNF and replace optimizer
    clnf = CLNF(model_dir="pyclnf/pyclnf/models")
    clnf.optimizer = InstrumentedNURLMSOptimizer(
        capture_response_maps=True,
        target_landmarks=list(range(68)),  # All landmarks
        regularization=clnf.optimizer.regularization,
        max_iterations=clnf.optimizer.max_iterations,
        convergence_threshold=clnf.optimizer.convergence_threshold,
        sigma=clnf.optimizer.sigma,
        weight_multiplier=clnf.optimizer.weight_multiplier,
    )

    pdm_parser = PDMParser("pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")
    calc_params = CalcParams(pdm_parser)

    # Open video
    video_path = "S Data/Normal Cohort/IMG_0942.MOV"
    if not os.path.exists(video_path):
        video_path = "Patient Data/Normal Cohort/IMG_0942.MOV"

    cap = cv2.VideoCapture(video_path)
    print(f"Opened video: {video_path}")

    # Process frames
    diagnostics = []
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n  Processing frame {frame_idx}...")

        # Set context
        clnf.optimizer.set_frame_context(frame_idx, cpp_landmarks[frame_idx])

        # Detection
        bboxes, _ = detector.detect(frame)
        if bboxes is None or len(bboxes) == 0:
            print(f"    No face detected")
            continue

        # CLNF fitting
        start = time.time()
        py_landmarks, info = clnf.fit(frame, bboxes[0][:4])
        elapsed = time.time() - start

        # Get diagnostics
        iter_diagnostics = clnf.optimizer.get_diagnostics()

        # Create frame diagnostic
        diag = FrameDiagnostic(
            frame_idx=frame_idx,
            cpp_landmarks=cpp_landmarks[frame_idx],
            py_landmarks=py_landmarks,
            iterations=iter_diagnostics,
        )
        diag.compute_errors()
        diagnostics.append(diag)

        # Print summary
        region_errors = diag.get_region_errors()
        print(f"    Time: {elapsed*1000:.1f}ms")
        print(f"    Mean error: {np.mean(diag.landmark_errors):.3f} px")
        print(f"    Jaw error: {region_errors.get('jaw', 0):.3f} px")
        print(f"    Iterations captured: {len(iter_diagnostics)}")

        # Check response map capture
        if iter_diagnostics:
            n_landmarks_captured = len(iter_diagnostics[0].landmarks)
            sample_lm = list(iter_diagnostics[0].landmarks.values())[0] if iter_diagnostics[0].landmarks else None
            has_response_map = sample_lm is not None and sample_lm.response_map is not None
            print(f"    Landmarks captured: {n_landmarks_captured}")
            print(f"    Response maps: {'Yes' if has_response_map else 'No'}")

    cap.release()

    print(f"\n  DIAGNOSTIC TEST: PASSED ({len(diagnostics)} frames processed)")
    return True


def test_ablation_generator():
    """Test the ablation experiment generator."""
    print("\n" + "=" * 60)
    print("TEST 2: Ablation Experiment Generator")
    print("=" * 60)

    from bigred200.ablation.generate_experiments import (
        create_experiment_manifest,
        generate_latin_hypercube,
        DEFAULT_PARAMETER_RANGES,
    )

    # Generate small test manifest
    print("Generating Latin Hypercube sample (20 configs)...")
    df = generate_latin_hypercube(DEFAULT_PARAMETER_RANGES, n_samples=20, seed=42)

    print(f"\nGenerated {len(df)} configurations:")
    print(df.head(10).to_string())

    # Verify all parameters present
    expected_params = ['sigma', 'regularization', 'max_iterations',
                       'convergence_threshold', 'weight_multiplier']
    for param in expected_params:
        if param not in df.columns:
            print(f"  ERROR: Missing parameter {param}")
            return False

    print(f"\n  GENERATOR TEST: PASSED")
    return True


def test_single_experiment():
    """Test running a single ablation experiment."""
    print("\n" + "=" * 60)
    print("TEST 3: Single Ablation Experiment")
    print("=" * 60)

    from bigred200.ablation.run_single_experiment import run_experiment

    # Define test config
    config = {
        'experiment_id': 0,
        'sigma': 2.0,
        'regularization': 15,
        'max_iterations': 15,
        'convergence_threshold': 0.05,
        'weight_multiplier': 5.0,
    }

    print(f"Config: {config}")

    # Find video path
    video_path = "S Data/Normal Cohort/IMG_0942.MOV"
    if not os.path.exists(video_path):
        video_path = "Patient Data/Normal Cohort/IMG_0942.MOV"

    cpp_csv_path = "validation_output_0942/IMG_0942.csv"

    print(f"\nRunning experiment on 10 frames...")
    start = time.time()

    results = run_experiment(
        config=config,
        video_path=video_path,
        cpp_csv_path=cpp_csv_path,
        max_frames=10,
    )

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Frames: {results['timing']['frames_processed']}/{results['timing']['total_frames']}")

    if results['metrics']['overall']['mean_error'] is not None:
        print(f"  Mean error: {results['metrics']['overall']['mean_error']:.3f} px")
        print(f"  Jaw error: {results['metrics']['regions']['jaw']['mean_error']:.3f} px")
        print(f"\n  EXPERIMENT TEST: PASSED")
        return True
    else:
        print("  ERROR: No valid frames processed")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HPC TOOLS LOCAL TEST")
    print("=" * 60)

    results = {}

    # Test 1: Diagnostic tool
    try:
        results['diagnostic'] = test_diagnostic_tool(max_frames=3)
    except Exception as e:
        print(f"\n  DIAGNOSTIC TEST: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results['diagnostic'] = False

    # Test 2: Ablation generator
    try:
        results['generator'] = test_ablation_generator()
    except Exception as e:
        print(f"\n  GENERATOR TEST: FAILED - {e}")
        results['generator'] = False

    # Test 3: Single experiment
    try:
        results['experiment'] = test_single_experiment()
    except Exception as e:
        print(f"\n  EXPERIMENT TEST: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results['experiment'] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Ready for Big Red 200 deployment.")
    else:
        print("\nSome tests failed. Please fix before deploying.")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
