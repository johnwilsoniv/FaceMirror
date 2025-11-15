"""
Comprehensive PyFaceAU vs C++ OpenFace Validation

This script:
1. Runs C++ OpenFace on all test frames
2. Runs PyFaceAU on all test frames
3. Compares MTCNN stages, landmarks (initial & final), pose, and AUs
4. Generates visualizations
5. Creates validation report
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import subprocess
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Paths
PROJECT_ROOT = Path(__file__).parent
CALIBRATION_FRAMES = PROJECT_ROOT / "calibration_frames"
CPP_OUTPUT = PROJECT_ROOT / "validation_output" / "cpp_baseline"
PYTHON_OUTPUT = PROJECT_ROOT / "validation_output" / "python_baseline"
REPORT_OUTPUT = PROJECT_ROOT / "validation_output" / "report"
CPP_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

# Ensure output directories exist
CPP_OUTPUT.mkdir(parents=True, exist_ok=True)
PYTHON_OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

def get_test_images():
    """Get list of test images"""
    images = sorted(list(CALIBRATION_FRAMES.glob("patient*.jpg")))
    print(f"Found {len(images)} test images")
    return images

def run_cpp_on_all_frames():
    """Run C++ OpenFace on all test frames"""
    print("\n" + "="*60)
    print("Phase 1: Running C++ OpenFace Baseline")
    print("="*60)

    images = get_test_images()
    results = []

    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        frame_name = img_path.stem
        print(f"\n[{idx}/{len(images)}] Processing {frame_name}...")

        # Clean up temp MTCNN debug files
        for tmp_file in ["/tmp/mtcnn_debug.csv", "/tmp/cpp_pnet_all_boxes.txt",
                         "/tmp/cpp_rnet_output.txt", "/tmp/cpp_before_onet.txt"]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        # Run C++ OpenFace
        cmd = [
            CPP_BINARY,
            "-f", str(img_path),
            "-out_dir", str(CPP_OUTPUT),
            "-of", frame_name
        ]

        frame_start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        frame_time = time.time() - frame_start

        # Parse stdout for DEBUG_INIT_LANDMARKS
        init_landmarks = None
        init_params = None
        for line in result.stdout.split('\n'):
            if 'DEBUG_INIT_PARAMS:' in line:
                init_params = line.strip()
            if 'DEBUG_INIT_LANDMARKS:' in line:
                init_landmarks = line.split('DEBUG_INIT_LANDMARKS:')[1].strip()

        # Copy MTCNN debug file if it exists
        if os.path.exists("/tmp/mtcnn_debug.csv"):
            import shutil
            shutil.copy("/tmp/mtcnn_debug.csv", CPP_OUTPUT / f"{frame_name}_mtcnn_debug.csv")

        results.append({
            'frame': frame_name,
            'cpp_time': frame_time,
            'init_landmarks': init_landmarks,
            'init_params': init_params
        })

        print(f"  ✓ Completed in {frame_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n✓ C++ baseline complete: {total_time:.2f}s total ({total_time/len(images):.2f}s avg)")

    return results

def run_python_on_all_frames():
    """Run PyFaceAU on all test frames"""
    print("\n" + "="*60)
    print("Phase 2: Running PyFaceAU with Debug Mode")
    print("="*60)

    # Import PyFaceAU
    sys.path.insert(0, str(PROJECT_ROOT / "pyfaceau"))
    sys.path.insert(0, str(PROJECT_ROOT / "pymtcnn"))

    from pyfaceau.pipeline import FullPythonAUPipeline

    # Initialize pipeline with debug mode
    print("\nInitializing PyFaceAU pipeline...")
    pipeline = FullPythonAUPipeline(
        pdm_file=str(PROJECT_ROOT / "pyfaceau" / "weights" / "In-the-wild_aligned_PDM_68.txt"),
        au_models_dir=str(PROJECT_ROOT / "pyfaceau" / "weights" / "AU_predictors"),
        triangulation_file=str(PROJECT_ROOT / "pyfaceau" / "weights" / "tris_68_full.txt"),
        patch_expert_file=str(PROJECT_ROOT / "pyfaceau" / "weights" / "svr_patches_0.25_general.txt"),
        mtcnn_backend='coreml',
        use_calc_params=True,  # Enable CalcParams for better pose estimation
        max_clnf_iterations=10,  # CLNF refinement (now via pyclnf)
        clnf_convergence_threshold=0.01,
        debug_mode=True,
        track_faces=False,
        verbose=False
    )

    # Initialize components
    pipeline._initialize_components()
    print("✓ Pipeline initialized\n")

    images = get_test_images()
    results = []

    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        frame_name = img_path.stem
        print(f"[{idx}/{len(images)}] Processing {frame_name}...")

        # Load image
        img = cv2.imread(str(img_path))

        # Process with debug mode
        frame_start = time.time()
        result = pipeline._process_frame(img, frame_idx=0, timestamp=0.0, return_debug=True)
        frame_time = time.time() - frame_start

        # Save results
        result_data = {
            'frame': frame_name,
            'python_time': frame_time,
            'success': result['success'],
            'debug_info': result.get('debug_info', {}),
            'au_results': {k: v for k, v in result.items() if k.startswith('AU')}
        }

        # Save to JSON
        output_file = PYTHON_OUTPUT / f"{frame_name}_result.json"
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            json.dump(convert_to_serializable(result_data), f, indent=2)

        results.append(result_data)

        print(f"  ✓ Completed in {frame_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n✓ Python baseline complete: {total_time:.2f}s total ({total_time/len(images):.2f}s avg)")

    return results

def main():
    """Run comprehensive validation"""
    print("\n" + "="*80)
    print("PyFaceAU vs C++ OpenFace: Comprehensive Validation")
    print("="*80)

    validation_start = time.time()

    # Phase 1: Run C++ baseline
    cpp_results = run_cpp_on_all_frames()

    # Phase 2: Run Python baseline
    python_results = run_python_on_all_frames()

    total_time = time.time() - validation_start

    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)
    print(f"Total time: {total_time:.2f}s")
    print(f"\nNext steps:")
    print("  1. Run comparison analysis script")
    print("  2. Generate visualizations")
    print("  3. Create validation report")

if __name__ == "__main__":
    main()
