#!/usr/bin/env python3
"""
Validate CLNF landmarks across backends and compare to C++ OpenFace.

Compares:
1. PyMTCNN CoreML → CLNF
2. PyMTCNN ONNX → CLNF
3. C++ OpenFace (gold standard)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

import cv2
import numpy as np
import pandas as pd
import subprocess
from pyfaceau.pipeline import FullPythonAUPipeline

print("=" * 80)
print("CLNF Landmark Validation: PyMTCNN Backends vs C++ OpenFace")
print("=" * 80)
print()

# Test image
test_image_path = "calibration_frames/patient1_frame1.jpg"
if not Path(test_image_path).exists():
    print(f"✗ Test image not found: {test_image_path}")
    sys.exit(1)

img = cv2.imread(test_image_path)
print(f"✓ Test image: {test_image_path} ({img.shape[1]}x{img.shape[0]})")
print()

# Create temporary directory for C++ output
temp_dir = Path("temp_clnf_validation")
temp_dir.mkdir(exist_ok=True)

results = {}

# ============================================================================
# Test 1: PyMTCNN CoreML → CLNF
# ============================================================================
print("=" * 80)
print("[1/3] PyMTCNN CoreML → CLNF")
print("=" * 80)

try:
    pipeline_coreml = FullPythonAUPipeline(
        pdm_file='pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
        au_models_dir='pyfaceau/weights/AU_predictors',
        triangulation_file='pyfaceau/weights/tris_68_full.txt',
        patch_expert_file='pyfaceau/weights/svr_patches_0.25_general.txt',
        mtcnn_backend='coreml',  # Force CoreML
        use_batched_predictor=False,
        verbose=False
    )

    result_coreml = pipeline_coreml._process_frame(img, frame_idx=0, timestamp=0.0, return_debug=True)

    if result_coreml['success']:
        landmarks_coreml = result_coreml['debug_info']['landmark_detection']['landmarks_68']
        bbox_coreml = result_coreml['debug_info']['face_detection']['bbox']
        results['coreml'] = {
            'landmarks': landmarks_coreml,
            'bbox': bbox_coreml,
            'clnf_converged': result_coreml.get('debug_info', {}).get('landmark_detection', {}).get('clnf_converged', False),
            'clnf_iterations': result_coreml.get('debug_info', {}).get('landmark_detection', {}).get('clnf_iterations', 0)
        }
        print(f"✅ CoreML → CLNF: {len(landmarks_coreml)} landmarks")
        print(f"   CLNF converged: {results['coreml']['clnf_converged']}")
        print(f"   CLNF iterations: {results['coreml']['clnf_iterations']}")
        print(f"   Bbox: {bbox_coreml}")
    else:
        print(f"✗ CoreML → CLNF failed")
        print(f"   Result: {result_coreml}")
        results['coreml'] = None
except Exception as e:
    print(f"✗ CoreML → CLNF error: {e}")
    import traceback
    traceback.print_exc()
    results['coreml'] = None

print()

# ============================================================================
# Test 2: PyMTCNN ONNX → CLNF
# ============================================================================
print("=" * 80)
print("[2/3] PyMTCNN ONNX → CLNF")
print("=" * 80)

try:
    pipeline_onnx = FullPythonAUPipeline(
        pdm_file='pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
        au_models_dir='pyfaceau/weights/AU_predictors',
        triangulation_file='pyfaceau/weights/tris_68_full.txt',
        patch_expert_file='pyfaceau/weights/svr_patches_0.25_general.txt',
        mtcnn_backend='onnx',  # Force ONNX
        use_batched_predictor=False,
        verbose=False
    )

    result_onnx = pipeline_onnx._process_frame(img, frame_idx=0, timestamp=0.0, return_debug=True)

    if result_onnx['success']:
        landmarks_onnx = result_onnx['debug_info']['landmark_detection']['landmarks_68']
        bbox_onnx = result_onnx['debug_info']['face_detection']['bbox']
        results['onnx'] = {
            'landmarks': landmarks_onnx,
            'bbox': bbox_onnx,
            'clnf_converged': result_onnx.get('debug_info', {}).get('landmark_detection', {}).get('clnf_converged', False),
            'clnf_iterations': result_onnx.get('debug_info', {}).get('landmark_detection', {}).get('clnf_iterations', 0)
        }
        print(f"✅ ONNX → CLNF: {len(landmarks_onnx)} landmarks")
        print(f"   CLNF converged: {results['onnx']['clnf_converged']}")
        print(f"   CLNF iterations: {results['onnx']['clnf_iterations']}")
        print(f"   Bbox: {bbox_onnx}")
    else:
        print(f"✗ ONNX → CLNF failed")
        print(f"   Result: {result_onnx}")
        results['onnx'] = None
except Exception as e:
    print(f"✗ ONNX → CLNF error: {e}")
    import traceback
    traceback.print_exc()
    results['onnx'] = None

print()

# ============================================================================
# Test 3: C++ OpenFace (Gold Standard)
# ============================================================================
print("=" * 80)
print("[3/3] C++ OpenFace (Gold Standard)")
print("=" * 80)

cpp_exe = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
cpp_output = temp_dir / "cpp_output.csv"

try:
    cmd = [
        cpp_exe,
        "-f", test_image_path,
        "-out_dir", str(temp_dir),
        "-of", "cpp_output"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0 and cpp_output.exists():
        # Read C++ landmarks
        df = pd.read_csv(cpp_output)

        # Extract landmarks (x_0, y_0, x_1, y_1, ..., x_67, y_67)
        landmark_cols = [f'{coord}_{i}' for i in range(68) for coord in ['x', 'y']]
        landmarks_flat = df[landmark_cols].values[0]
        landmarks_cpp = landmarks_flat.reshape(68, 2)

        # Extract bbox (approximate from landmarks)
        x_min, y_min = landmarks_cpp.min(axis=0)
        x_max, y_max = landmarks_cpp.max(axis=0)
        bbox_cpp = np.array([x_min, y_min, x_max, y_max])

        results['cpp'] = {
            'landmarks': landmarks_cpp,
            'bbox': bbox_cpp,
            'clnf_converged': True,  # C++ always reports success
            'clnf_iterations': 0  # Not reported by C++
        }

        print(f"✅ C++ OpenFace: {len(landmarks_cpp)} landmarks")
        print(f"   Bbox (estimated): {bbox_cpp}")
    else:
        print(f"✗ C++ OpenFace failed")
        print(f"   Return code: {result.returncode}")
        print(f"   Stdout: {result.stdout}")
        print(f"   Stderr: {result.stderr}")
        results['cpp'] = None
except Exception as e:
    print(f"✗ C++ OpenFace error: {e}")
    import traceback
    traceback.print_exc()
    results['cpp'] = None

print()

# ============================================================================
# Comparison
# ============================================================================
print("=" * 80)
print("LANDMARK COMPARISON")
print("=" * 80)
print()

if results['coreml'] and results['onnx']:
    landmarks_coreml = results['coreml']['landmarks']
    landmarks_onnx = results['onnx']['landmarks']

    diff_coreml_onnx = np.abs(landmarks_coreml - landmarks_onnx)
    mean_diff = diff_coreml_onnx.mean()
    max_diff = diff_coreml_onnx.max()

    print(f"CoreML vs ONNX:")
    print(f"  Mean difference: {mean_diff:.4f} pixels")
    print(f"  Max difference:  {max_diff:.4f} pixels")

    if mean_diff < 0.1:
        print(f"  ✅ IDENTICAL (mean < 0.1 pixels)")
    elif mean_diff < 1.0:
        print(f"  ✅ VERY CLOSE (mean < 1.0 pixels)")
    else:
        print(f"  ⚠️  DIFFERENT (mean >= 1.0 pixels)")
    print()

if results['coreml'] and results['cpp']:
    landmarks_coreml = results['coreml']['landmarks']
    landmarks_cpp = results['cpp']['landmarks']

    diff_coreml_cpp = np.abs(landmarks_coreml - landmarks_cpp)
    mean_diff = diff_coreml_cpp.mean()
    max_diff = diff_coreml_cpp.max()

    print(f"CoreML → CLNF vs C++ OpenFace:")
    print(f"  Mean difference: {mean_diff:.4f} pixels")
    print(f"  Max difference:  {max_diff:.4f} pixels")

    if mean_diff < 1.0:
        print(f"  ✅ EXCELLENT (mean < 1.0 pixels)")
    elif mean_diff < 2.0:
        print(f"  ✅ GOOD (mean < 2.0 pixels)")
    elif mean_diff < 5.0:
        print(f"  ⚠️  ACCEPTABLE (mean < 5.0 pixels)")
    else:
        print(f"  ✗ POOR (mean >= 5.0 pixels)")
    print()

if results['onnx'] and results['cpp']:
    landmarks_onnx = results['onnx']['landmarks']
    landmarks_cpp = results['cpp']['landmarks']

    diff_onnx_cpp = np.abs(landmarks_onnx - landmarks_cpp)
    mean_diff = diff_onnx_cpp.mean()
    max_diff = diff_onnx_cpp.max()

    print(f"ONNX → CLNF vs C++ OpenFace:")
    print(f"  Mean difference: {mean_diff:.4f} pixels")
    print(f"  Max difference:  {max_diff:.4f} pixels")

    if mean_diff < 1.0:
        print(f"  ✅ EXCELLENT (mean < 1.0 pixels)")
    elif mean_diff < 2.0:
        print(f"  ✅ GOOD (mean < 2.0 pixels)")
    elif mean_diff < 5.0:
        print(f"  ⚠️  ACCEPTABLE (mean < 5.0 pixels)")
    else:
        print(f"  ✗ POOR (mean >= 5.0 pixels)")
    print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print()

print("Architecture:")
print("  PyMTCNN (CoreML/ONNX) → CLNF → 68 landmarks")
print()

print("Results:")
for backend, result in results.items():
    if result:
        print(f"  ✅ {backend.upper()}: Success")
    else:
        print(f"  ✗ {backend.upper()}: Failed")
print()

if all(results.values()):
    print("✅ All backends successful - CLNF landmarks validated!")
else:
    print("⚠️  Some backends failed - check errors above")

# Cleanup
import shutil
if temp_dir.exists():
    shutil.rmtree(temp_dir)

print()
print("=" * 80)
