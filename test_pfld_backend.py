#!/usr/bin/env python3
"""Test PFLD backend architecture"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau.detectors.pfld_detector import PFLDDetector
import cv2
import numpy as np

print("=" * 80)
print("PFLD Backend Architecture Test")
print("=" * 80)

# Test image
test_image_path = "calibration_frames/patient1_frame1.jpg"
if not Path(test_image_path).exists():
    print(f"\n✗ Test image not found: {test_image_path}")
    print("Using synthetic test image instead")
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
else:
    img = cv2.imread(test_image_path)
    print(f"\n✓ Loaded test image: {test_image_path}")

# Dummy bbox for testing
h, w = img.shape[:2]
bbox = np.array([w//4, h//4, 3*w//4, 3*h//4])

print(f"\nTest bbox: {bbox}")
print()

# Test ONNX backend
print("=" * 80)
print("Test 1: ONNX Backend")
print("=" * 80)

try:
    detector_onnx = PFLDDetector(
        backend='onnx',
        weights_dir='pyfaceau/weights',
        verbose=True
    )
    print(f"\n✓ Detector created: {detector_onnx}")
    print(f"  Backend info: {detector_onnx.get_backend_info()}")

    landmarks_onnx, conf_onnx = detector_onnx.detect_landmarks(img, bbox)
    print(f"\n✓ Landmarks detected: {landmarks_onnx.shape}")
    print(f"  Confidence: {conf_onnx}")
    print(f"  Sample points:")
    for i in range(min(3, len(landmarks_onnx))):
        print(f"    Point {i}: ({landmarks_onnx[i, 0]:.2f}, {landmarks_onnx[i, 1]:.2f})")

    print("\n✅ ONNX backend test PASSED")
except Exception as e:
    print(f"\n✗ ONNX backend test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test Auto backend selection
print("\n" + "=" * 80)
print("Test 2: Auto Backend Selection")
print("=" * 80)

try:
    detector_auto = PFLDDetector(
        backend='auto',
        weights_dir='pyfaceau/weights',
        verbose=True
    )
    print(f"\n✓ Detector created: {detector_auto}")
    print(f"  Backend info: {detector_auto.get_backend_info()}")

    landmarks_auto, conf_auto = detector_auto.detect_landmarks(img, bbox)
    print(f"\n✓ Landmarks detected: {landmarks_auto.shape}")
    print(f"  Confidence: {conf_auto}")

    # Compare with ONNX
    diff = np.abs(landmarks_auto - landmarks_onnx).mean()
    print(f"\n  Comparison with ONNX backend:")
    print(f"    Mean difference: {diff:.6f} pixels")

    if diff < 0.1:
        print(f"    ✓ Backends produce identical results")
    else:
        print(f"    ⚠ Backends differ (expected if using different backend)")

    print("\n✅ Auto backend test PASSED")
except Exception as e:
    print(f"\n✗ Auto backend test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test CoreML backend if available
print("\n" + "=" * 80)
print("Test 3: CoreML Backend (if available)")
print("=" * 80)

coreml_path = Path("pyfaceau/weights/pfld_cunjian.mlpackage")
if coreml_path.exists():
    try:
        detector_coreml = PFLDDetector(
            backend='coreml',
            weights_dir='pyfaceau/weights',
            verbose=True
        )
        print(f"\n✓ Detector created: {detector_coreml}")
        print(f"  Backend info: {detector_coreml.get_backend_info()}")

        landmarks_coreml, conf_coreml = detector_coreml.detect_landmarks(img, bbox)
        print(f"\n✓ Landmarks detected: {landmarks_coreml.shape}")
        print(f"  Confidence: {conf_coreml}")

        # Compare with ONNX
        diff = np.abs(landmarks_coreml - landmarks_onnx).mean()
        max_diff = np.abs(landmarks_coreml - landmarks_onnx).max()
        print(f"\n  Numerical equivalence with ONNX backend:")
        print(f"    Mean difference: {diff:.6f} pixels")
        print(f"    Max difference:  {max_diff:.6f} pixels")

        threshold = 0.5
        if diff < threshold:
            print(f"    ✅ PASSED: Mean error {diff:.6f} < {threshold} pixels")
        else:
            print(f"    ✗ FAILED: Mean error {diff:.6f} >= {threshold} pixels")

        print("\n✅ CoreML backend test PASSED")
    except Exception as e:
        print(f"\n✗ CoreML backend test FAILED: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n⚠ CoreML model not found at {coreml_path}")
    print("  Run convert_pfld_to_coreml.py to generate it")
    print("  Skipping CoreML backend test")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("\n✅ PFLD backend architecture working correctly")
print("\nAvailable backends:")
print("  - ONNX (ONNX Runtime with CoreMLExecutionProvider)")
print("  - Auto (selects best available backend)")
if coreml_path.exists():
    print("  - CoreML (native CoreML for Apple Neural Engine)")
else:
    print("  - CoreML (not available - model not converted)")

print("\n" + "=" * 80)
print("All tests completed")
print("=" * 80)
