"""
Phase 1: MTCNN Regression Diagnostics

Quick diagnostics to identify why ONNX/CoreML backends diverge from C++ baseline
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add pymtcnn to path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

print("="*80)
print("MTCNN Regression Diagnostics - Phase 1")
print("="*80)

# Test 1: Check ONNX Runtime version
print("\n" + "="*80)
print("Test 1: ONNX Runtime Version")
print("="*80)
try:
    import onnxruntime
    print(f"✓ ONNX Runtime version: {onnxruntime.__version__}")
    print(f"  Available providers: {onnxruntime.get_available_providers()}")
except ImportError as e:
    print(f"✗ ONNX Runtime not available: {e}")

# Test 2: Check CoreML availability
print("\n" + "="*80)
print("Test 2: CoreML Availability")
print("="*80)
try:
    import coremltools
    print(f"✓ CoreML Tools version: {coremltools.__version__}")
except ImportError as e:
    print(f"✗ CoreML Tools not available: {e}")

# Test 3: Verify model files exist
print("\n" + "="*80)
print("Test 3: Model Files")
print("="*80)

pymtcnn_root = Path(__file__).parent / "pymtcnn" / "pymtcnn"
onnx_models_dir = pymtcnn_root / "models" / "onnx"
coreml_models_dir = pymtcnn_root / "models" / "coreml"

print(f"\nONNX models directory: {onnx_models_dir}")
if onnx_models_dir.exists():
    for model_file in sorted(onnx_models_dir.glob("*.onnx")):
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"  ✓ {model_file.name}: {size_mb:.2f} MB")
else:
    print(f"  ✗ Directory not found!")

print(f"\nCoreML models directory: {coreml_models_dir}")
if coreml_models_dir.exists():
    for model_dir in sorted(coreml_models_dir.glob("*.mlpackage")):
        print(f"  ✓ {model_dir.name}")
else:
    print(f"  ✗ Directory not found!")

# Test 4: Check thresholds in backend code
print("\n" + "="*80)
print("Test 4: Backend Thresholds")
print("="*80)

try:
    from pymtcnn.backends.coreml_backend import CoreMLMTCNN
    from pymtcnn.backends.onnx_backend import ONNXMTCNN

    # Create detectors to check default thresholds
    coreml_detector = CoreMLMTCNN()
    onnx_detector = ONNXMTCNN()

    print(f"\nCoreML thresholds: {coreml_detector.thresholds}")
    print(f"ONNX thresholds:   {onnx_detector.thresholds}")

    if coreml_detector.thresholds == onnx_detector.thresholds:
        print("✓ Thresholds match between backends")
    else:
        print("✗ WARNING: Threshold mismatch detected!")

    print(f"\nCoreML min_face_size: {coreml_detector.min_face_size}")
    print(f"ONNX min_face_size:   {onnx_detector.min_face_size}")

    print(f"\nCoreML factor: {coreml_detector.factor}")
    print(f"ONNX factor:   {onnx_detector.factor}")

except Exception as e:
    print(f"✗ Error loading backends: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Run both backends on test image with debug mode
print("\n" + "="*80)
print("Test 5: Backend Detection Comparison")
print("="*80)

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

if not os.path.exists(TEST_IMAGE):
    print(f"✗ Test image not found: {TEST_IMAGE}")
else:
    print(f"\nTest image: {TEST_IMAGE}")
    img = cv2.imread(TEST_IMAGE)
    print(f"Image shape: {img.shape}")

    try:
        from pymtcnn import MTCNN

        # Test CoreML
        print("\n--- CoreML Backend ---")
        detector_coreml = MTCNN(backend='coreml', debug_mode=True)
        result_coreml = detector_coreml.detect(img)

        if len(result_coreml) == 3:
            bboxes_coreml, landmarks_coreml, debug_coreml = result_coreml
            print(f"PNet boxes: {debug_coreml.get('pnet', {}).get('num_boxes', 'N/A')}")
            print(f"RNet boxes: {debug_coreml.get('rnet', {}).get('num_boxes', 'N/A')}")
            print(f"ONet boxes: {debug_coreml.get('onet', {}).get('num_boxes', 'N/A')}")
            print(f"Final boxes: {len(bboxes_coreml)}")
            if len(bboxes_coreml) > 0:
                print(f"BBox: {bboxes_coreml[0]}")
        else:
            bboxes_coreml, landmarks_coreml = result_coreml
            print(f"Final boxes: {len(bboxes_coreml)}")
            print("✗ Debug info not returned")

        # Test ONNX
        print("\n--- ONNX Backend ---")
        detector_onnx = MTCNN(backend='onnx', debug_mode=True)
        result_onnx = detector_onnx.detect(img)

        if len(result_onnx) == 3:
            bboxes_onnx, landmarks_onnx, debug_onnx = result_onnx
            print(f"PNet boxes: {debug_onnx.get('pnet', {}).get('num_boxes', 'N/A')}")
            print(f"RNet boxes: {debug_onnx.get('rnet', {}).get('num_boxes', 'N/A')}")
            print(f"ONet boxes: {debug_onnx.get('onet', {}).get('num_boxes', 'N/A')}")
            print(f"Final boxes: {len(bboxes_onnx)}")
            if len(bboxes_onnx) > 0:
                print(f"BBox[0]: {bboxes_onnx[0]}")
                if len(bboxes_onnx) > 1:
                    print(f"BBox[1]: {bboxes_onnx[1]}")
                if len(bboxes_onnx) > 2:
                    print(f"BBox[2]: {bboxes_onnx[2]}")
        else:
            bboxes_onnx, landmarks_onnx = result_onnx
            print(f"Final boxes: {len(bboxes_onnx)}")
            print("✗ Debug info not returned")

        # Compare results
        print("\n--- Comparison ---")
        print(f"C++ baseline: PNet=95, RNet=95, ONet=1, Final=1")
        print(f"CoreML: PNet={debug_coreml.get('pnet', {}).get('num_boxes', '?')}, RNet={debug_coreml.get('rnet', {}).get('num_boxes', '?')}, ONet={debug_coreml.get('onet', {}).get('num_boxes', '?')}, Final={len(bboxes_coreml)}")
        print(f"ONNX:   PNet={debug_onnx.get('pnet', {}).get('num_boxes', '?')}, RNet={debug_onnx.get('rnet', {}).get('num_boxes', '?')}, ONet={debug_onnx.get('onet', {}).get('num_boxes', '?')}, Final={len(bboxes_onnx)}")

        # Calculate IoU if both detected faces
        if len(bboxes_coreml) > 0 and len(bboxes_onnx) > 0:
            from compare_mtcnn_simple import compute_iou
            cpp_bbox = [301.938, 782.149, 400.586, 400.585]
            iou_coreml = compute_iou(cpp_bbox, bboxes_coreml[0])
            iou_onnx = compute_iou(cpp_bbox, bboxes_onnx[0])
            print(f"\nIoU vs C++ baseline:")
            print(f"  CoreML: {iou_coreml:.4f} {'✓ PASS' if iou_coreml > 0.9 else '✗ FAIL'}")
            print(f"  ONNX:   {iou_onnx:.4f} {'✓ PASS' if iou_onnx > 0.9 else '✗ FAIL'}")

    except Exception as e:
        print(f"✗ Error running detection: {e}")
        import traceback
        traceback.print_exc()

# Test 6: Check for preprocessing differences
print("\n" + "="*80)
print("Test 6: Preprocessing Check")
print("="*80)

try:
    from pymtcnn.backends.coreml_backend import CoreMLMTCNN
    from pymtcnn.backends.onnx_backend import ONNXMTCNN

    # Check if both have same preprocessing method
    import inspect

    print("\nCoreML _preprocess signature:")
    coreml_preprocess = inspect.signature(CoreMLMTCNN._preprocess)
    print(f"  {coreml_preprocess}")

    print("\nONNX _preprocess signature:")
    onnx_preprocess = inspect.signature(ONNXMTCNN._preprocess)
    print(f"  {onnx_preprocess}")

except Exception as e:
    print(f"✗ Error checking preprocessing: {e}")

print("\n" + "="*80)
print("Diagnostics Complete")
print("="*80)
