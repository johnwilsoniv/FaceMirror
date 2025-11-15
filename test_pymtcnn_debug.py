"""Test PyMTCNN debug mode implementation"""

import cv2
import numpy as np
from pymtcnn import MTCNN

# Load a test image
test_image_path = "test_images/test_frame.jpg"
img = cv2.imread(test_image_path)

if img is None:
    print(f"❌ Could not load test image: {test_image_path}")
    exit(1)

print(f"✓ Loaded test image: {img.shape}")
print()

# Test ONNX backend debug mode
print("=" * 60)
print("Testing ONNX Backend Debug Mode")
print("=" * 60)

detector_onnx = MTCNN(backend='onnx', verbose=True)
print()

# Test with return_debug=True
bboxes, landmarks, debug_info = detector_onnx.detect(img, return_debug=True)

print(f"\n✓ ONNX Detection Results:")
print(f"  Final boxes: {len(bboxes)}")
print(f"  Final landmarks: {len(landmarks)}")
print()

print("Debug Info:")
for stage in ['pnet', 'rnet', 'onet', 'final']:
    info = debug_info[stage]
    print(f"  {stage.upper()}:")
    print(f"    num_boxes: {info['num_boxes']}")
    if 'time_ms' in info:
        print(f"    time_ms: {info['time_ms']:.2f}ms")
    if 'total_time_ms' in info:
        print(f"    total_time_ms: {info['total_time_ms']:.2f}ms")

print()

# Test CoreML backend if on macOS
import platform
if platform.system() == 'Darwin':
    print("=" * 60)
    print("Testing CoreML Backend Debug Mode")
    print("=" * 60)

    try:
        detector_coreml = MTCNN(backend='coreml', verbose=True)
        print()

        bboxes, landmarks, debug_info = detector_coreml.detect(img, return_debug=True)

        print(f"\n✓ CoreML Detection Results:")
        print(f"  Final boxes: {len(bboxes)}")
        print(f"  Final landmarks: {len(landmarks)}")
        print()

        print("Debug Info:")
        for stage in ['pnet', 'rnet', 'onet', 'final']:
            info = debug_info[stage]
            print(f"  {stage.upper()}:")
            print(f"    num_boxes: {info['num_boxes']}")
            if 'time_ms' in info:
                print(f"    time_ms: {info['time_ms']:.2f}ms")
            if 'total_time_ms' in info:
                print(f"    total_time_ms: {info['total_time_ms']:.2f}ms")

        print("\n✓ CoreML debug mode works!")

    except ImportError as e:
        print(f"⚠ CoreML not available: {e}")

print()
print("=" * 60)
print("✓ All debug mode tests passed!")
print("=" * 60)
