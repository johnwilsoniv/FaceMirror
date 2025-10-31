#!/usr/bin/env python3
"""Test CoreML compilation - wait for it to finish"""

import time
import sys

print("=" * 70)
print("CoreML Compilation Test")
print("=" * 70)
print("")
print("This will test if CoreML compilation works when we wait long enough.")
print("Expected: 30-90 seconds for first-time compilation")
print("")

sys.path.insert(0, '../pyfhog/src')

print("[1/2] Loading RetinaFace with CoreML (be patient!)...")
print("")
start = time.time()

from onnx_retinaface_detector import ONNXRetinaFaceDetector

detector = ONNXRetinaFaceDetector(
    'weights/retinaface_mobilenet025_coreml.onnx',
    use_coreml=True  # Enable CoreML
)

load_time = time.time() - start
print("")
print(f"✓ Loaded in {load_time:.1f} seconds")
print(f"  Backend: {detector.backend}")
print("")

print("[2/2] Testing face detection...")
import cv2
import numpy as np

# Create a dummy image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

start = time.time()
detections, _ = detector.detect_faces(test_image)
detect_time = (time.time() - start) * 1000

print(f"✓ Detection took {detect_time:.2f}ms")
print(f"  Faces found: {len(detections)}")
print("")

print("=" * 70)
if detector.backend == 'coreml':
    print("✅ CoreML IS WORKING!")
    print("")
    print("Performance:")
    print(f"  First load: {load_time:.1f}s (includes compilation)")
    print(f"  Detection:  {detect_time:.2f}ms")
    print("")
    print("Next time will be instant (CoreML model is now cached!)")
else:
    print("⚠️ CoreML fell back to CPU")
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Detection: {detect_time:.2f}ms")
print("=" * 70)
