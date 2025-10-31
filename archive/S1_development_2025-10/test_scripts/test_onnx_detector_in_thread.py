#!/usr/bin/env python3
"""
Minimal test: Use ONNXRetinaFaceDetector (pipeline's detector) in Thread
Compare with OptimizedFaceDetector which works
"""

import multiprocessing
import threading
import os
import sys
import time

# Set fork method
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Fork method set")
    except RuntimeError:
        print("⚠ Fork already set")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

print("\n" + "=" * 80)
print("TEST: ONNXRetinaFaceDetector in Thread (like pipeline uses)")
print("=" * 80)

def worker_function():
    """Create ONNXRetinaFaceDetector in thread"""
    import numpy as np
    from onnx_retinaface_detector import ONNXRetinaFaceDetector

    print("[Thread] Worker started")
    print("[Thread] Creating ONNXRetinaFaceDetector...")

    start = time.time()
    detector = ONNXRetinaFaceDetector(
        onnx_model_path='weights/retinaface_mobilenet025_coreml.onnx',
        use_coreml=True
    )
    init_time = time.time() - start
    print(f"[Thread] ✓ Init: {init_time:.2f}s")
    print(f"[Thread] Backend: {detector.backend}")

    print("[Thread] Warmup with small images...")
    for i in range(3):
        print(f"[Thread]   Warmup {i+1}...", end=" ", flush=True)
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_faces(dummy, resize=1.0)
        print(f"OK ({len(detections)} faces)")

    print("[Thread] ✓ Completed without crash!")
    return True

print("[Main] Starting worker thread...")

result_container = {'success': False}

def wrapper():
    result_container['success'] = worker_function()

thread = threading.Thread(target=wrapper, daemon=False, name="CoreMLWorker")
thread.start()
print("[Main] Thread launched, waiting...")

thread.join(timeout=60.0)

if thread.is_alive():
    print("\n❌ Thread timeout")
    sys.exit(1)

print("\n" + "=" * 80)
if result_container['success']:
    print("✅ SUCCESS! ONNXRetinaFaceDetector works in Thread!")
else:
    print("⚠️ Failed")
print("=" * 80)
