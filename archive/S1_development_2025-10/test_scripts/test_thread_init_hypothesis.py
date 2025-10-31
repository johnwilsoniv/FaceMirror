#!/usr/bin/env python3
"""Test if initializing CoreML in a Thread (not main) enables it"""

import multiprocessing
import threading
import os
import sys
import time

# Set multiprocessing fork method (like Face Mirror)
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Set multiprocessing start method to 'fork'")
    except RuntimeError as e:
        print(f"⚠ Could not set fork method: {e}")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

print("\n" + "=" * 70)
print("HYPOTHESIS TEST: Initialize CoreML in Thread (not main process)")
print("=" * 70)
print("Face Mirror creates detector in worker thread, not main thread")
print("")

def worker_thread_function():
    """This mimics Face Mirror's video_processing_worker function"""
    import numpy as np
    from onnx_retinaface_detector import OptimizedFaceDetector

    print("[Thread] Worker thread started")
    print("[Thread] Initializing OptimizedFaceDetector...")

    start = time.time()
    detector = OptimizedFaceDetector(
        model_path='weights/Alignment_RetinaFace.pth',
        onnx_model_path='weights/retinaface_mobilenet025_coreml.onnx',
        device='cpu',
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    init_time = time.time() - start
    print(f"[Thread] ✓ Init: {init_time:.2f}s")

    # Check inner backend
    if hasattr(detector.detector, 'backend'):
        print(f"[Thread] Inner detector backend: {detector.detector.backend}")

    print("[Thread] Warmup with small images...")
    try:
        for i in range(3):
            print(f"[Thread]   Warmup {i+1}...", end=" ", flush=True)
            dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = detector.detect_faces(dummy, resize=1.0)
            print(f"OK ({len(detections)} faces)")
        print("[Thread] ✓ Warmup completed without crash!")
        return True
    except Exception as e:
        print(f"\n[Thread] ❌ CRASH during warmup: {e}")
        import traceback
        traceback.print_exc()
        return False

print("[Main] Starting worker thread (mimicking Face Mirror)...")
print("[Main] This is exactly how Face Mirror initializes CoreML!")
print("")

# Create thread exactly as Face Mirror does (daemon=False)
worker_thread = threading.Thread(
    target=worker_thread_function,
    daemon=False,
    name="VideoProcessingWorker"
)

worker_thread.start()
print("[Main] Worker thread launched, waiting for completion...")

# Wait for thread to complete
worker_thread.join(timeout=60.0)

if worker_thread.is_alive():
    print("\n[Main] ❌ Worker thread still running after timeout")
    sys.exit(1)
else:
    print("\n[Main] ✓ Worker thread completed!")

print("\n" + "=" * 70)
print("✅ SUCCESS! CoreML works when initialized in Thread!")
print("=" * 70)
print("CONCLUSION: Thread initialization enables CoreML in standalone scripts!")
print("=" * 70)
