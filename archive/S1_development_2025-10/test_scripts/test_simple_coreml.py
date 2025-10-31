#!/usr/bin/env python3
"""Simplest CoreML test - Initialize in Thread"""

import multiprocessing
import threading
import os
import sys

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Fork method set")
    except RuntimeError:
        pass

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

print("\n" + "=" * 70)
print("SIMPLE CoreML TEST - Initialize Detector in Thread")
print("=" * 70)
print("")

def worker():
    """Initialize and use detector INSIDE thread"""
    print("[Thread] Initializing pipeline with CoreML...")

    from full_python_au_pipeline import FullPythonAUPipeline
    import time

    t0 = time.time()
    pipeline = FullPythonAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='In-the-wild_aligned_PDM_68.txt',
        au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
        triangulation_file='tris_68_full.txt',
        use_calc_params=True,
        use_coreml=True,
        verbose=False
    )
    init_time = time.time() - t0

    print(f"[Thread] ✓ Init: {init_time:.2f}s")
    print(f"[Thread] Backend: {pipeline.face_detector.backend}")

    # Process 5 frames
    print("[Thread] Processing 5 frames...")
    video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

    t0 = time.time()
    results = pipeline.process_video(video_path, None, max_frames=5)
    process_time = time.time() - t0

    success = results['success'].sum()
    print(f"[Thread] ✓ Success: {success}/5")
    print(f"[Thread] Time: {process_time:.2f}s ({(process_time/5)*1000:.0f}ms/frame)")

print("[Main] Creating worker thread...")
thread = threading.Thread(target=worker, daemon=False, name="CoreMLWorker")
thread.start()
print("[Main] Waiting for thread...")
thread.join(timeout=120)

if thread.is_alive():
    print("\n[Main] ❌ Timeout!")
    sys.exit(1)
else:
    print("\n[Main] ✅ SUCCESS! CoreML works!")
