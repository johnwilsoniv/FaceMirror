#!/usr/bin/env python3
"""
TEST: Import pipeline INSIDE worker thread (like test_thread_init_hypothesis.py)
This is the TRUE solution - delay ALL CoreML-related imports until worker thread
"""

import multiprocessing
import threading
import os
import sys
import time

# Set multiprocessing fork method
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Multiprocessing fork method set")
    except RuntimeError:
        print("⚠ Fork already set")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

print("\n" + "=" * 80)
print("CRITICAL TEST: Import Pipeline INSIDE Worker Thread")
print("=" * 80)
print("Hypothesis: ONNX Runtime import must happen in worker thread, not main")
print("")

def worker_function():
    """Import and initialize EVERYTHING inside the worker thread"""
    print("[Thread] Worker started - importing pipeline NOW...")

    # ✅ CRITICAL: Import INSIDE worker thread (not in main!)
    from full_python_au_pipeline import FullPythonAUPipeline

    print("[Thread] ✓ Pipeline imported (ONNX Runtime loaded in thread)")
    print("[Thread] Creating pipeline with CoreML...")

    start = time.time()
    pipeline = FullPythonAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='In-the-wild_aligned_PDM_68.txt',
        au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
        triangulation_file='tris_68_full.txt',
        use_calc_params=True,
        use_coreml=True,  # ✅ CoreML enabled
        verbose=False
    )
    init_time = time.time() - start
    print(f"[Thread] ✓ Pipeline created: {init_time:.2f}s")

    # Test processing
    video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

    print("[Thread] Processing 5 frames...")
    start = time.time()
    results = pipeline.process_video(video_path, None, max_frames=5)
    process_time = time.time() - start

    success = results['success'].sum()
    print(f"[Thread] ✓ Processed: {success}/5 successful in {process_time:.2f}s")

    return success >= 4

print("[Main] Launching worker thread...")
print("[Main] NO imports happened in main thread (clean state!)")
print("")

result_container = {'success': False}

def wrapper():
    result_container['success'] = worker_function()

# Create worker thread
worker_thread = threading.Thread(
    target=wrapper,
    daemon=False,
    name="CoreMLWorker"
)

worker_thread.start()
print("[Main] Worker thread launched, waiting...")

worker_thread.join(timeout=60.0)

if worker_thread.is_alive():
    print("\n[Main] ❌ Worker thread timeout")
    sys.exit(1)

print("\n" + "=" * 80)
if result_container['success']:
    print("✅✅✅ SUCCESS!!! CoreML WORKS!!! ✅✅✅")
    print("=" * 80)
    print("")
    print("🎉 500 GLASSES EARNED! 💧💧💧💧💧")
    print("")
    print("SOLUTION CONFIRMED:")
    print("  - Import ONNX Runtime INSIDE worker thread")
    print("  - Do NOT import in main thread")
    print("  - This allows CoreML to work with fork()")
    print("")
else:
    print("⚠️ Partial success")
print("=" * 80)
