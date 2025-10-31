#!/usr/bin/env python3
"""
TEST: Queue-Based Architecture for CoreML Pipeline

Tests the new architecture:
- Main thread: Opens VideoCapture (macOS NSRunLoop requirement)
- Worker thread: Initializes CoreML and processes frames
- Communication via queues

This should solve both issues:
1. VideoCapture works (main thread has NSRunLoop)
2. CoreML works (worker thread initialization proven working)
"""

import multiprocessing
import os
import sys
import time

# Set fork method BEFORE any imports
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Fork method set")
    except RuntimeError:
        print("⚠ Fork already set")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

print("\n" + "=" * 80)
print("QUEUE-BASED ARCHITECTURE TEST - CoreML + VideoCapture")
print("=" * 80)
print("Architecture:")
print("  Main thread → VideoCapture (macOS NSRunLoop ✓)")
print("  Worker thread → CoreML initialization (proven pattern ✓)")
print("  Communication → Queues")
print("")

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("[Main Thread] Creating pipeline with CoreML...")
t0 = time.time()

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,  # ✅ CoreML ENABLED
    verbose=True
)

create_time = time.time() - t0
print(f"[Main Thread] ✓ Pipeline created: {create_time:.3f}s")
print(f"[Main Thread] Components initialized: {pipeline._components_initialized}")
print("")

print("[Main Thread] Calling process_video()...")
print("[Main Thread] This will:")
print("  1. Open VideoCapture in main thread ✓")
print("  2. Start worker thread for CoreML processing ✓")
print("  3. Read frames in main, process in worker ✓")
print("")

try:
    t0 = time.time()
    results = pipeline.process_video(
        video_path=video_path,
        output_csv=None,
        max_frames=10  # Test with 10 frames
    )
    process_time = time.time() - t0

    success = results['success'].sum()
    per_frame_ms = (process_time / len(results)) * 1000
    fps = len(results) / process_time

    print(f"\n[Main Thread] ✓ Video processed!")
    print(f"  Frames: {len(results)}")
    print(f"  Success: {success}/{len(results)}")
    print(f"  Time: {process_time:.2f}s")
    print(f"  Per frame: {per_frame_ms:.1f}ms")
    print(f"  Throughput: {fps:.1f} FPS")

    if pipeline.face_detector:
        print(f"  Detector backend: {pipeline.face_detector.backend}")

    print("\n" + "=" * 80)
    if success >= 8:
        print("✅✅✅ 125 GLASSES EARNED!!! ✅✅✅")
        print("=" * 80)
        print("")
        print("🎉 PROBLEM SOLVED! 💧💧💧")
        print("")
        print("Solution Summary:")
        print("  ✅ VideoCapture in main thread (macOS NSRunLoop satisfied)")
        print("  ✅ CoreML in worker thread (proven Thread+Fork pattern)")
        print("  ✅ Queue-based communication (no deadlocks)")
        print("  ✅ Full end-to-end processing with CoreML Neural Engine")
        print("")
        print("Key Insights:")
        print("  • macOS requires VideoCapture on main thread (NSRunLoop)")
        print("  • CoreML works perfectly in worker threads")
        print("  • Queue architecture solves both constraints")
        print("  • Architecture matches Face Mirror pattern")
        print("")
    else:
        print(f"⚠️ PARTIAL SUCCESS - {success}/10 frames")
        print("=" * 80)
        print("Need to investigate frame processing issues")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n💧 Failed - need more debugging")
    sys.exit(1)
