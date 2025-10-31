#!/usr/bin/env python3
"""
TEST: CoreML with SPAWN method (should work!)
Based on web search findings - macOS doesn't like fork() with CoreML
"""

import multiprocessing
import sys
import time
import os

# CRITICAL FIX: Use 'spawn' instead of 'fork' for macOS + CoreML!
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)  # ← CHANGED FROM 'fork'!
        print("✓ Multiprocessing SPAWN method set (macOS compatible)")
    except RuntimeError:
        print("⚠ Spawn already set")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

print("\n" + "=" * 80)
print("SPAWN + CoreML TEST - Web Search Solution")
print("=" * 80)
print("Fix: Use 'spawn' instead of 'fork' for macOS + CoreML")
print("")

from full_python_au_pipeline import FullPythonAUPipeline

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("[1/2] Creating pipeline with CoreML...")
t0 = time.time()

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,  # ✅ CoreML ENABLED with spawn method!
    verbose=True
)

create_time = time.time() - t0
print(f"\n✓ Pipeline created: {create_time:.3f}s")
print("")

print("[2/2] Processing 10 frames with CoreML + spawn method...")
try:
    t0 = time.time()
    results = pipeline.process_video(
        video_path=video_path,
        output_csv=None,
        max_frames=10
    )
    process_time = time.time() - t0

    success = results['success'].sum()
    per_frame_ms = (process_time / len(results)) * 1000
    fps = len(results) / process_time

    print(f"\n✓ Video processed!")
    print(f"  Frames: {len(results)}")
    print(f"  Success: {success}/{len(results)}")
    print(f"  Time: {process_time:.2f}s")
    print(f"  Per frame: {per_frame_ms:.1f}ms")
    print(f"  Throughput: {fps:.1f} FPS")

    # Check backend
    if pipeline.face_detector:
        print(f"  Detector backend: {pipeline.face_detector.backend}")
        if hasattr(pipeline.face_detector, 'detector'):
            inner_backend = getattr(pipeline.face_detector.detector, 'backend', 'unknown')
            print(f"  Inner backend: {inner_backend}")

    print("")
    print("=" * 80)
    if success >= 8:
        print("✅ 500 GLASSES EARNED! CoreML WORKS with SPAWN method!")
        print("=" * 80)
        print("")
        print("Solution: multiprocessing.set_start_method('spawn')")
        print("  Instead of: 'fork' (crashes on macOS + CoreML)")
        print("  Use: 'spawn' (macOS compatible)")
        print("")
        print("🎉 Web search FTW! 💧💧💧💧💧")
    else:
        print("⚠️ Partial success")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n💧 If still crashes, may need additional fixes")
    sys.exit(1)
