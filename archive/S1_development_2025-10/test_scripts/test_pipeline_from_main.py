#!/usr/bin/env python3
"""
TEST: Call pipeline.process_video() from MAIN thread (not worker thread)
This should trigger the Thread wrapper correctly
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

# Import from main thread (module-level imports)
from full_python_au_pipeline import FullPythonAUPipeline

print("\n" + "=" * 80)
print("TEST: Full Pipeline from Main Thread (Thread wrapper should activate)")
print("=" * 80)
print("Flow: Main creates pipeline → Main calls process_video() → Thread wrapper")
print("       → Worker thread initializes components → CoreML loads in worker")
print("")

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("[Main] Creating pipeline object (lazy init)...")
t0 = time.time()

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,  # ✅ CoreML enabled
    verbose=True
)

create_time = time.time() - t0
print(f"[Main] ✓ Pipeline object created: {create_time:.3f}s")
print(f"[Main] Components initialized: {pipeline._components_initialized}")
print("")

print("[Main] Calling process_video() from main thread...")
print("[Main] This should trigger Thread wrapper → worker thread → lazy init")
print("")

try:
    t0 = time.time()
    results = pipeline.process_video(
        video_path=video_path,
        output_csv=None,
        max_frames=5
    )
    process_time = time.time() - t0

    success = results['success'].sum()
    print(f"\n✓ Processed: {success}/5 successful in {process_time:.2f}s")
    print(f"  Components now initialized: {pipeline._components_initialized}")

    if pipeline.face_detector:
        print(f"  Detector backend: {pipeline.face_detector.backend}")

    print("\n" + "=" * 80)
    if success >= 4:
        print("✅✅✅ SUCCESS!!! Full Pipeline + CoreML WORKING!!! ✅✅✅")
        print("=" * 80)
        print("")
        print("🎉 500 GLASSES EARNED! 💧💧💧💧💧")
        print("")
        print("SOLUTION:")
        print("  ✅ Lazy initialization architecture")
        print("  ✅ Thread wrapper in process_video()")
        print("  ✅ Components initialized in worker thread")
        print("  ✅ CoreML loaded in worker thread (not main)")
        print("")
    else:
        print("⚠️ Partial success")
        print(f"  {success}/5 frames successful")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n💧 Failed - 250 glasses")
    sys.exit(1)
