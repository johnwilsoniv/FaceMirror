#!/usr/bin/env python3
"""Test process_video() with detailed logging"""

import multiprocessing
import sys
import os

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Fork set")
    except RuntimeError:
        print("⚠ Fork already set")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

print("1. Imported ✓")

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    verbose=False  # Turn off pipeline verbose to see our messages
)

print("2. Pipeline created ✓")
print(f"   Components initialized: {pipeline._components_initialized}")

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("3. About to call process_video()...")
print("   This should trigger Thread wrapper + lazy initialization")

try:
    results = pipeline.process_video(video_path, None, max_frames=3)
    print(f"\n4. ✅ SUCCESS! Processed {len(results)} frames")
    print(f"   Success: {results['success'].sum()}/{len(results)}")
    if pipeline.face_detector:
        print(f"   Backend: {pipeline.face_detector.backend}")
    print("\n🎉 500 GLASSES EARNED!")
except Exception as e:
    print(f"\n4. ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
