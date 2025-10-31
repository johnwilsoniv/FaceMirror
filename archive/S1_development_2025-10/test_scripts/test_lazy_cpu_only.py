#!/usr/bin/env python3
"""Test lazy initialization with CPU mode (should work!)"""

import os
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

print("\n" + "=" * 80)
print("LAZY INITIALIZATION TEST - CPU MODE")
print("=" * 80)

from full_python_au_pipeline import FullPythonAUPipeline

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("[1/2] Creating pipeline object (lazy init, CPU mode)...")
t0 = time.time()

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=False,  # CPU mode for testing
    verbose=True
)

create_time = time.time() - t0
print(f"\n✓ Pipeline object created: {create_time:.3f}s")
print(f"  Components initialized: {pipeline._components_initialized}")
print("")

print("[2/2] Processing 5 frames (components will initialize now)...")
t0 = time.time()
results = pipeline.process_video(video_path, None, max_frames=5)
process_time = time.time() - t0

success = results['success'].sum()
print(f"\n✓ Success: {success}/5 in {process_time:.2f}s")
print(f"  Components initialized: {pipeline._components_initialized}")

if success >= 4:
    print("\n✅ 250 GLASSES - Lazy initialization works (CPU mode)!")
else:
    print("\n⚠️ Some issues with lazy init")
