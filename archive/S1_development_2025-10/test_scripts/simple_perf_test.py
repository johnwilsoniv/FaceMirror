#!/usr/bin/env python3
"""Simplest possible performance test"""

import sys
import time
import os

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

print("Simple Performance Test")
print("=" * 60)

sys.path.insert(0, '../pyfhog/src')

# Import
print("\n[1/3] Import...")
from full_python_au_pipeline import FullPythonAUPipeline
print("OK")

# Initialize
print("\n[2/3] Initialize...")
t0 = time.time()
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    verbose=False
)
init_time = time.time() - t0
print(f"OK ({init_time:.2f}s)")
print(f"Backend: {pipeline.face_detector.backend}")

# Process 5 frames only
print("\n[3/3] Process 5 frames...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

t0 = time.time()
results = pipeline.process_video(video_path, None, max_frames=5)
process_time = time.time() - t0

success = results['success'].sum()
print(f"OK - {success}/5 successful")
print(f"Time: {process_time:.2f}s")
print(f"Per frame: {(process_time/5)*1000:.0f}ms")
print(f"FPS: {5/process_time:.1f}")

print("\n" + "=" * 60)
print(f"Pipeline: {(process_time/5)*1000:.0f}ms/frame")
print(f"C++ Hybrid: 705ms/frame")
print(f"Speedup: {705/((process_time/5)*1000):.1f}x")
print("=" * 60)
