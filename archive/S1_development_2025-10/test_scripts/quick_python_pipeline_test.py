#!/usr/bin/env python3
"""
Quick Python Pipeline Test - Process 5 frames and measure performance
"""

import time
import cv2
from pathlib import Path

print("=" * 80)
print("QUICK PYTHON PIPELINE PERFORMANCE TEST")
print("=" * 80)
print("")

# Test video
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

if not Path(video_path).exists():
    print(f"❌ Video not found: {video_path}")
    exit(1)

print(f"Video: {Path(video_path).name}")
print("Processing: 5 frames")
print("")

# Import pipeline
print("[1/3] Importing pipeline...")
start = time.time()
from full_python_au_pipeline import FullPythonAUPipeline
import_time = time.time() - start
print(f"✓ Import: {import_time:.2f}s")
print("")

# Initialize
print("[2/3] Initializing components...")
start = time.time()

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    verbose=False  # Quiet for clean output (CoreML now disabled in pipeline)
)

init_time = time.time() - start
print(f"✓ Init: {init_time:.2f}s")
print("")

# Process frames
print("[3/3] Processing 5 frames...")
start = time.time()

results = pipeline.process_video(
    video_path=video_path,
    output_csv=None,
    max_frames=5
)

process_time = time.time() - start
print(f"✓ Processing: {process_time:.2f}s")
print("")

# Results
success = results['success'].sum()
fail = len(results) - success

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Frames: {len(results)} (Success: {success}, Failed: {fail})")
print(f"Processing time: {process_time:.3f}s")
print(f"Per frame: {(process_time/len(results)*1000):.2f}ms")
print(f"Throughput: {len(results)/process_time:.2f} FPS")
print("")

# Compare to hybrid
hybrid_per_frame = (35.24 / 50) * 1000  # 704.8 ms
python_per_frame = (process_time / len(results)) * 1000

print("COMPARISON:")
print(f"  C++ Hybrid: {hybrid_per_frame:.2f}ms/frame")
print(f"  Full Python: {python_per_frame:.2f}ms/frame")
print(f"  Speedup: {hybrid_per_frame/python_per_frame:.1f}x")
print("")

# Sample AUs
if success > 0:
    print("SAMPLE AUs (first successful frame):")
    success_frame = results[results['success'] == True].iloc[0]
    au_cols = [c for c in results.columns if c.startswith('AU') and c.endswith('_r')]
    for au in sorted(au_cols)[:5]:  # First 5 AUs
        print(f"  {au}: {success_frame[au]:.3f}")
    print(f"  ... ({len(au_cols)} total AUs)")

print("")
print("✅ TEST COMPLETE")
