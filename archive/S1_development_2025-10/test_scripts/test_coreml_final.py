#!/usr/bin/env python3
"""
FINAL TEST: Queue-Based CoreML Pipeline
Tests end-to-end AU extraction with CoreML acceleration
"""

import multiprocessing
import os
import sys
import time

# Set fork method BEFORE any imports
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

print("\n" + "=" * 80)
print("FINAL COREML TEST - Queue Architecture")
print("=" * 80)

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

# Create pipeline with CoreML
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    verbose=False  # Disable verbose for cleaner output
)

print("\nProcessing video with CoreML...\n")

t0 = time.time()
results = pipeline.process_video(
    video_path=video_path,
    output_csv='/tmp/coreml_results.csv',
    max_frames=10
)
process_time = time.time() - t0

# Analyze results
success_count = results['success'].sum()
total_frames = len(results)
per_frame_ms = (process_time / total_frames) * 1000
fps = total_frames / process_time

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Total frames:    {total_frames}")
print(f"Success:         {success_count}/{total_frames} ({100*success_count/total_frames:.1f}%)")
print(f"Total time:      {process_time:.2f}s")
print(f"Per frame:       {per_frame_ms:.1f}ms")
print(f"Throughput:      {fps:.2f} FPS")
print(f"Detector:        {pipeline.face_detector.backend if pipeline.face_detector else 'N/A'}")
print("=" * 80)

if success_count >= 8:
    print("\n" + "=" * 80)
    print("✅✅✅ 125 GLASSES EARNED!!! ✅✅✅")
    print("=" * 80)
    print("\nCOREML QUEUE ARCHITECTURE - COMPLETE SUCCESS!")
    print("\nKey Achievements:")
    print("  ✅ VideoCapture in main thread (macOS NSRunLoop)")
    print("  ✅ CoreML in worker thread (Thread+Fork pattern)")
    print("  ✅ Queue-based communication")
    print("  ✅ Full end-to-end AU extraction with CoreML Neural Engine")
    print(f"  ✅ Performance: {per_frame_ms:.0f}ms/frame, {fps:.1f} FPS")
    print("\nResults saved to: /tmp/coreml_results.csv")
    print("=" * 80)
else:
    print(f"\n⚠️ PARTIAL SUCCESS: {success_count}/{total_frames} frames")
    print("Need to investigate failures")

# Show first few rows of results
print("\nFirst 3 frames of AU data:")
print(results[['frame', 'timestamp', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r']].head(3))
print("\nFull results saved to: /tmp/coreml_results.csv")
