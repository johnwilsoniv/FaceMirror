#!/usr/bin/env python3
"""
Performance Test: CoreML Queue Architecture vs CPU Mode

Tests the Full Python AU pipeline with:
1. CPU mode (baseline)
2. CoreML mode (new queue architecture)

Measures:
- Initialization time
- Per-frame processing time
- Total throughput (FPS)
- Success rate
"""

import multiprocessing
import os
import sys
import time
import pandas as pd

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
print("PERFORMANCE TEST: CoreML Queue Architecture vs CPU Mode")
print("=" * 80)

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
test_frames = 50  # Test with 50 frames for good statistics

results_summary = []

# Test parameters
configs = [
    {
        'name': 'CPU Mode',
        'use_coreml': False,
        'model': 'weights/retinaface_mobilenet025_coreml.onnx'
    },
    {
        'name': 'CoreML Queue Architecture',
        'use_coreml': True,
        'model': 'weights/retinaface_mobilenet025_coreml.onnx'
    }
]

for config in configs:
    print("\n" + "=" * 80)
    print(f"Testing: {config['name']}")
    print("=" * 80)

    # Create pipeline
    print("\n1. Initializing pipeline...")
    t0 = time.time()

    pipeline = FullPythonAUPipeline(
        retinaface_model=config['model'],
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='In-the-wild_aligned_PDM_68.txt',
        au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
        triangulation_file='tris_68_full.txt',
        use_calc_params=True,
        use_coreml=config['use_coreml'],
        verbose=False
    )

    init_time = time.time() - t0
    print(f"   ✓ Initialization: {init_time:.2f}s")

    # Process video
    print(f"\n2. Processing {test_frames} frames...")
    t0 = time.time()

    results = pipeline.process_video(
        video_path=video_path,
        output_csv=None,
        max_frames=test_frames
    )

    process_time = time.time() - t0

    # Analyze results
    success_count = results['success'].sum()
    total_frames = len(results)
    success_rate = (success_count / total_frames * 100) if total_frames > 0 else 0
    per_frame_ms = (process_time / total_frames * 1000) if total_frames > 0 else 0
    fps = total_frames / process_time if process_time > 0 else 0

    print(f"\n3. Results:")
    print(f"   Total frames:    {total_frames}")
    print(f"   Success:         {success_count}/{total_frames} ({success_rate:.1f}%)")
    print(f"   Total time:      {process_time:.2f}s")
    print(f"   Per frame:       {per_frame_ms:.1f}ms")
    print(f"   Throughput:      {fps:.2f} FPS")

    if hasattr(pipeline, 'face_detector') and pipeline.face_detector:
        backend = pipeline.face_detector.backend
        print(f"   Detector:        {backend}")

    # Store results
    results_summary.append({
        'Mode': config['name'],
        'Init Time (s)': init_time,
        'Total Time (s)': process_time,
        'Frames': total_frames,
        'Success': success_count,
        'Success Rate (%)': success_rate,
        'Per Frame (ms)': per_frame_ms,
        'FPS': fps,
        'Backend': backend if hasattr(pipeline, 'face_detector') and pipeline.face_detector else 'N/A'
    })

    # Show sample AU values
    if success_count > 0:
        print(f"\n4. Sample AU values (first successful frame):")
        first_success = results[results['success'] == True].iloc[0]
        aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r']
        for au in aus:
            if au in first_success:
                print(f"   {au}: {first_success[au]:.3f}")

# Summary comparison
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

df = pd.DataFrame(results_summary)
print("\n" + df.to_string(index=False))

if len(results_summary) == 2:
    cpu_fps = results_summary[0]['FPS']
    coreml_fps = results_summary[1]['FPS']
    cpu_ms = results_summary[0]['Per Frame (ms)']
    coreml_ms = results_summary[1]['Per Frame (ms)']

    if cpu_fps > 0:
        speedup = coreml_fps / cpu_fps
        time_reduction = ((cpu_ms - coreml_ms) / cpu_ms * 100) if cpu_ms > 0 else 0

        print("\n" + "=" * 80)
        print("SPEEDUP ANALYSIS")
        print("=" * 80)
        print(f"CoreML Speedup:     {speedup:.2f}x faster")
        print(f"Time Reduction:     {time_reduction:.1f}% faster")
        print(f"CPU:                {cpu_ms:.0f}ms/frame ({cpu_fps:.2f} FPS)")
        print(f"CoreML:             {coreml_ms:.0f}ms/frame ({coreml_fps:.2f} FPS)")
        print(f"Time Saved:         {cpu_ms - coreml_ms:.0f}ms per frame")

        if speedup >= 2.0:
            print("\n🎉 EXCELLENT! CoreML achieves 2x+ speedup!")
        elif speedup >= 1.5:
            print("\n✅ GOOD! CoreML achieves significant speedup!")
        else:
            print("\n⚠️ CoreML speedup lower than expected")

print("\n" + "=" * 80)
print("Performance test complete!")
print("=" * 80)
