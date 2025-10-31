#!/usr/bin/env python3
"""Quick Performance Test - CoreML vs CPU (10 frames each)"""

import multiprocessing
import os, sys, time

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("=" * 80)
print("QUICK PERFORMANCE TEST: CoreML vs CPU (10 frames)")
print("=" * 80)

# Test 1: CPU Mode
print("\n[1/2] Testing CPU Mode...")
try:
    pipeline_cpu = FullPythonAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='In-the-wild_aligned_PDM_68.txt',
        au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
        triangulation_file='tris_68_full.txt',
        use_calc_params=True,
        use_coreml=False,
        verbose=False
    )

    t0 = time.time()
    results_cpu = pipeline_cpu.process_video(video_path, output_csv=None, max_frames=10)
    cpu_time = time.time() - t0

    cpu_success = results_cpu['success'].sum()
    cpu_fps = len(results_cpu) / cpu_time if cpu_time > 0 else 0
    cpu_ms = (cpu_time / len(results_cpu) * 1000) if len(results_cpu) > 0 else 0

    print(f"✓ CPU Mode: {cpu_success}/10 frames, {cpu_ms:.0f}ms/frame, {cpu_fps:.2f} FPS")
except Exception as e:
    print(f"✗ CPU Mode failed: {e}")
    cpu_time = None

# Test 2: CoreML Mode
print("\n[2/2] Testing CoreML Mode...")
try:
    pipeline_coreml = FullPythonAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='In-the-wild_aligned_PDM_68.txt',
        au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
        triangulation_file='tris_68_full.txt',
        use_calc_params=True,
        use_coreml=True,
        verbose=False
    )

    t0 = time.time()
    results_coreml = pipeline_coreml.process_video(video_path, output_csv=None, max_frames=10)
    coreml_time = time.time() - t0

    coreml_success = results_coreml['success'].sum()
    coreml_fps = len(results_coreml) / coreml_time if coreml_time > 0 else 0
    coreml_ms = (coreml_time / len(results_coreml) * 1000) if len(results_coreml) > 0 else 0

    print(f"✓ CoreML Mode: {coreml_success}/10 frames, {coreml_ms:.0f}ms/frame, {coreml_fps:.2f} FPS")
except Exception as e:
    print(f"✗ CoreML Mode failed: {e}")
    import traceback
    traceback.print_exc()
    coreml_time = None

# Comparison
if cpu_time and coreml_time:
    speedup = cpu_time / coreml_time
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"CPU Mode:     {cpu_ms:.0f}ms/frame ({cpu_fps:.2f} FPS)")
    print(f"CoreML Mode:  {coreml_ms:.0f}ms/frame ({coreml_fps:.2f} FPS)")
    print(f"Speedup:      {speedup:.2f}x")
    if speedup >= 2.0:
        print("\n🎉 EXCELLENT! CoreML is 2x+ faster!")
    elif speedup >= 1.5:
        print("\n✅ GOOD! CoreML provides significant speedup!")
    else:
        print(f"\n⚠️ CoreML speedup: {speedup:.2f}x (expected 2-3x)")
print("=" * 80)
