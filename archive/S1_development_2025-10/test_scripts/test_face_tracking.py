#!/usr/bin/env python3
"""Test face tracking optimization"""
import multiprocessing, os, sys, time
if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.path.insert(0, '../pyfhog/src')
from full_python_au_pipeline import FullPythonAUPipeline

print("=" * 80)
print("FACE TRACKING TEST")
print("=" * 80)
print()
print("Comparing:")
print("  1. With face tracking (default)")
print("  2. Without face tracking")
print()
print("Expected: ~3x speedup with tracking (skip RetinaFace on most frames)")
print("=" * 80)
print()

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

# Test 1: WITH face tracking (default)
print("TEST 1: WITH FACE TRACKING")
print("-" * 80)
pipeline_with_tracking = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    track_faces=True,  # ← Face tracking ON
    verbose=False
)

t0 = time.time()
results_with = pipeline_with_tracking.process_video(
    video_path,
    output_csv=None,
    max_frames=20  # Test 20 frames
)
time_with = time.time() - t0

success = results_with['success'].sum()
print(f"\n✅ Results: {success}/20 frames in {time_with:.2f}s")
print(f"   Per frame: {time_with/20*1000:.0f}ms")
print(f"   FPS: {20/time_with:.2f}")
print(f"   Frames since last detection: {pipeline_with_tracking.frames_since_detection}")
print(f"   Detection failures: {pipeline_with_tracking.detection_failures}")
print()

# Test 2: WITHOUT face tracking
print("TEST 2: WITHOUT FACE TRACKING")
print("-" * 80)
pipeline_without_tracking = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    track_faces=False,  # ← Face tracking OFF
    verbose=False
)

t0 = time.time()
results_without = pipeline_without_tracking.process_video(
    video_path,
    output_csv=None,
    max_frames=20  # Test 20 frames
)
time_without = time.time() - t0

success = results_without['success'].sum()
print(f"\n✅ Results: {success}/20 frames in {time_without:.2f}s")
print(f"   Per frame: {time_without/20*1000:.0f}ms")
print(f"   FPS: {20/time_without:.2f}")
print()

# Compare
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"With tracking:    {time_with/20*1000:.0f}ms/frame ({20/time_with:.2f} FPS)")
print(f"Without tracking: {time_without/20*1000:.0f}ms/frame ({20/time_without:.2f} FPS)")
print()
speedup = time_without / time_with
print(f"Speedup: {speedup:.2f}x faster! 🚀")
print()

# AU comparison (make sure tracking doesn't hurt accuracy)
au_cols = [col for col in results_with.columns if col.startswith('AU') and col.endswith('_r')]
if au_cols:
    print("AU Accuracy Check (first 5 AUs):")
    for au_col in sorted(au_cols)[:5]:
        with_mean = results_with[au_col].mean()
        without_mean = results_without[au_col].mean()
        diff_pct = abs(with_mean - without_mean) / (without_mean + 1e-6) * 100
        print(f"  {au_col}: with={with_mean:.3f}, without={without_mean:.3f}, diff={diff_pct:.1f}%")

print()
print("=" * 80)
