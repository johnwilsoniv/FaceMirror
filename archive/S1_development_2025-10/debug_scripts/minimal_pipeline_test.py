#!/usr/bin/env python3
"""Minimal test to verify pipeline works with CoreML disabled"""

import time
import sys

print("Starting minimal pipeline test...")
print("")

# Test 1: Import
print("[1/4] Testing imports...")
try:
    sys.path.insert(0, '../pyfhog/src')
    from full_python_au_pipeline import FullPythonAUPipeline
    print("✓ Pipeline imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize
print("\n[2/4] Initializing pipeline (CoreML disabled)...")
start = time.time()
try:
    pipeline = FullPythonAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='In-the-wild_aligned_PDM_68.txt',
        au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
        triangulation_file='tris_68_full.txt',
        use_calc_params=True,
        verbose=False
    )
    init_time = time.time() - start
    print(f"✓ Pipeline initialized in {init_time:.2f}s")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Process one frame
print("\n[3/4] Processing 1 frame...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

start = time.time()
try:
    results = pipeline.process_video(
        video_path=video_path,
        output_csv=None,
        max_frames=1
    )
    process_time = time.time() - start
    print(f"✓ Processed 1 frame in {process_time:.2f}s")
    print(f"  Success: {results['success'].sum()}")
except Exception as e:
    print(f"✗ Processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check results
print("\n[4/4] Checking results...")
if results['success'].iloc[0]:
    au_cols = [c for c in results.columns if c.startswith('AU')]
    print(f"✓ Got {len(au_cols)} AU predictions")
    print(f"  Sample: {au_cols[0]}={results[au_cols[0]].iloc[0]:.3f}")
else:
    print("✗ Frame processing failed")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print(f"Init time: {init_time:.2f}s")
print(f"Process 1 frame: {process_time:.2f}s")
print(f"Per-frame estimate: {process_time * 1000:.0f}ms")
print("")
