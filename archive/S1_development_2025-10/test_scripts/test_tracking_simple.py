#!/usr/bin/env python3
"""Simple test to verify face tracking is working"""
import multiprocessing, os, sys
if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.path.insert(0, '../pyfhog/src')
from full_python_au_pipeline import FullPythonAUPipeline

print("=" * 80)
print("FACE TRACKING VERIFICATION TEST")
print("=" * 80)
print()

# Create pipeline WITH face tracking
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    track_faces=True,  # ← Face tracking ON
    verbose=True
)

print("\nProcessing 10 frames with face tracking enabled...")
print("Watch for 'Using cached bbox' messages after frame 0!")
print()

results = pipeline.process_video(
    "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4",
    output_csv=None,
    max_frames=10
)

success = results['success'].sum()
print()
print("=" * 80)
print("FACE TRACKING STATISTICS")
print("=" * 80)
print(f"Successful frames: {success}/10")
print(f"Frames since last detection: {pipeline.frames_since_detection}")
print(f"Detection failures (re-detections): {pipeline.detection_failures}")
print()

if pipeline.frames_since_detection > 5:
    print("✅ Face tracking is WORKING! RetinaFace was skipped on most frames.")
else:
    print("⚠️ Face tracking may not be working - detection ran too frequently")

print()
print("=" * 80)
