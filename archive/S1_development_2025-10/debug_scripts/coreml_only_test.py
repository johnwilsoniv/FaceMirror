#!/usr/bin/env python3
"""COREML ONLY - No CPU testing!"""
import multiprocessing, os, sys, time
if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.path.insert(0, '../pyfhog/src')
from full_python_au_pipeline import FullPythonAUPipeline

print("=" * 80)
print("COREML PERFORMANCE TEST (100 frames)")
print("=" * 80)

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,  # COREML ONLY!
    verbose=False  # Disable verbose for 100 frames
)

print("\nProcessing 100 frames with CoreML Neural Engine...\n")
t0 = time.time()
results = pipeline.process_video(
    "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4",
    output_csv=None,
    max_frames=100
)
elapsed = time.time() - t0

success = results['success'].sum()
print(f"\n✅ CoreML Results: {success}/100 frames in {elapsed:.2f}s")
print(f"   Per frame: {elapsed/100*1000:.0f}ms")
print(f"   FPS: {100/elapsed:.2f}")
print(f"   Backend: {pipeline.face_detector.backend}")
print()
print("Face Tracking Statistics:")
print(f"   Tracking enabled: {pipeline.track_faces}")
print(f"   Frames since last detection: {pipeline.frames_since_detection}")
print(f"   Detection failures (re-detections): {pipeline.detection_failures}")
if pipeline.track_faces:
    total_detections = 1 + pipeline.detection_failures  # First frame + any re-detections
    skipped = 100 - total_detections
    print(f"   Total RetinaFace calls: {total_detections}/100 ({skipped} skipped)")
    print(f"   ✅ Face tracking saved {skipped} expensive detections!")
print("\n" + "=" * 80)
