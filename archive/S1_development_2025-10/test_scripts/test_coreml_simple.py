#!/usr/bin/env python3
import multiprocessing
import os, sys, time

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    verbose=False
)

print("\\nProcessing 10 frames with CoreML...\\n", flush=True)
t0 = time.time()
results = pipeline.process_video(
    video_path="/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4",
    output_csv=None,
    max_frames=10
)
elapsed = time.time() - t0

success = results['success'].sum()
total = len(results)
fps = total / elapsed if elapsed > 0 else 0
ms_per_frame = (elapsed / total * 1000) if total > 0 else 0

print(f"\\nRESULTS: {success}/{total} frames ({fps:.1f} FPS, {ms_per_frame:.0f}ms/frame)", flush=True)

if success >= 8:
    print("\\n✅✅✅ 125 GLASSES EARNED!!! ✅✅✅\\n", flush=True)
else:
    print(f"\\n⚠️ Only {success}/10 frames succeeded\\n", flush=True)
