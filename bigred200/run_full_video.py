#!/usr/bin/env python3
"""
Run full video through Python pipeline on BR200.

Usage (on BR200):
    cd ~/pyfaceau
    module load python/3.11.13
    export PYTHONPATH=$PWD:$PWD/pyclnf:$PWD/pyfaceau:$PWD/pymtcnn:$PYTHONPATH

    # Interactive (for debugging):
    srun -A r01984 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=01:00:00 \
        python bigred200/run_full_video.py --video test_data/IMG_0942.MOV --output /tmp/br200_full_video_aus.csv

    # Or submit as batch job:
    sbatch bigred200/submit_full_video.sh
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))

# Limit threads for HPC
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'


def main():
    parser = argparse.ArgumentParser(description="Run full video through AU pipeline")
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--output', default='/tmp/br200_full_video_aus.csv', help='Output CSV')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames (None=all)')
    args = parser.parse_args()

    from pyfaceau.pipeline import FullPythonAUPipeline

    # Model paths
    pdm_file = project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir = project_root / "pyfaceau/weights/AU_predictors"
    triangulation_file = project_root / "pyfaceau/weights/tris_68_full.txt"

    # Check alternate paths
    if not pdm_file.exists():
        pdm_file = project_root / "pyclnf/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    if not au_models_dir.exists():
        au_models_dir = project_root / "pyclnf/pyfaceau/weights/AU_predictors"
    if not triangulation_file.exists():
        triangulation_file = project_root / "pyclnf/pyfaceau/weights/tris_68_full.txt"

    print("=" * 70)
    print("BR200 FULL VIDEO PROCESSING")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Max frames: {args.max_frames or 'ALL'}")
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = FullPythonAUPipeline(
        pdm_file=str(pdm_file),
        au_models_dir=str(au_models_dir),
        triangulation_file=str(triangulation_file),
        patch_expert_file="",
        mtcnn_backend='onnx',  # ONNX on Linux
        use_calc_params=True,
        track_faces=False,  # Disabled for accuracy
        use_batched_predictor=True,
        max_clnf_iterations=10,
        clnf_convergence_threshold=0.005,  # Gold standard for sub-pixel accuracy
        verbose=True
    )

    print(f"Face detector backend: {pipeline.face_detector.backend if pipeline.face_detector else 'not initialized'}")
    print()

    # Process video
    start = time.time()
    df = pipeline.process_video(
        video_path=args.video,
        output_csv=args.output,
        max_frames=args.max_frames
    )
    elapsed = time.time() - start

    success_count = df['success'].sum()
    fps = success_count / elapsed if elapsed > 0 else 0

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Frames: {success_count}/{len(df)} successful")
    print(f"Time: {elapsed:.1f}s ({fps:.2f} FPS)")
    print(f"Output: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
