#!/usr/bin/env python3
"""
HPC Batch AU Pipeline for BigRed200

Optimized for SLURM array jobs - processes one video per task sequentially.
This is the most efficient approach because:
1. CLNF landmark detection (410MB model) is the bottleneck
2. Running median requires sequential processing
3. Face tracking provides ~3-5x speedup within a video

Usage with SLURM Array Jobs:
    sbatch --array=0-99 submit_batch.sh videos.txt results/

This will process 100 videos in parallel, each on its own CPU.

Single video usage:
    python hpc_batch_pipeline.py --video input.mp4 --output results/output.csv
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

# Limit threads for HPC (single-threaded per job is most efficient)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'


def process_single_video(
    video_path: str,
    output_csv: str,
    track_faces: bool = False,
    max_frames: int = None,
    verbose: bool = True
):
    """
    Process a single video with the AU pipeline

    Args:
        video_path: Path to input video
        output_csv: Path for output CSV
        track_faces: Enable face tracking (disabled by default for accuracy)
        max_frames: Optional frame limit
        verbose: Print progress

    Returns:
        dict with processing stats
    """
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

    if verbose:
        print(f"Processing: {video_path}")
        print(f"Output: {output_csv}")
        print(f"Face tracking: {'enabled' if track_faces else 'disabled'}")

    # Initialize pipeline
    pipeline = FullPythonAUPipeline(
        pdm_file=str(pdm_file),
        au_models_dir=str(au_models_dir),
        triangulation_file=str(triangulation_file),
        patch_expert_file="",
        mtcnn_backend='onnx',  # ONNX on Linux HPC
        use_calc_params=True,
        track_faces=track_faces,  # Disabled by default for accuracy
        use_batched_predictor=True,
        max_clnf_iterations=10,
        clnf_convergence_threshold=0.005,  # Gold standard for sub-pixel accuracy
        verbose=verbose
    )

    # Process video
    start = time.time()
    df = pipeline.process_video(
        video_path=video_path,
        output_csv=output_csv,
        max_frames=max_frames
    )
    elapsed = time.time() - start

    success_count = df['success'].sum() if 'success' in df.columns else len(df)
    fps = success_count / elapsed if elapsed > 0 else 0

    stats = {
        'video': str(video_path),
        'output': str(output_csv),
        'frames': len(df),
        'success': int(success_count),
        'time': elapsed,
        'fps': fps
    }

    if verbose:
        print(f"Complete: {success_count}/{len(df)} frames in {elapsed:.1f}s ({fps:.2f} FPS)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="HPC Batch AU Pipeline - optimized for SLURM array jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options
    parser.add_argument('--video', help='Single video file')
    parser.add_argument('--video-list', help='File with video paths (one per line)')
    parser.add_argument('--array-index', type=int,
                        help='SLURM array task index (uses SLURM_ARRAY_TASK_ID if not set)')

    # Output options
    parser.add_argument('--output', help='Output CSV (single video mode)')
    parser.add_argument('--output-dir', help='Output directory (batch mode)')

    # Processing options
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable face tracking (slower but more accurate)')
    parser.add_argument('--max-frames', type=int, help='Max frames per video')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    verbose = not args.quiet
    track_faces = not args.no_tracking

    # Determine which video to process
    if args.video:
        # Single video mode
        video_path = args.video
        output_csv = args.output
        if not output_csv:
            video_p = Path(video_path)
            output_csv = str(video_p.parent / f"{video_p.stem}_aus.csv")

    elif args.video_list:
        # Batch mode - get video from list using array index
        array_idx = args.array_index
        if array_idx is None:
            array_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

        with open(args.video_list) as f:
            videos = [line.strip() for line in f if line.strip()]

        if array_idx >= len(videos):
            print(f"Array index {array_idx} exceeds video count {len(videos)}")
            return 1

        video_path = videos[array_idx]

        # Output to directory
        output_dir = Path(args.output_dir or 'results')
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem
        output_csv = str(output_dir / f"{video_name}_aus.csv")

        if verbose:
            print(f"SLURM array task {array_idx}/{len(videos)}")

    else:
        parser.error("Either --video or --video-list is required")
        return 1

    # Process the video
    try:
        stats = process_single_video(
            video_path=video_path,
            output_csv=output_csv,
            track_faces=track_faces,
            max_frames=args.max_frames,
            verbose=verbose
        )

        # Print summary for SLURM logs
        print(f"\nSUMMARY: {stats['success']}/{stats['frames']} frames, "
              f"{stats['time']:.1f}s, {stats['fps']:.2f} FPS")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
