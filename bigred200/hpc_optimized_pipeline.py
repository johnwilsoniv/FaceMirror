#!/usr/bin/env python3
"""
HPC Optimized AU Pipeline for BigRed200

Optimizes performance for AMD EPYC 7742 (128 CPUs, 251GB RAM) while
maintaining accuracy validated against local pipeline.

Parallelization Strategy:
=========================
Level 1: Multi-video parallelization (process N videos simultaneously)
Level 2: Per-video parallel frame processing (M workers per video)

The running median update must be sequential per-video, but feature
extraction (detection, landmarks, alignment, HOG) can be parallelized.

Key Optimizations:
- ONNX threading optimized for AMD EPYC
- Configurable parallelism levels based on node resources
- Efficient frame batching to minimize process overhead
- Memory-efficient frame queuing

Usage:
    # Single video with parallel frames
    python hpc_optimized_pipeline.py --video input.mp4 --output results.csv --workers 8

    # Multiple videos in parallel
    python hpc_optimized_pipeline.py --video-list videos.txt --output-dir results/ \
        --video-parallel 4 --workers-per-video 8
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))


def get_optimal_thread_config(total_cpus: int, num_videos: int = 1) -> Dict:
    """
    Calculate optimal thread configuration for BR200 AMD EPYC

    AMD EPYC 7742: 64 cores, 128 threads (SMT)
    - ONNX Runtime works best with physical cores for compute-bound work
    - Memory bandwidth is shared, so diminishing returns past ~32 workers

    Args:
        total_cpus: Total CPUs available (from SLURM)
        num_videos: Number of videos to process in parallel

    Returns:
        Dict with configuration
    """
    # For AMD EPYC, use physical cores (half of hyperthreads)
    physical_cores = max(1, total_cpus // 2)

    if num_videos == 1:
        # Single video: all cores for frame parallelization
        workers_per_video = min(physical_cores, 32)  # Cap at 32 for efficiency
        ort_threads = 1  # Single thread per ONNX inference (parallelism at worker level)
    else:
        # Multiple videos: distribute cores
        workers_per_video = max(2, physical_cores // num_videos)
        workers_per_video = min(workers_per_video, 16)  # Cap per-video workers
        ort_threads = 1

    return {
        'workers_per_video': workers_per_video,
        'ort_threads': ort_threads,
        'video_parallel': num_videos,
        'total_workers': workers_per_video * num_videos
    }


def configure_threading(ort_threads: int = 1):
    """Configure threading for optimal HPC performance"""
    # Limit per-thread parallelism (we parallelize at process level)
    os.environ['OMP_NUM_THREADS'] = str(ort_threads)
    os.environ['MKL_NUM_THREADS'] = str(ort_threads)
    os.environ['NUMBA_NUM_THREADS'] = str(ort_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(ort_threads)
    os.environ['ORT_NUM_THREADS'] = str(ort_threads)

    # ONNX Runtime specific settings
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'  # Reduce spin-wait overhead


# Configure threading before imports
configure_threading(1)

import cv2


# Global worker state
_worker_pipeline = None
_worker_id = None
_worker_counter = None  # Shared counter for staggered init


def init_worker(init_params, stagger_delay=2.0):
    """Initialize worker process with pipeline instance (staggered)"""
    global _worker_pipeline, _worker_id, _worker_counter

    # Get worker ID from shared counter for staggered initialization
    if _worker_counter is not None:
        with _worker_counter.get_lock():
            _worker_id = _worker_counter.value
            _worker_counter.value += 1
    else:
        _worker_id = 0

    # Stagger initialization to avoid I/O contention on 410MB CEN models
    # Each worker waits (worker_id * stagger_delay) seconds before loading
    if stagger_delay > 0 and _worker_id > 0:
        time.sleep(_worker_id * stagger_delay)

    # Import here to avoid issues with forking
    from pyfaceau.pipeline import FullPythonAUPipeline

    # Configure per-worker threading
    configure_threading(init_params.get('ort_threads', 1))

    _worker_pipeline = FullPythonAUPipeline(
        pdm_file=init_params['pdm_file'],
        au_models_dir=init_params['au_models_dir'],
        triangulation_file=init_params['triangulation_file'],
        patch_expert_file=init_params.get('patch_expert_file', ''),
        mtcnn_backend='onnx',  # Always ONNX on HPC
        use_calc_params=True,
        track_faces=False,  # Each frame independent in parallel mode
        use_batched_predictor=True,
        max_clnf_iterations=init_params.get('max_clnf_iterations', 10),
        clnf_convergence_threshold=init_params.get('clnf_convergence_threshold', 0.005),
        verbose=False
    )

    # Initialize components
    _worker_pipeline._initialize_components()


def process_frame_worker(frame_data: Tuple) -> Optional[Dict]:
    """
    Worker function to extract features from a single frame (Steps 1-6)

    Steps performed:
    1. Face detection (PyMTCNN ONNX)
    2. Landmark detection (CLNF)
    3. 3D pose estimation (CalcParams)
    4. Face alignment
    5. HOG feature extraction
    6. Geometric feature extraction

    Returns HOG and geometric features for running median update in main process.
    Does NOT predict AUs (requires running median from main process).
    """
    global _worker_pipeline

    frame_idx, frame_bytes, frame_shape, fps = frame_data

    try:
        # Reconstruct frame from bytes
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)

        # Import pyfhog here to avoid import issues in worker
        import pyfhog

        # Step 1: Face detection
        detections, _ = _worker_pipeline.face_detector.detect_faces(frame)
        if len(detections) == 0:
            return None

        det = detections[0]
        bbox = det[:4]  # [x, y, width, height]

        # Step 2: Landmark detection with CLNF
        bbox_pyclnf = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        landmarks_68, info = _worker_pipeline.landmark_detector.fit(frame, bbox_pyclnf)

        # Step 3: 3D pose estimation
        if _worker_pipeline.use_calc_params and _worker_pipeline.calc_params:
            params_global, params_local = _worker_pipeline.calc_params.calc_params(landmarks_68)
            scale = params_global[0]
            rx, ry, rz = params_global[1:4]
            tx, ty = params_global[4:6]
        else:
            scale = 1.0
            rx = ry = rz = 0.0
            tx = bbox[0] + bbox[2] / 2
            ty = bbox[1] + bbox[3] / 2
            params_local = np.zeros(34)

        # Step 4: Face alignment
        aligned_face = _worker_pipeline.face_aligner.align_face(
            image=frame,
            landmarks_68=landmarks_68,
            pose_tx=tx,
            pose_ty=ty,
            p_rz=rz,
            apply_mask=True,
            triangulation=_worker_pipeline.triangulation
        )

        # Step 5: HOG feature extraction
        hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
        # Apply transpose to match C++ OpenFace HOG ordering
        hog_features = hog_features.reshape(12, 12, 31).transpose(1, 0, 2).flatten()
        hog_features = hog_features.astype(np.float32)

        # Step 6: Geometric feature extraction
        geom_features = _worker_pipeline.pdm_parser.extract_geometric_features(params_local)
        geom_features = geom_features.astype(np.float32)

        return {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'hog_features': hog_features.tobytes(),
            'hog_shape': hog_features.shape,
            'geom_features': geom_features.tobytes(),
            'geom_shape': geom_features.shape
        }

    except Exception as e:
        return None


def get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video"""
    cap = cv2.VideoCapture(video_path)
    rotation = 0
    if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    cap.release()
    return rotation


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation to frame based on metadata"""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class HPCOptimizedPipeline:
    """
    HPC-optimized AU extraction pipeline

    Uses multiprocessing to parallelize feature extraction while
    maintaining sequential running median updates for accuracy.
    """

    def __init__(
        self,
        pdm_file: str,
        au_models_dir: str,
        triangulation_file: str,
        patch_expert_file: str = "",
        num_workers: int = 8,
        batch_size: int = 32,
        max_clnf_iterations: int = 10,
        clnf_convergence_threshold: float = 0.005,  # Gold standard for sub-pixel accuracy
        verbose: bool = True
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose

        self.init_params = {
            'pdm_file': pdm_file,
            'au_models_dir': au_models_dir,
            'triangulation_file': triangulation_file,
            'patch_expert_file': patch_expert_file,
            'max_clnf_iterations': max_clnf_iterations,
            'clnf_convergence_threshold': clnf_convergence_threshold,
            'ort_threads': 1
        }

        # Initialize main process pipeline for AU prediction
        from pyfaceau.pipeline import FullPythonAUPipeline

        self.main_pipeline = FullPythonAUPipeline(
            pdm_file=pdm_file,
            au_models_dir=au_models_dir,
            triangulation_file=triangulation_file,
            patch_expert_file=patch_expert_file,
            mtcnn_backend='onnx',
            use_calc_params=True,
            track_faces=False,
            use_batched_predictor=True,
            max_clnf_iterations=max_clnf_iterations,
            clnf_convergence_threshold=clnf_convergence_threshold,
            verbose=verbose
        )
        self.main_pipeline._initialize_components()

        if verbose:
            print(f"HPC Pipeline initialized with {num_workers} workers")
            print(f"Batch size: {batch_size}")

    def process_video(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process video with parallel feature extraction

        Args:
            video_path: Path to input video
            output_csv: Optional output CSV path
            max_frames: Optional limit on frames

        Returns:
            DataFrame with AU predictions
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = get_video_rotation(str(video_path))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        if self.verbose:
            print(f"\nProcessing: {video_path.name}")
            print(f"  FPS: {fps:.2f}, Frames: {total_frames}, Rotation: {rotation}°")
            print(f"  Workers: {self.num_workers}, Batch: {self.batch_size}")

        results = []
        frame_idx = 0
        start_time = time.time()

        # Create shared counter for staggered worker initialization
        global _worker_counter
        _worker_counter = mp.Value('i', 0)

        # Stagger delay: 2 seconds between workers to avoid I/O contention
        # when loading 410MB CEN patch experts
        stagger_delay = 2.0

        if self.verbose:
            print(f"  Initializing {self.num_workers} workers (staggered, ~{stagger_delay}s apart)...")

        # Create worker pool with staggered initializer
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(self.init_params, stagger_delay)
        ) as pool:

            while frame_idx < total_frames:
                batch_start = time.time()

                # Read batch of frames
                batch_data = []
                for _ in range(self.batch_size):
                    if frame_idx >= total_frames:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Apply rotation
                    if rotation:
                        frame = rotate_frame(frame, rotation)

                    # Pack frame data for worker
                    batch_data.append((
                        frame_idx,
                        frame.tobytes(),
                        frame.shape,
                        fps
                    ))
                    frame_idx += 1

                if not batch_data:
                    break

                # Process batch in parallel
                feature_results = pool.map(process_frame_worker, batch_data)

                # Sequential AU prediction (requires running median)
                for i, feat_result in enumerate(feature_results):
                    if feat_result is None:
                        results.append({
                            'frame': batch_data[i][0],  # frame_idx from batch_data
                            'timestamp': batch_data[i][0] / fps,
                            'success': False
                        })
                        continue

                    # Reconstruct features
                    hog_features = np.frombuffer(
                        feat_result['hog_features'],
                        dtype=np.float32
                    ).reshape(feat_result['hog_shape'])

                    geom_features = np.frombuffer(
                        feat_result['geom_features'],
                        dtype=np.float32
                    ).reshape(feat_result['geom_shape'])

                    # Update running median (sequential)
                    idx = feat_result['frame_idx']
                    update_histogram = (idx % 2 == 1)
                    self.main_pipeline.running_median.update(
                        hog_features,
                        geom_features,
                        update_histogram=update_histogram
                    )
                    running_median = self.main_pipeline.running_median.get_combined_median()

                    # Predict AUs
                    au_results = self.main_pipeline._predict_aus(
                        hog_features,
                        geom_features,
                        running_median
                    )

                    result = {
                        'frame': idx,
                        'timestamp': feat_result['timestamp'],
                        'success': True
                    }
                    result.update(au_results)
                    results.append(result)

                # Progress
                if self.verbose:
                    batch_time = time.time() - batch_start
                    elapsed = time.time() - start_time
                    current_fps = frame_idx / elapsed if elapsed > 0 else 0
                    batch_fps = len(batch_data) / batch_time if batch_time > 0 else 0
                    eta = (total_frames - frame_idx) / current_fps if current_fps > 0 else 0

                    print(f"  Progress: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%) "
                          f"Batch: {batch_fps:.1f} FPS, Overall: {current_fps:.1f} FPS, ETA: {eta:.0f}s")

        cap.release()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        total_time = time.time() - start_time
        success_count = df['success'].sum() if 'success' in df.columns else len(df)
        overall_fps = success_count / total_time if total_time > 0 else 0

        if self.verbose:
            print(f"\n  Complete: {success_count}/{len(df)} frames in {total_time:.1f}s ({overall_fps:.1f} FPS)")

        # Save CSV
        if output_csv:
            df.to_csv(output_csv, index=False)
            if self.verbose:
                print(f"  Saved: {output_csv}")

        return df


def process_video_job(args: Tuple) -> Dict:
    """
    Process a single video (for multi-video parallelization)

    Args:
        args: (video_path, output_path, init_params, workers_per_video, max_frames)

    Returns:
        Dict with results summary
    """
    video_path, output_path, init_params, workers, max_frames = args

    try:
        pipeline = HPCOptimizedPipeline(
            pdm_file=init_params['pdm_file'],
            au_models_dir=init_params['au_models_dir'],
            triangulation_file=init_params['triangulation_file'],
            patch_expert_file=init_params.get('patch_expert_file', ''),
            num_workers=workers,
            batch_size=32,
            verbose=False  # Quiet in multi-video mode
        )

        start = time.time()
        df = pipeline.process_video(video_path, output_path, max_frames)
        elapsed = time.time() - start

        success_count = df['success'].sum() if 'success' in df.columns else len(df)

        return {
            'video': str(video_path),
            'output': str(output_path),
            'frames': len(df),
            'success': int(success_count),
            'time': elapsed,
            'fps': success_count / elapsed if elapsed > 0 else 0,
            'status': 'success'
        }

    except Exception as e:
        return {
            'video': str(video_path),
            'output': str(output_path) if output_path else None,
            'status': 'failed',
            'error': str(e)
        }


def process_video_list(
    video_list: List[str],
    output_dir: str,
    init_params: Dict,
    video_parallel: int = 4,
    workers_per_video: int = 8,
    max_frames: Optional[int] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Process multiple videos in parallel

    Args:
        video_list: List of video paths
        output_dir: Output directory for CSVs
        init_params: Pipeline initialization parameters
        video_parallel: Number of videos to process simultaneously
        workers_per_video: Workers per video
        max_frames: Optional frame limit per video
        verbose: Print progress

    Returns:
        List of result summaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare job arguments
    jobs = []
    for video_path in video_list:
        video_path = Path(video_path)
        output_path = output_dir / f"{video_path.stem}_aus.csv"
        jobs.append((
            str(video_path),
            str(output_path),
            init_params,
            workers_per_video,
            max_frames
        ))

    if verbose:
        print(f"Processing {len(jobs)} videos")
        print(f"  Video parallelism: {video_parallel}")
        print(f"  Workers per video: {workers_per_video}")
        print(f"  Total processes: {video_parallel * workers_per_video}")
        print()

    # Process videos in parallel
    results = []
    start_time = time.time()

    # Use ProcessPoolExecutor for video-level parallelism
    # Note: This creates nested parallelism (videos x workers)
    with ProcessPoolExecutor(max_workers=video_parallel) as executor:
        for i, result in enumerate(executor.map(process_video_job, jobs)):
            results.append(result)
            if verbose:
                status = result['status']
                if status == 'success':
                    print(f"  [{i+1}/{len(jobs)}] {Path(result['video']).name}: "
                          f"{result['frames']} frames, {result['fps']:.1f} FPS")
                else:
                    print(f"  [{i+1}/{len(jobs)}] {Path(result['video']).name}: FAILED - {result.get('error', 'unknown')}")

    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_frames = sum(r.get('frames', 0) for r in results if r['status'] == 'success')

    if verbose:
        print()
        print(f"Complete: {success_count}/{len(jobs)} videos")
        print(f"Total frames: {total_frames}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Throughput: {total_frames/total_time:.1f} FPS")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HPC Optimized AU Pipeline for BigRed200",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', help='Single video file')
    input_group.add_argument('--video-list', help='File containing video paths (one per line)')

    # Output options
    parser.add_argument('--output', help='Output CSV (single video mode)')
    parser.add_argument('--output-dir', help='Output directory (multi-video mode)')

    # Parallelism options
    parser.add_argument('--workers', type=int, default=8,
                        help='Workers per video (default: 8)')
    parser.add_argument('--video-parallel', type=int, default=1,
                        help='Videos to process simultaneously (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Frames per batch (default: 32)')
    parser.add_argument('--auto-config', action='store_true',
                        help='Auto-configure based on SLURM resources')

    # Processing options
    parser.add_argument('--max-frames', type=int, help='Max frames per video')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Auto-configure based on SLURM
    if args.auto_config:
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
        num_videos = len(open(args.video_list).readlines()) if args.video_list else 1
        config = get_optimal_thread_config(cpus, min(num_videos, 8))
        args.workers = config['workers_per_video']
        args.video_parallel = config['video_parallel']
        print(f"Auto-configured: {args.workers} workers/video, {args.video_parallel} parallel videos")

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

    init_params = {
        'pdm_file': str(pdm_file),
        'au_models_dir': str(au_models_dir),
        'triangulation_file': str(triangulation_file),
        'patch_expert_file': ''
    }

    print("=" * 70)
    print("HPC OPTIMIZED AU PIPELINE")
    print("=" * 70)

    if args.video:
        # Single video mode
        if not args.output:
            video_path = Path(args.video)
            args.output = str(video_path.parent / f"{video_path.stem}_optimized_aus.csv")

        print(f"Video: {args.video}")
        print(f"Output: {args.output}")
        print(f"Workers: {args.workers}")
        print()

        pipeline = HPCOptimizedPipeline(
            pdm_file=str(pdm_file),
            au_models_dir=str(au_models_dir),
            triangulation_file=str(triangulation_file),
            num_workers=args.workers,
            batch_size=args.batch_size,
            verbose=args.verbose or True
        )

        start = time.time()
        df = pipeline.process_video(args.video, args.output, args.max_frames)
        elapsed = time.time() - start

        print()
        print("=" * 70)
        print("COMPLETE")
        print("=" * 70)
        success_count = df['success'].sum() if 'success' in df.columns else len(df)
        print(f"Frames: {success_count}/{len(df)}")
        print(f"Time: {elapsed:.1f}s ({success_count/elapsed:.1f} FPS)")
        print(f"Output: {args.output}")

    else:
        # Multi-video mode
        if not args.output_dir:
            args.output_dir = 'optimized_results'

        with open(args.video_list) as f:
            video_list = [line.strip() for line in f if line.strip()]

        print(f"Videos: {len(video_list)}")
        print(f"Output dir: {args.output_dir}")
        print(f"Video parallelism: {args.video_parallel}")
        print(f"Workers per video: {args.workers}")
        print()

        results = process_video_list(
            video_list=video_list,
            output_dir=args.output_dir,
            init_params=init_params,
            video_parallel=args.video_parallel,
            workers_per_video=args.workers,
            max_frames=args.max_frames,
            verbose=args.verbose or True
        )

        # Save summary
        summary_path = Path(args.output_dir) / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
