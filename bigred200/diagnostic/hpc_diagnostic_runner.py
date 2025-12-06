"""
HPC Diagnostic Runner for Parallel Frame Processing

Uses Python multiprocessing to process video frames in parallel,
collecting detailed diagnostic data from each frame.

Designed for Big Red 200 with 64+ CPU cores.
Uses staggered worker initialization to avoid I/O contention.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from multiprocessing import Pool, cpu_count, Manager
import threading
from tqdm import tqdm
import time
import json

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyclnf'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pymtcnn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyfaceau'))

from .data_structures import FrameDiagnostic, save_all_diagnostics


# Global variables for worker processes
_worker_detector = None
_worker_clnf = None
_worker_calc_params = None
_init_lock = None


def init_diagnostic_worker(lock, capture_response_maps: bool = True,
                           target_landmarks: Optional[List[int]] = None):
    """
    Initialize worker process with instrumented CLNF.

    Called once per worker process.
    Uses staggered initialization to avoid I/O contention.
    """
    global _worker_detector, _worker_clnf, _worker_calc_params, _init_lock
    _init_lock = lock

    # Acquire lock - only one worker initializes at a time
    lock.acquire()
    try:
        # Limit threads per worker to avoid contention
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfaceau.alignment.calc_params import CalcParams
        from pyfaceau.features.pdm import PDMParser

        from .instrumented_optimizer import InstrumentedNURLMSOptimizer

        # Initialize detector
        _worker_detector = MTCNN()

        # Initialize CLNF
        _worker_clnf = CLNF(model_dir="pyclnf/pyclnf/models")

        # Replace optimizer with instrumented version
        _worker_clnf.optimizer = InstrumentedNURLMSOptimizer(
            capture_response_maps=capture_response_maps,
            target_landmarks=target_landmarks,
            regularization=_worker_clnf.optimizer.regularization,
            max_iterations=_worker_clnf.optimizer.max_iterations,
            convergence_threshold=_worker_clnf.optimizer.convergence_threshold,
            sigma=_worker_clnf.optimizer.sigma,
            weight_multiplier=_worker_clnf.optimizer.weight_multiplier,
        )

        # Initialize pose estimator
        pdm_parser = PDMParser("pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")
        _worker_calc_params = CalcParams(pdm_parser)
    finally:
        # Release lock after a small delay to let this worker finish I/O
        # before next worker starts (0.1s stagger between workers)
        threading.Timer(0.1, lock.release).start()


def process_frame_diagnostic(args: Tuple) -> Optional[FrameDiagnostic]:
    """
    Process a single frame and return diagnostics.

    Args:
        args: Tuple of (frame_idx, frame_data_bytes, cpp_landmarks, cpp_pose, video_name)

    Returns:
        FrameDiagnostic or None if processing failed
    """
    frame_idx, frame_data, cpp_landmarks, cpp_pose, video_name = args

    global _worker_detector, _worker_clnf, _worker_calc_params

    start_time = time.time()

    # Create diagnostic object
    diag = FrameDiagnostic(
        frame_idx=frame_idx,
        video_name=video_name,
    )

    try:
        # Decode frame from bytes
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        # Set C++ reference
        diag.cpp_landmarks = cpp_landmarks
        diag.cpp_pose = tuple(cpp_pose) if cpp_pose is not None else None

        # Set context for optimizer
        _worker_clnf.optimizer.set_frame_context(frame_idx, cpp_landmarks)

        # Face detection
        bboxes, _ = _worker_detector.detect(frame)
        if bboxes is None or len(bboxes) == 0:
            diag.success = False
            diag.error_message = "No face detected"
            return diag

        diag.detection_bbox = bboxes[0][:4]

        # CLNF fitting (uses instrumented optimizer)
        py_landmarks, info = _worker_clnf.fit(frame, diag.detection_bbox)
        py_landmarks = py_landmarks.astype(np.float32)
        diag.py_landmarks = py_landmarks

        # Get diagnostics from optimizer
        diag.iterations = _worker_clnf.optimizer.get_diagnostics()

        # Pose estimation
        global_params, local_params = _worker_calc_params.calc_params(py_landmarks)
        diag.py_pose = tuple(global_params[1:4])  # rx, ry, rz
        diag.py_params = global_params

        # Compute errors
        diag.compute_errors()

        diag.success = True

    except Exception as e:
        diag.success = False
        diag.error_message = str(e)

    diag.processing_time_ms = (time.time() - start_time) * 1000
    return diag


class HPCDiagnosticRunner:
    """
    Run diagnostic analysis across frames using multiprocessing.

    Designed for Big Red 200 HPC cluster with 64+ cores.
    """

    def __init__(self,
                 n_workers: Optional[int] = None,
                 capture_response_maps: bool = True,
                 target_landmarks: Optional[List[int]] = None):
        """
        Initialize HPC diagnostic runner.

        Args:
            n_workers: Number of worker processes (default: from SLURM or CPU count)
            capture_response_maps: Whether to save full response maps
            target_landmarks: Specific landmarks to track (None = all 68)
        """
        self.n_workers = n_workers or int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
        self.capture_response_maps = capture_response_maps
        self.target_landmarks = target_landmarks

        print(f"HPCDiagnosticRunner initialized with {self.n_workers} workers")
        if capture_response_maps:
            print("  Response maps: ENABLED (full detail)")
        else:
            print("  Response maps: DISABLED (summary only)")

    def load_cpp_reference(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load C++ OpenFace reference data from CSV.

        Returns:
            cpp_landmarks: (n_frames, 68, 2) landmark positions
            cpp_pose: (n_frames, 3) pose angles (rx, ry, rz)
        """
        df = pd.read_csv(csv_path)
        n_frames = len(df)

        # Extract landmarks
        cpp_landmarks = np.zeros((n_frames, 68, 2))
        for i in range(68):
            cpp_landmarks[:, i, 0] = df[f'x_{i}'].values
            cpp_landmarks[:, i, 1] = df[f'y_{i}'].values

        # Extract pose
        cpp_pose = df[['pose_Rx', 'pose_Ry', 'pose_Rz']].values

        return cpp_landmarks, cpp_pose

    def load_video_frames(self, video_path: str,
                          max_frames: Optional[int] = None) -> List[bytes]:
        """
        Load video frames into memory as encoded bytes.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to load (None = all)

        Returns:
            List of JPEG-encoded frame bytes
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        print(f"Loading {total_frames} frames from {video_path}...")
        frames_data = []

        for i in tqdm(range(total_frames), desc="Loading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            # Encode as JPEG for multiprocessing
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames_data.append(encoded.tobytes())

        cap.release()
        print(f"Loaded {len(frames_data)} frames")
        return frames_data

    def process_video(self,
                      video_path: str,
                      cpp_csv_path: str,
                      output_dir: str,
                      max_frames: Optional[int] = None) -> List[FrameDiagnostic]:
        """
        Process video with diagnostic collection.

        Args:
            video_path: Path to video file
            cpp_csv_path: Path to C++ OpenFace CSV output
            output_dir: Directory to save diagnostic output
            max_frames: Maximum frames to process (None = all)

        Returns:
            List of FrameDiagnostic objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load reference data
        print(f"\nLoading C++ reference from {cpp_csv_path}...")
        cpp_landmarks, cpp_pose = self.load_cpp_reference(cpp_csv_path)
        print(f"  {len(cpp_landmarks)} frames of reference data")

        # Load video frames
        frames_data = self.load_video_frames(video_path, max_frames)
        n_frames = len(frames_data)

        # Extract video name
        video_name = Path(video_path).stem

        # Prepare tasks
        tasks = []
        for i in range(n_frames):
            cpp_lm = cpp_landmarks[i] if i < len(cpp_landmarks) else None
            cpp_p = cpp_pose[i] if i < len(cpp_pose) else None
            tasks.append((i, frames_data[i], cpp_lm, cpp_p, video_name))

        # Process in parallel with staggered worker initialization
        print(f"\nProcessing {n_frames} frames with {self.n_workers} workers (staggered init)...")
        start_time = time.time()

        # Create a manager lock for staggered initialization
        manager = Manager()
        init_lock = manager.Lock()

        # Create initializer args (lock first, then other args)
        init_args = (init_lock, self.capture_response_maps, self.target_landmarks)

        with Pool(self.n_workers, initializer=init_diagnostic_worker,
                  initargs=init_args) as pool:
            results = list(tqdm(
                pool.imap(process_frame_diagnostic, tasks),
                total=len(tasks),
                desc="Processing"
            ))

        elapsed = time.time() - start_time
        fps = n_frames / elapsed

        # Filter out failed frames
        successful = [r for r in results if r is not None and r.success]
        failed = n_frames - len(successful)

        print(f"\nProcessing complete:")
        print(f"  Successful: {len(successful)}/{n_frames}")
        print(f"  Failed: {failed}")
        print(f"  Time: {elapsed:.1f}s ({fps:.1f} fps)")

        # Save diagnostics
        print(f"\nSaving diagnostics to {output_dir}...")
        save_all_diagnostics(successful, output_dir,
                             save_response_maps=self.capture_response_maps)

        # Save summary statistics
        self._save_summary(successful, output_dir)

        return results

    def _save_summary(self, diagnostics: List[FrameDiagnostic], output_dir: Path):
        """Save summary statistics."""
        if not diagnostics:
            return

        # Compute aggregate statistics
        all_errors = np.concatenate([d.landmark_errors for d in diagnostics if d.landmark_errors is not None])

        # Region-wise statistics
        region_errors = {}
        for d in diagnostics:
            if d.landmark_errors is not None:
                for region, error in d.get_region_errors().items():
                    if region not in region_errors:
                        region_errors[region] = []
                    region_errors[region].append(error)

        summary = {
            'n_frames': len(diagnostics),
            'overall': {
                'mean_error': float(np.mean(all_errors)),
                'std_error': float(np.std(all_errors)),
                'max_error': float(np.max(all_errors)),
                'median_error': float(np.median(all_errors)),
            },
            'regions': {
                region: {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'max': float(np.max(errors)),
                }
                for region, errors in region_errors.items()
            }
        }

        with open(output_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"\nOverall: {summary['overall']['mean_error']:.3f} +/- {summary['overall']['std_error']:.3f} px")
        print(f"Max: {summary['overall']['max_error']:.3f} px")
        print("\nPer-region errors:")
        for region, stats in summary['regions'].items():
            print(f"  {region:18s}: {stats['mean']:.3f} +/- {stats['std']:.3f} px")


def main():
    """Main entry point for diagnostic analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='HPC Diagnostic Analysis for pyCLNF')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--cpp-reference', required=True, help='Path to C++ OpenFace CSV')
    parser.add_argument('--output-dir', default='diagnostic_output', help='Output directory')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--no-response-maps', action='store_true', help='Disable response map saving')
    parser.add_argument('--target-landmarks', type=str, default=None,
                        help='Comma-separated landmark indices to track (default: all)')

    args = parser.parse_args()

    # Parse target landmarks
    target_landmarks = None
    if args.target_landmarks:
        target_landmarks = [int(x.strip()) for x in args.target_landmarks.split(',')]

    # Create runner
    runner = HPCDiagnosticRunner(
        n_workers=args.workers,
        capture_response_maps=not args.no_response_maps,
        target_landmarks=target_landmarks,
    )

    # Process video
    runner.process_video(
        video_path=args.video,
        cpp_csv_path=args.cpp_reference,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()
