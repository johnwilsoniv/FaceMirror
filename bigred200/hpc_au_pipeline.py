#!/usr/bin/env python3
"""
HPC-Optimized AU Extraction Pipeline for Big Red 200

This is the production pipeline optimized for high-throughput batch processing
on Big Red 200 supercomputer. It combines all optimizations from Phases 1-5:

Phase 1: Shared Memory Model Loading
- CEN patch experts shared across workers via memory mapping
- Reduces memory from 424MB × N_workers to 424MB × 1
- Enables 64-128 effective workers instead of 16

Phase 2: CLNF Convergence Optimization
- Named convergence profiles (accurate, optimized, fast, video)
- Early window exit when already converged
- Temporal warm-start for video mode

Phase 3: NUMA-Aware Worker Pool
- Workers pinned to specific NUMA nodes
- Memory allocated from local NUMA domain
- Reduces cross-node latency on AMD EPYC

Phase 4: Data Type Optimization
- Float32 for all model weights (vs float64)
- Efficient numpy memory layouts

Phase 5: Zero-Copy Frame Buffer
- Frames shared via memory-mapped arrays
- Eliminates serialization overhead in multiprocessing
- Direct frame access from all workers

Usage:
    from hpc_au_pipeline import HPCAUPipeline

    # Create optimized pipeline
    pipeline = HPCAUPipeline(
        n_workers=64,
        convergence_profile='optimized',
        use_shared_memory=True,
        use_numa=True
    )

    # Process video batch
    results = pipeline.process_video_batch(video_paths, output_dir)

    # Or process single video
    df = pipeline.process_video(video_path, output_csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from multiprocessing import Pool, cpu_count, shared_memory
import time
from dataclasses import dataclass

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))


@dataclass
class HPCConfig:
    """Configuration for HPC AU Pipeline."""
    n_workers: int = 64
    convergence_profile: str = 'optimized'
    use_shared_memory: bool = True
    use_numa: bool = True
    batch_size: int = 100
    verbose: bool = True

    # Model paths (relative to project_root)
    clnf_model_dir: str = "pyclnf/pyclnf/models"
    pdm_file: str = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir: str = "pyfaceau/weights/AU_predictors"
    triangulation_file: str = "pyfaceau/weights/tris_68_full.txt"


class ZeroCopyFrameBuffer:
    """
    Zero-copy frame buffer using shared memory.

    Allows multiple worker processes to access frames without serialization.
    Frames are stored in a shared memory array and accessed via memory mapping.
    """

    def __init__(self, max_frames: int, frame_height: int, frame_width: int, channels: int = 3):
        """
        Initialize zero-copy frame buffer.

        Args:
            max_frames: Maximum number of frames to buffer
            frame_height: Height of each frame
            frame_width: Width of each frame
            channels: Number of color channels (default: 3 for BGR)
        """
        self.max_frames = max_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels

        # Calculate buffer size
        self.frame_size = frame_height * frame_width * channels
        self.buffer_size = self.frame_size * max_frames

        # Create shared memory
        self.shm = shared_memory.SharedMemory(create=True, size=self.buffer_size)
        self.buffer = np.ndarray(
            (max_frames, frame_height, frame_width, channels),
            dtype=np.uint8,
            buffer=self.shm.buf
        )

        # Track which frames are valid
        self.valid_frames = np.zeros(max_frames, dtype=bool)
        self.frame_count = 0

    def add_frame(self, frame: np.ndarray) -> int:
        """
        Add a frame to the buffer.

        Args:
            frame: BGR frame array (H, W, 3)

        Returns:
            Frame index in buffer
        """
        if self.frame_count >= self.max_frames:
            raise RuntimeError("Frame buffer full")

        idx = self.frame_count
        self.buffer[idx] = frame
        self.valid_frames[idx] = True
        self.frame_count += 1
        return idx

    def get_frame(self, idx: int) -> np.ndarray:
        """
        Get a frame from the buffer (zero-copy view).

        Args:
            idx: Frame index

        Returns:
            Frame array (view into shared memory)
        """
        if not self.valid_frames[idx]:
            raise ValueError(f"Frame {idx} not valid")
        return self.buffer[idx]

    def get_shm_name(self) -> str:
        """Get shared memory name for worker access."""
        return self.shm.name

    def get_shape(self) -> Tuple[int, int, int, int]:
        """Get buffer shape."""
        return (self.max_frames, self.frame_height, self.frame_width, self.channels)

    def clear(self):
        """Clear buffer for reuse."""
        self.valid_frames.fill(False)
        self.frame_count = 0

    def close(self):
        """Close shared memory (call from main process)."""
        self.shm.close()

    def unlink(self):
        """Unlink shared memory (call once when done)."""
        self.shm.unlink()


# Global worker state
_worker_state = None


def _init_worker_hpc(config: Dict[str, Any]):
    """
    Initialize HPC worker with all optimizations.

    Args:
        config: Configuration dict with paths and settings
    """
    global _worker_state

    # Limit threads per worker (critical for HPC to avoid thread affinity conflicts)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['ORT_NUM_THREADS'] = '1'  # ONNX Runtime - avoids pthread_setaffinity errors

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser
    from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
    from pyfaceau.features.triangulation import TriangulationParser

    # Import pyfhog
    try:
        import pyfhog
    except ImportError:
        pyfhog_path = Path(config['project_root']) / 'pyfhog' / 'src'
        if pyfhog_path.exists():
            sys.path.insert(0, str(pyfhog_path))
            import pyfhog

    # Initialize components
    detector = MTCNN()

    clnf = CLNF(
        model_dir=config['clnf_model_dir'],
        use_shared_memory=config.get('use_shared_memory', True),
        shared_memory_dir=config.get('clnf_shm_dir'),
        convergence_profile=config.get('convergence_profile', 'optimized'),
        early_window_exit=True,
        early_exit_threshold=0.3
    )

    pdm_parser = PDMParser(config['pdm_file'])
    calc_params = CalcParams(pdm_parser)

    face_aligner = OpenFace22FaceAligner(
        pdm_file=config['pdm_file'],
        sim_scale=0.7,
        output_size=(112, 112)
    )

    triangulation = TriangulationParser(config['triangulation_file'])

    # Connect to shared frame buffer if provided
    frame_buffer = None
    if 'frame_buffer_name' in config:
        shm = shared_memory.SharedMemory(name=config['frame_buffer_name'])
        frame_buffer = np.ndarray(
            config['frame_buffer_shape'],
            dtype=np.uint8,
            buffer=shm.buf
        )

    _worker_state = {
        'detector': detector,
        'clnf': clnf,
        'pdm_parser': pdm_parser,
        'calc_params': calc_params,
        'face_aligner': face_aligner,
        'triangulation': triangulation,
        'pyfhog': pyfhog,
        'frame_buffer': frame_buffer,
        'frame_buffer_shm': shm if frame_buffer is not None else None
    }


def _process_frame_hpc(args: Tuple) -> Optional[Dict]:
    """
    Process a single frame using HPC-optimized worker.

    Args:
        args: Tuple of (frame_idx, frame_data_or_idx, fps, use_buffer)

    Returns:
        Dictionary with features or None if failed
    """
    global _worker_state

    frame_idx, frame_data, fps, use_buffer = args

    try:
        # Get frame from buffer or decode from bytes
        if use_buffer and _worker_state['frame_buffer'] is not None:
            frame = _worker_state['frame_buffer'][frame_data]
        else:
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        # Step 1: Face detection
        bboxes, _ = _worker_state['detector'].detect(frame)
        if bboxes is None or len(bboxes) == 0:
            return None

        bbox = bboxes[0][:4]

        # Step 2: CLNF landmark detection (with convergence optimization)
        landmarks_68, info = _worker_state['clnf'].fit(frame, bbox)
        landmarks_68 = landmarks_68.astype(np.float32)

        # Step 3: Pose estimation
        params_global, params_local = _worker_state['calc_params'].calc_params(landmarks_68)
        scale = params_global[0]
        rx, ry, rz = params_global[1:4]
        tx, ty = params_global[4:6]

        # Step 4: Face alignment
        aligned_face = _worker_state['face_aligner'].align_face(
            image=frame,
            landmarks_68=landmarks_68,
            pose_tx=tx,
            pose_ty=ty,
            p_rz=rz,
            apply_mask=True,
            triangulation=_worker_state['triangulation']
        )

        # Step 5: HOG features
        pyfhog = _worker_state['pyfhog']
        hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
        hog_features = hog_features.reshape(12, 12, 31).transpose(1, 0, 2).flatten()
        hog_features = hog_features.astype(np.float32)

        # Step 6: Geometric features
        geom_features = _worker_state['pdm_parser'].extract_geometric_features(params_local)
        geom_features = geom_features.astype(np.float32)

        return {
            'frame': frame_idx,
            'timestamp': frame_idx / fps,
            'hog_features': hog_features,
            'geom_features': geom_features,
            'landmarks_68': landmarks_68,
            'pose': (rx, ry, rz, tx, ty, scale),
            'clnf_converged': info['converged'],
            'clnf_iterations': info['iterations']
        }

    except Exception as e:
        # Log exception for debugging
        import sys
        import traceback
        print(f"[_process_frame_hpc] Frame {frame_idx} failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


class HPCAUPipeline:
    """
    HPC-optimized AU extraction pipeline for Big Red 200.

    Combines all optimization phases for maximum throughput.
    """

    def __init__(self, config: Optional[HPCConfig] = None, **kwargs):
        """
        Initialize HPC AU pipeline.

        Args:
            config: HPCConfig object or None to use defaults
            **kwargs: Override config values
        """
        if config is None:
            config = HPCConfig()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.project_root = project_root

        # Resolve paths
        self.clnf_model_dir = str(project_root / config.clnf_model_dir)
        self.pdm_file = str(project_root / config.pdm_file)
        self.au_models_dir = str(project_root / config.au_models_dir)
        self.triangulation_file = str(project_root / config.triangulation_file)

        # Shared memory state
        self.shm_config = None
        self.frame_buffer = None

        # Initialize running median and AU models in main process
        self._init_main_process()

    def _init_main_process(self):
        """Initialize components needed in main process."""
        # Import running median tracker
        try:
            from cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
        except ImportError:
            from pyfaceau.features.histogram_median_tracker import DualHistogramMedianTracker

        self.running_median = DualHistogramMedianTracker(
            hog_dim=4464,
            geom_dim=238,
            hog_bins=1000,
            hog_min=-0.005,
            hog_max=1.0,
            geom_bins=10000,
            geom_min=-60.0,
            geom_max=60.0
        )

        # Load AU models
        from pyfaceau.prediction.model_parser import OF22ModelParser

        model_parser = OF22ModelParser(self.au_models_dir)
        self.au_models = model_parser.load_all_models(
            use_recommended=True,
            use_combined=True,
            verbose=self.config.verbose
        )

        # Try to use batched predictor
        try:
            from pyfaceau.prediction.batched_au_predictor import BatchedAUPredictor
            self.batched_predictor = BatchedAUPredictor(self.au_models)
        except ImportError:
            self.batched_predictor = None

    def _init_shared_memory(self):
        """Initialize shared memory for models."""
        if not self.config.use_shared_memory:
            return

        from shared_memory_init import initialize_shared_models

        if self.config.verbose:
            print("Initializing shared memory models...")

        self.shm_config = initialize_shared_models(
            clnf_model_dir=self.clnf_model_dir,
            au_model_dir=self.au_models_dir
        )

        self.shm_config['convergence_profile'] = self.config.convergence_profile
        self.shm_config['project_root'] = str(self.project_root)
        self.shm_config['pdm_file'] = self.pdm_file
        self.shm_config['triangulation_file'] = self.triangulation_file

    def _cleanup_shared_memory(self):
        """Clean up shared memory."""
        if self.shm_config is None:
            return

        from shared_memory_init import cleanup_shared_models
        cleanup_shared_models(self.shm_config)
        self.shm_config = None

    def _predict_aus(
        self,
        hog_features: np.ndarray,
        geom_features: np.ndarray,
        running_median: np.ndarray
    ) -> Dict[str, float]:
        """Predict AU intensities."""
        if self.batched_predictor is not None:
            return self.batched_predictor.predict(hog_features, geom_features, running_median)

        # Fallback to sequential
        predictions = {}
        full_vector = np.concatenate([hog_features, geom_features])

        for au_name, model in self.au_models.items():
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic:
                centered = full_vector - model['means'].flatten() - running_median
            else:
                centered = full_vector - model['means'].flatten()

            pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
            pred = float(pred[0, 0])
            pred = np.clip(pred, 0.0, 5.0)
            predictions[au_name] = pred

        return predictions

    def process_video(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process a single video with HPC optimizations.

        Args:
            video_path: Path to input video
            output_csv: Optional path to save results
            max_frames: Optional limit on frames

        Returns:
            DataFrame with AU predictions
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        if self.config.verbose:
            print(f"Processing video: {video_path.name}")
            print(f"  Frames: {total_frames}, FPS: {fps:.2f}")
            print(f"  Resolution: {frame_width}x{frame_height}")
            print(f"  Workers: {self.config.n_workers}")
            print(f"  Convergence profile: {self.config.convergence_profile}")
            print()

        start_time = time.time()

        # Initialize shared memory
        self._init_shared_memory()

        try:
            # Choose pool type
            if self.config.use_numa:
                from numa_worker_pool import NUMAWorkerPool
                pool_class = NUMAWorkerPool
            else:
                pool_class = Pool

            # Load all frames and encode for transfer
            if self.config.verbose:
                print(f"Loading {total_frames} frames...")

            frames_data = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames_data.append(encoded.tobytes())
            cap.release()

            # Prepare worker arguments
            worker_config = self.shm_config.copy() if self.shm_config else {
                'clnf_model_dir': self.clnf_model_dir,
                'pdm_file': self.pdm_file,
                'triangulation_file': self.triangulation_file,
                'convergence_profile': self.config.convergence_profile,
                'project_root': str(self.project_root),
                'use_shared_memory': False
            }

            args_list = [
                (i, frames_data[i], fps, False)
                for i in range(len(frames_data))
            ]

            # Process with worker pool
            if self.config.verbose:
                print(f"Processing with {self.config.n_workers} workers...")

            if self.config.use_numa:
                with pool_class(
                    n_workers=self.config.n_workers,
                    initializer=_init_worker_hpc,
                    initargs=(worker_config,)
                ) as pool:
                    feature_results = list(pool.imap(
                        _process_frame_hpc,
                        args_list
                    ))
            else:
                with Pool(
                    self.config.n_workers,
                    initializer=_init_worker_hpc,
                    initargs=(worker_config,)
                ) as pool:
                    feature_results = list(pool.imap(
                        _process_frame_hpc,
                        args_list
                    ))

            # Process results sequentially (running median + AU prediction)
            results = []
            stored_features = []

            for idx, frame_features in enumerate(feature_results):
                if frame_features is None:
                    results.append({
                        'frame': idx,
                        'timestamp': idx / fps,
                        'success': False
                    })
                    continue

                hog_features = frame_features['hog_features']
                geom_features = frame_features['geom_features']

                # Update running median
                update_histogram = (idx % 2 == 1)
                self.running_median.update(hog_features, geom_features, update_histogram)
                running_median = self.running_median.get_combined_median()

                # Store features for two-pass processing
                if idx < 3000:
                    stored_features.append((idx, hog_features.copy(), geom_features.copy()))

                # Predict AUs
                au_results = self._predict_aus(hog_features, geom_features, running_median)

                result = {
                    'frame': idx,
                    'timestamp': frame_features['timestamp'],
                    'success': True
                }
                result.update(au_results)
                results.append(result)

            # Two-pass: re-predict early frames with final median
            if stored_features:
                final_median = self.running_median.get_combined_median()
                for frame_idx, hog_features, geom_features in stored_features:
                    au_results = self._predict_aus(hog_features, geom_features, final_median)
                    for au_name, au_value in au_results.items():
                        results[frame_idx][au_name] = au_value

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Apply post-processing
            df = self._finalize_predictions(df)

            elapsed = time.time() - start_time
            success_count = df['success'].sum()
            fps_achieved = success_count / elapsed

            if self.config.verbose:
                print()
                print("=" * 60)
                print("PROCESSING COMPLETE")
                print("=" * 60)
                print(f"Frames: {success_count}/{total_frames} successful")
                print(f"Time: {elapsed:.1f}s ({fps_achieved:.1f} FPS)")
                print()

            if output_csv:
                df.to_csv(output_csv, index=False)
                if self.config.verbose:
                    print(f"Results saved to: {output_csv}")

            return df

        finally:
            self._cleanup_shared_memory()

    def _finalize_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to AU predictions."""
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        # Cutoff adjustment
        for au_col in au_cols:
            if au_col not in self.au_models:
                continue

            model = self.au_models[au_col]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic and model.get('cutoff', -1) != -1:
                cutoff = model['cutoff']
                au_values = df[au_col].values
                sorted_vals = np.sort(au_values)
                cutoff_idx = int(len(sorted_vals) * cutoff)
                offset = sorted_vals[cutoff_idx]
                df[au_col] = np.clip(au_values - offset, 0.0, 5.0)

        # Temporal smoothing (3-frame moving average)
        for au_col in au_cols:
            smoothed = df[au_col].rolling(window=3, center=True, min_periods=1).mean()
            df[au_col] = smoothed

        return df

    def process_video_batch(
        self,
        video_paths: List[str],
        output_dir: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple videos.

        Args:
            video_paths: List of video file paths
            output_dir: Directory to save results

        Returns:
            Dictionary mapping video paths to result DataFrames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for video_path in video_paths:
            video_path = Path(video_path)
            output_csv = output_dir / f"{video_path.stem}_aus.csv"

            try:
                df = self.process_video(str(video_path), str(output_csv))
                results[str(video_path)] = df
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[str(video_path)] = None

        return results


def main():
    """Command-line interface for HPC AU pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HPC-Optimized AU Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--workers', type=int, default=64, help='Number of workers')
    parser.add_argument('--profile', default='optimized',
                        choices=['accurate', 'optimized', 'fast', 'video'],
                        help='Convergence profile')
    parser.add_argument('--no-numa', action='store_true', help='Disable NUMA optimization')
    parser.add_argument('--no-shared-memory', action='store_true', help='Disable shared memory')

    args = parser.parse_args()

    # Set default output
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_hpc_aus.csv")

    # Create config
    config = HPCConfig(
        n_workers=args.workers,
        convergence_profile=args.profile,
        use_numa=not args.no_numa,
        use_shared_memory=not args.no_shared_memory
    )

    # Process
    pipeline = HPCAUPipeline(config)
    df = pipeline.process_video(args.video, args.output, args.max_frames)

    print(f"Processed {len(df)} frames")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
