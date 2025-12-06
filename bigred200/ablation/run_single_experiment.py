"""
Run Single Ablation Experiment

Worker script for SLURM array jobs.
Loads configuration from manifest using SLURM_ARRAY_TASK_ID,
processes test frames, and saves results.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyclnf'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pymtcnn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyfaceau'))


# Landmark region definitions
REGIONS = {
    'jaw': list(range(0, 17)),
    'right_eyebrow': list(range(17, 22)),
    'left_eyebrow': list(range(22, 27)),
    'nose': list(range(27, 36)),
    'right_eye': list(range(36, 42)),
    'left_eye': list(range(42, 48)),
    'outer_mouth': list(range(48, 60)),
    'inner_mouth': list(range(60, 68)),
}


def load_config_from_manifest(manifest_path: str, task_id: int) -> Dict:
    """
    Load experiment configuration from manifest CSV.

    Args:
        manifest_path: Path to experiment manifest CSV
        task_id: SLURM_ARRAY_TASK_ID (row index)

    Returns:
        Dict with experiment parameters
    """
    df = pd.read_csv(manifest_path)

    if task_id >= len(df):
        raise ValueError(f"Task ID {task_id} exceeds manifest size {len(df)}")

    row = df.iloc[task_id]
    config = row.to_dict()

    return config


def load_cpp_reference(csv_path: str) -> tuple:
    """Load C++ OpenFace reference data."""
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    cpp_landmarks = np.zeros((n_frames, 68, 2))
    for i in range(68):
        cpp_landmarks[:, i, 0] = df[f'x_{i}'].values
        cpp_landmarks[:, i, 1] = df[f'y_{i}'].values

    cpp_pose = df[['pose_Rx', 'pose_Ry', 'pose_Rz']].values

    return cpp_landmarks, cpp_pose


def run_experiment(config: Dict,
                   video_path: str,
                   cpp_csv_path: str,
                   max_frames: int = 100) -> Dict:
    """
    Run single experiment with given configuration.

    Args:
        config: Parameter configuration dict
        video_path: Path to video file
        cpp_csv_path: Path to C++ reference CSV
        max_frames: Maximum frames to process

    Returns:
        Results dict with metrics
    """
    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser

    # Initialize pipeline with experiment parameters
    detector = MTCNN()

    # Extract CLNF parameters from config
    clnf_params = {}
    for key in ['regularization', 'max_iterations', 'convergence_threshold',
                'sigma', 'weight_multiplier']:
        if key in config:
            clnf_params[key] = config[key]

    clnf = CLNF(model_dir="pyclnf/pyclnf/models", **clnf_params)

    pdm_parser = PDMParser("pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")
    calc_params = CalcParams(pdm_parser)

    # Load reference data
    cpp_landmarks, cpp_pose = load_cpp_reference(cpp_csv_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    # Process frames
    all_errors = []
    region_errors = {r: [] for r in REGIONS}
    pose_errors = {'rx': [], 'ry': [], 'rz': []}
    processing_times = []
    failed = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        try:
            # Detection
            bboxes, _ = detector.detect(frame)
            if bboxes is None or len(bboxes) == 0:
                failed += 1
                continue

            # CLNF fitting
            py_landmarks, info = clnf.fit(frame, bboxes[0][:4])
            py_landmarks = py_landmarks.astype(np.float32)

            # Pose estimation
            global_params, _ = calc_params.calc_params(py_landmarks)

            # Compute errors
            cpp_lm = cpp_landmarks[frame_idx]
            lm_errors = np.linalg.norm(py_landmarks - cpp_lm, axis=1)
            all_errors.append(np.mean(lm_errors))

            # Per-region errors
            for region, indices in REGIONS.items():
                region_errors[region].append(np.mean(lm_errors[indices]))

            # Pose errors
            cpp_rx, cpp_ry, cpp_rz = cpp_pose[frame_idx]
            py_rx, py_ry, py_rz = global_params[1:4]
            pose_errors['rx'].append(abs(np.degrees(py_rx - cpp_rx)))
            pose_errors['ry'].append(abs(np.degrees(py_ry - cpp_ry)))
            pose_errors['rz'].append(abs(np.degrees(py_rz - cpp_rz)))

        except Exception as e:
            failed += 1
            continue

        processing_times.append(time.time() - start_time)

    cap.release()

    # Compute summary metrics
    results = {
        'experiment_id': int(config.get('experiment_id', 0)),
        'config': {k: v for k, v in config.items() if k != 'experiment_id'},
        'metrics': {
            'overall': {
                'mean_error': float(np.mean(all_errors)) if all_errors else None,
                'std_error': float(np.std(all_errors)) if all_errors else None,
                'max_error': float(np.max(all_errors)) if all_errors else None,
                'median_error': float(np.median(all_errors)) if all_errors else None,
            },
            'regions': {
                region: {
                    'mean_error': float(np.mean(errors)) if errors else None,
                    'std_error': float(np.std(errors)) if errors else None,
                }
                for region, errors in region_errors.items()
            },
            'pose': {
                'rx_mean': float(np.mean(pose_errors['rx'])) if pose_errors['rx'] else None,
                'ry_mean': float(np.mean(pose_errors['ry'])) if pose_errors['ry'] else None,
                'rz_mean': float(np.mean(pose_errors['rz'])) if pose_errors['rz'] else None,
            },
        },
        'timing': {
            'frames_processed': len(all_errors),
            'frames_failed': failed,
            'total_frames': total_frames,
            'mean_time_ms': float(np.mean(processing_times) * 1000) if processing_times else None,
            'fps': len(processing_times) / sum(processing_times) if processing_times else None,
        }
    }

    return results


def main():
    """Main entry point for single experiment."""
    parser = argparse.ArgumentParser(description='Run single ablation experiment')
    parser.add_argument('--manifest', required=True, help='Path to experiment manifest CSV')
    parser.add_argument('--task-id', type=int, default=None,
                        help='Task ID (default: from SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--cpp-reference', required=True, help='Path to C++ reference CSV')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames to process')
    parser.add_argument('--output-dir', default='bigred200/ablation/results/raw',
                        help='Output directory for results')

    args = parser.parse_args()

    # Get task ID
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    print(f"=" * 60)
    print(f"Ablation Experiment: Task {task_id}")
    print(f"=" * 60)

    # Load configuration
    config = load_config_from_manifest(args.manifest, task_id)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run experiment
    print(f"\nProcessing {args.max_frames} frames from {args.video}...")
    start_time = time.time()

    results = run_experiment(
        config=config,
        video_path=args.video,
        cpp_csv_path=args.cpp_reference,
        max_frames=args.max_frames,
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Print summary
    metrics = results['metrics']
    if metrics['overall']['mean_error'] is not None:
        print(f"\nResults:")
        print(f"  Overall error: {metrics['overall']['mean_error']:.3f} +/- {metrics['overall']['std_error']:.3f} px")
        print(f"  Jaw error: {metrics['regions']['jaw']['mean_error']:.3f} px")
        print(f"  Eyes error: {(metrics['regions']['left_eye']['mean_error'] + metrics['regions']['right_eye']['mean_error']) / 2:.3f} px")
        print(f"  Frames: {results['timing']['frames_processed']}/{results['timing']['total_frames']}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"experiment_{task_id:05d}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
