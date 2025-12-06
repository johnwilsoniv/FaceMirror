"""
PyFaceAU Pipeline Validation - HPC Version
Optimized for Big Red 200 with multiprocessing and shared memory.

This script processes video frames in parallel using multiple CPU cores.

SHARED MEMORY OPTIMIZATION:
- Models are loaded once and shared across all workers via memory mapping
- Eliminates 424MB × N_workers memory duplication
- Removes staggered initialization (no longer needed)
- Enables 32-48+ effective workers instead of 16

NUMA OPTIMIZATION (Phase 3):
- Workers are pinned to specific NUMA nodes for memory locality
- Reduces cross-node memory access latency on AMD EPYC
- Enables efficient use of 64-128 cores per node

CONVERGENCE OPTIMIZATION (Phase 2):
- Uses 'optimized' convergence profile for faster processing
- Early window exit when already converged
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import threading
import time

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))

# Use workers based on available CPUs
# With shared memory + NUMA, we can use more workers effectively
N_WORKERS = min(128, int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())))
print(f"Using {N_WORKERS} worker processes")

# Enable shared memory by default (set to False for legacy behavior)
USE_SHARED_MEMORY = os.environ.get('USE_SHARED_MEMORY', '1') == '1'

# Enable NUMA-aware worker placement (Phase 3)
USE_NUMA = os.environ.get('USE_NUMA', '1') == '1'

# Convergence profile for CLNF (Phase 2)
# Options: 'accurate', 'optimized', 'fast', 'video'
CONVERGENCE_PROFILE = os.environ.get('CONVERGENCE_PROFILE', 'optimized')

# Shared memory configuration (set by main process)
_shm_config = None


def init_worker_shared(config):
    """Initialize worker process with shared memory models.

    This is much faster than the old init_worker() because models are
    memory-mapped instead of loaded from disk.

    Args:
        config: Configuration dict with shared memory paths and settings
    """
    global _shm_config
    _shm_config = config

    # Limit threads per worker (critical for HPC to avoid thread affinity conflicts)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['ORT_NUM_THREADS'] = '1'  # ONNX Runtime - avoids pthread_setaffinity errors

    global detector, clnf, calc_params, pdm_parser

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser

    detector = MTCNN()

    # Phase 2: Use convergence profile for faster processing
    convergence_profile = config.get('convergence_profile', 'optimized')

    clnf = CLNF(
        model_dir=config['clnf_model_dir'],
        use_shared_memory=True,
        shared_memory_dir=config['clnf_shm_dir'],
        convergence_profile=convergence_profile,  # Phase 2: HPC-optimized convergence
        early_window_exit=True,  # Phase 2: Skip windows when converged
        early_exit_threshold=0.3  # Phase 2: 0.3px threshold
    )
    pdm_parser = PDMParser(str(project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"))
    calc_params = CalcParams(pdm_parser)


# Legacy initialization (kept for backwards compatibility)
_init_lock = None


def init_worker(lock):
    """Initialize worker process with pipeline components (LEGACY).

    Uses a lock with staggered release to prevent all workers from
    loading models simultaneously (which causes I/O contention).

    NOTE: Use init_worker_shared() instead for better performance.
    """
    global _init_lock
    _init_lock = lock

    lock.acquire()
    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        global detector, clnf, calc_params, pdm_parser

        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfaceau.alignment.calc_params import CalcParams
        from pyfaceau.features.pdm import PDMParser

        detector = MTCNN()
        clnf = CLNF(model_dir="pyclnf/pyclnf/models")
        pdm_parser = PDMParser("pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")
        calc_params = CalcParams(pdm_parser)
    finally:
        threading.Timer(0.1, lock.release).start()


def process_frame_data(args):
    """Process a single frame (worker function)."""
    frame_idx, frame_data, cpp_lm, cpp_pose_frame = args

    # Decode frame from bytes
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

    cpp_rx, cpp_ry, cpp_rz = cpp_pose_frame

    # Derive C++ bbox from landmarks
    x_min, y_min = cpp_lm.min(axis=0)
    x_max, y_max = cpp_lm.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    cpp_bbox = np.array([x_min - w*0.1, y_min - h*0.1, x_max + w*0.1, y_max + h*0.1])

    # Python pipeline
    try:
        bboxes, _ = detector.detect(frame)
        if bboxes is None or len(bboxes) == 0:
            return None

        mtcnn_bbox = bboxes[0][:4]

        py_landmarks, info = clnf.fit(frame, mtcnn_bbox)
        py_landmarks = py_landmarks.astype(np.float32)

        global_params, local_params = calc_params.calc_params(py_landmarks)
        py_rx, py_ry, py_rz = global_params[1:4]

    except Exception:
        return None

    # Compute metrics
    # IoU
    x1 = max(mtcnn_bbox[0], cpp_bbox[0])
    y1 = max(mtcnn_bbox[1], cpp_bbox[1])
    x2 = min(mtcnn_bbox[2], cpp_bbox[2])
    y2 = min(mtcnn_bbox[3], cpp_bbox[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (mtcnn_bbox[2] - mtcnn_bbox[0]) * (mtcnn_bbox[3] - mtcnn_bbox[1])
    box2_area = (cpp_bbox[2] - cpp_bbox[0]) * (cpp_bbox[3] - cpp_bbox[1])
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)

    # Landmark error
    lm_error = np.mean(np.linalg.norm(py_landmarks - cpp_lm, axis=1))

    # Pose errors (degrees)
    pose_rx_err = abs(np.degrees(py_rx - cpp_rx))
    pose_ry_err = abs(np.degrees(py_ry - cpp_ry))
    pose_rz_err = abs(np.degrees(py_rz - cpp_rz))

    return {
        'frame': frame_idx,
        'bbox_iou': iou,
        'landmark_error_mean': lm_error,
        'pose_rx_error': pose_rx_err,
        'pose_ry_error': pose_ry_err,
        'pose_rz_error': pose_rz_err,
    }


def main():
    start_time = time.time()

    print("=" * 70)
    print("PyFaceAU Pipeline Validation - HPC Version")
    print("=" * 70)

    # Load C++ reference data
    cpp_csv = pd.read_csv("validation_output_0942/IMG_0942.csv")
    print(f"C++ reference: {len(cpp_csv)} frames")

    # Extract C++ landmarks
    cpp_landmarks = np.zeros((len(cpp_csv), 68, 2))
    for i in range(68):
        cpp_landmarks[:, i, 0] = cpp_csv[f'x_{i}'].values
        cpp_landmarks[:, i, 1] = cpp_csv[f'y_{i}'].values

    # Extract C++ pose
    cpp_pose = cpp_csv[['pose_Rx', 'pose_Ry', 'pose_Rz']].values

    # Load video frames into memory
    video_path = "S Data/Normal Cohort/IMG_0942.MOV"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nLoading {total_frames} frames into memory...")
    frames_data = []
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        # Encode frame as bytes for multiprocessing
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames_data.append(encoded.tobytes())
    cap.release()

    print(f"Loaded {len(frames_data)} frames")

    # Prepare arguments for parallel processing
    # Pass only per-frame data, not full arrays (avoids serialization overhead)
    args_list = [
        (i, frames_data[i], cpp_landmarks[i], cpp_pose[i])
        for i in range(len(frames_data))
    ]

    # Choose initialization method based on USE_SHARED_MEMORY setting
    if USE_SHARED_MEMORY:
        # Initialize shared memory models ONCE in main process
        from shared_memory_init import initialize_shared_models, cleanup_shared_models

        print(f"\nInitializing shared memory models...")
        shm_config = initialize_shared_models()

        # Add convergence profile to config (Phase 2)
        shm_config['convergence_profile'] = CONVERGENCE_PROFILE
        print(f"Using convergence profile: {CONVERGENCE_PROFILE}")

        # Choose worker pool type based on USE_NUMA setting (Phase 3)
        if USE_NUMA:
            from numa_worker_pool import NUMAWorkerPool, detect_numa_topology

            # Detect and report NUMA topology
            topology = detect_numa_topology()
            print(f"\nNUMA Topology:")
            print(f"  NUMA nodes: {topology['num_nodes']}")
            print(f"  Total CPUs: {topology['num_cpus']}")
            if topology['is_numa']:
                for i, cpus in enumerate(topology['cpus_per_node']):
                    print(f"  Node {i}: {len(cpus)} CPUs")

            # Process with NUMA-aware pool
            print(f"\nProcessing with {N_WORKERS} workers (shared memory + NUMA)...")
            try:
                with NUMAWorkerPool(
                    n_workers=N_WORKERS,
                    initializer=init_worker_shared,
                    initargs=(shm_config,)
                ) as pool:
                    # Report NUMA distribution
                    stats = pool.get_numa_stats()
                    print(f"  Workers per NUMA node: {stats['workers_per_node']}")

                    results = list(tqdm(
                        pool.imap(process_frame_data, args_list),
                        total=len(args_list),
                        desc="Processing"
                    ))
            finally:
                # Clean up shared memory
                cleanup_shared_models(shm_config)
        else:
            # Standard pool without NUMA awareness
            print(f"\nProcessing with {N_WORKERS} workers (shared memory, no NUMA)...")
            try:
                with Pool(N_WORKERS, initializer=init_worker_shared, initargs=(shm_config,)) as pool:
                    results = list(tqdm(
                        pool.imap(process_frame_data, args_list),
                        total=len(args_list),
                        desc="Processing"
                    ))
            finally:
                # Clean up shared memory
                cleanup_shared_models(shm_config)
    else:
        # Legacy mode with staggered initialization
        manager = Manager()
        init_lock = manager.Lock()

        print(f"\nProcessing with {N_WORKERS} workers (staggered init, legacy mode)...")
        with Pool(N_WORKERS, initializer=init_worker, initargs=(init_lock,)) as pool:
            results = list(tqdm(
                pool.imap(process_frame_data, args_list),
                total=len(args_list),
                desc="Processing"
            ))

    # Filter out failed frames
    results = [r for r in results if r is not None]
    failed = total_frames - len(results)

    # Compute statistics
    bbox_ious = [r['bbox_iou'] for r in results]
    landmark_errors = [r['landmark_error_mean'] for r in results]
    pose_rx = [r['pose_rx_error'] for r in results]
    pose_ry = [r['pose_ry_error'] for r in results]
    pose_rz = [r['pose_rz_error'] for r in results]

    elapsed = time.time() - start_time
    fps = len(results) / elapsed

    # Print results
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Shared Memory: {USE_SHARED_MEMORY}")
    print(f"  NUMA-aware: {USE_NUMA}")
    print(f"  Convergence Profile: {CONVERGENCE_PROFILE}")

    print(f"\nFrames: {len(results)}/{total_frames} successful ({failed} failed)")
    print(f"Processing time: {elapsed:.1f}s ({fps:.1f} fps)")

    print("\n--- BOUNDING BOX ---")
    mean_iou = np.mean(bbox_ious)
    target_iou = 0.98
    status_iou = "PASS" if mean_iou >= target_iou else "FAIL"
    print(f"Mean IOU:    {mean_iou:.4f} +/- {np.std(bbox_ious):.4f}")
    print(f"Min IOU:     {np.min(bbox_ious):.4f}")
    print(f"Target:      {target_iou:.2f}")
    print(f"Status:      {status_iou}")

    print("\n--- LANDMARKS ---")
    mean_lm = np.mean(landmark_errors)
    target_lm = 2.0
    status_lm = "PASS" if mean_lm < target_lm else "FAIL"
    print(f"Mean Error:  {mean_lm:.3f} px +/- {np.std(landmark_errors):.3f}")
    print(f"Max Error:   {np.max(landmark_errors):.3f} px")
    print(f"Target:      < {target_lm} px")
    print(f"Status:      {status_lm}")

    print("\n--- POSE ESTIMATION ---")
    print(f"  RX Error:   {np.mean(pose_rx):.2f} +/- {np.std(pose_rx):.2f} degrees")
    print(f"  RY Error:   {np.mean(pose_ry):.2f} +/- {np.std(pose_ry):.2f} degrees")
    print(f"  RZ Error:   {np.mean(pose_rz):.2f} +/- {np.std(pose_rz):.2f} degrees")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("validation_results_hpc.csv", index=False)
    print(f"\nDetailed results saved to: validation_results_hpc.csv")

    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    if mean_iou >= target_iou and mean_lm < target_lm:
        print("STATUS: ALL CHECKS PASSED")
    else:
        print("STATUS: ISSUES FOUND")
        if mean_iou < target_iou:
            print(f"  - BBox IOU ({mean_iou:.2%}) below target ({target_iou:.0%})")
        if mean_lm >= target_lm:
            print(f"  - Landmark error ({mean_lm:.2f}px) exceeds target (<{target_lm}px)")


if __name__ == '__main__':
    main()
