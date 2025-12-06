#!/usr/bin/env python3
"""
Quick test script for HPC AU Pipeline on Big Red 200.

This tests the pipeline with a small number of frames to verify
all components are working before running a full batch job.

Usage:
    python test_hpc_pipeline.py --video "S Data/Normal Cohort/IMG_0942.MOV" --max-frames 50
"""

import os
import sys
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))


def test_numa_detection():
    """Test NUMA topology detection."""
    print("=" * 60)
    print("TEST 1: NUMA Topology Detection")
    print("=" * 60)

    from numa_worker_pool import detect_numa_topology

    topology = detect_numa_topology()
    print(f"  NUMA available: {topology['is_numa']}")
    print(f"  NUMA nodes: {topology['num_nodes']}")
    print(f"  Total CPUs: {topology['num_cpus']}")

    if topology['is_numa']:
        for i, cpus in enumerate(topology['cpus_per_node']):
            mem = topology['memory_per_node'][i] if i < len(topology['memory_per_node']) else 'N/A'
            print(f"  Node {i}: {len(cpus)} CPUs, {mem} GB")

    print("  [PASSED]")
    return topology


def test_shared_memory_init():
    """Test shared memory model initialization."""
    print("\n" + "=" * 60)
    print("TEST 2: Shared Memory Initialization")
    print("=" * 60)

    from shared_memory_init import initialize_shared_models, cleanup_shared_models

    start = time.time()
    try:
        config = initialize_shared_models()
        init_time = time.time() - start
        print(f"  Initialization time: {init_time:.2f}s")
        print(f"  CLNF shared dir: {config['clnf_shm_dir']}")
        print(f"  AU shared dir: {config['au_shm_dir']}")
        print("  [PASSED]")
        return config
    except Exception as e:
        print(f"  [FAILED] {e}")
        return None


def test_worker_init(shm_config):
    """Test worker initialization."""
    print("\n" + "=" * 60)
    print("TEST 3: Worker Initialization")
    print("=" * 60)

    from hpc_au_pipeline import _init_worker_hpc

    config = shm_config.copy() if shm_config else {}
    config['project_root'] = str(project_root)
    config['clnf_model_dir'] = str(project_root / "pyclnf/pyclnf/models")
    config['pdm_file'] = str(project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")
    config['triangulation_file'] = str(project_root / "pyfaceau/weights/tris_68_full.txt")
    config['convergence_profile'] = 'optimized'
    config['use_shared_memory'] = shm_config is not None

    start = time.time()
    try:
        _init_worker_hpc(config)
        init_time = time.time() - start
        print(f"  Worker init time: {init_time:.2f}s")
        print("  [PASSED]")
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_frame(video_path):
    """Test processing a single frame."""
    print("\n" + "=" * 60)
    print("TEST 4: Single Frame Processing")
    print("=" * 60)

    import cv2
    import numpy as np
    from hpc_au_pipeline import _process_frame_hpc, _worker_state

    # Load a frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ret:
        print(f"  [FAILED] Could not read frame from {video_path}")
        return False

    print(f"  Frame shape: {frame.shape}")

    # First, test the detector directly to verify it's working
    print("  Testing detector directly...")
    if _worker_state is None:
        print("  [FAILED] _worker_state is None (worker not initialized?)")
        return False

    detector = _worker_state['detector']
    bboxes, _ = detector.detect(frame)
    if bboxes is None or len(bboxes) == 0:
        print("  [DEBUG] Direct detection on raw frame: NO FACES")
    else:
        print(f"  [DEBUG] Direct detection on raw frame: {len(bboxes)} faces")
        bbox = bboxes[0]
        print(f"  [DEBUG] First bbox: {bbox[:4]}")

    # Encode frame
    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    frame_data = encoded.tobytes()

    # Test detection on decoded frame
    decoded = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    print(f"  Decoded shape: {decoded.shape}")
    bboxes2, _ = detector.detect(decoded)
    if bboxes2 is None or len(bboxes2) == 0:
        print("  [DEBUG] Direct detection on decoded frame: NO FACES")
    else:
        print(f"  [DEBUG] Direct detection on decoded frame: {len(bboxes2)} faces")
        bbox = bboxes2[0]
        print(f"  [DEBUG] First bbox: {bbox[:4]}")

    start = time.time()
    try:
        result = _process_frame_hpc((0, frame_data, fps, False))
        proc_time = time.time() - start

        if result is None:
            print(f"  [FAILED] Frame processing returned None (no face detected?)")
            return False

        print(f"  Processing time: {proc_time*1000:.1f}ms")
        print(f"  HOG features: {result['hog_features'].shape}")
        print(f"  Geom features: {result['geom_features'].shape}")
        print(f"  CLNF converged: {result['clnf_converged']}")
        print(f"  CLNF iterations: {result['clnf_iterations']}")
        print("  [PASSED]")
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline(video_path, max_frames=50, n_workers=4):
    """Test the full pipeline."""
    print("\n" + "=" * 60)
    print(f"TEST 5: Full Pipeline ({max_frames} frames, {n_workers} workers)")
    print("=" * 60)

    from hpc_au_pipeline import HPCAUPipeline, HPCConfig

    config = HPCConfig(
        n_workers=n_workers,
        convergence_profile='optimized',
        use_shared_memory=True,
        use_numa=True,
        verbose=True
    )

    try:
        pipeline = HPCAUPipeline(config)

        start = time.time()
        df = pipeline.process_video(str(video_path), max_frames=max_frames)
        elapsed = time.time() - start

        success_count = df['success'].sum()
        fps = success_count / elapsed

        print(f"\n  Results:")
        print(f"    Frames processed: {success_count}/{len(df)}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    FPS: {fps:.1f}")

        # Show AU statistics
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        if au_cols:
            print(f"\n  AU Statistics:")
            for au_col in sorted(au_cols)[:5]:  # Show first 5
                mean_val = df[df['success']][au_col].mean()
                print(f"    {au_col}: mean={mean_val:.3f}")

        print("\n  [PASSED]")
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test HPC AU Pipeline")
    parser.add_argument('--video', default="S Data/Normal Cohort/IMG_0942.MOV",
                        help='Test video path')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Max frames for pipeline test')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers for pipeline test')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Skip full pipeline test')

    args = parser.parse_args()

    video_path = project_root / args.video
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        print("Available videos:")
        for p in (project_root / "S Data").rglob("*.MOV"):
            print(f"  {p.relative_to(project_root)}")
        sys.exit(1)

    print("=" * 60)
    print("HPC AU Pipeline - Test Suite")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Max frames: {args.max_frames}")
    print(f"Workers: {args.workers}")
    print()

    results = []

    # Test 1: NUMA detection
    topology = test_numa_detection()
    results.append(("NUMA Detection", topology is not None))

    # Test 2: Shared memory init
    shm_config = test_shared_memory_init()
    results.append(("Shared Memory Init", shm_config is not None))

    # Test 3: Worker init
    worker_ok = test_worker_init(shm_config)
    results.append(("Worker Init", worker_ok))

    # Test 4: Single frame
    if worker_ok:
        frame_ok = test_single_frame(video_path)
        results.append(("Single Frame", frame_ok))
    else:
        results.append(("Single Frame", False))

    # Test 5: Full pipeline
    if not args.skip_pipeline:
        pipeline_ok = test_pipeline(video_path, args.max_frames, args.workers)
        results.append(("Full Pipeline", pipeline_ok))

    # Cleanup shared memory
    if shm_config:
        from shared_memory_init import cleanup_shared_models
        cleanup_shared_models(shm_config)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Ready for production.")
        return 0
    else:
        print("Some tests failed. Check output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
