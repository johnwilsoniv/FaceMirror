#!/usr/bin/env python3
"""
Shared Memory Initialization for Big Red 200 HPC

This module initializes shared memory model files ONCE before spawning workers,
allowing all workers to share model weights via memory mapping.

Memory Savings:
- CEN Patch Experts: 424MB × N_workers → 424MB × 1 (93% reduction)
- AU SVR Models: 10MB × N_workers → 10MB × 1

Usage:
    # In main process before creating worker pool:
    from shared_memory_init import initialize_shared_models, cleanup_shared_models

    # Initialize (run once)
    shm_config = initialize_shared_models()

    # Create worker pool with shared memory config
    with Pool(N_WORKERS, initializer=init_worker_shared,
              initargs=(shm_config,)) as pool:
        ...

    # Cleanup when done
    cleanup_shared_models(shm_config)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Default shared memory locations (use /dev/shm on Linux for best performance)
DEFAULT_CLNF_SHM_DIR = "/dev/shm/pyclnf_models"
DEFAULT_AU_SHM_DIR = "/dev/shm/pyfaceau_models"

# For non-Linux systems, use temp directory
if not Path("/dev/shm").exists():
    import tempfile
    DEFAULT_CLNF_SHM_DIR = str(Path(tempfile.gettempdir()) / "pyclnf_models")
    DEFAULT_AU_SHM_DIR = str(Path(tempfile.gettempdir()) / "pyfaceau_models")


def initialize_shared_models(
    clnf_model_dir: str = "pyclnf/pyclnf/models",
    au_model_dir: str = "pyfaceau/weights/AU_predictors",
    clnf_shm_dir: str = DEFAULT_CLNF_SHM_DIR,
    au_shm_dir: str = DEFAULT_AU_SHM_DIR,
    force: bool = False
) -> Dict[str, str]:
    """
    Initialize shared memory model files.

    Call this ONCE in the main process before creating the worker pool.

    Args:
        clnf_model_dir: Path to CLNF model directory
        au_model_dir: Path to AU predictors directory
        clnf_shm_dir: Shared memory directory for CLNF models
        au_shm_dir: Shared memory directory for AU models
        force: If True, recreate even if already exists

    Returns:
        Configuration dict to pass to workers
    """
    # Add paths if needed
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "pyclnf") not in sys.path:
        sys.path.insert(0, str(project_root / "pyclnf"))
    if str(project_root / "pyfaceau") not in sys.path:
        sys.path.insert(0, str(project_root / "pyfaceau"))

    # Resolve relative paths to absolute
    clnf_model_dir = str((project_root / clnf_model_dir).resolve())
    au_model_dir = str((project_root / au_model_dir).resolve())

    print("=" * 60)
    print("INITIALIZING SHARED MEMORY MODELS")
    print("=" * 60)
    print(f"  CLNF models: {clnf_model_dir}")
    print(f"  AU models: {au_model_dir}")
    print(f"  CLNF shared memory: {clnf_shm_dir}")
    print(f"  AU shared memory: {au_shm_dir}")

    # Initialize CLNF models
    from pyclnf.core.shared_model_loader import SharedModelManager, SharedAUModelManager

    clnf_created = SharedModelManager.initialize(clnf_shm_dir, clnf_model_dir, force=force)

    # Initialize AU models
    au_created = SharedAUModelManager.initialize(au_shm_dir, au_model_dir, force=force)

    print("=" * 60)
    print(f"  CLNF models: {'created' if clnf_created else 'already exists'}")
    print(f"  AU models: {'created' if au_created else 'already exists'}")
    print("=" * 60)

    return {
        'clnf_model_dir': clnf_model_dir,
        'au_model_dir': au_model_dir,
        'clnf_shm_dir': clnf_shm_dir,
        'au_shm_dir': au_shm_dir,
        'use_shared_memory': True
    }


def cleanup_shared_models(config: Dict[str, str]):
    """
    Clean up shared memory model files.

    Call this after all workers have finished.

    Args:
        config: Configuration dict from initialize_shared_models()
    """
    from pyclnf.core.shared_model_loader import SharedModelManager, SharedAUModelManager

    print("Cleaning up shared memory models...")
    SharedModelManager.cleanup(config['clnf_shm_dir'])
    SharedAUModelManager.cleanup(config['au_shm_dir'])
    print("Shared memory cleanup complete")


def init_worker_shared(config: Dict[str, str]):
    """
    Initialize worker process with shared memory models.

    This is much faster than init_worker() because models are memory-mapped
    instead of loaded from disk.

    Args:
        config: Configuration dict from initialize_shared_models()
    """
    # Limit threads per worker to avoid contention
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    global detector, clnf, calc_params, pdm_parser, au_models

    # Add paths if needed
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "pyclnf") not in sys.path:
        sys.path.insert(0, str(project_root / "pyclnf"))
    if str(project_root / "pyfaceau") not in sys.path:
        sys.path.insert(0, str(project_root / "pyfaceau"))
    if str(project_root / "pymtcnn") not in sys.path:
        sys.path.insert(0, str(project_root / "pymtcnn"))

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser
    from pyclnf.core.shared_model_loader import SharedAUModels

    # Load detector (small, no sharing needed)
    detector = MTCNN()

    # Load CLNF with shared memory
    clnf = CLNF(
        model_dir=config['clnf_model_dir'],
        use_shared_memory=True,
        shared_memory_dir=config['clnf_shm_dir']
    )

    # Load AU models from shared memory
    au_models = SharedAUModels(config['au_shm_dir'])

    # Load PDM parser (small, no sharing needed)
    pdm_file = str(project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt")
    pdm_parser = PDMParser(pdm_file)
    calc_params = CalcParams(pdm_parser)


# For testing
if __name__ == "__main__":
    import time

    print("Testing shared memory initialization...")
    start = time.time()

    config = initialize_shared_models()
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")

    # Test worker initialization
    print("\nTesting worker initialization...")
    worker_start = time.time()
    init_worker_shared(config)
    worker_time = time.time() - worker_start
    print(f"Worker init time: {worker_time:.2f}s")

    print(f"\nLoaded components:")
    print(f"  detector: {type(detector)}")
    print(f"  clnf: {type(clnf)}")
    print(f"  au_models: {len(au_models)} models")

    # Cleanup
    cleanup_shared_models(config)
