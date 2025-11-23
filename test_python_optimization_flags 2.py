#!/usr/bin/env python3
"""
Test the effect of Python optimization flags on performance.
-O : Remove assert statements and __debug__ code
-OO : Also remove docstrings
"""

import time
import sys
import numpy as np
from pathlib import Path

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def benchmark_simple_operations():
    """
    Test basic operations that might benefit from -O flag.
    """
    print(f"Python optimization level: {sys.flags.optimize}")
    print(f"Debug mode: {__debug__}")
    print(f"Assertions enabled: {not sys.flags.optimize}")
    print()

    # Test with assertions and debug code
    n_iterations = 1000000

    # Test 1: Assertions
    print("Test 1: Assertion overhead")
    data = np.random.randn(100)

    start = time.perf_counter()
    for i in range(n_iterations):
        assert len(data) > 0, "Data must not be empty"
        assert data.dtype == np.float64, "Data must be float64"
        result = np.sum(data)
    assertion_time = (time.perf_counter() - start) * 1000

    print(f"  Time with assertions: {assertion_time:.1f}ms")

    # Test 2: Debug blocks
    print("\nTest 2: Debug block overhead")
    start = time.perf_counter()
    for i in range(n_iterations):
        if __debug__:
            # This code is removed with -O flag
            debug_info = f"Iteration {i}"
            debug_check = len(data) > 0
        result = np.sum(data)
    debug_time = (time.perf_counter() - start) * 1000

    print(f"  Time with debug blocks: {debug_time:.1f}ms")

    # Test 3: Pipeline initialization
    print("\nTest 3: Pipeline initialization")
    start = time.perf_counter()

    # Suppress output
    import io
    import warnings
    warnings.filterwarnings('ignore')
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        from pymtcnn import MTCNN
        from pyclnf import CLNF

        # Initialize components
        detector = MTCNN()
        clnf = CLNF(
            model_dir="pyclnf/models",
            max_iterations=5,
            convergence_threshold=0.5,
            debug_mode=False
        )
        init_time = (time.perf_counter() - start) * 1000
    finally:
        sys.stdout = old_stdout

    print(f"  Initialization time: {init_time:.1f}ms")

    return {
        'assertions': assertion_time,
        'debug': debug_time,
        'init': init_time
    }


def main():
    print("=" * 60)
    print("PYTHON OPTIMIZATION FLAG TEST")
    print("=" * 60)
    print("\nRun this script with different flags:")
    print("  Normal: python3 test_python_optimization_flags.py")
    print("  -O:     python3 -O test_python_optimization_flags.py")
    print("  -OO:    python3 -OO test_python_optimization_flags.py")
    print()

    results = benchmark_simple_operations()

    print("\n" + "=" * 60)
    print("EXPECTED IMPROVEMENTS WITH -O FLAG")
    print("=" * 60)
    print("- Assertions removed: ~5-10% speedup in tight loops")
    print("- Debug blocks removed: ~2-5% speedup")
    print("- Overall pipeline: ~3-7% speedup expected")
    print("\nTo run optimized pipeline:")
    print("  python3 -O optimized_au_pipeline.py")


if __name__ == "__main__":
    main()