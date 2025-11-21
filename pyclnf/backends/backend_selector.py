"""
Automatic backend selection for CEN patch experts.

Chooses the fastest available backend based on hardware capabilities:
1. CoreML (Apple Neural Engine) on macOS
2. ONNX + CUDA (NVIDIA GPUs)
3. Pure Python (fallback)
"""
import platform
import sys


def select_best_backend(force_backend: str = None, verbose: bool = True):
    """
    Select the best available CEN backend for current hardware.

    Priority (fastest first):
    1. CoreML (macOS with Apple Silicon or Intel+ANE)
    2. ONNX + CUDA (NVIDIA GPUs)
    3. Pure Python (CPU fallback)

    Args:
        force_backend: Force a specific backend ("coreml", "onnx", "python")
        verbose: Print backend selection info

    Returns:
        backend: Initialized backend instance
    """
    if force_backend:
        backend_name = force_backend.lower()
        if verbose:
            print(f"[CEN Backend] Forcing backend: {backend_name}")

        if backend_name == "coreml":
            try:
                from .coreml_backend import CoreMLCENBackend
                backend = CoreMLCENBackend()
                if verbose:
                    print(f"[CEN Backend] ✓ CoreML loaded")
                return backend
            except Exception as e:
                if verbose:
                    print(f"[CEN Backend] ✗ CoreML failed: {e}")
                raise

        elif backend_name == "onnx":
            try:
                from .onnx_backend import ONNXCENBackend
                backend = ONNXCENBackend()
                if verbose:
                    print(f"[CEN Backend] ✓ ONNX loaded")
                return backend
            except Exception as e:
                if verbose:
                    print(f"[CEN Backend] ✗ ONNX failed: {e}")
                raise

        elif backend_name == "python":
            # Use pure Python implementation (current CENPatchExpert)
            if verbose:
                print(f"[CEN Backend] Using pure Python backend")
            return None  # Signal to use existing implementation

        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    # Auto-select best backend
    system = platform.system()

    # Try CoreML first on macOS
    if system == "Darwin":
        try:
            from .coreml_backend import CoreMLCENBackend
            backend = CoreMLCENBackend()
            if verbose:
                print(f"[CEN Backend] Auto-selected: CoreML (Apple Neural Engine)")
            return backend
        except ImportError:
            if verbose:
                print(f"[CEN Backend] CoreML not available (coremltools not installed)")
        except Exception as e:
            if verbose:
                print(f"[CEN Backend] CoreML init failed: {e}")

    # Try ONNX with CUDA
    try:
        from .onnx_backend import ONNXCENBackend
        backend = ONNXCENBackend()
        if hasattr(backend, 'has_cuda') and backend.has_cuda:
            if verbose:
                print(f"[CEN Backend] Auto-selected: ONNX + CUDA")
            return backend
        elif verbose:
            print(f"[CEN Backend] ONNX available but no CUDA detected")
            print(f"[CEN Backend] Auto-selected: ONNX (CPU)")
        return backend
    except ImportError:
        if verbose:
            print(f"[CEN Backend] ONNX not available (onnxruntime not installed)")
    except Exception as e:
        if verbose:
            print(f"[CEN Backend] ONNX init failed: {e}")

    # Fallback to pure Python
    if verbose:
        print(f"[CEN Backend] Auto-selected: Pure Python (CPU)")
        print(f"[CEN Backend] Install coremltools (macOS) or onnxruntime-gpu for acceleration")

    return None  # Use existing Python implementation


def benchmark_backends(image_patch: "np.ndarray", iterations: int = 100):
    """
    Benchmark all available backends.

    Args:
        image_patch: Sample image patch for testing
        iterations: Number of iterations for timing

    Returns:
        results: Dict of backend_name -> fps
    """
    import time
    import numpy as np

    results = {}

    # Test CoreML
    try:
        from .coreml_backend import CoreMLCENBackend
        backend = CoreMLCENBackend()
        backend.load_models("pyclnf/models")

        start = time.time()
        for _ in range(iterations):
            _ = backend.response(image_patch, landmark_idx=36, scale=0.25)
        elapsed = time.time() - start
        fps = iterations / elapsed
        results['CoreML'] = fps
        backend.cleanup()
    except:
        results['CoreML'] = None

    # Test ONNX
    try:
        from .onnx_backend import ONNXCENBackend
        backend = ONNXCENBackend()
        backend.load_models("pyclnf/models")

        start = time.time()
        for _ in range(iterations):
            _ = backend.response(image_patch, landmark_idx=36, scale=0.25)
        elapsed = time.time() - start
        fps = iterations / elapsed
        backend_name = 'ONNX+CUDA' if backend.has_cuda else 'ONNX+CPU'
        results[backend_name] = fps
        backend.cleanup()
    except:
        results['ONNX'] = None

    # Test Pure Python
    try:
        from ..core.cen_patch_expert import CENPatchExpert, CENPatchExperts
        cen = CENPatchExperts("pyclnf/models")
        expert = cen.patch_experts[0][36]  # Scale 0.25, landmark 36

        start = time.time()
        for _ in range(iterations):
            _ = expert.response(image_patch)
        elapsed = time.time() - start
        fps = iterations / elapsed
        results['Python'] = fps
    except:
        results['Python'] = None

    return results
