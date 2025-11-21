"""
Hardware-accelerated backends for CEN patch experts.

Provides CoreML (Apple Neural Engine) and ONNX (CPU/CUDA) backends
for fast response map computation.
"""

__all__ = []

# Import available backends
try:
    from .coreml_backend import CoreMLCENBackend
    __all__.append("CoreMLCENBackend")
except ImportError:
    CoreMLCENBackend = None

try:
    from .onnx_backend import ONNXCENBackend
    __all__.append("ONNXCENBackend")
except ImportError:
    ONNXCENBackend = None

from .backend_selector import select_best_backend
__all__.append("select_best_backend")
