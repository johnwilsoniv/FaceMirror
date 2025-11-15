# PyFaceAU → PyMTCNN Migration Plan

## Executive Summary

**Goal:** Make PyMTCNN the primary face detector for PyFaceAU, removing RetinaFace dependency.

**Why:**
- ✅ **Faster:** 34 FPS (macOS) vs ~20 FPS (RetinaFace)
- ✅ **Cross-platform:** CUDA, CoreML, CPU support
- ✅ **Simpler:** One detector package instead of RetinaFace + weights
- ✅ **Better maintained:** PyMTCNN v1.1.0 just published with multi-backend support

## Current Status: PyFaceAU is Already Cross-Platform!

### Component Analysis:

```
Component              | Type        | Cross-Platform | Notes
-----------------------|-------------|----------------|------------------------
Face Detection         | ONNX        | ✅ Yes         | Can swap to PyMTCNN
Landmark Detection     | ONNX        | ✅ Yes         | PFLD model
3D Pose Estimation     | Pure Python | ✅ Yes         | CalcParams
Face Alignment         | Pure Python | ✅ Yes         | OpenFace22 algorithm
HOG Feature Extraction | Pure Python | ✅ Yes         | PyFHOG
Geometric Features     | Pure Python | ✅ Yes         | PDM-based
AU Prediction          | scikit-learn| ✅ Yes         | SVR models
Median Tracking        | Cython*     | ✅ Yes*        | *Has Python fallback
Rotation Update        | Cython*     | ✅ Yes*        | *Has Python fallback
```

**Conclusion:** PyFaceAU is 100% cross-platform. Cython is just for performance optimization!

## Migration Steps

### Phase 1: Make PyMTCNN the Default (Backward Compatible)

**Files to Update:**
1. `pyfaceau/pipeline.py` - Change default detector
2. `pyfaceau/__init__.py` - Update imports
3. `pyfaceau/setup.py` - Add pymtcnn as dependency
4. `pyfaceau/README.md` - Update documentation

**Changes:**

```python
# OLD (pyfaceau/pipeline.py):
from pyfaceau.detectors import ONNXRetinaFaceDetector

def __init__(self, retinaface_model: str, ...):
    self.face_detector = ONNXRetinaFaceDetector(retinaface_model)

# NEW (pyfaceau/pipeline.py):
from pyfaceau.detectors import PyMTCNNDetector, PYMTCNN_AVAILABLE

def __init__(self, face_detector: str = 'pymtcnn', ...):
    if face_detector == 'pymtcnn' and PYMTCNN_AVAILABLE:
        self.face_detector = PyMTCNNDetector(backend='auto')
    elif face_detector == 'retinaface':
        # Legacy support
        self.face_detector = ONNXRetinaFaceDetector(retinaface_model)
    else:
        raise ValueError(...)
```

### Phase 2: Update Installation

**Before:**
```bash
pip install pyfaceau
# User needs to separately get RetinaFace model
```

**After:**
```bash
pip install pyfaceau[cuda]     # NVIDIA GPU
pip install pyfaceau[coreml]   # Apple Silicon
pip install pyfaceau[onnx]     # CPU
```

**setup.py changes:**
```python
install_requires=[
    "numpy>=1.20.0",
    "opencv-python>=4.5.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "onnxruntime>=1.10.0",
    "pyfhog>=0.1.0",
    "pymtcnn>=1.1.0",  # NEW: Core dependency
],
extras_require={
    'cuda': ['pymtcnn[onnx-gpu]>=1.1.0'],
    'coreml': ['pymtcnn[coreml]>=1.1.0'],
    'onnx': ['pymtcnn[onnx]>=1.1.0'],
    'all': ['pymtcnn[all]>=1.1.0', 'coremltools>=7.0'],
    # Keep legacy support
    'retinaface': ['onnxruntime>=1.10.0'],
}
```

### Phase 3: Remove RetinaFace Models (Future)

Once migration is complete:
- Remove RetinaFace ONNX model from weights
- Simplify download_weights.py
- Update all examples

## Performance Comparison

### Current (RetinaFace):
```
Platform            | Face Detection | Total Pipeline
--------------------|----------------|---------------
Apple M3 (CoreML)   | ~20 FPS        | ~4.6 FPS
NVIDIA GPU          | ~20 FPS        | ~5 FPS
CPU                 | ~5 FPS         | ~2 FPS
```

### With PyMTCNN:
```
Platform            | Face Detection | Total Pipeline (estimated)
--------------------|----------------|---------------------------
Apple M3 (CoreML)   | 34 FPS (1.7x)  | ~7 FPS (1.5x)
NVIDIA GPU (CUDA)   | 50+ FPS (2.5x) | ~10 FPS (2x)
CPU                 | 5-10 FPS       | ~2-3 FPS
```

**Why pipeline doesn't scale linearly:**
- Face detection is only one step
- AU prediction, HOG extraction, alignment also take time
- But 1.5-2x overall speedup is still significant!

## Compilation Question: Do We Still Need It?

**Answer: NO for functionality, YES for performance**

### What Compilation Provides:

**Cython Extensions:**
1. **Histogram Median Tracker** - 260x faster
   - Python fallback exists but is slow
   - Used for temporal smoothing across frames

2. **Rotation Update** - 99.9% accuracy mode
   - Python fallback exists
   - Used in CalcParams for pose estimation

### GitHub Actions Wheel Building:

**Purpose:**
- Pre-compile Cython extensions for users
- Users get performance boost without needing compiler
- Convenience, not requirement

**We should keep it because:**
- ✅ Temporal smoothing is important for video processing
- ✅ 260x speedup is too good to lose
- ✅ Most users don't have compilers installed
- ✅ Wheels work seamlessly - user doesn't even know they're compiled

## Recommended Approach

### Option A: PyMTCNN as Default (Recommended)

**Pros:**
- Best performance out of the box
- Cross-platform by default
- Simpler dependency tree
- Users can still opt-in to RetinaFace if needed

**Cons:**
- Requires pymtcnn dependency
- Small breaking change (but easily fixable)

### Option B: Keep Both, Auto-Select

**Pros:**
- No breaking changes
- Maximum flexibility

**Cons:**
- More complex code
- Two face detectors to maintain
- Confusing for users

## Migration Timeline

### Immediate (v1.1.0):
1. ✅ Add PyMTCNNDetector wrapper
2. ✅ Add integration example
3. ✅ Add PYMTCNN_INTEGRATION.md docs
4. ⏳ Update pipeline to support both detectors
5. ⏳ Add pymtcnn to optional dependencies

### Next Release (v1.2.0):
1. Make PyMTCNN the default detector
2. Add deprecation warning for RetinaFace
3. Update all examples to use PyMTCNN
4. Update README with new installation instructions

### Future (v2.0.0):
1. Remove RetinaFace dependency entirely
2. Remove RetinaFace model from weights
3. Simplify codebase

## User Impact

### Breaking Changes:
**v1.1.0:** None - PyMTCNN is optional
**v1.2.0:** Default changed, but RetinaFace still works
**v2.0.0:** RetinaFace removed (with migration guide)

### Migration Path for Users:

**Current code:**
```python
from pyfaceau import FullPythonAUPipeline

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface.onnx',
    ...
)
```

**v1.2.0+ (recommended):**
```python
from pyfaceau import FullPythonAUPipeline

# Auto-selects PyMTCNN if available, falls back to RetinaFace
pipeline = FullPythonAUPipeline(
    face_detector='pymtcnn',  # or 'auto', 'retinaface'
    ...
)
```

**v2.0.0+:**
```python
from pyfaceau import FullPythonAUPipeline

# RetinaFace removed - PyMTCNN only
pipeline = FullPythonAUPipeline(
    # No face detector parameter needed - PyMTCNN is default
    ...
)
```

## Next Steps

1. **Update pipeline.py** to support face_detector parameter
2. **Update setup.py** to add pymtcnn dependency
3. **Update README.md** with PyMTCNN as recommended option
4. **Test cross-platform** (Linux/Windows/macOS)
5. **Release v1.2.0** with PyMTCNN as default

## Questions to Answer

1. ✅ Is PyFaceAU cross-platform? **YES**
2. ✅ Do we need compilation? **NO for functionality, YES for performance**
3. ✅ Can we use PyMTCNN as primary? **YES, recommended!**
4. ⏳ Should we remove RetinaFace? **Eventually (v2.0.0)**

---

**Recommendation: Proceed with PyMTCNN as the default detector starting in v1.2.0**
