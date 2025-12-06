# PyOpenFace: Python Implementation of OpenFace 2.2

A pure Python implementation of OpenFace 2.2 for facial Action Unit (AU) extraction.

**Last Updated:** December 5, 2025

---

## Overview

This codebase provides a Python-based pipeline for facial AU extraction that replicates the functionality of C++ OpenFace 2.2. The pipeline consists of four main packages that work together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         pyfaceau (API Layer)                        │
│                    FullPythonAUPipeline                             │
│                         pipeline.py                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   pymtcnn     │  │     pyclnf      │  │     pyfhog      │
│ Face Detection│  │ 68-pt Landmarks │  │ HOG Extraction  │
│   (CoreML)    │  │  (CLNF/CECLM)   │  │ (dlib wrapper)  │
└───────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Package Structure

### 1. pyfaceau (Main API) - `pyfaceau/`

The top-level API that orchestrates the entire AU extraction pipeline.

```
pyfaceau/
├── pyfaceau/
│   ├── __init__.py              # Exports FullPythonAUPipeline
│   ├── pipeline.py              # Main pipeline (CANONICAL ENTRY POINT)
│   ├── config.py                # Locked configuration constants
│   ├── alignment/
│   │   └── face_aligner.py      # Kabsch-based face alignment
│   ├── detectors/
│   │   └── pymtcnn_detector.py  # PyMTCNN wrapper
│   ├── features/
│   │   ├── hog_extractor.py     # HOG feature extraction
│   │   └── geometry.py          # Geometric features from PDM
│   ├── prediction/
│   │   ├── au_predictor.py      # AU prediction (SVR/SVM)
│   │   └── running_median.py    # Cython-optimized temporal smoothing
│   └── nn/
│       └── calc_params.py       # Numba-accelerated PDM fitting
└── weights/                      # Model weights
    ├── AU_predictors/           # SVR/SVM models for 17 AUs
    ├── In-the-wild_aligned_PDM_68.txt  # 34-mode PDM for alignment
    ├── tris_68_full.txt         # Triangulation for face mask
    └── svr_patches_*.txt        # Patch expert files
```

**Key File:** `pyfaceau/pyfaceau/pipeline.py` - The single entry point for all AU processing.

### 2. pyclnf (Landmark Detection) - `pyclnf/`

Python implementation of Constrained Local Neural Fields (CLNF) for 68-point facial landmark detection.

```
pyclnf/
├── pyclnf/
│   ├── __init__.py
│   ├── clnf.py                  # Main CLNF detector class
│   ├── core/
│   │   ├── optimizer.py         # CLNF optimization (mean-shift, Jacobian)
│   │   ├── pdm.py               # Point Distribution Model
│   │   ├── cen_patch_expert.py  # Neural network patch experts
│   │   └── eye_refinement.py    # Eye landmark refinement
│   └── models/
│       └── exported_pdm/        # 30-mode PDM (exported from C++)
│           ├── mean_shape.npy
│           ├── eigen_values.npy
│           └── princ_comp.npy
└── examples/
```

**Critical:** The PDM in `pyclnf/pyclnf/models/exported_pdm/` is the **30-mode** model exported from C++ OpenFace. This is the correct model for landmark detection.

### 3. pymtcnn (Face Detection) - `pymtcnn/`

Multi-task CNN face detector with CoreML backend for Apple Silicon.

```
pymtcnn/
├── pymtcnn/
│   ├── __init__.py
│   ├── detector.py              # Main MTCNN class
│   ├── backends/
│   │   ├── coreml.py           # CoreML backend (Apple Silicon)
│   │   ├── numpy_backend.py    # Pure NumPy fallback
│   │   └── cuda.py             # CUDA backend (optional)
│   └── models/
│       ├── pnet_fp32.mlpackage/
│       ├── rnet_fp32.mlpackage/
│       └── onet_fp32.mlpackage/
└── examples/
```

### 4. pyfhog (HOG Extraction) - `pyfhog/`

Fast Histogram of Oriented Gradients extraction using dlib's FHOG implementation.

```
pyfhog/
├── pyfhog/
│   ├── __init__.py
│   └── fhog.py              # Main FHOG extraction functions
├── src/
│   └── cpp/
│       └── fhog_wrapper.cpp # C++ wrapper for dlib FHOG
├── tests/
├── setup.py
└── pyproject.toml

Key function:
    pyfhog.extract_fhog(image, cell_size=8, num_bins=31) -> np.ndarray
```

**Note:** Local source available for modifications. Also installable via `pip install pyfhog`.

---

## Configuration

All critical parameters are locked in `pyfaceau/pyfaceau/config.py`:

```python
CLNF_CONFIG = {
    'max_iterations': 10,
    'convergence_threshold': 0.005,  # Gold standard (stricter than 0.01)
    'sigma': 2.25,                   # C++ CECLM default
    'use_eye_refinement': True,
    'convergence_profile': 'video',  # Template tracking
    'detector': False,               # pyfaceau handles detection
}

HOG_CONFIG = {
    'hog_dim': 4464,                 # 56×14 cells × 9 bins
    'hog_min': -0.005,               # CRITICAL: NOT 0.0
    'hog_max': 1.0,
}

AU_CONFIG = {
    'num_bins': 200,
    'min_val': -3.0,
    'max_val': 5.0,
    'cutoff_ratio': 0.10,            # 10th percentile baseline
    'skip_au17_cutoff': True,        # AU17 exception
}
```

---

## PDM Models

| Location | Modes | Purpose | Status |
|----------|-------|---------|--------|
| `pyclnf/pyclnf/models/exported_pdm/*.npy` | **30** | Landmark detection | ✅ CORRECT |
| `pyfaceau/weights/In-the-wild_aligned_PDM_68.txt` | 34 | Face alignment | ✅ OK (different purpose) |

The 30-mode PDM matches C++ OpenFace `pdm_68_aligned_menpo.txt` and is used for CLNF optimization.
The 34-mode PDM is used for face alignment reference shape only.

---

## Usage

### Basic API

```python
from pyfaceau import FullPythonAUPipeline

# Initialize pipeline
pipeline = FullPythonAUPipeline(
    mtcnn_backend='coreml'  # 'auto', 'coreml', 'numpy', 'cuda'
)

# Process video
results_df = pipeline.process_video("video.mp4")

# Process single frame
au_dict = pipeline.process_frame(frame)
```

### Import Path

```python
# Set PYTHONPATH before running:
# PYTHONPATH="pyfaceau:pyclnf:pymtcnn:." python your_script.py

# Or add to script:
import sys
sys.path.insert(0, 'pyfaceau')
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')
```

---

## Directory Structure

```
SplitFace Open3/
├── pyfaceau/                    # Main AU pipeline package
├── pyclnf/                      # Landmark detection package
├── pymtcnn/                     # Face detection package
├── pyfhog/                      # HOG feature extraction package
├── archive/                     # Archived/obsolete files
│   ├── debug_scripts/           # 200+ debug scripts
│   ├── pipeline_variants/       # Old pipeline versions
│   ├── nested_duplicates/       # Moved duplicates
│   │   ├── pyclnf_pyfaceau/    # Was nested in pyclnf
│   │   ├── pyclnf_pymtcnn/     # Was nested in pyclnf
│   │   └── S1 Face Mirror/     # Old project version
│   └── investigation_docs/      # Research documentation
├── compare_pipelines_100frames.py  # Validation script
├── CODE_STRUCTURE.md            # This file
├── AU_CORRELATION_INVESTIGATION.md  # Research findings
└── README.md
```

---

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Speed (vs C++) | **2.6x faster** | Python with CoreML |
| Upper Face AU Correlation | **98-99%** | AU01, AU02, AU05 |
| Lower Face AU Correlation | **52-72%** | AU12, AU17, AU26 |
| Overall AU Correlation | **70.8%** | Target: 92% |

### Known Limitations

1. **Lower face AUs** have lower correlation due to jaw landmark error (5-9 px vs <1 px for eyes)
2. **AU17 (chin raiser)** and **AU45 (blink)** are hardest to match
3. Root cause is pyCLNF landmark detection, not AU prediction

---

## Key Files

| File | Purpose |
|------|---------|
| `pyfaceau/pyfaceau/pipeline.py` | **CANONICAL ENTRY POINT** - Main AU pipeline |
| `pyfaceau/pyfaceau/config.py` | Locked configuration constants |
| `pyclnf/pyclnf/clnf.py` | CLNF landmark detector |
| `pyclnf/pyclnf/core/optimizer.py` | CLNF optimization with BORDER_REPLICATE |
| `pymtcnn/pymtcnn/detector.py` | MTCNN face detector |

---

## Dependencies

```
numpy
opencv-python
pandas
scipy
scikit-learn
coremltools  # For Apple Silicon
pyfhog       # pip install (system-wide)
```

---

## Applied Fixes (C++ Compatibility)

Critical fixes applied to match C++ OpenFace 2.2 behavior:

| Fix | File | Change | Status |
|-----|------|--------|--------|
| **#1 Border Handling** | `optimizer.py:1540+` | Use `BORDER_REPLICATE` instead of `-1e10` penalty | ✅ Applied |
| **#2 Precision** | `optimizer.py:732` | Use `float64` for sim_matrix | ✅ Applied |
| **#3 Temporal Tracking** | `clnf.py:799` | Lower correlation threshold 0.5→0.2 | ✅ Applied |
| **#4 Sigma Values** | N/A | Already loaded from model per-window-size | ✅ Verified |
| **#5 Regularization** | `optimizer.py:175-179` | reg=25.0, sigma=1.5, weight=0.0 | ✅ Applied |
| **#6 PDM Epsilon** | `pdm.py:776` | Removed `+ 1e-10` from regularization | ✅ Applied |
| **PDM Model** | `pyclnf/models/exported_pdm/` | 30-mode model exported from C++ | ✅ Applied |

**Result:** Mean landmark error reduced to 0.951 px (sub-pixel accuracy).

---

## Development Guidelines

1. **Single source of truth:** All AU processing goes through `pyfaceau/pipeline.py`
2. **No duplicate scripts:** Debug/test scripts go in `archive/`
3. **Config is locked:** Changes to `config.py` require thorough testing
4. **PDM models:** Use 30-mode for detection, 34-mode for alignment only
5. **Border handling:** Always use `BORDER_REPLICATE` in patch extraction

---

## Validation

Run the pipeline comparison test:

```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
PYTHONPATH="pyfaceau:pyclnf:pymtcnn:." python3 compare_pipelines_100frames.py
```

Expected output:
- 100 frames processed
- AU correlation ~70%
- Upper face AUs >98%
- Speed: 2-3 FPS Python, ~1 FPS C++
