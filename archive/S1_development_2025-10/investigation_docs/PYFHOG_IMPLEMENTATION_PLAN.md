# PyFHOG Implementation Plan - Cross-Platform FHOG Extraction

**Goal:** Create a minimal Python package that wraps dlib's FHOG extraction for cross-platform distribution.

**Date:** 2025-10-28
**Status:** Planning Phase
**Estimated Time:** 1-2 days development + 1 day CI setup

---

## Overview

Instead of:
- ❌ Porting 1,400 lines of C++ to Python (risky, slow)
- ❌ Bundling platform-specific binaries (messy distribution)

We create:
- ✅ **`pyfhog`** - A minimal C++ Python extension
- ✅ Wraps only dlib's `extract_fhog_features()` function
- ✅ Distributes as wheels for Mac/Linux/Windows
- ✅ Installs like any pip package: `pip install pyfhog`

---

## Project Structure

```
pyfhog/
├── README.md
├── LICENSE (Boost Software License - matches dlib)
├── setup.py                    # Build configuration
├── pyproject.toml              # Modern Python packaging
├── CMakeLists.txt              # CMake build (optional, for complex builds)
├── src/
│   ├── pyfhog/
│   │   ├── __init__.py         # Python package entry
│   │   └── _version.py
│   └── cpp/
│       ├── fhog_wrapper.cpp    # Main pybind11 wrapper (~200 lines)
│       └── dlib/               # Vendored dlib FHOG headers (minimal subset)
│           └── image_transforms/
│               ├── fhog.h
│               ├── fhog_abstract.h
│               └── (required dependencies)
├── tests/
│   ├── test_fhog_extraction.py
│   └── test_data/
│       └── sample_face.png
├── .github/
│   └── workflows/
│       └── build_wheels.yml    # GitHub Actions for CI/CD
└── requirements-dev.txt
```

---

## Implementation Steps

### **Phase 1: Local Development (macOS)**

#### Step 1.1: Create Package Structure
```bash
mkdir -p pyfhog/src/pyfhog
mkdir -p pyfhog/src/cpp
mkdir -p pyfhog/tests
```

#### Step 1.2: Write pybind11 Wrapper

**File: `src/cpp/fhog_wrapper.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dlib/image_transforms/fhog.h"
#include "dlib/array2d.h"
#include "dlib/pixel.h"

namespace py = pybind11;

// Convert NumPy array to dlib image
dlib::array2d<dlib::rgb_pixel> numpy_to_dlib_image(
    py::array_t<uint8_t> img_array
) {
    auto buf = img_array.request();

    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input must be HxWx3 RGB image");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];

    dlib::array2d<dlib::rgb_pixel> img(height, width);
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = (r * width + c) * 3;
            img[r][c].red = ptr[idx];
            img[r][c].green = ptr[idx + 1];
            img[r][c].blue = ptr[idx + 2];
        }
    }

    return img;
}

// Convert dlib FHOG output to NumPy array
py::array_t<double> dlib_hog_to_numpy(
    const dlib::array2d<dlib::matrix<float,31,1>>& hog
) {
    int num_rows = hog.nr();
    int num_cols = hog.nc();
    int num_features = 31;

    // Allocate NumPy array
    auto result = py::array_t<double>(num_rows * num_cols * num_features);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // Flatten in same order as OpenFace (y, x, orientation)
    int idx = 0;
    for (int y = 0; y < num_cols; ++y) {
        for (int x = 0; x < num_rows; ++x) {
            for (int o = 0; o < num_features; ++o) {
                ptr[idx++] = hog[x][y](o);
            }
        }
    }

    return result;
}

// Main extraction function
py::array_t<double> extract_fhog_features(
    py::array_t<uint8_t> image,
    int cell_size = 8
) {
    // Convert NumPy to dlib format
    auto dlib_img = numpy_to_dlib_image(image);

    // Extract FHOG using dlib
    dlib::array2d<dlib::matrix<float,31,1>> hog;
    dlib::extract_fhog_features(dlib_img, hog, cell_size);

    // Convert back to NumPy
    return dlib_hog_to_numpy(hog);
}

// Python module definition
PYBIND11_MODULE(_pyfhog, m) {
    m.doc() = "Fast FHOG feature extraction using dlib";

    m.def("extract_fhog_features", &extract_fhog_features,
          py::arg("image"),
          py::arg("cell_size") = 8,
          R"pbdoc(
              Extract Felzenszwalb HOG features from an image.

              Args:
                  image: NumPy array of shape (H, W, 3) in RGB format, dtype=uint8
                  cell_size: Size of HOG cells in pixels (default: 8)

              Returns:
                  1D NumPy array of FHOG features (flattened)

              Example:
                  >>> import pyfhog
                  >>> import numpy as np
                  >>> img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
                  >>> features = pyfhog.extract_fhog_features(img)
                  >>> features.shape
                  (4464,)  # For 96x96 image with cell_size=8
          )pbdoc");

    m.attr("__version__") = "0.1.0";
}
```

#### Step 1.3: Create `setup.py`

```python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

__version__ = '0.1.0'

# Determine include directories
include_dirs = [
    pybind11.get_include(),
    "src/cpp",  # For vendored dlib headers
]

ext_modules = [
    Extension(
        'pyfhog._pyfhog',
        ['src/cpp/fhog_wrapper.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],  # Optimization
    ),
]

setup(
    name='pyfhog',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/pyfhog',
    description='Fast FHOG feature extraction for Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['pyfhog'],
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    install_requires=['numpy>=1.20.0'],
    python_requires='>=3.8',
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
```

#### Step 1.4: Create Python Package

**File: `src/pyfhog/__init__.py`**

```python
"""
PyFHOG - Fast Felzenszwalb HOG feature extraction for Python

Provides a minimal wrapper around dlib's optimized FHOG implementation.
"""

from ._pyfhog import extract_fhog_features, __version__

__all__ = ['extract_fhog_features', '__version__']
```

#### Step 1.5: Vendor Minimal dlib Headers

Copy only the required dlib headers:
```bash
# From OpenFace's dlib installation
cp -r /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/3rdParty/dlib/include/dlib/image_transforms pyfhog/src/cpp/dlib/
cp -r /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/3rdParty/dlib/include/dlib/simd.h pyfhog/src/cpp/dlib/
# ... (copy minimal dependencies)
```

#### Step 1.6: Write Tests

**File: `tests/test_fhog_extraction.py`**

```python
import pytest
import numpy as np
import pyfhog

def test_fhog_extraction():
    """Test basic FHOG extraction"""
    # Create 96x96 test image
    img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

    # Extract features
    features = pyfhog.extract_fhog_features(img, cell_size=8)

    # Expected dimensions: (96/8)^2 * 31 = 12*12*31 = 4464
    assert features.shape == (4464,)
    assert features.dtype == np.float64

def test_fhog_vs_openface():
    """Validate against OpenFace .hog file"""
    # Load test image and reference .hog file
    import cv2
    from openface22_hog_parser import OF22HOGParser

    # Load aligned face
    img = cv2.imread("test_data/aligned_face.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract with pyfhog
    features_py = pyfhog.extract_fhog_features(img_rgb, cell_size=8)

    # Load OpenFace reference
    parser = OF22HOGParser("test_data/reference.hog")
    _, features_of = parser.parse()
    features_of = features_of[0]  # First frame

    # Validate correlation
    correlation = np.corrcoef(features_py, features_of)[0, 1]
    assert correlation > 0.999, f"Correlation too low: {correlation}"
```

#### Step 1.7: Build Locally

```bash
cd pyfhog
pip install pybind11 numpy
python setup.py build_ext --inplace
python -m pytest tests/
```

---

### **Phase 2: Cross-Platform Wheel Building**

#### GitHub Actions Workflow

**File: `.github/workflows/build_wheels.yml`**

```yaml
name: Build Wheels

on:
  push:
    branches: [ main ]
  release:
    types: [ created ]

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build wheel setuptools pybind11

      - name: Build wheel
        run: python -m build --wheel

      - name: Upload wheels
        uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: dist/*.whl

  upload_pypi:
    needs: [build_wheels]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    steps:
      - uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
```

---

## Validation Strategy

### Test 1: Exact Output Match
```python
# Ensure pyfhog produces IDENTICAL output to OpenFace
correlation = verify_pyfhog_vs_openface()
assert correlation > 0.9999
```

### Test 2: End-to-End AU Prediction
```python
# Use pyfhog in full pipeline
predictor = OpenFace22AUPredictor(fhog_backend='pyfhog')
results = predictor.predict_video(test_video)
correlation_with_of22 = validate_au_predictions(results)
assert correlation_with_of22 > 0.9996
```

---

## Benefits of This Approach

| Aspect | Benefit |
|--------|---------|
| **Cross-platform** | Single `pip install pyfhog` works on Mac/Linux/Windows |
| **Performance** | Full C++ SIMD speed (dlib unchanged) |
| **Accuracy** | Zero risk - uses proven dlib code |
| **Size** | ~200-500KB wheel (vs 1.2MB+ full binary) |
| **Maintenance** | Isolated - only track dlib FHOG changes |
| **Integration** | Drop-in replacement for OF C++ binary |
| **Distribution** | Standard PyPI distribution |

---

## Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1.1-1.3 | Project structure + wrapper code | 4 hours | ⏳ Pending |
| 1.4-1.6 | Python package + tests | 2 hours | ⏳ Pending |
| 1.7 | Local build and testing | 2 hours | ⏳ Pending |
| 2.1 | CI/CD setup (GitHub Actions) | 3 hours | ⏳ Pending |
| 2.2 | Cross-platform testing | 2 hours | ⏳ Pending |
| 2.3 | PyPI deployment | 1 hour | ⏳ Pending |
| **Total** | | **14 hours (~2 days)** | |

---

## Alternative: Use cibuildwheel

For even easier cross-platform wheel building:

```yaml
# .github/workflows/wheels.yml
name: Build wheels

on: [push, pull_request]

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v3

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.16.2
        env:
          CIBW_BUILD: cp38-* cp39-* cp310-* cp311-* cp312-*
          CIBW_SKIP: "*-musllinux_*"

      - uses: actions/upload-artifact@v3
        with:
          path: ./wheelhouse/*.whl
```

**`cibuildwheel` advantages:**
- Automatically builds for all platforms/Python versions
- Handles manylinux, macOS universal2, Windows
- Industry standard (used by NumPy, SciPy, etc.)

---

## Next Steps

1. **Create `pyfhog` repository structure**
2. **Write pybind11 wrapper** (200 lines)
3. **Vendor minimal dlib headers** (just FHOG subset)
4. **Local build and test** (macOS first)
5. **Validate against OpenFace .hog files** (r > 0.999)
6. **Set up CI/CD** (GitHub Actions + cibuildwheel)
7. **Test wheels on all platforms**
8. **Integrate into OpenFace22AUPredictor**

---

## Success Criteria

✅ `pip install pyfhog` works on Mac/Linux/Windows
✅ Correlation with OpenFace .hog files > 0.9999
✅ End-to-end AU prediction maintains r = 0.9996
✅ Wheel size < 1MB
✅ No external binary dependencies

---

**Ready to proceed?** We can start by creating the project structure and pybind11 wrapper!
