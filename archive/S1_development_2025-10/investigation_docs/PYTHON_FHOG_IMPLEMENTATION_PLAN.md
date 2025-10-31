# Python FHOG Implementation Plan

**Goal:** Implement pure Python FHOG (Felzenszwalb HOG) extraction matching dlib's `extract_fhog_features()` output

**Date:** 2025-10-28
**Phase:** 3 - FHOG Extraction

## OpenFace's FHOG Usage

**Source:** `Face_utils.cpp:238-269`

```cpp
dlib::array2d<dlib::matrix<float,31,1> > hog;
dlib::extract_fhog_features(dlib_warped_img, hog, cell_size);

// Output: num_cols x num_rows x 31 dimensions
// Flattened order: for y in cols, for x in rows, for o in 31
descriptor = cv::Mat_<double>(1, num_cols * num_rows * 31);
```

### Key Parameters
- **Cell size:** 8x8 pixels (OpenFace default)
- **Features per cell:** 31 dimensions
- **Feature composition:**
  - 18 signed gradient orientations (0-360°)
  - 9 unsigned gradient orientations (0-180°)
  - 4 texture features (gradient energy features)
- **Input:** Aligned face image (typically 112x112 pixels)
- **Output:** Flattened 1D vector (4464 dims for typical face)

## FHOG vs Standard HOG

**Standard HOG (Dalal-Triggs):**
- 9 unsigned orientation bins (0-180°)
- Used for pedestrian detection
- Implemented in scikit-image, OpenCV

**FHOG (Felzenszwalb):**
- 31 features per cell (18 signed + 9 unsigned + 4 texture)
- Used for deformable part models
- More discriminative for object detection
- **Implemented in dlib**

## Python Implementation Options

### Option 1: dlib Python Bindings ⭐ RECOMMENDED
**Pros:**
- Exact match to OpenFace's C++ dlib
- Already implemented and tested
- Fast (C++ backend)
- Guaranteed compatibility

**Cons:**
- External dependency
- May have installation issues on some platforms

**Implementation:**
```python
import dlib
import numpy as np
import cv2

def extract_fhog_features_dlib(image, cell_size=8):
    """
    Extract FHOG features using dlib (matches OpenFace exactly)

    Args:
        image: BGR or grayscale image (numpy array)
        cell_size: HOG cell size in pixels (default: 8)

    Returns:
        features: 1D numpy array of FHOG features
    """
    # Convert to dlib format
    if len(image.shape) == 2:  # Grayscale
        dlib_img = image
    else:  # BGR
        dlib_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract FHOG
    fhog = dlib.extract_fhog_features(dlib_img, cell_size)

    # Flatten in same order as OpenFace
    num_cols = fhog.shape[1]
    num_rows = fhog.shape[0]
    features = []

    for y in range(num_cols):
        for x in range(num_rows):
            for o in range(31):
                features.append(fhog[x, y][o])

    return np.array(features, dtype=np.float64)
```

**Validation Strategy:**
- Compare against OpenFace .hog files
- Verify correlation > 0.999
- Test on multiple face images

### Option 2: Port dlib FHOG to Pure Python
**Pros:**
- No external dependencies beyond numpy/scipy
- Full control over implementation
- Educational value

**Cons:**
- Complex implementation (~500 lines)
- Potential for subtle bugs
- Slower than C++ (unless using numba/cython)
- High risk of not matching exactly

**Complexity:** HIGH - not recommended unless Option 1 fails

### Option 3: scikit-image HOG with FHOG Parameters
**Status:** Not viable - scikit-image implements Dalal-Triggs HOG (9 bins), not FHOG (31 features)

## Recommended Approach: Option 1 (dlib Python)

### Phase 3.1: Setup and Validation
1. Install dlib Python bindings
2. Create test script to extract FHOG from sample image
3. Compare output against OpenFace .hog file
4. Verify feature dimensions match (should be 4464 for typical face)

### Phase 3.2: Integration
1. Create `python_fhog_extractor.py` module
2. Implement face alignment (similarity transform from landmarks)
3. Extract FHOG from aligned face
4. Format output to match OpenFace's 4464-dim vector

### Phase 3.3: End-to-End Pipeline
1. Integrate FHOG extractor with AU predictor
2. Test full pipeline: Image → Landmarks → Alignment → FHOG → AU Predictions
3. Validate against OpenFace 2.2 output
4. Performance benchmarking

## Expected Output Dimensions

For a 112x112 aligned face image with cell_size=8:
- Grid size: 112/8 = 14 cells per dimension
- Total cells: 14 x 14 = 196 cells
- Features per cell: 31
- **Total features: 196 x 31 = 6076 dimensions**

**Wait, our .hog files have 4464 dimensions!**

Let me check the actual aligned face size used by OpenFace...

## Investigation Needed
- What is the exact size of aligned faces in OpenFace?
- What alignment transform is used?
- Are there border/padding considerations?

Let me check the .hog file structure to reverse-engineer dimensions.

## .HOG File Format (from openface22_hog_parser.py)

```
Frame 0: 4464 features
→ 4464 / 31 = 144 cells
→ sqrt(144) = 12 x 12 grid
→ Image size: 12 * 8 = 96 pixels
```

**Conclusion:** OpenFace uses **96x96 aligned faces** (not 112x112)

## Corrected Parameters
- **Aligned face size:** 96x96 pixels
- **Cell size:** 8x8 pixels
- **Grid:** 12x12 cells
- **Features per cell:** 31
- **Total dimensions:** 12 * 12 * 31 = **4464** ✓

## Implementation Timeline

| Task | Estimated Time | Priority |
|------|---------------|----------|
| Install dlib Python | 15 mins | HIGH |
| Create FHOG extractor | 30 mins | HIGH |
| Test on sample image | 15 mins | HIGH |
| Validate against .hog file | 30 mins | HIGH |
| Implement face alignment | 1 hour | HIGH |
| End-to-end integration | 1 hour | MEDIUM |
| Testing & validation | 1 hour | MEDIUM |

**Total:** ~4-5 hours

## Success Criteria
- ✅ FHOG output matches OpenFace .hog files (correlation > 0.999)
- ✅ Correct dimensions (4464 features)
- ✅ End-to-end pipeline produces same AU predictions as OpenFace
- ✅ No C++ dependencies

## Fallback Plan
If dlib Python bindings are not available or cause issues:
1. Use OpenFace C++ binary temporarily
2. Plan to port dlib FHOG to pure Python in future iteration
3. Document the dependency clearly

## Next Steps
1. Check if dlib is already installed: `pip list | grep dlib`
2. If not, install: `pip install dlib`
3. Create test script to verify dlib.extract_fhog_features() exists
4. Proceed with implementation

## References
- dlib FHOG documentation: http://dlib.net/imaging.html#extract_fhog_features
- Felzenszwalb et al. "Object Detection with Discriminatively Trained Part Based Models" (2010)
- OpenFace Face_utils.cpp:238-269
