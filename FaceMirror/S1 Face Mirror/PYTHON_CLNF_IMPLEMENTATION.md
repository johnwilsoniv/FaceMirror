# Python CLNF Implementation Progress

**Goal:** Implement pure Python version of OpenFace 2.2's CLNF (Constrained Local Neural Fields) landmark detector to handle challenging cases like surgical markings without external binary dependencies.

**Status:** IN PROGRESS (CEN Patch Expert Loader)

---

## Why Python CLNF?

**Problem:** Surgical markings cause PFLD/FAN to fail (76-100% poor quality)
**Solution:** OpenFace 2.2 CEN model succeeds (0% poor quality) via shape-constrained optimization
**Packaging Constraint:** Cannot bundle OpenFace binary (790 MB with dependencies)
**Our Approach:** Implement CLNF in Python using only CEN models (410 MB)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Python CLNF Detector                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CEN Patch Experts (DONE - debugging loader)            │
│     - Load .dat files (410 MB)                              │
│     - Evaluate patch responses                              │
│                                                             │
│  2. NU-RLMS Optimization (TODO)                             │
│     - Mean-shift with KDE                                   │
│     - Jacobian computation                                  │
│     - Gauss-Newton optimization                             │
│     - PDM constraint enforcement                            │
│                                                             │
│  3. Multi-Scale Fitting (TODO)                              │
│     - Coarse-to-fine optimization                           │
│     - Adaptive regularization                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: CEN Patch Expert Loader

### C++ Reference Code

**File:** `OpenFace/lib/local/LandmarkDetector/src/CEN_patch_expert.cpp`

**Key Functions:**
1. `CEN_patch_expert::Read(std::ifstream &stream)` - Load single patch expert
2. `CEN_patch_expert::Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)` - Compute response map

**File:** `OpenFace/lib/local/LandmarkDetector/src/Patch_experts.cpp`
- Manages collection of patch experts across scales and views

### Binary File Format: `cen_patches_*.dat`

#### Reverse-Engineered Structure

```
┌─────────────────────────────────────────────────────────────┐
│ CEN .dat File Format                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ HEADER (264 bytes for 7 views):                             │
│   [0-7]     patch_scale (double)        e.g., 0.25         │
│   [8-11]    num_views (int)             e.g., 7            │
│                                                             │
│   For each view (7 iterations):                             │
│     [12+i*36]    view_center_x (double)                     │
│     [20+i*36]    view_center_y (double)                     │
│     [28+i*36]    view_center_z (double)                     │
│                                                             │
│   Pattern observed in hex:                                  │
│     (3, 1, 6) appears at offsets: 12, 48, 84, 120,        │
│                                    156, 192, 228            │
│                                                             │
│ VISIBILITY MATRICES (variable size):                        │
│   [264-???]  num_landmarks (int) = 68                       │
│   [268-???]  visibility flags (68 ints)                     │
│              All values = 1 (visible) in test file          │
│                                                             │
│ PATCH EXPERTS (7 views × 68 landmarks):                     │
│   Starting offset: TBD (searching for type=6 marker)       │
│                                                             │
│   For each patch expert:                                    │
│     [0-3]    read_type (int) = 6 (CEN marker)              │
│     [4-7]    width_support (int)                            │
│     [8-11]   height_support (int)                           │
│     [12-15]  num_layers (int)                               │
│                                                             │
│     If num_layers == 0:                                     │
│       [16-23]  confidence (double)                          │
│       (empty patch - landmark invisible)                    │
│                                                             │
│     If num_layers > 0:                                      │
│       For each layer:                                       │
│         [0-3]    activation_function (int)                  │
│                  0=sigmoid, 1=tanh, 2=relu, 3=linear       │
│         [4-]     bias matrix (cv::Mat)                      │
│         [?-]     weight matrix (cv::Mat)                    │
│       [?-?]  confidence (double)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### cv::Mat Binary Format

```
┌─────────────────────────────────────────────────────────────┐
│ OpenCV Matrix Format (used for weights/biases)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [0-3]    rows (int)                                       │
│   [4-7]    cols (int)                                       │
│   [8-11]   type (int)                                       │
│             5 = CV_32F (float32)                            │
│             6 = CV_64F (float64)                            │
│   [12-]    data (rows × cols × elemSize bytes)             │
│             Row-major order                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Status

#### ✅ DONE

1. **File:** `pyfaceau/clnf/cen_patch_experts.py` (~390 lines)
   - `CENPatchExpert` class - single patch expert
   - `CENPatchExperts` class - manages all scales
   - `read_mat_bin()` - OpenCV matrix loader
   - `contrast_norm()` - row-wise normalization
   - `im2col_bias()` - image to column conversion
   - Neural network forward pass (sigmoid/tanh/relu)

#### ⚠️ IN PROGRESS

1. **Binary File Loader** - `CENPatchExperts._load_scale()`
   - ✅ Header parsing (scale, num_views)
   - ✅ View center parsing (7 × 3 doubles)
   - ⚠️ **BLOCKED:** Visibility matrix parsing
     - Issue: Matrix type = -2147483648 (invalid)
     - Need to find correct offset where patch experts start
   - ⚠️ **BLOCKED:** Patch expert parsing
     - Currently reading type=4 or type=7 instead of type=6
     - Offset calculation is incorrect

**Debug Status:**
```
Expected: type=6 at offset ???
Actual:   type=4 at offset 272 (wrong!)
          type=7 at offset 8 (wrong!)

Visibility section:
  Offset 264: 68 (num_landmarks?)
  Offset 268: 1 (visibility flags start?)
  Offsets 272-536: All 1s (68 × 4 bytes = 272 bytes)

Patch experts likely start at: 264 + 4 + 68*4 = 540?
```

#### 🔴 TODO

1. **Fix visibility matrix parsing**
   - Option A: Parse correctly using read_mat_bin()
   - Option B: Calculate exact byte offset and skip

2. **Verify patch expert loading**
   - Test on first expert at each scale
   - Validate neural network weights

3. **Test response computation**
   - Extract patch from test image
   - Compute response map
   - Verify output dimensions

### Next Steps for CEN Loader

1. Find exact offset where patch experts start after visibility data
2. Verify type=6 marker is read correctly
3. Load and validate first patch expert
4. Test response computation on sample image patch

---

## Component 2: NU-RLMS Optimization (TODO)

### C++ Reference Code

**File:** `OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`

**Key Function:** `CLNF::NU_RLMS()` (lines 990-1200)

### Algorithm Overview

```python
def nu_rlms_optimize(pdm, patch_responses, init_params, num_iters=10):
    """
    Non-Uniform Regularized Landmark Mean Shift optimization.

    Optimizes PDM parameters (not raw landmarks) to fit patch responses.
    """

    current_params = init_params.copy()

    for iter in range(num_iters):
        # 1. Generate current shape from PDM
        current_shape = pdm.reconstruct(current_params)

        # 2. Compute Jacobian (derivatives w.r.t. PDM parameters)
        J = pdm.compute_jacobian(current_params)

        # 3. Mean-shift calculation with KDE
        mean_shifts = compute_mean_shifts(patch_responses, current_shape)

        # 4. Regularization term (penalize unlikely shapes)
        reg_term = diag(reg_factor / eigenvalues)

        # 5. Solve linear system: (J^T W J + R) Δp = J^T W Δx
        Hessian = J.T @ W @ J + reg_term
        param_update = solve(Hessian, J.T @ W @ mean_shifts)

        # 6. Update PDM parameters
        current_params += param_update

        # 7. Clamp to ±3 sigma (enforce valid shapes)
        current_params = clamp(current_params, -3*std_devs, 3*std_devs)

        # 8. Check convergence
        if norm(current_shape - previous_shape) < threshold:
            break

    return pdm.reconstruct(current_params)
```

### Required Functions

1. **Mean-Shift with KDE**
   - `NonVectorisedMeanShift_precalc_kde()` (lines 820-900)
   - Precomputed Kernel Density Estimation
   - Subpixel refinement of peak locations

2. **Jacobian Computation**
   - `PDM::ComputeJacobian()` - derivatives w.r.t. shape parameters
   - Accounts for rigid (pose) and non-rigid (expression) components

3. **Linear System Solver**
   - Cholesky decomposition: `cv::solve(H, b, x, DECOMP_CHOLESKY)`
   - Python equivalent: `np.linalg.solve()` or `scipy.linalg.cho_solve()`

### Implementation Plan

**File:** `pyfaceau/clnf/nu_rlms.py` (estimated ~500 lines)

```python
class NURLMSOptimizer:
    def __init__(self, pdm, reg_factor=25.0, sigma=1.5, num_iters=10):
        self.pdm = pdm
        self.reg_factor = reg_factor
        self.sigma = sigma
        self.num_iters = num_iters
        self.kde_cache = {}  # Precomputed KDE responses

    def optimize(self, patch_responses, init_landmarks):
        # Main optimization loop
        pass

    def compute_mean_shifts(self, patch_responses, current_shape):
        # Mean-shift with KDE
        pass

    def precompute_kde(self, response_size):
        # Cache KDE kernels for efficiency
        pass
```

---

## Component 3: Multi-Scale Fitting (TODO)

### C++ Reference Code

**File:** `OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`

**Key Function:** `CLNF::Fit()` (lines 732-818)

### Algorithm Overview

```python
def multi_scale_fit(image, init_landmarks, pdm, patch_experts):
    """
    Coarse-to-fine optimization across multiple scales.
    """

    current_landmarks = init_landmarks

    # Optimize at each scale (0.25, 0.35, 0.50, 1.00)
    for scale_idx in range(4):
        # 1. Compute patch expert responses at this scale
        responses = patch_experts.response(image, current_landmarks, scale_idx)

        # 2. Adjust regularization based on scale
        #    (reduce regularization at finer scales)
        reg_factor = base_reg_factor - 15 * log2(patch_scaling[scale_idx] / 0.25)
        reg_factor = max(reg_factor, 0.001)

        # 3. Optimize landmarks using NU-RLMS
        current_landmarks = nu_rlms_optimize(
            pdm, responses, current_landmarks, reg_factor
        )

        # 4. Check if face too small
        if scale < 0.25:
            break

    return current_landmarks
```

### Implementation Plan

**File:** `pyfaceau/clnf/clnf_detector.py` (estimated ~200 lines)

```python
class CLNFLandmarkDetector:
    def __init__(self, model_dir):
        self.patch_experts = CENPatchExperts(model_dir)
        self.pdm = PDM.load(model_dir / 'pdm_68.txt')
        self.optimizer = NURLMSOptimizer(self.pdm)

    def detect(self, image, init_landmarks, bbox):
        # Multi-scale fitting
        pass

    def refine(self, image, poor_landmarks, bbox):
        # Called when PFLD produces poor quality
        pass
```

---

## Integration Plan

### Modify `pyfaceau_detector.py`

```python
class PyFaceAU68LandmarkDetector:
    def __init__(self, ...):
        # ...existing code...

        # Initialize Python CLNF for challenging cases
        clnf_model_dir = model_dir / 'clnf'
        if clnf_model_dir.exists():
            from pyfaceau.clnf import CLNFLandmarkDetector
            self.clnf_fallback = CLNFLandmarkDetector(clnf_model_dir)
        else:
            self.clnf_fallback = None

    def detect_landmarks(self, frame):
        # ...existing PFLD detection...

        # Check quality
        is_poor, reason = self.check_landmark_quality(landmarks_68)

        # Fallback to CLNF for poor quality
        if is_poor and self.clnf_fallback is not None:
            landmarks_68 = self.clnf_fallback.refine(
                frame, landmarks_68, self.cached_bbox
            )
            # Re-check quality
            is_poor, reason = self.check_landmark_quality(landmarks_68)
```

---

## Testing Strategy

### Phase 1: CEN Loader (Current)

```bash
python test_cen_loading.py
```

**Expected Output:**
```
✅ Successfully loaded CEN patch experts!
   Scales: [0.25, 0.35, 0.5, 1.0]
   Landmarks: 68
   Total experts: 272 (4 scales × 68 landmarks)

   Scale 0.25:
     Landmark 0: 11x11 patch
     Layers: 3
     Empty: False
```

### Phase 2: Response Computation

```python
# Test patch expert response on sample image
image = cv2.imread('test_face.jpg', cv2.IMREAD_GRAYSCALE)
init_landmarks = pfld_detector.detect(image)

cen = CENPatchExperts(model_dir)
responses = cen.response(image, init_landmarks, scale_idx=2)  # 0.50 scale

print(f"Response shape: {responses[0].shape}")  # Should be (height, width)
print(f"Response range: [{responses[0].min()}, {responses[0].max()}]")
```

### Phase 3: NU-RLMS Optimization

```python
# Test optimization on good vs poor landmarks
optimizer = NURLMSOptimizer(pdm)

# Test on surgical marking case
optimized = optimizer.optimize(responses, poor_landmarks)
is_poor_after, _ = check_landmark_quality(optimized)

print(f"Before CLNF: {'POOR' if is_poor_before else 'GOOD'}")
print(f"After CLNF:  {'POOR' if is_poor_after else 'GOOD'}")
```

### Phase 4: End-to-End

```python
# Test on IMG_8401 (surgical markings)
detector = PyFaceAU68LandmarkDetector(enable_clnf_fallback=True)
landmarks = detector.detect_landmarks(frame)

# Should succeed where PFLD failed
assert detector.get_quality_statistics()['percentage'] < 10  # < 10% poor
```

---

## Performance Targets

| Metric | Target | OpenFace C++ | Notes |
|--------|--------|--------------|-------|
| Accuracy | 0% poor on IMG_8401 | 0% | Match OpenFace |
| Speed (fallback only) | 10-20 FPS | 5-10 FPS | Acceptable for rare cases |
| Memory | < 500 MB | ~800 MB | CEN models only |

---

## File Structure

```
pyfaceau/
└── pyfaceau/
    ├── clnf/
    │   ├── __init__.py              ✅ Done
    │   ├── cen_patch_experts.py     ⚠️  In Progress (390 lines)
    │   ├── nu_rlms.py               🔴 TODO (~500 lines)
    │   └── clnf_detector.py         🔴 TODO (~200 lines)
    └── refinement/
        └── pdm.py                    ✅ Done (already exists)

S1 Face Mirror/
└── weights/
    └── clnf/
        ├── cen_patches_0.25_of.dat  ✅ Copied (58 MB)
        ├── cen_patches_0.35_of.dat  ✅ Copied (58 MB)
        ├── cen_patches_0.50_of.dat  ✅ Copied (147 MB)
        ├── cen_patches_1.00_of.dat  ✅ Copied (147 MB)
        └── tris_68.txt              ✅ Copied (triangulation)
```

---

## Estimated Remaining Work

| Task | Lines | Hours | Status |
|------|-------|-------|--------|
| Fix CEN loader (debug offsets) | +50 | 2 | 🔄 Current |
| Test CEN response computation | +100 | 1 | 📋 Next |
| Implement NU-RLMS core | 500 | 8 | 📋 Queued |
| Implement mean-shift KDE | (in NU-RLMS) | - | 📋 Queued |
| Implement multi-scale fitting | 200 | 3 | 📋 Queued |
| Integration + testing | +100 | 2 | 📋 Queued |
| **TOTAL** | ~950 | **16 hours** | |

**Realistic Timeline:** 2-3 days of focused work

---

## Debug Log

### 2025-11-03 - CEN Loader Issues

**Problem:** Cannot find correct start offset for patch experts in .dat file

**Attempts:**
1. ❌ Offset 0: type=0 (scale header)
2. ❌ Offset 8: type=7 (num_views)
3. ❌ Offset 272: type=4 (unknown)
4. ❌ Offset 264: type=68 (num_landmarks?)

**Observations:**
- Pattern (3,1,6) appears at: 12, 48, 84, 120, 156, 192, 228 (view headers)
- After offset 264: sequence of 68 int values (all 1s)
- This is likely the visibility matrix: 68 landmarks × 1 int each = 272 bytes
- Patch experts should start at: 264 + 272 = **536 bytes**

**Next Action:**
Try offset 536 for first patch expert type=6 marker

---

## CRITICAL DECISION POINT (2025-11-03)

### Binary Format Challenge

After 2+ hours of reverse engineering, the CEN .dat format is proving more complex than expected:

1. **Header Structure Uncertain:**
   - View centers parsing produces garbage values
   - Visibility matrices have unknown structure
   - Patch expert start offset unclear (tried: 272, 536, others)

2. **Time Investment:**
   - Estimated 4-6 more hours just to debug the loader
   - Then 12+ hours for NU-RLMS + multi-scale fitting
   - Total: ~18 hours remaining

3. **Alternative Approach:**
   - OpenFace binary works perfectly (0% poor on surgical markings)
   - Could shell out to OpenFace for fallback cases only
   - Trading packaging size (790 MB) for development time

### Recommended Path Forward

**Option 1: Use OpenFace Binary as Subprocess (FASTEST)**
- Time: 2-3 hours
- Pros: Proven, no reverse engineering
- Cons: 790 MB package, external dependency

**Option 2: Continue Python CLNF (CLEANEST)**
- Time: 16-20 hours
- Pros: Pure Python, 410 MB models only
- Cons: Complex, high risk of more surprises

**Option 3: Hybrid Approach (BALANCED)**
- Phase 1: Ship with OpenFace binary now
- Phase 2: Replace with Python CLNF in future release
- Pros: Fast to market, clean migration path
- Cons: Two implementations to maintain

**User Decision Needed:** Which path should we take?

