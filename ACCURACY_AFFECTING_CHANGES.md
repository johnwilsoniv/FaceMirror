# Accuracy-Affecting Changes Since Gold Standard

**Gold Standard Commit**: `b6afcdf2` (91.85% accuracy)
**Current State**: Multiple optimization commits

This document lists ALL changes that may affect AU prediction accuracy. Review carefully before reverting.

---

## 1. CLNF Parameter Changes (clnf.py)

### 1.1 Regularization
```python
# Gold Standard (line 53)
regularization: float = 35,  # Match C++ OpenFace default reg_factor=35

# Current (line 53)
regularization: float = 20,  # Optimal for Python (C++ uses 35 but Python needs lower)
```
**Impact**: Lower regularization = less shape constraint = potentially worse landmark accuracy

### 1.2 Max Iterations
```python
# Gold Standard (line 54)
max_iterations: int = 10,

# Current (line 54)
max_iterations: int = 5,  # Reduced for faster processing (was 10)
```
**Impact**: Fewer iterations = less refinement = worse landmark accuracy

### 1.3 Convergence Threshold
```python
# Gold Standard (line 55)
convergence_threshold: float = 0.005,

# Current (line 55)
convergence_threshold: float = 0.5,  # Mean per-landmark change threshold (pixels)
```
**Impact**: 100x higher threshold = early stopping = significantly worse accuracy

### 1.4 Patch Scaling
```python
# Gold Standard (line 96)
self.patch_scaling = [0.25, 0.35, 0.5]

# Current (line 98)
self.patch_scaling = [0.25, 0.35, 0.5, 1.0]
```
**Impact**: Added 1.0 scale - may be beneficial or neutral

### 1.5 Window Size Filtering
```python
# Gold Standard (lines 110-114)
filtered_windows = [ws for ws in self.window_sizes if ws in available_windows]
if filtered_windows != self.window_sizes:
    print(f"Filtering window sizes to those with sigma components: ...")
    self.window_sizes = filtered_windows

# Current (lines 118-121)
missing_windows = [ws for ws in self.window_sizes if ws not in available_windows]
if missing_windows:
    print(f"Note: No sigma components for window sizes {missing_windows}, using identity transform")
```
**Impact**: Now processes window sizes without sigma components (window size 5) - may affect accuracy

---

## 2. Optimizer Changes (optimizer.py)

### 2.1 Numba JIT Mean-Shift (NEW - lines 27-80)
```python
@jit(nopython=True, cache=True)
def _kde_mean_shift_numba(response_map, dx, dy, a, kde_weights):
    """Numba-optimized KDE-based mean-shift computation."""
    ...
```
**Impact**: Should be numerically identical but floating-point differences possible due to:
- Different order of operations
- fastmath optimizations
- Loop unrolling

### 2.2 Convergence Check Formula - RIGID Phase (lines 237-244)
```python
# Gold Standard
# Check convergence: ||current - previous|| < 0.01 (C++ line 1173)
if previous_landmarks is not None:
    shape_change = np.linalg.norm(current_landmarks - previous_landmarks)
    if shape_change < 0.01:
        rigid_converged = True
        break

# Current
# Check convergence: ||current - previous|| < threshold
# Use mean per-landmark change for more intuitive threshold
if previous_landmarks is not None:
    shape_change = np.linalg.norm(current_landmarks - previous_landmarks)
    mean_change = shape_change / np.sqrt(len(current_landmarks))
    if mean_change < self.convergence_threshold:
        rigid_converged = True
        break
```
**Impact**: Changed from fixed 0.01 norm threshold to configurable per-landmark mean threshold
- With threshold=0.5: stops when mean change < 0.5px (vs total change < 0.01 before)
- Equivalent: 0.5 * sqrt(68) ≈ 4.1 total norm vs 0.01 = **410x earlier stopping**

### 2.3 Convergence Check Formula - NON-RIGID Phase (lines 329-338)
```python
# Same change as rigid phase
```
**Impact**: Same as above

### 2.4 Debug Print Conditions (multiple locations)
```python
# Gold Standard
if window_size == 11:
    print(...)

# Current
if window_size == 11 and self.debug_mode:
    print(...)
```
**Impact**: None on accuracy (only affects output)

---

## 3. CEN Patch Expert Changes (cen_patch_expert.py)

### 3.1 Numba JIT Response Computation (NEW)
```python
@njit(fastmath=True, cache=True)
def _response_core_numba(...):
    """Numba-optimized core response computation for 2-layer CEN."""
    ...
```
**Impact**: Potential floating-point differences due to:
- `fastmath=True` allows non-IEEE-compliant optimizations
- Loop reordering
- Different accumulation order in matrix multiply

### 3.2 Numba JIT im2col (NEW)
```python
@njit(fastmath=True, cache=True)
def _im2col_bias_numba(input_patch, width, height, output):
    """Numba-optimized im2col with bias column."""
    ...
```
**Impact**: Should be identical but verify patch ordering matches C++

### 3.3 Numba JIT Forward Pass (NEW)
```python
@njit(fastmath=True, cache=True)
def _forward_pass_numba(layer_input, weights_T, bias, activation_type):
    """Numba-optimized single layer forward pass."""
    ...
```
**Impact**: Potential differences in:
- Sigmoid clamping values
- Accumulation order in matmul

### 3.4 Numba JIT Contrast Normalization (NEW)
```python
@njit(fastmath=True, cache=True)
def _contrast_norm_numba(patches):
    """Numba-optimized contrast normalization."""
    ...
```
**Impact**: Division and sqrt operations may have different precision

### 3.5 Mirrored Patch Expert (NEW)
```python
class MirroredCENPatchExpert:
    """Wrapper for CEN patch expert that applies horizontal flipping."""
    ...
```
**Impact**: Uses mirror landmarks for empty patches - should improve accuracy

### 3.6 Response Function Routing
```python
# Gold Standard
# Direct numpy computation always

# Current
if NUMBA_AVAILABLE and len(self.weights) == 2:
    return _response_core_numba(...)
# Fallback for non-2-layer networks
```
**Impact**: Most patches use Numba path which may differ numerically

---

## 4. Cython Modules (NEW FILES)

### 4.1 optimizer_cython.pyx
- Cython version of NURLMSOptimizer
- May have different numerical behavior than Python version
- Check if being used in production

### 4.2 optimizer_cython.cpp (Generated)
- Compiled Cython code
- Verify not accidentally being imported

---

## 5. Summary of High-Impact Changes

### CRITICAL (Likely causing accuracy drop):
1. **convergence_threshold**: 0.005 → 0.5 (100x higher, ~410x earlier stopping)
2. **Convergence formula**: Fixed 0.01 → per-landmark mean (different calculation)
3. **max_iterations**: 10 → 5 (50% fewer iterations)
4. **regularization**: 35 → 20 (43% lower shape constraint)

### MODERATE (May affect accuracy):
5. **Numba fastmath**: Non-IEEE floating point
6. **Window size 5**: Now processed without sigma components
7. **patch_scaling 1.0**: Added finest scale

### LOW RISK (Unlikely to affect):
8. Debug print conditions
9. MirroredCENPatchExpert (should improve accuracy)
10. Cython modules (if not imported)

---

## 6. Recommended Reversion Order

To restore accuracy while keeping some optimizations:

### Phase 1: Revert Critical Parameters
1. `convergence_threshold`: 0.5 → 0.005
2. `max_iterations`: 5 → 10
3. `regularization`: 20 → 35
4. Convergence formula: Use fixed 0.01 threshold

### Phase 2: Test Numba Paths
5. Disable `fastmath=True` in Numba decorators
6. Compare Numba vs pure Python outputs numerically

### Phase 3: Investigate Window Size 5
7. Test with/without window size 5 processing
8. Check if sigma identity transform is correct

---

## 7. Quick Revert Commands

To revert clnf.py parameters to gold standard:
```bash
git show b6afcdf2:pyclnf/clnf.py > /tmp/clnf_gold.py
# Then manually copy parameter values
```

To see exact gold standard values:
```bash
git show b6afcdf2:pyclnf/clnf.py | grep -A5 "def __init__"
git show b6afcdf2:pyclnf/core/optimizer.py | grep -A5 "Check convergence"
```

---

## 8. Validation Command

After reverting, run validation:
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
PYTHONPATH="pyfaceau:pyclnf:pymtcnn:." python3 archive/test_scripts/validate_full_pipeline.py
```

Expected result: Mean AU correlation > 0.90

---

*Document generated: 2024-11-21*
*Current accuracy: ~49% (needs investigation)*
*Target accuracy: >90%*
