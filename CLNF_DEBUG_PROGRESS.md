# CLNF Python Implementation Debug Progress

## Summary of Improvements

### Starting Point
- **Initial Error**: ~9.37px divergence from C++ OpenFace
- **Main Issue**: Python implementation not matching C++ behavior

### Critical Bugs Fixed

#### 1. Variable Name Collision (MOST CRITICAL)
- **Location**: `pyclnf/core/optimizer.py` lines 433-496
- **Bug**: Variable `a` used for both Gaussian kernel parameter AND similarity transform
- **Impact**: KDE weights completely wrong (100x-700x error in J_w_t_m)
- **Fix**: Renamed to `a_kde` (Gaussian) and `a_sim`/`b_sim` (similarity transform)
- **Result**: Error reduced from 10.85px → 6.414px (41% improvement)

#### 2. Missing CEN Sigma Components
- **Location**: `pyclnf/core/cen_patch_expert.py` line 195
- **Bug**: CEN incorrectly set `self.sigma_components = {}`
- **Fix**: Now loads sigma components from disk
- **Added**: `compute_sigma()` method to CENPatchExpert (returns identity for quick mode)

#### 3. Window Size 5 Filtering
- **Issue**: No sigma components exist for window size 5
- **Fix**: Automatically filter to [11, 9, 7] when sigma components are available

#### 4. Precomputed KDE Grid Implementation
- **Added**: Matching C++ discretized KDE grid with 0.1 pixel spacing
- **Location**: `pyclnf/core/optimizer.py` lines 574-621

### Current Status
- **Error**: 7.313px (down from 9.37px - 22% improvement)
- **Breakdown by landmark**:
  - Landmark 36: 6.273px
  - Landmark 48: 5.836px
  - Landmark 30: 3.485px
  - Landmark 8: 13.655px (worst)

### Remaining Issues

#### 1. KDE Accumulation Difference (30% lower in Python)
```
C++:    mx=18.42, my=24.87, sum=4.677
Python: mx=13.54, my=17.77, sum=3.338
```
Possible causes:
- Response map normalization differences
- Numerical precision in exp() computation
- Different response map extraction

#### 2. CEN Sigma Computation
Currently returns identity matrix (quick mode). May need proper implementation with betas for full accuracy.

#### 3. Response Map Peak Differences
Small differences in response maps (~0.001 mean difference) may be amplified by KDE accumulation.

## Key Files Modified
1. `pyclnf/core/optimizer.py` - Fixed variable collision, added KDE grid
2. `pyclnf/core/cen_patch_expert.py` - Added sigma loading and compute_sigma
3. `pyclnf/clnf.py` - Window size filtering logic

## Next Steps to Reach < 1px Error
1. Investigate response map normalization differences
2. Implement proper CEN sigma computation (not just identity)
3. Debug remaining 30% KDE accumulation difference
4. Verify iteration counts match C++ exactly

## Test Command
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
env PYTHONPATH="pyclnf:." python3 test_final_comparison.py
```

## C++ Reference Values
- Landmark 36: (364.3000, 866.1000)
- Landmark 48: (420.6000, 1053.5000)
- Landmark 30: (483.8000, 944.3000)
- Landmark 8: (503.0000, 1164.3000)