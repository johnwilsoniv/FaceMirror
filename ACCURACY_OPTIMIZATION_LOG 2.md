# Accuracy Optimization Log

## Goal
Reduce Python CLNF landmark error from ~2.5px to <0.5px (matching C++ ~0.67px)

## Test Configuration
- **Video**: `/Patient Data/Normal Cohort/Shorty.mov`
- **Frame**: 30 (primary), 160 (secondary)
- **Metric**: Mean pixel error across 68 landmarks vs C++ OpenFace

---

## Results Summary

| Date | Configuration | Overall | Eyes | Max | Notes |
|------|--------------|---------|------|-----|-------|
| Baseline | Before fixes | 2.98px | 2.59px | - | With eye refinement |
| Test 1 | min_iterations=5 added | 2.58px | 2.69px | 5.47px | 13% improvement |
| Test 2 | reg=25 | 2.61px | 2.62px | 5.57px | Slightly worse overall |
| Test 3 | windows=[11,9,7], scales=[0.25,0.35,0.5] | 8.14px | 8.51px | 14.29px | REGRESSION - sigma broken |
| Test 4 | Restored windows=[11,9,7,5] | 2.68px | 2.55px | 5.77px | Back to baseline |
| Test 5 | threshold=0.5 | 2.58px | 2.69px | 5.47px | Best so far |
| Test 6 | No eye refinement | 2.57px | 2.63px | 5.47px | Slightly better |
| **Test 7** | **base_landmarks_nonrigid = initial** | **1.23px** | **2.38px** | **3.32px** | **2x improvement!** |
| Test 8 | + eye refinement enabled | 1.37px | 3.17px | 6.37px | Eye refinement HURTS |
| Test 9 | reg=25, no eye refine | 1.08px | 2.29px | 3.33px | 12% better |
| Test 10 | threshold=0.1 | 1.08px | 2.30px | 3.34px | No change |
| Test 11 | reg=30 | 1.00px | 2.25px | 3.37px | Broke 1px! |
| **Test 12** | **reg=35** | **0.74px** | **2.11px** | **3.33px** | **BEST - Near C++ 0.67px!** |
| Test 13 | reg=40 | 0.84px | 2.08px | 3.36px | Worse than reg=35 |

---

## Key Fixes Applied

### 1. min_iterations Parameter (Completed)
- **Files**: `pyclnf/core/optimizer.py`, `pyclnf/clnf.py`
- **Change**: Added `min_iterations=5` to prevent early convergence termination
- **Impact**: 13% improvement (2.98px → 2.58px)

### 2. Non-Rigid Base Landmarks Fix (Completed) - CRITICAL
- **File**: `pyclnf/core/optimizer.py` line 296
- **Change**: `base_landmarks_nonrigid = landmarks_2d_initial.copy()` instead of rigid-updated params
- **Impact**: 2x improvement (2.57px → 1.23px)

### 3. Window Sizes (Reverted)
- **Finding**: Removing WS=5 and scale=1.0 breaks sigma adjustment
- **Current**: `windows=[11, 9, 7, 5]`, `scales=[0.25, 0.35, 0.5, 1.0]`
- **Note**: sigma=2.25 requires scale_max=1.0

---

## Current Best Configuration

```python
clnf = CLNF(
    'pyclnf/models',
    regularization=35,           # C++ video mode default (was 20)
    convergence_threshold=0.5,   # Default
    window_sizes=[11, 9, 7, 5],  # Default
    min_iterations=5,            # Prevents early termination
    use_eye_refinement=False     # DISABLED - causes regression
)
```

**Result**: **0.74px** mean error (frame 30) - within 0.07px of C++ 0.67px!

### Per-Region Breakdown (Best Result)
| Region | Mean Error | Status |
|--------|------------|--------|
| Jaw (0-16) | 0.70px | GOOD |
| R Eyebrow (17-21) | 0.39px | **EXCELLENT** |
| L Eyebrow (22-26) | 0.59px | GOOD |
| Nose (27-35) | 0.39px | **EXCELLENT** |
| **R Eye (36-41)** | **2.29px** | **BOTTLENECK** |
| **L Eye (42-47)** | **1.94px** | **BOTTLENECK** |
| Outer Lip (48-59) | **0.24px** | **EXCELLENT** |
| Inner Lip (60-67) | **0.23px** | **EXCELLENT** |

---

## Current Status

### Achieved
- Overall: 1.23px mean error
- All points under 5px threshold
- Face center distance: 0.4px

### Remaining Issues
- Eyes: 2.38px (higher than other regions)
- Goal: <0.5px (need 2.5x more improvement)

### Per-Region Breakdown (Test 7)
| Region | Mean Error |
|--------|------------|
| Jaw (0-16) | 1.14px |
| R Eyebrow (17-21) | 1.13px |
| L Eyebrow (22-26) | 0.65px |
| Nose (27-35) | 1.14px |
| **R Eye (36-41)** | **2.49px** |
| **L Eye (42-47)** | **2.28px** |
| Outer Lip (48-59) | 0.83px |
| Inner Lip (60-67) | 0.80px |

---

## Next Steps

1. **Test eye refinement** with the new base_landmarks fix
2. **Investigate eye-specific issues** - why 2.38px when other regions are <1.2px
3. **Parameter tuning** - test different regularization values
4. **Verify on multiple frames** - ensure consistency

---

## Investigation Notes

### Why removing WS=5 broke accuracy
- sigma adjustment formula: `sigma = 1.75 + 0.25 * log2(scale_max/0.25)`
- With scale_max=1.0: sigma=2.25
- With scale_max=0.5: sigma=2.0
- The 0.25 sigma difference significantly affects KDE mean-shift

### The base_landmarks_nonrigid fix
- Documentation said use `landmarks_2d_initial`
- Code had been changed to use rigid-updated params
- Using initial landmarks allows larger mean-shift vectors in non-rigid phase
- This gives the optimizer more room to refine positions

---

## Files Modified

1. `pyclnf/core/optimizer.py`
   - Line 107: Added `min_iterations` parameter
   - Line 241: RIGID convergence check with min_iterations
   - Line 327: NON-RIGID convergence check with min_iterations
   - Line 296: `base_landmarks_nonrigid = landmarks_2d_initial.copy()`

2. `pyclnf/clnf.py`
   - Line 66: Added `min_iterations` parameter
   - Line 148: Pass min_iterations to optimizer

---

## References

- `CLNF_CONVERGENCE_INVESTIGATION.md` - Original investigation document
- C++ OpenFace accuracy: ~0.67px
- Target accuracy: <0.5px
