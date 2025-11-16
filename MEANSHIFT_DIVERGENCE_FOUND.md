# Mean-Shift Divergence Found! 🎯

## Critical Discovery

**The divergence occurs in the MEAN-SHIFT COMPUTATION at ITER0!**

Both Python and C++ start with IDENTICAL landmark positions, but compute DIFFERENT mean-shift vectors.

## Comparison at ITER0, Window Size 11

### Python Mean-Shift Vectors:
```
Landmark_36: ms=(-8.3294, -2.0448) mag=8.5768
Landmark_48: ms=(-4.0023,  2.2635) mag=4.5980
Landmark_30: ms=(-17.4900, 1.9149) mag=17.5946
Landmark_8:  ms=(-4.0337, -3.7181) mag=5.4859
```

### C++ Mean-Shift Vectors:
```
Landmark_36: ms=(-11.955,  3.5888) mag=12.482
Landmark_48: ms=(-1.26259, 4.78668) mag=4.9504
Landmark_30: ms=(-18.397,  2.94177) mag=18.6308
Landmark_8:  ms=(-6.31102, -7.14387) mag=9.53225
```

### Differences:

| Landmark | ΔX | ΔY | Δ Magnitude |
|----------|----|----|-------------|
| 36 | +3.626 | -5.633 | +3.905 (45% error!) |
| 48 | -2.740 | -2.523 | +0.352 (7% error) |
| 30 | +0.907 | -1.027 | +0.637 (4% error) |
| 8 | +2.277 | +3.426 | +4.044 (74% error!) |

**Landmark 36 and Landmark 8 have MASSIVE mean-shift errors!**

## Root Cause Analysis

The mean-shift computation has two main components:

1. **Response Map Computation**
   - Patch expert evaluation at current landmark position
   - Warping to reference coordinates (if enabled)
   - Sigma component weighting (if enabled)

2. **KDE Mean-Shift Calculation**
   - Weighted average using Gaussian kernel
   - Computing centroid of response distribution

The divergence must be in one of these two components.

## Next Steps

1. **Compare response maps** - Check if Python and C++ produce identical response maps for landmark 36
2. **Compare KDE calculation** - If response maps match, the problem is in the KDE mean-shift algorithm
3. **Check sigma components** - Verify sigma weighting is applied correctly

## Hypothesis

Most likely culprit: **Response map computation differs** between Python and C++.

Possible causes:
- Different patch expert evaluation
- Different warping/interpolation
- Different sigma component application
- Different normalization

Let's compare the actual response maps to confirm!
