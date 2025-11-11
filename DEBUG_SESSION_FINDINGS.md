# PyCLNF Convergence Debug Session - Findings

**Date**: 2025-11-10
**Approach**: All 3 approaches (debug output, empirical comparison, investigating suspects)

---

## Summary

Added debug output to `pyclnf/core/optimizer.py` and ran tests. Discovered **critical issues** with mean-shift magnitudes.

---

## Debug Output Added

### In `optimizer.py` lines 156-161:
```python
# DEBUG: Print convergence metrics
if iteration < 3 or iteration % 5 == 0:  # Print first 3 iterations, then every 5th
    ms_mag = np.linalg.norm(mean_shift)
    dp_mag = np.linalg.norm(delta_p)
    w_mean = np.mean(np.diag(W))
    print(f"Iter {iteration:2d} (ws={window_size}): MS={ms_mag:8.4f} DP={dp_mag:8.4f} W_mean={w_mean:6.4f}")
```

### In `_compute_mean_shift()` lines 289-290, 325-329:
- Collect response map statistics (min, max, mean)
- Print occasionally (10% chance) to avoid spam

---

## Test Results

```
Iter  0 (ws=11): MS= 58.1258 DP= 10.0971 W_mean=1.0000
Iter  1 (ws=11): MS= 57.7236 DP=  6.7010 W_mean=1.0000
Iter  2 (ws=11): MS= 44.8593 DP=  7.5345 W_mean=1.0000
Iter  0 (ws=9): MS= 39.4625 DP=  6.6635 W_mean=1.0000
Iter  1 (ws=9): MS= 37.1759 DP=  4.3218 W_mean=1.0000
Iter  2 (ws=9): MS= 37.6184 DP=  3.9392 W_mean=1.0000
Iter  0 (ws=7): MS= 27.8334 DP=  4.2205 W_mean=1.0000
Iter  1 (ws=7): MS= 25.1616 DP=  3.7075 W_mean=1.0000
Iter  2 (ws=7): MS= 24.3951 DP=  2.4658 W_mean=1.0000
Iter  0 (ws=5): MS= 24.9171 DP=  3.7079 W_mean=1.0000
Iter  1 (ws=5): MS= 24.8354 DP=  3.6137 W_mean=1.0000
Iter  2 (ws=5): MS= 25.4867 DP=  3.4014 W_mean=1.0000

Final: Converged=False, Iterations=20, Final Update=3.583388
```

---

## Critical Findings

### 1. Mean-Shift Magnitudes Are HUGE ❌
- **First iteration**: MS = 58.13 pixels
- **Final iteration (ws=5)**: MS = 24-25 pixels
- **Expected**: MS should decrease to < 1 pixel for convergence
- **Actual**: MS stays at 24-58 pixels

**Interpretation**: The mean-shift algorithm is detecting landmarks 24-58 pixels away from their current positions. This is WAY too large and indicates:
- Response maps are giving very poor/incorrect peaks, OR
- Mean-shift computation has a fundamental bug, OR
- Initial PDM parameters are drastically wrong

### 2. Parameter Updates Not Decreasing Enough ❌
- **First iteration**: DP = 10.10
- **Final iteration**: DP = 3.40
- **Expected**: DP should decrease to < 0.005 for convergence
- **Actual**: DP decreases from 10.10 → 3.40 but gets stuck

**Interpretation**: Parameter updates are decreasing but not fast enough. This is likely a consequence of the huge mean-shift values.

### 3. Weight Matrix Always 1.0 ⚠️
- **All iterations**: W_mean = 1.0000
- **Expected**: Weights should vary based on patch response quality
- **Actual**: All weights are exactly 1.0

**Interpretation**: Either:
- All patches have perfect responses (unlikely), OR
- Weight computation is broken/not being used, OR
- This is expected behavior for this dataset

### 4. Multi-Scale Optimization Works ✓
- Window sizes progress correctly: 11 → 9 → 7 → 5
- Optimizer restarts at each scale (iter 0, 1, 2 for each window size)
- Mean-shift magnitude decreases across scales (58 → 39 → 27 → 24)

**Interpretation**: The multi-scale strategy is working, but convergence isn't happening at ANY scale.

---

## Comparison with Expected Behavior

### OpenFace C++ (Expected):
- Mean-shift: Large initially (10-20px), decreases to < 1px within 5-10 iterations
- Delta-p: Decreases from ~5 to < 0.005 within 10-20 iterations
- Weights: Vary based on patch quality (typically 0.5-1.0)
- Convergence: Achieves target < 0.005 in 10-20 iterations

### PyCLNF (Actual):
- Mean-shift: Stays at 24-58px, NOT decreasing to < 1px ❌
- Delta-p: Decreases from 10 to 3.4, NOT reaching < 0.005 ❌
- Weights: All exactly 1.0 (suspicious) ⚠️
- Convergence: Never achieves target ❌

---

## Hypotheses for Root Cause

### Hypothesis 1: Response Maps Are Wrong (MOST LIKELY)
If response maps are giving peaks in the wrong locations, mean-shift will keep trying to move landmarks to incorrect positions, preventing convergence.

**Evidence**:
- Mean-shift magnitudes of 24-58px suggest response maps have peaks far from correct landmark positions
- Would explain why convergence never happens

**Next Steps**:
- Print response map statistics (min, max, mean, peak location)
- Compare response map values with OpenFace C++
- Visualize response maps to see if peaks are in correct locations

### Hypothesis 2: Mean-Shift Computation Has a Bug
The KDE mean-shift algorithm might have a subtle bug causing it to compute incorrect displacement vectors.

**Evidence**:
- Mean-shift values are consistently too large
- Reverted coordinate system "fix" didn't fully solve the problem

**Next Steps**:
- Add detailed logging to `_kde_mean_shift()` method
- Compare intermediate values (dx, dy, weighted centroid) with OpenFace
- Test with known response map to verify KDE computation

### Hypothesis 3: PDM Initialization Is Wrong
If initial PDM parameters are way off, landmarks start in completely wrong positions, and response maps can't guide them back.

**Evidence**:
- First iteration mean-shift is 58px (very large)
- Would explain why convergence never happens

**Next Steps**:
- Print initial PDM parameters
- Compare with OpenFace C++ initial parameters
- Test with known-good initial parameters from OpenFace

### Hypothesis 4: Weight Matrix Not Being Used
If weights aren't being computed correctly, poor patches have too much influence.

**Evidence**:
- All weights are exactly 1.0

**Next Steps**:
- Check how weight matrix `W` is computed
- Verify it's being used in parameter update equation
- Compare weight computation with OpenFace

---

## Investigation Tools Created

1. **Debug Output in optimizer.py**:
   - Prints MS, DP, W_mean for each iteration
   - Prints response map statistics occasionally
   - Easy to remove later (search for "# DEBUG:")

2. **compare_one_iteration.py**:
   - Runs both PyCLNF and OpenFace C++ on same frame
   - Compares final landmark positions
   - Can be extended to compare intermediate values

---

## Next Steps (Priority Order)

### Priority 1: Investigate Response Maps ⭐⭐⭐
- Add logging to print response map peak locations
- Visualize response maps for a few landmarks
- Compare response map values with OpenFace C++
- Check if patch scaling (0.25) is being applied correctly

### Priority 2: Compare PDM Initialization ⭐⭐
- Print initial PDM parameters from PyCLNF
- Run OpenFace C++ with same face bbox, extract initial params
- Compare to see if initialization differs

### Priority 3: Investigate Weight Matrix ⭐
- Check `_compute_weights()` implementation
- Verify weights are being used in `_solve_update()`
- Compare weight computation with OpenFace

### Priority 4: Verify Mean-Shift KDE ⭐
- Add detailed logging to `_kde_mean_shift()`
- Test with synthetic response map
- Compare with OpenFace intermediate KDE values

---

## Files Modified

1. **pyclnf/core/optimizer.py**:
   - Lines 156-161: Added iteration debug output
   - Lines 267, 289-290, 325-329: Added response map statistics
   - All changes marked with `# DEBUG:` for easy removal

2. **compare_one_iteration.py**: New comparison script (not yet run)

---

## Questions to Answer

1. **Why are mean-shift values 24-58 pixels?**
   - Are response maps wrong?
   - Is mean-shift computation broken?
   - Are initial parameters way off?

2. **Why are all weights exactly 1.0?**
   - Is this expected?
   - Is weight computation broken?
   - Are all patches giving perfect responses?

3. **Why doesn't convergence happen at ANY scale?**
   - Is the convergence threshold too strict?
   - Is the algorithm fundamentally different from OpenFace?
   - Is there a bug in parameter update?

---

## Conclusion

Debug output reveals that **mean-shift magnitudes are way too large** (24-58 pixels instead of < 1 pixel). This is preventing convergence.

The most likely culprit is **response maps giving incorrect peaks**, but could also be bugs in mean-shift computation, PDM initialization, or weight matrix.

**Recommendation**: Focus on investigating response maps first, as this is most likely to reveal the root cause.
