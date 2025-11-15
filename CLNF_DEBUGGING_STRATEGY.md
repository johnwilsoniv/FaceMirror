# CLNF Debugging Strategy: Python 9.646px vs C++ Ground Truth

## Current Status

### Initialization (PDM from bbox)
- **C++ init error**: 18.5px (vs C++ final)
- **Python init error**: 16.8px (vs C++ final) ✓ Better than C++!
- **Bbox difference**: ~4-6px (negligible)

### After CLNF Refinement
- **C++ final error**: 0px (ground truth reference)
- **Python final error**: 9.646px (8.4px for first 3 frames)
- **C++ CLNF improvement**: 18.5px (converged fully)
- **Python CLNF improvement**: 8.4px (only 50% of the way)

### Convergence
- **C++**: Converges (hits threshold)
- **Python**: Never converges (0% convergence, always hits max iterations)
- **Python parameter updates (DP)**: 3-8px (vs 0.01px threshold)

---

## Debugging Hypothesis

**Since Python initialization is BETTER than C++, but C++ final is much better, the issue is in the CLNF optimization loop, not in the initialization.**

The CLNF optimization loop consists of:
1. **Patch Expert Evaluation** (response maps)
2. **Mean-shift computation** (finding optimal landmark positions)
3. **Parameter update** (NU-RLMS optimizer)
4. **Regularization** (PDM shape constraint)

---

## Debugging Plan

### Phase 1: Verify Patch Expert Response Maps
**Goal**: Ensure Python and C++ patch experts produce similar response maps

**Test**: Compare response map values for the same landmark position
- Extract C++ response maps (may need to add debug output to OpenFace)
- Extract Python response maps (already have debug output)
- Compare:
  - Response map dimensions
  - Response map value ranges (min, max, mean)
  - Peak locations
  - Peak values

**Expected**: Response maps should be nearly identical if patch experts loaded correctly

**Files to check**:
- `pyclnf/core/patch_expert.py` - Response map computation
- `pyclnf/core/optimizer.py:267-268` - Debug logging for response maps

### Phase 2: Verify Mean-Shift Computation
**Goal**: Ensure mean-shift correctly finds response map peaks

**Test**: For each iteration, compare:
- Mean-shift magnitude (MS values in debug output)
- Direction of shift
- Peak detection method

**Python debug output shows**:
```
Iter  0 (ws=11): MS= 63.0717 DP= 13.4618
Iter  1 (ws=11): MS= 48.8527 DP=  8.6780
```

**Expected**: Mean-shift should decrease monotonically and converge

**Files to check**:
- `pyclnf/core/optimizer.py` - Mean-shift implementation
- Look for peak finding logic

### Phase 3: Verify Parameter Update (NU-RLMS)
**Goal**: Ensure optimizer correctly updates PDM parameters

**Test**: Compare parameter update equations
- Check Hessian computation
- Check gradient computation
- Check regularization application
- Check step size/learning rate

**Key parameters**:
- Regularization: r=25.0 (current) vs r=1.0 (previous)
- Convergence threshold: 0.005 (current) vs 0.01 (previous)

**Files to check**:
- `pyclnf/core/optimizer.py` - Full NU-RLMS implementation
- Compare against OpenFace C++ code: `LandmarkDetectorUtils/PDM.cpp`

### Phase 4: Verify Window Size Strategy
**Goal**: Ensure correct window sizes and iteration distribution

**Current setup**:
- Window sizes: [11, 9, 7] (after removing ws=5)
- Iterations: 10 total (distributed across window sizes)

**Test**:
- Verify C++ uses same window sizes
- Verify iteration distribution matches C++
- Check if window size affects convergence

**Files to check**:
- `pyclnf/clnf.py:58` - Window sizes parameter
- `pyclnf/core/optimizer.py` - Window size iteration logic

### Phase 5: Check Image Preprocessing
**Goal**: Ensure image passed to patch experts is preprocessed identically

**Test**: Compare:
- Image normalization
- Image warping/cropping
- Grayscale conversion
- Any Gaussian blur or filtering

**Files to check**:
- `pyclnf/clnf.py:92-96` - Grayscale conversion
- Patch expert preprocessing

### Phase 6: Verify Regularization Impact
**Goal**: Determine if regularization is too strict

**Test**: Run validation with different regularization values
- r=1.0 (previous, gave 10.898px)
- r=25.0 (current, gives 9.646px)
- r=10.0 (middle ground)
- r=0.1 (very loose)

**Expected**: Lower r should allow more flexibility but may overfit

---

## Immediate Next Steps (Prioritized)

### Step 1: Extract C++ CLNF Debug Info ⭐ CRITICAL
**Action**: Modify C++ OpenFace to output CLNF iteration details

Add to `FeatureExtraction` or create standalone test:
```cpp
// For each CLNF iteration:
printf("Iter %d (ws=%d): MS=%.4f DP=%.4f\n", iter, window_size, mean_shift, param_delta);

// For response maps:
printf("Response map: min=%.6f max=%.6f mean=%.6f\n", min_val, max_val, mean_val);
```

**Why critical**: Without C++ iteration data, we're debugging blind

### Step 2: Compare Response Map Statistics
**Action**: Run Python and C++ on same frame, compare response map stats

**Script**: Create `compare_response_maps.py`
- Load same image + bbox for both
- Run first CLNF iteration only
- Compare response map values landmark-by-landmark

### Step 3: Parameter Update Analysis
**Action**: Check if parameter updates are in the right direction

**Script**: Create `analyze_parameter_updates.py`
- Track parameter values across iterations
- Verify they're moving toward better fit (lower error)
- Check for oscillation or divergence

### Step 4: Convergence Threshold Investigation
**Action**: Test if threshold is too strict

**Quick test**: Change convergence threshold from 0.01 to 1.0
- If it now converges but accuracy doesn't improve → threshold issue
- If it still doesn't converge → optimizer issue

---

## Success Criteria

**Minimum acceptable**: 6.5px final error (close to Nov 14's 6.17px)
**Stretch goal**: <5px final error (approaching C++ accuracy)
**Convergence goal**: >50% of frames converge before max iterations

---

## Risk Areas

### High Risk (Most likely causes):
1. **Response map computation** - Patch expert evaluation differences
2. **NU-RLMS optimizer** - Parameter update equation bugs
3. **Regularization** - Too strict or incorrect application

### Medium Risk:
4. **Window size strategy** - Incorrect iteration distribution
5. **Mean-shift** - Peak finding differences
6. **Image preprocessing** - Subtle differences in input

### Low Risk (Unlikely given good initialization):
7. **PDM** - Shape model issues
8. **Bbox** - Detection differences

---

## Tools Needed

1. **C++ Debug Builds**: OpenFace with CLNF debug output
2. **Comparison Scripts**: Python scripts to extract and compare intermediate values
3. **Visualization**: Plot response maps, parameter evolution, error curves
4. **Statistical Analysis**: Compare distributions of response values

---

## Timeline Estimate

- **Phase 1 (Response maps)**: 2-3 hours
- **Phase 2 (Mean-shift)**: 1-2 hours
- **Phase 3 (NU-RLMS)**: 3-4 hours (most complex)
- **Phase 4 (Window sizes)**: 1 hour
- **Phase 5 (Preprocessing)**: 1-2 hours
- **Phase 6 (Regularization)**: 1 hour

**Total**: 9-13 hours of focused debugging

---

## Notes

- Python init is better than C++ init (16.8 vs 18.5px) → **PDM and bbox are working correctly** ✓
- Python improves by 8.4px, C++ improves by 18.5px → **Optimization loop difference**
- Python never converges → **Likely parameter update issue or threshold issue**
- Current 9.646px is clinically acceptable but we can do better
