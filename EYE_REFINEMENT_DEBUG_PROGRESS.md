# Eye Refinement Debugging Progress

## Goal
Match C++ OpenFace eye landmark refinement quality (<0.5px error)

## Current Status
- **Landmark 36 error**: 1.45px (improved from 2.38px)
- **Overall**: Eye refinement now IMPROVES results (was making them worse)
- **Remaining issue**: X direction movement still inverted

---

## Completed Fixes

### 1. uint8 to float32 Bug (eye_patch_expert.py:1007)
- **Issue**: Image patches were uint8 instead of float32
- **Impact**: Response values were quantized
- **Status**: FIXED

### 2. Removed Incorrect 0.5 Damping (eye_patch_expert.py:1279, 1334)
- **Issue**: Python had 0.5 damping, C++ has none in eye model
- **Impact**: Under-stepping by 50%
- **Status**: FIXED

### 3. Added PDM Re-fitting (clnf.py:340-343)
- **Issue**: C++ calls CalcParams + CalcShape2D after eye refinement
- **Impact**: Missing regularization through main model shape constraints
- **Status**: FIXED - Implemented full fit_to_landmarks in pdm.py

---

## Verified Components (Match C++ within tolerance)

| Component | Tolerance | Verified |
|-----------|-----------|----------|
| CCNF Response | 0.6% | Yes |
| KDE Mean-shift | 0.2% | Yes |
| sim_ref_to_img transform | 0.08% | Yes |
| Raw mean-shift (eye lm8) | <3% | Yes |
| Parameter update (iter 0) | ~6% | Yes |

---

## Current Discrepancy Analysis

### Landmark 36 Movement
- **C++ moved**: (-1.019, 3.862)
- **Python moved**: (-0.707, 3.604) [after fit_to_landmarks]
- **Ratios**: X: 0.69x, Y: 0.93x

### The X Direction Problem
Python consistently moves landmark 36 in the **wrong X direction**:
- Needed: +0.377 px (rightward)
- Python: -0.707 px (leftward)
- Ratio: -1.87x

This suggests the mean-shifts are pointing the wrong way for some landmarks.

---

## Debugging Strategy

### Phase 1: Verify Initial Eye Landmark Positions
**Goal**: Ensure eye model starts from same positions as C++

1. Compare initial eye_8 position:
   - C++: (391.3150, 827.5953)
   - Python: (391.3010, 827.5868)
   - Diff: (-0.014, -0.009) - **OK, very close**

2. Check all 28 eye landmarks initial positions match

### Phase 2: Compare Response Maps Side-by-Side
**Goal**: Find if peak positions differ

For Eye_8 (left eye outer corner):
- C++ response 3x3: Need to extract and compare
- Python response 3x3:
  ```
  0.1737 0.1316 0.1265
  0.3617 0.3565 0.1579
  0.4673 0.5035 0.2382  <- Peak at (2,1)
  ```

**Key Question**: Is the C++ peak also at (2,1)?

### Phase 3: Track Mean-Shift Direction Per Landmark
**Goal**: Find which landmarks have inverted mean-shifts

Compare RIGID iteration 0 mean-shifts for all eyelid landmarks:
```
Eye_8 (lm36):  Python (-0.410, 0.918) vs C++ (-0.399, 0.941)  <- Close!
Eye_10 (lm37): Compare
Eye_12 (lm38): Compare
Eye_14 (lm39): Compare
Eye_16 (lm40): Compare
Eye_18 (lm41): Compare
```

If individual mean-shifts match but cumulative result is wrong, the issue is in:
- Jacobian computation
- Parameter update composition

### Phase 4: Compare Jacobian Matrices
**Goal**: Verify J matrix matches C++

The rigid Jacobian columns are:
1. Scale
2-4. Rotation (wx, wy, wz)
5-6. Translation (tx, ty)

Log J @ mean_shift for first iteration and compare.

### Phase 5: Compare Parameter Updates Step-by-Step
**Goal**: Track divergence iteration-by-iteration

For window_size=3, compare after each of 5 RIGID iterations:
```
Iter 0: C++ params vs Python params
Iter 1: C++ params vs Python params
...
```

The divergence likely compounds, so find when it starts.

### Phase 6: Verify fit_to_landmarks Convergence
**Goal**: Ensure the new CalcParams equivalent works correctly

Check:
- Does error decrease each iteration?
- Does it converge within reasonable iterations?
- Are final params close to C++ final params?

---

## Likely Root Causes (Prioritized)

### 1. **Jacobian Rotation Derivative Signs** (HIGH)
The Jacobian for rotation parameters involves cross products. Any sign error would cause mean-shifts to point wrong direction for rotation-dependent movements.

**Action**: Compare J[:,1:4] (rotation columns) element-by-element with C++

### 2. **Rotation Composition in update_params** (MEDIUM)
The small-angle rotation matrix R2 construction might have wrong signs.

**Action**: Verify R2 matrix matches C++ comment format

### 3. **Cumulative Error in Initial Landmark Projection** (MEDIUM)
The eye_landmarks start from main model output. Small errors compound through eye refinement.

**Action**: Check if C++ uses exactly same initial landmarks

### 4. **Window Size Progression** (LOW)
Python uses [3, 5], C++ might use [3, 5, 9] or different order.

**Action**: Verify window_sizes match C++ hierarchical_params

---

## Debug Output Files

| File | Contents |
|------|----------|
| /tmp/cpp_eye_model_detailed.txt | C++ eye model iterations |
| /tmp/python_eye_model_detailed.txt | Python eye model iterations |
| /tmp/cpp_eye_lm8_params.txt | C++ landmark 8 debug |
| /tmp/cpp_eye_lm8_meanshift.txt | C++ mean-shift KDE values |
| /tmp/python_eye_rigid_iter0.txt | Python RIGID iter 0 details |
| /tmp/python_eye_raw_meanshift.txt | Python raw mean-shifts |

---

## Next Steps

1. **Add Jacobian logging** to Python RIGID iteration 0
2. **Extract C++ Jacobian** for same iteration
3. **Compare column-by-column** to find sign or formula errors
4. **Test with single iteration** to isolate per-step error vs cumulative

---

## Success Criteria

- [ ] Eye refinement reduces error for ALL eye landmarks (36-47)
- [ ] Mean error < 1.0px
- [ ] X and Y movement directions match C++
- [ ] Movement magnitudes within 10% of C++
