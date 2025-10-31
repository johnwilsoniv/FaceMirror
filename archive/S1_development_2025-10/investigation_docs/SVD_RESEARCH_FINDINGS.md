# SVD Implementation Differences: Research Findings

## Summary of Key Findings

Based on research into numpy.linalg.svd vs OpenCV cv::SVD, here are the critical differences that could explain our rotation discrepancies.

---

## 1. Algorithm Differences

### NumPy (numpy.linalg.svd)
- **Backend:** LAPACK's xGESDD (divide-and-conquer algorithm)
- **Speed:** Fast
- **Accuracy:** Standard precision
- **Best for:** General-purpose matrices

### OpenCV (cv::SVD)
- **Backend:** Custom Jacobi-based implementation (since OpenCV 2.3+)
  - OpenCV 2.2 used LAPACK (fast)
  - OpenCV 2.3+ uses custom JacobiSVDImpl (slower but sometimes more accurate)
- **Speed:** Several times slower than LAPACK methods
- **Accuracy:** Higher accuracy for well-conditioned matrices
- **Best for:** Matrices expressible as A = B × D (B well-conditioned, D diagonal)

**CRITICAL:** Different algorithms can produce systematically different results!

---

## 2. Sign Ambiguity in SVD

### The Problem
**SVD decomposition is only defined up to a sign:**
- U and V can have arbitrary column signs
- If `A = U × S × V^T`, then also `A = (-U) × S × (-V)^T`
- Different implementations may choose different signs

### Impact on Kabsch Algorithm
```
Rotation = V^T × correction × U^T
```

If U or V have flipped signs, the rotation matrix will be different!

**Example:**
- If V's second column is flipped: rotation could differ by 180°
- If U's first column is flipped: rotation direction could reverse
- If multiple columns differ: arbitrary rotation difference!

### Determinant Check in Kabsch
Both implementations check determinant to prevent reflection:
```python
d = np.linalg.det(Vt.T @ U.T)
if d > 0:
    corr = [[1, 0], [0, 1]]
else:
    corr = [[1, 0], [0, -1]]
```

**But this only catches reflections, not sign differences in U/V!**

---

## 3. Numerical Precision Differences

### Float32 vs Float64
- **OpenCV:** Typically uses `float32` throughout
- **NumPy:** Defaults to `float64`, but we convert to `float32`

**Impact:**
- Small precision differences accumulate through matrix operations
- Could cause ~0.1-1° differences
- Unlikely to cause 5-30° differences (but possible!)

### Precision Characteristics
- Singular value differences: ~10^-6 to 10^-7
- Singular vector differences: Larger, especially for:
  - Matrices with small singular values
  - Matrices with repeated singular values
  - Ill-conditioned matrices

---

## 4. Specific Issues Found in Research

### Issue 1: Sign Inconsistency in U and V
> "Almost all entries in U and V matrices can have different signs between implementations"

**This could be our problem!**
- If numpy and OpenCV choose different signs for U/V columns
- The rotation matrix will be systematically different
- Could easily cause 5-30° rotation differences

### Issue 2: Point-Set Dependent Behavior
> "Rotation angle 'fixup' differs based on input point-sets, sometimes requiring corrections of 180°, 360°, or even 180+angle"

**This could explain:**
- Why with-eyes (24 points) gives -5°
- Why without-eyes (16 points) gives +30°
- Different point sets → different SVD signs → different rotations

### Issue 3: Covariance Matrix Construction
> "Problem may be in the construction of the covariance matrix"

In Kabsch, we compute: `H = src.T @ dst`

**Could:**
- Matrix operation order matter?
- Transpose operations differ?
- @ operator vs * operator produce different results?

---

## 5. Why This Could Explain Everything

### Our Observations Match SVD Sign Issues

**Observation 1:** Python gives consistent -5° with eyes
- → SVD signs are consistent for this point set
- → But different from C++ SVD signs

**Observation 2:** Python gives consistent +30° without eyes
- → SVD signs are consistent for this point set
- → But different from both C++ and with-eyes

**Observation 3:** Both have good stability within their configuration
- → Each configuration produces consistent SVD signs
- → The difference is between configurations, not within

**Observation 4:** Expression changes affect with-eyes but not without-eyes
- → Eye landmarks moving changes the matrix
- → Changes which sign choices SVD makes
- → Without eyes, matrix is more stable → consistent signs

---

## 6. Specific Hypothesis

### The Root Cause:
**numpy and OpenCV make different sign choices in SVD decomposition**

When computing `SVD(src.T @ dst)`:
1. Both get correct singular values (S)
2. Both get mathematically valid U and V
3. But U and V have **arbitrary column signs**
4. numpy chooses one set of signs
5. OpenCV chooses different signs
6. This propagates through Kabsch → different rotation matrices!

### Why It's Stable Within Each Configuration:
- For a given point set, SVD signs are deterministic
- Same input → same SVD output from same library
- So each approach is internally consistent
- But cross-library comparison fails

### Why Different Point Sets Give Different Results:
- 24 points (with eyes) → Matrix M1 → SVD chooses signs1 → -5° rotation
- 16 points (without eyes) → Matrix M2 → SVD chooses signs2 → +30° rotation
- Same library, different inputs, different sign choices!

---

## 7. How to Investigate This

### Test 1: Compare SVD Output Directly

```python
# Create the SAME test matrix
src = landmarks_68[RIGID_INDICES]  # Extract rigid points
dst = reference_shape[RIGID_INDICES]

# Compute covariance matrix
H = src.T @ dst

# Python SVD
U_py, S_py, Vt_py = np.linalg.svd(H)

# Would need C++ SVD output to compare:
# U_cpp, S_cpp, Vt_cpp = cv::SVD(H)

# Compare:
# S values should be nearly identical
# U and V might have sign differences!
```

### Test 2: Force Consistent Signs

Some approaches to fix sign ambiguity:
1. **Largest magnitude convention:** Force largest element in each column to be positive
2. **Determinant sign:** Ensure det(U) > 0 and det(V) > 0
3. **Reference-based:** Choose signs based on dot product with reference

**Try in Python:**
```python
def fix_svd_signs(U, Vt):
    """Force consistent sign convention"""
    # Make largest element in each column positive
    for i in range(U.shape[1]):
        if np.abs(U[:, i]).max() == -U[:, i].min():
            U[:, i] *= -1
            Vt[i, :] *= -1
    return U, Vt
```

### Test 3: Match OpenCV's SVD Exactly

Could we use OpenCV's SVD in Python?
```python
import cv2

# Convert to OpenCV format
H_cv = cv2.Mat(H)

# Use OpenCV SVD
w, u, vt = cv2.SVDecomp(H_cv)

# This should match C++ OpenCV exactly!
```

---

## 8. What This Means for Our Solution

### If Sign Ambiguity Is the Problem:

**Option A:** Use OpenCV SVD in Python
- Import cv2
- Use cv2.SVDecomp instead of np.linalg.svd
- Should match C++ exactly!

**Option B:** Implement Sign Normalization
- Add sign fixing after SVD
- Choose same convention as OpenCV
- May need trial and error

**Option C:** Learn the Offset
- If Python consistently differs by X degrees
- Apply -X degree correction
- We know: -5° for with-eyes, -30° for without-eyes

---

## 9. Next Investigative Steps

### Step 1: Test OpenCV SVD in Python (CRITICAL)
Replace:
```python
U, S, Vt = np.linalg.svd(src.T @ dst)
```

With:
```python
w, u, vt = cv2.SVDecomp(src.T @ dst)
```

**Hypothesis:** This will match C++ rotation exactly!

### Step 2: Compare SVD Outputs
Print U, S, Vt from both numpy and OpenCV for same input
- Are singular values identical?
- Do U/V have sign differences?
- Which columns differ?

### Step 3: Test Sign Normalization
Implement consistent sign convention
- Test if it improves stability
- Test if it matches C++ better

---

## 10. Confidence Level

**Likelihood this is the root cause: HIGH (70-80%)**

Reasons:
1. ✓ Explains different rotations for different point sets
2. ✓ Explains consistency within each approach
3. ✓ Explains expression sensitivity (changes matrix → changes signs)
4. ✓ Well-documented issue with SVD implementations
5. ✓ Easy to test (use OpenCV SVD in Python)

**If we use cv2.SVDecomp in Python and it matches C++, we've found the answer!**

---

## Summary

**Key Finding:** SVD sign ambiguity is likely the root cause.

**Evidence:**
- Different SVD implementations choose different column signs for U and V
- This propagates through Kabsch algorithm
- Results in systematically different rotation matrices
- Explains all our observations

**Test:** Replace numpy SVD with OpenCV SVD in Python

**If it works:** Problem solved! Pure Python solution using cv2.SVDecomp

**If it doesn't:** Need to investigate other causes from systematic analysis
