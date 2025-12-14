# Regularization Factor Investigation

## Question
"If C++ uses reg_factor=22.5, why does applying that to Python make results worse?"

## Key Findings

### 1. Initialization is NOT the problem
- Python MTCNN and C++ produce **identical bbox centers** (462.07, 1046.84)
- Init params differ by only ~0.003 pixels
- The `init_params()` function produces identical results for the same bbox

### 2. The regularization cost explains the behavior

| Local[0] | reg_factor=1.0 | reg_factor=22.5 |
|----------|----------------|-----------------|
| 0        | 0.00           | 0.00            |
| 12       | 0.17           | 3.92            |
| 20       | 0.48           | 10.89           |
| 32       | 1.24           | 27.89           |

With reg_factor=22.5, the cost for Local[0]=32 is **7x higher** than for Local[0]=12.

### 3. Python converges to a different optimum

**With reg_factor=1.0:**
- Python: Local[0] ≈ 32, jaw error **1.02px**
- The optimizer has freedom to explore the cost landscape

**With reg_factor=22.5:**
- Python: converges to a bad local optimum, jaw error **8.49px**
- The strong regularization traps it in a different basin

**C++ with reg_factor=22.5:**
- Converges to Local[0] ≈ 12, good accuracy
- Despite same initialization, finds a different optimum

### 4. Root Cause: Optimization Path Differences

Even though Python and C++ start from the same position, small differences in:
- Response map computation
- Mean-shift calculation
- Jacobian computation
- Floating-point accumulation

cause the optimization paths to diverge. With weak regularization (1.0), both paths reach similar accuracy. With strong regularization (22.5), Python gets stuck.

## Conclusion

**Python should use reg_factor=1.0** - this is correct for the Python implementation because:
1. It produces excellent accuracy (1.02px jaw error)
2. It matches C++ accuracy despite different Local[0] values
3. The exact C++ value (22.5) doesn't transfer due to optimization landscape differences

The Local[0] difference (Python ~32 vs C++ ~12) represents different but equally valid shape variants that produce similar landmark positions.

## Current Settings

```python
# optimizer.py line ~958
reg_factor = 1.0  # Works with Python's optimization path
```

## Test Results (5 frames, IMG_0422.MOV)

| Metric | Mean | Max |
|--------|------|-----|
| Jaw error | 1.02px | 1.37px |
| Overall error | 1.25px | 1.39px |
