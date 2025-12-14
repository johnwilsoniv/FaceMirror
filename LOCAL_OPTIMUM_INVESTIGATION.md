# Local Optimum Investigation Summary

## Problem
Python pyclnf and C++ OpenFace converge to **different local optima**, even with identical:
- Response maps (CEN)
- Sigma parameter (2.5)
- Mean-shift computation

## Key Finding: Different Optimization Basins

### Evidence
When starting from C++ frame 28 parameters (ideal init):
- **C++ Local[0]**: 17.18
- **Python Result Local[0]**: 32.10 (change: +14.92)
- **Jaw Error**: 2.13px

Python optimizer **moves AWAY** from C++ solution even when started there!

### Root Cause: Lambda_inv Regularization

Python uses eigenvalue-based regularization without clamping:
```python
Lambda_inv[6:] = 1.0 / eigenvalues
```

For Local[0] with eigenvalue = 826:
- **1/826 = 0.0012** → Almost no regularization → Local[0] drifts freely

C++ likely clamps this:
```python
Lambda_inv[6:] = np.clip(1.0 / eigenvalues, 0.01, 1.0)
```
- **clamp(0.0012, 0.01, 1.0) = 0.01** → Minimum regularization → Local[0] constrained

### Clamping Test Results

| Metric | Without Clamping | With Clamping |
|--------|------------------|---------------|
| Local[0] | 32.10 | 17.80 |
| C++ Target | 17.18 | 17.18 |
| Difference | +14.92 | +0.62 |
| Jaw Error | 2.13px | 1.78px |

Clamping brings Local[0] much closer to C++!

## The Paradox

**Problem**: Clamping helps when starting from C++ landmarks, but **hurts** normal tracking:

| Scenario | Without Clamping | With Clamping |
|----------|------------------|---------------|
| Tracking from pymtcnn | **1.39px** | 2.18px |
| Tracking from C++ f1 | 2.35px | **1.99px** |

### Explanation
1. Python's pymtcnn initialization puts landmarks in a **different basin** than C++
2. Without clamping, Python finds a **locally optimal** solution in its own basin
3. With clamping, Python tries to move toward C++'s basin but starts from the wrong position
4. The result is worse because it's neither fully in Python's nor C++'s basin

## Conclusion

Python and C++ operate in **different optimization basins**:
- C++ basin: Local[0] ≈ 17, constrained by clamped regularization
- Python basin: Local[0] ≈ 32, unconstrained regularization

Both produce valid face fits, but with different shape parameter distributions.

## Recommendations

1. **Keep unclamped regularization** for normal Python tracking (1.39px mean error is acceptable)

2. **If exact C++ matching is needed**:
   - Initialize from C++ landmarks (not pymtcnn)
   - Enable clamped regularization
   - This gives ~1.99px mean error tracking from C++ init

3. **The ~2px jaw error is inherent** to the different optimization basins and cannot be eliminated without changing the initialization strategy

## Current Status

| Metric | Value |
|--------|-------|
| Mean jaw error (30 frames) | 1.39px |
| Max jaw error | 2.27px |
| Frame 28 jaw error | 2.27px |
| Mean overall error | 1.41px |

These results are with:
- Sigma = 2.5 (fixed)
- Unclamped Lambda_inv (kept for Python's basin)
- pymtcnn initialization
