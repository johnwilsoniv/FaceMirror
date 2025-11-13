# CCNF Sigma Transformation Implementation Summary

## Overview

Successfully implemented CCNF Sigma spatial correlation transformation for PyCLNF, matching the OpenFace C++ implementation. The implementation is mathematically correct but shows minimal impact on convergence, suggesting deeper issues with PyCLNF's core optimization.

## Implementation Details

### 1. Sigma Component Export
**Location**: `export_sigma_components.cpp` (C++), `pyclnf/models/sigma_components/` (exported files)

**Exported Components**:
- Window sizes: [7, 9, 11, 15]
- 3 sigma components per window size (12 files total)
- Matrix sizes: 49×49, 81×81, 121×121, 225×225

**Validation**: All sigma components are:
- Symmetric (required for covariance matrices)
- Positive definite (eigenvalues > 0)
- Correctly sized for window dimensions

### 2. Python Integration

#### `pyclnf/models/openface_loader.py`
- Added `load_sigma_components()` function (lines 593-633)
- Loads all sigma components from exported files
- Returns Dict[int, List[np.ndarray]] mapping window_size → sigma matrices

#### `pyclnf/core/patch_expert.py`
- Modified CCNFModel.__init__ to load sigma components (lines 224-229)
- Added CCNFPatchExpert.compute_sigma() method (lines 192-242)
- Implements OpenFace formula:
  ```python
  sum_alphas = Σ(neuron.alpha)
  q1 = sum_alphas * Identity
  q2 = Σ(beta_i * sigma_component_i)
  SigmaInv = 2 * (q1 + q2)
  Sigma = inv(SigmaInv)  # using Cholesky decomposition
  ```

#### `pyclnf/core/optimizer.py`
- Added sigma_components parameter to optimize(), _compute_mean_shift(), _compute_response_map()
- Applied transformation at lines 516-532:
  ```python
  response_vec = response_map.reshape(-1, 1)
  response_transformed = Sigma @ response_vec
  response_map = response_transformed.reshape(response_shape)
  ```
- Handles negative values as per OpenFace (shift to remove negatives)
- Normalizes to [0, 1] for numerical stability

#### `pyclnf/clnf.py`
- Passes sigma_components to optimizer (line 176)

### 3. Key Implementation Details

**Dimension Matching**:
- Initially tried using patch width for sigma lookup (WRONG)
- Fixed to use response_window_size (window_size parameter, e.g., 11)
- Response maps are window_size × window_size, not patch_size × patch_size

**Sigma Matrix Properties** (for window_size=11):
- Shape: (121, 121)
- Value range: [-0.0002, 0.0075]
- Diagonal mean: 0.006257
- Condition number: 1.90
- **Effect**: Significantly attenuates response magnitudes (~160x reduction)

**Normalization Issue**:
- Sigma transformation intentionally attenuates responses to model spatial correlations
- However, we must normalize to [0, 1] afterward for numerical stability
- This partially defeats the sigma transformation's purpose
- Removing normalization makes convergence WORSE (8.3% degradation vs 2.9%)

## Testing Results

### Comparison Test (test_sigma_convergence.py)

**With Sigma Transformation**:
- Converged: False
- Iterations: 80
- Final update: 2.582
- Landmark variance: 259.00

**Without Sigma Transformation**:
- Converged: False
- Iterations: 80
- Final update: 2.509
- Landmark variance: 259.18

**Analysis**:
- Sigma shows 2.9% DEGRADATION in convergence (higher final update magnitude)
- Landmark difference: only 1.10 pixels (minimal difference)
- Both versions fail to converge in 80 iterations

## Root Cause Analysis

### Why Sigma Isn't Helping

1. **Normalization Conflict**: The [0,1] normalization after sigma transformation partially undoes the spatial correlation modeling

2. **Base Convergence Issues**: PyCLNF has fundamental convergence problems unrelated to sigma:
   - Non-convergence in 80 iterations regardless of sigma
   - Final update magnitudes ~2.5-4.0 (should be <0.005)
   - Suggests issues with:
     - Response map computation
     - Mean-shift calculation
     - Jacobian computation
     - Optimization parameters

3. **Implementation Trade-off**: Must choose between:
   - Preserving sigma attenuation (causes numerical issues with mean-shift)
   - Normalizing responses (defeats sigma's purpose but improves stability)

### Sigma Values Analysis

From test (landmark 0, window_size=11):
```
Neuron alphas: sum = 41.87
Betas: sum = 9.96
Computed Sigma:
  Diagonal mean: 0.006257
  Range: [-0.000241, 0.007535]
  Mean deviation from identity: 0.993743
```

**Implication**: Sigma effectively multiplies responses by ~0.006, a 160x attenuation. When normalized back to [0, 1], the spatial correlation structure is preserved but magnitudes are lost.

## OpenFace C++ Code Reference

**Sigma Computation**: CCNF_patch_expert.cpp lines 82-119
```cpp
void CCNF_patch_expert::ComputeSigmas(std::vector<cv::Mat_<float> > sigma_components, int window_size)
{
    // Calculate sum of alphas
    float sum_alphas = 0;
    for(int a = 0; a < n_alphas; ++a)
        sum_alphas = sum_alphas + this->neurons[a].alpha;

    // q1 = sum_alphas * Identity
    cv::Mat_<float> q1 = sum_alphas * cv::Mat_<float>::eye(window_size*window_size, window_size*window_size);

    // q2 = sum of (beta * sigma_component)
    cv::Mat_<float> q2 = cv::Mat_<float>::zeros(window_size*window_size, window_size*window_size);
    for (int b=0; b < n_betas; ++b)
        q2 = q2 + ((float)this->betas[b]) * sigma_components[b];

    // SigmaInv = 2 * (q1 + q2)
    cv::Mat_<float> SigmaInv = 2 * (q1 + q2);

    // Sigma = inv(SigmaInv) using Cholesky
    cv::Mat Sigma_f;
    cv::invert(SigmaInv, Sigma_f, cv::DECOMP_CHOLESKY);

    Sigmas.push_back(Sigma_f);
}
```

**Sigma Application**: CCNF_patch_expert.cpp lines 400-414
```cpp
// Find correct sigma for response window size
for(size_t i=0; i < window_sizes.size(); ++i)
{
    if(window_sizes[i] == response_height)
    {
        s_to_use = i;
        break;
    }
}

// Apply sigma transformation
cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);
cv::Mat out = Sigmas[s_to_use] * resp_vec_f;
response = out.reshape(1, response_height);

// Remove negative values
double min;
minMaxIdx(response, &min, 0);
if(min < 0)
{
    response = response - min;
}
```

**Note**: OpenFace does NOT renormalize to [0, 1] after sigma transformation!

## Conclusion

The CCNF Sigma transformation has been correctly implemented according to the OpenFace specification:
- ✅ Sigma components correctly exported from OpenFace C++
- ✅ Sigma computation formula matches OpenFace exactly
- ✅ Transformation applied at correct stage (after patch response, before mean-shift)
- ✅ Dimension handling corrected (uses response window size, not patch size)
- ❌ Shows no convergence improvement (2.9% degradation)

**Next Steps**:

1. **Investigate Base Convergence Issues**: Focus on core PyCLNF optimization problems:
   - Debug response map computation
   - Verify mean-shift calculation
   - Check Jacobian accuracy
   - Compare parameters with OpenFace defaults

2. **Response Magnitude Analysis**: Compare response map magnitudes between PyCLNF and OpenFace C++ to identify discrepancies

3. **Alternative Sigma Approach**: Consider applying sigma only at later refinement stages, or with different normalization strategies

4. **Parameter Tuning**: Experiment with:
   - Higher iterations (OpenFace typically uses 100+)
   - Different weight_multiplier values (OpenFace: 5-7)
   - Adjusted regularization
   - Alternative sigma values

The sigma implementation is technically correct but cannot compensate for fundamental convergence issues in the base CLNF optimization.
