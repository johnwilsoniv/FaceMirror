# PyCLNF Response Map Quality Investigation Report

## Executive Summary

**Root Cause Identified:** PyCLNF applies an **incorrect normalization step** that does not exist in OpenFace C++. This normalization destroys relative magnitude information between response maps, making all response maps have identical max values regardless of patch confidence, leading to poor convergence.

**Diagnosis:** Poor response map quality due to incorrect post-processing (normalization)

**Impact:** Convergence failure - final update ~2.5 instead of <0.005

---

## Investigation Findings

### 1. Code Comparison: Response Map Computation

#### A. OpenFace C++ (CCNF_patch_expert.cpp lines 356-415)

**Response computation workflow:**
```cpp
// 1. Sum neuron responses (lines 375-385)
for(size_t i = 0; i < neurons.size(); i++) {
    if(neurons[i].alpha > 1e-4) {
        neurons[i].Response(...);
        response = response + neuron_response;
    }
}

// 2. Apply Sigma transformation (lines 400-404)
cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);
cv::Mat out = Sigmas[s_to_use] * resp_vec_f;
response = out.reshape(1, response_height);

// 3. Remove negative values ONLY (lines 406-413)
double min;
minMaxIdx(response, &min, 0);
if(min < 0) {
    response = response - min;
}
// NO FURTHER NORMALIZATION!
```

**Key Point:** OpenFace C++ does NOT normalize response maps to [0,1] range. It only shifts them to remove negative values.

#### B. PyCLNF (optimizer.py lines 534-547)

**Response computation workflow:**
```python
# 1. Sum neuron responses (lines 475-510)
for i in range(window_size):
    for j in range(window_size):
        patch = self._extract_patch(...)
        if patch is not None:
            response_map[i, j] = patch_expert.compute_response(patch)

# 2. Apply Sigma transformation (lines 512-533)
sigma_comps = sigma_components[response_window_size]
Sigma = patch_expert.compute_sigma(sigma_comps, window_size=response_window_size)
response_vec = response_map.reshape(-1, 1)
response_transformed = Sigma @ response_vec
response_map = response_transformed.reshape(response_shape)

# 3. Remove negative values (lines 536-538)
min_val = response_map.min()
if min_val < 0:
    response_map = response_map - min_val

# 4. ⚠️ EXTRA NORMALIZATION NOT IN OPENFACE! (lines 540-545)
max_val = response_map.max()
if max_val > 0:
    response_map = response_map / max_val  # ← INCORRECT!
```

**Key Point:** PyCLNF adds an extra normalization step (dividing by max value) that does NOT exist in OpenFace C++.

---

### 2. Diagnostic Results

#### Test Setup
- Image: Real face image (1920x2160 pixels)
- Patch experts: 11x11 patches, 7 neurons each
- Window size: 11x11 search window
- Test landmarks: 30 (nose), 36 (eye), 48 (mouth)

#### A. Raw Response Statistics (Landmark 30, before sigma)

```
Min:   0.0000042176
Max:   3.8919210434
Mean:  0.5779808205
Std:   0.9697044094
Range: 3.8919168258
```

**Observation:** Raw responses show good variation (~1000x dynamic range)

#### B. After Sigma Transformation (before normalization)

```
Min:   -0.0012426617
Max:    0.0238330910
Mean:   0.0036438426
Std:    0.0062555159
```

**Observation:** Sigma transformation reduces magnitude by ~160x (3.89 → 0.024), which is expected behavior for spatial correlation modeling.

#### C. After PyCLNF Normalization (INCORRECT)

```
Min:   0.0000000000
Max:   1.0000000000
Mean:  0.1948700000
Peak sharpness: 1.554
Dynamic range:  5.270
```

**Critical Issue:** All response maps now have max=1.0, destroying relative confidence information!

---

### 3. Impact Analysis

#### A. Why This Breaks Convergence

The NU-RLMS optimizer relies on response map magnitudes to weight landmark updates:

```python
# In mean-shift computation (optimizer.py line 387):
weight = kde_weight * response_map[ii, jj]
```

When ALL response maps are normalized to [0,1]:

1. **Lost Confidence Information:** A high-confidence patch (trained on good data) and a low-confidence patch (trained on noisy data) produce identical max responses (1.0)

2. **Unreliable Landmarks Get Equal Weight:** The optimizer trusts unreliable landmarks as much as reliable ones

3. **Poor Mean-Shift Estimates:** KDE mean-shift is computed from normalized responses, giving equal importance to all landmarks regardless of patch quality

4. **Large Parameter Updates:** Without proper weighting, parameter updates can overshoot, causing oscillation and convergence failure

#### B. Comparison: With vs Without Normalization

**Landmark 30 (Nose - High Confidence, patch_confidence=0.410):**
- Raw max after sigma: 0.0238
- After normalization: 1.0000

**Landmark 36 (Eye - Higher Confidence, patch_confidence=0.517):**
- Raw max after sigma: 0.0357 (should be higher than landmark 30)
- After normalization: 1.0000 (identical to landmark 30!)

**Result:** Normalization makes the optimizer treat both landmarks identically, even though landmark 36 should have ~1.5x more influence.

#### C. Why Sigma Transformation Showed No Improvement

The diagnostic shows:
- Sigma transformation reduces dynamic range: 7.072 → 5.270 (-25%)
- But then normalization rescales everything to [0,1] anyway
- Net effect: Sigma transformation's attenuation is completely undone by normalization

This explains why implementing sigma transformation showed no convergence improvement!

---

### 4. Secondary Findings

#### A. Neuron Response Computation: ✓ CORRECT

PyCLNF's neuron response computation matches OpenFace C++:

```python
# PyCLNF (patch_expert.py lines 166-173)
correlation = np.sum(weights_centered * features_centered) / (weight_norm * feature_norm)
sigmoid_input = correlation * norm_weights + bias
response = (2.0 * alpha) * self._sigmoid(sigmoid_input)
```

```cpp
// OpenFace C++ (CCNF_patch_expert.cpp line 282)
*p++ = (2 * alpha) * 1.0 /(1.0 + exp( -(*q1++ * norm_weights + bias )));
```

**Verdict:** Neuron computation is correct.

#### B. Cross-Correlation: ✓ CORRECT

PyCLNF computes TM_CCOEFF_NORMED correctly:
- Centering: weights_centered = weights - mean(weights)
- Normalization: correlation / (||weights|| * ||features||)

**Verdict:** Cross-correlation is correct.

#### C. Sigma Computation: ✓ CORRECT

```python
# PyCLNF (patch_expert.py lines 214-244)
sum_alphas = sum(neuron['alpha'] for neuron in self.neurons)
q1 = sum_alphas * np.eye(matrix_size)
q2 = sum(self.betas[i] * sigma_components[i])
SigmaInv = 2.0 * (q1 + q2)
Sigma = np.linalg.inv(SigmaInv)
```

```cpp
// OpenFace C++ (CCNF_patch_expert.cpp lines 92-117)
sum_alphas = sum_alphas + this->neurons[a].alpha;
q1 = sum_alphas * cv::Mat_<float>::eye(window_size*window_size, window_size*window_size);
q2 = q2 + ((float)this->betas[b]) * sigma_components[b];
SigmaInv = 2 * (q1 + q2);
cv::invert(SigmaInv, Sigma_f, cv::DECOMP_CHOLESKY);
```

**Verdict:** Sigma computation is correct.

---

## Root Cause Summary

**The Problem:**
Lines 540-545 in `/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/core/optimizer.py`:

```python
# Normalize to [0, 1] range for numerical stability
# Note: This partially defeats sigma transformation's attenuation,
# but is necessary to keep response magnitudes reasonable for mean-shift
max_val = response_map.max()
if max_val > 0:
    response_map = response_map / max_val
```

**Why It's Wrong:**
1. This normalization does NOT exist in OpenFace C++
2. It destroys relative confidence information between landmarks
3. It defeats the purpose of sigma transformation
4. The comment acknowledges it "partially defeats sigma transformation" but incorrectly justifies it

**The Fix:**
Remove lines 540-545. The normalization is NOT necessary - OpenFace C++ works fine without it. The sigma-transformed response magnitudes are already appropriate for mean-shift computation.

---

## Recommended Fix

### File: `/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/core/optimizer.py`

**Lines 534-547 should be changed from:**
```python
# OpenFace CCNF Response normalization (CCNF_patch_expert.cpp lines 406-413)
# After computing responses, remove negative values by shifting
min_val = response_map.min()
if min_val < 0:
    response_map = response_map - min_val

# Normalize to [0, 1] range for numerical stability
# Note: This partially defeats sigma transformation's attenuation,
# but is necessary to keep response magnitudes reasonable for mean-shift
max_val = response_map.max()
if max_val > 0:
    response_map = response_map / max_val

return response_map
```

**To:**
```python
# OpenFace CCNF Response normalization (CCNF_patch_expert.cpp lines 406-413)
# After computing responses, remove negative values by shifting
# NOTE: OpenFace C++ does NOT normalize to [0,1] - that would destroy
# relative confidence information between landmarks!
min_val = response_map.min()
if min_val < 0:
    response_map = response_map - min_val

return response_map
```

---

## Expected Impact of Fix

### Before Fix:
- All response maps have max=1.0
- High-confidence and low-confidence landmarks treated equally
- Sigma transformation has no effect (immediately undone by normalization)
- Final update: ~2.5 (convergence failure)

### After Fix:
- Response maps retain relative magnitudes (e.g., 0.024 vs 0.036)
- High-confidence landmarks have proportionally larger influence
- Sigma transformation properly attenuates unreliable responses
- Expected final update: <0.005 (successful convergence)

### Confidence Level: **VERY HIGH**

This is a clear, unambiguous bug:
1. Code in PyCLNF differs from OpenFace C++
2. The difference destroys critical information (relative magnitudes)
3. The paper explicitly states "NU-RLMS performs better on more reliable response maps"
4. Normalization makes reliable and unreliable responses indistinguishable

---

## References

1. **OpenFace C++ Implementation:**
   - File: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/CCNF_patch_expert.cpp`
   - Response() method: lines 356-415
   - Key observation: Only removes negative values (lines 406-413), no [0,1] normalization

2. **PyCLNF Implementation:**
   - File: `/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/core/optimizer.py`
   - _compute_response_map() method: lines 408-547
   - Bug location: lines 540-545 (incorrect normalization)

3. **Diagnostic Script:**
   - File: `/Users/johnwilsoniv/Documents/SplitFace Open3/diagnose_response_maps.py`
   - Key finding: "WARNING: Sigma transformation REDUCES dynamic range significantly!" followed by normalization undoing it

4. **Research Paper:**
   - "NU-RLMS performs better on more reliable response maps"
   - Normalization makes reliable and unreliable maps indistinguishable

---

## Conclusion

The convergence issue in PyCLNF is caused by an **incorrect normalization step** that destroys relative response magnitude information. This makes the NU-RLMS optimizer treat all landmarks equally regardless of patch confidence, leading to poor parameter updates and convergence failure.

**The fix is simple:** Remove lines 540-545 in optimizer.py to match OpenFace C++ behavior.

This explains why:
1. Sigma transformation showed no improvement (normalization undid it)
2. Convergence fails (unreliable landmarks get equal weight)
3. Final updates are large (~2.5) instead of small (<0.005)

The fix should immediately improve convergence by allowing the optimizer to properly weight high-confidence landmarks over low-confidence ones.
