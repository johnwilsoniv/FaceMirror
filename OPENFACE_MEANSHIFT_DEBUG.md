# OpenFace Mean-Shift to PDM Parameter Update Analysis

## URGENT DEBUGGING FINDINGS

### Summary
The convergence bug appears to be in how **mean-shift vectors are projected onto the Jacobian** to produce parameter updates. The mean-shift vectors correctly compute landmark displacement targets, but the Jacobian-based projection may not be properly converting those image-space displacements to PDM parameter changes.

---

## 1. MEAN-SHIFT COMPUTATION PIPELINE

### Function: `NonVectorisedMeanShift_precalc_kde()` (Lines 820-935)
**Location:** `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`

#### What it does:
1. **Precomputes KDE responses** - Creates a lookup table of Gaussian kernel evaluations over a discretized response map grid
2. **For each landmark**, computes the mean-shift vector by:
   - Evaluating the KDE kernel at the current landmark position (dx, dy)
   - Computing a weighted average position from the response map
   - The weight is: `w = patch_response[ii,jj] * KDE_kernel[ii,jj]`

#### Key Computation (lines 902-928):
```cpp
// For each landmark i:
float mx = 0.0, my = 0.0, sum = 0.0;

for(int ii = 0; ii < resp_size; ii++)
{
    for(int jj = 0; jj < resp_size; jj++)
    {
        float v = patch_response[ii,jj] * kde_kernel[ii,jj];  // Weight
        sum += v;
        mx += v*jj;          // Weighted x accumulation
        my += v*ii;          // Weighted y accumulation
    }
}

// CRITICAL LINE:
float msx = (mx/sum - dx);   // Mean-shift in x
float msy = (my/sum - dy);   // Mean-shift in y
```

#### KDE Kernel Definition (line 854):
```cpp
v = exp(a*(vx+vy));  // where a = -0.5/(sigma*sigma)
```
This is a **Gaussian kernel** with parameter `sigma` (from parameters.sigma).

#### OUTPUT:
- `mean_shifts` matrix: size (2n x 1) where n = number of landmarks
  - Rows 0 to n-1: x-displacements for each landmark
  - Rows n to 2n-1: y-displacements for each landmark
- **These are in response map space**, need transformation to image space

---

## 2. COORDINATE SPACE TRANSFORMATIONS (Lines 1071-1083)

```cpp
// Transform offset from BASE shape to CURRENT shape position
cv::Mat_<float> offsets;
cv::Mat((current_shape_2D - base_shape_2D) * cv::Mat(sim_img_to_ref).t())
    .convertTo(offsets, CV_32F);

dxs = offsets.col(0) + (resp_size-1)/2;  // Center on response map
dys = offsets.col(1) + (resp_size-1)/2;

// Line 1077: COMPUTE MEAN-SHIFTS
NonVectorisedMeanShift_precalc_kde(mean_shifts, patch_expert_responses, 
    dxs, dys, resp_size, a, scale, view_id, kde_resp_precalc);

// Lines 1080-1083: TRANSFORM BACK TO IMAGE SPACE
cv::Mat_<float> mean_shifts_2D = (mean_shifts.reshape(1, 2)).t();
mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t();
mean_shifts = cv::Mat(mean_shifts_2D.t()).reshape(1, n*2);
```

**IMPORTANT:** 
- Mean-shifts are computed in **response map space**
- Must be transformed using `sim_ref_to_img` to get back to **image space**
- This transformation involves a 2x2 similarity matrix (rotation + scale)

---

## 3. THE NU_RLMS OPTIMIZATION LOOP (Lines 990-1191)

This is the core parameter update algorithm. Here's the flow:

### Step 1: Compute Jacobian (Lines 1052-1063)
```cpp
cv::Mat_<float> J, J_w_t;

if(rigid)
    pdm.ComputeRigidJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
else
    pdm.ComputeJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
```

**Jacobian Matrix:**
- Size: (2n) x (6 + m)
  - Rows 0 to n-1: derivatives of x-coordinates w.r.t. parameters
  - Rows n to 2n-1: derivatives of y-coordinates w.r.t. parameters
  - Columns 0-5: rigid parameters [scale, rot_x, rot_y, rot_z, tx, ty]
  - Columns 6 to 6+m-1: non-rigid parameters (PDM modes)

### Step 2: Compute Mean-Shifts (Line 1077)
See section 1 above.

### Step 3: PROJECT MEAN-SHIFTS ONTO JACOBIAN (Line 1107)
**THIS IS THE CRITICAL STEP:**

```cpp
// projection of the meanshifts onto the jacobians
cv::Mat_<float> J_w_t_m = J_w_t * mean_shifts;
```

**What this does:**
- J_w_t: Weighted transpose of Jacobian, size (6+m) x (2n)
- mean_shifts: (2n) x 1 vector of landmark displacements
- Result J_w_t_m: (6+m) x 1 vector of **how much each parameter should change**

**Mathematical interpretation:**
```
delta_parameters = J_w_t * mean_shifts
```

This is solving:
```
mean_shifts ≈ J * delta_parameters
```

### Step 4: Add Regularization (Lines 1110-1113)
```cpp
if(!rigid)
{
    // Subtract regularization penalty to keep parameters close to mean
    J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) 
        - regTerm(cv::Rect(6,6, m, m)) * current_local;
}
```

**Regularization formula:**
- regTerm: diagonal matrix with values `parameters.reg_factor / eigenvalues[j]`
- Adds penalty: `-lambda_j * current_local[j]` for each mode j
- Effect: Biases parameters toward zero (mean shape) to avoid wild deviations

### Step 5: Build Hessian and Solve (Lines 1115-1128)
```cpp
cv::Mat_<float> Hessian = regTerm.clone();

// Hessian = J_w_t * J + regTerm  (using BLAS for speed)
sgemm_(N, N, &J.cols, &J_w_t.rows, &J_w_t.cols, &alpha1, 
    (float*)J.data, &J.cols, (float*)J_w_t.data, &J_w_t.cols, 
    &beta1, (float*)Hessian.data, &J.cols);

// Solve: Hessian * param_update = J_w_t_m
cv::solve(Hessian, J_w_t_m, param_update, cv::DECOMP_CHOLESKY);
```

**This is Gauss-Newton optimization:**
```
(J^T W J + lambda*I) * delta_p = J^T W * mean_shifts
```

- J: Jacobian
- W: Weight matrix (diagonal, from patch expert confidences)
- lambda: Regularization from regTerm
- Solves using Cholesky decomposition

### Step 6: Apply Parameter Update (Lines 1130-1134)
```cpp
pdm.UpdateModelParameters(param_update, current_local, current_global);
pdm.Clamp(current_local, current_global, parameters);
```

---

## 4. JACOBIAN COMPUTATION DETAILS (PDM.cpp Lines 346-450)

### What the Jacobian Represents:
For each 2D landmark coordinate, compute how it changes with respect to each parameter.

**Format:**
```
J[i, j] = d(shape_2D[i]) / d(param[j])

Where shape_2D is computed as:
  X_2D = s * (R[0,0]*X_3D + R[0,1]*Y_3D + R[0,2]*Z_3D) + tx
  Y_2D = s * (R[1,0]*X_3D + R[1,1]*Y_3D + R[1,2]*Z_3D) + ty

Parameters:
  0: scale (s)
  1-3: rotation Euler angles (omega_x, omega_y, omega_z)
  4: translation x (tx)
  5: translation y (ty)
  6+: local parameters (PDM modes)
```

### Key Jacobian Entries (lines 397-419):

**Scaling term (parameter 0):**
```cpp
J[i, 0] = X_3D * R[0,0] + Y_3D * R[0,1] + Z_3D * R[0,2]
J[i+n, 0] = X_3D * R[1,0] + Y_3D * R[1,1] + Z_3D * R[1,2]
```

**Rotation terms (parameters 1-3):**
Uses small-angle approximation with perturbation matrix R':
```cpp
R' = [1,  -wz,  wy]
     [wz,  1,  -wx]
     [-wy, wx,  1]

J[i, 1] = s * (Y_3D * R[0,2] - Z_3D * R[0,1])  // rotation x
J[i, 2] = -s * (X_3D * R[0,2] - Z_3D * R[0,0]) // rotation y
J[i, 3] = s * (X_3D * R[0,1] - Y_3D * R[0,0])  // rotation z
```

**Translation terms (parameters 4-5):**
```cpp
J[i, 4] = 1.0    // d(X_2D)/d(tx)
J[i+n, 5] = 1.0  // d(Y_2D)/d(ty)
```

**Local parameters (parameters 6+m):**
For each PDM mode j:
```cpp
J[i, 6+j] = s * (R[0,0]*dX_3D/dmode[j] + R[0,1]*dY_3D/dmode[j] + R[0,2]*dZ_3D/dmode[j])
```

### Weighting (lines 422-449):
If weight matrix W is not identity, multiply Jacobian by weights:
```cpp
J_w[i, j] = W[i, i] * J[i, j]

Jacob_t_w = J_w.transpose()
```

---

## 5. PARAMETER UPDATE APPLICATION (PDM.cpp Lines 454-506)

```cpp
void PDM::UpdateModelParameters(const cv::Mat_<float>& delta_p, 
    cv::Mat_<float>& params_local, cv::Vec6f& params_global)
{
    // Global parameters:
    params_global[0] += delta_p[0];  // scale (DIRECT ADDITION)
    params_global[4] += delta_p[4];  // tx (DIRECT ADDITION)
    params_global[5] += delta_p[5];  // ty (DIRECT ADDITION)
    
    // Rotation (SPECIAL HANDLING):
    // 1. Construct perturbation matrix R' from delta_p[1:3]
    R' = [[1,        -delta_p[3], delta_p[2]],
          [delta_p[3], 1,         -delta_p[1]],
          [-delta_p[2], delta_p[1], 1]]
    
    // 2. Combine with current rotation: R_new = R_old * R'
    R_new = R_old * R'
    
    // 3. Extract new Euler angles
    params_global[1:3] = rotationMatrix_to_euler(R_new)
    
    // Local parameters (DIRECT ADDITION):
    if(delta_p.size > 6)
        params_local += delta_p[6:end]
}
```

**Important:** 
- Scale and translation: Direct addition
- Rotation: Matrix multiplication (small-angle perturbation)
- Local parameters: Direct addition

---

## 6. SHAPE RENDERING FROM PARAMETERS (PDM.cpp Lines 159-188)

```cpp
void PDM::CalcShape2D(cv::Mat_<float>& out_shape, 
    const cv::Mat_<float>& params_local, const cv::Vec6f& params_global)
{
    // 1. Build 3D shape from local parameters
    CalcShape3D(Shape_3D, params_local);  // Uses: mean_shape + sum(params_local[j] * princ_comp[j])
    
    // 2. Apply rigid transformation
    s = params_global[0];
    R = euler_to_rotation_matrix(params_global[1:3]);
    tx = params_global[4];
    ty = params_global[5];
    
    // 3. Project to 2D with weak-perspective
    for(each landmark i):
        X_2D = s * (R[0,0]*X_3D + R[0,1]*Y_3D + R[0,2]*Z_3D) + tx
        Y_2D = s * (R[1,0]*X_3D + R[1,1]*Y_3D + R[1,2]*Z_3D) + ty
}
```

---

## POTENTIAL BUGS - DEBUGGING CHECKLIST

### Issue 1: Jacobian-Mean-Shift Projection Mismatch
**Symptom:** Response maps are correct, but landmarks don't move.

**Check:**
1. Is the Jacobian computed at the CURRENT shape (line 1039)?
   - Yes, explicitly: `pdm.CalcShape2D(current_shape, current_local, current_global)`

2. Is J correctly mapping image-space displacements to parameter changes?
   - Verify by checking: J is size (2n) x (6+m) ✓
   - J rows correspond to x,y coordinates ✓
   - J columns correspond to parameters in correct order ✓

3. Is the weight matrix W correctly applied?
   - Check line 1028: `GetWeightMatrix(WeightMatrix, scale, view_id, parameters)`
   - W should be diagonal with patch confidence values

### Issue 2: Coordinate Space Transformation Error
**Symptom:** Mean-shift vectors are wrong magnitude or direction.

**Check:**
1. Are dxs/dys computed correctly? (Lines 1074-1075)
   - Should map from image space to response map space
   - Uses `sim_img_to_ref` to transform offset

2. Is the transformation back correct? (Lines 1080-1083)
   - Uses `sim_ref_to_img`
   - These should be inverse transformations

3. Is the response map size correct?
   - Used as resp_size in mean-shift computation
   - Must match patch expert response resolution

### Issue 3: Regularization Overpower
**Symptom:** Parameters drift back toward zero even with strong responses.

**Check:**
1. What is parameters.reg_factor?
   - Default should be reasonable (e.g., 0.1 to 1.0)
   - If too high, regularization dominates

2. Is regTerm computation correct? (Lines 1020-1024)
   ```cpp
   regularisations[6+j] = reg_factor / eigenvalues[j]
   ```
   - Higher eigenvalue = lower regularization ✓
   - This is correct Bayesian prior

### Issue 4: Convergence Termination
**Symptom:** Optimization stops too early.

**Check:**
1. Shape change threshold (Lines 1043-1047):
   ```cpp
   if(norm(current_shape, previous_shape) < 0.01) break;
   ```
   - 0.01 pixels might be too small
   - Check against your image resolution

2. Number of iterations (Line 1036):
   - `parameters.num_optimisation_iteration` - is it sufficient?

### Issue 5: Parameter Clamping
**Symptom:** Valid parameter updates are being truncated.

**Check:**
1. Line 1134: `pdm.Clamp(current_local, current_global, parameters)`
   - Is this restricting parameters too much?
   - Check what min/max bounds are applied

---

## CRITICAL EQUATIONS TO VERIFY

### Equation 1: Gauss-Newton Solution
```
delta_p = (J^T W J + lambda*I)^-1 * J^T W * mean_shifts
```

In code (lines 1115-1128):
- `Hessian = J_w_t * J + regTerm` ✓
- `solve(Hessian, J_w_t_m, param_update)` where `J_w_t_m = J_w_t * mean_shifts` ✓

### Equation 2: Weak-Perspective Projection
```
[x_2D]   [s*R[0,0]  s*R[0,1]  s*R[0,2]  1  0] [x_3D]   [tx]
[y_2D] = [s*R[1,0]  s*R[1,1]  s*R[1,2]  0  1] [y_3D] + [ty]
                                         [z_3D]
```

In code (lines 185-186):
```cpp
X_2D = s * (R[0,0]*X_3D + R[0,1]*Y_3D + R[0,2]*Z_3D) + tx
Y_2D = s * (R[1,0]*X_3D + R[1,1]*Y_3D + R[1,2]*Z_3D) + ty
```
✓ Correct

### Equation 3: Mean-Shift Calculation
```
mean_shift_x = sum(w_ij * jj) / sum(w_ij) - dx
mean_shift_y = sum(w_ij * ii) / sum(w_ij) - dy

where w_ij = patch_response[ii,jj] * KDE[ii,jj]
      KDE[ii,jj] = exp(-0.5*(dx-jj)^2/(sigma^2)) * exp(-0.5*(dy-ii)^2/(sigma^2))
```

In code (lines 915-928):
```cpp
v = (*p++) * (*kde_it++);  // w_ij
sum += v;
mx += v*jj;
my += v*ii;

msx = (mx/sum - dx);
msy = (my/sum - dy);
```
✓ Correct (ii/jj convention: ii=y, jj=x)

---

## RECOMMENDED DEBUGGING STEPS

1. **Add debug output** at line 1107:
   ```cpp
   std::cout << "Mean shifts norm: " << cv::norm(mean_shifts) << std::endl;
   std::cout << "Jacobian norm: " << cv::norm(J) << std::endl;
   std::cout << "Param update: " << param_update.t() << std::endl;
   ```

2. **Verify Jacobian invertibility:**
   - Check condition number of Hessian
   - Verify it's not singular or near-singular

3. **Test with synthetic data:**
   - Create a known shape shift
   - Verify Jacobian predicts correct parameter change
   - Verify mean-shift detects correct direction

4. **Check weight matrix:**
   - Print patch expert confidences
   - Verify they're not all zero or one

5. **Verify response maps:**
   - Compare response map peaks with expected landmark positions
   - Check if multiple peaks exist (ambiguous responses)

6. **Trace a single iteration:**
   - For iteration 1, manually compute all intermediate values
   - Verify each step matches the code

---

## KEY FILES TO INSPECT

1. **Mean-shift computation:**
   `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp` (lines 820-935)

2. **NU_RLMS optimization loop:**
   `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp` (lines 990-1191)

3. **Jacobian computation:**
   `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/PDM.cpp` (lines 346-450)

4. **Parameter update:**
   `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/PDM.cpp` (lines 454-506)

5. **Shape calculation:**
   `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/PDM.cpp` (lines 159-188)

