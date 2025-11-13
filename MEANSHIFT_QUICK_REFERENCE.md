# OpenFace Mean-Shift to PDM Update: Quick Reference

## The Complete Data Flow

```
1. PATCH EXPERTS
   ├─ Generate response maps for each landmark
   └─ Response[i] = heatmap showing confidence at each pixel

2. MEAN-SHIFT COMPUTATION (NonVectorisedMeanShift_precalc_kde)
   ├─ Input: Response maps, current landmark position (dx, dy)
   ├─ Kernel: Gaussian KDE with parameter sigma
   ├─ Operation: Weighted average of response map using KDE as weight
   └─ Output: mean_shifts = (2n x 1) vector of displacements IN RESPONSE-MAP SPACE

3. COORDINATE TRANSFORMATION
   ├─ Transform mean_shifts from response-map space TO image space
   ├─ Uses: sim_ref_to_img (inverse similarity transformation)
   └─ Output: mean_shifts in IMAGE SPACE

4. JACOBIAN COMPUTATION (ComputeJacobian)
   ├─ Input: Current PDM parameters (local, global)
   ├─ Compute: 2D derivatives of landmarks w.r.t. parameters
   │  J[i, j] = d(landmark_2D[i]) / d(parameter[j])
   ├─ Jacobian size: (2n) x (6 + m)
   │  - 2n rows: x,y coordinates for n landmarks
   │  - 6 cols: rigid parameters [scale, rot_x, rot_y, rot_z, tx, ty]
   │  - m cols: non-rigid (PDM modes)
   └─ Output: J and J_w_t (weighted transpose)

5. GAUSS-NEWTON SOLVER (Lines 1107-1128)
   ├─ Project mean_shifts onto Jacobian: J_w_t_m = J_w_t * mean_shifts
   ├─ Add regularization: J_w_t_m -= regTerm * current_local
   ├─ Build Hessian: H = J_w_t * J + regTerm
   ├─ Solve: H * delta_p = J_w_t_m (Cholesky decomposition)
   └─ Output: delta_p = (6 + m) parameter changes

6. PARAMETER UPDATE (UpdateModelParameters)
   ├─ Scale: param[0] += delta_p[0]
   ├─ Translation: param[4] += delta_p[4], param[5] += delta_p[5]
   ├─ Rotation: param[1:3] = rot_matrix_to_euler(R_old * R_perturbation)
   └─ Local: param_local += delta_p[6:end]

7. SHAPE RENDERING (CalcShape2D)
   ├─ Build 3D: shape_3D = mean_shape + sum(param_local[j] * princ_comp[j])
   ├─ Transform: shape_2D = s * (R * shape_3D) + [tx, ty]
   └─ Output: 2D landmark positions
```

---

## Critical Equations

### Mean-Shift (in response-map space):
```
mx = sum(patch_response[ii,jj] * KDE[ii,jj] * jj)
my = sum(patch_response[ii,jj] * KDE[ii,jj] * ii)
sum = sum(patch_response[ii,jj] * KDE[ii,jj])

mean_shift_x = (mx / sum) - dx
mean_shift_y = (my / sum) - dy

KDE[ii,jj] = exp(-0.5*((dx-jj)^2 + (dy-ii)^2) / sigma^2)
```

### Jacobian-based parameter update:
```
J_w_t_m = J_w_t * mean_shifts                         [project to param space]
H = J_w_t * J + regTerm                              [build Hessian]
delta_p = H^-1 * J_w_t_m                             [solve system]

Final: param_new = param_old + delta_p (with special handling for rotation)
```

---

## What Can Go Wrong

### 1. Mean-shifts computed but Jacobian matrix is zero/singular
**Symptom:** `J_w_t * mean_shifts` gives zero even with large mean_shifts

**Root cause:**
- Jacobian rows are zero (landmark not in parameter space)
- J_w_t is not transpose (shape mismatch)
- Weight matrix W zeros out all rows

**Debug:**
```cpp
std::cout << "J shape: " << J.size() << " norm: " << cv::norm(J) << std::endl;
std::cout << "J_w_t shape: " << J_w_t.size() << " norm: " << cv::norm(J_w_t) << std::endl;
std::cout << "mean_shifts norm: " << cv::norm(mean_shifts) << std::endl;
std::cout << "J_w_t_m (before regTerm): " << J_w_t_m.t() << std::endl;
```

### 2. Mean-shifts large but parameter updates small
**Symptom:** Jacobian-mean-shift projection gives tiny updates

**Root cause:**
- Hessian is singular/near-singular (condition number >> 1)
- Cholesky decomposition fails or gives numerical garbage
- Regularization is too strong

**Debug:**
```cpp
// Check Hessian condition number
cv::Mat_<float> D;
cv::eigen(Hessian, D);
float cond = D.at<float>(0,0) / D.at<float>(D.rows-1, 0);
std::cout << "Hessian condition number: " << cond << std::endl;
```

### 3. Parameter updates applied but landmarks don't move
**Symptom:** delta_p is nonzero, but CalcShape2D gives same output

**Root cause:**
- Parameter update calculation wrong (scale/rotation/translation)
- UpdateModelParameters has a bug (especially rotation handling)
- CalcShape2D doesn't use updated parameters

**Debug:**
```cpp
// Before UpdateModelParameters
cv::Mat_<float> shape_before;
pdm.CalcShape2D(shape_before, current_local, current_global);

// Call UpdateModelParameters

// After
cv::Mat_<float> shape_after;
pdm.CalcShape2D(shape_after, current_local, current_global);

std::cout << "Shape change: " << cv::norm(shape_after - shape_before) << std::endl;
```

### 4. Coordinate space transformation wrong
**Symptom:** Mean-shifts in wrong direction or magnitude

**Root cause:**
- dxs/dys computed incorrectly (lines 1074-1075)
- sim_img_to_ref or sim_ref_to_img is transposed/inverted
- Response map size doesn't match

**Debug:**
```cpp
std::cout << "Current shape diff norm: " << cv::norm(current_shape - base_shape) << std::endl;
std::cout << "dxs/dys range: [" << dxs.begin()[0] << ", " << dxs.begin()[0] << "]" << std::endl;
std::cout << "Response size: " << resp_size << std::endl;
std::cout << "sim_img_to_ref:\n" << sim_img_to_ref << std::endl;
```

---

## Key Line Numbers in Source Code

| What | File | Lines |
|------|------|-------|
| Mean-shift computation | LandmarkDetectorModel.cpp | 820-935 |
| NU_RLMS loop | LandmarkDetectorModel.cpp | 990-1191 |
| Jacobian-mean-shift projection | LandmarkDetectorModel.cpp | 1107 |
| Parameter update in Gauss-Newton | LandmarkDetectorModel.cpp | 1128 |
| Apply parameter update | LandmarkDetectorModel.cpp | 1131 |
| Jacobian computation | PDM.cpp | 346-450 |
| Parameter update function | PDM.cpp | 454-506 |
| Shape rendering | PDM.cpp | 159-188 |

---

## Validation Checklist

- [ ] Response maps show clear peaks near true landmarks
- [ ] Mean-shift vectors point from current position toward peak
- [ ] Mean-shift magnitude is reasonable (< response_size/2)
- [ ] Jacobian is full-rank (or near full-rank)
- [ ] Hessian condition number < 1e6
- [ ] Parameter updates are nonzero
- [ ] Parameter updates are reasonable magnitude (< 0.1 * current_params)
- [ ] CalcShape2D produces visible landmark movement after update
- [ ] Iterative convergence: each iteration moves landmarks closer to true position

---

## Hypothesis: Why It Might Not Work

**Most likely:** Jacobian rows corresponding to non-visible landmarks are being zeroed out (lines 1091-1099), but **not all landmarks in the weight matrix are also zeroed**, causing a dimension mismatch.

**Second most likely:** sim_ref_to_img transformation is wrong direction or scale, so mean-shifts are in wrong units for the Jacobian.

**Third most likely:** Weight matrix has very small values, making J_w_t numerically unstable, and Hessian becomes singular.

