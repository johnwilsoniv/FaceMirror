# Strategic Instrumentation Points for Convergence Debugging

## Location 1: After Mean-Shift Computation (Line 1077)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify mean-shifts are computed correctly

Add after line 1077:
```cpp
// DEBUG: Check mean-shift computation
float ms_norm = cv::norm(mean_shifts);
std::cout << "[ITER " << iter << "] Mean-shift norm: " << ms_norm << std::endl;

// Check a few landmarks
for(int i = 0; i < std::min(3, n); i++)
{
    std::cout << "  Landmark " << i << ": ms=("
              << mean_shifts.at<float>(i,0) << ", "
              << mean_shifts.at<float>(i+n,0) << ")"
              << std::endl;
}
```

---

## Location 2: Before/After Coordinate Transform (Lines 1080-1083)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify coordinate transformation is correct

Add before line 1080 (save a copy):
```cpp
cv::Mat_<float> mean_shifts_before_transform = mean_shifts.clone();
```

Add after line 1083:
```cpp
// DEBUG: Check coordinate transformation
float ms_norm_after = cv::norm(mean_shifts);
std::cout << "[ITER " << iter << "] After transform: norm="
          << ms_norm_after << " (was " << ms_norm << ")" << std::endl;

// Verify transformation is invertible
if(ms_norm_after < ms_norm * 0.5 || ms_norm_after > ms_norm * 2.0)
{
    std::cout << "WARNING: Coordinate transform magnitude changed significantly!" << std::endl;
    std::cout << "  sim_ref_to_img: " << sim_ref_to_img << std::endl;
    std::cout << "  sim_img_to_ref: " << sim_img_to_ref << std::endl;
}
```

---

## Location 3: Jacobian Computation (After Line 1062)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify Jacobian is non-singular

Add after line 1063:
```cpp
// DEBUG: Check Jacobian
std::cout << "[ITER " << iter << "] Jacobian: rows=" << J.rows << " cols=" << J.cols
          << " norm=" << cv::norm(J) << std::endl;

// Check if Jacobian has zero rows (invisible landmarks)
int zero_rows = 0;
for(int i = 0; i < J.rows; i++)
{
    if(cv::norm(J.row(i)) < 1e-6)
        zero_rows++;
}
std::cout << "  Zero rows (invisible landmarks): " << zero_rows << "/" << J.rows << std::endl;
```

---

## Location 4: After Jacobian-Mean-Shift Projection (Line 1107)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify projection step

Add after line 1107:
```cpp
// DEBUG: Check Jacobian-mean-shift projection
std::cout << "[ITER " << iter << "] J_w_t_m before regTerm: norm=" << cv::norm(J_w_t_m)
          << std::endl;

// Print parameter update components
std::cout << "  Components: [";
for(int i = 0; i < std::min(6, (int)J_w_t_m.rows); i++)
{
    std::cout << J_w_t_m.at<float>(i, 0) << " ";
}
if(J_w_t_m.rows > 6) std::cout << "... (" << J_w_t_m.rows-6 << " local params)";
std::cout << "]" << std::endl;
```

---

## Location 5: Hessian and Solve (Lines 1115-1128)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify Hessian is well-conditioned

Add before line 1128:
```cpp
// DEBUG: Check Hessian condition
cv::Mat_<float> eigenvalues;
cv::eigen(Hessian, eigenvalues);

float max_eig = eigenvalues.at<float>(0, 0);
float min_eig = eigenvalues.at<float>(eigenvalues.rows-1, 0);
float cond_num = (min_eig > 1e-10) ? max_eig / min_eig : 1e10;

std::cout << "[ITER " << iter << "] Hessian: size=" << Hessian.rows << "x" << Hessian.cols
          << " cond_num=" << cond_num
          << " eig_range=[" << min_eig << ", " << max_eig << "]" << std::endl;

if(cond_num > 1e6)
    std::cout << "WARNING: Hessian is ill-conditioned!" << std::endl;
```

Add after line 1128:
```cpp
// DEBUG: Check solve result
std::cout << "[ITER " << iter << "] param_update: norm=" << cv::norm(param_update)
          << " norm/J_w_t_m_norm=" << (cv::norm(J_w_t_m) > 1e-10 ? cv::norm(param_update)/cv::norm(J_w_t_m) : 0)
          << std::endl;

if(cv::norm(param_update) < 1e-8)
    std::cout << "WARNING: Parameter update is near-zero!" << std::endl;
```

---

## Location 6: After Parameter Update (Line 1131)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify parameters actually changed

Add after line 1134:
```cpp
// DEBUG: Check parameter update effect
cv::Mat_<float> new_shape;
pdm.CalcShape2D(new_shape, current_local, current_global);
float shape_change = cv::norm(new_shape - current_shape);

std::cout << "[ITER " << iter << "] After param update:" << std::endl;
std::cout << "  Shape change: " << shape_change << " pixels" << std::endl;
std::cout << "  Global params: [s=" << current_global[0] << " tx=" << current_global[4] 
          << " ty=" << current_global[5] << "]" << std::endl;
std::cout << "  Local param norm: " << cv::norm(current_local) << std::endl;

if(shape_change < 0.01)
    std::cout << "WARNING: Minimal shape change despite non-zero param_update!" << std::endl;
```

---

## Location 7: Iteration Loop (Line 1036)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify convergence criteria

Modify the iteration loop header:
```cpp
for(int iter = 0; iter < parameters.num_optimisation_iteration; iter++)
{
    std::cout << "\n========== ITERATION " << iter << " ==========" << std::endl;
    
    // ... rest of loop ...
    
    if(iter > 0)
    {
        float shape_norm = cv::norm(current_shape - previous_shape);
        std::cout << "Shape change from previous: " << shape_norm << std::endl;
        
        if(shape_norm < 0.01)
        {
            std::cout << "Converged (shape change < 0.01)" << std::endl;
            break;
        }
    }
    
    // ... rest of loop ...
}
```

---

## Location 8: Weight Matrix Check (After Line 1028)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Verify weights are reasonable

Add after line 1028:
```cpp
// DEBUG: Check weight matrix
float min_w = 1e10, max_w = -1e10;
int zero_w = 0;
for(int i = 0; i < WeightMatrix.rows; i++)
{
    float w = WeightMatrix.at<float>(i, i);
    if(w < 1e-10) zero_w++;
    min_w = std::min(min_w, w);
    max_w = std::max(max_w, w);
}
std::cout << "[Before NU_RLMS] WeightMatrix: min=" << min_w << " max=" << max_w
          << " zeros=" << zero_w << "/" << WeightMatrix.rows << std::endl;

if(zero_w > WeightMatrix.rows / 2)
    std::cout << "WARNING: Over 50% of weights are zero!" << std::endl;
```

---

## Location 9: Mean-Shift in NonVectorisedMeanShift_precalc_kde (Lines 927-931)
**File:** `LandmarkDetectorModel.cpp`
**Purpose:** Debug mean-shift computation per landmark

Add before line 927:
```cpp
// DEBUG: Per-landmark mean-shift (first few only)
if(i < 3)
{
    std::cout << "  [Landmark " << i << "]" << std::endl;
    std::cout << "    Position: dx=" << dx << " dy=" << dy << std::endl;
    std::cout << "    Sum (weight): " << sum << std::endl;
    std::cout << "    Weighted center: mx=" << (mx/sum) << " my=" << (my/sum) << std::endl;
}
```

---

## Quick-Start Debug Session

To enable all debug output:

1. Add this at the start of NU_RLMS (line 993):
```cpp
const bool DEBUG_CONVERGENCE = true;
```

2. Wrap all debug statements with:
```cpp
if(DEBUG_CONVERGENCE)
{
    // ... debug output ...
}
```

3. Compile with debug symbols:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
```

4. Run with output to file:
```bash
./your_program 2>&1 | tee debug_output.log
```

5. Analyze the log for the step where things go wrong

---

## Expected Output Pattern (Working Case)

```
========== ITERATION 0 ==========
[ITER 0] Mean-shift norm: 15.3
  Landmark 0: ms=(2.5, 3.2)
  Landmark 1: ms=(1.8, -2.1)
  ...
[ITER 0] After transform: norm=15.5 (was 15.3)
[ITER 0] Jacobian: rows=134 cols=36 norm=245.3
  Zero rows (invisible landmarks): 0/134
[ITER 0] J_w_t_m before regTerm: norm=12.4
  Components: [0.23 -0.15 0.08 0.12 1.5 -2.1 ... (30 local params)]
[ITER 0] Hessian: size=36x36 cond_num=45.2 eig_range=[0.008, 0.36]
[ITER 0] param_update: norm=3.2 norm/J_w_t_m_norm=0.26
[ITER 0] After param update:
  Shape change: 8.3 pixels
  Global params: [s=0.82 tx=245.1 ty=156.3]
  Local param norm: 2.1
Shape change from previous: 8.3

========== ITERATION 1 ==========
[ITER 1] Mean-shift norm: 8.1
  ...
```

---

## What to Look For

1. **Mean-shift norm steadily decreases** → Good convergence signal
2. **Shape change > 1 pixel per iteration** → Parameters are moving
3. **Hessian condition number < 100** → Solver should be stable
4. **param_update has mix of positive/negative values** → Not stuck in saturation

---

## Red Flags

- Mean-shift norm = 0 → Response maps are flat or wrong
- Shape change = 0 despite non-zero param_update → CalcShape2D bug
- Hessian condition > 1e6 → Matrix near singular
- Weight matrix all zeros → No visible landmarks
- Coordinate transform magnitude changes wildly → sim matrices wrong

