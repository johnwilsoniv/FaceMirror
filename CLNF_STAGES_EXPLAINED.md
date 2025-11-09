# CLNF (Constrained Local Neural Fields) - Complete Stage Breakdown

## Overview

CLNF is a facial landmark detection model with three main components:
1. **Point Distribution Model (PDM)** - Statistical model of facial shape
2. **Local Neural Field (LNF) Patch Experts** - Novel landmark detectors
3. **Non-Uniform Regularised Landmark Mean-Shift (NU-RLMS)** - Optimization strategy

## The Three Core Components

### 1. Point Distribution Model (PDM)

The PDM models landmark locations using:

```
xi = s · R2D · (x̄i + Φiq) + t
```

Where:
- **x̄i**: Mean position of landmark i
- **Φi**: 3×m principal component matrix (learned from training data)
- **q**: m-dimensional non-rigid shape parameters
- **s**: Scale
- **t**: Translation [tx, ty]
- **w**: Orientation [wx, wy, wz] (axis-angle representation)
- **R2D**: First two rows of 3×3 rotation matrix

Full parameter vector: **p = [s, t, w, q]**

### 2. Local Neural Field (LNF) Patch Expert

The LNF patch expert models the conditional probability P(y|X) where:
- **X**: Input pixel intensities in support region (e.g., 11×11 patch)
- **y**: Output alignment probabilities

**Potential Function:**
```
Ψ = Σi Σk αk·fk(yi, X, θk) + Σi,j Σk βk·gk(yi, yj) + Σi,j Σk γk·lk(yi, yj)
```

**Three Types of Features:**

**a) Vertex Features fk** (Neural Network Layer):
```
fk(yi, X, θk) = -(yi - h(θk, xi))²
h(θ, x) = 1/(1 + e^(-θᵀx))  [sigmoid activation]
```
- Maps input pixels to alignment probability through one-layer neural network
- θk: Weight vector for neuron k (acts as convolution kernel)
- αk: Reliability weight for neuron k

**b) Edge Features gk** (Spatial Similarity):
```
gk(yi, yj) = -½ S(gk)i,j (yi - yj)²
```
- Enforces smoothness between neighboring pixels
- S(g1): Returns 1 for horizontal/vertical neighbors
- S(g2): Returns 1 for diagonal neighbors
- Makes response maps smoother with fewer peaks

**c) Edge Features lk** (Sparsity Constraint):
```
lk(yi, yj) = -½ S(lk)i,j (yi + yj)²
```
- Enforces that only ONE peak should be present in the response
- Penalizes multiple high-probability regions
- S(l): Returns 1 when nodes are 4-6 edges apart

### 3. Non-Uniform Regularised Landmark Mean-Shift (NU-RLMS)

**Key Innovation:** Weights patch experts by their reliability.

**Objective Function:**
```
arg min_Δp (||p + Δp||²Λ⁻¹ + ||JΔp - v||²W)
```

Where:
- **Λ⁻¹**: Prior on parameter p (Gaussian for non-rigid, uniform for rigid)
- **J**: Jacobian of landmark locations w.r.t. parameters p
- **v**: Mean-shift vectors from patch responses
- **W**: Diagonal weight matrix based on patch expert reliabilities

**Update Rule (Tikhonov Regularised Gauss-Newton):**
```
Δp = -(JᵀWJ + rΛ⁻¹)⁻¹(rΛ⁻¹p - JᵀWv)
```

**Weight Matrix Construction:**
```
W = w · diag(c1, ..., cn, c1, ..., cn)
```
- ci: Correlation coefficient of patch expert i on validation data
- w: Empirically determined scaling factor
- Computed separately for each scale and view

## Complete CLNF Fitting Pipeline

### Stage 0: Initialization

**Input:**
- Face detection bbox (e.g., from Viola-Jones)
- Video frame

**Initialize Parameters:**
- Rigid parameters (s, t, w) from face detector
- Non-rigid parameters q = 0

### Stage 1: Multi-Scale Setup

**Hierarchical Processing:**
- Multiple scales (e.g., 17px, 23px, 30px interocular distance)
- Multiple orientations (e.g., -20°, 0°, +20° yaw)
- Fit coarse-to-fine (smallest scale → largest)

**Typical Configuration:**
- Areas of interest: [15×15, 21×21, 21×21] for each scale
- Patch support regions: 11×11 pixels
- Parameters: ρ=1.5-2.0, r=25, w=5-7

### Stage 2: Patch Expert Evaluation (Per Landmark)

For each landmark i at current estimate xᶜi:

**2.1 Extract Support Region**
- Sample 11×11 pixel patch around xᶜi
- Vectorize to xi ∈ R¹²¹

**2.2 Compute Vertex Features**
- Apply neural network: h(Θ, xi) [K1 neurons]
- Each neuron acts as learned convolution kernel
- Weight by αk (reliability of each neuron)

**2.3 Compute Edge Features**
- Spatial similarity gk: Smooth neighboring responses
- Sparsity lk: Suppress multiple peaks

**2.4 Generate Response Map**
- Evaluate LNF at all locations in area of interest (e.g., 15×15 grid)
- Output: πxi = Probability map for landmark i

**Result:** Response map shows probability of correct alignment at each location

### Stage 3: Mean-Shift Vector Calculation

For each landmark i:

**Gaussian Kernel Density Estimation:**
```
vi = [Σy∈Ψi πy·N(xᶜi; y, ρI)] / [Σz∈Ψi πz·N(xᶜi; z, ρI)] - xᶜi
```

Where:
- xᶜi: Current landmark estimate
- Ψi: Area of interest for landmark i
- πy: Alignment probability at location y
- N(xᶜi; y, ρI): Gaussian kernel (bandwidth ρ)
- ρ: Empirically set (typically 1.5-2.0)

**Physical Meaning:** vi points in the direction of highest probability, weighted by distance.

### Stage 4: Parameter Update (NU-RLMS)

**4.1 Compute Jacobian J**
- Partial derivatives of landmark positions w.r.t. parameters p
- Evaluated at current parameter estimate
- Size: 2n × (6+m) where n=number of landmarks

**4.2 Apply Reliability Weighting**
- Weight mean-shift vectors by W (patch expert reliabilities)
- More reliable patches get higher influence

**4.3 Regularization**
- Apply shape prior Λ⁻¹ to prevent impossible shapes
- Regularization strength r (typically 25)

**4.4 Solve for Update**
```
Δp = -(JᵀWJ + rΛ⁻¹)⁻¹(rΛ⁻¹p - JᵀWv)
```

**4.5 Update Parameters**
```
p ← p + Δp
```

### Stage 5: Convergence Check

**Termination Criteria:**
- ||Δp|| < threshold (parameter change is small)
- Maximum iterations reached (typically 10-20 per scale)
- Alignment error stops decreasing

If not converged, return to Stage 2.

### Stage 6: Scale Progression

Once converged at current scale:
- Move to next finer scale
- Use current parameters as initialization
- Repeat Stages 2-5

## Key Differences from Standard CLM

### 1. Patch Expert: SVR → LNF

**Standard CLM (SVR):**
- Linear Support Vector Regressor
- No spatial relationships
- Noisy response maps

**CLNF (LNF):**
- Non-linear neural network layer
- Spatial smoothness (gk features)
- Sparsity constraints (lk features)
- Cleaner response maps with single clear peak

### 2. Optimization: RLMS → NU-RLMS

**Standard RLMS:**
- Treats all patch experts equally
- W = I (identity matrix)

**NU-RLMS:**
- Weights patches by reliability ci
- Learned from validation performance
- Better accuracy on challenging cases

## Computational Details

### Training Phase

**Data Requirements:**
- Images with labeled landmarks
- Multiple scales and orientations
- Positive and negative samples (on/off landmark)

**Training Steps:**

1. **PDM Training:**
   - Collect landmark positions from training set
   - Perform PCA to get mean shape x̄ and components Φ
   - Typically retain 95-99% variance

2. **LNF Patch Expert Training:**
   - For each landmark, sample patches at various offsets
   - Synthetic response maps: yi = N(zi; z, σ=1)
   - Optimize parameters {α, β, γ, Θ} using constrained BFGS
   - Maximize log-likelihood: L(α,β,γ,Θ) = ΣM log P(y⁽q⁾|x⁽q⁾)

3. **Reliability Estimation:**
   - Evaluate each patch expert on validation set
   - Compute correlation between predicted and ground truth
   - Store as ci for each patch expert

### Inference Phase

**Per Frame Processing:**
1. Face detection: ~10ms
2. Multi-scale LNF evaluation: ~80-400ms (depends on scale)
3. NU-RLMS optimization: ~20ms per iteration
4. Total: ~100-500ms per frame (Matlab implementation)

**C++ Implementation:**
- 10 images/sec on Multi-PIE (constrained)
- 2 images/sec on in-the-wild data
- 3.5GHz dual core Intel i7

## Mathematical Formulation Summary

### Complete Optimization Problem

```
p* = arg min_p [R(p) + Σi Di(xi; I)]
```

Where:
- **R(p) = ||p||²Λ⁻¹**: Shape regularization
- **Di(xi; I)**: Misalignment of landmark i

### Iterative Solution (NU-RLMS)

```
p⁽ᵗ⁺¹⁾ = p⁽ᵗ⁾ + Δp
Δp = -(JᵀWJ + rΛ⁻¹)⁻¹(rΛ⁻¹p - JᵀWv)
```

### LNF Response (Gaussian Form)

```
P(y|X) = N(y; μ, Σ)
μ = Σd
Σ⁻¹ = 2(A + B + C)
d = 2αᵀh(ΘX)
```

Where A, B, C encode vertex, similarity, and sparsity features respectively.

## Practical Considerations

### 1. Multi-View Robustness
- Train separate patch experts for different orientations
- Initialize at multiple poses, select best fit
- Typical: 5 initializations at (0,0,0), (0,±30,0), (0,0,±30) degrees

### 2. Failure Detection
- Monitor alignment error R(p) + Σi Di(xi; I)
- High error indicates failed detection
- Can trigger re-initialization or MTCNN fallback

### 3. Temporal Smoothing
- Track parameters across frames
- Use previous frame as initialization
- Apply temporal filtering on landmarks

## Comparison with PyFaceAU Pipeline

### OpenFace (Using CLNF):
```
MTCNN bbox → CLNF (starts from bbox only) → 68 landmarks
```
- CLNF estimates ALL 68 landmarks from scratch
- Only uses bbox for initialization
- MTCNN's 5 landmarks discarded

### PyFaceAU:
```
RetinaFace (bbox + 5 landmarks) → PFLD → 68 landmarks → SVR CLNF refinement
```
- PFLD directly predicts 68 landmarks
- Uses 5 landmarks for alignment
- SVR CLNF refines specific landmarks (targeted refinement)

**Key Difference:** OpenFace CLNF does full 68-point estimation, PyFaceAU uses targeted refinement on pre-detected 68 points.

## References

- Baltrusaitis et al. "Constrained Local Neural Fields for robust facial landmark detection in the wild" ICCV 2013
- Saragih et al. "Deformable Model Fitting by Regularized Landmark Mean-Shift" IJCV 2011
- Implementation: OpenFace 2.2 (~/repo/fea_tool/external_libs/openFace/)
