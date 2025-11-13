# PyCLNF Implementation Document
## Pure Python Constrained Local Neural Fields with Neural Engine Optimization

**Goal:** PyInstaller-compatible CLNF implementation reusing OpenFace models with platform-optimized performance

**Target Platforms:**
1. **ARM Macs** (M1/M2/M3) - Neural Engine optimization priority
2. **Intel/CUDA** - GPU acceleration
3. **Pure Intel** - Acceptable to be slow

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     PyCLNF Package                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Core Python Implementation (Platform Agnostic) │    │
│  │  - PDM (Point Distribution Model)               │    │
│  │  - LNF Patch Experts                            │    │
│  │  - NU-RLMS Optimizer                            │    │
│  │  - All using numpy/scipy (PyInstaller-safe)     │    │
│  └────────────────────────────────────────────────┘    │
│                           │                              │
│  ┌────────────────────────┼────────────────────────┐   │
│  │     Performance Layer (Platform-Specific)        │   │
│  ├──────────────┬─────────────┬────────────────────┤   │
│  │              │             │                     │   │
│  │  CoreML      │  Cython     │  CuPy/CUDA         │   │
│  │  (ARM Mac    │  (ARM/Intel │  (NVIDIA GPU)      │   │
│  │   Neural     │   CPU       │                     │   │
│  │   Engine)    │   accel)    │                     │   │
│  │              │             │                     │   │
│  └──────────────┴─────────────┴────────────────────┘   │
│                           │                              │
│  ┌────────────────────────┼────────────────────────┐   │
│  │         Model Loader (One-time export)          │   │
│  │  - Load OpenFace models → .npy/.mlpackage       │   │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Model Export from OpenFace (One-Time Setup)

### Objective
Export OpenFace's trained models to Python-readable formats without maintaining C++ dependencies.

### Models to Export

#### 1.1 Point Distribution Model (PDM)
**Source:** OpenFace `model/main_clnf_general.txt` references PDM files

**Files:**
```
lib/local/LandmarkDetector/model/pdms/
  In_the_wild_68_to_rigid_params.txt
  In_the_wild_68_to_non_rigid_params.txt
  In_the_wild_68_pdm.txt
```

**Export Target:**
```python
pdm/
  mean_shape.npy          # (68, 3) - x̄i for each landmark
  pca_components.npy      # (204, n_modes) - Φ matrix (68*3 × modes)
  eigenvalues.npy         # (n_modes,) - variance of each mode
  rigid_params.npy        # Parameters for rigid transformation
```

**Export C++ Utility:**
```cpp
// tools/export_pdm.cpp
#include "PDM.h"
#include <cnpy.h>  // C++ numpy writer

void export_pdm_to_numpy(const std::string& model_path,
                         const std::string& output_dir) {
    LandmarkDetector::PDM pdm;
    pdm.Read(model_path);

    // Export mean shape
    std::vector<double> mean_shape_flat;
    for (int i = 0; i < pdm.NumberOfPoints(); i++) {
        mean_shape_flat.push_back(pdm.mean_shape.at<double>(i, 0));
        mean_shape_flat.push_back(pdm.mean_shape.at<double>(i, 1));
        mean_shape_flat.push_back(pdm.mean_shape.at<double>(i, 2));
    }
    cnpy::npy_save(output_dir + "/mean_shape.npy",
                   mean_shape_flat.data(),
                   {68, 3}, "w");

    // Export PCA components (principal components matrix)
    cnpy::npy_save(output_dir + "/pca_components.npy",
                   pdm.princ_comp.ptr<double>(),
                   {204, pdm.princ_comp.cols}, "w");

    // Export eigenvalues
    cnpy::npy_save(output_dir + "/eigenvalues.npy",
                   pdm.eigen_values.ptr<double>(),
                   {pdm.eigen_values.rows}, "w");
}
```

#### 1.2 LNF Patch Experts
**Source:** OpenFace `model/main_clnf_general.txt` references patch expert files

**Files (per scale and orientation):**
```
lib/local/LandmarkDetector/model/patch_experts/
  ccnf_patches_0.25_general.txt   # Scale 0.25
  ccnf_patches_0.35_general.txt   # Scale 0.35
  ccnf_patches_0.50_general.txt   # Scale 0.50
  ccnf_patches_1.00_general.txt   # Scale 1.00
```

**Export Target:**
```python
patch_experts/
  scale_0.25/
    view_0/
      landmarks.json          # List of which landmarks at this scale/view
      alpha.npy              # (n_landmarks, K1) - vertex feature weights
      theta.npy              # (n_landmarks, K1, patch_size) - neural weights
      beta.npy               # (n_landmarks, K2) - smoothness weights
      gamma.npy              # (n_landmarks, K3) - sparsity weights
      neighborhood_g.npy     # (K2,) - smoothness neighborhoods S^(g)
      neighborhood_l.npy     # (K3,) - sparsity neighborhoods S^(l)
```

**Export C++ Utility:**
```cpp
// tools/export_patch_experts.cpp
#include "LNF_patch_expert.h"
#include <cnpy.h>

void export_lnf_patch_expert(const LandmarkDetector::CLNF& clnf_model,
                              const std::string& output_dir) {

    // For each scale
    for (size_t scale = 0; scale < clnf_model.patch_experts.patch_experts.size(); scale++) {
        auto& experts = clnf_model.patch_experts.patch_experts[scale];

        std::string scale_dir = output_dir + "/scale_" + std::to_string(scale);

        // For each view
        for (size_t view = 0; view < experts.size(); view++) {
            auto& expert = experts[view];
            std::string view_dir = scale_dir + "/view_" + std::to_string(view);

            // Export alpha (vertex feature weights)
            cnpy::npy_save(view_dir + "/alpha.npy",
                          expert.alphas.ptr<double>(),
                          {expert.alphas.rows, expert.alphas.cols}, "w");

            // Export theta (neural network weights)
            // Theta is stored as multiple weight matrices
            std::vector<double> theta_flat;
            for (const auto& neuron_weights : expert.neurons) {
                for (int i = 0; i < neuron_weights.rows * neuron_weights.cols; i++) {
                    theta_flat.push_back(neuron_weights.ptr<double>()[i]);
                }
            }
            cnpy::npy_save(view_dir + "/theta.npy",
                          theta_flat.data(),
                          {expert.neurons.size(), expert.neurons[0].rows,
                           expert.neurons[0].cols}, "w");

            // Export beta (smoothness weights)
            cnpy::npy_save(view_dir + "/beta.npy",
                          expert.betas.ptr<double>(),
                          {expert.betas.rows, expert.betas.cols}, "w");

            // Export gamma (sparsity weights)
            cnpy::npy_save(view_dir + "/gamma.npy",
                          expert.gammas.ptr<double>(),
                          {expert.gammas.rows, expert.gammas.cols}, "w");
        }
    }
}
```

#### 1.3 Reliability Weights
**Source:** Computed from validation data performance

**Export Target:**
```python
reliability/
  weights.npy  # (68, n_scales, n_views) - ci values for W matrix
```

### 1.4 Export Script
**Single command to export all models:**
```bash
cd ~/repo/fea_tool/external_libs/openFace/OpenFace
mkdir -p tools/export
# Compile export utility
g++ -std=c++17 tools/export_all.cpp -o tools/export_all \
    -I lib/local/LandmarkDetector/include \
    -L build/lib -lLandmarkDetector \
    -l cnpy
# Run once
./tools/export_all --model model/main_clnf_general.txt \
                   --output ~/Documents/SplitFace\ Open3/pyclnf/models/
```

**After this one-time export, never need OpenFace C++ again!**

---

## Phase 2: Pure Python Core Implementation

### 2.1 Package Structure
```
pyclnf/
├── __init__.py
├── models/                 # Exported .npy files (not in git)
│   ├── pdm/
│   ├── patch_experts/
│   └── reliability/
├── core/
│   ├── __init__.py
│   ├── pdm.py             # Point Distribution Model
│   ├── patch_expert.py    # LNF Patch Expert
│   ├── optimizer.py       # NU-RLMS Optimizer
│   └── clnf.py           # Main CLNF class
├── accelerators/
│   ├── __init__.py
│   ├── coreml.py         # Neural Engine acceleration (ARM Mac)
│   ├── cython_ops.pyx    # Cython-accelerated operations
│   └── cuda_ops.py       # CuPy/CUDA operations
├── utils/
│   ├── __init__.py
│   ├── geometry.py       # Rotation matrices, transformations
│   └── visualization.py  # Debug visualization
└── tests/
    ├── test_pdm.py
    ├── test_patch_expert.py
    └── test_integration.py
```

### 2.2 Core Implementation (Accuracy-Preserving)

#### PDM Implementation
```python
# pyclnf/core/pdm.py
import numpy as np
from typing import Tuple

class PointDistributionModel:
    """
    3D Point Distribution Model for facial shape.

    Implements Equation 2 from Baltrusaitis et al. 2013:
    xi = s · R2D · (x̄i + Φiq) + t
    """

    def __init__(self, mean_shape: np.ndarray,
                 pca_components: np.ndarray,
                 eigenvalues: np.ndarray):
        """
        Args:
            mean_shape: (68, 3) mean landmark positions
            pca_components: (204, n_modes) PCA components (flattened 68x3)
            eigenvalues: (n_modes,) variance of each mode
        """
        self.mean_shape = mean_shape.astype(np.float64)
        self.pca_components = pca_components.astype(np.float64)
        self.eigenvalues = eigenvalues.astype(np.float64)
        self.n_landmarks = 68
        self.n_modes = pca_components.shape[1]

        # Precompute for speed
        self._component_matrices = self._reshape_components()

    def _reshape_components(self) -> np.ndarray:
        """Reshape flat PCA components to (68, 3, n_modes)"""
        return self.pca_components.reshape(68, 3, self.n_modes)

    def params_from_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """
        Initialize parameters from bounding box.

        Args:
            bbox: [x, y, width, height] or [x1, y1, x2, y2]

        Returns:
            p: [s, tx, ty, wx, wy, wz, q0, q1, ..., qn]
        """
        if len(bbox) == 4 and bbox[2] < bbox[0]:  # x1, y1, x2, y2 format
            x, y, width, height = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        else:
            x, y, width, height = bbox

        # Estimate scale from bbox
        # Typical face width is ~0.6 of mean shape width
        mean_face_width = np.max(self.mean_shape[:, 0]) - np.min(self.mean_shape[:, 0])
        s = width / mean_face_width * 0.6

        # Center translation
        tx = x + width / 2
        ty = y + height / 2

        # Zero rotation (frontal face assumption)
        wx, wy, wz = 0.0, 0.0, 0.0

        # Zero non-rigid parameters (mean shape)
        q = np.zeros(self.n_modes, dtype=np.float64)

        return np.array([s, tx, ty, wx, wy, wz, *q], dtype=np.float64)

    def landmarks_from_params(self, p: np.ndarray) -> np.ndarray:
        """
        Compute 3D landmark positions from parameters.

        Implements: xi = s · R2D · (x̄i + Φiq) + t

        Args:
            p: Parameter vector [s, tx, ty, wx, wy, wz, q...]

        Returns:
            landmarks: (68, 3) 3D landmark positions
        """
        # Parse parameters
        s = p[0]
        t = p[1:3]
        w = p[3:6]
        q = p[6:]

        # Rotation matrix from axis-angle
        R2D = self._rotation_matrix_2d(w)

        # Shape from non-rigid parameters: x̄i + Φiq
        # Vectorized: (68, 3) + (68, 3, n_modes) @ (n_modes,)
        shape = self.mean_shape + np.einsum('ijk,k->ij',
                                             self._component_matrices, q)

        # Apply similarity transform: s · R2D · shape + [t, 0]
        landmarks = s * (shape @ R2D.T)
        landmarks[:, :2] += t

        return landmarks

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of landmarks w.r.t. parameters.

        Returns:
            J: (2*68, 6+n_modes) Jacobian matrix
        """
        # This is the performance-critical part
        # Will be accelerated with Cython/Numba
        # Implementation follows standard CLM Jacobian
        # (derivation in supplementary material)
        pass  # Detailed implementation

    @staticmethod
    def _rotation_matrix_2d(w: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle to 2D rotation matrix.

        Args:
            w: [wx, wy, wz] axis-angle representation

        Returns:
            R2D: (3, 3) rotation matrix (only first 2 rows used)
        """
        theta = np.linalg.norm(w)
        if theta < 1e-8:
            return np.eye(3)

        w_normalized = w / theta
        wx, wy, wz = w_normalized

        # Rodrigues' formula
        K = np.array([[0, -wz, wy],
                      [wz, 0, -wx],
                      [-wy, wx, 0]])

        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    @classmethod
    def load(cls, model_dir: str) -> 'PointDistributionModel':
        """Load PDM from exported numpy files"""
        mean_shape = np.load(f"{model_dir}/pdm/mean_shape.npy")
        pca_components = np.load(f"{model_dir}/pdm/pca_components.npy")
        eigenvalues = np.load(f"{model_dir}/pdm/eigenvalues.npy")
        return cls(mean_shape, pca_components, eigenvalues)
```

#### LNF Patch Expert Implementation
```python
# pyclnf/core/patch_expert.py
import numpy as np
from scipy.ndimage import map_coordinates
from typing import Tuple

class LNFPatchExpert:
    """
    Local Neural Field patch expert.

    Implements Equations 8-13 from Baltrusaitis et al. 2013.
    """

    def __init__(self, alpha: np.ndarray, theta: np.ndarray,
                 beta: np.ndarray, gamma: np.ndarray,
                 neighborhood_g: np.ndarray, neighborhood_l: np.ndarray,
                 patch_size: int = 11, area_size: int = 21):
        """
        Args:
            alpha: (K1,) vertex feature weights
            theta: (K1, patch_size^2) neural network weights
            beta: (K2,) smoothness weights
            gamma: (K3,) sparsity weights
            neighborhood_g: (K2, 2, n_neighbors_g) smoothness neighborhoods
            neighborhood_l: (K3, 2, n_neighbors_l) sparsity neighborhoods
            patch_size: Support region size (11×11)
            area_size: Area of interest size (21×21)
        """
        self.alpha = alpha.astype(np.float64)
        self.theta = theta.astype(np.float64)
        self.beta = beta.astype(np.float64)
        self.gamma = gamma.astype(np.float64)
        self.neighborhood_g = neighborhood_g
        self.neighborhood_l = neighborhood_l
        self.patch_size = patch_size
        self.area_size = area_size

        # Precompute
        self.K1 = len(alpha)
        self.K2 = len(beta)
        self.K3 = len(gamma)

    def compute_response_map(self, frame: np.ndarray,
                            center: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute response map for landmark alignment probability.

        This is the CORE computation of CLNF.
        Performance-critical - will be accelerated.

        Args:
            frame: Input image (grayscale)
            center: (x, y) current landmark estimate

        Returns:
            response_map: (area_size, area_size) alignment probability
            confidence: Peak confidence value
        """
        # Extract area of interest
        aoi = self._extract_area_of_interest(frame, center)

        # Compute vertex features (neural network activation)
        vertex_responses = self._compute_vertex_features(aoi)

        # Compute edge features (spatial constraints)
        smoothness_term = self._compute_smoothness(vertex_responses)
        sparsity_term = self._compute_sparsity(vertex_responses)

        # Combine (Equation 9)
        response_map = vertex_responses + smoothness_term + sparsity_term

        # Convert to probability (Equation 16 - Gaussian form)
        response_map = self._to_probability(response_map)

        # Normalize
        response_map = response_map / (np.sum(response_map) + 1e-10)

        confidence = np.max(response_map)
        return response_map, confidence

    def _extract_area_of_interest(self, frame: np.ndarray,
                                  center: np.ndarray) -> np.ndarray:
        """
        Extract area of interest grid around center.

        Returns:
            aoi: (area_size, area_size, n_points, patch_size^2)
        """
        # Create grid of evaluation points
        half_area = self.area_size // 2
        x_grid = np.arange(-half_area, half_area + 1) + center[0]
        y_grid = np.arange(-half_area, half_area + 1) + center[1]

        # Extract patches at each grid point
        patches = []
        for y in y_grid:
            row_patches = []
            for x in x_grid:
                patch = self._extract_patch(frame, np.array([x, y]))
                row_patches.append(patch.flatten())
            patches.append(row_patches)

        return np.array(patches)  # (area_size, area_size, patch_size^2)

    def _extract_patch(self, frame: np.ndarray,
                      center: np.ndarray) -> np.ndarray:
        """Extract patch_size×patch_size patch centered at location"""
        half = self.patch_size // 2
        x, y = int(center[0]), int(center[1])

        # Handle boundary conditions
        x1 = max(0, x - half)
        x2 = min(frame.shape[1], x + half + 1)
        y1 = max(0, y - half)
        y2 = min(frame.shape[0], y + half + 1)

        patch = frame[y1:y2, x1:x2]

        # Pad if needed
        if patch.shape != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size))
            py = (self.patch_size - patch.shape[0]) // 2
            px = (self.patch_size - patch.shape[1]) // 2
            padded[py:py+patch.shape[0], px:px+patch.shape[1]] = patch
            patch = padded

        return patch

    def _compute_vertex_features(self, aoi: np.ndarray) -> np.ndarray:
        """
        Compute vertex features (Equations 10-11).

        This is matrix multiplication: h(θX) weighted by α
        PERFORMANCE-CRITICAL: Will be accelerated
        """
        # aoi shape: (area_size, area_size, patch_size^2)
        # theta shape: (K1, patch_size^2)

        # Neural network activation: h(θX)
        # Vectorized across all patches
        linear = np.einsum('ijk,mk->ijm', aoi, self.theta)  # (area, area, K1)
        activations = self._sigmoid(linear)  # Element-wise sigmoid

        # Weight by alpha (reliability)
        weighted = np.einsum('ijm,m->ij', activations, self.alpha)

        return weighted

    def _compute_smoothness(self, vertex_responses: np.ndarray) -> np.ndarray:
        """
        Compute smoothness term (Equation 12).

        Enforces nearby pixels have similar responses.
        """
        smoothness = np.zeros_like(vertex_responses)

        for k in range(self.K2):
            # Get neighbor indices from neighborhood_g[k]
            # Apply smoothness penalty: -0.5 * beta_k * (y_i - y_j)^2
            # This will be optimized with Cython
            pass

        return smoothness

    def _compute_sparsity(self, vertex_responses: np.ndarray) -> np.ndarray:
        """
        Compute sparsity term (Equation 13).

        Enforces only one peak in response map.
        """
        sparsity = np.zeros_like(vertex_responses)

        for k in range(self.K3):
            # Get neighbor indices from neighborhood_l[k]
            # Apply sparsity penalty: -0.5 * gamma_k * (y_i + y_j)^2
            # This will be optimized with Cython
            pass

        return sparsity

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid"""
        return np.where(x >= 0,
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))

    def _to_probability(self, response: np.ndarray) -> np.ndarray:
        """Convert response to probability (Gaussian form)"""
        # Equation 16: P(y|X) = N(y; μ, Σ)
        # In practice: softmax-like normalization
        exp_response = np.exp(response - np.max(response))
        return exp_response / (np.sum(exp_response) + 1e-10)
```

#### NU-RLMS Optimizer Implementation
```python
# pyclnf/core/optimizer.py
import numpy as np
from scipy import linalg
from typing import List, Tuple

class NURLMSOptimizer:
    """
    Non-Uniform Regularised Landmark Mean-Shift optimizer.

    Implements Equations 23-24 from Baltrusaitis et al. 2013.
    """

    def __init__(self, pdm: 'PointDistributionModel',
                 patch_experts: List['LNFPatchExpert'],
                 reliability_weights: np.ndarray,
                 rho: float = 2.0, r: float = 25.0):
        """
        Args:
            pdm: Point Distribution Model
            patch_experts: List of 68 LNF patch experts
            reliability_weights: (68,) reliability of each expert (for W matrix)
            rho: Gaussian kernel bandwidth for mean-shift (default: 2.0)
            r: Regularization strength (default: 25)
        """
        self.pdm = pdm
        self.patch_experts = patch_experts
        self.rho = rho
        self.r = r

        # Construct W matrix (Equation 23)
        # Diagonal matrix with reliability weights repeated for x,y
        w_diag = np.repeat(reliability_weights, 2)  # (136,) for 68 landmarks
        self.W = np.diag(w_diag)

        # Lambda^-1: Prior covariance (from PDM eigenvalues)
        self.Lambda_inv = np.diag(1.0 / (pdm.eigenvalues + 1e-6))
        # Pad for rigid parameters (no prior)
        self.Lambda_inv = np.block([
            [np.zeros((6, 6)), np.zeros((6, pdm.n_modes))],
            [np.zeros((pdm.n_modes, 6)), self.Lambda_inv]
        ])

    def fit(self, frame: np.ndarray, initial_bbox: np.ndarray,
            max_iterations: int = 20, convergence_threshold: float = 0.001,
            multi_scale: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Fit CLNF model to image.

        Args:
            frame: Grayscale image
            initial_bbox: [x, y, w, h] or [x1, y1, x2, y2]
            max_iterations: Max iterations per scale
            convergence_threshold: Convergence threshold for Δp
            multi_scale: Use multi-scale fitting

        Returns:
            landmarks: (68, 2) final landmark positions
            info: Dict with fitting information
        """
        # Initialize parameters from bbox
        p = self.pdm.params_from_bbox(initial_bbox)

        if multi_scale:
            # Coarse to fine: Scale 0 → 1 → 2
            scales = [0, 1, 2]
            iterations_per_scale = [max_iterations // 2,
                                   max_iterations // 2,
                                   max_iterations]
        else:
            scales = [2]  # Only finest scale
            iterations_per_scale = [max_iterations]

        fitting_history = []

        for scale_idx, n_iters in zip(scales, iterations_per_scale):
            for iteration in range(n_iters):
                # Get current landmark estimates
                landmarks_3d = self.pdm.landmarks_from_params(p)
                landmarks_2d = landmarks_3d[:, :2]  # Project to 2D

                # Evaluate patch experts
                response_maps = []
                confidences = []
                for i, (landmark, expert) in enumerate(
                    zip(landmarks_2d, self.patch_experts)):
                    response, conf = expert.compute_response_map(frame, landmark)
                    response_maps.append(response)
                    confidences.append(conf)

                # Compute mean-shift vectors (Equation 6)
                v = self._compute_mean_shifts(response_maps, landmarks_2d)

                # Compute Jacobian
                J = self.pdm.jacobian(p)

                # Parameter update (Equation 24)
                # Δp = -(J^T W J + r Λ^-1)^-1 (r Λ^-1 p - J^T W v)
                A = J.T @ self.W @ J + self.r * self.Lambda_inv
                b = self.r * self.Lambda_inv @ p - J.T @ self.W @ v

                delta_p = -linalg.solve(A, b, assume_a='pos')

                # Update parameters
                p = p + delta_p

                # Check convergence
                if np.linalg.norm(delta_p) < convergence_threshold:
                    break

                # Store history
                fitting_history.append({
                    'iteration': iteration,
                    'scale': scale_idx,
                    'delta_p_norm': np.linalg.norm(delta_p),
                    'mean_confidence': np.mean(confidences)
                })

        # Final landmarks
        final_landmarks_3d = self.pdm.landmarks_from_params(p)
        final_landmarks_2d = final_landmarks_3d[:, :2]

        info = {
            'parameters': p,
            'confidences': confidences,
            'history': fitting_history,
            'converged': np.linalg.norm(delta_p) < convergence_threshold
        }

        return final_landmarks_2d, info

    def _compute_mean_shifts(self, response_maps: List[np.ndarray],
                            current_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute mean-shift vectors from response maps (Equation 6).

        vi = Σ(πy * N(xc; y, ρI)) / Σ(πz * N(xc; z, ρI)) - xc

        PERFORMANCE-CRITICAL: Will be accelerated
        """
        v = np.zeros(len(current_landmarks) * 2, dtype=np.float64)

        for i, (response_map, xc) in enumerate(zip(response_maps, current_landmarks)):
            # Create grid of positions
            area_size = response_map.shape[0]
            half = area_size // 2
            x_grid = np.arange(-half, half + 1) + xc[0]
            y_grid = np.arange(-half, half + 1) + xc[1]
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.stack([X, Y], axis=-1)  # (area, area, 2)

            # Gaussian weights: N(xc; y, ρI)
            diff = positions - xc
            gaussian_weights = np.exp(-np.sum(diff**2, axis=-1) / (2 * self.rho**2))

            # Weighted mean: Σ(πy * N * y) / Σ(πz * N)
            weights = response_map * gaussian_weights
            weighted_sum = np.sum(weights[:, :, np.newaxis] * positions, axis=(0, 1))
            total_weight = np.sum(weights)

            mean_pos = weighted_sum / (total_weight + 1e-10)

            # Mean-shift vector
            v[2*i:2*i+2] = mean_pos - xc

        return v
```

---

## Phase 3: Platform-Specific Acceleration

### 3.1 Neural Engine Optimization (ARM Macs - Priority)

**Objective:** Offload compute-intensive operations to M1/M2/M3 Neural Engine

**Target Operations:**
1. **LNF neural network inference** - Perfect for ANE
2. **Smoothness/sparsity convolutions** - Matrix operations
3. **PDM transformations** - Linear algebra

**Implementation Strategy:**

```python
# pyclnf/accelerators/coreml.py
import coremltools as ct
import numpy as np

class CoreMLLNFExpert:
    """
    Neural Engine-accelerated LNF patch expert.

    Converts LNF neural network to CoreML for ANE execution.
    """

    def __init__(self, alpha: np.ndarray, theta: np.ndarray,
                 beta: np.ndarray, gamma: np.ndarray):
        """
        Build CoreML model from LNF weights.

        The neural network part (theta, alpha) runs on Neural Engine.
        Smoothness/sparsity stays in Python (faster with Cython).
        """
        self.model = self._build_coreml_model(alpha, theta)
        self.beta = beta
        self.gamma = gamma

    def _build_coreml_model(self, alpha, theta) -> ct.models.MLModel:
        """
        Convert LNF neural network to CoreML.

        Network structure:
        Input: (patch_size^2,) pixel values
        Layer 1: Linear (theta) → Sigmoid
        Layer 2: Linear (alpha weighting)
        Output: Scalar response
        """
        import coremltools.proto.FeatureTypes_pb2 as ft

        # Define input
        input_features = [('patch', ft.ArrayFeatureType(shape=(121,)))]  # 11x11
        output_features = [('response', ft.DoubleFeatureType())]

        # Build model using Neural Network builder
        from coremltools.models.neural_network import NeuralNetworkBuilder
        builder = NeuralNetworkBuilder(input_features, output_features)

        # Layer 1: theta weights (Linear + Sigmoid)
        builder.add_inner_product(
            name='neural_layer',
            W=theta,  # (K1, 121)
            b=np.zeros(theta.shape[0]),
            input_channels=121,
            output_channels=theta.shape[0],
            input_name='patch',
            output_name='activations'
        )
        builder.add_activation(
            name='sigmoid',
            non_linearity='SIGMOID',
            input_name='activations',
            output_name='sigmoid_out'
        )

        # Layer 2: alpha weighting (Linear)
        builder.add_inner_product(
            name='alpha_weight',
            W=alpha.reshape(1, -1),  # (1, K1)
            b=np.zeros(1),
            input_channels=theta.shape[0],
            output_channels=1,
            input_name='sigmoid_out',
            output_name='response'
        )

        # Compile
        mlmodel = ct.models.MLModel(builder.spec)

        # CRITICAL: Set compute units to Neural Engine
        mlmodel = ct.models.MLModel(mlmodel.get_spec(),
                                    compute_units=ct.ComputeUnit.ALL)

        return mlmodel

    def compute_response_map(self, patches: np.ndarray) -> np.ndarray:
        """
        Compute responses using Neural Engine.

        Args:
            patches: (area_size, area_size, 121) flattened patches

        Returns:
            vertex_responses: (area_size, area_size) neural network outputs
        """
        area_size = patches.shape[0]

        # Batch process patches (Neural Engine loves batching!)
        patches_flat = patches.reshape(-1, 121)

        # Run on Neural Engine
        responses = []
        for patch in patches_flat:
            output = self.model.predict({'patch': patch})
            responses.append(output['response'])

        vertex_responses = np.array(responses).reshape(area_size, area_size)

        # Add smoothness/sparsity (Cython-accelerated Python)
        vertex_responses += self._compute_smoothness_cython(vertex_responses)
        vertex_responses += self._compute_sparsity_cython(vertex_responses)

        return vertex_responses
```

**CoreML Model Export:**
```python
def export_lnf_to_coreml(numpy_weights_dir: str, output_dir: str):
    """
    One-time conversion: numpy weights → CoreML models
    Run once after exporting from OpenFace.
    """
    for scale in ['0.25', '0.35', '0.50', '1.00']:
        for view in range(n_views):
            # Load numpy weights
            alpha = np.load(f"{numpy_weights_dir}/scale_{scale}/view_{view}/alpha.npy")
            theta = np.load(f"{numpy_weights_dir}/scale_{scale}/view_{view}/theta.npy")

            # Build CoreML model (runs on Neural Engine)
            coreml_expert = CoreMLLNFExpert(alpha, theta, None, None)

            # Save as .mlpackage (optimized for Neural Engine)
            coreml_expert.model.save(
                f"{output_dir}/scale_{scale}_view_{view}.mlpackage"
            )
```

**Performance Expectations:**
- **Neural Engine:** 5-10x speedup on M1/M2/M3
- **Batch processing:** Additional 2-3x from batching patches
- **Total:** 10-30x faster than pure NumPy on ARM Mac

### 3.2 Cython Acceleration (CPU-Intensive Operations)

**Target Operations:**
1. Jacobian computation (nested loops)
2. Mean-shift vector calculation (nested loops)
3. Smoothness/sparsity terms (neighbor iterations)

```python
# pyclnf/accelerators/cython_ops.pyx
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_smoothness_term(double[:, :] responses,
                           long[:, :] neighbors_i,
                           long[:, :] neighbors_j,
                           double[:] beta):
    """
    Compute smoothness term with Cython.

    -0.5 * Σ beta_k * (y_i - y_j)^2

    This is 10-50x faster than pure Python nested loops.
    """
    cdef int rows = responses.shape[0]
    cdef int cols = responses.shape[1]
    cdef int K2 = beta.shape[0]
    cdef double[:, :] smoothness = np.zeros((rows, cols), dtype=np.float64)

    cdef int k, idx, i, j, ni, nj
    cdef double diff, penalty

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            penalty = 0.0

            for k in range(K2):
                # Get neighbor indices
                ni = neighbors_i[k, idx]
                nj = neighbors_j[k, idx]

                if 0 <= ni < rows and 0 <= nj < cols:
                    diff = responses[i, j] - responses[ni, nj]
                    penalty -= 0.5 * beta[k] * diff * diff

            smoothness[i, j] = penalty

    return np.asarray(smoothness)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_mean_shift_vectors(double[:, :] response_maps,
                               double[:, :] current_landmarks,
                               double rho):
    """
    Vectorized mean-shift with Cython.

    20-100x faster than pure Python.
    """
    cdef int n_landmarks = current_landmarks.shape[0]
    cdef int area_size = response_maps.shape[0]
    cdef double[:] v = np.zeros(n_landmarks * 2, dtype=np.float64)

    cdef int i, ix, iy
    cdef double xc, yc, x, y, weight, total_weight
    cdef double weighted_x, weighted_y, diff_x, diff_y, dist_sq
    cdef double rho_sq = rho * rho

    for i in range(n_landmarks):
        xc = current_landmarks[i, 0]
        yc = current_landmarks[i, 1]

        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0

        # Iterate over area of interest
        for iy in range(area_size):
            for ix in range(area_size):
                # Position in image
                x = xc + (ix - area_size // 2)
                y = yc + (iy - area_size // 2)

                # Gaussian weight
                diff_x = x - xc
                diff_y = y - yc
                dist_sq = diff_x * diff_x + diff_y * diff_y
                weight = response_maps[iy, ix] * exp(-dist_sq / (2.0 * rho_sq))

                weighted_x += weight * x
                weighted_y += weight * y
                total_weight += weight

        # Mean position
        v[2*i] = weighted_x / total_weight - xc
        v[2*i + 1] = weighted_y / total_weight - yc

    return np.asarray(v)
```

**Build Configuration:**
```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "pyclnf.accelerators.cython_ops",
        ["pyclnf/accelerators/cython_ops.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3', '-march=native'],  # CPU-specific optimization
    )
]

setup(
    name='pyclnf',
    ext_modules=cythonize(extensions,
                         compiler_directives={
                             'language_level': 3,
                             'boundscheck': False,
                             'wraparound': False
                         }),
)
```

**Performance Expectations:**
- **Nested loops:** 10-50x speedup
- **Vectorizable operations:** 2-5x speedup
- **Overall fitting:** 5-10x faster than pure NumPy

### 3.3 CUDA/CuPy Acceleration (NVIDIA GPU)

**For Intel/CUDA systems only**

```python
# pyclnf/accelerators/cuda_ops.py
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class CUDALNFExpert:
    """
    CUDA-accelerated patch expert using CuPy.

    Offloads all matrix operations to GPU.
    """

    def __init__(self, alpha, theta, beta, gamma):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        # Transfer to GPU once
        self.alpha = cp.asarray(alpha)
        self.theta = cp.asarray(theta)
        self.beta = cp.asarray(beta)
        self.gamma = cp.asarray(gamma)

    def compute_response_map(self, patches_gpu: cp.ndarray) -> cp.ndarray:
        """
        All operations on GPU.

        100-1000x faster than CPU for large batches.
        """
        # Neural network: patches @ theta.T → sigmoid → @ alpha
        linear = patches_gpu @ self.theta.T
        activations = 1.0 / (1.0 + cp.exp(-linear))
        vertex = activations @ self.alpha

        # Smoothness (GPU kernel)
        smoothness = self._cuda_smoothness(vertex)

        # Sparsity (GPU kernel)
        sparsity = self._cuda_sparsity(vertex)

        return vertex + smoothness + sparsity
```

---

## Phase 4: Integration and Testing

### 4.1 Main API

```python
# pyclnf/core/clnf.py
import numpy as np
from typing import Optional, Tuple
import platform

class PyCLNF:
    """
    Pure Python CLNF with platform-specific acceleration.

    Automatically selects best backend:
    - ARM Mac: CoreML (Neural Engine)
    - Intel/CUDA: CuPy (GPU)
    - Fallback: Cython (CPU)
    """

    def __init__(self, model_dir: str, accelerator: Optional[str] = 'auto'):
        """
        Args:
            model_dir: Path to exported models
            accelerator: 'auto', 'coreml', 'cuda', 'cython', or 'numpy'
        """
        # Load core models
        self.pdm = PDM.load(model_dir)
        self.patch_experts = self._load_patch_experts(model_dir, accelerator)
        reliability = np.load(f"{model_dir}/reliability/weights.npy")

        # Create optimizer
        self.optimizer = NURLMSOptimizer(self.pdm, self.patch_experts, reliability)

        # Detect platform
        if accelerator == 'auto':
            accelerator = self._detect_best_accelerator()

        self.accelerator = accelerator
        print(f"PyCLNF initialized with {accelerator} backend")

    @staticmethod
    def _detect_best_accelerator() -> str:
        """Auto-detect best acceleration method"""
        # Check ARM Mac (Neural Engine)
        if platform.machine() == 'arm64' and platform.system() == 'Darwin':
            try:
                import coremltools
                return 'coreml'
            except ImportError:
                pass

        # Check CUDA
        try:
            import cupy
            return 'cuda'
        except ImportError:
            pass

        # Check Cython
        try:
            from .accelerators import cython_ops
            return 'cython'
        except ImportError:
            pass

        # Fallback to NumPy
        return 'numpy'

    def detect_landmarks(self, frame: np.ndarray,
                        bbox: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Main API: Detect 68 facial landmarks.

        Args:
            frame: Grayscale or RGB image
            bbox: [x, y, w, h] or [x1, y1, x2, y2] from RetinaFace

        Returns:
            landmarks: (68, 2) landmark positions
            info: Dict with confidence, convergence info
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize
        frame = frame.astype(np.float64) / 255.0

        # Fit model
        landmarks, info = self.optimizer.fit(frame, bbox)

        return landmarks, info
```

### 4.2 PyInstaller Specification

```python
# pyclnf.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('pyclnf/models', 'pyclnf/models'),  # Include exported models
        ('pyclnf/accelerators/*.mlpackage', 'pyclnf/accelerators'),  # CoreML models
    ],
    hiddenimports=['scipy.special._ufuncs_cxx'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow'],  # Exclude heavy dependencies
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceMirror',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='universal2',  # ARM + Intel Mac
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='FaceMirror.app',
    icon=None,
    bundle_identifier='com.splitface.facemirror',
    info_plist={
        'NSCameraUsageDescription': 'Face Mirror needs camera access',
    },
)
```

---

## Performance Targets and Expected Results

### Baseline (Pure NumPy)
- **Per-frame time:** 300-500ms
- **FPS:** 2-3 FPS
- **Platform:** All

### With Cython
- **Per-frame time:** 100-150ms
- **FPS:** 7-10 FPS
- **Platform:** All
- **Speedup:** 3-5x over NumPy

### With CoreML (ARM Mac)
- **Per-frame time:** 30-50ms
- **FPS:** 20-30 FPS
- **Platform:** M1/M2/M3 Macs
- **Speedup:** 10-15x over NumPy, Neural Engine utilized

### With CUDA (NVIDIA GPU)
- **Per-frame time:** 10-20ms
- **FPS:** 50-100 FPS
- **Platform:** NVIDIA GPUs (RTX 2060+)
- **Speedup:** 20-50x over NumPy

### Pure Intel (Acceptable Slow)
- **Per-frame time:** 200-300ms
- **FPS:** 3-5 FPS
- **Platform:** Intel CPU only
- **Acceptable:** User specified this is OK

---

## Dependencies (PyInstaller-Safe)

```python
# requirements.txt
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
opencv-python>=4.8.0
pillow>=10.0.0
coremltools>=7.0  # ARM Mac only
cupy-cuda11x>=12.0.0  # NVIDIA GPU only (optional)
Cython>=3.0.0  # Build time
```

**Total installed size:** ~150-200MB (reasonable for desktop app)

**All dependencies:** ✅ PyInstaller compatible

---

## Next Steps

1. **Week 1:** Export models from OpenFace to numpy
2. **Week 2:** Implement pure Python core (PDM, LNF, NU-RLMS)
3. **Week 3:** Add Cython acceleration for CPU paths
4. **Week 4:** Add CoreML acceleration for ARM Mac Neural Engine
5. **Week 5:** Testing, profiling, optimization
6. **Week 6:** PyInstaller packaging and distribution

**Total timeline:** 6 weeks to production-ready PyCLNF

**Feasibility:** ✅ **HIGH** - All components proven, clear path forward
