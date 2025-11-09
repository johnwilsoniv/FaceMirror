"""
LNF Patch Expert - Response map computation for CLNF

Implements the Linear Neural Field patch expert from OpenFace CCNF model.

The patch expert computes a response map R(x, y) for each landmark that indicates
the likelihood of the landmark being at position (x, y) in the image patch.

Response computation:
    R(x, y) = Σ_k β_k * σ(α_k * ||w_k^T * f(x, y) + b_k||)

Where:
    - f(x, y): Image features at position (x, y)
    - w_k: Weight vector for neuron k
    - b_k: Bias for neuron k
    - α_k: Scaling factor for neuron k
    - β_k: Linear combination weight for neuron k
    - σ: Sigmoid activation function
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import cv2


class CCNFPatchExpert:
    """CCNF patch expert for a single landmark at a specific scale."""

    def __init__(self, patch_dir: str):
        """
        Load CCNF patch expert from exported NumPy files.

        Args:
            patch_dir: Directory containing metadata.npz and neuron_*.npz files
        """
        self.patch_dir = Path(patch_dir)

        # Load patch metadata
        meta = np.load(self.patch_dir / 'metadata.npz')
        self.width = int(meta['width'])
        self.height = int(meta['height'])
        self.betas = meta['betas']
        self.patch_confidence = float(meta['patch_confidence'])

        # Count neuron files to determine num_neurons
        neuron_files = sorted(self.patch_dir.glob('neuron_*.npz'))
        self.num_neurons = len(neuron_files)

        # Load neurons
        self.neurons = []
        for neuron_file in neuron_files:
            neuron_data = np.load(neuron_file)

            neuron = {
                'type': int(neuron_data['neuron_type']),
                'weights': neuron_data['weights'],
                'bias': float(neuron_data['bias']),
                'alpha': float(neuron_data['alpha']),
                'norm_weights': float(neuron_data['norm_weights'])
            }
            self.neurons.append(neuron)

    def compute_response(self, image_patch: np.ndarray) -> float:
        """
        Compute response (confidence) for this patch expert.

        The response indicates how likely the landmark is at the CENTER
        of this image patch.

        In CCNF, neurons are organized into groups by sigma component (feature scale).
        The betas weight the contribution of each sigma component group.

        Args:
            image_patch: Grayscale image patch, shape (height, width)

        Returns:
            response: Scalar confidence value
        """
        # Ensure patch is correct size
        if image_patch.shape != (self.height, self.width):
            image_patch = cv2.resize(image_patch, (self.width, self.height))

        # Extract features from patch (gradient magnitude)
        features = self._extract_features(image_patch)

        # Group neurons by sigma component (approximately equal division)
        n_sigmas = len(self.betas)
        neurons_per_sigma = len(self.neurons) / n_sigmas

        # Accumulate weighted responses per sigma component
        total_response = 0.0

        for sigma_idx in range(n_sigmas):
            # Get neurons for this sigma component
            start_idx = int(sigma_idx * neurons_per_sigma)
            end_idx = int((sigma_idx + 1) * neurons_per_sigma) if sigma_idx < n_sigmas - 1 else len(self.neurons)

            # Sum neuron responses for this sigma component
            sigma_response = 0.0
            for neuron_idx in range(start_idx, end_idx):
                neuron = self.neurons[neuron_idx]
                neuron_response = self._compute_neuron_response(features, neuron)
                sigma_response += neuron_response

            # Weight by beta for this sigma component
            total_response += self.betas[sigma_idx] * sigma_response

        return float(total_response)

    def _extract_features(self, image_patch: np.ndarray) -> np.ndarray:
        """
        Extract features from image patch.

        For CCNF, uses gradient magnitude as the primary feature.

        Args:
            image_patch: Grayscale image patch, shape (height, width)

        Returns:
            features: Gradient magnitude feature map, shape (height, width)
        """
        # Convert to float
        patch_float = image_patch.astype(np.float32) / 255.0

        # Compute gradients using Sobel
        grad_x = cv2.Sobel(patch_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch_float, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude (edge strength)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        return grad_mag

    def _compute_neuron_response(self, features: np.ndarray, neuron: dict) -> np.ndarray:
        """
        Compute response for a single neuron using correlation.

        CCNF neurons compute: response = Σ(pixels) W ⊙ f
        Where ⊙ is element-wise multiplication (correlation).

        Then apply: σ(α * (response + bias))

        Args:
            features: Feature map (gradient magnitude), shape (height, width)
            neuron: Neuron parameters (weights, bias, alpha)

        Returns:
            response: Single scalar response value
        """
        weights = neuron['weights']  # Shape: (height, width)
        bias = neuron['bias']
        alpha = neuron['alpha']

        # Ensure features and weights are same size
        if features.shape != weights.shape:
            features = cv2.resize(features, (weights.shape[1], weights.shape[0]))

        # Compute correlation (element-wise multiplication and sum)
        # This computes the dot product between weight vector and feature vector
        response_value = np.sum(weights * features) + bias

        # Apply sigmoid activation: σ(α * response)
        # Return as a scalar - we'll accumulate these across neurons
        response = self._sigmoid(alpha * response_value)

        return response

    def _sigmoid(self, x):
        """Numerically stable sigmoid function."""
        if isinstance(x, np.ndarray):
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )
        else:
            # Scalar version
            if x >= 0:
                return 1 / (1 + np.exp(-x))
            else:
                return np.exp(x) / (1 + np.exp(x))

    def get_info(self) -> dict:
        """Get patch expert information."""
        return {
            'width': self.width,
            'height': self.height,
            'num_neurons': self.num_neurons,
            'patch_confidence': self.patch_confidence,
            'num_betas': len(self.betas)
        }


class CCNFModel:
    """
    Complete CCNF model with multi-view, multi-scale patch experts.

    Manages loading patch experts for:
    - Multiple views (face orientations: frontal, profile, etc.)
    - Multiple scales (0.25, 0.35, 0.5 × interocular distance)
    - 68 landmarks per view
    """

    def __init__(self, model_base_dir: str, scales: Optional[List[float]] = None):
        """
        Load CCNF model from exported NumPy files.

        Args:
            model_base_dir: Base directory containing exported_ccnf_* folders
            scales: List of scales to load (default: [0.25, 0.35, 0.5])
        """
        self.model_base_dir = Path(model_base_dir)
        self.scales = scales or [0.25, 0.35, 0.5]

        # Load multi-scale models
        self.scale_models = {}
        for scale in self.scales:
            scale_dir = self.model_base_dir / f'exported_ccnf_{scale}'
            if scale_dir.exists():
                self.scale_models[scale] = self._load_scale_model(scale_dir)
            else:
                print(f"Warning: Scale {scale} model not found at {scale_dir}")

    def _load_scale_model(self, scale_dir: Path) -> dict:
        """
        Load all patch experts for one scale.

        Args:
            scale_dir: Directory for one scale (e.g., exported_ccnf_0.25/)

        Returns:
            Dictionary with structure: {view_idx: {landmark_idx: CCNFPatchExpert}}
        """
        # Load global metadata
        global_meta = np.load(scale_dir / 'global_metadata.npz')
        num_views = int(global_meta['num_views'])
        num_landmarks = int(global_meta['num_landmarks'])

        scale_model = {
            'num_views': num_views,
            'num_landmarks': num_landmarks,
            'patch_scaling': float(global_meta['patch_scaling']),
            'views': {}
        }

        # Load each view
        for view_idx in range(num_views):
            view_dir = scale_dir / f'view_{view_idx:02d}'
            if not view_dir.exists():
                continue

            # Load view metadata
            view_meta_path = scale_dir / f'view_{view_idx:02d}_metadata.npz'
            view_meta = np.load(view_meta_path)

            view_data = {
                'center': view_meta['center'],
                'visibility': view_meta['visibility'],
                'patches': {}
            }

            # Load patch experts for this view
            for landmark_idx in range(num_landmarks):
                patch_dir = view_dir / f'patch_{landmark_idx:02d}'

                # Check if patch exists (some landmarks not visible in some views)
                if patch_dir.exists() and (patch_dir / 'metadata.npz').exists():
                    try:
                        patch_expert = CCNFPatchExpert(str(patch_dir))
                        view_data['patches'][landmark_idx] = patch_expert
                    except Exception as e:
                        print(f"Warning: Failed to load patch {landmark_idx} in view {view_idx}: {e}")

            scale_model['views'][view_idx] = view_data

        return scale_model

    def get_patch_expert(self, scale: float, view_idx: int, landmark_idx: int) -> Optional[CCNFPatchExpert]:
        """
        Get patch expert for specific scale, view, and landmark.

        Args:
            scale: Patch scale (0.25, 0.35, or 0.5)
            view_idx: View index (0-6)
            landmark_idx: Landmark index (0-67)

        Returns:
            CCNFPatchExpert or None if not available
        """
        if scale not in self.scale_models:
            return None

        scale_model = self.scale_models[scale]
        if view_idx not in scale_model['views']:
            return None

        view_data = scale_model['views'][view_idx]
        return view_data['patches'].get(landmark_idx)

    def get_best_view(self, pose: np.ndarray) -> int:
        """
        Select best view based on head pose.

        Args:
            pose: Head pose [pitch, yaw, roll] in degrees

        Returns:
            Best view index
        """
        # Use first scale to get view centers
        if not self.scale_models:
            return 0

        first_scale = list(self.scale_models.values())[0]

        # Find view with closest center to current pose
        best_view = 0
        min_distance = float('inf')

        for view_idx, view_data in first_scale['views'].items():
            center = view_data['center'].flatten()

            # Compute distance (primarily based on yaw)
            distance = np.linalg.norm(center - pose)

            if distance < min_distance:
                min_distance = distance
                best_view = view_idx

        return best_view

    def get_info(self) -> dict:
        """Get CCNF model information."""
        info = {
            'scales': list(self.scale_models.keys()),
            'scale_models': {}
        }

        for scale, model in self.scale_models.items():
            num_patches = sum(
                len(view_data['patches'])
                for view_data in model['views'].values()
            )
            info['scale_models'][scale] = {
                'num_views': model['num_views'],
                'num_landmarks': model['num_landmarks'],
                'patch_scaling': model['patch_scaling'],
                'total_patches': num_patches
            }

        return info


def test_patch_expert():
    """Test patch expert loading and response computation."""
    print("=" * 60)
    print("Testing LNF Patch Expert Implementation")
    print("=" * 60)

    # Test 1: Load CCNF model
    print("\nTest 1: Load CCNF model")
    model_dir = "pyclnf/models"
    ccnf = CCNFModel(model_dir)

    info = ccnf.get_info()
    print(f"  Loaded scales: {info['scales']}")
    for scale, scale_info in info['scale_models'].items():
        print(f"  Scale {scale}:")
        print(f"    Views: {scale_info['num_views']}")
        print(f"    Landmarks: {scale_info['num_landmarks']}")
        print(f"    Total patches: {scale_info['total_patches']}")

    # Test 2: Get specific patch expert
    print("\nTest 2: Get specific patch expert")
    scale = 0.25
    view_idx = 0
    landmark_idx = 30  # Nose tip (typically visible in frontal view)

    patch_expert = ccnf.get_patch_expert(scale, view_idx, landmark_idx)
    if patch_expert:
        patch_info = patch_expert.get_info()
        print(f"  Patch expert for landmark {landmark_idx}, view {view_idx}, scale {scale}:")
        print(f"    Size: {patch_info['width']}×{patch_info['height']}")
        print(f"    Neurons: {patch_info['num_neurons']}")
        print(f"    Confidence: {patch_info['patch_confidence']:.3f}")

        # Test 3: Compute response on random patch
        print("\nTest 3: Compute response")
        test_patch = np.random.randint(0, 256, (patch_info['height'], patch_info['width']), dtype=np.uint8)
        response = patch_expert.compute_response(test_patch)
        print(f"    Input patch shape: {test_patch.shape}")
        print(f"    Response value: {response:.6f}")
        print(f"    Response type: {type(response)}")
    else:
        print(f"  Patch expert not found for landmark {landmark_idx}, view {view_idx}")

    # Test 4: View selection
    print("\nTest 4: View selection based on pose")
    test_poses = [
        np.array([0, 0, 0]),      # Frontal
        np.array([0, 30, 0]),     # Right profile
        np.array([0, -30, 0]),    # Left profile
    ]

    for pose in test_poses:
        best_view = ccnf.get_best_view(pose)
        print(f"  Pose {pose} -> View {best_view}")

    print("\n" + "=" * 60)
    print("✓ LNF Patch Expert Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_patch_expert()
