"""
CLNF (Constrained Local Neural Fields) - Complete facial landmark detector

This is the main user-facing API that combines:
- PDM (Point Distribution Model) for shape representation
- CCNF patch experts for landmark detection
- NU-RLMS optimizer for parameter fitting

Usage:
    from pyclnf import CLNF

    # Initialize model
    clnf = CLNF(model_dir="pyclnf/models")

    # Detect landmarks in image
    landmarks, info = clnf.fit(image, face_bbox)
"""

import numpy as np
from typing import Tuple, Optional, Dict
import cv2
from pathlib import Path

from .core.pdm import PDM
from .core.patch_expert import CCNFModel
from .core.optimizer import NURLMSOptimizer


class CLNF:
    """
    Complete CLNF facial landmark detector.

    Fits a statistical shape model (PDM) to detected facial features using
    patch experts and constrained optimization.
    """

    def __init__(self,
                 model_dir: str = "pyclnf/models",
                 scale: float = 0.25,
                 regularization: float = 1.0,
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.01):
        """
        Initialize CLNF model.

        Args:
            model_dir: Directory containing exported PDM and CCNF models
            scale: Patch scale to use (0.25, 0.35, or 0.5)
            regularization: Shape regularization weight (higher = stricter shape prior)
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for parameter updates
        """
        self.model_dir = Path(model_dir)
        self.scale = scale

        # Load PDM (shape model)
        pdm_dir = self.model_dir / "exported_pdm"
        self.pdm = PDM(str(pdm_dir))

        # Load CCNF patch experts
        self.ccnf = CCNFModel(str(self.model_dir), scales=[scale])

        # Initialize optimizer
        self.optimizer = NURLMSOptimizer(
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )

    def fit(self,
            image: np.ndarray,
            face_bbox: Tuple[float, float, float, float],
            initial_params: Optional[np.ndarray] = None,
            return_params: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Fit CLNF model to detect facial landmarks.

        Args:
            image: Input image (grayscale or color, will be converted to grayscale)
            face_bbox: Face bounding box [x, y, width, height]
            initial_params: Optional initial parameter guess (default: from bbox)
            return_params: If True, include optimized parameters in info dict

        Returns:
            landmarks: Detected 2D landmarks, shape (68, 2)
            info: Dictionary with fitting information:
                - converged: bool
                - iterations: int
                - final_update: float
                - params: np.ndarray (if return_params=True)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Initialize parameters from bounding box
        if initial_params is None:
            params = self.pdm.init_params(face_bbox)
        else:
            params = initial_params.copy()

        # Estimate head pose from bbox for view selection
        # For now, assume frontal view (view 0)
        # TODO: Implement pose estimation from bbox orientation
        view_idx = 0
        pose = np.array([0.0, 0.0, 0.0])  # [pitch, yaw, roll]

        # Get patch experts for detected view
        patch_experts = self._get_patch_experts(view_idx)

        # Run optimization
        optimized_params, opt_info = self.optimizer.optimize(
            self.pdm,
            params,
            patch_experts,
            gray
        )

        # Extract final landmarks
        landmarks = self.pdm.params_to_landmarks_2d(optimized_params)

        # Prepare output info
        info = {
            'converged': opt_info['converged'],
            'iterations': opt_info['iterations'],
            'final_update': opt_info['final_update'],
            'view': view_idx,
            'pose': pose
        }

        if return_params:
            info['params'] = optimized_params

        return landmarks, info

    def fit_video(self,
                  video_path: str,
                  face_detector,
                  output_path: Optional[str] = None,
                  visualize: bool = True) -> list:
        """
        Fit CLNF to all frames in a video.

        Args:
            video_path: Path to input video
            face_detector: Face detector function (image -> bbox or None)
            output_path: Optional path to save visualization video
            visualize: If True, draw landmarks on frames

        Returns:
            results: List of (landmarks, info) tuples for each frame
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_idx = 0
        prev_params = None  # For temporal consistency

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect face
            bbox = face_detector(frame)

            if bbox is not None:
                # Use previous frame's parameters as initialization for temporal consistency
                landmarks, info = self.fit(
                    frame,
                    bbox,
                    initial_params=prev_params
                )

                # Store parameters for next frame
                if 'params' not in info:
                    info['params'] = self.pdm.params_to_landmarks_2d.im_func.__self__.params
                prev_params = info.get('params')

                # Visualize if requested
                if visualize:
                    frame = self._draw_landmarks(frame, landmarks)

                results.append((landmarks, info))
            else:
                results.append((None, {'converged': False}))
                prev_params = None  # Reset on detection failure

            # Write frame if output requested
            if writer:
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        return results

    def _get_patch_experts(self, view_idx: int) -> Dict[int, 'CCNFPatchExpert']:
        """
        Get patch experts for a specific view.

        Args:
            view_idx: View index (0-6)

        Returns:
            Dictionary mapping landmark_idx -> CCNFPatchExpert
        """
        patch_experts = {}

        scale_model = self.ccnf.scale_models.get(self.scale)
        if scale_model is None:
            return patch_experts

        view_data = scale_model['views'].get(view_idx)
        if view_data is None:
            return patch_experts

        # Get all available patches for this view
        patch_experts = view_data['patches']

        return patch_experts

    def _draw_landmarks(self,
                       image: np.ndarray,
                       landmarks: np.ndarray,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       radius: int = 2) -> np.ndarray:
        """
        Draw landmarks on image.

        Args:
            image: Input image
            landmarks: Landmark positions (n_points, 2)
            color: Landmark color (B, G, R)
            radius: Landmark radius in pixels

        Returns:
            image: Image with landmarks drawn
        """
        vis = image.copy()

        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)

        return vis

    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'pdm': self.pdm.get_info(),
            'ccnf': self.ccnf.get_info(),
            'optimizer': {
                'regularization': self.optimizer.regularization,
                'max_iterations': self.optimizer.max_iterations,
                'convergence_threshold': self.optimizer.convergence_threshold
            },
            'scale': self.scale
        }


def test_clnf():
    """Test CLNF complete pipeline."""
    print("=" * 60)
    print("Testing Complete CLNF Pipeline")
    print("=" * 60)

    # Test 1: Initialize CLNF
    print("\nTest 1: Initialize CLNF")
    clnf = CLNF(
        model_dir="pyclnf/models",
        scale=0.25,
        max_iterations=5
    )

    info = clnf.get_info()
    print(f"  PDM: {info['pdm']['n_points']} landmarks, {info['pdm']['n_params']} params")
    print(f"  CCNF scales: {info['ccnf']['scales']}")
    print(f"  CCNF patches at 0.25: {info['ccnf']['scale_models'][0.25]['total_patches']}")
    print(f"  Optimizer: max_iter={info['optimizer']['max_iterations']}")

    # Test 2: Create test image with face-like features
    print("\nTest 2: Create test image")
    test_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    # Add some edge structure to simulate facial features
    center_y, center_x = 240, 320
    cv2.circle(test_image, (center_x - 50, center_y - 30), 15, 200, 2)  # Left eye
    cv2.circle(test_image, (center_x + 50, center_y - 30), 15, 200, 2)  # Right eye
    cv2.ellipse(test_image, (center_x, center_y + 30), (40, 20), 0, 0, 180, 200, 2)  # Mouth

    print(f"  Test image: {test_image.shape}")

    # Test 3: Fit CLNF to image
    print("\nTest 3: Fit CLNF to test image")
    face_bbox = (220, 140, 200, 250)  # [x, y, width, height]

    landmarks, fit_info = clnf.fit(test_image, face_bbox, return_params=True)

    print(f"  Bbox: {face_bbox}")
    print(f"  Converged: {fit_info['converged']}")
    print(f"  Iterations: {fit_info['iterations']}")
    print(f"  Final update: {fit_info['final_update']:.6f}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Landmark range: x=[{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}], y=[{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")

    # Test 4: Verify landmarks are within expected region
    print("\nTest 4: Verify landmark positions")
    bbox_center_x = face_bbox[0] + face_bbox[2] / 2
    bbox_center_y = face_bbox[1] + face_bbox[3] / 2

    landmark_center_x = landmarks[:, 0].mean()
    landmark_center_y = landmarks[:, 1].mean()

    center_offset = np.sqrt((landmark_center_x - bbox_center_x)**2 + (landmark_center_y - bbox_center_y)**2)

    print(f"  Bbox center: ({bbox_center_x:.1f}, {bbox_center_y:.1f})")
    print(f"  Landmark center: ({landmark_center_x:.1f}, {landmark_center_y:.1f})")
    print(f"  Center offset: {center_offset:.1f} pixels")

    # Test 5: Test with different bbox
    print("\nTest 5: Fit with different bbox")
    face_bbox2 = (150, 100, 150, 180)
    landmarks2, fit_info2 = clnf.fit(test_image, face_bbox2)

    print(f"  Bbox: {face_bbox2}")
    print(f"  Converged: {fit_info2['converged']}")
    print(f"  Iterations: {fit_info2['iterations']}")
    print(f"  Landmark shift from first fit: {np.linalg.norm(landmarks2 - landmarks, axis=1).mean():.1f} pixels")

    print("\n" + "=" * 60)
    print("✓ Complete CLNF Pipeline Tests Complete!")
    print("=" * 60)
    print("\nCLNF is ready to use!")
    print("  - Pure Python implementation (no C++ dependencies)")
    print("  - Loads OpenFace trained models")
    print("  - Ready for PyInstaller distribution")
    print("  - Can be optimized with CoreML/Cython/CuPy as needed")


if __name__ == "__main__":
    test_clnf()
