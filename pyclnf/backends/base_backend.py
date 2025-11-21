"""
Base backend interface for CEN patch experts.

Defines the interface that all backends (CoreML, ONNX, Pure Python) must implement.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List


class CENBackend(ABC):
    """
    Abstract base class for CEN patch expert backends.

    All backends must implement the response() method to compute
    response maps for a given image patch.
    """

    def __init__(self):
        """Initialize the backend."""
        self.backend_name = "unknown"
        self.is_loaded = False

    @abstractmethod
    def load_models(self, model_dir: str, scales: List[float] = None):
        """
        Load CEN patch expert models for all landmarks and scales.

        Args:
            model_dir: Directory containing CEN model files
            scales: List of scales to load (e.g., [0.25, 0.35, 0.5, 1.0])
        """
        pass

    @abstractmethod
    def response(self,
                 area_of_interest: np.ndarray,
                 landmark_idx: int,
                 scale: float) -> np.ndarray:
        """
        Compute patch expert response map for a given image patch.

        Args:
            area_of_interest: Grayscale image patch (H, W) as uint8 or float32
            landmark_idx: Landmark index (0-67)
            scale: Patch scale to use (0.25, 0.35, 0.5, or 1.0)

        Returns:
            response: Response map (response_height, response_width) as float32
        """
        pass

    @abstractmethod
    def batch_response(self,
                      areas_of_interest: List[np.ndarray],
                      landmark_indices: List[int],
                      scales: List[float]) -> List[np.ndarray]:
        """
        Compute response maps for multiple patches in batch (if supported).

        Args:
            areas_of_interest: List of image patches
            landmark_indices: List of landmark indices
            scales: List of scales for each patch

        Returns:
            responses: List of response maps
        """
        pass

    def get_patch_confidence(self, landmark_idx: int, scale: float) -> float:
        """
        Get patch expert confidence for a landmark at a specific scale.

        Args:
            landmark_idx: Landmark index (0-67)
            scale: Patch scale

        Returns:
            confidence: Patch confidence value
        """
        # Default implementation - subclasses can override
        return 1.0

    def cleanup(self):
        """Release backend resources."""
        self.is_loaded = False

    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.backend_name} ({status})"
