"""
CoreML backend for CEN patch experts (Apple Neural Engine acceleration).

Provides hardware-accelerated inference on Apple Silicon and Intel Macs with ANE.

Expected performance:
- M1/M2/M3: 20-50 FPS per patch expert
- Batch processing: 30-60 FPS
- Power efficiency: 5-10x better than CPU

Installation:
    pip install coremltools

Model export required:
    python pyclnf/backends/export_to_coreml.py
"""
import numpy as np
from pathlib import Path
from typing import List, Dict
from .base_backend import CENBackend

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class CoreMLCENBackend(CENBackend):
    """
    CoreML backend for CEN patch expert inference.

    Uses Apple Neural Engine for hardware acceleration on macOS.
    """

    def __init__(self):
        """Initialize CoreML backend."""
        super().__init__()
        self.backend_name = "CoreML"

        if not COREML_AVAILABLE:
            raise ImportError(
                "coremltools not installed. Install with: pip install coremltools"
            )

        # Model storage
        # models[scale][landmark_idx] = coreml_model
        self.models: Dict[float, Dict[int, 'ct.models.MLModel']] = {}
        self.confidences: Dict[float, Dict[int, float]] = {}

    def load_models(self, model_dir: str, scales: List[float] = None):
        """
        Load CoreML CEN models from disk.

        Args:
            model_dir: Directory containing exported .mlpackage files
            scales: List of scales to load

        Expected structure:
            model_dir/
                coreml_cen/
                    cen_lm00_scale0.25.mlpackage/
                    cen_lm00_scale0.35.mlpackage/
                    ...
                    cen_lm67_scale1.0.mlpackage/
        """
        model_path = Path(model_dir) / "coreml_cen"
        if not model_path.exists():
            raise FileNotFoundError(
                f"CoreML CEN models not found at {model_path}\n"
                f"Run: python pyclnf/backends/export_to_coreml.py"
            )

        if scales is None:
            scales = [0.25, 0.35, 0.5, 1.0]

        print(f"Loading CoreML CEN models from {model_path}...")

        for scale in scales:
            self.models[scale] = {}
            self.confidences[scale] = {}

            for landmark_idx in range(68):
                model_file = model_path / f"cen_lm{landmark_idx:02d}_scale{scale:.2f}.mlpackage"

                if not model_file.exists():
                    # Skip missing landmarks (some may be invisible at certain orientations)
                    continue

                try:
                    # Load CoreML model
                    model = ct.models.MLModel(str(model_file))
                    self.models[scale][landmark_idx] = model

                    # Load confidence from metadata if available
                    if hasattr(model, 'user_defined_metadata'):
                        confidence = model.user_defined_metadata.get('confidence', 1.0)
                        self.confidences[scale][landmark_idx] = float(confidence)
                    else:
                        self.confidences[scale][landmark_idx] = 1.0

                except Exception as e:
                    print(f"Warning: Failed to load landmark {landmark_idx} at scale {scale}: {e}")
                    continue

            num_loaded = len(self.models[scale])
            print(f"  [CoreML] Scale {scale:.2f}: {num_loaded}/68 landmarks loaded")

        self.is_loaded = True
        print(f"✓ CoreML CEN backend loaded")

    def response(self,
                 area_of_interest: np.ndarray,
                 landmark_idx: int,
                 scale: float) -> np.ndarray:
        """
        Compute response map using CoreML model.

        Args:
            area_of_interest: Grayscale image patch (H, W) uint8 or float32
            landmark_idx: Landmark index (0-67)
            scale: Patch scale (0.25, 0.35, 0.5, 1.0)

        Returns:
            response_map: Response map (response_height, response_width) float32
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if scale not in self.models:
            raise ValueError(f"Scale {scale} not loaded")

        if landmark_idx not in self.models[scale]:
            # Return zero response for missing landmarks
            height = area_of_interest.shape[0] - 11 + 1
            width = area_of_interest.shape[1] - 11 + 1
            return np.zeros((height, width), dtype=np.float32)

        # Prepare input
        if area_of_interest.dtype != np.float32:
            area_of_interest = area_of_interest.astype(np.float32)

        # Normalize to [0, 1] if needed
        if area_of_interest.max() > 1.0:
            area_of_interest = area_of_interest / 255.0

        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        input_data = area_of_interest[np.newaxis, np.newaxis, :, :]

        # Run inference
        model = self.models[scale][landmark_idx]
        output = model.predict({'input': input_data})

        # Extract response map
        # Output shape should be (1, 1, response_h, response_w)
        response_map = output['response']

        # Remove batch and channel dims
        if response_map.ndim == 4:
            response_map = response_map[0, 0]
        elif response_map.ndim == 3:
            response_map = response_map[0]

        return response_map.astype(np.float32)

    def batch_response(self,
                      areas_of_interest: List[np.ndarray],
                      landmark_indices: List[int],
                      scales: List[float]) -> List[np.ndarray]:
        """
        Compute response maps for multiple patches.

        Note: CoreML doesn't efficiently batch across different models,
        so this just calls response() in a loop. For true batching,
        use ONNX backend.

        Args:
            areas_of_interest: List of image patches
            landmark_indices: List of landmark indices
            scales: List of scales

        Returns:
            responses: List of response maps
        """
        responses = []
        for aoi, lm_idx, scale in zip(areas_of_interest, landmark_indices, scales):
            response = self.response(aoi, lm_idx, scale)
            responses.append(response)
        return responses

    def get_patch_confidence(self, landmark_idx: int, scale: float) -> float:
        """Get patch confidence for a landmark."""
        if scale in self.confidences and landmark_idx in self.confidences[scale]:
            return self.confidences[scale][landmark_idx]
        return 1.0

    def cleanup(self):
        """Release CoreML models."""
        self.models.clear()
        self.confidences.clear()
        self.is_loaded = False
