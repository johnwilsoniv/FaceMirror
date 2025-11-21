"""
ONNX backend for CEN patch experts (CPU/CUDA acceleration).

Provides hardware-accelerated inference on NVIDIA GPUs and optimized CPU execution.

Expected performance:
- RTX 3080/4090: 50-100 FPS per patch expert
- CPU (AVX2): 10-20 FPS
- Batch processing: 2-3x speedup

Installation:
    # For CUDA support
    pip install onnxruntime-gpu

    # For CPU only
    pip install onnxruntime

Model export required:
    python pyclnf/backends/export_to_onnx.py
"""
import numpy as np
from pathlib import Path
from typing import List, Dict
from .base_backend import CENBackend

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXCENBackend(CENBackend):
    """
    ONNX backend for CEN patch expert inference.

    Supports both CPU and CUDA acceleration.
    """

    def __init__(self):
        """Initialize ONNX backend."""
        super().__init__()
        self.backend_name = "ONNX"

        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not installed. Install with:\n"
                "  GPU: pip install onnxruntime-gpu\n"
                "  CPU: pip install onnxruntime"
            )

        # Check for CUDA
        providers = ort.get_available_providers()
        self.has_cuda = 'CUDAExecutionProvider' in providers

        if self.has_cuda:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.backend_name = "ONNX+CUDA"
        else:
            self.providers = ['CPUExecutionProvider']
            self.backend_name = "ONNX+CPU"

        # Model storage
        # sessions[scale][landmark_idx] = onnx_session
        self.sessions: Dict[float, Dict[int, ort.InferenceSession]] = {}
        self.confidences: Dict[float, Dict[int, float]] = {}

    def load_models(self, model_dir: str, scales: List[float] = None):
        """
        Load ONNX CEN models from disk.

        Args:
            model_dir: Directory containing exported .onnx files
            scales: List of scales to load

        Expected structure:
            model_dir/
                onnx_cen/
                    cen_lm00_scale0.25.onnx
                    cen_lm00_scale0.35.onnx
                    ...
                    cen_lm67_scale1.0.onnx
        """
        model_path = Path(model_dir) / "onnx_cen"
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX CEN models not found at {model_path}\n"
                f"Run: python pyclnf/backends/export_to_onnx.py"
            )

        if scales is None:
            scales = [0.25, 0.35, 0.5, 1.0]

        print(f"Loading ONNX CEN models from {model_path}...")
        print(f"Using providers: {self.providers}")

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        for scale in scales:
            self.sessions[scale] = {}
            self.confidences[scale] = {}

            for landmark_idx in range(68):
                model_file = model_path / f"cen_lm{landmark_idx:02d}_scale{scale:.2f}.onnx"

                if not model_file.exists():
                    # Skip missing landmarks
                    continue

                try:
                    # Create ONNX Runtime session
                    session = ort.InferenceSession(
                        str(model_file),
                        sess_options=sess_options,
                        providers=self.providers
                    )
                    self.sessions[scale][landmark_idx] = session

                    # Load confidence from metadata if available
                    metadata = session.get_modelmeta()
                    if metadata and hasattr(metadata, 'custom_metadata_map'):
                        confidence = metadata.custom_metadata_map.get('confidence', '1.0')
                        self.confidences[scale][landmark_idx] = float(confidence)
                    else:
                        self.confidences[scale][landmark_idx] = 1.0

                except Exception as e:
                    print(f"Warning: Failed to load landmark {landmark_idx} at scale {scale}: {e}")
                    continue

            num_loaded = len(self.sessions[scale])
            print(f"  [ONNX] Scale {scale:.2f}: {num_loaded}/68 landmarks loaded")

        self.is_loaded = True
        print(f"✓ {self.backend_name} backend loaded")

    def response(self,
                 area_of_interest: np.ndarray,
                 landmark_idx: int,
                 scale: float) -> np.ndarray:
        """
        Compute response map using ONNX model.

        Args:
            area_of_interest: Grayscale image patch (H, W) uint8 or float32
            landmark_idx: Landmark index (0-67)
            scale: Patch scale (0.25, 0.35, 0.5, 1.0)

        Returns:
            response_map: Response map (response_height, response_width) float32
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if scale not in self.sessions:
            raise ValueError(f"Scale {scale} not loaded")

        if landmark_idx not in self.sessions[scale]:
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
        session = self.sessions[scale][landmark_idx]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        outputs = session.run([output_name], {input_name: input_data})
        response_map = outputs[0]

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

        ONNX can batch efficiently when landmarks are the same,
        but for different landmarks we process sequentially.

        Args:
            areas_of_interest: List of image patches
            landmark_indices: List of landmark indices
            scales: List of scales

        Returns:
            responses: List of response maps
        """
        # Group by (landmark_idx, scale) for batching
        batches = {}
        for i, (aoi, lm_idx, scale) in enumerate(zip(areas_of_interest, landmark_indices, scales)):
            key = (lm_idx, scale)
            if key not in batches:
                batches[key] = []
            batches[key].append((i, aoi))

        # Process each batch
        results = [None] * len(areas_of_interest)
        for (lm_idx, scale), items in batches.items():
            if len(items) == 1:
                # Single item - process normally
                idx, aoi = items[0]
                results[idx] = self.response(aoi, lm_idx, scale)
            else:
                # Batch process
                indices, aois = zip(*items)
                batch_input = np.stack([
                    aoi.astype(np.float32) / 255.0 if aoi.max() > 1.0 else aoi.astype(np.float32)
                    for aoi in aois
                ])
                batch_input = batch_input[:, np.newaxis, :, :]  # Add channel dim

                # Run batched inference
                session = self.sessions[scale][lm_idx]
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name

                batch_outputs = session.run([output_name], {input_name: batch_input})
                batch_response = batch_outputs[0]

                # Distribute results
                for i, idx in enumerate(indices):
                    if batch_response.ndim == 4:
                        results[idx] = batch_response[i, 0].astype(np.float32)
                    else:
                        results[idx] = batch_response[i].astype(np.float32)

        return results

    def get_patch_confidence(self, landmark_idx: int, scale: float) -> float:
        """Get patch confidence for a landmark."""
        if scale in self.confidences and landmark_idx in self.confidences[scale]:
            return self.confidences[scale][landmark_idx]
        return 1.0

    def cleanup(self):
        """Release ONNX sessions."""
        self.sessions.clear()
        self.confidences.clear()
        self.is_loaded = False
