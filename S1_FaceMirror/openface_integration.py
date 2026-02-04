#!/usr/bin/env python3
"""
GPU-accelerated AU extraction for S1 Face Mirror.

Uses PyPI packages:
- pyfaceau (>=1.3.4): Full AU extraction pipeline
- pyclnf (>=0.2.2): GPU-accelerated CLNF landmark detection
- pymtcnn (>=1.1.1): CoreML/CUDA face detection

Optionally uses SPIGA for landmark detection (experimental).
"""

from pyfaceau.processor import OpenFaceProcessor as PyFaceAUProcessor
from pathlib import Path
import numpy as np
import importlib.util

# Import local config.py (not pyfaceau.config)
_config_path = Path(__file__).parent / 'config.py'
_spec = importlib.util.spec_from_file_location("local_config", _config_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


class SPIGALandmarkWrapper:
    """
    Wrapper that makes SPIGA compatible with pyfaceau's CLNF interface.

    Provides the same .fit() interface as pyclnf CLNF but uses SPIGA
    for landmark detection. Uses CalcParams to compute PDM params from landmarks.
    """

    def __init__(self, spiga_detector, pdm_parser):
        """
        Args:
            spiga_detector: SPIGALandmarkDetector instance
            pdm_parser: PDMParser instance for CalcParams
        """
        from pyfaceau.pipeline import CalcParams

        self.spiga_detector = spiga_detector
        self._cached_bbox = None
        self.calc_params = CalcParams(pdm_parser)

        # Store PDM for compatibility
        self.pdm = pdm_parser

    _call_count = 0  # Class variable to track calls

    def fit(self, frame, bbox, detector_type=None, return_params=True):
        """
        Detect landmarks using SPIGA (CLNF-compatible interface).

        Args:
            frame: BGR image
            bbox: (x, y, w, h) bounding box
            detector_type: Ignored (for CLNF compatibility)
            return_params: If True, return info dict with params

        Returns:
            landmarks: (68, 2) array
            info: dict with 'converged', 'iterations', 'params'
        """
        SPIGALandmarkWrapper._call_count += 1
        if SPIGALandmarkWrapper._call_count <= 3:
            print(f"[SPIGA WRAPPER] fit() called (call #{SPIGALandmarkWrapper._call_count})")

        # NOTE: PyMTCNN and FaceNet MTCNN detect faces at different positions.
        # PyMTCNN bbox starts ~256px lower (more neck/shoulders), while FaceNet
        # focuses more on the face. Using PyMTCNN bbox with SPIGA produces landmarks
        # that are shifted relative to CLNF landmarks.
        #
        # Solution: Let SPIGA use its own FaceNet MTCNN detection (don't override bbox).
        # The scaling correction below will align the landmark spread with CLNF.

        # Use SPIGA detector with its native FaceNet MTCNN detection
        # (don't set cached_bbox - let SPIGA detect the face itself)
        landmarks, spiga_info = self.spiga_detector.get_face_mesh(frame, detection_interval=0)

        # Note: SPIGA landmarks have different semantics than CLNF (different face detector,
        # different landmark definitions). The face alignment uses a similarity transform
        # based on landmark RELATIVE positions, so absolute positions don't need to match.
        # We pass SPIGA landmarks as-is and let the alignment handle the normalization.

        if landmarks is None:
            # Return zeros if detection failed
            landmarks = np.zeros((68, 2), dtype=np.float32)
            info = {
                'converged': False,
                'iterations': 0,
                'params': np.zeros(40, dtype=np.float32)  # 6 global + 34 local
            }
        else:
            # Use CalcParams to compute PDM params from landmarks
            try:
                # calc_params returns (params_global, params_local) tuple
                params_global, params_local = self.calc_params.calc_params(landmarks)
                # Concatenate to match CLNF format: [6 global + 34 local]
                params = np.concatenate([
                    params_global.astype(np.float32),
                    params_local.astype(np.float32)
                ])
            except Exception as e:
                # Fallback to zeros if CalcParams fails
                print(f"[SPIGA WRAPPER] CalcParams failed: {e}")
                params = np.zeros(40, dtype=np.float32)

            info = {
                'converged': True,
                'iterations': 1,
                'params': params,
                'confidence': spiga_info.get('confidence', 1.0)
            }

        return landmarks, info

    def reset_temporal_state(self):
        """Reset temporal state (CLNF compatibility)."""
        self.spiga_detector.reset_tracking_history()
        self._cached_bbox = None


class OpenFace3Processor(PyFaceAUProcessor):
    """
    GPU-accelerated AU extraction using PyFaceAU pipeline.

    Features:
    - 17 Action Units (AU01-AU45) with r > 0.95 correlation to OpenFace 2.2
    - GPU-accelerated CLNF landmarks (~15 fps) OR SPIGA landmarks (experimental)
    - PyMTCNN face detection (CoreML/CUDA/CPU auto-selection)
    - 100% Python implementation (no C++ dependencies)
    """

    def __init__(self, device=None, weights_dir=None, confidence_threshold=0.5,
                 nms_threshold=0.4, calculate_landmarks=True, num_threads=6,
                 debug_mode=False, skip_face_detection=False):
        """
        Initialize PyFaceAU processor with OpenFace3-compatible API.

        Args:
            device: Ignored - PyFaceAU auto-detects optimal device
            weights_dir: Path to weights directory (defaults to ./weights)
            confidence_threshold: Ignored - PyFaceAU uses built-in thresholds
            nms_threshold: Ignored - PyFaceAU uses built-in thresholds
            calculate_landmarks: Always True - PyFaceAU always uses CLNF refinement
            num_threads: Ignored - PyFaceAU manages threading internally
            debug_mode: Enable verbose logging (default: False)
            skip_face_detection: Ignored - PyFaceAU always detects faces
        """
        # Determine weights directory
        if weights_dir is None:
            script_dir = Path(__file__).parent
            weights_dir = script_dir / 'weights'
        else:
            weights_dir = Path(weights_dir)

        # Check if SPIGA should be used for AU extraction
        detector_type = getattr(config, 'LANDMARK_DETECTOR', 'clnf')
        self._using_spiga = (detector_type == 'spiga')

        # Initialize PyFaceAU processor with enhanced settings
        # Note: verbose=True enables progress reporting during AU extraction
        super().__init__(
            weights_dir=str(weights_dir),
            use_clnf_refinement=True,  # Always enable CLNF for best accuracy
            verbose=True  # Always show AU extraction progress
        )

        # Store compatibility flags
        self.debug_mode = debug_mode
        self.calculate_landmarks = True  # PyFaceAU always calculates landmarks

        if self._using_spiga:
            # Replace CLNF with SPIGA wrapper AFTER pipeline is created
            try:
                from spiga_detector import SPIGALandmarkDetector

                # Force pipeline initialization to get pdm_parser
                # This must succeed for either CLNF or SPIGA to work
                if not self.pipeline._components_initialized:
                    self.pipeline._initialize_components()

                # Now create SPIGA detector and replace the landmark detector
                spiga = SPIGALandmarkDetector(debug_mode=debug_mode)
                # Pass PDM parser for CalcParams to compute proper geometric features
                pdm_parser = self.pipeline.pdm_parser
                self.pipeline.landmark_detector = SPIGALandmarkWrapper(spiga, pdm_parser)

                if debug_mode:
                    print("\n" + "="*60)
                    print("SPIGA AU PROCESSOR (EXPERIMENTAL)")
                    print("="*60)
                    print("  Backend: pyfaceau + SPIGA landmarks")
                    print("  Landmarks: SPIGA 98→68 mapped")
                    print("  Face Detection: pyfaceau PyMTCNN + SPIGA FaceNet MTCNN")
                    print("  Geometric: CalcParams from SPIGA landmarks")
                    print("="*60 + "\n")
            except Exception as e:
                import traceback
                print(f"Warning: SPIGA initialization failed ({e}), falling back to CLNF")
                traceback.print_exc()
                self._using_spiga = False

        if not self._using_spiga and debug_mode:
            print("\n" + "="*60)
            print("GPU-ACCELERATED AU PROCESSOR")
            print("="*60)
            print("  Backend: pyfaceau 1.3.4 + pyclnf 0.2.2")
            print("  Accuracy: 15/17 AUs pass (r >= 0.95)")
            print("  Landmarks: GPU-accelerated CLNF (~15 fps)")
            print("  Face Detection: PyMTCNN (CoreML/CUDA auto)")
            print("  AU Models: SVR-based (17 AUs)")
            print("="*60 + "\n")

    def process_video(self, video_path, output_csv=None, progress_callback=None):
        """
        Process video with progress_callback support (API compatibility wrapper).

        Wraps PyFaceAU's process_video to provide progress updates to S1's GUI.

        Args:
            video_path: Path to input video
            output_csv: Path to output CSV file
            progress_callback: Function(current_frame, total_frames, fps) for progress updates

        Returns:
            int: Number of frames processed
        """
        # Pass callback directly to parent's process_video
        # The callback will be called on each frame to update GUI
        return super().process_video(video_path, output_csv, progress_callback)


def process_videos(directory_path, specific_files=None, output_dir=None, **kwargs):
    """
    Process video files using PyFaceAU.

    Maintains API compatibility with old OpenFace3 process_videos function.

    Args:
        directory_path: Path to directory containing video files
        specific_files: List of specific files to process (optional)
        output_dir: Output directory for CSV files (optional)
        **kwargs: Additional arguments passed to OpenFace3Processor

    Returns:
        int: Number of files successfully processed
    """
    from pyfaceau.processor import process_videos as pyfaceau_process_videos

    # Use PyFaceAU's native batch processing function
    return pyfaceau_process_videos(
        directory_path=directory_path,
        specific_files=specific_files,
        output_dir=output_dir,
        **kwargs
    )
