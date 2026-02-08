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
    for landmark detection. Uses 2D Procrustes to correct landmark proportions
    for stable Kabsch alignment, and CalcParams for geometric features.
    """

    # Same rigid indices used by face_aligner's Kabsch alignment
    RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

    # SPIGA-to-CLNF alignment calibration constants
    # Measured from 50 frames across 5 patients (3 paralysis + 2 normal)
    # Scale: corrects SPIGA rigid points being more tightly clustered than CLNF
    # Offset: corrects SPIGA landmark centroid shift vs CLNF CalcParams center
    SPIGA_SCALE_CORRECTION = 1.0346
    SPIGA_OFFSET_X = 0.22
    SPIGA_OFFSET_Y = 2.37

    def __init__(self, spiga_detector, pdm_parser, reg_factor=10.0):
        """
        Args:
            spiga_detector: SPIGALandmarkDetector instance
            pdm_parser: PDMParser instance for CalcParams
            reg_factor: Regularization factor for CalcParams (higher = more constrained)
        """
        from pyfaceau.alignment.calc_params import CalcParams

        self.spiga_detector = spiga_detector
        self._cached_bbox = None
        self.calc_params = CalcParams(pdm_parser, reg_factor=reg_factor)

        # Store PDM for compatibility
        self.pdm = pdm_parser

    def fit(self, frame, bbox, detector_type=None, return_params=True):
        """
        Detect landmarks using SPIGA (CLNF-compatible interface).

        Returns Procrustes-corrected landmarks (PDM proportions at detected position)
        for alignment, and stores raw SPIGA landmarks in info['raw_landmarks'].

        Args:
            frame: BGR image
            bbox: (x, y, w, h) bounding box
            detector_type: Ignored (for CLNF compatibility)
            return_params: If True, return info dict with params

        Returns:
            landmarks: (68, 2) Procrustes-corrected landmarks (for alignment)
            info: dict with 'converged', 'iterations', 'params', 'raw_landmarks'
        """
        # Use SPIGA detector with its native FaceNet MTCNN detection
        raw_landmarks, spiga_info = self.spiga_detector.get_face_mesh(frame, detection_interval=0)

        if raw_landmarks is None:
            landmarks = np.zeros((68, 2), dtype=np.float32)
            info = {
                'converged': False,
                'iterations': 0,
                'params': np.zeros(40, dtype=np.float32),
                'raw_landmarks': np.zeros((68, 2), dtype=np.float32)
            }
        else:
            try:
                # Procrustes-correct: PDM proportions at SPIGA position (stable alignment)
                landmarks = self.calc_params.procrustes_correct(
                    raw_landmarks, self.RIGID_INDICES
                )

                # Apply scale calibration: expand Procrustes-corrected landmarks to match
                # CLNF's RMS spread. SPIGA rigid points are ~3.5% more tightly clustered
                # than CLNF's, causing the Kabsch alignment to over-zoom.
                centroid = np.mean(landmarks, axis=0)
                landmarks = (landmarks - centroid) * self.SPIGA_SCALE_CORRECTION + centroid

                # CalcParams on raw landmarks for geometric features
                params_global, params_local = self.calc_params.calc_params(raw_landmarks)

                # Use raw SPIGA landmark centroid for position, with calibrated offset
                raw_centroid = np.mean(raw_landmarks, axis=0)
                params_global[4] = raw_centroid[0] + self.SPIGA_OFFSET_X
                params_global[5] = raw_centroid[1] + self.SPIGA_OFFSET_Y

                # Concatenate to match CLNF format: [6 global + 34 local]
                params = np.concatenate([
                    params_global.astype(np.float32),
                    params_local.astype(np.float32)
                ])
            except Exception as e:
                print(f"[SPIGA WRAPPER] CalcParams failed: {e}")
                landmarks = raw_landmarks
                params = np.zeros(40, dtype=np.float32)

            info = {
                'converged': True,
                'iterations': 1,
                'params': params,
                'raw_landmarks': raw_landmarks,
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
                self.pipeline.landmark_detector = SPIGALandmarkWrapper(spiga, pdm_parser, reg_factor=10.0)

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
