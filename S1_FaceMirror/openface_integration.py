#!/usr/bin/env python3
"""
GPU-accelerated AU extraction for S1 Face Mirror.

Uses PyPI packages:
- pyfaceau (>=1.3.4): Full AU extraction pipeline
- pyclnf (>=0.2.2): GPU-accelerated CLNF landmark detection
- pymtcnn (>=1.1.1): CoreML/CUDA face detection
"""

from pyfaceau.processor import OpenFaceProcessor as PyFaceAUProcessor
from pathlib import Path


class OpenFace3Processor(PyFaceAUProcessor):
    """
    GPU-accelerated AU extraction using PyFaceAU pipeline.

    Features:
    - 17 Action Units (AU01-AU45) with r > 0.95 correlation to OpenFace 2.2
    - GPU-accelerated CLNF landmarks (~15 fps)
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

        if debug_mode:
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
