#!/usr/bin/env python3
"""
PyFaceAU integration for AU extraction from mirrored videos.

This module provides a simplified wrapper around pyfaceau.processor.OpenFaceProcessor
for seamless integration with S1 Face Mirror.
"""

# Re-export PyFaceAU's OpenFaceProcessor as our main processor
from pyfaceau.processor import OpenFaceProcessor as PyFaceAUProcessor
from pathlib import Path


class OpenFace3Processor(PyFaceAUProcessor):
    """
    Drop-in replacement for OpenFace 3.0 using PyFaceAU.

    This maintains API compatibility with the old OpenFace3Processor
    while using the more accurate PyFaceAU backend.

    Features:
    - 17 Action Units (AU01-AU45) with r > 0.92 correlation to OpenFace 2.2
    - CLNF landmark refinement for improved accuracy
    - 100% Python implementation (no C++ dependencies)
    - CoreML Neural Engine acceleration on Apple Silicon
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
        super().__init__(
            weights_dir=str(weights_dir),
            use_clnf_refinement=True,  # Always enable CLNF for best accuracy
            verbose=debug_mode  # Only show detailed logs in debug mode
        )

        # Store compatibility flags
        self.debug_mode = debug_mode
        self.calculate_landmarks = True  # PyFaceAU always calculates landmarks

        if debug_mode:
            print("\n" + "="*60)
            print("PYFACEAU PROCESSOR INITIALIZED")
            print("="*60)
            print("  Backend: Pure Python AU extraction")
            print("  Accuracy: r > 0.92 correlation with OpenFace 2.2")
            print("  Landmark System: 68-point PFLD + CLNF refinement")
            print("  Face Detection: RetinaFace (CoreML accelerated)")
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
        import cv2
        from pathlib import Path

        # Get total frames for progress reporting
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Store callback in pipeline for worker thread access
        if progress_callback and hasattr(self, 'pipeline'):
            self.pipeline._progress_callback = progress_callback
            self.pipeline._total_frames = total_frames
            self.pipeline._video_fps = fps

        # Call parent's process_video
        result = super().process_video(video_path, output_csv)

        # Clean up callback reference
        if hasattr(self, 'pipeline'):
            self.pipeline._progress_callback = None

        return result


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
