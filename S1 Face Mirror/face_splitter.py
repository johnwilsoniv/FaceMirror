from pyfaceau_detector import PyFaceAU68LandmarkDetector
from face_mirror import FaceMirror
from video_processor import VideoProcessor
import numpy as np
import gc

class StableFaceSplitter:
    """Main class that encapsulates face detection, mirroring, and video processing"""

    def __init__(self, debug_mode=False, device='cpu', num_threads=6, progress_callback=None, skip_face_detection=False):
        """
        Initialize with PyFaceAU 68-point detector

        Args:
            debug_mode: Enable debug output
            device: 'cpu' or 'cuda' (ignored - PyFaceAU auto-detects)
            num_threads: Number of threads for parallel frame processing (default: 6)
            progress_callback: Optional callback function for progress updates (stage, current, total, message)
            skip_face_detection: Skip RetinaFace entirely, use default bbox (experimental)
        """
        # Create PyFaceAU 68-point landmark detector with CLNF refinement
        # skip_redetection=True: Only run RetinaFace once (first frame), then track
        # skip_face_detection=True: Skip RetinaFace entirely (experimental, uses default bbox)
        self.landmark_detector = PyFaceAU68LandmarkDetector(
            debug_mode=debug_mode,
            device=device,
            skip_redetection=not skip_face_detection,  # If skipping detection, also skip redetection
            skip_face_detection=skip_face_detection,
            use_clnf_refinement=True  # Enable CLNF for better midline accuracy
        )
        if debug_mode:
            print("Using PyFaceAU 68-point detector with CLNF refinement")

        # Create component objects
        self.face_mirror = FaceMirror(self.landmark_detector)
        self.video_processor = VideoProcessor(
            self.landmark_detector,
            self.face_mirror,
            debug_mode=debug_mode,
            num_threads=num_threads,
            progress_callback=progress_callback
        )

        self.debug_mode = debug_mode
        self.progress_callback = progress_callback

        # Warm up models and freeze for GC optimization
        self._warmup_models()

    def _warmup_models(self):
        """
        Warm up models with dummy inference to trigger PyTorch graph optimization.
        Also freeze models from garbage collection to reduce GC overhead.
        """
        if self.debug_mode:
            print("Warming up face detection models...")

        # Create dummy frame (480p resolution)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm up face detection
        try:
            _ = self.landmark_detector.get_face_mesh(dummy_frame)
            if self.debug_mode:
                print("  Face detector warmed up")
        except Exception:
            pass  # Ignore warm-up errors

        # Run garbage collection once, then freeze model objects
        gc.collect()
        gc.freeze()

        if self.debug_mode:
            print("  Models frozen from garbage collection (reduces GC overhead)")

    def process_video(self, input_path, output_dir):
        """Process a video file and create mirrored outputs"""
        return self.video_processor.process_video(input_path, output_dir)
    
    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        self.landmark_detector.reset_tracking_history()
