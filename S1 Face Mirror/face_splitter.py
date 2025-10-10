from openface3_detector import OpenFace3LandmarkDetector
from face_mirror import FaceMirror
from video_processor import VideoProcessor

class StableFaceSplitter:
    """Main class that encapsulates face detection, mirroring, and video processing"""

    def __init__(self, debug_mode=False, device='cpu', num_threads=6, progress_callback=None):
        """
        Initialize with OpenFace 3.0 detector

        Args:
            debug_mode: Enable debug output
            device: 'cpu' or 'cuda' for GPU acceleration
            num_threads: Number of threads for parallel frame processing (default: 6)
            progress_callback: Optional callback function for progress updates (stage, current, total, message)
        """
        # Create OpenFace 3.0 landmark detector
        self.landmark_detector = OpenFace3LandmarkDetector(
            debug_mode=debug_mode,
            device=device
        )
        if debug_mode:
            print("Using OpenFace 3.0 detector")

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

    def process_video(self, input_path, output_dir):
        """Process a video file and create mirrored outputs"""
        return self.video_processor.process_video(input_path, output_dir)
    
    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        self.landmark_detector.reset_tracking_history()
