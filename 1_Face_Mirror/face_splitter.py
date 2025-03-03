from face_landmarks import FaceLandmarkDetector
from face_mirror import FaceMirror
from video_processor import VideoProcessor

class StableFaceSplitter:
    """Main class that encapsulates face detection, mirroring, and video processing"""
    
    def __init__(self, debug_mode=False):
        """Initialize with optional debug mode"""
        # Create component objects
        self.landmark_detector = FaceLandmarkDetector(debug_mode=debug_mode)
        self.face_mirror = FaceMirror(self.landmark_detector)
        self.video_processor = VideoProcessor(
            self.landmark_detector, 
            self.face_mirror,
            debug_mode=debug_mode
        )
        
        self.debug_mode = debug_mode
    
    def process_video(self, input_path, output_dir):
        """Process a video file and create mirrored outputs"""
        return self.video_processor.process_video(input_path, output_dir)
    
    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        self.landmark_detector.reset_tracking_history()
