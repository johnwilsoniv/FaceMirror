"""
Frame extraction module for facial AU analysis.
Extracts frames from videos at points of maximal expression.
"""

import os
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialFrameExtractor:
    """
    Extracts frames from videos at points of maximal expression.
    """
    
    def __init__(self):
        """Initialize the frame extractor."""
        pass
    
    def extract_frames(self, analyzer, video_path, output_dir, patient_id, results, action_descriptions):
        """
        Extract frames from video at points of maximal expression
        and save them as images with appropriate labels.
        
        Args:
            analyzer: The analyzer instance
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frames
            patient_id (str): Patient ID
            results (dict): Results dictionary
            action_descriptions (dict): Dictionary of action descriptions
            
        Returns:
            tuple: (bool, dict) Success flag and dictionary of frame paths
        """
        if not results:
            logger.error("No analysis results available for frame extraction")
            return False, {}
        
        # Create output directory if it doesn't exist
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return False, {}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video FPS: {fps}")
        logger.info(f"Total frames: {frame_count}")
        
        # Initialize frame paths dictionary
        frame_paths = {}
        
        # Extract frames for each action
        for action, info in results.items():
            frame_num = int(info['max_frame'])
            
            # Set video to the right frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Could not read frame {frame_num}")
                continue
            
            # Create image label with patient ID, action, and frame number
            action_desc = action_descriptions.get(action, action)
            label = f"{patient_id}_{action_desc}_frame{frame_num}"
            
            # Add label text to the image
            labeled_frame = frame.copy()
            cv2.putText(
                labeled_frame, 
                label, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Save the image
            output_path = os.path.join(patient_output_dir, f"{label}.jpg")
            cv2.imwrite(output_path, labeled_frame)
            logger.info(f"Saved {output_path}")
            
            # Save original frame (without text) for visualization
            # Save as PNG for better quality and to avoid compression artifacts
            original_output_path = os.path.join(patient_output_dir, f"{label}_original.png")
            cv2.imwrite(original_output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Store the path for use in visualization
            frame_paths[action] = original_output_path
        
        # Release the video capture
        cap.release()
        
        logger.info(f"Extracted {len(frame_paths)} frames from video for patient {patient_id}")
        return True, frame_paths
