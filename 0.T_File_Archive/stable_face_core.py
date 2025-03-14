import cv2
import numpy as np
from pathlib import Path
import logging
import dlib
from scipy.spatial import Delaunay
from video_rotation import process_video_rotation
import math
from sklearn.linear_model import RANSACRegressor

# Import components from other files
from face_detection import preprocess_frame, update_face_roi, get_face_mesh, validate_landmarks, estimate_head_pose
from midline_calculation import get_facial_midline
from face_mirroring import create_mirrored_faces, preprocess_tilted_frame
from visualization import create_debug_frame
from stability_analysis import analyze_midline_stability


class StableFaceSplitter:
    def __init__(self, debug_mode=False, log_midline=True):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)

        # Face tracking parameters
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Initialize smoothing history
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.history_size = 8  # Increased from 5 for better smoothing
        
        # Head pose tracking
        self.pose_history = []
        self.pose_history_size = 3
        
        # Previous frame data for stability metrics
        self.prev_midline_glabella = None
        self.prev_midline_chin = None
        self.prev_head_pitch = None
        self.prev_head_yaw = None
        self.prev_head_roll = None
        
        # Add tracking for midline angle to maintain orientation consistency
        self.prev_midline_angle = None
        
        # Dynamic history size based on motion
        self.min_history_size = 5  # Increased from 3
        self.max_history_size = 10 # Increased from 8
        self.motion_threshold = 5.0  # pixels
        
        # Midline logging
        self.log_midline = log_midline
        if log_midline:
            log_path = Path('midline_log.csv')
            # Create or overwrite the log file with headers
            with open(log_path, 'w') as f:
                headers = [
                    "frame", 
                    # Midline coordinates
                    "glabella_x", "glabella_y", "chin_x", "chin_y",
                    # Head orientation 
                    "pitch", "yaw", "roll", "eye_angle",
                    # Raw landmarks used for midline
                    "left_brow_x", "left_brow_y", "right_brow_x", "right_brow_y",
                    "nose_bridge_x", "nose_bridge_y", "nose_tip_x", "nose_tip_y",
                    "philtrum_x", "philtrum_y", "chin_point_x", "chin_point_y",
                    # Stability metrics
                    "midline_angle", "midline_slope",
                    "glabella_delta_x", "glabella_delta_y", "chin_delta_x", "chin_delta_y",
                    "pitch_delta", "yaw_delta", "roll_delta",
                    # Method information
                    "tilt_factor", "method", "quality", "smoothing_strength"
                ]
                f.write(",".join(headers) + "\n")
        
        self.debug_mode = debug_mode
        if debug_mode or log_midline:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        # Reset face tracking parameters
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Reset smoothing history
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.pose_history = []
        
        # Reset stability metrics
        self.prev_midline_glabella = None
        self.prev_midline_chin = None
        self.prev_head_pitch = None
        self.prev_head_yaw = None
        self.prev_head_roll = None
        self.prev_midline_angle = None  # Reset angle tracking

        # Reset face ROI if it exists
        if hasattr(self, 'face_roi'):
            self.face_roi = None

    def process_video(self, input_path, output_dir, analyze_stability=True):
        """Process video file with progress tracking"""
        # Reset tracking history at the start of each video
        self.reset_tracking_history()

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing video: {input_path.name}")

        # Process video rotation
        print("Checking video rotation...")
        rotated_input_path = output_dir / f"{input_path.stem}_rotated{input_path.suffix}"
        rotated_input_path = process_video_rotation(str(input_path), str(rotated_input_path))

        # Update output filenames to reflect anatomical sides
        anatomical_right_output = output_dir / f"{input_path.stem}_right_mirrored.mp4"
        anatomical_left_output = output_dir / f"{input_path.stem}_left_mirrored.mp4"
        debug_output = output_dir / f"{input_path.stem}_debug.mp4"
        
        # Create midline log with video name prefix if logging is enabled
        midline_log_path = output_dir / f"{input_path.stem}_midline_log.csv"
        if hasattr(self, 'log_midline') and self.log_midline:
            # Reset log file for this video with enhanced headers
            with open(midline_log_path, 'w') as f:
                headers = [
                    "frame", 
                    # Midline coordinates
                    "glabella_x", "glabella_y", "chin_x", "chin_y",
                    # Head orientation 
                    "pitch", "yaw", "roll", "eye_angle",
                    # Raw landmarks used for midline
                    "left_brow_x", "left_brow_y", "right_brow_x", "right_brow_y",
                    "nose_bridge_x", "nose_bridge_y", "nose_tip_x", "nose_tip_y",
                    "philtrum_x", "philtrum_y", "chin_point_x", "chin_point_y",
                    # Stability metrics
                    "midline_angle", "midline_slope",
                    "glabella_delta_x", "glabella_delta_y", "chin_delta_x", "chin_delta_y",
                    "pitch_delta", "yaw_delta", "roll_delta",
                    # Method information
                    "tilt_factor", "method", "quality", "smoothing_strength", "status"
                ]
                f.write(",".join(headers) + "\n")

        # Open video and get properties
        cap = cv2.VideoCapture(str(rotated_input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {rotated_input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps

        print(f"\nVideo details:")
        print(f"- Resolution: {width}x{height}")
        print(f"- Frames: {total_frames}")
        print(f"- Duration: {duration:.1f} seconds")
        print(f"- FPS: {fps}")

        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        right_writer = cv2.VideoWriter(str(anatomical_right_output), fourcc, fps, (width, height))
        left_writer = cv2.VideoWriter(str(anatomical_left_output), fourcc, fps, (width, height))
        debug_writer = cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))

        # Process frames
        frame_count = 0
        last_progress = -1
        print("\nProcessing frames...")
        
        # Keep track of failed frames for quality metrics
        failed_frames = 0
        invalid_frames = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Update progress every 1%
                progress = int((frame_count / total_frames) * 100)
                if progress != last_progress:
                    print(f"\rProgress: {progress}% ({frame_count}/{total_frames} frames)", end="")
                    last_progress = progress
                
                # Update frame count before processing (for logging)
                self.frame_count = frame_count

                # Process frame
                landmarks, _ = self.get_face_mesh(frame)
                
                midline_status = "no_face"  # Default status for logging
                log_data = {}  # Initialize empty log data
                
                if landmarks is not None:
                    # Validate landmarks
                    if self.validate_landmarks(landmarks):
                        # Generate mirrored faces - use a separate call to get facial midline first for logging
                        glabella, chin, log_data = self.get_facial_midline(landmarks, frame.shape)
                        right_face, left_face = self.create_mirrored_faces(frame, landmarks)
                        debug_frame = self.create_debug_frame(frame, landmarks)
                        processed_frames += 1
                        midline_status = "success"
                    else:
                        # Invalid landmarks - use original frame
                        right_face, left_face = frame.copy(), frame.copy()
                        
                        # Add message to debug frame
                        debug_frame = frame.copy()
                        cv2.putText(debug_frame, "Invalid landmarks detected", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        invalid_frames += 1
                        midline_status = "invalid_landmarks"
                else:
                    # No face detected - use original frame
                    right_face, left_face = frame.copy(), frame.copy()
                    debug_frame = frame.copy()
                    
                    # Add message to debug frame
                    cv2.putText(debug_frame, "No face detected", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    failed_frames += 1
                
                # Add frame count to debug frame
                cv2.putText(debug_frame, f"Frame: {frame_count}", (30, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # If logging is enabled, log the data
                if hasattr(self, 'log_midline') and self.log_midline:
                    if midline_status == "success" and log_data:
                        # We have valid data to log
                        with open(midline_log_path, 'a') as f:
                            log_values = [
                                str(frame_count),
                                # Midline coordinates
                                f"{glabella[0]:.2f}", f"{glabella[1]:.2f}", 
                                f"{chin[0]:.2f}", f"{chin[1]:.2f}",
                                # Head orientation
                                f"{log_data['pitch']:.4f}", f"{log_data['yaw']:.4f}", 
                                f"{log_data['roll']:.4f}", f"{log_data['eye_angle']:.2f}",
                                # Raw landmarks
                                f"{log_data['left_brow_x']:.2f}", f"{log_data['left_brow_y']:.2f}",
                                f"{log_data['right_brow_x']:.2f}", f"{log_data['right_brow_y']:.2f}",
                                f"{log_data['nose_bridge_x']:.2f}", f"{log_data['nose_bridge_y']:.2f}",
                                f"{log_data['nose_tip_x']:.2f}", f"{log_data['nose_tip_y']:.2f}",
                                f"{log_data['philtrum_x']:.2f}", f"{log_data['philtrum_y']:.2f}",
                                f"{log_data['chin_point_x']:.2f}", f"{log_data['chin_point_y']:.2f}",
                                # Stability metrics
                                f"{log_data['midline_angle']:.2f}", f"{log_data['midline_slope']:.4f}",
                                f"{log_data['glabella_delta_x']:.2f}", f"{log_data['glabella_delta_y']:.2f}",
                                f"{log_data['chin_delta_x']:.2f}", f"{log_data['chin_delta_y']:.2f}",
                                f"{log_data['pitch_delta']:.4f}", f"{log_data['yaw_delta']:.4f}", 
                                f"{log_data['roll_delta']:.4f}",
                                # Method information
                                f"{log_data['tilt_factor']:.4f}", f"{log_data['method']}",
                                f"{log_data['quality']}", f"{log_data['smoothing_strength']}",
                                midline_status
                            ]
                            f.write(",".join(log_values) + "\n")
                    else:
                        # Log the failure with empty values
                        with open(midline_log_path, 'a') as f:
                            # Create a line with the frame count, zeros for all metrics, and the status
                            empty_values = ["0" for _ in range(34)]  # 34 = total columns - frame - status
                            log_values = [str(frame_count)] + empty_values + [midline_status]
                            f.write(",".join(log_values) + "\n")
                    
                    # Log to console periodically
                    if self.debug_mode and frame_count % 30 == 0 and midline_status == "success":
                        self.logger.info(
                            f"Frame {frame_count} - Head: Pitch={log_data['pitch']:.2f}, "
                            f"Yaw={log_data['yaw']:.2f}, Roll={log_data['roll']:.2f}, "
                            f"Method: {log_data['method']}, Quality: {log_data['quality']}"
                        )

                right_writer.write(right_face.astype(np.uint8))
                left_writer.write(left_face.astype(np.uint8))
                debug_writer.write(debug_frame)

            except Exception as e:
                if self.debug_mode:
                    self.logger.warning(f"\nError processing frame {frame_count}: {str(e)}")
                right_writer.write(frame)
                left_writer.write(frame)
                
                # Add error message to debug frame
                error_frame = frame.copy()
                cv2.putText(error_frame, f"Error: {str(e)[:50]}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                debug_writer.write(error_frame)
                
                # Log the error if logging is enabled
                if hasattr(self, 'log_midline') and self.log_midline:
                    with open(midline_log_path, 'a') as f:
                        empty_values = ["0" for _ in range(34)]  # 34 = total columns - frame - status
                        log_values = [str(frame_count)] + empty_values + ["error"]
                        f.write(",".join(log_values) + "\n")
                
                failed_frames += 1

            frame_count += 1

        # Clean up
        cap.release()
        right_writer.release()
        left_writer.release()
        debug_writer.release()
        
        # Calculate quality metrics
        success_rate = processed_frames / frame_count * 100 if frame_count > 0 else 0
        invalid_rate = invalid_frames / frame_count * 100 if frame_count > 0 else 0
        failure_rate = failed_frames / frame_count * 100 if frame_count > 0 else 0
        
        print(f"\nProcessing complete: {frame_count} frames processed")
        print(f"Quality metrics:")
        print(f"- Successfully processed: {processed_frames}/{frame_count} frames ({success_rate:.1f}%)")
        print(f"- Invalid landmarks: {invalid_frames}/{frame_count} frames ({invalid_rate:.1f}%)")
        print(f"- Failed detections: {failed_frames}/{frame_count} frames ({failure_rate:.1f}%)")
        
        # Log info about midline log if enabled
        if hasattr(self, 'log_midline') and self.log_midline:
            print(f"\nComprehensive midline and head pose log saved to: {midline_log_path}")
            
            # Analyze stability data if requested
            if analyze_stability:
                print("Analyzing midline stability...")
                self.analyze_midline_stability(midline_log_path)

        # Determine the list of output files
        output_files = [str(anatomical_right_output), str(anatomical_left_output), str(debug_output)]
        if str(rotated_input_path) != str(input_path):
            output_files.append(str(rotated_input_path))
        if hasattr(self, 'log_midline') and self.log_midline:
            output_files.append(str(midline_log_path))
            # Add stability analysis image if it was generated
            stability_image = midline_log_path.with_suffix('.png')
            if stability_image.exists():
                output_files.append(str(stability_image))

        print("\nOutput files:")
        for f in output_files:
            print(f"- {Path(f).name}")
        print("")

        return output_files

# Add methods from other files
preprocess_frame = preprocess_frame
update_face_roi = update_face_roi
get_face_mesh = get_face_mesh
validate_landmarks = validate_landmarks
estimate_head_pose = estimate_head_pose
get_facial_midline = get_facial_midline
create_mirrored_faces = create_mirrored_faces
preprocess_tilted_frame = preprocess_tilted_frame
create_debug_frame = create_debug_frame
analyze_midline_stability = analyze_midline_stability

# Attach the methods to the class
StableFaceSplitter.preprocess_frame = preprocess_frame
StableFaceSplitter.update_face_roi = update_face_roi
StableFaceSplitter.get_face_mesh = get_face_mesh
StableFaceSplitter.validate_landmarks = validate_landmarks
StableFaceSplitter.estimate_head_pose = estimate_head_pose
StableFaceSplitter.get_facial_midline = get_facial_midline
StableFaceSplitter.create_mirrored_faces = create_mirrored_faces
StableFaceSplitter.preprocess_tilted_frame = preprocess_tilted_frame
StableFaceSplitter.create_debug_frame = create_debug_frame
StableFaceSplitter.analyze_midline_stability = analyze_midline_stability
