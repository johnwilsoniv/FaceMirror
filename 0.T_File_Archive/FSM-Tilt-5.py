import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import dlib
from scipy.spatial import Delaunay
from video_rotation import process_video_rotation
import math
from sklearn.linear_model import RANSACRegressor


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

    def preprocess_frame(self, frame):
        """Enhance frame for better face detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised, gray

    def update_face_roi(self, face_rect):
        """Update face ROI with temporal smoothing"""
        if face_rect is None:
            return

        current_roi = [face_rect.left(), face_rect.top(),
                       face_rect.right(), face_rect.bottom()]

        if not hasattr(self, 'face_roi') or self.face_roi is None:
            self.face_roi = current_roi
        else:
            # Smooth ROI transition
            alpha = 0.7  # Smoothing factor
            self.face_roi = [int(alpha * prev + (1 - alpha) * curr)
                             for prev, curr in zip(self.face_roi, current_roi)]

    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimate 3D head pose to improve tilt handling"""
        if landmarks is None:
            return None, None, None
        
        # 2D image points - selected facial landmarks
        image_points = np.array([
            landmarks[30],    # Nose tip
            landmarks[8],     # Chin
            landmarks[36],    # Left eye left corner
            landmarks[45],    # Right eye right corner
            landmarks[48],    # Left mouth corner
            landmarks[54]     # Right mouth corner
        ], dtype=np.float32)
        
        # 3D model points (approximate)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        # Camera matrix (approximate)
        height, width = frame_shape[:2]
        focal_length = width  # Approximate based on image width
        center = (width/2, height/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1), dtype=np.float32)
        
        try:
            # Find rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Directly calculate Euler angles using more robust method
                # Based on https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
                
                # Rotation around X-axis (pitch)
                pitch = -math.asin(rotation_matrix[2, 0])
                
                # Rotation around Y-axis (yaw)
                cos_pitch = math.cos(pitch)
                if abs(cos_pitch) > 1e-10:
                    yaw = math.atan2(rotation_matrix[2, 1] / cos_pitch, 
                                     rotation_matrix[2, 2] / cos_pitch)
                else:
                    yaw = 0.0
                
                # Rotation around Z-axis (roll)
                if abs(cos_pitch) > 1e-10:
                    roll = math.atan2(rotation_matrix[1, 0] / cos_pitch, 
                                     rotation_matrix[0, 0] / cos_pitch)
                else:
                    roll = math.atan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
                
                return pitch, yaw, roll
        except Exception as e:
            if self.debug_mode:
                self.logger.warning(f"Head pose estimation failed: {str(e)}")
        
        return None, None, None

    def get_face_mesh(self, frame, detection_interval=2):
        """Get facial landmarks with improved tilt handling"""
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply detection more frequently when head is tilted
        if self.last_landmarks is not None:
            # Calculate current head tilt from last landmarks
            left_eye = np.mean(self.last_landmarks[36:42], axis=0)
            right_eye = np.mean(self.last_landmarks[42:48], axis=0)
            tilt_angle = np.arctan2(right_eye[1] - left_eye[1], 
                                  right_eye[0] - left_eye[0])
            
            # Adjust detection interval based on tilt
            tilt_factor = abs(np.sin(tilt_angle))
            adjusted_interval = max(1, int(detection_interval * (1 - tilt_factor)))
        else:
            adjusted_interval = detection_interval
        
        # Decide whether to run detection
        run_detection = (self.frame_count % adjusted_interval == 0) or (self.last_face is None)
        
        if run_detection:
            # Try standard detection first
            faces = self.detector(gray)
            
            # If standard detection fails, try with upsampling for better detection of tilted faces
            if not faces:
                faces = self.detector(gray, 1)
            
            # If still no faces and we have previous landmarks, try ROI-based detection
            if not faces and self.last_landmarks is not None:
                # Estimate search area based on previous landmarks
                min_x = np.min(self.last_landmarks[:, 0]) - 40
                min_y = np.min(self.last_landmarks[:, 1]) - 40
                max_x = np.max(self.last_landmarks[:, 0]) + 40
                max_y = np.max(self.last_landmarks[:, 1]) + 40
                
                # Create a region of interest and try detection there
                min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
                max_x = min(gray.shape[1], int(max_x))
                max_y = min(gray.shape[0], int(max_y))
                
                if min_x < max_x and min_y < max_y:  # Ensure valid ROI
                    roi = gray[min_y:max_y, min_x:max_x]
                    if roi.size > 0:
                        roi_faces = self.detector(roi)
                        if roi_faces:
                            # Adjust coordinates back to full frame
                            adjusted_face = roi_faces[0]
                            adjusted_face = dlib.rectangle(
                                int(adjusted_face.left() + min_x),
                                int(adjusted_face.top() + min_y),
                                int(adjusted_face.right() + min_x),
                                int(adjusted_face.bottom() + min_y)
                            )
                            faces = [adjusted_face]
            
            if not faces:
                self.last_face = None
                self.last_landmarks = None
                return None, None
                
            self.last_face = max(faces, key=lambda rect: rect.width() * rect.height())
            self.update_face_roi(self.last_face)
        
        if self.last_face is not None:
            # Extract landmarks
            landmarks = self.predictor(gray, self.last_face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)
            
            # Estimate head pose
            pitch, yaw, roll = self.estimate_head_pose(points, frame.shape)
            
            # Store pose information if available
            if pitch is not None and yaw is not None and roll is not None:
                self.pose_history.append((pitch, yaw, roll))
                if len(self.pose_history) > self.pose_history_size:
                    self.pose_history.pop(0)
            
            # Dynamically adjust history size based on movement
            if len(self.landmarks_history) > 0:
                movement = np.mean(np.linalg.norm(points - self.landmarks_history[-1], axis=1))
                if movement > self.motion_threshold:
                    # High movement - use shorter history
                    self.history_size = self.min_history_size
                else:
                    # Low movement - gradually increase history size
                    self.history_size = min(self.history_size + 1, self.max_history_size)
            
            # Apply temporal smoothing
            self.landmarks_history.append(points)
            if len(self.landmarks_history) > self.history_size:
                self.landmarks_history.pop(0)
            
            # Calculate weighted average with explicit type handling
            weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Initialize smoothed points with the correct type
            smoothed_points = np.zeros_like(points, dtype=np.float32)
            
            # Apply weighted average
            for pts, w in zip(self.landmarks_history, weights):
                smoothed_points += pts * w
            
            # Apply tilt correction if we have pose information
            if self.pose_history:
                avg_roll = np.mean([p[2] for p in self.pose_history])  # Average roll angle
                if abs(avg_roll) > 0.1:  # Only correct significant tilt
                    # Calculate face center
                    face_center = np.mean(smoothed_points, axis=0)
                    
                    # Create rotation matrix
                    correction_angle = -avg_roll * 0.7  # Partial correction (70%)
                    cos_angle = np.cos(correction_angle)
                    sin_angle = np.sin(correction_angle)
                    rotation_mat = np.array([
                        [cos_angle, -sin_angle],
                        [sin_angle, cos_angle]
                    ])
                    
                    # Apply rotation around face center
                    centered_points = smoothed_points - face_center
                    rotated_points = np.dot(centered_points, rotation_mat.T)
                    smoothed_points = rotated_points + face_center
            
            # Ensure integer coordinates for final output
            smoothed_points = np.round(smoothed_points).astype(np.int32)
            
            # Create triangulation
            triangles = Delaunay(smoothed_points)
            self.last_landmarks = smoothed_points
            
            return smoothed_points, triangles
        
        return None, None

    def validate_landmarks(self, landmarks):
        """Validate landmark detection quality"""
        if landmarks is None:
            return False

        # Check for outliers
        distances = np.linalg.norm(landmarks - landmarks.mean(axis=0), axis=1)
        if len(distances) > 0:
            z_scores = (distances - distances.mean()) / max(distances.std(), 1e-6)
            if np.any(np.abs(z_scores) > 3):
                return False

        # Check for minimum face size
        face_width = landmarks[:, 0].max() - landmarks[:, 0].min()
        face_height = landmarks[:, 1].max() - landmarks[:, 1].min()
        if face_width < 60 or face_height < 60:
            return False
            
        # Check for facial tilt quality
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        tilt_angle = np.arctan2(right_eye[1] - left_eye[1], 
                              right_eye[0] - left_eye[0])
                              
        # Extreme tilt might lead to poor mirroring
        if abs(tilt_angle) > 0.5:  # About 30 degrees
            return False

        return True

    def get_facial_midline(self, landmarks, frame_shape=None):
        """Calculate the anatomical midline points with improved stability and orientation consistency"""
        if landmarks is None:
            return None, None, {}  # Return empty dict for logging data

        # Convert to float32 for calculations
        landmarks = landmarks.astype(np.float32)
        
        # Get midline-related landmarks
        nose_bridge = [landmarks[27], landmarks[28], landmarks[29], landmarks[30]]  # More nose points
        philtrum = landmarks[33]
        nose_tip = landmarks[30]
        chin = landmarks[8]
        
        # Get key points for logging
        left_medial_brow = landmarks[21]
        right_medial_brow = landmarks[22]
        
        # Detect head tilt angle based on eye positions
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], 
                              right_eye_center[0] - left_eye_center[0])
        
        # Get 3D head pose
        pitch, yaw, roll = None, None, None
        if hasattr(self, 'pose_history') and self.pose_history:
            # Use smoothed pose if available
            pitch = np.mean([p[0] for p in self.pose_history if p[0] is not None])
            yaw = np.mean([p[1] for p in self.pose_history if p[1] is not None])
            roll = np.mean([p[2] for p in self.pose_history if p[2] is not None])
        
        # Log eye angle (tilt) in degrees
        eye_angle_degrees = np.degrees(eye_angle)
        
        # Weight the midline points according to stability during tilt
        # (nose bridge and philtrum points are more stable during tilt)
        midline_points = np.vstack([
            np.mean(landmarks[[21, 22]], axis=0),  # Glabella (original)
            *nose_bridge,  # Nose bridge points (more stable)
            philtrum,      # Philtrum (stable)
            chin           # Chin (original, less stable with tilt)
        ])
        
        # Calculate weights - give more importance to stable features
        weights = np.array([0.6, 0.8, 0.9, 0.9, 0.8, 0.8, 0.5])
        weights = weights / np.sum(weights)  # Normalize
        
        # Tilt factor (used for logging)
        tilt_factor = abs(np.sin(eye_angle))
        
        # Get coordinates for fitting
        x = midline_points[:, 0]
        y = midline_points[:, 1]
        
        # Always use RANSAC for consistency instead of switching methods
        # This is the key change - always use RANSAC as the most reliable approach
        try:
            # Use RANSAC to get a more robust line fit with improved parameters
            X = x.reshape(-1, 1)
            # Use a lower residual threshold (3.0) for better outlier rejection and more trials
            ransac = RANSACRegressor(min_samples=3, max_trials=200, residual_threshold=3.0)
            ransac.fit(X, y, sample_weight=weights)
            
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            # Calculate the midline angle
            current_angle = np.arctan(slope)
            
            # Add orientation consistency - ensure the line always points from top to bottom
            # This prevents the 180-degree flips seen in the data
            if hasattr(self, 'prev_midline_angle') and self.prev_midline_angle is not None:
                # Check if we need to flip the orientation to maintain consistency
                angle_diff = abs((current_angle - self.prev_midline_angle + np.pi) % (2 * np.pi) - np.pi)
                
                # If the angle change is too large (more than 45 degrees), flip the line
                if angle_diff > np.pi/4:
                    # Flip the line by rotating 180 degrees
                    slope = -slope
                    # Recalculate intercept to maintain the line through the facial center
                    centroid_x = np.sum(x * weights)
                    centroid_y = np.sum(y * weights)
                    intercept = centroid_y - slope * centroid_x
                    current_angle = np.arctan(slope)
            
            # Store current angle for next frame
            self.prev_midline_angle = current_angle
            
            # Use centroid of facial features for better consistency
            centroid_x = np.sum(x * weights)
            centroid_y = np.sum(y * weights)
            
            # Calculate midline points based on the face's vertical extent
            # This ensures endpoints are consistently positioned relative to the face
            face_top = np.min(landmarks[:, 1])
            face_bottom = np.max(landmarks[:, 1])
            face_height = face_bottom - face_top
            
            # Calculate line y-values at 20% above and 20% below the face
            y_top = face_top - 0.2 * face_height
            y_bottom = face_bottom + 0.2 * face_height
            
            # Calculate corresponding x-values using the line equation
            # y = mx + b => x = (y - b) / m
            if abs(slope) > 1e-6:  # Avoid division by zero
                x_top = (y_top - intercept) / slope
                x_bottom = (y_bottom - intercept) / slope
            else:
                # For near-vertical lines, use the centroid x-coordinate
                x_top = centroid_x
                x_bottom = centroid_x
            
            # Create line endpoints
            glabella = np.array([x_top, y_top])
            chin = np.array([x_bottom, y_bottom])
            
            # Track the method used for logging
            method = "RANSAC"
            quality = "good"
            
        except Exception as e:
            # Fallback to centroid method if RANSAC fails
            if self.debug_mode:
                self.logger.warning(f"RANSAC fitting failed: {str(e)}")
            
            # Calculate weighted centroid
            centroid_x = np.sum(x * weights)
            centroid_y = np.sum(y * weights)
            
            # Use standard weighted least squares
            numer = np.sum(weights * (x - centroid_x) * (y - centroid_y))
            denom = np.sum(weights * (x - centroid_x) ** 2)
            
            if abs(denom) > 1e-10:  # Avoid division by zero
                slope = numer / denom
                intercept = centroid_y - slope * centroid_x
                
                # Calculate the midline angle
                current_angle = np.arctan(slope)
                
                # Add orientation consistency - ensure the line always points from top to bottom
                if hasattr(self, 'prev_midline_angle') and self.prev_midline_angle is not None:
                    # Check if we need to flip the orientation to maintain consistency
                    angle_diff = abs((current_angle - self.prev_midline_angle + np.pi) % (2 * np.pi) - np.pi)
                    
                    # If the angle change is too large (more than 45 degrees), flip the line
                    if angle_diff > np.pi/4:
                        # Flip the line by rotating 180 degrees
                        slope = -slope
                        # Recalculate intercept to maintain the line through the facial center
                        intercept = centroid_y - slope * centroid_x
                        current_angle = np.arctan(slope)
                
                # Store current angle for next frame
                self.prev_midline_angle = current_angle
                
                # Calculate midline points based on the face's vertical extent
                # This ensures endpoints are consistently positioned relative to the face
                face_top = np.min(landmarks[:, 1])
                face_bottom = np.max(landmarks[:, 1])
                face_height = face_bottom - face_top
                
                # Calculate line y-values at 20% above and 20% below the face
                y_top = face_top - 0.2 * face_height
                y_bottom = face_bottom + 0.2 * face_height
                
                # Calculate corresponding x-values using the line equation
                if abs(slope) > 1e-6:  # Avoid division by zero
                    x_top = (y_top - intercept) / slope
                    x_bottom = (y_bottom - intercept) / slope
                else:
                    # For near-vertical lines, use the centroid x-coordinate
                    x_top = centroid_x
                    x_bottom = centroid_x
                
                # Create line endpoints
                glabella = np.array([x_top, y_top])
                chin = np.array([x_bottom, y_bottom])
            else:
                # Fallback to original method if fitting fails
                glabella = np.mean(landmarks[[21, 22]], axis=0)
                chin = landmarks[8]
                
                # No reliable angle calculation in this case
                self.prev_midline_angle = None
            
            method = "fallback"
            quality = "fair"
        
        # If frame shape is provided, extend the line to cover the entire frame
        if frame_shape is not None:
            height, width = frame_shape[:2]
            
            # Recalculate glabella and chin to extend across the frame
            direction = chin - glabella
            if np.any(direction):
                direction = direction / np.sqrt(np.sum(direction ** 2))
                
                # Parametric line equation: point = start + t * direction
                # Find t values for intersections with frame boundaries
                t_vals = []
                
                # Top boundary (y=0)
                if abs(direction[1]) > 1e-6:
                    t = -glabella[1] / direction[1]
                    x_intersect = glabella[0] + t * direction[0]
                    if 0 <= x_intersect <= width:
                        t_vals.append(t)
                
                # Bottom boundary (y=height-1)
                if abs(direction[1]) > 1e-6:
                    t = (height - 1 - glabella[1]) / direction[1]
                    x_intersect = glabella[0] + t * direction[0]
                    if 0 <= x_intersect <= width:
                        t_vals.append(t)
                
                # Left boundary (x=0)
                if abs(direction[0]) > 1e-6:
                    t = -glabella[0] / direction[0]
                    y_intersect = glabella[1] + t * direction[1]
                    if 0 <= y_intersect <= height:
                        t_vals.append(t)
                
                # Right boundary (x=width-1)
                if abs(direction[0]) > 1e-6:
                    t = (width - 1 - glabella[0]) / direction[0]
                    y_intersect = glabella[1] + t * direction[1]
                    if 0 <= y_intersect <= height:
                        t_vals.append(t)
                
                # Sort t values
                t_vals.sort()
                
                # Use the first and last t values to get the endpoints
                if len(t_vals) >= 2:
                    glabella = glabella + t_vals[0] * direction
                    chin = glabella + (t_vals[-1] - t_vals[0]) * direction
        
        # Calculate midline angle and slope for logging
        try:
            midline_direction = chin - glabella
            midline_angle = np.arctan2(midline_direction[1], midline_direction[0])
            midline_angle_degrees = np.degrees(midline_angle)
            if abs(midline_direction[0]) > 1e-6:
                midline_slope = midline_direction[1] / midline_direction[0]
            else:
                midline_slope = float('inf')  # Vertical line
        except Exception:
            midline_angle_degrees = 0
            midline_slope = 0
            
        # Calculate stability metrics
        glabella_delta_x = 0
        glabella_delta_y = 0
        chin_delta_x = 0
        chin_delta_y = 0
        pitch_delta = 0
        yaw_delta = 0
        roll_delta = 0
        
        if hasattr(self, 'prev_midline_glabella') and self.prev_midline_glabella is not None:
            glabella_delta_x = glabella[0] - self.prev_midline_glabella[0]
            glabella_delta_y = glabella[1] - self.prev_midline_glabella[1]
            
        if hasattr(self, 'prev_midline_chin') and self.prev_midline_chin is not None:
            chin_delta_x = chin[0] - self.prev_midline_chin[0]
            chin_delta_y = chin[1] - self.prev_midline_chin[1]
            
        if pitch is not None and hasattr(self, 'prev_head_pitch') and self.prev_head_pitch is not None:
            pitch_delta = pitch - self.prev_head_pitch
            
        if yaw is not None and hasattr(self, 'prev_head_yaw') and self.prev_head_yaw is not None:
            yaw_delta = yaw - self.prev_head_yaw
            
        if roll is not None and hasattr(self, 'prev_head_roll') and self.prev_head_roll is not None:
            roll_delta = roll - self.prev_head_roll
        
        # Store current values for next frame
        self.prev_midline_glabella = glabella
        self.prev_midline_chin = chin
        self.prev_head_pitch = pitch
        self.prev_head_yaw = yaw
        self.prev_head_roll = roll
        
        # Prepare data for logging
        log_data = {
            # Head orientation
            "pitch": pitch if pitch is not None else 0,
            "yaw": yaw if yaw is not None else 0,
            "roll": roll if roll is not None else 0,
            "eye_angle": eye_angle_degrees,
            
            # Raw landmarks used for midline
            "left_brow_x": left_medial_brow[0],
            "left_brow_y": left_medial_brow[1],
            "right_brow_x": right_medial_brow[0],
            "right_brow_y": right_medial_brow[1],
            "nose_bridge_x": nose_bridge[1][0],
            "nose_bridge_y": nose_bridge[1][1],
            "nose_tip_x": nose_tip[0],
            "nose_tip_y": nose_tip[1],
            "philtrum_x": philtrum[0],
            "philtrum_y": philtrum[1],
            "chin_point_x": chin[0],
            "chin_point_y": chin[1],
            
            # Midline characteristics
            "midline_angle": midline_angle_degrees,
            "midline_slope": midline_slope,
            
            # Stability metrics
            "glabella_delta_x": glabella_delta_x,
            "glabella_delta_y": glabella_delta_y,
            "chin_delta_x": chin_delta_x,
            "chin_delta_y": chin_delta_y,
            "pitch_delta": pitch_delta,
            "yaw_delta": yaw_delta,
            "roll_delta": roll_delta,
            
            # Method information
            "tilt_factor": tilt_factor,
            "method": method,
            "quality": quality,
            "smoothing_strength": len(self.landmarks_history)
        }
        
        # Add to history for temporal smoothing
        self.glabella_history.append(glabella)
        self.chin_history.append(chin)
        
        if len(self.glabella_history) > self.history_size:
            self.glabella_history.pop(0)
        if len(self.chin_history) > self.history_size:
            self.chin_history.pop(0)
        
        # Calculate smooth midline points with exponential weighting
        # This gives more weight to recent frames while still maintaining stability
        if len(self.glabella_history) > 1:
            weights = np.exp(np.linspace(-1, 0, len(self.glabella_history)))
            weights = weights / np.sum(weights)
            
            smooth_glabella = np.zeros_like(glabella)
            smooth_chin = np.zeros_like(chin)
            
            for i, (g, c) in enumerate(zip(self.glabella_history, self.chin_history)):
                smooth_glabella += g * weights[i]
                smooth_chin += c * weights[i]
        else:
            smooth_glabella = glabella
            smooth_chin = chin
            
        # Add debugging to find any problematic calls
        try:
            # Detect if this is called from somewhere unexpected
            import inspect
            stack = inspect.stack()
            if len(stack) > 1:
                caller = stack[1].function
                if caller not in ['create_mirrored_faces', 'create_debug_frame', 'process_video']:
                    if self.debug_mode:
                        print(f"WARNING: get_facial_midline called from {caller}")
                    # Return just two values for backward compatibility
                    return smooth_glabella, smooth_chin
        except Exception:
            pass
            
        return smooth_glabella, smooth_chin, log_data

    def create_mirrored_faces(self, frame, landmarks):
        """Create mirrored faces by reflecting exactly along the anatomical midline with tilt compensation"""
        height, width = frame.shape[:2]

        # Get midline points with frame shape for full extension
        glabella, chin, _ = self.get_facial_midline(landmarks, frame.shape)

        if glabella is None or chin is None:
            return frame.copy(), frame.copy()

        # Calculate midline direction vector with tilt compensation
        direction = chin - glabella
        
        # Get head pose information if available
        if self.pose_history:
            avg_roll = np.mean([p[2] for p in self.pose_history])
            
            # Apply tilt correction to the direction vector
            if abs(avg_roll) > 0.1:  # Only correct significant tilt
                # Rotate the direction vector to compensate for head tilt
                correction_angle = -avg_roll * 0.3  # Partial correction (30%)
                
                # Create rotation matrix
                cos_angle = np.cos(correction_angle)
                sin_angle = np.sin(correction_angle)
                rotation_mat = np.array([
                    [cos_angle, -sin_angle],
                    [sin_angle, cos_angle]
                ])
                
                # Apply rotation to direction vector
                direction = np.dot(rotation_mat, direction)
        
        # Normalize the direction vector
        direction = direction / np.sqrt(np.sum(direction ** 2))

        # Create meshgrid of coordinates
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

        # Reshape coordinates for vectorized operations
        coords_reshaped = coords.reshape(-1, 2)

        # Calculate perpendicular vector (pointing to anatomical right)
        # Note: OpenCV's coordinate system has y increasing downward
        perpendicular = np.array([-direction[1], direction[0]])

        # Calculate signed distances from midline
        # Positive distances are on anatomical right, negative on anatomical left
        diff = coords_reshaped - glabella
        distances = np.dot(diff, perpendicular)

        # Calculate reflection points
        reflection = coords_reshaped - 2 * np.outer(distances, perpendicular)
        reflection = reflection.reshape(height, width, 2)

        # Clip reflection coordinates to image bounds
        reflection[..., 0] = np.clip(reflection[..., 0], 0, width - 1)
        reflection[..., 1] = np.clip(reflection[..., 1], 0, height - 1)
        reflection = reflection.astype(np.int32)

        # Create output images
        anatomical_right_face = frame.copy()
        anatomical_left_face = frame.copy()

        # Create side masks based on signed distance
        distances = distances.reshape(height, width)
        anatomical_right_mask = distances >= 0  # Points on anatomical right
        anatomical_left_mask = distances < 0  # Points on anatomical left

        # Apply reflections
        for i in range(3):  # For each color channel
            # Anatomical right face: keep right side, reflect left side
            reflected_vals = frame[reflection[..., 1], reflection[..., 0], i]
            anatomical_right_face[..., i][anatomical_left_mask] = reflected_vals[anatomical_left_mask]

            # Anatomical left face: keep left side, reflect right side
            anatomical_left_face[..., i][anatomical_right_mask] = reflected_vals[anatomical_right_mask]

        # Apply adaptive gradient blending along the midline
        # Wider blending for more severe tilt
        blend_width = 20  # Base width in pixels
        
        # Adjust blend width based on head tilt if available
        if self.pose_history:
            avg_roll = np.mean([p[2] for p in self.pose_history])
            tilt_factor = abs(np.sin(avg_roll))
            blend_width = int(blend_width * (1 + tilt_factor))

        # Calculate points along the midline
        y_range = np.arange(height)
        if abs(direction[1]) > 1e-6:  # Avoid division by zero
            t = (y_range - glabella[1]) / direction[1]
            x_midline = (glabella[0] + t * direction[0]).astype(int)
            x_midline = np.clip(x_midline, blend_width // 2, width - blend_width // 2 - 1)

            # Create blending weights
            blend_weights = np.linspace(0, 1, blend_width)

            # Vectorized blending
            for i in range(blend_width):
                weight = blend_weights[i]
                x_offset = i - blend_width // 2
                x_coords = np.clip(x_midline + x_offset, 0, width - 1)

                # Apply blending using array indexing
                blend_weight = weight if x_offset >= 0 else (1 - weight)
                anatomical_right_face[y_range, x_coords] = (
                        anatomical_right_face[y_range, x_coords] * (1 - blend_weight) +
                        frame[y_range, x_coords] * blend_weight
                )
                anatomical_left_face[y_range, x_coords] = (
                        anatomical_left_face[y_range, x_coords] * (1 - blend_weight) +
                        frame[y_range, x_coords] * blend_weight
                )

        return anatomical_right_face, anatomical_left_face

    def create_debug_frame(self, frame, landmarks):
        """Create debug visualization with all 68 landmarks, anatomical midline, and tilt angle"""
        debug_frame = frame.copy()

        if landmarks is not None:
            # Draw all 68 landmarks as dots
            for point in landmarks:
                x, y = point.astype(int)
                # Draw point
                cv2.circle(debug_frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dot

            # Highlight medial eyebrow points (21, 22)
            left_medial_brow = tuple(map(int, landmarks[21]))
            right_medial_brow = tuple(map(int, landmarks[22]))
            cv2.circle(debug_frame, left_medial_brow, 4, (255, 0, 0), -1)  # Blue dot
            cv2.circle(debug_frame, right_medial_brow, 4, (255, 0, 0), -1)  # Blue dot

            # Get midline points
            glabella, chin, _ = self.get_facial_midline(landmarks, frame.shape)

            if glabella is not None and chin is not None:
                # Convert points to integer coordinates for drawing
                glabella = tuple(map(int, glabella))
                chin = tuple(map(int, chin))

                # Draw midline points
                cv2.circle(debug_frame, glabella, 4, (0, 255, 0), -1)  # Green dot for glabella
                cv2.circle(debug_frame, chin, 4, (0, 255, 0), -1)  # Green dot for chin

                # Calculate and draw the extended midline
                height, width = frame.shape[:2]
                direction = np.array([chin[0] - glabella[0], chin[1] - glabella[1]])

                if np.any(direction):
                    direction = direction / np.sqrt(np.sum(direction ** 2))
                    extension_length = np.sqrt(height ** 2 + width ** 2)

                    top_point = (
                        int(glabella[0] - direction[0] * extension_length),
                        int(glabella[1] - direction[1] * extension_length)
                    )
                    bottom_point = (
                        int(chin[0] + direction[0] * extension_length),
                        int(chin[1] + direction[1] * extension_length)
                    )

                    # Draw the extended midline
                    cv2.line(debug_frame, top_point, bottom_point, (0, 0, 255), 2)  # Red line

            # Calculate and display tilt information
            left_eye_center = np.mean(landmarks[36:42], axis=0).astype(int)
            right_eye_center = np.mean(landmarks[42:48], axis=0).astype(int)
            
            # Draw eye centers and eye line
            cv2.circle(debug_frame, tuple(left_eye_center), 5, (255, 0, 255), -1)  # Magenta dot
            cv2.circle(debug_frame, tuple(right_eye_center), 5, (255, 0, 255), -1)  # Magenta dot
            cv2.line(debug_frame, tuple(left_eye_center), tuple(right_eye_center), (255, 0, 255), 2)  # Magenta line
            
            # Calculate tilt angle
            tilt_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], 
                                   right_eye_center[0] - left_eye_center[0])
            tilt_degrees = np.degrees(tilt_angle)
            
            # Display tilt information
            cv2.putText(debug_frame, f"Tilt: {tilt_degrees:.1f} deg", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        
            # Display quality indicator based on tilt
            quality = "Good" if abs(tilt_degrees) < 10 else "Fair" if abs(tilt_degrees) < 25 else "Poor"
            quality_color = (0, 255, 0) if quality == "Good" else (0, 255, 255) if quality == "Fair" else (0, 0, 255)
            cv2.putText(debug_frame, f"Quality: {quality}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

            # Draw face boundary
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.polylines(debug_frame, [hull], True, (255, 255, 0), 2)

            # Add legend
            legend_y = 30
            cv2.putText(debug_frame, "Yellow: All landmarks", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(debug_frame, "Blue: Medial eyebrow points", (10, legend_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(debug_frame, "Green: Calculated midline points", (10, legend_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_frame, "Red: Extended midline", (10, legend_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(debug_frame, "Magenta: Eye centers & tilt", (10, legend_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return debug_frame
        
    def preprocess_tilted_frame(self, frame, landmarks):
        """Optionally preprocess frame to correct severe tilt before mirroring"""
        if landmarks is None or len(self.pose_history) == 0:
            return frame
            
        # Get average roll angle from pose history
        avg_roll = np.mean([p[2] for p in self.pose_history])
        
        # Only correct severe tilt
        if abs(avg_roll) < 0.2:  # Less than ~11 degrees
            return frame
            
        # Calculate face center
        face_center = tuple(map(int, np.mean(landmarks, axis=0)))
        
        # Get rotation matrix
        correction_angle = -avg_roll  # Full correction
        angle_degrees = np.degrees(correction_angle)
        rotation_matrix = cv2.getRotationMatrix2D(face_center, angle_degrees, 1.0)
        
        # Apply rotation
        height, width = frame.shape[:2]
        corrected_frame = cv2.warpAffine(frame, rotation_matrix, (width, height), 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_REPLICATE)
                                         
        # Note: we would need to transform landmarks too if we used this approach
        # For now, we're not using this function directly - instead applying smaller 
        # corrections throughout the pipeline
        
        return corrected_frame

    def analyze_midline_stability(self, log_path):
        """Analyze midline stability data from the log file"""
        if not Path(log_path).exists():
            print(f"Log file not found: {log_path}")
            return
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Load the data
            df = pd.read_csv(log_path)
            
            # Calculate frame-to-frame changes
            df['total_position_delta'] = np.sqrt(
                df['glabella_delta_x']**2 + df['glabella_delta_y']**2 +
                df['chin_delta_x']**2 + df['chin_delta_y']**2
            )
            
            # Calculate head movement metrics
            df['total_head_movement'] = np.sqrt(
                df['pitch_delta']**2 + df['yaw_delta']**2 + df['roll_delta']**2
            )
            
            # Calculate midline angle delta
            df['midline_angle_delta'] = df['midline_angle'].diff().abs()
            
            # Mark method changes
            df['method_change'] = df['method'] != df['method'].shift(1)
            
            # Create stability analysis plots
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot 1: Midline position changes
            axes[0].plot(df['frame'], df['total_position_delta'], 'b-', label='Position Change')
            axes[0].set_title('Midline Position Stability')
            axes[0].set_ylabel('Pixel Change')
            axes[0].grid(True)
            axes[0].legend()
            
            # Plot 2: Head pose and midline angle
            axes[1].plot(df['frame'], df['roll'], 'r-', label='Head Roll')
            axes[1].plot(df['frame'], df['midline_angle'], 'b-', label='Midline Angle')
            axes[1].set_title('Head Pose vs Midline Angle')
            axes[1].set_ylabel('Angle (degrees)')
            axes[1].grid(True)
            axes[1].legend()
            
            # Plot 3: Method changes and quality
            sc = axes[2].scatter(df['frame'], df['midline_angle_delta'], 
                              c=df['method'].map({'standard': 0, 'RANSAC': 1, 'fallback': 2}),
                              cmap='viridis', s=10, alpha=0.7)
            # Highlight method changes
            method_changes = df[df['method_change']]
            axes[2].scatter(method_changes['frame'], method_changes['midline_angle_delta'],
                         color='red', s=30, marker='x')
            
            axes[2].set_title('Midline Angle Changes and Method Transitions')
            axes[2].set_ylabel('Angle Delta (degrees)')
            axes[2].set_xlabel('Frame')
            axes[2].grid(True)
            
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=sc.cmap(0), markersize=8, label='Standard'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=sc.cmap(0.5), markersize=8, label='RANSAC'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=sc.cmap(1), markersize=8, label='Fallback'),
                plt.Line2D([0], [0], marker='x', color='red', markersize=8, label='Method Change')
            ]
            axes[2].legend(handles=legend_elements)
            
            # Add overall statistics
            stat_text = (
                f"Total Frames: {len(df)}\n"
                f"Avg Position Delta: {df['total_position_delta'].mean():.2f} px\n"
                f"Avg Angle Delta: {df['midline_angle_delta'].mean():.2f}°\n"
                f"Method Changes: {df['method_change'].sum()}\n"
                f"Standard Method: {(df['method'] == 'standard').mean()*100:.1f}%\n"
                f"RANSAC Method: {(df['method'] == 'RANSAC').mean()*100:.1f}%\n"
                f"Fallback Method: {(df['method'] == 'fallback').mean()*100:.1f}%"
            )
            
            plt.figtext(0.02, 0.02, stat_text, fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the figure
            output_path = str(Path(log_path).with_suffix('.png'))
            plt.savefig(output_path)
            plt.close()
            
            print(f"Stability analysis saved to: {output_path}")
            
            # Create a detailed summary
            summary = {
                'total_frames': len(df),
                'mean_position_delta': df['total_position_delta'].mean(),
                'max_position_delta': df['total_position_delta'].max(),
                'method_changes': df['method_change'].sum(),
                'standard_method_percent': (df['method'] == 'standard').mean()*100,
                'ransac_method_percent': (df['method'] == 'RANSAC').mean()*100,
                'fallback_method_percent': (df['method'] == 'fallback').mean()*100,
                'mean_head_movement': df['total_head_movement'].mean(),
                'stability_rating': 'Unknown',
                'method_stability': 'Unknown'
            }
            
            # Add stability ratings
            if summary['mean_position_delta'] < 1:
                summary['stability_rating'] = 'Excellent'
            elif summary['mean_position_delta'] < 2:
                summary['stability_rating'] = 'Good'
            elif summary['mean_position_delta'] < 5:
                summary['stability_rating'] = 'Fair'
            else:
                summary['stability_rating'] = 'Poor'
                
            method_changes_per_frame = summary['method_changes'] / summary['total_frames']
            if method_changes_per_frame < 0.01:
                summary['method_stability'] = 'Excellent'
            elif method_changes_per_frame < 0.05:
                summary['method_stability'] = 'Good'
            elif method_changes_per_frame < 0.1:
                summary['method_stability'] = 'Fair'
            else:
                summary['method_stability'] = 'Poor'
            
            # Print summary
            print("\nStability Analysis Summary:")
            print(f"- Total Frames: {summary['total_frames']}")
            print(f"- Average Midline Movement: {summary['mean_position_delta']:.2f} pixels")
            print(f"- Maximum Midline Movement: {summary['max_position_delta']:.2f} pixels")
            print(f"- Method Changes: {summary['method_changes']} ({method_changes_per_frame*100:.1f}%)")
            print(f"- Method Usage: Standard {summary['standard_method_percent']:.1f}%, "
                 f"RANSAC {summary['ransac_method_percent']:.1f}%, "
                 f"Fallback {summary['fallback_method_percent']:.1f}%")
            print(f"- Overall Stability Rating: {summary['stability_rating']}")
            print(f"- Method Stability Rating: {summary['method_stability']}")
            
            return summary
            
        except Exception as e:
            print(f"Error analyzing stability data: {str(e)}")
            return None

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


def main():
    """Command-line interface with enhanced stability analysis"""
    try:
        root = tk.Tk()
        root.withdraw()
        input_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not input_paths:
            return

        output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ask if the user wants to enable stability analysis
        analyze_stability = messagebox.askyesno(
            "Stability Analysis", 
            "Would you like to generate stability analysis graphs?\n\n"
            "This requires pandas and matplotlib to be installed."
        )

        splitter = StableFaceSplitter(debug_mode=True, log_midline=True)
        results = []

        for input_path in input_paths:
            try:
                outputs = splitter.process_video(input_path, output_dir, analyze_stability)
                results.append({
                    'input': input_path,
                    'success': True,
                    'outputs': outputs
                })
            except Exception as e:
                results.append({
                    'input': input_path,
                    'success': False,
                    'error': str(e)
                })

        summary = "Processing Results:\n\n"
        for result in results:
            if result['success']:
                summary += f"✓ {Path(result['input']).name}\n"

                # Separate output types
                left_video = next((f for f in result['outputs'] if 'left_mirrored' in f), None)
                right_video = next((f for f in result['outputs'] if 'right_mirrored' in f), None)
                debug_video = next((f for f in result['outputs'] if 'debug' in f), None)
                rotated_video = next((f for f in result['outputs'] if 'rotated' in f), None)
                log_file = next((f for f in result['outputs'] if 'midline_log.csv' in f), None)
                stability_image = next((f for f in result['outputs'] if 'midline_log.png' in f), None)

                if left_video:
                    summary += f"  - Left: {Path(left_video).name}\n"
                if right_video:
                    summary += f"  - Right: {Path(right_video).name}\n"
                if debug_video:
                    summary += f"  - Debug: {Path(debug_video).name}\n"
                if rotated_video:
                    summary += f"  - Rotated: {Path(rotated_video).name}\n"
                if log_file:
                    summary += f"  - Log: {Path(log_file).name}\n"
                if stability_image:
                    summary += f"  - Stability Analysis: {Path(stability_image).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
