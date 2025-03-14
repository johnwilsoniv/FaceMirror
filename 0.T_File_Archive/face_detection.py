import cv2
import numpy as np
import math
import dlib
from scipy.spatial import Delaunay

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
