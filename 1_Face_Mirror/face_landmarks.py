import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay

class FaceLandmarkDetector:
    """Handles face detection and landmark tracking"""
    
    def __init__(self, debug_mode=False):
        """Initialize face detector and landmark predictor"""
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Tracking history
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0
        
        # Initialize smoothing history
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.history_size = 5
        
        # Face ROI tracking
        self.face_roi = None
        
        # Head pose history for stability calculation
        self.yaw_history = []
        self.pose_history_size = 10
        
        # Frame quality history
        self.frame_quality_history = []
        
        # Debug mode
        self.debug_mode = debug_mode
    
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
        self.yaw_history = []
        self.frame_quality_history = []

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

        if self.face_roi is None:
            self.face_roi = current_roi
        else:
            # Smooth ROI transition
            alpha = 0.7  # Smoothing factor
            self.face_roi = [int(alpha * prev + (1 - alpha) * curr)
                            for prev, curr in zip(self.face_roi, current_roi)]
    
    def get_face_mesh(self, frame, detection_interval=2):
        """Get facial landmarks with temporal smoothing"""
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % detection_interval == 0 or self.last_face is None:
            faces = self.detector(gray)
            if not faces:
                self.last_face = None
                self.last_landmarks = None
                return None, None
            self.last_face = max(faces, key=lambda rect: rect.width() * rect.height())

        if self.last_face is not None:
            # Extract landmarks
            landmarks = self.predictor(gray, self.last_face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)  # Explicitly set dtype

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

            # Ensure integer coordinates for final output
            smoothed_points = np.round(smoothed_points).astype(np.int32)

            # Update yaw history
            yaw = self.calculate_head_pose(smoothed_points)
            self.yaw_history.append(yaw)
            if len(self.yaw_history) > self.pose_history_size:
                self.yaw_history.pop(0)

            # Calculate frame quality
            quality = self.calculate_frame_quality(smoothed_points)
            self.frame_quality_history.append(quality)
            if len(self.frame_quality_history) > self.pose_history_size:
                self.frame_quality_history.pop(0)

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
        z_scores = (distances - distances.mean()) / distances.std()
        if np.any(np.abs(z_scores) > 3):
            return False

        # Check for minimum face size
        face_width = landmarks[:, 0].max() - landmarks[:, 0].min()
        face_height = landmarks[:, 1].max() - landmarks[:, 1].min()
        if face_width < 60 or face_height < 60:
            return False

        return True
    
    def get_facial_midline(self, landmarks):
        """Calculate the anatomical midline points"""
        if landmarks is None:
            return None, None

        # Convert to float32 for calculations
        landmarks = landmarks.astype(np.float32)

        # Get medial eyebrow points
        left_medial_brow = landmarks[21]
        right_medial_brow = landmarks[22]

        # Calculate glabella and chin
        glabella = (left_medial_brow + right_medial_brow) / 2
        chin = landmarks[8]

        # Add to history
        self.glabella_history.append(glabella)
        self.chin_history.append(chin)

        if len(self.glabella_history) > self.history_size:
            self.glabella_history.pop(0)
        if len(self.chin_history) > self.history_size:
            self.chin_history.pop(0)

        # Calculate smooth midline points
        smooth_glabella = np.mean(self.glabella_history, axis=0)
        smooth_chin = np.mean(self.chin_history, axis=0)

        return smooth_glabella, smooth_chin

    def calculate_head_pose(self, landmarks):
        """
        Calculate head yaw using multiple facial reference points to handle
        facial asymmetry and deviated nasal features
        
        Returns:
        - yaw: Head rotation angle in degrees
        """
        if landmarks is None:
            return None
        
        # Convert to float32 for calculations
        landmarks = landmarks.astype(np.float32)
        
        # Use the existing midline calculation for efficiency
        glabella, chin = self.get_facial_midline(landmarks)
        
        if glabella is None or chin is None:
            return None
        
        # Calculate midline direction vector
        direction = chin - glabella
        if np.linalg.norm(direction) < 1e-6:
            return None
            
        direction = direction / np.linalg.norm(direction)
        
        # Calculate a point that's directly below glabella on the vertical midline
        vert_point = glabella + np.array([0, 100])
        
        # Calculate face center line (vertical line through center of face)
        # This will be more stable than just using the x-coordinate of glabella
        center_landmarks = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]  # Nose bridge, philtrum, center of lips, chin
        center_points = landmarks[center_landmarks]
        center_x = np.mean(center_points[:, 0])
        
        # Establish pairs of symmetric landmarks to compare
        # Each pair should be roughly symmetric across the face
        landmark_pairs = [
            # Eyes outer corners
            (36, 45),  # Left eye outer, Right eye outer
            # Eyes inner corners
            (39, 42),  # Left eye inner, Right eye inner
            # Eyebrows outer
            (17, 26),  # Left eyebrow outer, Right eyebrow outer
            # Mouth corners
            (48, 54),  # Left mouth corner, Right mouth corner
            # Cheeks/jaw
            (1, 15),   # Left jaw, Right jaw
            (4, 12)    # Left cheek, Right cheek
        ]
        
        # Calculate horizontal displacement ratios for each pair
        yaw_estimates = []
        weights = []
        
        for left_idx, right_idx in landmark_pairs:
            left_point = landmarks[left_idx]
            right_point = landmarks[right_idx]
            
            # Distance from center to each point
            left_dist = center_x - left_point[0]
            right_dist = right_point[0] - center_x
            
            # In a perfectly front-facing image, these should be equal
            # Calculate normalized ratio difference (will be ~0 for front-facing)
            if left_dist > 0 and right_dist > 0:
                # Get average distance as normalization factor
                avg_dist = (left_dist + right_dist) / 2
                
                # Calculate the ratio difference
                ratio_diff = (right_dist - left_dist) / avg_dist
                
                # Convert to degree estimate (scaling factor based on empirical observation)
                # A ratio difference of 1.0 would correspond to about 45 degrees
                yaw_estimate = ratio_diff * 45.0
                
                # Assign weights based on reliability of the landmark pair
                # Eye corners are usually most reliable
                weight = 1.0
                if (left_idx, right_idx) in [(36, 45), (39, 42)]:
                    weight = 2.0  # Higher weight for eye landmarks
                
                yaw_estimates.append(yaw_estimate)
                weights.append(weight)
        
        # Calculate weighted average of all estimates
        if yaw_estimates:
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            return np.average(yaw_estimates, weights=weights)
        
        # Fallback to simple nose offset if paired method fails
        eyes_center = (landmarks[39] + landmarks[42]) / 2
        nose_tip = landmarks[30]
        nose_offset = nose_tip[0] - eyes_center[0]
        face_width = np.linalg.norm(landmarks[36] - landmarks[45])
        if face_width > 0:
            normalized_offset = nose_offset / (face_width / 2)
            return normalized_offset * 45.0
            
        return None

    def validate_head_pose(self, landmarks, threshold=10.0):
        """
        Validates if the head yaw is within acceptable threshold for facial analysis
        
        Parameters:
        - landmarks: Facial landmarks array
        - threshold: Maximum allowed yaw angle in degrees
        
        Returns:
        - valid: Boolean indicating if yaw is valid
        - warning: Warning message if yaw is invalid, None otherwise
        """
        # Calculate head yaw angle
        yaw = self.calculate_head_pose(landmarks)
        
        if yaw is None:
            return False, "No valid head pose detected"
        
        # Check if yaw exceeds threshold
        if abs(yaw) > threshold:
            return False, f"Head rotation ({abs(yaw):.1f}°) exceeds {threshold}°"
        
        return True, None

    def calculate_face_stability(self):
        """
        Calculate face stability based on yaw history
        
        Returns:
        - stability: Float between 0.0 and 1.0 (1.0 = perfectly stable)
        - is_stable: Boolean indicating if face is considered stable
        """
        if len(self.yaw_history) < 3:  # Need at least 3 frames to calculate stability
            return 0.0, False
        
        # Filter out None values
        valid_yaw = [y for y in self.yaw_history if y is not None]
        
        if len(valid_yaw) < 3:
            return 0.0, False
        
        # Calculate standard deviation of yaw
        yaw_std = np.std(valid_yaw)
        
        # Convert to stability score (0.0 to 1.0)
        # A yaw_std of 0 means perfect stability (1.0 score)
        # A yaw_std of 5 or more means poor stability (0.0 score)
        max_std = 5.0  # Maximum expected standard deviation for normalization
        stability = max(0.0, 1.0 - (yaw_std / max_std))
        
        # Consider stable if stability score is above threshold
        stability_threshold = 0.7
        is_stable = stability >= stability_threshold
        
        return stability, is_stable

    def calculate_frame_quality(self, landmarks):
        """
        Calculate frame quality score based on head yaw
        
        Returns:
        - quality: Float between 0.0 and 1.0 (1.0 = perfect quality)
        """
        if landmarks is None:
            return 0.0
        
        # Calculate head yaw
        yaw = self.calculate_head_pose(landmarks)
        
        if yaw is None:
            return 0.0
        
        # Define ideal ranges for yaw
        ideal_range = 3.0    # Ideal range for yaw: ±3° (updated from 5°)
        
        # Calculate quality component for yaw (1.0 if within ideal range, decreasing as angle increases)
        # Yaw is critical for mirroring, so it has a steep penalty
        yaw_quality = max(0.0, 1.0 - (abs(yaw) - ideal_range) / 7.0) if abs(yaw) > ideal_range else 1.0
        
        # Calculate stability factor if we have history
        stability, _ = self.calculate_face_stability() if len(self.yaw_history) >= 3 else (0.5, False)
        
        # Weight the components to get overall quality
        overall_quality = (yaw_quality * 0.9) + (stability * 0.1)
        
        return overall_quality
