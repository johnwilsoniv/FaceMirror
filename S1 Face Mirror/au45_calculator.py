#!/usr/bin/env python3
"""
AU45 (Blink) Calculator using Eye Aspect Ratio (EAR)

Calculates AU45 intensity (0-5 scale) from facial landmarks using the
Eye Aspect Ratio method, which is the gold standard for blink detection.

EAR Formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
Where p1-p6 are eye corner and lid landmarks

AU45 Scale:
- 0.0: Eyes fully open
- 5.0: Eyes tightly closed
"""

import numpy as np


class AU45Calculator:
    """Calculate AU45 (Blink) from eye landmarks using Eye Aspect Ratio"""

    def __init__(self):
        """
        Initialize AU45 calculator with WFLW 98-point landmark indices

        WFLW 98-point format eye landmarks:
        - Right eye: indices 60-67 (8 points) - patient's right, image left
        - Left eye: indices 68-75 (8 points) - patient's left, image right

        Note: WFLW uses image coordinates, so left/right are from image perspective
        """
        # WFLW 98-point eye landmark indices
        # These are the standard indices for WFLW format
        self.right_eye_indices = list(range(60, 68))  # 8 points
        self.left_eye_indices = list(range(68, 76))   # 8 points

        # EAR calibration parameters
        # These values are empirically determined from WFLW 98-point landmarks
        # WFLW uses 8 points per eye (not 6), resulting in higher EAR values
        self.ear_open = 0.75      # Typical EAR when eyes are fully open (WFLW format)
        self.ear_closed = 0.60    # Typical EAR when eyes are closed during blink (WFLW format)
        self.ear_threshold = 0.68 # Below this threshold = eyes are closing

        # Temporal smoothing to reduce jitter
        self.au45_history = []
        self.history_size = 3  # Smooth over last 3 frames

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio for a single eye

        Uses the standard 6-point EAR formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Where:
        - p1, p4 are the horizontal eye corners
        - p2, p3, p5, p6 are the vertical eyelid points

        Args:
            eye_landmarks: (N, 2) numpy array of eye landmark coordinates
                          Must have at least 6 points

        Returns:
            float: Eye Aspect Ratio, or None if calculation fails
        """
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return None

        try:
            # Use first 6 points for standard EAR calculation
            # WFLW format: [outer_corner, upper_lid, upper_lid, inner_corner, lower_lid, lower_lid]
            p1 = eye_landmarks[0]  # Outer corner
            p2 = eye_landmarks[1]  # Upper lid point 1
            p3 = eye_landmarks[2]  # Upper lid point 2
            p4 = eye_landmarks[3]  # Inner corner
            p5 = eye_landmarks[4]  # Lower lid point 1
            p6 = eye_landmarks[5]  # Lower lid point 2

            # Calculate vertical distances (eyelid opening height)
            vertical_dist_1 = np.linalg.norm(p2 - p6)
            vertical_dist_2 = np.linalg.norm(p3 - p5)

            # Calculate horizontal distance (eye width)
            horizontal_dist = np.linalg.norm(p1 - p4)

            # Avoid division by zero
            if horizontal_dist < 1e-6:
                return None

            # Calculate EAR
            ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)

            return ear

        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return None

    def ear_to_au45(self, ear):
        """
        Convert Eye Aspect Ratio to AU45 intensity on 0-5 scale

        Mapping:
        - EAR >= ear_open (0.25): AU45 = 0 (fully open)
        - EAR <= ear_closed (0.05): AU45 = 5 (tightly closed)
        - Linear interpolation between these values

        Args:
            ear: Eye Aspect Ratio (float)

        Returns:
            float: AU45 intensity (0.0 to 5.0), or np.nan if ear is None
        """
        if ear is None:
            return np.nan

        # Clamp EAR to expected range
        ear_clamped = np.clip(ear, self.ear_closed, self.ear_open)

        # Calculate closure ratio
        # 0 = fully open, 1 = fully closed
        closure_ratio = 1.0 - ((ear_clamped - self.ear_closed) /
                               (self.ear_open - self.ear_closed))

        # Map to 0-5 scale (OpenFace intensity scale)
        au45_intensity = closure_ratio * 5.0

        return float(au45_intensity)

    def calculate_au45_from_landmarks(self, landmarks_98, debug=False):
        """
        Calculate AU45 (blink) intensity from WFLW 98-point landmarks

        This is the main public interface for AU45 calculation.

        Process:
        1. Extract left and right eye landmarks
        2. Calculate EAR for each eye
        3. Average the EARs
        4. Convert to AU45 intensity (0-5 scale)
        5. Apply temporal smoothing

        Args:
            landmarks_98: (98, 2) numpy array of WFLW facial landmarks
            debug: Print debug information (default: False)

        Returns:
            float: AU45 intensity (0.0 to 5.0), or np.nan if calculation fails
        """
        if landmarks_98 is None or len(landmarks_98) < 76:
            if debug:
                print(f"  [AU45 Debug] landmarks_98 is None or too short")
            return np.nan

        try:
            # Extract eye landmarks using WFLW indices
            right_eye = landmarks_98[self.right_eye_indices]
            left_eye = landmarks_98[self.left_eye_indices]

            if debug:
                print(f"  [AU45 Debug] Right eye landmarks shape: {right_eye.shape}")
                print(f"  [AU45 Debug] Left eye landmarks shape: {left_eye.shape}")

            # Calculate EAR for both eyes
            right_ear = self.calculate_ear(right_eye)
            left_ear = self.calculate_ear(left_eye)

            if debug:
                print(f"  [AU45 Debug] Right EAR: {right_ear}, Left EAR: {left_ear}")

            # Average both eyes (or use available eye if one fails)
            if right_ear is not None and left_ear is not None:
                avg_ear = (right_ear + left_ear) / 2.0
            elif right_ear is not None:
                avg_ear = right_ear
            elif left_ear is not None:
                avg_ear = left_ear
            else:
                return np.nan

            # Convert EAR to AU45 intensity
            au45_raw = self.ear_to_au45(avg_ear)

            if np.isnan(au45_raw):
                return np.nan

            # Apply temporal smoothing to reduce jitter
            self.au45_history.append(au45_raw)
            if len(self.au45_history) > self.history_size:
                self.au45_history.pop(0)

            # Weighted average (give more weight to recent frames)
            weights = np.linspace(0.5, 1.0, len(self.au45_history))
            weights = weights / np.sum(weights)
            au45_smooth = np.average(self.au45_history, weights=weights)

            # Clamp to valid range
            au45_final = np.clip(au45_smooth, 0.0, 5.0)

            return float(au45_final)

        except Exception as e:
            print(f"Error calculating AU45 from landmarks: {e}")
            return np.nan

    def reset(self):
        """
        Reset smoothing history

        Call this between videos to prevent temporal smoothing from
        carrying over between different video sessions.
        """
        self.au45_history = []


def test_au45_calculator():
    """Test function to verify AU45 calculator with sample landmarks"""
    print("Testing AU45 Calculator...")

    # Create sample landmarks (98 points)
    # Simulate fully open eyes (high EAR ~0.25)
    landmarks_open = np.random.rand(98, 2) * 100
    # Set eye landmarks to simulate open eyes
    for i in range(60, 76):
        landmarks_open[i] = np.array([50 + (i % 8) * 5, 50 + np.random.rand() * 2])

    calculator = AU45Calculator()
    au45_open = calculator.calculate_au45_from_landmarks(landmarks_open)
    print(f"AU45 (open eyes): {au45_open:.2f} (expected ~0.0)")

    # Simulate closed eyes (low EAR ~0.05)
    landmarks_closed = landmarks_open.copy()
    for i in range(60, 76):
        landmarks_closed[i][1] = 50  # Collapse vertical distance

    calculator.reset()
    au45_closed = calculator.calculate_au45_from_landmarks(landmarks_closed)
    print(f"AU45 (closed eyes): {au45_closed:.2f} (expected ~5.0)")

    print("AU45 Calculator test complete!")


if __name__ == "__main__":
    test_au45_calculator()
