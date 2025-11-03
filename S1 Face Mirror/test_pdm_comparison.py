#!/usr/bin/env python3
"""
Generate comparison images showing original vs PDM-constrained landmarks.
"""

import cv2
import numpy as np
from pyfaceau_detector import PyFaceAU68LandmarkDetector

def create_comparison_images():
    """Create side-by-side comparison of original vs PDM-constrained landmarks."""

    print('Creating landmark comparison images...')
    print('=' * 70)

    # Load problematic video
    cap = cv2.VideoCapture('/tmp/IMG_8401_rotated.MOV')

    # Initialize detector WITHOUT PDM fallback (original behavior)
    detector_original = PyFaceAU68LandmarkDetector(
        debug_mode=False,
        skip_face_detection=False,
        use_clnf_refinement=True,
        enable_pdm_fallback=False  # Disabled
    )

    # Initialize detector WITH PDM fallback
    detector_pdm = PyFaceAU68LandmarkDetector(
        debug_mode=False,
        skip_face_detection=False,
        use_clnf_refinement=True,
        enable_pdm_fallback=True  # Enabled
    )

    # Process frame 0
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        return

    # Get landmarks from both detectors
    print("Getting original landmarks...")
    landmarks_original, _ = detector_original.get_face_mesh(frame)

    print("Getting PDM-constrained landmarks...")
    landmarks_pdm, _ = detector_pdm.get_face_mesh(frame)

    # Create comparison image
    h, w = frame.shape[:2]
    comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)

    # Left: Original landmarks
    frame_original = frame.copy()
    if landmarks_original is not None:
        # Check quality
        is_poor, reason = detector_original.check_landmark_quality(landmarks_original)

        # Draw landmarks
        for (x, y) in landmarks_original:
            cv2.circle(frame_original, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Add quality label
        quality_text = f"Original: {reason.upper()}"
        cv2.putText(frame_original, quality_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw midline
        x_coords = landmarks_original[:, 0]
        x_center = (np.min(x_coords) + np.max(x_coords)) / 2
        cv2.line(frame_original, (int(x_center), 0), (int(x_center), h), (0, 0, 255), 2)

    # Right: PDM-constrained landmarks
    frame_pdm = frame.copy()
    if landmarks_pdm is not None:
        # Check quality
        is_poor, reason = detector_pdm.check_landmark_quality(landmarks_pdm)

        # Draw landmarks
        for (x, y) in landmarks_pdm:
            cv2.circle(frame_pdm, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Add quality label
        quality_text = f"PDM-Constrained: {reason.upper()}"
        cv2.putText(frame_pdm, quality_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if not is_poor else (0, 165, 255), 2)

        # Draw midline
        x_coords = landmarks_pdm[:, 0]
        x_center = (np.min(x_coords) + np.max(x_coords)) / 2
        cv2.line(frame_pdm, (int(x_center), 0), (int(x_center), h), (0, 255, 0), 2)

    # Combine images
    comparison[:, 0:w] = frame_original
    comparison[:, w+20:] = frame_pdm

    # Add separator
    comparison[:, w:w+20] = 255

    # Save comparison
    output_path = '/tmp/IMG_8401_landmark_comparison.jpg'
    cv2.imwrite(output_path, comparison)
    print(f"\nComparison saved: {output_path}")
    print('=' * 70)

    # Print statistics
    if landmarks_original is not None and landmarks_pdm is not None:
        print("\nLandmark Statistics:")
        print("-" * 70)

        # Calculate displacement
        displacement = np.linalg.norm(landmarks_pdm - landmarks_original, axis=1)
        print(f"  Mean displacement: {np.mean(displacement):.2f} pixels")
        print(f"  Max displacement: {np.max(displacement):.2f} pixels")

        # Calculate clustering ratio
        x_orig = landmarks_original[:, 0]
        x_pdm = landmarks_pdm[:, 0]

        x_center_orig = (np.min(x_orig) + np.max(x_orig)) / 2
        x_center_pdm = (np.min(x_pdm) + np.max(x_pdm)) / 2

        cluster_orig = max(np.sum(x_orig < x_center_orig), np.sum(x_orig >= x_center_orig)) / 68.0
        cluster_pdm = max(np.sum(x_pdm < x_center_pdm), np.sum(x_pdm >= x_center_pdm)) / 68.0

        print(f"\n  Clustering ratio:")
        print(f"    Original:        {cluster_orig:.2f}")
        print(f"    PDM-constrained: {cluster_pdm:.2f}")

        # Calculate spatial distribution
        x_std_orig = np.std(x_orig)
        y_std_orig = np.std(landmarks_original[:, 1])
        x_std_pdm = np.std(x_pdm)
        y_std_pdm = np.std(landmarks_pdm[:, 1])

        print(f"\n  Spatial distribution (std dev):")
        print(f"    Original:        X={x_std_orig:.1f}px  Y={y_std_orig:.1f}px")
        print(f"    PDM-constrained: X={x_std_pdm:.1f}px  Y={y_std_pdm:.1f}px")
        print('=' * 70)


if __name__ == '__main__':
    create_comparison_images()
