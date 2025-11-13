#!/usr/bin/env python3
"""
Test sigma transformation impact on convergence.

Compares CLNF convergence with and without sigma spatial correlation modeling.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add pyclnf to path
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pyclnf import CLNF


def test_sigma_impact():
    """Test convergence with and without sigma transformation."""

    print("=" * 70)
    print("Testing Sigma Transformation Impact on Convergence")
    print("=" * 70)

    # Load a real test image
    test_image_path = Path("Patient Data/Normal Cohort/IMG_0434.MOV")

    if test_image_path.exists():
        # Use first frame from video
        cap = cv2.VideoCapture(str(test_image_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to read video frame, using synthetic image")
            frame = create_synthetic_face()
        else:
            print(f"✓ Loaded frame from {test_image_path.name}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("Test video not found, using synthetic image")
        gray = create_synthetic_face()

    # Define a reasonable face bounding box
    height, width = gray.shape
    face_size = min(width, height) // 2
    face_bbox = (
        width // 2 - face_size // 2,
        height // 2 - face_size // 2,
        face_size,
        face_size
    )

    print(f"\nImage shape: {gray.shape}")
    print(f"Face bbox: {face_bbox}")

    # Test 1: WITH sigma transformation
    print("\n" + "=" * 70)
    print("Test 1: CLNF WITH Sigma Transformation (Spatial Correlation Modeling)")
    print("=" * 70)

    clnf_with_sigma = CLNF(
        model_dir="pyclnf/models",
        max_iterations=20,  # Increase for better convergence
        regularization=25.0,
        sigma=1.5,
        weight_multiplier=5.0  # Apply patch confidence weighting
    )

    # Verify sigma components loaded
    if clnf_with_sigma.ccnf.sigma_components:
        print(f"✓ Sigma components loaded for windows: {list(clnf_with_sigma.ccnf.sigma_components.keys())}")
    else:
        print("✗ ERROR: Sigma components NOT loaded!")

    landmarks_with_sigma, info_with_sigma = clnf_with_sigma.fit(
        gray, face_bbox, return_params=True
    )

    print(f"\nResults WITH Sigma:")
    print(f"  Converged: {info_with_sigma['converged']}")
    print(f"  Iterations: {info_with_sigma['iterations']}")
    print(f"  Final update: {info_with_sigma['final_update']:.6f}")
    print(f"  Landmark variance: {np.std(landmarks_with_sigma):.2f}")

    # Test 2: WITHOUT sigma transformation (remove sigma components)
    print("\n" + "=" * 70)
    print("Test 2: CLNF WITHOUT Sigma Transformation (No Spatial Correlation)")
    print("=" * 70)

    clnf_without_sigma = CLNF(
        model_dir="pyclnf/models",
        max_iterations=20,
        regularization=25.0,
        sigma=1.5,
        weight_multiplier=5.0
    )

    # Manually disable sigma components
    clnf_without_sigma.ccnf.sigma_components = None
    print("✓ Sigma components disabled for comparison")

    landmarks_without_sigma, info_without_sigma = clnf_without_sigma.fit(
        gray, face_bbox, return_params=True
    )

    print(f"\nResults WITHOUT Sigma:")
    print(f"  Converged: {info_without_sigma['converged']}")
    print(f"  Iterations: {info_without_sigma['iterations']}")
    print(f"  Final update: {info_without_sigma['final_update']:.6f}")
    print(f"  Landmark variance: {np.std(landmarks_without_sigma):.2f}")

    # Compare results
    print("\n" + "=" * 70)
    print("Comparison Analysis")
    print("=" * 70)

    landmark_diff = np.linalg.norm(landmarks_with_sigma - landmarks_without_sigma, axis=1).mean()

    print(f"\nConvergence:")
    print(f"  WITH sigma:    {info_with_sigma['converged']} ({info_with_sigma['iterations']} iters)")
    print(f"  WITHOUT sigma: {info_without_sigma['converged']} ({info_without_sigma['iterations']} iters)")

    if info_with_sigma['converged'] and not info_without_sigma['converged']:
        print("  ✓ Sigma transformation IMPROVED convergence!")
    elif info_with_sigma['converged'] == info_without_sigma['converged']:
        print("  ~ Both achieved same convergence state")
    else:
        print("  ✗ Sigma transformation did NOT improve convergence")

    print(f"\nFinal Update Magnitude:")
    print(f"  WITH sigma:    {info_with_sigma['final_update']:.6f}")
    print(f"  WITHOUT sigma: {info_without_sigma['final_update']:.6f}")
    improvement = ((info_without_sigma['final_update'] - info_with_sigma['final_update'])
                   / info_without_sigma['final_update'] * 100)
    print(f"  Improvement: {improvement:+.1f}%")

    print(f"\nLandmark Difference:")
    print(f"  Mean position difference: {landmark_diff:.2f} pixels")

    # Visualize results
    print("\n" + "=" * 70)
    print("Generating Visualization")
    print("=" * 70)

    vis = create_comparison_visualization(
        gray,
        landmarks_with_sigma,
        landmarks_without_sigma,
        face_bbox,
        info_with_sigma,
        info_without_sigma
    )

    output_path = "test_output/sigma_comparison.jpg"
    Path("test_output").mkdir(exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"✓ Saved visualization to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if info_with_sigma['final_update'] < info_without_sigma['final_update']:
        print("✓ Sigma transformation is WORKING - reduces final update magnitude")
    else:
        print("✗ Sigma transformation may need tuning - no improvement observed")

    print("\nSigma transformation models spatial correlations in patch response maps,")
    print("which should improve robustness and convergence stability.")


def create_synthetic_face():
    """Create a synthetic face image for testing."""
    image = np.random.randint(50, 150, (480, 640), dtype=np.uint8)

    # Add facial features
    cx, cy = 320, 240

    # Eyes
    cv2.circle(image, (cx - 60, cy - 40), 20, 220, -1)
    cv2.circle(image, (cx + 60, cy - 40), 20, 220, -1)
    cv2.circle(image, (cx - 60, cy - 40), 8, 50, -1)
    cv2.circle(image, (cx + 60, cy - 40), 8, 50, -1)

    # Nose
    pts = np.array([[cx, cy], [cx - 15, cy + 30], [cx + 15, cy + 30]], dtype=np.int32)
    cv2.fillPoly(image, [pts], 200)

    # Mouth
    cv2.ellipse(image, (cx, cy + 60), (50, 25), 0, 0, 180, 220, -1)

    # Face outline
    cv2.ellipse(image, (cx, cy), (120, 150), 0, 0, 360, 200, 3)

    return image


def create_comparison_visualization(gray, lm_with, lm_without, bbox, info_with, info_without):
    """Create side-by-side comparison visualization."""

    # Convert to color
    vis_with = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    vis_without = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

    # Draw bounding box
    x, y, w, h = bbox
    cv2.rectangle(vis_with, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
    cv2.rectangle(vis_without, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

    # Draw landmarks
    for i, (lm_x, lm_y) in enumerate(lm_with):
        cv2.circle(vis_with, (int(lm_x), int(lm_y)), 2, (0, 255, 0), -1)

    for i, (lm_x, lm_y) in enumerate(lm_without):
        cv2.circle(vis_without, (int(lm_x), int(lm_y)), 2, (0, 0, 255), -1)

    # Add text labels
    cv2.putText(vis_with, "WITH Sigma Transform", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_with, f"Converged: {info_with['converged']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_with, f"Iters: {info_with['iterations']}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_with, f"Update: {info_with['final_update']:.4f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(vis_without, "WITHOUT Sigma Transform", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_without, f"Converged: {info_without['converged']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis_without, f"Iters: {info_without['iterations']}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis_without, f"Update: {info_without['final_update']:.4f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Concatenate side by side
    vis = np.hstack([vis_with, vis_without])

    return vis


if __name__ == "__main__":
    test_sigma_impact()
