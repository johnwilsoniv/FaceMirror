#!/usr/bin/env python3
"""
Analyze KDE accumulation differences between C++ and Python.
"""
import numpy as np
import cv2
from pyclnf.clnf import CLNF

def analyze_kde_accumulation():
    """Compare KDE accumulation values between C++ and Python."""

    print("="*80)
    print("KDE ACCUMULATION ANALYSIS")
    print("="*80)

    # Test parameters
    image_path = "calibration_frames/patient1_frame1.jpg"
    cpp_bbox = (301.938, 782.149, 400.586, 400.585)

    # Initialize CLNF with explicit debug mode
    clnf = CLNF(
        model_dir="pyclnf/models",
        regularization=35,
        max_iterations=1,  # Only one iteration for analysis
        convergence_threshold=0.005,
        sigma=1.5,
        weight_multiplier=0.0,
        window_sizes=[11],  # Only first window size
        detector=None,
        debug_mode=True
    )

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    print(f"\nTest setup:")
    print(f"  Image: {image_path}")
    print(f"  Bbox: {cpp_bbox}")
    print(f"  Sigma: 1.5")
    print(f"  a_kde = -0.5 / (1.5 * 1.5) = {-0.5 / (1.5 * 1.5):.8f}")

    # Run one iteration
    landmarks, info = clnf.fit(image, face_bbox=cpp_bbox)

    print("\n" + "="*80)
    print("EXPECTED C++ VALUES (from debug output):")
    print("-"*80)
    print("Landmark 36 mean-shift computation:")
    print("  dx=5.50, dy=5.50 (center of 11x11 window)")
    print("  KDE grid index: 5550 (55*101 + 55)")
    print("  Accumulation: mx=18.42, my=24.87, sum=4.677")
    print("  Mean-shift result: ms_x=1.438, ms_y=0.823")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("-"*80)
    print("""
The 30% accumulation difference (C++ sum=4.677 vs Python sum=3.338) suggests:

1. Response map values might be scaled differently
2. KDE weights might still have subtle differences
3. Numerical precision differences in exp() computation

Next steps:
1. Compare raw response map values element-by-element
2. Verify KDE grid computation matches exactly
3. Check if response map normalization differs
    """)

if __name__ == "__main__":
    analyze_kde_accumulation()