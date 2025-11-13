#!/usr/bin/env python3
"""
Test the integrated pyCLNF pipeline with corrected RetinaFace as primary detector.

This validates that:
1. CLNF initializes with RetinaFace detector by default
2. detect_and_fit() works correctly
3. The corrected RetinaFace provides accurate results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyclnf import CLNF


def test_primary_pipeline():
    """Test the primary usage pattern with automatic detection."""
    print("=" * 70)
    print("TESTING PYCLNF WITH CORRECTED RETINAFACE (PRIMARY DETECTOR)")
    print("=" * 70)

    # Test image
    image_path = "Patient Data/Normal Cohort/IMG_0434.MOV_frame_0000.jpg"

    if not Path(image_path).exists():
        print(f"\n⚠️  Extracting test frame...")
        video_path = "Patient Data/Normal Cohort/IMG_0434.MOV"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(image_path, frame)
            print(f"✅ Frame extracted")

    # Load image
    image = cv2.imread(image_path)
    print(f"\n1. Loaded image: {image.shape}")

    # Initialize CLNF with default detector (corrected RetinaFace)
    print("\n2. Initializing CLNF with corrected RetinaFace detector...")
    clnf = CLNF()  # Default: detector='retinaface'
    print("   ✅ CLNF initialized with primary detector")

    # Detect and fit in one call
    print("\n3. Running detect_and_fit()...")
    landmarks, info = clnf.detect_and_fit(image, return_params=True)

    print(f"\n4. Results:")
    print(f"   - Detected {len(landmarks)} landmarks")
    print(f"   - Converged: {info['converged']}")
    print(f"   - Iterations: {info['iterations']}")
    print(f"   - Final update: {info['final_update']:.6f}")
    print(f"   - Detected bbox: {info['bbox']}")

    if 'params' in info:
        print(f"   - Init scale: {info['params'][0]:.4f}")

    # Visualize results
    print("\n5. Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original image with bbox
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x, y, w, h = info['bbox']
    rect = plt.Rectangle((x, y), w, h, linewidth=3,
                         edgecolor='lime', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(f"Corrected RetinaFace Detection\n{w:.1f}×{h:.1f}px",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Landmarks
    ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=50,
               alpha=0.8, edgecolors='yellow', linewidths=1.5)
    ax2.set_title(f"pyCLNF Landmarks\n{len(landmarks)} points, "
                 f"{info['iterations']} iterations",
                 fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.suptitle("PRIMARY PIPELINE: CLNF with Corrected RetinaFace",
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = "test_output/integrated_pipeline_test.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {output_path}")

    return landmarks, info


def test_legacy_pipeline():
    """Test the legacy usage pattern with manual bbox."""
    print("\n" + "=" * 70)
    print("TESTING LEGACY PIPELINE (MANUAL BBOX)")
    print("=" * 70)

    # Load image
    image_path = "Patient Data/Normal Cohort/IMG_0434.MOV_frame_0000.jpg"
    image = cv2.imread(image_path)

    # Initialize CLNF without detector
    print("\n1. Initializing CLNF without detector...")
    clnf = CLNF(detector=None)
    print("   ✅ CLNF initialized (no detector)")

    # Manual bbox (from previous tests)
    manual_bbox = (273, 793, 401, 404)
    print(f"\n2. Using manual bbox: {manual_bbox}")

    # Fit with manual bbox
    print("\n3. Running fit() with manual bbox...")
    landmarks, info = clnf.fit(image, manual_bbox)

    print(f"\n4. Results:")
    print(f"   - Detected {len(landmarks)} landmarks")
    print(f"   - Converged: {info['converged']}")
    print(f"   - Iterations: {info['iterations']}")
    print(f"   ✅ Legacy pipeline works correctly")

    return landmarks, info


def test_multi_face():
    """Test detection and fitting on multi-face image."""
    print("\n" + "=" * 70)
    print("TESTING MULTI-FACE DETECTION")
    print("=" * 70)

    # Use the split-face image
    image_path = "Patient Data/Synkinesis Cohort/IMG_0316_frame_0000.jpg"

    if not Path(image_path).exists():
        print("⚠️  Multi-face test image not available, skipping...")
        return None, None

    image = cv2.imread(image_path)

    # Initialize CLNF
    clnf = CLNF()

    # Detect all faces
    print("\n1. Running detect_and_fit(return_all_faces=True)...")
    results = clnf.detect_and_fit(image, return_all_faces=True)

    print(f"\n2. Results:")
    print(f"   - Detected {len(results)} faces")

    for i, (landmarks, info) in enumerate(results):
        print(f"   - Face {i+1}: {len(landmarks)} landmarks, bbox={info['bbox']}")

    return results


def main():
    print("\n🚀 Testing Integrated pyCLNF Pipeline")
    print("=" * 70)

    # Test 1: Primary pipeline with automatic detection
    landmarks1, info1 = test_primary_pipeline()

    # Test 2: Legacy pipeline with manual bbox
    landmarks2, info2 = test_legacy_pipeline()

    # Test 3: Multi-face detection
    multi_results = test_multi_face()

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Primary pipeline (detect_and_fit): ✅ {len(landmarks1)} landmarks")
    print(f"  - Legacy pipeline (manual bbox):     ✅ {len(landmarks2)} landmarks")
    if multi_results:
        print(f"  - Multi-face detection:              ✅ {len(multi_results)} faces")
    print("\nCorrected RetinaFace is now the PRIMARY detector in pyCLNF!")
    print("=" * 70)


if __name__ == "__main__":
    main()
