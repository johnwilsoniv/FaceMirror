#!/usr/bin/env python3
"""
Simplified comparison: C++ OpenFace 2.2 vs PyfaceLM wrapper

This validates that PyfaceLM accurately wraps the C++ OpenFace binary.
Once validated, we can integrate PyfaceLM into S1 pipeline for AU extraction.
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import subprocess
import tempfile
import time

# Add PyfaceLM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "PyfaceLM"))

from pyfacelm import CLNFDetector, visualize_landmarks


def run_cpp_openface(image_path, binary_path):
    """Run C++ OpenFace binary directly"""
    print(f"\n[1] Running C++ OpenFace 2.2...")
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cmd = [
            str(binary_path),
            "-f", str(image_path),
            "-out_dir", str(tmpdir),
            "-2Dfp",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"  ✗ Failed: {result.stderr[:200]}")
            return None, None, None

        # Parse CSV
        csv_file = tmpdir / f"{Path(image_path).stem}.csv"
        landmarks, confidence = parse_openface_csv(csv_file)

    elapsed = time.time() - start
    print(f"  ✓ Success: {landmarks.shape}, conf={confidence:.3f}, time={elapsed:.3f}s")

    return landmarks, confidence, elapsed


def run_pyfacelm(image_path, detector):
    """Run PyfaceLM wrapper"""
    print(f"\n[2] Running PyfaceLM wrapper...")
    start = time.time()

    landmarks, confidence, bbox = detector.detect(image_path)

    elapsed = time.time() - start
    print(f"  ✓ Success: {landmarks.shape}, conf={confidence:.3f}, time={elapsed:.3f}s")

    return landmarks, confidence, elapsed


def parse_openface_csv(csv_file):
    """Parse OpenFace CSV output"""
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    values = lines[1].strip().split(',')

    # Extract 68 landmarks
    landmarks = []
    for i in range(68):
        try:
            x_idx = header.index(f'x_{i}')
            y_idx = header.index(f'y_{i}')
        except ValueError:
            x_idx = header.index(f' x_{i}')
            y_idx = header.index(f' y_{i}')

        x = float(values[x_idx])
        y = float(values[y_idx])
        landmarks.append([x, y])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Extract confidence
    try:
        conf_idx = header.index('confidence')
    except ValueError:
        conf_idx = header.index(' confidence')

    confidence = float(values[conf_idx])

    return landmarks, confidence


def compare_landmarks(lm1, lm2, name1="C++ OpenFace", name2="PyfaceLM"):
    """Compare two sets of landmarks"""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {name2} vs {name1} (gold standard)")
    print(f"{'='*70}")

    if lm1 is None or lm2 is None:
        print("  ✗ Cannot compare (missing landmarks)")
        return

    error = np.abs(lm2 - lm1)
    mean_error = error.mean()
    max_error = error.max()
    rmse = np.sqrt((error ** 2).mean())

    print(f"\n  Mean absolute error: {mean_error:.6f} px")
    print(f"  Max absolute error:  {max_error:.6f} px")
    print(f"  RMSE:                {rmse:.6f} px")

    if mean_error < 0.01:
        print(f"\n  ✓ PERFECT MATCH! ({mean_error:.8f}px error)")
    elif mean_error < 1.0:
        print(f"\n  ✓ Excellent match ({mean_error:.3f}px error)")
    else:
        print(f"\n  ✗ Significant difference ({mean_error:.3f}px error)")

    return {
        "mean": float(mean_error),
        "max": float(max_error),
        "rmse": float(rmse)
    }


def create_comparison_vis(image_path, lm1, lm2, conf1, conf2, output_path):
    """Create side-by-side visualization"""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    # Draw C++ OpenFace landmarks (green)
    img1 = img.copy()
    for x, y in lm1:
        cv2.circle(img1, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.putText(img1, "C++ OpenFace 2.2 (Gold Standard)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img1, f"Confidence: {conf1:.3f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw PyfaceLM landmarks (blue)
    img2 = img.copy()
    for x, y in lm2:
        cv2.circle(img2, (int(x), int(y)), 2, (255, 0, 0), -1)
    cv2.putText(img2, "PyfaceLM Wrapper", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img2, f"Confidence: {conf2:.3f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show difference
    error = np.abs(lm2 - lm1)
    mean_error = error.mean()
    cv2.putText(img2, f"Error vs Gold: {mean_error:.4f}px", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Combine
    comparison = np.hstack([img1, img2])

    cv2.imwrite(str(output_path), comparison)
    print(f"\n✓ Visualization saved: {output_path}")


def main():
    """Run comparison on test images"""
    print("="*70)
    print("C++ OPENFACE 2.2 vs PYFACELM WRAPPER COMPARISON")
    print("="*70)

    # Setup
    binary_path = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")
    test_images = [
        "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg",
        "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_9330.jpg",
    ]
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/test_output")
    output_dir.mkdir(exist_ok=True)

    # Initialize PyfaceLM
    print("\nInitializing PyfaceLM...")
    detector = CLNFDetector()

    # Process each image
    all_results = []

    for image_path in test_images:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"\n✗ Skipping {image_path.name} (not found)")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {image_path.name}")
        print(f"{'='*70}")

        # Run both pipelines
        lm1, conf1, time1 = run_cpp_openface(image_path, binary_path)
        lm2, conf2, time2 = run_pyfacelm(image_path, detector)

        # Compare
        if lm1 is not None and lm2 is not None:
            error_stats = compare_landmarks(lm1, lm2)

            # Visualize
            output_path = output_dir / f"{image_path.stem}_cpp_vs_pyfacelm.jpg"
            create_comparison_vis(image_path, lm1, lm2, conf1, conf2, output_path)

            all_results.append({
                "image": image_path.name,
                "error": error_stats,
                "time_cpp": time1,
                "time_pyfacelm": time2
            })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for result in all_results:
        print(f"\n{result['image']}:")
        print(f"  Error:        {result['error']['mean']:.6f} px (mean)")
        print(f"  Time (C++):   {result['time_cpp']:.3f}s")
        print(f"  Time (Wrap):  {result['time_pyfacelm']:.3f}s")

    if all_results:
        avg_error = np.mean([r['error']['mean'] for r in all_results])
        print(f"\nAverage error across all images: {avg_error:.6f} px")

        if avg_error < 0.01:
            print("\n✓ VALIDATION SUCCESSFUL: PyfaceLM wrapper is identical to C++ OpenFace!")
            print("  Ready to integrate into S1 pipeline for AU extraction.")
        else:
            print(f"\n✗ Validation failed: {avg_error:.3f}px error detected")

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
