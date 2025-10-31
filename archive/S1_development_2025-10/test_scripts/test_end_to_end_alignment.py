#!/usr/bin/env python3
"""
End-to-End Test: Raw Frame → Python Alignment → pyfhog → HOG Features

This is the critical test: does our Python face alignment produce HOG features
that match OpenFace C++? If yes, then the alignment is "good enough" even if
not pixel-perfect.

Test: Raw video frame + landmarks + pose → align_face() → pyfhog → compare with C++ .hog
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner
from openface22_hog_parser import OF22HOGParser
from scipy.stats import pearsonr

try:
    import pyfhog
    PYFHOG_AVAILABLE = True
except ImportError:
    PYFHOG_AVAILABLE = False
    print("⚠ Warning: pyfhog not installed. Install with: pip install pyfhog")


def main():
    if not PYFHOG_AVAILABLE:
        print("\n❌ Cannot run end-to-end test without pyfhog")
        print("Please install: pip install pyfhog")
        return

    print("=" * 70)
    print("End-to-End Alignment + FHOG Test")
    print("=" * 70)

    # Paths
    pdm_file = "In-the-wild_aligned_PDM_68.txt"
    csv_file = "of22_validation/IMG_0942_left_mirrored.csv"
    video_file = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
    cpp_hog_file = "pyfhog_validation_output/IMG_0942_left_mirrored.hog"

    # Initialize
    print("\n[1/6] Initializing...")
    aligner = OpenFace22FaceAligner(pdm_file)
    hog_parser = OF22HOGParser(cpp_hog_file)
    frame_indices, all_hog_features = hog_parser.parse()
    print(f"✓ Loaded {len(frame_indices)} frames from C++ .hog file")

    # Load validation data (test evenly-spaced frames like validation script)
    print("\n[2/6] Loading validation data...")
    df = pd.read_csv(csv_file)
    num_test_frames = 10
    # Select evenly spaced frames
    total_frames = len(df)
    frame_indices_to_test = np.linspace(0, total_frames - 1, num_test_frames, dtype=int)
    test_frames = df.iloc[frame_indices_to_test]
    print(f"✓ Testing {num_test_frames} evenly-spaced frames")

    # Open video
    print("\n[3/6] Opening video...")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")
    print(f"✓ Video opened")

    # Process frames
    print("\n[4/6] Processing frames through Python pipeline...")
    print("  (Raw frame → align → pyfhog → 4464-dim HOG features)")
    print("-" * 70)

    results = []
    for idx, row in test_frames.iterrows():
        frame_num = int(row['frame'])

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ Warning: Could not read frame {frame_num}")
            continue

        # Extract landmarks and pose
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        x = row[x_cols].values.astype(np.float32)
        y = row[y_cols].values.astype(np.float32)
        landmarks_68 = np.stack([x, y], axis=1)
        pose_tx = row['p_tx']
        pose_ty = row['p_ty']

        # Python alignment
        aligned_face_bgr = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty)

        # Convert BGR → RGB for pyfhog
        aligned_face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)

        # Extract HOG features with pyfhog
        python_hog = pyfhog.extract_fhog_features(aligned_face_rgb, cell_size=8)

        # Load C++ HOG features
        # OpenFace writes all frame indices as 1.0, so use sequential index
        # frame_num is 1-indexed, array is 0-indexed
        cpp_frame_idx = frame_num - 1

        if cpp_frame_idx >= len(all_hog_features):
            print(f"⚠ Warning: Frame {frame_num} out of range (have {len(all_hog_features)} frames)")
            continue

        cpp_hog = all_hog_features[cpp_frame_idx]

        # Compare
        if python_hog.shape != cpp_hog.shape:
            print(f"✗ Frame {frame_num}: Shape mismatch! Python={python_hog.shape}, C++={cpp_hog.shape}")
            continue

        mse = np.mean((python_hog - cpp_hog) ** 2)
        corr, _ = pearsonr(python_hog, cpp_hog)
        max_diff = np.max(np.abs(python_hog - cpp_hog))

        status = "✓" if corr > 0.99 else "✗"
        print(f"{status} Frame {frame_num:4d}: MSE={mse:10.6f}, r={corr:.6f}, max_diff={max_diff:.4f}")

        results.append({
            'frame': frame_num,
            'mse': mse,
            'correlation': corr,
            'max_diff': max_diff
        })

    cap.release()

    # Summary
    print("\n[5/6] Results Summary")
    print("=" * 70)

    if len(results) == 0:
        print("⚠ No frames processed!")
        return

    results_df = pd.DataFrame(results)

    print(f"Frames processed: {len(results_df)}")
    print(f"\nMSE Statistics:")
    print(f"  Mean:   {results_df['mse'].mean():.6f}")
    print(f"  Median: {results_df['mse'].median():.6f}")
    print(f"  Min:    {results_df['mse'].min():.6f}")
    print(f"  Max:    {results_df['mse'].max():.6f}")

    print(f"\nCorrelation Statistics:")
    print(f"  Mean:   {results_df['correlation'].mean():.6f}")
    print(f"  Median: {results_df['correlation'].median():.6f}")
    print(f"  Min:    {results_df['correlation'].min():.6f}")
    print(f"  Max:    {results_df['correlation'].max():.6f}")

    print(f"\nMax Difference Statistics:")
    print(f"  Mean:   {results_df['max_diff'].mean():.4f}")
    print(f"  Max:    {results_df['max_diff'].max():.4f}")

    # Success criteria
    print("\n[6/6] Final Assessment")
    print("=" * 70)

    mean_corr = results_df['correlation'].mean()

    if mean_corr > 0.999:
        print("🎉 EXCELLENT: HOG features match nearly perfectly! (r > 0.999)")
        print("   Python alignment is production-ready!")
    elif mean_corr > 0.99:
        print("✓ GOOD: HOG features match very well (r > 0.99)")
        print("   Python alignment should work for AU prediction")
    elif mean_corr > 0.95:
        print("⚠ ACCEPTABLE: HOG features match reasonably (r > 0.95)")
        print("   Python alignment may work, but could be improved")
    else:
        print("✗ POOR: HOG features don't match well (r < 0.95)")
        print("   Alignment needs more work before production use")

    print("=" * 70)


if __name__ == "__main__":
    main()
