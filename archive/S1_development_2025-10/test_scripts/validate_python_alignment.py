#!/usr/bin/env python3
"""
Validate Python Face Alignment Against OpenFace C++

This script compares the Python implementation of OpenFace 2.2 face alignment
with the C++ reference implementation to ensure pixel-level accuracy.

Test data:
- OpenFace C++ aligned faces: pyfhog_validation_output/IMG_0942_left_mirrored_aligned/
- Original video: /Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4
- Landmarks + pose: of22_validation/IMG_0942_left_mirrored.csv

Success criteria:
- Target: MSE < 1.0, correlation r > 0.99 (near pixel-perfect)
- Acceptable: MSE < 5.0, correlation r > 0.95
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner
from triangulation_parser import TriangulationParser
from scipy.stats import pearsonr


def load_validation_data(csv_path: str, num_frames: int = 10):
    """
    Load landmarks and pose parameters from OpenFace CSV

    Args:
        csv_path: Path to OpenFace CSV file
        num_frames: Number of frames to validate (default: 10)

    Returns:
        DataFrame with frame, landmarks, and pose data
    """
    df = pd.read_csv(csv_path)

    # Select evenly spaced frames for validation
    total_frames = len(df)
    if num_frames >= total_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    df_sample = df.iloc[frame_indices].copy()

    print(f"Loaded validation data: {len(df_sample)} frames")
    print(f"  Frame range: {df_sample['frame'].min()} - {df_sample['frame'].max()}")

    return df_sample


def extract_landmarks_and_pose(row):
    """
    Extract 68 landmarks and pose from CSV row

    Args:
        row: DataFrame row

    Returns:
        Tuple of (landmarks_68, pose_tx, pose_ty)
        - landmarks_68: (68, 2) array
        - pose_tx, pose_ty: float
    """
    # Extract x and y coordinates
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)

    landmarks_68 = np.stack([x, y], axis=1)  # (68, 2)

    # Extract pose translation
    # Use PDM parameters (p_tx, p_ty), not world space pose (pose_Tx, pose_Ty)
    # params_global in C++ is [scale, rx, ry, rz, tx, ty] from PDM
    pose_tx = row['p_tx']
    pose_ty = row['p_ty']

    return landmarks_68, pose_tx, pose_ty


def compute_metrics(img1, img2):
    """
    Compute comparison metrics between two images

    Args:
        img1: First image
        img2: Second image

    Returns:
        Dictionary with MSE and correlation metrics
    """
    # Convert to float for accurate computation
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Mean Squared Error
    mse = np.mean((img1_float - img2_float) ** 2)

    # Pixel-wise correlation
    pixels1 = img1_float.flatten()
    pixels2 = img2_float.flatten()
    correlation, _ = pearsonr(pixels1, pixels2)

    # Per-channel metrics
    metrics = {
        'mse': mse,
        'correlation': correlation,
        'rmse': np.sqrt(mse),
        'max_diff': np.max(np.abs(img1_float - img2_float))
    }

    return metrics


def save_comparison(python_aligned, cpp_aligned, frame_num, output_dir):
    """
    Save side-by-side comparison images

    Args:
        python_aligned: Python aligned face
        cpp_aligned: C++ aligned face
        frame_num: Frame number
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create difference image (amplified for visibility)
    diff = np.abs(python_aligned.astype(np.float32) - cpp_aligned.astype(np.float32))
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

    # Create side-by-side comparison
    comparison = np.hstack([
        python_aligned,
        cpp_aligned,
        diff_amplified
    ])

    # Add labels
    comparison_labeled = comparison.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_labeled, "Python", (10, 20), font, 0.5, (0, 255, 0), 1)
    cv2.putText(comparison_labeled, "C++", (122, 20), font, 0.5, (0, 255, 0), 1)
    cv2.putText(comparison_labeled, "Diff x10", (234, 20), font, 0.5, (0, 255, 0), 1)

    # Save
    output_path = output_dir / f"frame_{frame_num:04d}_comparison.png"
    cv2.imwrite(str(output_path), comparison_labeled)

    return output_path


def main():
    """Main validation routine"""
    print("=" * 70)
    print("OpenFace 2.2 Python Face Alignment Validation")
    print("=" * 70)

    # Paths
    pdm_file = "In-the-wild_aligned_PDM_68.txt"
    csv_file = "of22_validation/IMG_0942_left_mirrored.csv"
    video_file = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
    cpp_aligned_dir = "pyfhog_validation_output/IMG_0942_left_mirrored_aligned"
    output_dir = "alignment_validation_output"

    # Initialize aligner and triangulation
    print("\n[1/5] Initializing face aligner...")
    aligner = OpenFace22FaceAligner(pdm_file, sim_scale=0.7, output_size=(112, 112))

    tris_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/tris_68.txt"
    triangulation = TriangulationParser(tris_file)

    # Load validation data
    print("\n[2/5] Loading validation data...")
    df = load_validation_data(csv_file, num_frames=10)

    # Open video
    print("\n[3/5] Opening video...")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")
    print(f"  Video opened: {video_file}")

    # Validate frames
    print("\n[4/5] Validating alignment on frames...")
    print("-" * 70)

    results = []
    cpp_aligned_path = Path(cpp_aligned_dir)

    for idx, row in df.iterrows():
        frame_num = int(row['frame'])

        # Read frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # OpenFace uses 1-indexed frames
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ Warning: Could not read frame {frame_num}")
            continue

        # Extract landmarks and pose
        landmarks_68, pose_tx, pose_ty = extract_landmarks_and_pose(row)

        # Align with Python (no masking - OpenFace doesn't mask for HOG extraction)
        python_aligned = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty,
                                           apply_mask=False)

        # Load C++ aligned face
        # OpenFace uses format: frame_det_00_NNNNNN.bmp
        cpp_aligned_file = cpp_aligned_path / f"frame_det_00_{frame_num:06d}.bmp"
        if not cpp_aligned_file.exists():
            print(f"⚠ Warning: C++ aligned face not found: {cpp_aligned_file}")
            continue

        cpp_aligned = cv2.imread(str(cpp_aligned_file))

        # Compute metrics
        metrics = compute_metrics(python_aligned, cpp_aligned)

        # Save comparison
        comparison_path = save_comparison(python_aligned, cpp_aligned, frame_num, output_dir)

        # Print results
        status = "✓" if metrics['mse'] < 5.0 else "✗"
        print(f"{status} Frame {frame_num:4d}: MSE={metrics['mse']:8.4f}, r={metrics['correlation']:.6f}, max_diff={metrics['max_diff']:6.2f}")

        results.append({
            'frame': frame_num,
            **metrics,
            'comparison_path': comparison_path
        })

    cap.release()

    # Summary statistics
    print("\n[5/5] Validation Summary")
    print("-" * 70)

    if len(results) == 0:
        print("⚠ No frames validated!")
        return

    results_df = pd.DataFrame(results)

    print(f"Frames validated: {len(results_df)}")
    print(f"\nMSE Statistics:")
    print(f"  Mean:   {results_df['mse'].mean():.4f}")
    print(f"  Median: {results_df['mse'].median():.4f}")
    print(f"  Min:    {results_df['mse'].min():.4f}")
    print(f"  Max:    {results_df['mse'].max():.4f}")

    print(f"\nCorrelation Statistics:")
    print(f"  Mean:   {results_df['correlation'].mean():.6f}")
    print(f"  Median: {results_df['correlation'].median():.6f}")
    print(f"  Min:    {results_df['correlation'].min():.6f}")
    print(f"  Max:    {results_df['correlation'].max():.6f}")

    print(f"\nMax Difference Statistics:")
    print(f"  Mean:   {results_df['max_diff'].mean():.2f}")
    print(f"  Median: {results_df['max_diff'].median():.2f}")
    print(f"  Max:    {results_df['max_diff'].max():.2f}")

    # Success criteria
    print("\n" + "=" * 70)
    mean_mse = results_df['mse'].mean()
    mean_corr = results_df['correlation'].mean()

    if mean_mse < 1.0 and mean_corr > 0.99:
        print("🎉 SUCCESS: Near pixel-perfect alignment! (MSE < 1.0, r > 0.99)")
    elif mean_mse < 5.0 and mean_corr > 0.95:
        print("✓ SUCCESS: Acceptable alignment (MSE < 5.0, r > 0.95)")
    else:
        print("✗ NEEDS IMPROVEMENT: Alignment accuracy below target")

    print(f"\nComparison images saved to: {output_dir}/")
    print("=" * 70)

    # Save results to CSV
    results_csv = Path(output_dir) / "validation_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")


if __name__ == "__main__":
    main()
