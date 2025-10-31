#!/usr/bin/env python3
"""
Test our Python face alignment by running it through C++ AU prediction

Strategy:
1. Use Python to align faces (current inverse-CSV-p_rz approach)
2. Save aligned faces to temp directory
3. Run C++ FeatureExtraction on aligned faces to get AU predictions
4. Compare to baseline C++ AU predictions (from original video)

This tests if our alignment is "good enough" for the AU models.
"""

import numpy as np
import cv2
import pandas as pd
import subprocess
import tempfile
import shutil
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner
from triangulation_parser import TriangulationParser

# Configuration
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
OPENFACE_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
PDM_FILE = "In-the-wild_aligned_PDM_68.txt"
TRIS_FILE = "tris_68_full.txt"

def create_aligned_video_python(csv_path, video_path, output_path, use_mask=True):
    """Create video of Python-aligned faces"""
    print("Creating video of Python-aligned faces...")

    # Load components
    aligner = OpenFace22FaceAligner(PDM_FILE)
    if use_mask:
        triangulation = TriangulationParser(TRIS_FILE)
    else:
        triangulation = None

    # Load CSV and video
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (112, 112))

    print(f"Processing {frame_count} frames...")

    for frame_num in range(1, min(frame_count + 1, len(df) + 1)):
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}/{frame_count}")

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        if not ret:
            break

        # Get CSV data
        row = df[df['frame'] == frame_num]
        if len(row) == 0:
            # Frame not in CSV, write black frame
            black_frame = np.zeros((112, 112, 3), dtype=np.uint8)
            out.write(black_frame)
            continue

        row = row.iloc[0]

        # Get landmarks and pose
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        x = row[x_cols].values.astype(np.float32)
        y = row[y_cols].values.astype(np.float32)
        landmarks = np.stack([x, y], axis=1)

        pose_tx = row['p_tx']
        pose_ty = row['p_ty']
        p_rz = row['p_rz']

        # Align face
        try:
            if use_mask:
                aligned = aligner.align_face(frame, landmarks, pose_tx, pose_ty, p_rz,
                                            apply_mask=True, triangulation=triangulation)
            else:
                aligned = aligner.align_face(frame, landmarks, pose_tx, pose_ty, p_rz)

            out.write(aligned)
        except Exception as e:
            print(f"  Error on frame {frame_num}: {e}")
            black_frame = np.zeros((112, 112, 3), dtype=np.uint8)
            out.write(black_frame)

    cap.release()
    out.release()
    print(f"✓ Saved aligned video: {output_path}")

def run_openface_on_aligned_video(video_path, output_dir):
    """Run C++ OpenFace on pre-aligned video"""
    print(f"\nRunning C++ OpenFace on aligned video...")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")

    cmd = [
        OPENFACE_BINARY,
        "-f", video_path,
        "-out_dir", output_dir,
        "-2Dfp",  # Output 2D landmarks
        "-aus",   # Output AUs
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("✗ OpenFace failed!")
        print("STDERR:", result.stderr)
        return None

    # Find output CSV
    csv_files = list(Path(output_dir).glob("*.csv"))
    if not csv_files:
        print("✗ No CSV output found!")
        return None

    csv_path = csv_files[0]
    print(f"✓ OpenFace completed: {csv_path}")
    return csv_path

def compare_au_predictions(baseline_csv, test_csv):
    """Compare AU predictions between baseline and test"""
    print("\n" + "=" * 80)
    print("AU Prediction Comparison")
    print("=" * 80)

    df_baseline = pd.read_csv(baseline_csv)
    df_test = pd.read_csv(test_csv)

    # AU columns
    au_cols = [col for col in df_baseline.columns if col.startswith('AU') and col.endswith('_r')]

    print(f"\nComparing {len(au_cols)} AUs across {len(df_baseline)} frames\n")

    results = {}
    for au in au_cols:
        if au not in df_test.columns:
            print(f"  {au}: MISSING in test output")
            continue

        baseline_vals = df_baseline[au].values
        test_vals = df_test[au].values

        # Handle different lengths
        min_len = min(len(baseline_vals), len(test_vals))
        baseline_vals = baseline_vals[:min_len]
        test_vals = test_vals[:min_len]

        # Compute correlation
        if np.std(baseline_vals) > 0 and np.std(test_vals) > 0:
            correlation = np.corrcoef(baseline_vals, test_vals)[0, 1]
        else:
            correlation = 0.0

        # Compute RMSE
        rmse = np.sqrt(np.mean((baseline_vals - test_vals) ** 2))

        results[au] = {'correlation': correlation, 'rmse': rmse}

        print(f"  {au}: r={correlation:.4f}, RMSE={rmse:.4f}")

    # Overall statistics
    correlations = [r['correlation'] for r in results.values() if not np.isnan(r['correlation'])]
    mean_correlation = np.mean(correlations)
    min_correlation = np.min(correlations)
    max_correlation = np.max(correlations)

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"  Mean correlation: {mean_correlation:.4f}")
    print(f"  Min correlation:  {min_correlation:.4f}")
    print(f"  Max correlation:  {max_correlation:.4f}")
    print("\nInterpretation:")
    if mean_correlation > 0.95:
        print("  ✓ EXCELLENT - Python alignment is virtually identical to C++")
    elif mean_correlation > 0.90:
        print("  ✓ GOOD - Python alignment is very close to C++")
    elif mean_correlation > 0.80:
        print("  ~ ACCEPTABLE - Python alignment is reasonably close")
    else:
        print("  ✗ POOR - Python alignment needs improvement")
    print("=" * 80)

    return results

def main():
    print("=" * 80)
    print("Testing Python Alignment with C++ AU Prediction")
    print("=" * 80)

    # Create temp directory for aligned video and outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Create aligned video using Python
        aligned_video_path = temp_path / "python_aligned.mp4"
        create_aligned_video_python(CSV_PATH, VIDEO_PATH, str(aligned_video_path), use_mask=True)

        # Step 2: Run C++ OpenFace on Python-aligned video
        openface_output_dir = temp_path / "openface_output"
        openface_output_dir.mkdir()

        test_csv = run_openface_on_aligned_video(str(aligned_video_path), str(openface_output_dir))
        if test_csv is None:
            print("✗ Test failed - could not run OpenFace")
            return

        # Step 3: Compare AU predictions
        baseline_csv = CSV_PATH
        compare_au_predictions(baseline_csv, test_csv)

if __name__ == "__main__":
    main()
