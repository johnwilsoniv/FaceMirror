#!/usr/bin/env python3
"""
Three-Group AU Extraction Comparison on Video

Tests 500 frames with:
- Group 1 (Gold): C++ OpenFace 2.2 full pipeline (landmarks + AUs)
- Group 2 (Current): pyfaceau pipeline (C++ features + Python SVR)
- Group 3 (New): PyfaceLM landmarks + pyfaceau AU extraction

Priority: Accuracy (not speed)
Target: 500 frames at ~20 FPS
"""

import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import time
import subprocess
import tempfile
from typing import Tuple, Dict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "PyfaceLM"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pyfaceau"))

from pyfacelm import CLNFDetector


class ThreeGroupComparison:
    """Compare three AU extraction pipelines on video"""

    def __init__(self):
        """Initialize all components"""
        print("="*80)
        print("THREE-GROUP AU EXTRACTION COMPARISON")
        print("="*80)

        # Paths
        self.openface_binary = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")
        self.models_dir = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/AU_predictors")
        self.pdm_file = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/model/pdms/In-the-wild_aligned_PDM_68.txt")

        # Validate
        if not self.openface_binary.exists():
            raise FileNotFoundError(f"OpenFace binary not found: {self.openface_binary}")

        # Initialize PyfaceLM (Group 3)
        print("\nInitializing PyfaceLM...")
        self.pyfacelm = CLNFDetector()

        # Initialize pyfaceau AU predictor (Groups 2 & 3)
        print("\nInitializing pyfaceau AU predictor...")
        from pyfaceau.prediction.au_predictor import OpenFace22AUPredictor
        self.pyfaceau_au = OpenFace22AUPredictor(
            str(self.openface_binary),
            str(self.models_dir),
            str(self.pdm_file)
        )

        print("\n" + "="*80)

    def run_group1_cpp_full(self, video_path: Path, output_dir: Path) -> pd.DataFrame:
        """
        Group 1: C++ OpenFace 2.2 full pipeline

        Returns landmarks + AUs directly from C++ binary
        """
        print(f"\n{'='*80}")
        print("GROUP 1: C++ OpenFace 2.2 Full Pipeline (Gold Standard)")
        print(f"{'='*80}")

        start_time = time.time()

        # Run C++ OpenFace
        cmd = [
            str(self.openface_binary),
            "-f", str(video_path),
            "-out_dir", str(output_dir),
            "-2Dfp",  # 2D landmarks
            "-aus",   # Action Units
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return None

        # Load CSV
        csv_file = output_dir / f"{video_path.stem}.csv"
        if not csv_file.exists():
            print(f"ERROR: CSV not found: {csv_file}")
            return None

        df = pd.read_csv(csv_file)
        elapsed = time.time() - start_time
        fps = len(df) / elapsed

        print(f"\n✓ Success!")
        print(f"  Frames processed: {len(df)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Columns: {len(df.columns)}")

        # Show AU columns
        au_cols = [c for c in df.columns if c.startswith('AU') and ('_r' in c or '_c' in c)]
        print(f"  AUs extracted: {len(au_cols)//2} ({au_cols[:6]}...)")

        return df

    def run_group2_pyfaceau_full(self, video_path: Path) -> pd.DataFrame:
        """
        Group 2: pyfaceau full pipeline

        C++ feature extraction + Python SVR prediction
        """
        print(f"\n{'='*80}")
        print("GROUP 2: pyfaceau Pipeline (Current S1)")
        print(f"{'='*80}")

        start_time = time.time()

        # Use pyfaceau AU predictor
        print("\nRunning pyfaceau AU prediction...")
        df = self.pyfaceau_au.predict_video(str(video_path), verbose=True)

        elapsed = time.time() - start_time
        fps = len(df) / elapsed

        print(f"\n✓ Success!")
        print(f"  Frames processed: {len(df)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")

        return df

    def run_group3_pyfacelm_plus_pyfaceau(self, video_path: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Group 3: PyfaceLM landmarks + pyfaceau AU extraction

        This tests if we can use PyfaceLM for accurate landmarks,
        then feed those into pyfaceau's AU extraction pipeline.

        Challenge: pyfaceau AU predictor expects C++ OpenFace features (FHOG + PDM),
        not just landmarks. So we have two options:

        A) Use PyfaceLM landmarks, but still run C++ OpenFace for FHOG extraction
        B) Implement pure Python AU extraction from PyfaceLM landmarks

        For now, implementing Option A.
        """
        print(f"\n{'='*80}")
        print("GROUP 3: PyfaceLM + pyfaceau AU")
        print(f"{'='*80}")

        start_time = time.time()

        # Extract frames
        print("\nExtracting frames from video...")
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"  Extracted {frame_count} frames...")

        cap.release()
        print(f"  Total frames: {len(frames)}")

        # Get landmarks from PyfaceLM
        print("\nDetecting landmarks with PyfaceLM...")
        landmarks_list = []
        confidences = []

        for i, frame in enumerate(frames):
            # Save frame temporarily
            temp_frame = output_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(temp_frame), frame)

            # Detect
            landmarks, conf, bbox = self.pyfacelm.detect(str(temp_frame))
            landmarks_list.append(landmarks)
            confidences.append(conf)

            # Cleanup
            temp_frame.unlink()

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(frames)} frames...")

        landmarks_df = pd.DataFrame({
            'frame': range(len(landmarks_list)),
            'confidence': confidences
        })

        # Add landmark coordinates
        for pt in range(68):
            landmarks_df[f'x_{pt}'] = [lm[pt, 0] for lm in landmarks_list]
            landmarks_df[f'y_{pt}'] = [lm[pt, 1] for lm in landmarks_list]

        elapsed_landmarks = time.time() - start_time
        fps_landmarks = len(frames) / elapsed_landmarks

        print(f"\n✓ Landmarks complete!")
        print(f"  Frames: {len(landmarks_df)}")
        print(f"  Time: {elapsed_landmarks:.2f}s")
        print(f"  FPS: {fps_landmarks:.2f}")

        # Now run pyfaceau AU prediction
        # Note: This still uses C++ for FHOG extraction
        print("\nRunning pyfaceau AU extraction...")
        au_df = self.pyfaceau_au.predict_video(str(video_path), verbose=True)

        elapsed_total = time.time() - start_time
        fps_total = len(frames) / elapsed_total

        print(f"\n✓ Total pipeline complete!")
        print(f"  Total time: {elapsed_total:.2f}s")
        print(f"  Overall FPS: {fps_total:.2f}")

        return landmarks_df, au_df

    def compare_results(
        self,
        group1_df: pd.DataFrame,
        group2_df: pd.DataFrame,
        group3_landmarks: pd.DataFrame,
        group3_aus: pd.DataFrame,
        output_dir: Path
    ):
        """Compare all three groups"""
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")

        # Extract AU columns (intensity values)
        au_cols_intensity = [c for c in group1_df.columns if c.startswith('AU') and c.endswith('_r')]

        print(f"\nAUs to compare: {len(au_cols_intensity)}")
        print(f"  {au_cols_intensity}")

        # Compare Group 2 vs Group 1
        if group2_df is not None:
            print(f"\n{'-'*80}")
            print("Group 2 (pyfaceau) vs Group 1 (C++ Gold)")
            print(f"{'-'*80}")

            correlations = {}
            for au_col in au_cols_intensity:
                if au_col in group2_df.columns:
                    corr = np.corrcoef(
                        group1_df[au_col].values,
                        group2_df[au_col].values
                    )[0, 1]
                    correlations[au_col] = corr
                    print(f"  {au_col}: r = {corr:.4f}")

            avg_corr = np.mean(list(correlations.values()))
            print(f"\n  Average correlation: r = {avg_corr:.4f}")

        # Compare landmarks: Group 3 vs Group 1
        print(f"\n{'-'*80}")
        print("Group 3 (PyfaceLM) Landmarks vs Group 1 (C++ Gold)")
        print(f"{'-'*80}")

        # Extract landmarks from Group 1
        landmark_errors = []
        for i in range(min(len(group1_df), len(group3_landmarks))):
            lm1 = np.array([[group1_df.loc[i, f'x_{j}'], group1_df.loc[i, f'y_{j}']] for j in range(68)])
            lm3 = np.array([[group3_landmarks.loc[i, f'x_{j}'], group3_landmarks.loc[i, f'y_{j}']] for j in range(68)])

            error = np.abs(lm3 - lm1).mean()
            landmark_errors.append(error)

        mean_landmark_error = np.mean(landmark_errors)
        max_landmark_error = np.max(landmark_errors)

        print(f"  Mean landmark error: {mean_landmark_error:.6f} px")
        print(f"  Max landmark error:  {max_landmark_error:.6f} px")

        # Compare AUs: Group 3 vs Group 1
        if group3_aus is not None:
            print(f"\n{'-'*80}")
            print("Group 3 (PyfaceLM + pyfaceau AU) vs Group 1 (C++ Gold)")
            print(f"{'-'*80}")

            correlations_g3 = {}
            for au_col in au_cols_intensity:
                if au_col in group3_aus.columns:
                    corr = np.corrcoef(
                        group1_df[au_col].values[:len(group3_aus)],
                        group3_aus[au_col].values
                    )[0, 1]
                    correlations_g3[au_col] = corr
                    print(f"  {au_col}: r = {corr:.4f}")

            avg_corr_g3 = np.mean(list(correlations_g3.values()))
            print(f"\n  Average correlation: r = {avg_corr_g3:.4f}")

        # Save comparison report
        report_path = output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write("THREE-GROUP AU EXTRACTION COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Group 1 frames: {len(group1_df)}\n")
            f.write(f"Group 2 frames: {len(group2_df) if group2_df is not None else 0}\n")
            f.write(f"Group 3 frames: {len(group3_landmarks)}\n\n")
            f.write(f"Landmark error (Group 3 vs 1): {mean_landmark_error:.6f} px\n\n")
            if group2_df is not None:
                f.write(f"AU correlation (Group 2 vs 1): r = {avg_corr:.4f}\n")
            if group3_aus is not None:
                f.write(f"AU correlation (Group 3 vs 1): r = {avg_corr_g3:.4f}\n")

        print(f"\n✓ Comparison report saved: {report_path}")


def main():
    """Run three-group comparison on 500-frame video"""

    # Setup
    video_path = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/test_500_frames/IMG_0942_500frames.mov")
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/test_500_frames")
    output_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Initialize comparison
    comparison = ThreeGroupComparison()

    # Run Group 1 (C++ Gold Standard)
    group1_output = output_dir / "cpp_output"
    group1_output.mkdir(exist_ok=True)
    group1_df = comparison.run_group1_cpp_full(video_path, group1_output)

    if group1_df is None:
        print("ERROR: Group 1 failed, cannot continue")
        return

    # Run Group 2 (pyfaceau)
    group2_df = comparison.run_group2_pyfaceau_full(video_path)

    # Run Group 3 (PyfaceLM + pyfaceau)
    group3_output = output_dir / "group3_output"
    group3_output.mkdir(exist_ok=True)
    group3_landmarks, group3_aus = comparison.run_group3_pyfacelm_plus_pyfaceau(video_path, group3_output)

    # Compare
    comparison.compare_results(
        group1_df,
        group2_df,
        group3_landmarks,
        group3_aus,
        output_dir
    )

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
