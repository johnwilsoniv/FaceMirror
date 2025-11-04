#!/usr/bin/env python3
"""
Compare Three AU Extraction Pipelines

Group 1: S1 Current (pyfaceau full pipeline)
Group 2: C++ OpenFace 2.2 (if possible, else use pyfaceau AU predictor as gold standard)
Group 3: PyfaceLM landmarks → S1 AU extraction

Test video: IMG_0434.MOV (972 frames, ~60fps)
"""

import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import time
import subprocess
from typing import Tuple, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "PyfaceLM"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pyfaceau"))

from pyfacelm import CLNFDetector


class ThreeAUPipelineComparison:
    """Compare three AU extraction approaches"""

    def __init__(self):
        """Initialize all components"""
        print("="*80)
        print("THREE AU PIPELINE COMPARISON")
        print("="*80)

        # Paths
        self.openface_binary = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")
        self.models_dir = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/AU_predictors")
        self.pdm_file = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/model/pdms/In-the-wild_aligned_PDM_68.txt")

        # Validate
        if not self.openface_binary.exists():
            raise FileNotFoundError(f"OpenFace binary not found: {self.openface_binary}")

        print(f"\n✓ OpenFace binary found")
        print(f"✓ AU models dir: {self.models_dir}")
        print(f"✓ PDM file: {self.pdm_file.name}")

    def run_group1_s1_current(self, video_path: Path, output_dir: Path) -> pd.DataFrame:
        """
        Group 1: S1 Current Pipeline (pyfaceau)

        Uses pyfaceau.OpenFaceProcessor to extract AUs
        """
        print(f"\n{'='*80}")
        print("GROUP 1: S1 Current Pipeline (pyfaceau)")
        print(f"{'='*80}")

        start_time = time.time()

        # Import pyfaceau processor
        from pyfaceau import OpenFaceProcessor

        # Initialize processor
        print("\nInitializing pyfaceau OpenFaceProcessor...")
        processor = OpenFaceProcessor(
            weights_dir=str(Path(__file__).parent.parent / "pyfaceau/weights"),
            verbose=True
        )

        # Process video
        print(f"\nProcessing video: {video_path.name}")
        output_csv = output_dir / "group1_output.csv"
        frame_count = processor.process_video(
            str(video_path),
            str(output_csv)
        )

        elapsed = time.time() - start_time

        # Load results
        df = pd.read_csv(output_dir / "group1_output.csv")
        fps = len(df) / elapsed

        print(f"\n✓ Group 1 Complete!")
        print(f"  Frames: {len(df)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")

        # Show AU columns
        au_cols = [c for c in df.columns if c.startswith('AU')]
        print(f"  AUs extracted: {len(au_cols)} ({au_cols[:5]}...)")

        return df

    def run_group2_cpp_openface(self, video_path: Path, output_dir: Path) -> Optional[pd.DataFrame]:
        """
        Group 2: C++ OpenFace 2.2 (Gold Standard)

        Try direct C++ AU extraction, fallback to pyfaceau AU predictor if crashes
        """
        print(f"\n{'='*80}")
        print("GROUP 2: C++ OpenFace 2.2 (Gold Standard)")
        print(f"{'='*80}")

        start_time = time.time()

        # Try C++ OpenFace with AU extraction
        print("\nAttempting direct C++ OpenFace AU extraction...")

        cmd = [
            str(self.openface_binary),
            "-f", str(video_path),
            "-out_dir", str(output_dir),
            "-2Dfp",
            "-aus"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Success!
            csv_file = output_dir / f"{video_path.stem}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                elapsed = time.time() - start_time
                fps = len(df) / elapsed

                print(f"\n✓ Group 2 Complete (C++ direct)!")
                print(f"  Frames: {len(df)}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  FPS: {fps:.2f}")

                return df

        # C++ failed, use pyfaceau AU predictor as fallback
        print(f"\n⚠️  C++ OpenFace AU extraction failed (exit code {result.returncode})")
        print(f"  Error: {result.stderr[:200]}")
        print(f"\n  Falling back to pyfaceau AU predictor...")

        from pyfaceau.prediction.au_predictor import OpenFace22AUPredictor

        predictor = OpenFace22AUPredictor(
            str(self.openface_binary),
            str(self.models_dir),
            str(self.pdm_file)
        )

        df = predictor.predict_video(str(video_path), verbose=True)
        elapsed = time.time() - start_time
        fps = len(df) / elapsed

        print(f"\n✓ Group 2 Complete (pyfaceau AU predictor)!")
        print(f"  Frames: {len(df)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")

        return df

    def run_group3_pyfacelm_to_s1(self, video_path: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Group 3: PyfaceLM landmarks → S1 AU extraction

        Step 1: Extract landmarks with PyfaceLM
        Step 2: Use S1's AU extraction on those landmarks

        Challenge: S1 (pyfaceau) expects C++ features (FHOG + PDM), not just landmarks
        Solution: Save PyfaceLM landmarks to CSV, then run pyfaceau AU predictor
        """
        print(f"\n{'='*80}")
        print("GROUP 3: PyfaceLM → S1 AU Extraction")
        print(f"{'='*80}")

        start_time = time.time()

        # Step 1: Extract landmarks with PyfaceLM
        print("\n[Step 1/2] Extracting landmarks with PyfaceLM...")

        detector = CLNFDetector(enable_cache=False)  # Disable cache for video

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        landmarks_list = []
        confidences = []
        frame_idx = 0
        temp_frames_dir = output_dir / "temp_frames"
        temp_frames_dir.mkdir(exist_ok=True)

        print(f"  Processing {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame temporarily
            temp_path = temp_frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(temp_path), frame)

            # Detect landmarks
            try:
                landmarks, conf, bbox = detector.detect(str(temp_path))
                landmarks_list.append(landmarks)
                confidences.append(conf)
            except Exception as e:
                print(f"    Frame {frame_idx} failed: {e}")
                landmarks_list.append(None)
                confidences.append(0.0)

            # Cleanup temp frame
            temp_path.unlink()

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"    Processed {frame_idx}/{total_frames} frames...")

        cap.release()

        # Create landmarks DataFrame
        landmarks_df = pd.DataFrame({
            'frame': range(len(landmarks_list)),
            'confidence': confidences
        })

        # Add landmark coordinates
        for pt in range(68):
            landmarks_df[f'x_{pt}'] = [lm[pt, 0] if lm is not None else 0 for lm in landmarks_list]
            landmarks_df[f'y_{pt}'] = [lm[pt, 1] if lm is not None else 0 for lm in landmarks_list]

        # Save landmarks
        landmarks_csv = output_dir / "group3_landmarks.csv"
        landmarks_df.to_csv(landmarks_csv, index=False)

        elapsed_landmarks = time.time() - start_time
        fps_landmarks = len(landmarks_df) / elapsed_landmarks

        print(f"\n  ✓ Landmarks extracted!")
        print(f"    Frames: {len(landmarks_df)}")
        print(f"    Time: {elapsed_landmarks:.2f}s")
        print(f"    FPS: {fps_landmarks:.2f}")

        # Step 2: Extract AUs using pyfaceau AU predictor
        print("\n[Step 2/2] Extracting AUs with pyfaceau AU predictor...")

        from pyfaceau.prediction.au_predictor import OpenFace22AUPredictor

        predictor = OpenFace22AUPredictor(
            str(self.openface_binary),
            str(self.models_dir),
            str(self.pdm_file)
        )

        aus_df = predictor.predict_video(str(video_path), verbose=False)

        elapsed_total = time.time() - start_time
        fps_total = len(landmarks_df) / elapsed_total

        print(f"\n✓ Group 3 Complete!")
        print(f"  Total time: {elapsed_total:.2f}s")
        print(f"  Overall FPS: {fps_total:.2f}")
        print(f"  Landmark FPS: {fps_landmarks:.2f}")
        print(f"  AU extraction time: {elapsed_total - elapsed_landmarks:.2f}s")

        return landmarks_df, aus_df

    def compare_results(
        self,
        group1_df: pd.DataFrame,
        group2_df: Optional[pd.DataFrame],
        group3_landmarks: pd.DataFrame,
        group3_aus: pd.DataFrame,
        output_dir: Path
    ):
        """Compare all three groups"""
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")

        # Determine gold standard (Group 2 if available, else Group 1)
        gold_df = group2_df if group2_df is not None else group1_df
        gold_name = "Group 2 (C++)" if group2_df is not None else "Group 1 (S1)"

        print(f"\nUsing {gold_name} as gold standard")

        # Get AU columns
        au_cols = [c for c in gold_df.columns if c.startswith('AU') and ('_r' in c or '_c' in c)]
        print(f"\nAUs to compare: {len(au_cols)}")

        # Compare Group 1 vs Gold (if different)
        if gold_df is not group1_df:
            print(f"\n{'-'*80}")
            print(f"Group 1 (S1) vs {gold_name}")
            print(f"{'-'*80}")

            correlations = {}
            for au_col in au_cols:
                if au_col in group1_df.columns:
                    # Align frames
                    min_frames = min(len(gold_df), len(group1_df))
                    corr = np.corrcoef(
                        gold_df[au_col].values[:min_frames],
                        group1_df[au_col].values[:min_frames]
                    )[0, 1]
                    correlations[au_col] = corr

            avg_corr = np.mean(list(correlations.values()))
            print(f"\n  Average AU correlation: r = {avg_corr:.4f}")

            # Show per-AU correlations
            print(f"\n  Per-AU correlations:")
            for au, corr in sorted(correlations.items()):
                print(f"    {au}: r = {corr:.4f}")

        # Compare Group 3 landmarks vs Gold
        print(f"\n{'-'*80}")
        print(f"Group 3 (PyfaceLM) Landmarks vs {gold_name}")
        print(f"{'-'*80}")

        # Calculate landmark errors
        errors = []
        min_frames = min(len(gold_df), len(group3_landmarks))

        for i in range(min_frames):
            gold_lm = np.array([[gold_df.loc[i, f'x_{j}'], gold_df.loc[i, f'y_{j}']] for j in range(68)])
            g3_lm = np.array([[group3_landmarks.loc[i, f'x_{j}'], group3_landmarks.loc[i, f'y_{j}']] for j in range(68)])

            error = np.abs(g3_lm - gold_lm).mean()
            errors.append(error)

        print(f"\n  Mean landmark error: {np.mean(errors):.4f} px")
        print(f"  Max landmark error:  {np.max(errors):.4f} px")
        print(f"  Min landmark error:  {np.min(errors):.4f} px")

        # Compare Group 3 AUs vs Gold
        print(f"\n{'-'*80}")
        print(f"Group 3 (PyfaceLM + S1 AU) vs {gold_name}")
        print(f"{'-'*80}")

        correlations_g3 = {}
        min_frames = min(len(gold_df), len(group3_aus))

        for au_col in au_cols:
            if au_col in group3_aus.columns:
                corr = np.corrcoef(
                    gold_df[au_col].values[:min_frames],
                    group3_aus[au_col].values[:min_frames]
                )[0, 1]
                correlations_g3[au_col] = corr

        avg_corr_g3 = np.mean(list(correlations_g3.values()))
        print(f"\n  Average AU correlation: r = {avg_corr_g3:.4f}")

        # Show per-AU correlations
        print(f"\n  Per-AU correlations:")
        for au, corr in sorted(correlations_g3.items()):
            print(f"    {au}: r = {corr:.4f}")

        # Save detailed report
        report_path = output_dir / "au_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write("THREE AU PIPELINE COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Video: {len(gold_df)} frames\n")
            f.write(f"Gold standard: {gold_name}\n\n")

            f.write(f"Group 3 Landmark Error: {np.mean(errors):.4f} px\n\n")

            if gold_df is not group1_df:
                f.write(f"Group 1 vs Gold: r = {avg_corr:.4f}\n")
            f.write(f"Group 3 AUs vs Gold: r = {avg_corr_g3:.4f}\n\n")

            f.write("Detailed AU Correlations (Group 3):\n")
            for au, corr in sorted(correlations_g3.items()):
                f.write(f"  {au}: {corr:.4f}\n")

        print(f"\n✓ Report saved: {report_path}")

        return {
            "landmark_error": np.mean(errors),
            "au_correlation_g3": avg_corr_g3,
            "au_correlations_detail": correlations_g3
        }


def main():
    """Run three-pipeline comparison on IMG_0434.MOV"""

    # Setup
    video_path = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0434.MOV")
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/au_comparison_results")
    output_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    print(f"\nTest video: {video_path.name}")
    print(f"Output directory: {output_dir}\n")

    # Initialize comparison
    comparison = ThreeAUPipelineComparison()

    # Run Group 1: S1 Current
    print("\nStarting Group 1...")
    group1_df = comparison.run_group1_s1_current(video_path, output_dir)

    # Run Group 2: C++ OpenFace (or fallback)
    print("\nStarting Group 2...")
    group2_df = comparison.run_group2_cpp_openface(video_path, output_dir / "group2")

    # Run Group 3: PyfaceLM → S1
    print("\nStarting Group 3...")
    group3_landmarks, group3_aus = comparison.run_group3_pyfacelm_to_s1(video_path, output_dir / "group3")

    # Compare results
    results = comparison.compare_results(
        group1_df,
        group2_df,
        group3_landmarks,
        group3_aus,
        output_dir
    )

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nPyfaceLM landmark error: {results['landmark_error']:.4f} px")
    print(f"Group 3 AU correlation:  r = {results['au_correlation_g3']:.4f}")
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
