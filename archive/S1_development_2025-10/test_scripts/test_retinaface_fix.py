#!/usr/bin/env python3
"""
Test script to verify the RetinaFace fix.

This script will:
1. Process IMG_0942_left_mirrored.mp4 with the FIXED OF3.0 (skip_face_detection=True)
2. Load the OF2.2 CSV for comparison
3. Generate comparison plots showing AU12_r and other AUs
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from openface_integration import OpenFace3Processor


def test_fix(video_path, of22_csv_path, output_dir):
    """
    Test the RetinaFace fix by processing a video and comparing to OF2.2

    Args:
        video_path: Path to mirrored video
        of22_csv_path: Path to OF2.2 CSV output
        output_dir: Directory for output files
    """
    video_path = Path(video_path)
    of22_csv_path = Path(of22_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TESTING RETINAFACE FIX")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"OF2.2 CSV: {of22_csv_path.name}")
    print(f"Output: {output_dir}")
    print()

    # ============================================================================
    # Step 1: Process with FIXED OF3.0 (skip_face_detection=True)
    # ============================================================================
    print("STEP 1: Processing with FIXED OpenFace 3.0")
    print("-" * 80)

    processor = OpenFace3Processor(
        device='cpu',
        skip_face_detection=True,  # THE FIX
        debug_mode=False
    )

    print(f"\nProcessor configuration:")
    print(f"  skip_face_detection = {processor.skip_face_detection}")
    print(f"  Using RetinaFace: {'NO (CORRECT!)' if processor.skip_face_detection else 'YES (PROBLEM!)'}")
    print()

    # Process video
    of30_csv_path = output_dir / "IMG_0942_left_mirrored_OF30_FIXED.csv"

    print(f"Processing video...")
    frame_count = processor.process_video(video_path, of30_csv_path)
    print(f"✓ Processed {frame_count} frames")
    print(f"✓ Saved to: {of30_csv_path}")
    print()

    # ============================================================================
    # Step 2: Load both CSVs
    # ============================================================================
    print("STEP 2: Loading CSV files for comparison")
    print("-" * 80)

    df_of22 = pd.read_csv(of22_csv_path)
    df_of30 = pd.read_csv(of30_csv_path)

    print(f"OF2.2 CSV: {len(df_of22)} rows")
    print(f"OF3.0 CSV: {len(df_of30)} rows")
    print()

    # ============================================================================
    # Step 3: Compare AU values
    # ============================================================================
    print("STEP 3: Comparing AU values")
    print("-" * 80)

    # Focus on key AUs
    key_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU20_r', 'AU25_r']

    # Check which AUs are available in both
    available_aus = [au for au in key_aus if au in df_of22.columns and au in df_of30.columns]

    print(f"Comparing {len(available_aus)} AUs: {', '.join(available_aus)}")
    print()

    # Compute statistics
    for au in available_aus:
        of22_values = df_of22[au].values
        of30_values = df_of30[au].values

        # Handle different lengths
        min_len = min(len(of22_values), len(of30_values))
        of22_values = of22_values[:min_len]
        of30_values = of30_values[:min_len]

        # Remove NaN values
        valid_mask = ~np.isnan(of22_values) & ~np.isnan(of30_values)
        of22_valid = of22_values[valid_mask]
        of30_valid = of30_values[valid_mask]

        if len(of22_valid) == 0:
            print(f"{au}: No valid data")
            continue

        # Compute metrics
        mae = np.mean(np.abs(of22_valid - of30_valid))
        rmse = np.sqrt(np.mean((of22_valid - of30_valid) ** 2))
        correlation = np.corrcoef(of22_valid, of30_valid)[0, 1]

        of22_mean = np.mean(of22_valid)
        of30_mean = np.mean(of30_valid)

        print(f"{au}:")
        print(f"  OF2.2 mean: {of22_mean:.3f}")
        print(f"  OF3.0 mean: {of30_mean:.3f}")
        print(f"  MAE:        {mae:.3f}")
        print(f"  RMSE:       {rmse:.3f}")
        print(f"  Correlation: {correlation:.3f}")
        print()

    # ============================================================================
    # Step 4: Generate visualizations
    # ============================================================================
    print("STEP 4: Generating comparison plots")
    print("-" * 80)

    # Create temporal comparison plot
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('OF2.2 vs OF3.0 (FIXED) - AU Comparison', fontsize=14, fontweight='bold')

    for idx, (ax, au) in enumerate(zip(axes.flat, available_aus)):
        if au not in df_of22.columns or au not in df_of30.columns:
            ax.axis('off')
            continue

        of22_values = df_of22[au].values
        of30_values = df_of30[au].values

        # Handle different lengths
        min_len = min(len(of22_values), len(of30_values))
        frames = np.arange(min_len)

        ax.plot(frames, of22_values[:min_len], 'b-', label='OF2.2', linewidth=2, alpha=0.7)
        ax.plot(frames, of30_values[:min_len], 'r-', label='OF3.0 (FIXED)', linewidth=2, alpha=0.7)

        ax.set_title(au, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Intensity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add correlation text
        valid_mask = ~np.isnan(of22_values[:min_len]) & ~np.isnan(of30_values[:min_len])
        if np.sum(valid_mask) > 0:
            correlation = np.corrcoef(of22_values[:min_len][valid_mask],
                                     of30_values[:min_len][valid_mask])[0, 1]
            ax.text(0.98, 0.98, f'r={correlation:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / "of22_vs_of30_fixed_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

    # Create focused AU12_r comparison
    if 'AU12_r' in available_aus:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle('AU12_r Detailed Comparison: OF2.2 vs OF3.0 (FIXED)',
                    fontsize=14, fontweight='bold')

        of22_au12 = df_of22['AU12_r'].values
        of30_au12 = df_of30['AU12_r'].values
        min_len = min(len(of22_au12), len(of30_au12))
        frames = np.arange(min_len)

        # Temporal plot
        ax1.plot(frames, of22_au12[:min_len], 'b-', label='OF2.2', linewidth=2)
        ax1.plot(frames, of30_au12[:min_len], 'r-', label='OF3.0 (FIXED)', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('AU12_r Intensity', fontsize=12)
        ax1.set_title('Temporal Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        valid_mask = ~np.isnan(of22_au12[:min_len]) & ~np.isnan(of30_au12[:min_len])
        ax2.scatter(of22_au12[:min_len][valid_mask], of30_au12[:min_len][valid_mask],
                   alpha=0.5, s=20)

        # Add diagonal line (perfect correlation)
        max_val = max(np.max(of22_au12[:min_len][valid_mask]),
                     np.max(of30_au12[:min_len][valid_mask]))
        ax2.plot([0, max_val], [0, max_val], 'k--', label='Perfect correlation', linewidth=2)

        correlation = np.corrcoef(of22_au12[:min_len][valid_mask],
                                 of30_au12[:min_len][valid_mask])[0, 1]

        ax2.set_xlabel('OF2.2 AU12_r', fontsize=12)
        ax2.set_ylabel('OF3.0 (FIXED) AU12_r', fontsize=12)
        ax2.set_title(f'Correlation: r={correlation:.3f}', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / "au12_detailed_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")

    # ============================================================================
    # Summary
    # ============================================================================
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("RESULTS:")

    # Check if AU12_r correlation improved
    if 'AU12_r' in available_aus:
        of22_au12 = df_of22['AU12_r'].values
        of30_au12 = df_of30['AU12_r'].values
        min_len = min(len(of22_au12), len(of30_au12))
        valid_mask = ~np.isnan(of22_au12[:min_len]) & ~np.isnan(of30_au12[:min_len])

        if np.sum(valid_mask) > 0:
            correlation = np.corrcoef(of22_au12[:min_len][valid_mask],
                                     of30_au12[:min_len][valid_mask])[0, 1]
            mae = np.mean(np.abs(of22_au12[:min_len][valid_mask] - of30_au12[:min_len][valid_mask]))

            print(f"  AU12_r correlation: {correlation:.3f}")
            print(f"  AU12_r MAE: {mae:.3f}")
            print()

            if correlation > 0.8:
                print("✓ EXCELLENT: High correlation - fix appears to work!")
            elif correlation > 0.6:
                print("✓ GOOD: Moderate correlation - significant improvement")
            else:
                print("⚠ WARNING: Low correlation - may need further investigation")

    print()
    print(f"Output files saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test RetinaFace fix')
    parser.add_argument('--video', type=str,
                       default='/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4',
                       help='Path to mirrored video')
    parser.add_argument('--of22-csv', type=str,
                       default='/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_left_mirrored.csv',
                       help='Path to OF2.2 CSV')
    parser.add_argument('--output-dir', type=str,
                       default='./retinaface_fix_test',
                       help='Output directory')

    args = parser.parse_args()

    test_fix(args.video, args.of22_csv, args.output_dir)
