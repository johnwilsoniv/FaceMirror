#!/usr/bin/env python3
"""
Process IMG_0942_left_mirrored.mp4 with the FIXED OF3.0 and compare to OF2.2
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from openface_integration import OpenFace3Processor

# Paths
video_path = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4")
of22_csv = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_left_mirrored.csv")
output_csv = Path("./test_output/IMG_0942_left_mirrored_OF30_FIXED.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTING RETINAFACE FIX - IMG_0942")
print("=" * 80)
print()

# Process with fixed OF3.0
print("Step 1: Processing video with FIXED OpenFace 3.0...")
print(f"  Video: {video_path.name}")
print(f"  Output: {output_csv}")
print()

processor = OpenFace3Processor(
    device='cpu',
    skip_face_detection=True,  # THE FIX
    debug_mode=False
)

print(f"Processor config: skip_face_detection={processor.skip_face_detection}")
print()

# Process video
frame_count = processor.process_video(video_path, output_csv)

print()
print(f"✓ Processed {frame_count} frames")
print()

# Load and compare
print("Step 2: Comparing with OpenFace 2.2...")
print("-" * 80)

df_of22 = pd.read_csv(of22_csv)
df_of30 = pd.read_csv(output_csv)

print(f"OF2.2: {len(df_of22)} frames")
print(f"OF3.0: {len(df_of30)} frames")
print()

# Compare AU12_r specifically
if 'AU12_r' in df_of22.columns and 'AU12_r' in df_of30.columns:
    of22_au12 = df_of22['AU12_r'].values
    of30_au12 = df_of30['AU12_r'].values

    min_len = min(len(of22_au12), len(of30_au12))
    of22_au12 = of22_au12[:min_len]
    of30_au12 = of30_au12[:min_len]

    # Remove NaNs
    valid = ~np.isnan(of22_au12) & ~np.isnan(of30_au12)

    if np.sum(valid) > 0:
        correlation = np.corrcoef(of22_au12[valid], of30_au12[valid])[0, 1]
        mae = np.mean(np.abs(of22_au12[valid] - of30_au12[valid]))

        of22_mean = np.mean(of22_au12[valid])
        of22_max = np.max(of22_au12[valid])
        of30_mean = np.mean(of30_au12[valid])
        of30_max = np.max(of30_au12[valid])

        print("AU12_r Comparison:")
        print(f"  OF2.2: mean={of22_mean:.3f}, max={of22_max:.3f}")
        print(f"  OF3.0: mean={of30_mean:.3f}, max={of30_max:.3f}")
        print(f"  Correlation: r={correlation:.3f}")
        print(f"  MAE: {mae:.3f}")
        print()

        if correlation > 0.7:
            print("✓ SUCCESS: High correlation - fix is working!")
        elif correlation > 0.5:
            print("✓ IMPROVED: Moderate correlation")
        else:
            print("⚠ WARNING: Low correlation - may need investigation")
    else:
        print("⚠ No valid AU12_r data for comparison")
else:
    print("⚠ AU12_r not found in one or both CSVs")

print()

# Compare all AUs
key_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU20_r', 'AU25_r']
available = [au for au in key_aus if au in df_of22.columns and au in df_of30.columns]

print(f"Comparing {len(available)} AUs:")
print("-" * 80)
for au in available:
    of22_vals = df_of22[au].values[:min_len]
    of30_vals = df_of30[au].values[:min_len]

    valid = ~np.isnan(of22_vals) & ~np.isnan(of30_vals)
    if np.sum(valid) > 10:
        corr = np.corrcoef(of22_vals[valid], of30_vals[valid])[0, 1]
        mae = np.mean(np.abs(of22_vals[valid] - of30_vals[valid]))
        print(f"{au:8s}: r={corr:.3f}, MAE={mae:.3f}")

print()
print("=" * 80)
print("Test complete! Check the comparison above.")
print(f"CSV saved to: {output_csv}")
print("=" * 80)
