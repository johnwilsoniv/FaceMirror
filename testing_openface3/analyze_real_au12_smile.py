#!/usr/bin/env python3
"""
Analyze Real AU12 (Smile) Signal from OpenFace 3.0 Output

OpenFace 3.0 CSV column AU03_r actually contains the real AU12 (Lip Corner Puller/Smile).
This script analyzes whether it provides a good signal for smile detection.
"""

import pandas as pd
import numpy as np

# Load OpenFace 3.0 output
csv_path = "openface3_output.csv"
df = pd.read_csv(csv_path)

print("="*80)
print("REAL AU12 (SMILE) SIGNAL ANALYSIS")
print("="*80)
print(f"\nVideo: IMG_0422.MOV (Normal Cohort)")
print(f"Frames analyzed: {len(df)}")
print()

# AU03_r contains the REAL AU12 (Lip Corner Puller/Smile)
real_au12 = df['AU03_r'].values

print("Column AU03_r - REAL AU12 (Lip Corner Puller/Smile)")
print("-" * 80)
print(f"Mean intensity:   {real_au12.mean():.6f}")
print(f"Max intensity:    {real_au12.max():.6f}")
print(f"Min intensity:    {real_au12.min():.6f}")
print(f"Std deviation:    {real_au12.std():.6f}")
print(f"Median:           {np.median(real_au12):.6f}")
print(f"Non-zero frames:  {np.count_nonzero(real_au12)}/{len(real_au12)} ({100*np.count_nonzero(real_au12)/len(real_au12):.1f}%)")
print()

# Frame-by-frame breakdown
print("Frame-by-Frame Values:")
print("-" * 80)
for i, val in enumerate(real_au12):
    if val > 0.001:  # Only show frames with meaningful activity
        print(f"  Frame {i:2d}: {val:.6f}")

if np.count_nonzero(real_au12 > 0.001) == 0:
    print("  (No frames with AU12 > 0.001)")
print()

# Compare to other AUs for context
print("Comparison to Other AUs:")
print("-" * 80)
au_columns = ['AU01_r', 'AU02_r', 'AU03_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU08_r']
actual_aus = ['AU01', 'AU06', 'AU12', 'AU15', 'AU17', 'AU02', 'AU09', 'AU10']

for csv_col, actual_au in zip(au_columns, actual_aus):
    mean_val = df[csv_col].mean()
    max_val = df[csv_col].max()
    nonzero = np.count_nonzero(df[csv_col] > 0.001)

    if actual_au == 'AU12':
        marker = " ← SMILE (real AU12)"
    else:
        marker = f" ({actual_au})"

    print(f"  {csv_col}: mean={mean_val:.6f}, max={max_val:.6f}, non-zero={nonzero:2d}{marker}")
print()

# Assessment
print("ASSESSMENT: Does AU12 Give Good Smile Signal?")
print("="*80)

# Calculate signal strength
mean_au12 = real_au12.mean()
max_au12 = real_au12.max()
nonzero_pct = 100 * np.count_nonzero(real_au12 > 0.001) / len(real_au12)

if max_au12 < 0.01:
    assessment = "❌ VERY WEAK - Essentially no smile detected"
    reason = f"Max value {max_au12:.6f} is extremely low (< 0.01)"
elif max_au12 < 0.1:
    assessment = "⚠️  WEAK - Minimal smile activity"
    reason = f"Max value {max_au12:.6f} is below typical AU threshold (0.1-0.5)"
elif max_au12 < 0.5:
    assessment = "⚙️  MODERATE - Some smile detected"
    reason = f"Max value {max_au12:.6f} shows activation but below strong smile levels"
else:
    assessment = "✓ STRONG - Good smile signal"
    reason = f"Max value {max_au12:.6f} indicates clear smile activation"

print(f"\n{assessment}")
print(f"\nReason: {reason}")
print(f"Context: This video likely shows neutral/resting face (Normal Cohort)")
print(f"         Only {nonzero_pct:.1f}% of frames show any AU12 activity (> 0.001)")
print()

# Recommendation
print("RECOMMENDATION:")
print("-" * 80)
if max_au12 < 0.1:
    print("To properly test AU12 for smile detection:")
    print("  1. Process a video with explicit 'Big Smile' (BS) action")
    print("  2. Expected AU12 values during smile: 0.5 - 5.0")
    print("  3. This neutral face video is not suitable for smile validation")
else:
    print("AU12 signal appears sufficient for smile detection.")
    print("Further validation recommended with explicit smile actions.")
print()

# Note about the video type
print("NOTE: This video (IMG_0422.MOV) appears to be a neutral/resting face")
print("      recording, which explains the very low AU12 values.")
print("      For smile validation, we need a video with 'BS' (Big Smile) action.")
print()
