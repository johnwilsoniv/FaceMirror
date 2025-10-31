#!/usr/bin/env python3
"""
Validate AU measurements by examining frames at min/max values
Compare OF2.2 vs OF3 to see which is clinically accurate

NO CODE CHANGES - ANALYSIS ONLY
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print("="*80)
print("AU Extreme Values Analysis - Clinical Validation")
print("="*80)

# Load CSV files
of3_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv')
of22_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP22.csv')

# Video file (left side)
video_path = '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_sourceOP3ORIG.MOV'

print(f"\nLoaded data:")
print(f"  OF3: {len(of3_left)} frames")
print(f"  OF2.2: {len(of22_left)} frames")
print(f"  Video: {video_path}")

# Key AUs to analyze
key_aus = ['AU01_r', 'AU02_r', 'AU12_r', 'AU20_r', 'AU45_r']

print("\n" + "="*80)
print("1. FINDING EXTREME VALUES")
print("="*80)

extremes = []

for au in key_aus:
    # OF3
    of3_vals = of3_left[au].dropna()
    of3_min_idx = of3_vals.idxmin()
    of3_max_idx = of3_vals.idxmax()
    of3_min_val = of3_vals.loc[of3_min_idx]
    of3_max_val = of3_vals.loc[of3_max_idx]
    of3_min_frame = of3_left.loc[of3_min_idx, 'frame']
    of3_max_frame = of3_left.loc[of3_max_idx, 'frame']

    # OF2.2
    of22_vals = of22_left[au].dropna()
    of22_min_idx = of22_vals.idxmin()
    of22_max_idx = of22_vals.idxmax()
    of22_min_val = of22_vals.loc[of22_min_idx]
    of22_max_val = of22_vals.loc[of22_max_idx]
    of22_min_frame = of22_left.loc[of22_min_idx, 'frame']
    of22_max_frame = of22_left.loc[of22_max_idx, 'frame']

    extremes.append({
        'AU': au,
        'of3_min_frame': int(of3_min_frame),
        'of3_min_val': of3_min_val,
        'of3_max_frame': int(of3_max_frame),
        'of3_max_val': of3_max_val,
        'of22_min_frame': int(of22_min_frame),
        'of22_min_val': of22_min_val,
        'of22_max_frame': int(of22_max_frame),
        'of22_max_val': of22_max_val,
    })

    print(f"\n{au}:")
    print(f"  OF3:  MIN at frame {int(of3_min_frame)} (value={of3_min_val:.3f})")
    print(f"        MAX at frame {int(of3_max_frame)} (value={of3_max_val:.3f})")
    print(f"  OF2.2: MIN at frame {int(of22_min_frame)} (value={of22_min_val:.3f})")
    print(f"        MAX at frame {int(of22_max_frame)} (value={of22_max_val:.3f})")

    # Check if they agree on timing
    frame_diff_max = abs(of3_max_frame - of22_max_frame)
    if frame_diff_max < 10:
        print(f"  ✓ Both agree on MAX timing (within {frame_diff_max} frames)")
    else:
        print(f"  ✗ Different MAX timing ({frame_diff_max} frames apart)")

print("\n" + "="*80)
print("2. EXTRACTING VIDEO FRAMES")
print("="*80)

def extract_frame(video_path, frame_number):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# Extract all frames we need
frames_to_extract = set()
for extreme in extremes:
    frames_to_extract.add(extreme['of3_min_frame'])
    frames_to_extract.add(extreme['of3_max_frame'])
    frames_to_extract.add(extreme['of22_min_frame'])
    frames_to_extract.add(extreme['of22_max_frame'])

print(f"Extracting {len(frames_to_extract)} unique frames...")

frame_cache = {}
for frame_num in frames_to_extract:
    frame = extract_frame(video_path, frame_num)
    if frame is not None:
        frame_cache[frame_num] = frame
        print(f"  ✓ Frame {frame_num}")
    else:
        print(f"  ✗ Frame {frame_num} (failed)")

print("\n" + "="*80)
print("3. CREATING VISUAL COMPARISON")
print("="*80)

# Create comparison figure for each AU
for extreme in extremes:
    au = extreme['AU']

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'{au} - Clinical Validation: OF3 vs OF2.2', fontsize=16, fontweight='bold')

    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # OF3 MIN
    ax1 = fig.add_subplot(gs[0, 0])
    if extreme['of3_min_frame'] in frame_cache:
        ax1.imshow(frame_cache[extreme['of3_min_frame']])
    ax1.set_title(f"OF3 MIN\nFrame {extreme['of3_min_frame']}\nValue: {extreme['of3_min_val']:.3f}", fontsize=10)
    ax1.axis('off')

    # OF3 MAX
    ax2 = fig.add_subplot(gs[0, 1])
    if extreme['of3_max_frame'] in frame_cache:
        ax2.imshow(frame_cache[extreme['of3_max_frame']])
    ax2.set_title(f"OF3 MAX\nFrame {extreme['of3_max_frame']}\nValue: {extreme['of3_max_val']:.3f}", fontsize=10)
    ax2.axis('off')

    # OF2.2 MIN
    ax3 = fig.add_subplot(gs[1, 0])
    if extreme['of22_min_frame'] in frame_cache:
        ax3.imshow(frame_cache[extreme['of22_min_frame']])
    ax3.set_title(f"OF2.2 MIN\nFrame {extreme['of22_min_frame']}\nValue: {extreme['of22_min_val']:.3f}", fontsize=10)
    ax3.axis('off')

    # OF2.2 MAX
    ax4 = fig.add_subplot(gs[1, 1])
    if extreme['of22_max_frame'] in frame_cache:
        ax4.imshow(frame_cache[extreme['of22_max_frame']])
    ax4.set_title(f"OF2.2 MAX\nFrame {extreme['of22_max_frame']}\nValue: {extreme['of22_max_val']:.3f}", fontsize=10)
    ax4.axis('off')

    # Time series plot showing where these frames are
    ax5 = fig.add_subplot(gs[:, 2:])
    ax5.plot(of3_left['frame'], of3_left[au], label='OF3', alpha=0.7, linewidth=2)
    ax5.plot(of22_left['frame'], of22_left[au], label='OF2.2', alpha=0.7, linewidth=2, linestyle='--')

    # Mark the extreme points
    ax5.axvline(extreme['of3_min_frame'], color='blue', linestyle=':', alpha=0.5, label='OF3 min')
    ax5.axvline(extreme['of3_max_frame'], color='blue', linestyle=':', alpha=0.5, label='OF3 max')
    ax5.axvline(extreme['of22_min_frame'], color='orange', linestyle=':', alpha=0.5, label='OF2.2 min')
    ax5.axvline(extreme['of22_max_frame'], color='orange', linestyle=':', alpha=0.5, label='OF2.2 max')

    ax5.set_xlabel('Frame Number')
    ax5.set_ylabel(f'{au} Intensity')
    ax5.set_title('Time Series with Extreme Points Marked')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/validation_{au}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

print("\n" + "="*80)
print("4. AU DEFINITIONS FOR REFERENCE")
print("="*80)

au_definitions = {
    'AU01': 'Inner Brow Raiser - Frontalis, pars medialis (raises inner eyebrow)',
    'AU02': 'Outer Brow Raiser - Frontalis, pars lateralis (raises outer eyebrow)',
    'AU12': 'Lip Corner Puller - Zygomaticus major (pulls lip corners up/back, smile)',
    'AU20': 'Lip Stretcher - Risorius (stretches lips horizontally)',
    'AU45': 'Blink - Orbicularis oculi (closes eyelid)'
}

print("\nWhat to look for in the frames:")
for au_code, definition in au_definitions.items():
    print(f"\n{au_code}_r: {definition}")

print("\n" + "="*80)
print("5. CLINICAL VALIDATION CHECKLIST")
print("="*80)

print("\nFor each AU, examine the frames and ask:")
print("  1. Does the MAX frame show clear activation of the AU?")
print("  2. Does the MIN frame show absence/minimal activation?")
print("  3. Which system (OF3 or OF2.2) captures the expected facial movement?")
print("  4. Are there frames where one system is clearly wrong?")

print("\nLook for:")
print("  - AU01: Inner eyebrow should be raised in MAX frame")
print("  - AU02: Outer eyebrow should be raised in MAX frame")
print("  - AU12: Smile/lip corner pull in MAX frame")
print("  - AU20: Horizontal lip stretch in MAX frame")
print("  - AU45: Eye closed/closing in MAX frame")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("\n1. Review the validation_AU*.png files")
print("2. For each AU, determine which system is more accurate:")
print("   - Does OF3's MAX frame show the AU clearly activated?")
print("   - Does OF2.2's MAX frame show the AU clearly activated?")
print("   - Which system's MIN frame truly shows minimal activation?")

print("\n3. Document your findings:")
print("   - Which AUs does OF3 detect accurately?")
print("   - Which AUs does OF2.2 detect accurately?")
print("   - Are there systematic biases (e.g., OF3 over-sensitive)?")

print("\n4. Decision point:")
print("   - If OF2.2 is consistently more accurate → stick with OF2.2")
print("   - If OF3 is more accurate → retrain pipeline for OF3")
print("   - If mixed → may need per-AU strategy")

print("\n" + "="*80)
