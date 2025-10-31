#!/usr/bin/env python3
"""
Diagnostic script to investigate AU12 (smile) inversion issue.

This script will:
1. Load frames at min/max AU values for each AU
2. Extract the full 8-AU vector for those frames
3. Analyze which AU index actually correlates with smiling
4. Determine if our AU mapping is correct or if index 4 is something else
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import our OpenFace 3.0 processor
from openface_integration import OpenFace3Processor

# Key frames identified from validation analysis
VALIDATION_FRAMES = {
    'AU01': {'min': 8, 'max': 1025},
    'AU02': {'min': 1, 'max': 965},
    'AU12': {'min': 937, 'max': 702},  # INVERTED - min shows smile!
    'AU20': {'min': 7, 'max': 505},
    'AU45': {'min': 36, 'max': 406},
}

def analyze_au_vectors_at_key_frames(video_path, csv_path):
    """
    Extract and analyze full AU vectors at key frames
    """
    print("="*80)
    print("AU12 INVERSION DIAGNOSTIC")
    print("="*80)

    # Load the CSV with all AU values
    df = pd.read_csv(csv_path)

    print(f"\nLoaded CSV with {len(df)} frames")
    print(f"AU columns: {[col for col in df.columns if col.startswith('AU') and col.endswith('_r')]}")

    # Initialize processor to get raw 8-AU vectors
    print("\nInitializing OpenFace 3.0 processor...")
    processor = OpenFace3Processor(
        device='cpu',
        calculate_landmarks=False,
        debug_mode=False
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    print(f"\nAnalyzing key frames from video...")
    print("-"*80)

    # Analyze AU12 frames specifically (the inverted AU)
    au12_frames = VALIDATION_FRAMES['AU12']

    print("\n🔍 ANALYZING AU12 (SMILE) FRAMES:")
    print("-"*80)

    for frame_type, frame_num in au12_frames.items():
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"  Error reading frame {frame_num}")
            continue

        print(f"\n{frame_type.upper()} AU12 Frame (Frame {frame_num}):")

        # Get CSV row values
        if frame_num < len(df):
            csv_row = df.iloc[frame_num]
            print(f"  CSV AU12_r value: {csv_row['AU12_r']:.3f}")

            # Print all AU values from CSV
            print(f"  All CSV AU_r values:")
            for au_num in [1, 2, 4, 6, 12, 15, 20, 25]:
                au_col = f'AU{au_num:02d}_r'
                if au_col in csv_row:
                    print(f"    {au_col}: {csv_row[au_col]:.3f}")

        # Get raw 8-AU vector from model
        try:
            # Detect face
            bbox = processor.detect_single_face(frame)
            if bbox is None:
                print("  ⚠️  No face detected")
                continue

            # Crop face
            x1, y1, x2, y2 = map(int, bbox[:4])
            cropped_face = frame[y1:y2, x1:x2]

            # Get raw AU output
            emotion_logits, gaze_output, au_output = processor.multitask_model.predict(cropped_face)

            # Convert to numpy
            if hasattr(au_output, 'detach'):
                au_output = au_output.detach().cpu().numpy().flatten()

            print(f"  Raw 8-AU vector from model:")
            for idx in range(8):
                print(f"    Index {idx}: {au_output[idx]:.4f}")

            # Analyze which index has highest/lowest value
            max_idx = np.argmax(au_output)
            min_idx = np.argmin(au_output)
            print(f"\n  Highest AU: Index {max_idx} = {au_output[max_idx]:.4f}")
            print(f"  Lowest AU:  Index {min_idx} = {au_output[min_idx]:.4f}")

        except Exception as e:
            print(f"  ❌ Error processing frame: {e}")
            import traceback
            traceback.print_exc()

    cap.release()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    print("\n📊 INTERPRETATION:")
    print("-"*80)
    print("If AU12 is truly inverted:")
    print("  - MIN frame (937, BIG SMILE) should have LOW value at index 4")
    print("  - MAX frame (702, NEUTRAL) should have HIGH value at index 4")
    print()
    print("If our mapping is correct:")
    print("  - Index 4 should correspond to AU12 (Lip Corner Puller)")
    print()
    print("If the mapping is wrong:")
    print("  - Index 4 might actually be AU15 (Lip Corner Depressor)")
    print("  - Or the indices are in a different order than we assumed")
    print("="*80)


if __name__ == "__main__":
    # Use the PyTorch OpenFace 3.0 output CSV
    video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
    csv_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv"

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
    elif not Path(csv_path).exists():
        print(f"Error: CSV not found: {csv_path}")
    else:
        analyze_au_vectors_at_key_frames(video_path, csv_path)
