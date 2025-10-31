#!/usr/bin/env python3
"""
Compare AU extraction performance: Cunjian PFLD vs FAN2

Tests which landmark detector produces better AU correlation with OpenFace 2.2 baseline.
Target: r > 0.80 correlation for most AUs
"""

import cv2
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from cunjian_pfld_detector import CunjianPFLDDetector
from fan2_landmark_detector import FAN2LandmarkDetector
from onnx_retinaface_detector import ONNXRetinaFaceDetector
from openface22_au_predictor import OpenFace22AUPredictor

# Test video and baseline
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
BASELINE_CSV = "of22_validation/IMG_0942_left_mirrored.csv"

# AU list from OpenFace 2.2
AUS = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
       'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']

print("="*80)
print("CUNJIAN PFLD vs FAN2 - AU EXTRACTION COMPARISON")
print("="*80)

# Load baseline
print(f"\nLoading baseline from: {BASELINE_CSV}")
baseline_df = pd.read_csv(BASELINE_CSV)
print(f"Baseline: {len(baseline_df)} frames")

# Initialize detectors
print("\n" + "="*80)
print("INITIALIZING DETECTORS")
print("="*80)

print("\nLoading face detector (RetinaFace)...")
face_detector = ONNXRetinaFaceDetector('weights/retinaface_mobilenet025_coreml.onnx')

print("Loading cunjian PFLD landmark detector...")
pfld_detector = CunjianPFLDDetector('weights/pfld_cunjian.onnx')
print(f"  {pfld_detector}")

print("Loading FAN2 landmark detector...")
fan2_detector = FAN2LandmarkDetector('weights/fan2_68_landmark.onnx')
print(f"  {fan2_detector}")

print("Loading OpenFace 2.2 AU predictor...")
OF_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
OF_MODELS = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
OF_PDM = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt"
au_predictor = OpenFace22AUPredictor(OF_BINARY, OF_MODELS, OF_PDM)
print("  AU predictor ready")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nVideo: {total_frames} frames")

# Test on first 50 frames
test_frames = min(50, len(baseline_df))
print(f"Testing first {test_frames} frames...")

# Storage for predictions
pfld_predictions = {au: [] for au in AUS}
fan2_predictions = {au: [] for au in AUS}
baseline_values = {au: [] for au in AUS}

print("\n" + "="*80)
print("PROCESSING FRAMES")
print("="*80)

for frame_idx in range(test_frames):
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        break

    # Get baseline AUs for this frame
    baseline_row = baseline_df.iloc[frame_idx]
    for au in AUS:
        baseline_values[au].append(baseline_row[f' {au}_r'])

    # Detect face
    faces = face_detector.detect_faces(frame)
    if len(faces) == 0:
        print(f"Frame {frame_idx}: No face detected")
        for au in AUS:
            pfld_predictions[au].append(0.0)
            fan2_predictions[au].append(0.0)
        continue

    # Use first face
    bbox = faces[0][:4]

    # ===== PFLD PIPELINE =====
    try:
        pfld_landmarks, _ = pfld_detector.detect_landmarks(frame, bbox)
        pfld_aus = au_predictor.predict_aus_from_landmarks(pfld_landmarks, frame)
        for au in AUS:
            pfld_predictions[au].append(pfld_aus.get(au, 0.0))
    except Exception as e:
        print(f"Frame {frame_idx}: PFLD failed - {e}")
        for au in AUS:
            pfld_predictions[au].append(0.0)

    # ===== FAN2 PIPELINE =====
    try:
        fan2_landmarks, _ = fan2_detector.detect_landmarks(frame, bbox)
        fan2_aus = au_predictor.predict_aus_from_landmarks(fan2_landmarks, frame)
        for au in AUS:
            fan2_predictions[au].append(fan2_aus.get(au, 0.0))
    except Exception as e:
        print(f"Frame {frame_idx}: FAN2 failed - {e}")
        for au in AUS:
            fan2_predictions[au].append(0.0)

    if (frame_idx + 1) % 10 == 0:
        print(f"  Processed {frame_idx + 1}/{test_frames} frames...")

cap.release()

# Calculate correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

pfld_correlations = {}
fan2_correlations = {}

for au in AUS:
    baseline_vals = np.array(baseline_values[au])
    pfld_vals = np.array(pfld_predictions[au])
    fan2_vals = np.array(fan2_predictions[au])

    # Calculate Pearson correlation
    if baseline_vals.std() > 0 and pfld_vals.std() > 0:
        pfld_r, pfld_p = pearsonr(baseline_vals, pfld_vals)
        pfld_correlations[au] = pfld_r
    else:
        pfld_correlations[au] = 0.0

    if baseline_vals.std() > 0 and fan2_vals.std() > 0:
        fan2_r, fan2_p = pearsonr(baseline_vals, fan2_vals)
        fan2_correlations[au] = fan2_r
    else:
        fan2_correlations[au] = 0.0

# Print results
print(f"\n{'AU':<8} {'PFLD r':<12} {'FAN2 r':<12} {'Winner':<15} {'Δ':<10}")
print("-"*60)

pfld_wins = 0
fan2_wins = 0

for au in AUS:
    pfld_r = pfld_correlations[au]
    fan2_r = fan2_correlations[au]

    if pfld_r > fan2_r:
        winner = "PFLD ✓"
        pfld_wins += 1
    elif fan2_r > pfld_r:
        winner = "FAN2 ✓"
        fan2_wins += 1
    else:
        winner = "Tie"

    delta = pfld_r - fan2_r

    print(f"{au:<8} {pfld_r:>6.3f}      {fan2_r:>6.3f}      {winner:<15} {delta:>+6.3f}")

# Summary statistics
pfld_mean = np.mean([pfld_correlations[au] for au in AUS])
pfld_median = np.median([pfld_correlations[au] for au in AUS])
pfld_above_80 = sum(1 for au in AUS if pfld_correlations[au] > 0.80)

fan2_mean = np.mean([fan2_correlations[au] for au in AUS])
fan2_median = np.median([fan2_correlations[au] for au in AUS])
fan2_above_80 = sum(1 for au in AUS if fan2_correlations[au] > 0.80)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\n{'Metric':<30} {'PFLD':<15} {'FAN2':<15}")
print("-"*60)
print(f"{'Mean correlation':<30} {pfld_mean:>6.3f}         {fan2_mean:>6.3f}")
print(f"{'Median correlation':<30} {pfld_median:>6.3f}         {fan2_median:>6.3f}")
print(f"{'AUs with r > 0.80':<30} {pfld_above_80:>6}/{len(AUS):<8} {fan2_above_80:>6}/{len(AUS):<8}")
print(f"{'Head-to-head wins':<30} {pfld_wins:>6}/{len(AUS):<8} {fan2_wins:>6}/{len(AUS):<8}")

print("\n" + "="*80)
print("OVERALL WINNER")
print("="*80)

if pfld_mean > fan2_mean:
    improvement = ((pfld_mean - fan2_mean) / fan2_mean) * 100
    print(f"\n🏆 PFLD WINS with {improvement:.1f}% better mean correlation")
    print(f"   PFLD: {pfld_mean:.3f} vs FAN2: {fan2_mean:.3f}")
elif fan2_mean > pfld_mean:
    improvement = ((fan2_mean - pfld_mean) / pfld_mean) * 100
    print(f"\n🏆 FAN2 WINS with {improvement:.1f}% better mean correlation")
    print(f"   FAN2: {fan2_mean:.3f} vs PFLD: {pfld_mean:.3f}")
else:
    print("\n🤝 TIE - Both models perform equally")

print("\n" + "="*80)
print("DECISION CRITERIA")
print("="*80)
print(f"Target: r > 0.80 for most AUs")
print(f"PFLD: {pfld_above_80}/{len(AUS)} AUs above threshold")
print(f"FAN2: {fan2_above_80}/{len(AUS)} AUs above threshold")

if pfld_above_80 >= fan2_above_80 and pfld_mean >= fan2_mean:
    print(f"\n✅ RECOMMENDATION: Use PFLD (better NME, faster, smaller, equal/better AU correlation)")
elif fan2_above_80 > pfld_above_80 or fan2_mean > pfld_mean:
    print(f"\n✅ RECOMMENDATION: Use FAN2 (better AU correlation despite worse NME)")
else:
    print(f"\n⚠️  RECOMMENDATION: Further testing needed or consider dlib fallback")
