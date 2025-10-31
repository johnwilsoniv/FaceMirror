#!/usr/bin/env python3
"""Test CalcParams fixes with full AU prediction pipeline - 50 frames"""

import cv2
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

from pdm_parser import PDMParser
from calc_params import CalcParams
from openface22_face_aligner import OpenFace22FaceAligner
from pyfhog_extractor import PyFHOGExtractor
from openface22_au_predictor import OpenFace22AUPredictor
from running_median_tracker import RunningMedianTracker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PDM_PATH = "In-the-wild_aligned_PDM_68.txt"
AU_MODELS_PATH = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"

print("="*80)
print("CalcParams Fix Validation - 50-Frame AU Correlation Test")
print("="*80)

# Load components
print("\n1. Loading components...")
pdm = PDMParser(PDM_PATH)
calc_params = CalcParams(pdm)
face_aligner = OpenFace22FaceAligner(pdm)
hog_extractor = PyFHOGExtractor()
au_predictor = OpenFace22AUPredictor(None, AU_MODELS_PATH, PDM_PATH)
tracker = RunningMedianTracker()
print("  ✓ All components loaded")

# Load test data
cap = cv2.VideoCapture(VIDEO_PATH)
df = pd.read_csv(CSV_PATH)
print(f"\n2. Test data loaded: {len(df)} frames")

# Test on 50 frames
TEST_FRAMES = 50
au_predictions = {au: [] for au in au_predictor.au_models.keys()}
au_baseline = {au: [] for au in au_predictor.au_models.keys()}

print(f"\n3. Processing {TEST_FRAMES} frames with CalcParams fixes...")

for frame_idx in range(TEST_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    # Get CSV data
    csv_row = df.iloc[frame_idx]

    # Get landmarks from CSV
    landmarks_2d = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks_2d[i, 0] = csv_row[f'x_{i}']
        landmarks_2d[i, 1] = csv_row[f'y_{i}']

    try:
        # Run CalcParams with fixes
        pose_params, shape_params = calc_params.calc_params(landmarks_2d)

        # Align face
        aligned_face = face_aligner.align_face(frame, landmarks_2d, pose_params)

        # Extract HOG features
        hog_features = hog_extractor.extract(aligned_face)

        # Get 3D shape
        shape_3d = pdm.calc_shape_3d(shape_params)

        # Concatenate features
        geom_features = np.concatenate([shape_3d.flatten(), shape_params])
        features = np.concatenate([hog_features, geom_features])

        # Predict AUs
        aus = au_predictor.predict_aus_from_features(features, tracker, is_dynamic=True)

        # Store predictions
        for au in au_predictor.au_models.keys():
            au_predictions[au].append(aus.get(au, 0.0))
            au_baseline[au].append(csv_row[f' {au}'])

    except Exception as e:
        print(f"  Frame {frame_idx}: Error - {e}")
        for au in au_predictor.au_models.keys():
            au_predictions[au].append(0.0)
            au_baseline[au].append(csv_row[f' {au}'])

    if (frame_idx + 1) % 10 == 0:
        print(f"  Processed {frame_idx + 1}/{TEST_FRAMES}...")

cap.release()

# Calculate correlations
print("\n" + "="*80)
print("AU Correlation Results")
print("="*80)

correlations = []
for au in sorted(au_predictor.au_models.keys()):
    pred = np.array(au_predictions[au])
    base = np.array(au_baseline[au])

    if base.std() > 0 and pred.std() > 0:
        r, p = pearsonr(base, pred)
        correlations.append(r)
        print(f"  {au}: r={r:.4f}")
    else:
        print(f"  {au}: r=N/A")

mean_r = np.mean(correlations)

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"  Before fixes:  r=0.4954 ❌ (BROKEN - ill-conditioned matrices)")
print(f"  Target (CSV):  r=0.8302 ✓ (CSV baseline)")
print(f"  After fixes:   r={mean_r:.4f}")

if mean_r > 0.75:
    improvement = ((mean_r - 0.4954) / 0.4954) * 100
    deficit = ((0.8302 - mean_r) / 0.8302) * 100
    print(f"\n✅ SUCCESS: {improvement:.1f}% improvement!")
    print(f"  Now {deficit:.1f}% below CSV baseline (acceptable)")
elif mean_r > 0.60:
    improvement = ((mean_r - 0.4954) / 0.4954) * 100
    print(f"\n⚠️  PARTIAL SUCCESS: {improvement:.1f}% improvement")
    print(f"  Still needs more work to reach r=0.83 target")
else:
    print(f"\n❌ FIXES INSUFFICIENT: Correlation still too low")

print("\n" + "="*80)
