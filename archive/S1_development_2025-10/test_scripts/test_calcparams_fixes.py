#!/usr/bin/env python3
"""
Quick test of CalcParams fixes:
1. Tikhonov regularization (lambda=1e-4)
2. SVD-based pseudo-inverse for ill-conditioned matrices
3. Isolated PDM copies to prevent state corruption
"""

import cv2
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

from pdm_parser import PDMParser
from calc_params import CalcParams
from face_aligner import FaceAligner
from pyfhog_extractor import PyFHOGExtractor
from openface22_au_predictor import OpenFace22AUPredictor
from running_median_tracker import RunningMedianTracker

# Suppress expected warnings
warnings.filterwarnings('ignore', category=np.linalg.LinAlgWarning)

print("="*80)
print("CalcParams Fix Validation - 50 Frame Test")
print("="*80)

# Test configuration
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PDM_PATH = "In-the-wild_aligned_PDM_68.txt"
AU_MODELS_PATH = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"

# Load components
print("\n1. Loading components...")
pdm = PDMParser(PDM_PATH)
calc_params = CalcParams(pdm)
face_aligner = FaceAligner(pdm)
hog_extractor = PyFHOGExtractor()
au_predictor = OpenFace22AUPredictor(None, AU_MODELS_PATH, PDM_PATH)
tracker = RunningMedianTracker()

# Load test data
print("\n2. Loading test data...")
cap = cv2.VideoCapture(VIDEO_PATH)
df = pd.read_csv(CSV_PATH)
print(f"  Video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
print(f"  CSV: {len(df)} rows")

# Test on 50 frames
TEST_FRAMES = 50
print(f"\n3. Testing CalcParams fixes on {TEST_FRAMES} frames...")

au_predictions = {au: [] for au in au_predictor.au_models.keys()}
au_baseline = {au: [] for au in au_predictor.au_models.keys()}

ill_conditioned_count = 0
total_iterations = 0

for frame_idx in range(TEST_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    # Get CSV ground truth
    csv_row = df.iloc[frame_idx]

    # Get landmarks from CSV
    landmarks_2d = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks_2d[i, 0] = csv_row[f'x_{i}']
        landmarks_2d[i, 1] = csv_row[f'y_{i}']

    # Run CalcParams with fixes
    try:
        pose_params, shape_params = calc_params.calc_params(landmarks_2d)

        # Extract features using CalcParams output
        aligned_face = face_aligner.align_face(frame, landmarks_2d, pose_params)
        hog_features = hog_extractor.extract(aligned_face)

        # Get 3D shape from CalcParams
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
        print(f"  Frame {frame_idx}: Failed - {e}")
        for au in au_predictor.au_models.keys():
            au_predictions[au].append(0.0)
            au_baseline[au].append(csv_row[f' {au}'])

    if (frame_idx + 1) % 10 == 0:
        print(f"  Processed {frame_idx + 1}/{TEST_FRAMES} frames...")

cap.release()

# Calculate correlations
print("\n" + "="*80)
print("AU Correlation Results (After Fixes)")
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
        print(f"  {au}: r=N/A (no variance)")

mean_r = np.mean(correlations)

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Mean correlation: {mean_r:.4f}")
print(f"\nComparison:")
print(f"  Before fixes: r=0.4954 (BROKEN)")
print(f"  Target:       r=0.8302 (CSV baseline)")
print(f"  After fixes:  r={mean_r:.4f}")

if mean_r > 0.75:
    improvement = ((mean_r - 0.4954) / 0.4954) * 100
    print(f"\n✅ SUCCESS: CalcParams fixes work! {improvement:.1f}% improvement")
elif mean_r > 0.65:
    print(f"\n⚠️  PARTIAL: Some improvement, but needs more work")
else:
    print(f"\n❌ FAILED: Fixes didn't resolve the issue")

print("\n" + "="*80)
