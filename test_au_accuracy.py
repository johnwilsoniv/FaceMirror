#!/usr/bin/env python3
"""Test AU accuracy with current alignment."""

import numpy as np
import cv2
import pandas as pd
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

import pyfhog
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.features.triangulation import TriangulationParser
from pyfaceau.prediction.model_parser import OF22ModelParser
from pyfaceau.prediction.batched_au_predictor import BatchedAUPredictor
from pyfaceau.features.pdm import PDMParser

print("=" * 60)
print("AU ACCURACY TEST")
print("=" * 60)

# Load aligner
aligner = OpenFace22FaceAligner(
    pdm_file='/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
    sim_scale=0.7, output_size=(112, 112)
)
triangulation = TriangulationParser('/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights/tris_68_full.txt')

# Load AU models
print("Loading AU models...")
parser = OF22ModelParser('/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights/AU_predictors')
models = parser.load_all_models(use_recommended=True, use_combined=True)
predictor = BatchedAUPredictor(models)
print(f"Loaded {len(models)} AU models")

# Load PDM for geometric features
pdm = PDMParser('/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt')

# Load video
video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV'
cap = cv2.VideoCapture(video_path)
frames = []
for i in range(20):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Load C++ data
df = pd.read_csv('/tmp/cpp_0422/IMG_0422.csv')

print(f"\nProcessing {len(frames)} frames...")

# Collect predictions
all_cpp = {au: [] for au in predictor.au_names}
all_py = {au: [] for au in predictor.au_names}

for frame_idx in range(len(frames)):
    row = df.iloc[frame_idx]

    # Get landmarks
    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']

    # Compute pose
    face_center = landmarks.mean(axis=0)
    pose_tx, pose_ty = face_center[0], face_center[1]

    # Align face WITH mask
    aligned = aligner.align_face(frames[frame_idx], landmarks, pose_tx, pose_ty,
                                  apply_mask=True, triangulation=triangulation)

    # Extract HOG
    hog = pyfhog.extract_fhog_features(aligned, cell_size=8)

    # Compute geometric features (simplified - just use landmark coordinates)
    # OpenFace uses local shape params + landmarks - for now just landmarks
    geom = landmarks.flatten()  # (136,)
    # Pad to match expected size (238)
    geom_padded = np.zeros(238)
    geom_padded[:136] = geom

    # Create running median (zeros for first frames = static prediction)
    running_median = np.zeros(4702)

    # Predict AUs
    aus = predictor.predict(hog, geom_padded, running_median)

    # Collect results
    for au_name in predictor.au_names:
        all_py[au_name].append(aus[au_name])
        cpp_val = row.get(au_name, 0)
        all_cpp[au_name].append(cpp_val)

# Compute correlations
print("\n" + "=" * 60)
print("AU CORRELATIONS")
print("=" * 60)

upper_face = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r']
lower_face = ['AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

print("\nUpper face AUs:")
for au_name in upper_face:
    if au_name in all_cpp:
        cpp = np.array(all_cpp[au_name])
        py = np.array(all_py[au_name])
        if len(cpp) > 1 and np.std(cpp) > 0 and np.std(py) > 0:
            corr = np.corrcoef(cpp, py)[0, 1]
        else:
            corr = 0
        status = '✅' if corr > 0.9 else ('⚠️' if corr > 0.7 else '❌')
        print(f"  {au_name}: corr={corr:.3f} {status}")

print("\nLower face AUs:")
for au_name in lower_face:
    if au_name in all_cpp:
        cpp = np.array(all_cpp[au_name])
        py = np.array(all_py[au_name])
        if len(cpp) > 1 and np.std(cpp) > 0 and np.std(py) > 0:
            corr = np.corrcoef(cpp, py)[0, 1]
        else:
            corr = 0
        status = '✅' if corr > 0.9 else ('⚠️' if corr > 0.7 else '❌')
        print(f"  {au_name}: corr={corr:.3f} {status}")
