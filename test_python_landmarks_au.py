#!/usr/bin/env python3
"""
Test AU prediction using Python CLNF landmarks vs C++ landmarks.

This test quantifies the gap between:
1. C++ landmarks → pyfaceau → AUs (baseline: ~96% correlation with OpenFace)
2. Python CLNF landmarks → pyfaceau → AUs (what we're measuring)

The difference reveals the impact of missing eye hierarchy refinement in pyCLNF.
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import sys
import functools

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR / "pyclnf"))
sys.path.insert(0, str(SCRIPT_DIR / "pymtcnn"))
sys.path.insert(0, str(SCRIPT_DIR / "pyfaceau"))
sys.path.insert(0, str(SCRIPT_DIR / "pyfhog"))
sys.path.insert(0, str(SCRIPT_DIR))

# pyCLNF imports
from pyclnf import CLNF

# pyfaceau imports
from pyfaceau.alignment.calc_params import CalcParams
from pyfaceau.features.pdm import PDMParser
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.prediction.model_parser import OF22ModelParser
from pyfaceau.features.triangulation import TriangulationParser

# Histogram tracker
try:
    from cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
    print("Using Cython tracker")
except ImportError:
    from pyfaceau.features.histogram_median_tracker import DualHistogramMedianTracker
    print("Using Python tracker")

# PyFHOG
try:
    import pyfhog
except ImportError:
    pyfhog_path = SCRIPT_DIR / "pyfhog" / "src"
    if pyfhog_path.exists():
        sys.path.insert(0, str(pyfhog_path))
        import pyfhog

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0422.MOV"
CPP_RESULTS_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/img0422_cpp_results/IMG_0422.csv"
PDM_FILE = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
AU_MODELS_DIR = "pyfaceau/weights/AU_predictors"
TRIANGULATION_FILE = "pyfaceau/weights/tris_68_full.txt"

AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']


def extract_cpp_landmarks(cpp_df, frame_idx):
    """Extract landmarks from C++ CSV as (68, 2) array."""
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    row = cpp_df.iloc[frame_idx]
    return np.column_stack([row[x_cols].values.astype(np.float32), row[y_cols].values.astype(np.float32)])


def predict_aus(au_models, full_vector, running_median):
    """Predict AUs given features and running median."""
    result = {}
    for au_name, model in au_models.items():
        is_dynamic = (model['model_type'] == 'dynamic')
        if is_dynamic:
            centered = full_vector - model['means'].flatten() - running_median
        else:
            centered = full_vector - model['means'].flatten()
        pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
        result[au_name] = float(np.clip(pred[0, 0], 0.0, 5.0))
    return result


def process_with_landmarks(frame, landmarks_2d, calc_params, face_aligner, pdm_parser,
                           triangulation, tracker, frame_idx, au_models):
    """Process a single frame with given landmarks."""
    params_global, params_local = calc_params.calc_params(landmarks_2d)
    tx, ty, rz = params_global[4], params_global[5], params_global[3]

    aligned_face = face_aligner.align_face(
        image=frame, landmarks_68=landmarks_2d, pose_tx=tx, pose_ty=ty, p_rz=rz,
        apply_mask=True, triangulation=triangulation
    )

    hog = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
    hog = hog.reshape(12, 12, 31).transpose(1, 0, 2).flatten().astype(np.float32)
    geom = pdm_parser.extract_geometric_features(params_local).astype(np.float32)

    update_histogram = (frame_idx % 2 == 1)
    tracker.update(hog, geom, update_histogram=update_histogram)
    running_median = tracker.get_combined_median()

    full_vector = np.concatenate([hog, geom])
    preds = predict_aus(au_models, full_vector, running_median)

    return preds, hog.copy(), geom.copy(), running_median.copy()


def main():
    print("=" * 80)
    print("PYTHON LANDMARKS vs C++ LANDMARKS AU COMPARISON")
    print("=" * 80)

    cpp_df = pd.read_csv(CPP_RESULTS_PATH)
    num_frames = len(cpp_df)
    print(f"Video: {VIDEO_PATH}")
    print(f"Frames: {num_frames}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Initialize shared components
    pdm_parser = PDMParser(PDM_FILE)
    calc_params = CalcParams(pdm_parser)
    face_aligner = OpenFace22FaceAligner(PDM_FILE, sim_scale=0.7, output_size=(112, 112))
    triangulation = TriangulationParser(TRIANGULATION_FILE)
    au_models = OF22ModelParser(AU_MODELS_DIR).load_all_models(use_recommended=True, use_combined=True, verbose=False)

    # Initialize pyCLNF
    print("\nInitializing pyCLNF...")
    clnf = CLNF()
    prev_landmarks = None

    # Two trackers - one for each landmark source
    tracker_cpp = DualHistogramMedianTracker(
        hog_dim=4464, geom_dim=238,
        hog_bins=1000, hog_min=-0.005, hog_max=1.0,
        geom_bins=10000, geom_min=-60.0, geom_max=60.0
    )
    tracker_py = DualHistogramMedianTracker(
        hog_dim=4464, geom_dim=238,
        hog_bins=1000, hog_min=-0.005, hog_max=1.0,
        geom_bins=10000, geom_min=-60.0, geom_max=60.0
    )

    cpp_results = []
    py_results = []
    landmark_diffs = []

    print(f"\nProcessing {num_frames} frames...")

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Get C++ landmarks
            cpp_landmarks = extract_cpp_landmarks(cpp_df, frame_idx)

            # Get Python CLNF landmarks using detect_and_fit
            try:
                py_landmarks, info = clnf.detect_and_fit(frame)
                py_landmarks = py_landmarks.astype(np.float32)
                prev_landmarks = py_landmarks
            except ValueError as e:
                # No face detected - use previous landmarks if available
                if prev_landmarks is not None:
                    py_landmarks = prev_landmarks
                else:
                    print(f"  Frame {frame_idx}: pyCLNF detection failed: {e}")
                    frame_idx += 1
                    continue

            # Compute landmark difference
            diff = np.mean(np.sqrt(np.sum((cpp_landmarks - py_landmarks)**2, axis=1)))
            landmark_diffs.append(diff)

            # Process with C++ landmarks
            cpp_preds, _, _, _ = process_with_landmarks(
                frame, cpp_landmarks, calc_params, face_aligner, pdm_parser,
                triangulation, tracker_cpp, frame_idx, au_models
            )
            cpp_preds['frame'] = frame_idx
            cpp_results.append(cpp_preds)

            # Process with Python landmarks
            py_preds, _, _, _ = process_with_landmarks(
                frame, py_landmarks, calc_params, face_aligner, pdm_parser,
                triangulation, tracker_py, frame_idx, au_models
            )
            py_preds['frame'] = frame_idx
            py_results.append(py_preds)

        except Exception as e:
            print(f"  Frame {frame_idx} error: {e}")

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Progress: {frame_idx}/{num_frames}")

    cap.release()

    cpp_df_pred = pd.DataFrame(cpp_results)
    py_df_pred = pd.DataFrame(py_results)

    print("\n" + "=" * 100)
    print("RESULTS: Python CLNF Landmarks vs C++ Landmarks")
    print("=" * 100)

    print(f"\nMean landmark difference (RMS): {np.mean(landmark_diffs):.2f} pixels")
    print(f"Max landmark difference (RMS): {np.max(landmark_diffs):.2f} pixels")

    # Compare both to OpenFace C++ AU predictions
    print("\n" + "-" * 100)
    print(f"{'AU':<10} {'C++→pyFAU':>12} {'Py→pyFAU':>12} {'Delta':>10} {'CPP AU Mean':>12}")
    print("-" * 100)

    cpp_corrs = []
    py_corrs = []

    for au in AU_COLUMNS:
        openface_vals = cpp_df[au].values[:len(cpp_df_pred)]
        cpp_vals = cpp_df_pred[au].values
        py_vals = py_df_pred[au].values

        # Make sure lengths match
        min_len = min(len(openface_vals), len(cpp_vals), len(py_vals))
        openface_vals = openface_vals[:min_len]
        cpp_vals = cpp_vals[:min_len]
        py_vals = py_vals[:min_len]

        cpp_corr = np.corrcoef(cpp_vals, openface_vals)[0, 1] if np.std(cpp_vals) > 0 else np.nan
        py_corr = np.corrcoef(py_vals, openface_vals)[0, 1] if np.std(py_vals) > 0 else np.nan

        cpp_corrs.append(cpp_corr)
        py_corrs.append(py_corr)

        delta = py_corr - cpp_corr if not (np.isnan(cpp_corr) or np.isnan(py_corr)) else np.nan
        cpp_au_mean = np.mean(openface_vals)

        cpp_str = f"{cpp_corr:.4f}" if not np.isnan(cpp_corr) else "N/A"
        py_str = f"{py_corr:.4f}" if not np.isnan(py_corr) else "N/A"
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"

        print(f"{au:<10} {cpp_str:>12} {py_str:>12} {delta_str:>10} {cpp_au_mean:>12.2f}")

    print("-" * 100)

    mean_cpp = np.nanmean(cpp_corrs)
    mean_py = np.nanmean(py_corrs)
    mean_delta = mean_py - mean_cpp

    print(f"{'OVERALL':<10} {mean_cpp:>12.4f} {mean_py:>12.4f} {mean_delta:>+10.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nC++ landmarks → pyfaceau:     {mean_cpp*100:.2f}% correlation with OpenFace")
    print(f"Python landmarks → pyfaceau:  {mean_py*100:.2f}% correlation with OpenFace")
    print(f"Gap (Python vs C++ landmarks): {mean_delta*100:+.2f}%")

    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)

    if mean_py > 0.90:
        print("Result: GOOD - Python landmarks achieve >90% correlation")
        print("Action: Eye hierarchy refinement may not be critical")
    elif mean_py > 0.80:
        print("Result: MODERATE - Python landmarks achieve 80-90% correlation")
        print("Action: Consider implementing eye refinement for production use")
    else:
        print("Result: SIGNIFICANT GAP - Python landmarks <80% correlation")
        print("Action: Eye hierarchy refinement should be implemented")

    # Eye-related AU breakdown
    eye_aus = ['AU05_r', 'AU07_r', 'AU45_r']
    non_eye_aus = [au for au in AU_COLUMNS if au not in eye_aus]

    eye_py_corrs = [py_corrs[AU_COLUMNS.index(au)] for au in eye_aus if not np.isnan(py_corrs[AU_COLUMNS.index(au)])]
    non_eye_py_corrs = [py_corrs[AU_COLUMNS.index(au)] for au in non_eye_aus if not np.isnan(py_corrs[AU_COLUMNS.index(au)])]

    if eye_py_corrs:
        print(f"\nEye-related AUs (AU05, AU07, AU45): {np.mean(eye_py_corrs)*100:.2f}%")
    if non_eye_py_corrs:
        print(f"Non-eye AUs: {np.mean(non_eye_py_corrs)*100:.2f}%")

    # Save results
    cpp_df_pred.to_csv("test_cpp_landmarks_predictions.csv", index=False)
    py_df_pred.to_csv("test_python_landmarks_predictions.csv", index=False)
    print("\nResults saved to:")
    print("  - test_cpp_landmarks_predictions.csv")
    print("  - test_python_landmarks_predictions.csv")


if __name__ == "__main__":
    main()
