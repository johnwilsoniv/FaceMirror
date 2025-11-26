#!/usr/bin/env python3
"""
Test full C++ postprocessing chain:
1. Two-pass re-prediction with final median
2. Cutoff-based offset subtraction (per-AU)
3. 3-frame smoothing
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

from pyfaceau.alignment.calc_params import CalcParams
from pyfaceau.features.pdm import PDMParser
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.prediction.model_parser import OF22ModelParser
from pyfaceau.features.triangulation import TriangulationParser

try:
    from cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
    print("Using Cython tracker")
except ImportError:
    from pyfaceau.features.histogram_median_tracker import DualHistogramMedianTracker
    print("Using Python tracker")

try:
    import pyfhog
except ImportError:
    pyfhog_path = SCRIPT_DIR / "pyfhog" / "src"
    if pyfhog_path.exists():
        sys.path.insert(0, str(pyfhog_path))
        import pyfhog

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0437.MOV"
CPP_RESULTS_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/img0437_cpp_results/IMG_0437.csv"
PDM_FILE = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
AU_MODELS_DIR = "pyfaceau/weights/AU_predictors"
TRIANGULATION_FILE = "pyfaceau/weights/tris_68_full.txt"

AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']


def extract_cpp_landmarks(cpp_df, frame_idx):
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
        result[au_name] = float(pred[0, 0])  # Don't clip yet - postprocessing will handle it
    return result


def apply_cutoff_offset(predictions_df, au_models):
    """
    Apply C++ cutoff-based offset subtraction.

    For each dynamic AU:
    1. Sort all prediction values
    2. Find the value at the cutoff percentile
    3. Subtract this offset from all predictions
    4. Clip to [0, 5]
    """
    result_df = predictions_df.copy()

    for au_name, model in au_models.items():
        if au_name not in result_df.columns:
            continue

        cutoff = model['cutoff']
        is_dynamic = (model['model_type'] == 'dynamic')

        if is_dynamic and cutoff > 0:
            # Sort predictions
            vals = result_df[au_name].values.copy()
            sorted_vals = np.sort(vals)

            # Find cutoff percentile value
            cutoff_idx = int(len(sorted_vals) * cutoff)
            cutoff_idx = min(cutoff_idx, len(sorted_vals) - 1)
            offset = sorted_vals[cutoff_idx]

            # Subtract offset
            result_df[au_name] = vals - offset

        # Clip to [0, 5]
        result_df[au_name] = np.clip(result_df[au_name].values, 0.0, 5.0)

    return result_df


def apply_smoothing(predictions_df, window_size=3):
    """Apply 3-frame moving average smoothing."""
    result_df = predictions_df.copy()

    for au_name in AU_COLUMNS:
        if au_name not in result_df.columns:
            continue

        vals = result_df[au_name].values.copy()
        smoothed = vals.copy()

        half_window = (window_size - 1) // 2
        for i in range(half_window, len(vals) - half_window):
            smoothed[i] = np.mean(vals[i - half_window:i + half_window + 1])

        result_df[au_name] = smoothed

    return result_df


def main():
    print("=" * 80)
    print("FULL POSTPROCESSING TEST (Two-Pass + Cutoff + Smoothing)")
    print("=" * 80)

    cpp_df = pd.read_csv(CPP_RESULTS_PATH)
    num_frames = len(cpp_df)
    print(f"Video: {num_frames} frames")

    cap = cv2.VideoCapture(VIDEO_PATH)

    pdm_parser = PDMParser(PDM_FILE)
    calc_params = CalcParams(pdm_parser)
    face_aligner = OpenFace22FaceAligner(PDM_FILE, sim_scale=0.7, output_size=(112, 112))
    triangulation = TriangulationParser(TRIANGULATION_FILE)
    au_models = OF22ModelParser(AU_MODELS_DIR).load_all_models(use_recommended=True, use_combined=True, verbose=False)

    # Print cutoff values
    print("\nAU Cutoff Values:")
    for au in AU_COLUMNS:
        if au in au_models:
            print(f"  {au}: {au_models[au]['cutoff']:.2f} ({au_models[au]['model_type']})")

    tracker = DualHistogramMedianTracker(
        hog_dim=4464, geom_dim=238,
        hog_bins=1000, hog_min=-0.005, hog_max=1.0,
        geom_bins=10000, geom_min=-60.0, geom_max=60.0
    )

    # Store features and running medians
    stored_data = []

    print(f"\nPASS 1: Processing {num_frames} frames...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            landmarks_2d = extract_cpp_landmarks(cpp_df, frame_idx)
            params_global, params_local = calc_params.calc_params(landmarks_2d)
            tx, ty, rz = params_global[4], params_global[5], params_global[3]

            aligned_face = face_aligner.align_face(
                image=frame, landmarks_68=landmarks_2d, pose_tx=tx, pose_ty=ty, p_rz=rz,
                apply_mask=True, triangulation=triangulation
            )

            hog = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
            hog = hog.reshape(12, 12, 31).transpose(1, 0, 2).flatten().astype(np.float32)
            geom = pdm_parser.extract_geometric_features(params_local).astype(np.float32)

            # Update tracker
            update_histogram = (frame_idx % 2 == 1)
            tracker.update(hog, geom, update_histogram=update_histogram)

            current_median = tracker.get_combined_median().copy()
            stored_data.append((hog.copy(), geom.copy(), current_median))

        except Exception as e:
            print(f"Frame {frame_idx} error: {e}")
            stored_data.append((None, None, None))

        frame_idx += 1

    cap.release()
    final_median = tracker.get_combined_median()

    print(f"\nComputing predictions with different postprocessing levels...")

    # 1. Single-pass (no postprocessing)
    pass1_results = []
    for i, (hog, geom, running_median) in enumerate(stored_data):
        if hog is None:
            pass1_results.append({'frame': i})
            continue
        full_vector = np.concatenate([hog, geom])
        p = predict_aus(au_models, full_vector, running_median)
        # Apply basic clipping only
        for au in AU_COLUMNS:
            if au in p:
                p[au] = np.clip(p[au], 0.0, 5.0)
        p['frame'] = i
        pass1_results.append(p)
    pass1_df = pd.DataFrame(pass1_results)

    # 2. Two-pass only (re-predict with final median)
    twopass_results = []
    for i, (hog, geom, _) in enumerate(stored_data):
        if hog is None:
            twopass_results.append({'frame': i})
            continue
        full_vector = np.concatenate([hog, geom])
        p = predict_aus(au_models, full_vector, final_median)
        for au in AU_COLUMNS:
            if au in p:
                p[au] = np.clip(p[au], 0.0, 5.0)
        p['frame'] = i
        twopass_results.append(p)
    twopass_df = pd.DataFrame(twopass_results)

    # 3. Two-pass + cutoff offset
    twopass_cutoff_results = []
    for i, (hog, geom, _) in enumerate(stored_data):
        if hog is None:
            twopass_cutoff_results.append({'frame': i})
            continue
        full_vector = np.concatenate([hog, geom])
        p = predict_aus(au_models, full_vector, final_median)
        p['frame'] = i
        twopass_cutoff_results.append(p)
    twopass_cutoff_df = pd.DataFrame(twopass_cutoff_results)
    twopass_cutoff_df = apply_cutoff_offset(twopass_cutoff_df, au_models)

    # 4. Full: Two-pass + cutoff + smoothing
    full_results = []
    for i, (hog, geom, _) in enumerate(stored_data):
        if hog is None:
            full_results.append({'frame': i})
            continue
        full_vector = np.concatenate([hog, geom])
        p = predict_aus(au_models, full_vector, final_median)
        p['frame'] = i
        full_results.append(p)
    full_df = pd.DataFrame(full_results)
    full_df = apply_cutoff_offset(full_df, au_models)
    full_df = apply_smoothing(full_df)

    print("\n" + "=" * 100)
    print("COMPARISON: Different Postprocessing Levels vs C++")
    print("=" * 100)

    print(f"\n{'AU':<10} {'Pass1':>10} {'TwoPass':>10} {'+Cutoff':>10} {'Full':>10} {'C++ Mean':>10}")
    print("-" * 60)

    p1_corrs, p2_corrs, cutoff_corrs, full_corrs = [], [], [], []

    for au in AU_COLUMNS:
        cpp_vals = cpp_df[au].values[:len(pass1_df)]

        p1_corr = np.corrcoef(pass1_df[au].values, cpp_vals)[0, 1]
        p2_corr = np.corrcoef(twopass_df[au].values, cpp_vals)[0, 1]
        cutoff_corr = np.corrcoef(twopass_cutoff_df[au].values, cpp_vals)[0, 1]
        full_corr = np.corrcoef(full_df[au].values, cpp_vals)[0, 1]

        p1_corrs.append(p1_corr)
        p2_corrs.append(p2_corr)
        cutoff_corrs.append(cutoff_corr)
        full_corrs.append(full_corr)

        cpp_mean = np.mean(cpp_vals)
        print(f"{au:<10} {p1_corr:>10.4f} {p2_corr:>10.4f} {cutoff_corr:>10.4f} {full_corr:>10.4f} {cpp_mean:>10.2f}")

    print("-" * 60)
    print(f"{'OVERALL':<10} {np.mean(p1_corrs):>10.4f} {np.mean(p2_corrs):>10.4f} {np.mean(cutoff_corrs):>10.4f} {np.mean(full_corrs):>10.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Pass 1 (single-pass):     {np.mean(p1_corrs)*100:.2f}%")
    print(f"Two-Pass:                 {np.mean(p2_corrs)*100:.2f}% ({(np.mean(p2_corrs)-np.mean(p1_corrs))*100:+.2f}%)")
    print(f"Two-Pass + Cutoff:        {np.mean(cutoff_corrs)*100:.2f}% ({(np.mean(cutoff_corrs)-np.mean(p1_corrs))*100:+.2f}%)")
    print(f"Full (+ Smoothing):       {np.mean(full_corrs)*100:.2f}% ({(np.mean(full_corrs)-np.mean(p1_corrs))*100:+.2f}%)")


if __name__ == "__main__":
    main()
