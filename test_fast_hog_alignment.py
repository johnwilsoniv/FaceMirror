#!/usr/bin/env python3
"""
Fast test for HOG alignment - NO CLNF loading.

Uses C++ landmarks from CSV to test face alignment and HOG extraction.
This isolates: Is the problem landmarks, alignment, or HOG?
"""

import numpy as np
import pandas as pd
import cv2
import sys
import os

os.environ['PYCLNF_DEBUG'] = '0'

sys.path.insert(0, 'pyfaceau')
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')


def load_cpp_hog(hog_path, frame_idx):
    """Load single frame of C++ HOG.

    OpenFace HOG format:
    - Global header: num_cols, num_rows, num_channels (12 bytes)
    - Per frame (4468 floats): indicator(1) + features(4464) + per-frame-header(3)
    """
    with open(hog_path, 'rb') as f:
        num_cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_channels = np.frombuffer(f.read(4), dtype=np.int32)[0]

        feature_dim = num_cols * num_rows * num_channels  # 4464
        frame_size = 1 + feature_dim + 3  # indicator + features + per-frame header = 4468

        # Seek to frame
        f.seek(12 + frame_idx * frame_size * 4)

        # Read frame data
        frame_data = np.frombuffer(f.read(frame_size * 4), dtype=np.float32)

        # Skip indicator (first value), take features (next 4464)
        features = frame_data[1:1+feature_dim]

        return features


def get_cpp_landmarks(cpp_df, frame_idx):
    """Extract 68 landmarks from C++ CSV."""
    row = cpp_df.iloc[frame_idx]
    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']
    return landmarks


def main():
    print("=" * 60)
    print("FAST HOG/ALIGNMENT TEST (No CLNF)")
    print("=" * 60)

    video_path = "Patient Data/Normal Cohort/IMG_0441.MOV"
    cpp_csv = "test_output/video_comparison/cpp/IMG_0441.csv"
    cpp_hog_path = "test_output/video_comparison/cpp/IMG_0441.hog"

    # Load C++ data
    cpp_df = pd.read_csv(cpp_csv)
    print(f"Loaded {len(cpp_df)} frames from C++ CSV")

    # Load lightweight components only
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser
    from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
    from pyfaceau.features.triangulation import TriangulationParser
    import pyfhog

    pdm_path = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    pdm_parser = PDMParser(pdm_path)
    calc_params = CalcParams(pdm_parser)
    face_aligner = OpenFace22FaceAligner(pdm_file=pdm_path, sim_scale=0.7, output_size=(112, 112))
    triangulation = TriangulationParser("pyfaceau/weights/tris_68_full.txt")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Test multiple frames
    test_frames = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    test_frames = [f for f in test_frames if f < len(cpp_df)]

    print(f"\nTesting {len(test_frames)} frames using C++ landmarks...")
    print()

    results = []

    for frame_idx in test_frames:
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Get C++ landmarks
        cpp_landmarks = get_cpp_landmarks(cpp_df, frame_idx)

        # Get C++ HOG
        cpp_hog = load_cpp_hog(cpp_hog_path, frame_idx)

        # Use CalcParams with C++ landmarks
        params_global, params_local = calc_params.calc_params(cpp_landmarks)
        tx, ty, rz = params_global[4], params_global[5], params_global[3]

        # Align face using C++ landmarks
        py_aligned = face_aligner.align_face(
            image=frame, landmarks_68=cpp_landmarks,
            pose_tx=tx, pose_ty=ty, p_rz=rz,
            apply_mask=True, triangulation=triangulation
        )

        # Extract HOG
        py_hog_raw = pyfhog.extract_fhog_features(py_aligned, cell_size=8)
        py_hog = py_hog_raw.reshape(12, 12, 31).transpose(1, 0, 2).flatten()

        # Compare
        corr = np.corrcoef(py_hog, cpp_hog)[0, 1]
        mae = np.mean(np.abs(py_hog - cpp_hog))

        results.append((frame_idx, corr, mae))
        print(f"Frame {frame_idx:4d}: HOG corr = {corr:.4f}, MAE = {mae:.6f}")

    cap.release()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    corrs = [r[1] for r in results]
    maes = [r[2] for r in results]

    print(f"\nHOG Correlation (using C++ landmarks):")
    print(f"  Mean: {np.mean(corrs):.4f}")
    print(f"  Min:  {np.min(corrs):.4f}")
    print(f"  Max:  {np.max(corrs):.4f}")

    print(f"\nHOG MAE:")
    print(f"  Mean: {np.mean(maes):.6f}")

    if np.mean(corrs) > 0.95:
        print("\n✅ Face alignment works correctly with C++ landmarks!")
        print("   Issue is in CLNF landmark detection.")
    elif np.mean(corrs) > 0.8:
        print("\n⚠️  Alignment mostly works but has some differences.")
        print("   Check pose parameters or triangulation.")
    else:
        print("\n❌ Face alignment differs significantly from C++.")
        print("   Need to debug alignment transform.")

    # Also test: What if we use the C++ aligned images directly?
    print("\n" + "=" * 60)
    print("BONUS: Test pyfhog on C++ aligned images")
    print("=" * 60)

    cpp_aligned_dir = "test_output/video_comparison/cpp/IMG_0441_aligned"
    if os.path.exists(cpp_aligned_dir):
        for frame_idx in test_frames[:3]:
            cpp_aligned_path = f"{cpp_aligned_dir}/frame_det_00_{frame_idx+1:06d}.bmp"
            if os.path.exists(cpp_aligned_path):
                cpp_aligned = cv2.imread(cpp_aligned_path)
                cpp_hog = load_cpp_hog(cpp_hog_path, frame_idx)

                py_hog_raw = pyfhog.extract_fhog_features(cpp_aligned, cell_size=8)
                py_hog = py_hog_raw.reshape(12, 12, 31).transpose(1, 0, 2).flatten()

                corr = np.corrcoef(py_hog, cpp_hog)[0, 1]
                print(f"Frame {frame_idx}: pyfhog on C++ image → corr = {corr:.4f}")
    else:
        print("C++ aligned images not found")


if __name__ == '__main__':
    main()
