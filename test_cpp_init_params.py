#!/usr/bin/env python3
"""
Test: Use C++ initial parameters in Python to verify initialization is the root cause.
"""

import numpy as np
import cv2
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pyclnf.core.pdm import PDM

def main():
    print("=" * 80)
    print("TEST: Use C++ initial params in Python")
    print("=" * 80)

    # Setup
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 160

    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    # Save frame for C++
    frame_path = "/tmp/debug_frame.png"
    cv2.imwrite(frame_path, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run C++ first to get ground truth and initial params
    print("\n1. Running C++ OpenFace...")
    bin_path = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    cmd = [
        bin_path,
        "-f", str(frame_path),
        "-out_dir", "/tmp/cpp_debug",
        "-2Dfp",
        "-pose"
    ]
    subprocess.run(cmd, capture_output=True)

    # Read C++ ground truth landmarks
    cpp_lm_file = Path("/tmp/cpp_debug/debug_frame.csv")
    if cpp_lm_file.exists():
        import csv
        with open(cpp_lm_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpp_landmarks = np.zeros((68, 2))
                for i in range(68):
                    cpp_landmarks[i, 0] = float(row[f'x_{i}'])
                    cpp_landmarks[i, 1] = float(row[f'y_{i}'])
                break
        print(f"  C++ ground truth: landmark 36 = ({cpp_landmarks[36, 0]:.2f}, {cpp_landmarks[36, 1]:.2f})")
    else:
        print("  Error: No C++ output found")
        return

    # Read C++ initial params from debug file
    cpp_init_file = Path("/tmp/cpp_init_landmarks_68.txt")
    cpp_init_params = None
    if cpp_init_file.exists():
        with open(cpp_init_file) as f:
            content = f.read()
            # Parse params
            lines = content.split('\n')
            params = np.zeros(40)  # 6 global + 34 local
            for line in lines:
                if 'params_global[0] (scale):' in line:
                    params[0] = float(line.split(':')[1])
                elif 'params_global[1] (rot_x):' in line:
                    params[1] = float(line.split(':')[1])
                elif 'params_global[2] (rot_y):' in line:
                    params[2] = float(line.split(':')[1])
                elif 'params_global[3] (rot_z):' in line:
                    params[3] = float(line.split(':')[1])
                elif 'params_global[4] (trans_x):' in line:
                    params[4] = float(line.split(':')[1])
                elif 'params_global[5] (trans_y):' in line:
                    params[5] = float(line.split(':')[1])
                elif 'params_local[' in line:
                    idx = int(line.split('[')[1].split(']')[0])
                    val = float(line.split(':')[1])
                    params[6 + idx] = val

            cpp_init_params = params
            print(f"  C++ init params: scale={params[0]:.4f}, rot=({params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f})")
            print(f"                   trans=({params[4]:.2f}, {params[5]:.2f})")

    # Load Python model
    print("\n2. Initializing Python pyclnf...")
    MODEL_DIR = Path("pyclnf/models")
    pdm = PDM(str(MODEL_DIR / "exported_pdm"))

    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=40,
        convergence_threshold=0.5,
        regularization=25,
        window_sizes=[11, 9, 7, 5],
        debug_mode=False
    )

    # Get bbox
    bbox_file = Path("/tmp/cpp_init_bbox.txt")
    with open(bbox_file, 'r') as f:
        for line in f:
            if line.startswith('bbox:'):
                parts = line.split()[1:]
                bbox = np.array([float(p) for p in parts])
                break

    print(f"  Bbox: {bbox}")

    # Test 1: Python with Python initialization
    print("\n3. Test 1: Python with Python initialization...")
    py_init_params = pdm.init_params(bbox)
    landmarks_py_init, info_py_init = clnf.fit(gray, bbox, initial_params=py_init_params, return_params=True)

    error_py_init = np.sqrt(np.sum((landmarks_py_init - cpp_landmarks)**2, axis=1)).mean()
    print(f"  Python init params: scale={py_init_params[0]:.4f}, rot=({py_init_params[1]:.4f}, {py_init_params[2]:.4f}, {py_init_params[3]:.4f})")
    print(f"  Result: {error_py_init:.2f} px mean error")

    # Test 2: Python with C++ initialization
    if cpp_init_params is not None:
        print("\n4. Test 2: Python with C++ initialization...")
        landmarks_cpp_init, info_cpp_init = clnf.fit(gray, bbox, initial_params=cpp_init_params, return_params=True)

        error_cpp_init = np.sqrt(np.sum((landmarks_cpp_init - cpp_landmarks)**2, axis=1)).mean()
        print(f"  C++ init params: scale={cpp_init_params[0]:.4f}, rot=({cpp_init_params[1]:.4f}, {cpp_init_params[2]:.4f}, {cpp_init_params[3]:.4f})")
        print(f"  Result: {error_cpp_init:.2f} px mean error")

        # Compare
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"  Python init → Python result: {error_py_init:.2f} px")
        print(f"  C++ init → Python result:    {error_cpp_init:.2f} px")
        print(f"  C++ ground truth:            0.00 px")
        print(f"\n  Improvement: {error_py_init - error_cpp_init:.2f} px ({(error_py_init - error_cpp_init)/error_py_init*100:.1f}%)")

        if error_cpp_init < error_py_init / 2:
            print("\n  ✓ CONFIRMED: Initialization is the root cause!")
        else:
            print("\n  ⚠️  Initialization helps but is not the only issue")

if __name__ == "__main__":
    main()
