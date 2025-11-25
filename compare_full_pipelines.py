#!/usr/bin/env python3
"""
Compare Python vs C++ pipelines with their own bbox detections.
Both pipelines use their native MTCNN face detector.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import subprocess
import os
import pandas as pd

def get_cpp_landmarks(image_path: str):
    """Run C++ FeatureExtraction to get landmarks."""
    out_dir = '/tmp/openface_cpp'
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'C++ stderr: {result.stderr}')
        return None, None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(out_dir, f'{base_name}.csv')

    df = pd.read_csv(csv_path)
    landmarks = np.zeros((68, 2))
    for i in range(68):
        for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = df[x_col].iloc[0]
                landmarks[i, 1] = df[y_col].iloc[0]
                break

    # Also extract eye landmarks (28 per eye = 56 total)
    eye_landmarks = None
    if ' eye_lmk_x_0' in df.columns:
        eye_landmarks = np.zeros((56, 2))
        for i in range(56):
            eye_landmarks[i, 0] = df[f' eye_lmk_x_{i}'].iloc[0]
            eye_landmarks[i, 1] = df[f' eye_lmk_y_{i}'].iloc[0]

    return landmarks, eye_landmarks

def run_python_pipeline(image_path: str):
    """Run Python CLNF pipeline with PyMTCNN detection."""
    from pyclnf.clnf import CLNF
    from pymtcnn import MTCNN

    clnf = CLNF('pyclnf/pyclnf/models')
    image = cv2.imread(image_path)

    # Detect face with PyMTCNN (CoreML backend)
    mtcnn = MTCNN()
    bboxes, _ = mtcnn.detect(image)

    if bboxes is None or len(bboxes) == 0:
        return None, None, None

    # Get first face bbox [x, y, width, height]
    bbox = [int(bboxes[0, 0]), int(bboxes[0, 1]),
            int(bboxes[0, 2]), int(bboxes[0, 3])]

    print(f'PyMTCNN bbox: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}')

    # Fit CLNF
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks, info = clnf.fit(gray, bbox)

    return landmarks, bbox, info

def main():
    image_path = '/tmp/comparison_frame_0000.jpg'

    print('=' * 70)
    print('FULL PIPELINE COMPARISON (Each uses own detection)')
    print('=' * 70)
    print('Both pipelines use their native MTCNN face detector')
    print()

    # Get C++ landmarks (C++ uses its own MTCNN detection)
    print('Running C++ pipeline with native MTCNN...')
    cpp_landmarks, cpp_eye_landmarks = get_cpp_landmarks(image_path)
    if cpp_landmarks is None:
        print('Error: C++ failed')
        return

    # Get Python landmarks with PyMTCNN
    print()
    print('Running Python pipeline with PyMTCNN...')
    python_landmarks, py_bbox, py_info = run_python_pipeline(image_path)
    if python_landmarks is None:
        print('Error: Python failed')
        return

    # Define landmark regions
    regions = {
        'Jaw (0-16)': range(0, 17),
        'Left Eyebrow (17-21)': range(17, 22),
        'Right Eyebrow (22-26)': range(22, 27),
        'Nose (27-35)': range(27, 36),
        'Left Eye (36-41)': range(36, 42),
        'Right Eye (42-47)': range(42, 48),
        'Outer Lip (48-59)': range(48, 60),
        'Inner Lip (60-67)': range(60, 68),
    }

    print()
    print('=' * 70)
    print('LANDMARK ACCURACY BY REGION')
    print('=' * 70)

    all_errors = []
    for region_name, indices in regions.items():
        errors = []
        for i in indices:
            diff = python_landmarks[i] - cpp_landmarks[i]
            error = np.linalg.norm(diff)
            errors.append(error)
            all_errors.append(error)

        mean_err = np.mean(errors)
        max_err = np.max(errors)
        print(f'{region_name:25s}: Mean={mean_err:6.2f}px  Max={max_err:6.2f}px')

    print('-' * 70)
    overall_mean = np.mean(all_errors)
    overall_max = np.max(all_errors)
    print(f'{"ALL LANDMARKS (0-67)":25s}: Mean={overall_mean:6.2f}px  Max={overall_max:6.2f}px')
    print()

    # Detailed eye comparison
    print('=' * 70)
    print('DETAILED EYE LANDMARKS (Critical for AU detection)')
    print('=' * 70)

    print()
    print('--- Left Eye (36-41) ---')
    left_errors = []
    for i in range(36, 42):
        diff = python_landmarks[i] - cpp_landmarks[i]
        error = np.linalg.norm(diff)
        left_errors.append(error)
        print(f'  LM{i}: C++({cpp_landmarks[i,0]:7.2f}, {cpp_landmarks[i,1]:7.2f}) '
              f'Py({python_landmarks[i,0]:7.2f}, {python_landmarks[i,1]:7.2f}) '
              f'Err={error:5.2f}px')

    print()
    print('--- Right Eye (42-47) ---')
    right_errors = []
    for i in range(42, 48):
        diff = python_landmarks[i] - cpp_landmarks[i]
        error = np.linalg.norm(diff)
        right_errors.append(error)
        print(f'  LM{i}: C++({cpp_landmarks[i,0]:7.2f}, {cpp_landmarks[i,1]:7.2f}) '
              f'Py({python_landmarks[i,0]:7.2f}, {python_landmarks[i,1]:7.2f}) '
              f'Err={error:5.2f}px')

    print()
    print('=' * 70)
    print('EYE SUMMARY')
    print('=' * 70)
    print(f'Left eye mean error:  {np.mean(left_errors):.2f}px')
    print(f'Right eye mean error: {np.mean(right_errors):.2f}px')
    print(f'Combined eye error:   {np.mean(left_errors + right_errors):.2f}px')

    # Visualize
    print()
    print('Saving visualization...')
    image = cv2.imread(image_path)

    # Draw C++ landmarks (green)
    for i, pt in enumerate(cpp_landmarks):
        cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

    # Draw Python landmarks (red)
    for i, pt in enumerate(python_landmarks):
        cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    # Add legend
    cv2.putText(image, 'Green: C++  Red: Python', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite('/tmp/pipeline_comparison.jpg', image)
    print('Saved to /tmp/pipeline_comparison.jpg')

if __name__ == '__main__':
    main()
