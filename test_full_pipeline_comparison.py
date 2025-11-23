#!/usr/bin/env python3
"""
Run full Python pipeline and compare with C++ output.
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
        return None
    
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
    return landmarks

def run_python_pipeline(image_path: str):
    """Run Python CLNF pipeline."""
    from pyclnf.clnf import CLNF
    from pymtcnn import MTCNN
    
    clnf = CLNF('pyclnf/models')
    image = cv2.imread(image_path)
    
    # Detect face with MTCNN
    mtcnn = MTCNN()
    bboxes, _ = mtcnn.detect(image)
    
    if bboxes is None or len(bboxes) == 0:
        return None
    
    # Get first face bbox [x, y, width, height]
    bbox = [int(bboxes[0, 0]), int(bboxes[0, 1]), 
            int(bboxes[0, 2]), int(bboxes[0, 3])]
    
    # Fit CLNF
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks, _ = clnf.fit(gray, bbox)
    
    return landmarks

def main():
    image_path = 'comparison_frame_0000.jpg'
    
    print("=== Full Pipeline Comparison ===\n")
    
    # Get C++ landmarks
    print("Running C++ pipeline...")
    cpp_landmarks = get_cpp_landmarks(image_path)
    if cpp_landmarks is None:
        print("Error: C++ failed")
        return
    
    # Get Python landmarks  
    print("Running Python pipeline...")
    python_landmarks = run_python_pipeline(image_path)
    if python_landmarks is None:
        print("Error: Python failed")
        return
    
    # Compare eye landmarks
    print("\n=== Left Eye Comparison (36-41) ===")
    left_errors = []
    for i in range(36, 42):
        diff = python_landmarks[i] - cpp_landmarks[i]
        error = np.linalg.norm(diff)
        left_errors.append(error)
        print(f"LM{i}: C++({cpp_landmarks[i,0]:.2f}, {cpp_landmarks[i,1]:.2f}) "
              f"Py({python_landmarks[i,0]:.2f}, {python_landmarks[i,1]:.2f}) "
              f"Err={error:.2f}px")
    
    print("\n=== Right Eye Comparison (42-47) ===")
    right_errors = []
    for i in range(42, 48):
        diff = python_landmarks[i] - cpp_landmarks[i]
        error = np.linalg.norm(diff)
        right_errors.append(error)
        print(f"LM{i}: C++({cpp_landmarks[i,0]:.2f}, {cpp_landmarks[i,1]:.2f}) "
              f"Py({python_landmarks[i,0]:.2f}, {python_landmarks[i,1]:.2f}) "
              f"Err={error:.2f}px")
    
    print(f"\nLeft eye mean error: {np.mean(left_errors):.3f}px")
    print(f"Right eye mean error: {np.mean(right_errors):.3f}px")
    print(f"Ratio (left/right): {np.mean(left_errors)/np.mean(right_errors):.2f}x")

if __name__ == '__main__':
    main()
