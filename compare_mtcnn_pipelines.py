#!/usr/bin/env python3
"""
Compare PyMTCNN Pipeline vs C++ OpenFace MTCNN Pipeline
Creates visual comparison with blue boxes (Python) and green boxes (C++)
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from cpp_mtcnn_detector import CPPMTCNNDetector

def parse_cpp_openface_output(output_csv):
    """Parse C++ OpenFace CSV output to extract face bboxes."""
    import pandas as pd

    # Read the CSV
    df = pd.read_csv(output_csv)

    # C++ OpenFace outputs x, y, w, h in pixels
    # Column names might be like: face_0_x, face_0_y, face_0_w, face_0_h
    # Or: x_0, y_0, w_0, h_0

    bboxes = []

    # Check for face detection columns
    # OpenFace typically outputs confidence and bbox for detected faces
    if ' confidence' in df.columns or 'confidence' in df.columns:
        # Has face detection info
        conf_col = ' confidence' if ' confidence' in df.columns else 'confidence'
        confidence = df[conf_col].iloc[0]

        if confidence > 0.5:  # Face detected
            # Try to find bbox columns
            x_col = None
            y_col = None
            w_col = None
            h_col = None

            for col in df.columns:
                col_lower = col.lower().strip()
                if 'face_0_x' in col_lower or col_lower == 'x_0':
                    x_col = col
                elif 'face_0_y' in col_lower or col_lower == 'y_0':
                    y_col = col
                elif 'face_0_w' in col_lower or col_lower == 'w_0':
                    w_col = col
                elif 'face_0_h' in col_lower or col_lower == 'h_0':
                    h_col = col

            if x_col and y_col and w_col and h_col:
                x = df[x_col].iloc[0]
                y = df[y_col].iloc[0]
                w = df[w_col].iloc[0]
                h = df[h_col].iloc[0]
                bboxes.append([x, y, w, h])

    return np.array(bboxes) if bboxes else np.array([])


def run_cpp_openface_mtcnn(image_path):
    """Run C++ OpenFace on image and extract MTCNN bboxes."""

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Run C++ OpenFace FeatureExtraction
        cpp_openface_bin = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

        cmd = [
            str(cpp_openface_bin),
            '-f', str(image_path),
            '-out_dir', str(temp_dir),
            '-verbose'
        ]

        print(f"Running C++ OpenFace MTCNN...")
        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR running C++ OpenFace:")
            print(result.stderr)
            return np.array([])

        # Parse output CSV to get bbox
        csv_files = list(temp_dir.glob('*.csv'))
        if not csv_files:
            print(f"No CSV output found")
            return np.array([])

        bboxes = parse_cpp_openface_output(csv_files[0])

        return bboxes


def extract_mtcnn_bbox_from_cpp_output(image_path):
    """
    Alternative: Run C++ with MTCNN debug output enabled.
    This assumes we've added debug logging to save MTCNN bboxes.
    """

    # For now, let's use a simpler approach:
    # Run C++ OpenFace and parse the detected face region
    # which comes from MTCNN internally

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        cpp_openface_bin = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

        cmd = [
            str(cpp_openface_bin),
            '-f', str(image_path),
            '-out_dir', str(temp_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse stdout for MTCNN bbox info (if we added debug logging)
        # For now, return empty since we need to add debug output

        # Check if there's a saved bbox file
        bbox_file = Path('/tmp/cpp_mtcnn_final_bbox.txt')
        if bbox_file.exists():
            bboxes = []
            with open(bbox_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, w, h = map(float, parts[:4])
                            bboxes.append([x, y, w, h])
            return np.array(bboxes)

        return np.array([])


def draw_comparison(image, py_bboxes, py_landmarks, cpp_bboxes, cpp_landmarks=None):
    """
    Draw comparison visualization.

    Args:
        image: Input image
        py_bboxes: Python MTCNN bboxes (N, 4) - [x, y, w, h]
        py_landmarks: Python MTCNN landmarks (N, 10) - [x1, y1, ..., x5, y5]
        cpp_bboxes: C++ MTCNN bboxes (N, 4) - [x, y, w, h]
        cpp_landmarks: C++ MTCNN landmarks (N, 10)
    """

    vis = image.copy()

    # Draw C++ bboxes in GREEN
    if len(cpp_bboxes) > 0:
        for bbox in cpp_bboxes:
            x, y, w, h = bbox.astype(int)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Green
            cv2.putText(vis, 'C++ MTCNN', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw Python bboxes in BLUE
    if len(py_bboxes) > 0:
        for bbox in py_bboxes:
            x, y, w, h = bbox.astype(int)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue
            cv2.putText(vis, 'Py MTCNN', (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw Python landmarks in RED (like the image example)
    if len(py_landmarks) > 0:
        for lm in py_landmarks:
            for i in range(5):
                x = int(lm[i*2])
                y = int(lm[i*2 + 1])
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # Red dots

    return vis


def main():
    """Compare PyMTCNN vs C++ OpenFace MTCNN."""

    print("="*80)
    print("COMPARING PyMTCNN vs C++ OpenFace MTCNN")
    print("="*80)

    # Test image
    test_image = 'cpp_mtcnn_test.jpg'

    if not os.path.exists(test_image):
        print(f"ERROR: Test image not found: {test_image}")
        # Try alternate paths
        alt_paths = [
            'calibration_frames/patient1_frame1.jpg',
            'Patient Data/Normal Cohort/IMG_0434.MOV'  # Frame 0
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                test_image = alt
                break

    print(f"\nTest image: {test_image}")

    # Load image
    img = cv2.imread(test_image)
    if img is None:
        print(f"ERROR: Could not load image: {test_image}")
        return

    print(f"Image shape: {img.shape}")

    # 1. Run Python MTCNN (our fixed implementation)
    print(f"\n{'='*80}")
    print(f"Running Python MTCNN (BGR-Fixed)...")
    print(f"{'='*80}")

    detector = CPPMTCNNDetector()
    py_bboxes, py_landmarks = detector.detect(img)

    print(f"\nPython MTCNN Results:")
    print(f"  Detected {len(py_bboxes)} faces")
    for i, (bbox, lm) in enumerate(zip(py_bboxes, py_landmarks)):
        x, y, w, h = bbox
        print(f"  Face {i+1}: ({x:.1f}, {y:.1f}, {w:.1f}x{h:.1f})")

    # 2. Run C++ OpenFace MTCNN
    print(f"\n{'='*80}")
    print(f"Running C++ OpenFace MTCNN...")
    print(f"{'='*80}")

    # For the comparison, let's use a known test image that we've already
    # processed with C++ and saved the output

    # Check if we have pre-saved C++ results
    cpp_bbox_file = Path('/tmp/cpp_mtcnn_final_bbox.txt')

    if not cpp_bbox_file.exists():
        # Run C++ to generate the output
        print(f"\nNo cached C++ results found. Running C++ OpenFace...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            cpp_openface_bin = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

            cmd = [
                str(cpp_openface_bin),
                '-f', test_image,
                '-out_dir', str(temp_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Check if bbox was saved
                if cpp_bbox_file.exists():
                    print(f"✓ C++ MTCNN bbox saved to {cpp_bbox_file}")
                else:
                    print(f"⚠ C++ ran successfully but no bbox file saved")
                    print(f"  You may need to add debug logging to C++ to save MTCNN bboxes")

    # Load C++ bboxes if available
    cpp_bboxes = []
    if cpp_bbox_file.exists():
        with open(cpp_bbox_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, w, h = map(float, parts[:4])
                        cpp_bboxes.append([x, y, w, h])
        cpp_bboxes = np.array(cpp_bboxes)

        print(f"\nC++ MTCNN Results:")
        print(f"  Detected {len(cpp_bboxes)} faces")
        for i, bbox in enumerate(cpp_bboxes):
            x, y, w, h = bbox
            print(f"  Face {i+1}: ({x:.1f}, {y:.1f}, {w:.1f}x{h:.1f})")
    else:
        print(f"\n⚠ No C++ MTCNN results available for comparison")
        print(f"  Creating visualization with Python results only...")
        cpp_bboxes = np.array([])

    # 3. Create comparison visualization
    print(f"\n{'='*80}")
    print(f"Creating Comparison Visualization...")
    print(f"{'='*80}")

    vis = draw_comparison(img, py_bboxes, py_landmarks, cpp_bboxes)

    # Save visualization
    output_path = 'mtcnn_pipeline_comparison.jpg'
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Saved comparison to: {output_path}")

    # Also create a side-by-side comparison
    if len(cpp_bboxes) > 0:
        # Calculate bbox differences
        print(f"\n{'='*80}")
        print(f"BBOX COMPARISON:")
        print(f"{'='*80}")

        if len(py_bboxes) == len(cpp_bboxes):
            for i, (py_bbox, cpp_bbox) in enumerate(zip(py_bboxes, cpp_bboxes)):
                diff = np.abs(py_bbox - cpp_bbox)
                print(f"\nFace {i+1}:")
                print(f"  Python: ({py_bbox[0]:.1f}, {py_bbox[1]:.1f}, {py_bbox[2]:.1f}x{py_bbox[3]:.1f})")
                print(f"  C++:    ({cpp_bbox[0]:.1f}, {cpp_bbox[1]:.1f}, {cpp_bbox[2]:.1f}x{cpp_bbox[3]:.1f})")
                print(f"  Diff:   ({diff[0]:.1f}, {diff[1]:.1f}, {diff[2]:.1f}x{diff[3]:.1f})")
                print(f"  Max diff: {diff.max():.1f} pixels")
        else:
            print(f"Different number of faces detected!")
            print(f"  Python: {len(py_bboxes)} faces")
            print(f"  C++:    {len(cpp_bboxes)} faces")

    print(f"\n{'='*80}")
    print(f"✅ COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"\nVisualization saved to: {output_path}")
    print(f"  - Green boxes: C++ OpenFace MTCNN")
    print(f"  - Blue boxes:  Python MTCNN (BGR-Fixed)")
    print(f"  - Red dots:    Python landmarks")


if __name__ == '__main__':
    main()
