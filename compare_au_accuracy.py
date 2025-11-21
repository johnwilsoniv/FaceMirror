#!/usr/bin/env python3
"""
Compare AU accuracy between Python pipeline and C++ OpenFace gold standard.
Includes landmark visualization and AU prediction comparison.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Optional
import json

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def process_frame_python(frame, detector, clnf, au_pipeline):
    """Process a single frame with Python pipeline."""
    result = {'success': False}

    # Detect face with MTCNN - returns (bboxes, confidences)
    detection = detector.detect(frame)

    if detection is not None and len(detection) > 0:
        # MTCNN returns tuple of (bboxes, confidences)
        if isinstance(detection, tuple) and len(detection) == 2:
            bboxes, confidences = detection
            if len(bboxes) > 0:
                # Get first face bbox [x, y, width, height]
                bbox = bboxes[0]
                # Convert to integers for bbox
                x, y, w, h = [int(v) for v in bbox]
                bbox_tuple = (x, y, w, h)

                # Fit CLNF landmarks directly with bbox
                landmarks, info = clnf.fit(frame, bbox_tuple)
                success = info.get('converged', False)

                # Use landmarks even if not fully converged, as long as we have them
                if landmarks is not None and len(landmarks) == 68:

                    # Get AU predictions using the full pipeline
                    try:
                        au_result = au_pipeline.process_frame(frame)
                        aus = au_result.get('aus', {}) if au_result else {}
                    except:
                        aus = {}

                    result = {
                        'success': True,
                        'bbox': [x, y, w, h],
                        'landmarks': landmarks,
                        'aus': aus,
                        'confidence': confidences[0] if len(confidences) > 0 else 1.0
                    }

    return result


def calculate_landmark_error(landmarks1, landmarks2):
    """Calculate mean pixel error between two sets of landmarks."""
    if landmarks1 is None or landmarks2 is None:
        return float('inf')

    # Ensure same shape
    if landmarks1.shape != landmarks2.shape:
        return float('inf')

    # Calculate Euclidean distance for each point
    distances = np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=1))
    return np.mean(distances)


def compare_au_predictions(python_aus, openface_row):
    """Compare AU predictions between Python and OpenFace."""
    au_comparison = {}

    # Standard AU list
    au_list = ['01', '02', '04', '05', '06', '07', '09', '10',
               '12', '14', '15', '17', '20', '23', '25', '26', '45']

    for au_num in au_list:
        of_intensity_key = f"AU{au_num}_r"
        of_class_key = f"AU{au_num}_c"

        if of_intensity_key in openface_row:
            # OpenFace values
            of_intensity = openface_row[of_intensity_key]
            of_class = openface_row.get(of_class_key, of_intensity > 0.5)

            # Python values
            py_key = f"au{int(au_num)}"
            py_intensity = python_aus.get(py_key, 0.0) if python_aus else 0.0
            py_class = py_intensity > 0.5

            # Calculate error
            intensity_error = abs(py_intensity - of_intensity)
            class_match = py_class == of_class

            au_comparison[f"AU{au_num}"] = {
                'python_intensity': py_intensity,
                'openface_intensity': of_intensity,
                'intensity_error': intensity_error,
                'python_class': py_class,
                'openface_class': of_class,
                'class_match': class_match
            }

    return au_comparison


def visualize_comparison(frame, python_result, openface_row, frame_num):
    """Create visualization comparing Python and OpenFace results."""
    h, w = frame.shape[:2]

    # Create side-by-side comparison
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # Python side (left)
    python_frame = frame.copy()

    # Draw Python results
    if python_result['success']:
        # Draw bbox
        if 'bbox' in python_result:
            x, y, bw, bh = python_result['bbox']
            cv2.rectangle(python_frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(python_frame, "Python MTCNN", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw landmarks
        if 'landmarks' in python_result:
            landmarks = python_result['landmarks']
            for i, (px, py) in enumerate(landmarks):
                cv2.circle(python_frame, (int(px), int(py)), 2, (0, 255, 0), -1)
                # Connect facial features
                if i > 0:
                    # Jaw line (0-16)
                    if i <= 16:
                        cv2.line(python_frame,
                                (int(landmarks[i-1][0]), int(landmarks[i-1][1])),
                                (int(px), int(py)), (0, 200, 0), 1)

    comparison[:, :w] = python_frame

    # OpenFace side (right)
    openface_frame = frame.copy()

    # Draw OpenFace landmarks
    landmarks_x = []
    landmarks_y = []
    for i in range(68):
        x_key = f'x_{i}'
        y_key = f'y_{i}'
        if x_key in openface_row and y_key in openface_row:
            landmarks_x.append(openface_row[x_key])
            landmarks_y.append(openface_row[y_key])

    if landmarks_x and landmarks_y:
        for i, (ox, oy) in enumerate(zip(landmarks_x, landmarks_y)):
            cv2.circle(openface_frame, (int(ox), int(oy)), 2, (0, 255, 255), -1)
            # Connect facial features
            if i > 0:
                # Jaw line (0-16)
                if i <= 16:
                    cv2.line(openface_frame,
                            (int(landmarks_x[i-1]), int(landmarks_y[i-1])),
                            (int(ox), int(oy)), (0, 200, 200), 1)

    comparison[:, w:] = openface_frame

    # Add labels
    cv2.putText(comparison, f"Python Pipeline - Frame {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, f"OpenFace C++ - Frame {frame_num}", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add performance info if available
    if python_result['success']:
        num_landmarks = len(python_result['landmarks']) if 'landmarks' in python_result else 0
        cv2.putText(comparison, f"Landmarks: {num_landmarks}", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.putText(comparison, f"Landmarks: {len(landmarks_x)}", (w + 10, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    return comparison


def main():
    """Run accuracy comparison between Python and OpenFace pipelines."""

    print("="*80)
    print("AU ACCURACY COMPARISON: Python vs OpenFace C++")
    print("="*80)

    # Video to process
    video_path = "Patient Data/Normal Cohort/Shorty.mov"

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Load OpenFace results (gold standard)
    openface_csv = "openface_output/Shorty.csv"
    if not Path(openface_csv).exists():
        print(f"Error: OpenFace results not found at {openface_csv}")
        print("Please run OpenFace first to generate gold standard results")
        return

    openface_df = pd.read_csv(openface_csv)
    print(f"Loaded OpenFace results: {len(openface_df)} frames")

    # Initialize Python pipeline
    print("\nInitializing Python pipeline...")

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    detector = MTCNN()
    clnf = CLNF(model_dir="pyclnf/models")
    au_pipeline = FullPythonAUPipeline(
        pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        au_models_dir="pyfaceau/weights/AU_predictors",
        triangulation_file="pyfaceau/weights/tris_68_full.txt",
        patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
        verbose=False
    )

    # Process video frames
    cap = cv2.VideoCapture(video_path)

    # Sample frames to compare (every 30th frame)
    sample_frames = list(range(0, min(180, len(openface_df)), 30))

    comparisons = []
    landmark_errors = []
    au_accuracies = []

    print(f"\nProcessing {len(sample_frames)} sample frames...")

    for frame_idx in sample_frames:
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        print(f"\nFrame {frame_idx}:")

        # Process with Python pipeline
        start_time = time.perf_counter()
        python_result = process_frame_python(frame, detector, clnf, au_pipeline)
        python_time = time.perf_counter() - start_time

        print(f"  Python processing: {python_time*1000:.1f}ms")

        # Get OpenFace results for this frame
        if frame_idx < len(openface_df):
            openface_row = openface_df.iloc[frame_idx]

            # Compare landmarks
            if python_result['success']:
                # Extract OpenFace landmarks
                of_landmarks = []
                for i in range(68):
                    if f'x_{i}' in openface_row and f'y_{i}' in openface_row:
                        of_landmarks.append([openface_row[f'x_{i}'], openface_row[f'y_{i}']])

                if of_landmarks and 'landmarks' in python_result:
                    of_landmarks = np.array(of_landmarks)
                    py_landmarks = python_result['landmarks']

                    # Calculate landmark error
                    landmark_error = calculate_landmark_error(py_landmarks, of_landmarks)
                    landmark_errors.append(landmark_error)
                    print(f"  Landmark error: {landmark_error:.2f} pixels")

                # Compare AU predictions
                au_comparison = compare_au_predictions(python_result.get('aus', {}), openface_row)

                # Calculate AU accuracy
                if au_comparison:
                    intensity_errors = [v['intensity_error'] for v in au_comparison.values()]
                    class_matches = [v['class_match'] for v in au_comparison.values()]

                    mean_intensity_error = np.mean(intensity_errors)
                    classification_accuracy = np.mean(class_matches) * 100

                    au_accuracies.append({
                        'frame': frame_idx,
                        'mean_intensity_error': mean_intensity_error,
                        'classification_accuracy': classification_accuracy
                    })

                    print(f"  AU intensity error: {mean_intensity_error:.3f}")
                    print(f"  AU classification accuracy: {classification_accuracy:.1f}%")

                # Create visualization
                comparison_img = visualize_comparison(frame, python_result, openface_row, frame_idx)

                # Save comparison image
                cv2.imwrite(f"comparison_frame_{frame_idx:04d}.jpg", comparison_img)
                print(f"  Saved comparison_frame_{frame_idx:04d}.jpg")

                comparisons.append({
                    'frame': frame_idx,
                    'python_result': python_result,
                    'au_comparison': au_comparison
                })

    cap.release()

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    if landmark_errors:
        print(f"\nLandmark Accuracy:")
        print(f"  Mean error: {np.mean(landmark_errors):.2f} pixels")
        print(f"  Std dev: {np.std(landmark_errors):.2f} pixels")
        print(f"  Min error: {np.min(landmark_errors):.2f} pixels")
        print(f"  Max error: {np.max(landmark_errors):.2f} pixels")

    if au_accuracies:
        mean_intensity_errors = [a['mean_intensity_error'] for a in au_accuracies]
        classification_accuracies = [a['classification_accuracy'] for a in au_accuracies]

        print(f"\nAU Prediction Accuracy:")
        print(f"  Mean intensity error: {np.mean(mean_intensity_errors):.3f}")
        print(f"  Mean classification accuracy: {np.mean(classification_accuracies):.1f}%")

    # Create accuracy plot
    if au_accuracies:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        frames = [a['frame'] for a in au_accuracies]
        intensity_errors = [a['mean_intensity_error'] for a in au_accuracies]
        class_accs = [a['classification_accuracy'] for a in au_accuracies]

        ax1.plot(frames, intensity_errors, 'b-', marker='o')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Mean AU Intensity Error')
        ax1.set_title('AU Intensity Error: Python vs OpenFace')
        ax1.grid(True, alpha=0.3)

        ax2.plot(frames, class_accs, 'g-', marker='s')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('AU Classification Accuracy (%)')
        ax2.set_title('AU Classification Accuracy: Python vs OpenFace')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('au_accuracy_comparison.png', dpi=150)
        print(f"\nSaved accuracy plot to au_accuracy_comparison.png")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()