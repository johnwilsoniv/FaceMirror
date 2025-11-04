#!/usr/bin/env python3
"""
Comprehensive comparison test: C++ OpenFace vs PyfaceLM vs Full Python Pipeline

Tests both landmark detection approaches on challenging patient images:
- IMG_8401: Surgical markings
- IMG_9330: Severe facial paralysis
"""

import sys
import os
import subprocess
import json
import numpy as np
import cv2
from pathlib import Path

# Add paths to import our implementations
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

# Configuration
OPENFACE_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
TEST_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test")
FRAMES_DIR = TEST_DIR / "frames"
RESULTS_DIR = TEST_DIR / "results"

# Test images
TEST_IMAGES = [
    "IMG_8401.jpg",
    "IMG_9330.jpg"
]


def run_cpp_openface(image_path, output_dir):
    """Run C++ OpenFace to get ground truth landmarks."""
    print(f"\n{'='*80}")
    print(f"RUNNING C++ OPENFACE ON: {image_path.name}")
    print(f"{'='*80}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run OpenFace
    cmd = [
        OPENFACE_BINARY,
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-2Dfp"  # Output 2D landmarks
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: OpenFace failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return None

    print(f"SUCCESS: C++ OpenFace completed")

    # Parse output CSV to get landmarks
    csv_file = output_dir / f"{image_path.stem}.csv"
    if not csv_file.exists():
        print(f"ERROR: Expected output file not found: {csv_file}")
        return None

    # Read landmarks from CSV
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            print(f"ERROR: CSV file is empty or malformed")
            return None

        header = lines[0].strip().split(',')
        values = lines[1].strip().split(',')

        # Find landmark columns (x_0, y_0, x_1, y_1, ... x_67, y_67)
        landmarks = []
        for i in range(68):
            try:
                # Try both with and without leading space
                try:
                    x_idx = header.index(f'x_{i}')
                    y_idx = header.index(f'y_{i}')
                except ValueError:
                    x_idx = header.index(f' x_{i}')
                    y_idx = header.index(f' y_{i}')

                x = float(values[x_idx])
                y = float(values[y_idx])
                landmarks.append([x, y])
            except (ValueError, IndexError) as e:
                print(f"ERROR parsing landmark {i}: {e}")
                return None

        landmarks = np.array(landmarks)
        print(f"Extracted {len(landmarks)} landmarks")

        # Also get confidence if available
        try:
            try:
                conf_idx = header.index('confidence')
            except ValueError:
                conf_idx = header.index(' confidence')
            confidence = float(values[conf_idx])
            print(f"Confidence: {confidence:.3f}")
        except (ValueError, IndexError):
            confidence = None

        return landmarks, confidence


def test_pyfacelm(image_path, initial_landmarks):
    """Test PyfaceLM CLNF refinement."""
    print(f"\n{'='*80}")
    print(f"TESTING PYFACELM ON: {image_path.name}")
    print(f"{'='*80}")

    try:
        from pyfacelm import CLNFDetector

        # Initialize detector
        print("Initializing PyfaceLM CLNFDetector...")
        detector = CLNFDetector(
            model_dir=MODEL_DIR,
            max_iterations=10,
            convergence_threshold=0.01
        )

        # Load image
        image = cv2.imread(str(image_path))
        print(f"Image shape: {image.shape}")

        # Refine landmarks using CLNF
        print("Running CLNF refinement...")
        refined_landmarks, converged, num_iterations = detector.refine_landmarks(
            image,
            initial_landmarks,
            scale_idx=2,  # 0.50 scale
            regularization=0.5,
            multi_scale=True
        )

        print(f"✓ Converged: {converged}")
        print(f"✓ Iterations: {num_iterations}")

        return refined_landmarks, converged, num_iterations

    except Exception as e:
        print(f"ERROR in PyfaceLM: {e}")
        import traceback
        traceback.print_exc()
        return None, False, 0


def test_full_python_pipeline(image_path):
    """Test full Python pipeline: MTCNN + CLNF."""
    print(f"\n{'='*80}")
    print(f"TESTING FULL PYTHON PIPELINE ON: {image_path.name}")
    print(f"{'='*80}")

    try:
        from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
        from pyfaceau.clnf.clnf_detector import CLNFDetector as PyFaceAUCLNF

        # Load image
        image = cv2.imread(str(image_path))
        print(f"Image shape: {image.shape}")

        # Step 1: MTCNN face detection
        print("\n[1/3] Running MTCNN face detection...")
        mtcnn = OpenFaceMTCNN(device='cpu')
        bboxes, landmarks_5pt = mtcnn.detect(image)

        if len(bboxes) == 0:
            print("ERROR: No faces detected by MTCNN")
            return None, None

        print(f"✓ Detected {len(bboxes)} face(s)")
        bbox = bboxes[0]  # Use first face
        print(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"  Confidence: {bbox[4]:.3f}")

        # Step 2: Initialize 68 landmarks from bbox
        print("\n[2/3] Initializing 68-point landmarks from bbox...")
        initial_68pt = bbox_to_68_landmarks(bbox)
        print(f"✓ Initialized 68 landmarks")

        # Step 3: CLNF refinement
        print("\n[3/3] Running CLNF refinement...")
        clnf = PyFaceAUCLNF(
            model_dir=MODEL_DIR,
            max_iterations=10,
            convergence_threshold=0.01
        )

        refined_landmarks, converged, num_iterations = clnf.refine_landmarks(
            image,
            initial_68pt,
            scale_idx=2,
            regularization=0.5,
            multi_scale=True
        )

        print(f"✓ Converged: {converged}")
        print(f"✓ Iterations: {num_iterations}")

        return refined_landmarks, converged

    except Exception as e:
        print(f"ERROR in full Python pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def bbox_to_68_landmarks(bbox):
    """Convert bounding box to rough 68-point landmark initialization."""
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Create a rough face template (normalized coordinates)
    # This is a simplified version - just enough to test
    template = np.array([
        # Jaw (0-16)
        [-0.3, 0.3], [-0.28, 0.35], [-0.25, 0.4], [-0.2, 0.45], [-0.15, 0.48],
        [-0.1, 0.5], [-0.05, 0.51], [0, 0.52], [0.05, 0.51], [0.1, 0.5],
        [0.15, 0.48], [0.2, 0.45], [0.25, 0.4], [0.28, 0.35], [0.3, 0.3],
        [0.32, 0.25], [0.33, 0.2],
        # Eyebrows (17-26)
        [-0.25, -0.1], [-0.2, -0.12], [-0.15, -0.13], [-0.1, -0.12], [-0.05, -0.1],
        [0.05, -0.1], [0.1, -0.12], [0.15, -0.13], [0.2, -0.12], [0.25, -0.1],
        # Nose (27-35)
        [0, -0.05], [0, 0.0], [0, 0.05], [0, 0.1],
        [-0.08, 0.12], [-0.04, 0.13], [0, 0.14], [0.04, 0.13], [0.08, 0.12],
        # Eyes (36-47)
        [-0.2, -0.05], [-0.17, -0.07], [-0.14, -0.07], [-0.11, -0.05],
        [-0.14, -0.04], [-0.17, -0.04],
        [0.11, -0.05], [0.14, -0.07], [0.17, -0.07], [0.2, -0.05],
        [0.17, -0.04], [0.14, -0.04],
        # Mouth (48-67)
        [-0.12, 0.25], [-0.09, 0.27], [-0.05, 0.28], [0, 0.285], [0.05, 0.28],
        [0.09, 0.27], [0.12, 0.25], [0.1, 0.28], [0.05, 0.3], [0, 0.305],
        [-0.05, 0.3], [-0.1, 0.28], [-0.08, 0.26], [-0.05, 0.27], [0, 0.275],
        [0.05, 0.27], [0.08, 0.26], [0.05, 0.27], [0, 0.275], [-0.05, 0.27],
    ])

    # Scale and translate to fit bbox
    landmarks = template.copy()
    landmarks[:, 0] = cx + landmarks[:, 0] * w
    landmarks[:, 1] = cy + landmarks[:, 1] * h

    return landmarks


def calculate_metrics(pred_landmarks, gt_landmarks):
    """Calculate error metrics between predicted and ground truth landmarks."""
    errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)

    metrics = {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'std_error': float(np.std(errors)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors)),
        'per_landmark_errors': errors.tolist()
    }

    return metrics


def visualize_comparison(image_path, cpp_lms, pyfacelm_lms, python_pipeline_lms, output_path):
    """Create visualization comparing all three approaches."""
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    # Create three columns
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = image.copy()
    vis[:, w:2*w] = image.copy()
    vis[:, 2*w:] = image.copy()

    # Draw landmarks
    # Column 1: C++ OpenFace (green)
    for x, y in cpp_lms:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.putText(vis, "C++ OpenFace (Ground Truth)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Column 2: PyfaceLM (blue)
    if pyfacelm_lms is not None:
        for x, y in pyfacelm_lms:
            cv2.circle(vis, (int(x + w), int(y)), 2, (255, 0, 0), -1)
        cv2.putText(vis, "PyfaceLM", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Column 3: Full Python Pipeline (red)
    if python_pipeline_lms is not None:
        for x, y in python_pipeline_lms:
            cv2.circle(vis, (int(x + 2*w), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(vis, "Python MTCNN+CLNF", (2*w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save visualization
    cv2.imwrite(str(output_path), vis)
    print(f"\n✓ Saved visualization: {output_path}")

    return vis


def main():
    """Run comprehensive comparison on all test images."""
    print("\n" + "="*80)
    print("COMPREHENSIVE LANDMARK DETECTION COMPARISON")
    print("="*80)
    print(f"OpenFace Binary: {OPENFACE_BINARY}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Test Images: {len(TEST_IMAGES)}")
    print("="*80)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for img_name in TEST_IMAGES:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING: {img_name}")
        print(f"{'#'*80}")

        image_path = FRAMES_DIR / img_name
        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}")
            continue

        results = {
            'image': img_name,
            'cpp_openface': {},
            'pyfacelm': {},
            'python_pipeline': {}
        }

        # Step 1: Run C++ OpenFace (ground truth)
        cpp_output_dir = RESULTS_DIR / "cpp_openface" / image_path.stem
        cpp_result = run_cpp_openface(image_path, cpp_output_dir)

        if cpp_result is None:
            print(f"ERROR: C++ OpenFace failed, skipping {img_name}")
            continue

        cpp_landmarks, cpp_confidence = cpp_result
        results['cpp_openface']['landmarks'] = cpp_landmarks.tolist()
        results['cpp_openface']['confidence'] = cpp_confidence

        # Step 2: Test PyfaceLM (using C++ landmarks as initialization)
        pyfacelm_landmarks, pyfacelm_converged, pyfacelm_iters = test_pyfacelm(
            image_path, cpp_landmarks
        )

        if pyfacelm_landmarks is not None:
            results['pyfacelm']['landmarks'] = pyfacelm_landmarks.tolist()
            results['pyfacelm']['converged'] = pyfacelm_converged
            results['pyfacelm']['iterations'] = pyfacelm_iters

            # Calculate metrics vs C++
            pyfacelm_metrics = calculate_metrics(pyfacelm_landmarks, cpp_landmarks)
            results['pyfacelm']['metrics'] = pyfacelm_metrics

            print(f"\nPyfaceLM vs C++ OpenFace:")
            print(f"  Mean error:   {pyfacelm_metrics['mean_error']:.2f} pixels")
            print(f"  Median error: {pyfacelm_metrics['median_error']:.2f} pixels")
            print(f"  Max error:    {pyfacelm_metrics['max_error']:.2f} pixels")

        # Step 3: Test full Python pipeline
        python_pipeline_landmarks, python_pipeline_converged = test_full_python_pipeline(image_path)

        if python_pipeline_landmarks is not None:
            results['python_pipeline']['landmarks'] = python_pipeline_landmarks.tolist()
            results['python_pipeline']['converged'] = python_pipeline_converged

            # Calculate metrics vs C++
            pipeline_metrics = calculate_metrics(python_pipeline_landmarks, cpp_landmarks)
            results['python_pipeline']['metrics'] = pipeline_metrics

            print(f"\nPython Pipeline vs C++ OpenFace:")
            print(f"  Mean error:   {pipeline_metrics['mean_error']:.2f} pixels")
            print(f"  Median error: {pipeline_metrics['median_error']:.2f} pixels")
            print(f"  Max error:    {pipeline_metrics['max_error']:.2f} pixels")

        # Step 4: Create visualization
        vis_path = RESULTS_DIR / f"{image_path.stem}_comparison.jpg"
        visualize_comparison(
            image_path,
            cpp_landmarks,
            pyfacelm_landmarks,
            python_pipeline_landmarks,
            vis_path
        )

        all_results[img_name] = results

    # Save all results to JSON
    results_json = RESULTS_DIR / "comparison_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_json}")
    print(f"Visualizations saved to: {RESULTS_DIR}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for img_name, result in all_results.items():
        print(f"\n{img_name}:")

        if 'metrics' in result.get('pyfacelm', {}):
            metrics = result['pyfacelm']['metrics']
            print(f"  PyfaceLM:         {metrics['mean_error']:.2f}px mean error")

        if 'metrics' in result.get('python_pipeline', {}):
            metrics = result['python_pipeline']['metrics']
            print(f"  Python Pipeline:  {metrics['mean_error']:.2f}px mean error")


if __name__ == "__main__":
    main()
