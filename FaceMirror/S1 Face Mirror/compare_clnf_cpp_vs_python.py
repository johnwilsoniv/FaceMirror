#!/usr/bin/env python3
"""
Compare C++ OpenFace CLNF vs Python CLNF step-by-step.

This script:
1. Loads OpenFace C++ landmarks as "ground truth"
2. Extracts MTCNN/FAN initialization from Python detector
3. Runs Python CLNF with debug logging
4. Compares iteration-by-iteration to identify divergence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
import json
from clnf_debug_logger import DebugCLNFDetector, print_iteration_summary, compare_iterations
from pyfaceau.detectors import ONNXRetinaFaceDetector, CunjianPFLDDetector


def load_openface_landmarks(csv_path, frame_idx):
    """
    Load OpenFace landmarks from CSV.

    Args:
        csv_path: Path to OpenFace CSV output
        frame_idx: Frame index (0-based)

    Returns:
        landmarks: 68-point landmarks (68, 2)
        metadata: OpenFace metadata dict
    """
    df = pd.read_csv(csv_path)
    row = df[df['frame'] == frame_idx + 1].iloc[0]

    landmarks = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']

    metadata = {
        'confidence': row['confidence'],
        'success': row['success'],
    }

    return landmarks, metadata


def extract_python_initialization(frame, weights_dir):
    """
    Extract initial landmarks using Python detector (RetinaFace + PFLD).

    Args:
        frame: BGR image (H, W, 3)
        weights_dir: Path to weights directory

    Returns:
        initial_landmarks: 68-point landmarks from PFLD (68, 2)
        bbox: Face bounding box [x1, y1, x2, y2]
    """
    # Initialize detectors
    face_detector = ONNXRetinaFaceDetector(
        onnx_model_path=str(weights_dir / 'retinaface_mobilenet025_coreml.onnx'),
        use_coreml=True
    )

    landmark_detector = CunjianPFLDDetector(
        model_path=str(weights_dir / 'pfld_cunjian.onnx'),
        use_coreml=True
    )

    # Detect face
    bboxes, kpts, scores = face_detector.detect_faces(frame)

    if len(bboxes) == 0:
        raise RuntimeError("No face detected")

    bbox = bboxes[0]

    # Detect landmarks
    landmarks, confidence = landmark_detector.detect_landmarks(frame, bbox)

    print(f"\nPython initialization:")
    print(f"  RetinaFace bbox: {bbox}")
    print(f"  PFLD confidence: {confidence:.3f}")
    print(f"  Landmarks range: X=[{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}], "
          f"Y=[{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")

    return landmarks, bbox


def run_python_clnf(frame, initial_landmarks, weights_dir, scale_idx=2, regularization=0.5):
    """
    Run Python CLNF with debug logging.

    Args:
        frame: BGR image (H, W, 3)
        initial_landmarks: Initial 68-point landmarks (68, 2)
        weights_dir: Path to weights directory
        scale_idx: CEN scale index (0-3)
        regularization: Regularization weight

    Returns:
        refined_landmarks: Refined landmarks (68, 2)
        converged: Whether optimization converged
        num_iterations: Number of iterations
        debug_history: List of iteration logs
    """
    # Initialize CLNF with debug logging (use clnf subdirectory)
    clnf = DebugCLNFDetector(
        model_dir=weights_dir / 'clnf',
        max_iterations=10,
        convergence_threshold=0.01
    )

    # Run CLNF refinement
    print(f"\nRunning Python CLNF (scale_idx={scale_idx}, regularization={regularization})...")
    refined_landmarks, converged, num_iterations = clnf.refine_landmarks(
        frame,
        initial_landmarks,
        scale_idx=scale_idx,
        regularization=regularization,
        multi_scale=False  # Single scale for comparison
    )

    print(f"  Converged: {converged}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Final landmarks range: X=[{refined_landmarks[:, 0].min():.1f}, {refined_landmarks[:, 0].max():.1f}], "
          f"Y=[{refined_landmarks[:, 1].min():.1f}, {refined_landmarks[:, 1].max():.1f}]")

    return refined_landmarks, converged, num_iterations, clnf.get_debug_history()


def compute_landmark_error(pred, gt):
    """
    Compute per-landmark and average L2 distance.

    Args:
        pred: Predicted landmarks (68, 2)
        gt: Ground truth landmarks (68, 2)

    Returns:
        per_landmark_error: Error for each landmark (68,)
        avg_error: Average error across all landmarks
    """
    per_landmark_error = np.sqrt(np.sum((pred - gt)**2, axis=1))
    avg_error = per_landmark_error.mean()
    return per_landmark_error, avg_error


def visualize_comparison(frame, python_init, python_clnf, openface_landmarks, save_path):
    """
    Visualize all three sets of landmarks.

    Args:
        frame: BGR image
        python_init: Python initialization landmarks (68, 2)
        python_clnf: Python CLNF refined landmarks (68, 2)
        openface_landmarks: OpenFace C++ landmarks (68, 2)
        save_path: Where to save visualization
    """
    vis = frame.copy()

    # Draw landmarks with different colors
    # Python init: RED
    for x, y in python_init:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)

    # Python CLNF: BLUE
    for x, y in python_clnf:
        cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

    # OpenFace C++: GREEN
    for x, y in openface_landmarks:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Add legend
    cv2.putText(vis, "Red: Python Init (PFLD)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, "Blue: Python CLNF", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(vis, "Green: OpenFace C++", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(save_path), vis)
    print(f"\nSaved visualization to: {save_path}")


def save_diagnostic_report(output_path, test_info, errors, debug_history):
    """
    Save detailed diagnostic report as JSON.

    Args:
        output_path: Where to save JSON report
        test_info: Dict with test metadata
        errors: Dict with error metrics
        debug_history: CLNF debug history
    """
    report = {
        'test_info': test_info,
        'errors': errors,
        'iterations': []
    }

    # Convert debug history to serializable format
    for log in debug_history:
        iteration_data = {
            'iteration': int(log['iteration']),
            'avg_movement': float(log['avg_movement']),
            'converged': bool(log['converged']),
            'scale': float(log['scale']),
            'translation': log['translation'].tolist(),
            'params': log['params'].tolist(),
            'landmarks': log['landmarks'].tolist(),
        }

        if 'residual_norm' in log:
            iteration_data['residual_norm'] = float(log['residual_norm'])

        if 'delta_params_norm' in log:
            iteration_data['delta_params_norm'] = float(log['delta_params_norm'])

        if 'jacobian_condition' in log:
            iteration_data['jacobian_condition'] = float(log['jacobian_condition'])

        if 'response_stats' in log:
            # Summarize response stats
            stats = log['response_stats']
            all_means = [s['mean'] for s in stats if s['mean'] > 0]
            all_maxs = [s['max'] for s in stats if s['max'] > 0]
            iteration_data['response_mean_avg'] = float(np.mean(all_means)) if all_means else 0.0
            iteration_data['response_max_avg'] = float(np.mean(all_maxs)) if all_maxs else 0.0

        report['iterations'].append(iteration_data)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Saved diagnostic report to: {output_path}")


def main():
    """Main comparison workflow."""

    print("="*80)
    print("C++ OPENFACE CLNF vs PYTHON CLNF COMPARISON")
    print("="*80)

    # Configuration
    weights_dir = Path(__file__).parent / 'weights'
    output_dir = Path('/tmp/clnf_diagnostic_results')
    output_dir.mkdir(exist_ok=True)

    test_cases = [
        {
            'name': 'IMG_9330',
            'video': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV',
            'openface_csv': '/tmp/openface_test_9330_rotated/IMG_9330_source.csv',
            'frame': 100,
        },
        {
            'name': 'IMG_8401',
            'video': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV',
            'openface_csv': '/tmp/openface_test_8401_rotated/IMG_8401_source.csv',
            'frame': 100,
        },
    ]

    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: {test['name']} frame {test['frame']}")
        print(f"{'='*80}")

        try:
            # Load frame
            cap = cv2.VideoCapture(test['video'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, test['frame'])
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"ERROR: Could not read frame {test['frame']}")
                continue

            # Load OpenFace C++ landmarks (ground truth)
            print("\nLoading OpenFace C++ landmarks...")
            openface_landmarks, openface_meta = load_openface_landmarks(
                test['openface_csv'],
                test['frame']
            )
            print(f"  OpenFace confidence: {openface_meta['confidence']:.3f}")
            print(f"  OpenFace success: {openface_meta['success']}")

            # Extract Python initialization
            python_init_landmarks, bbox = extract_python_initialization(frame, weights_dir)

            # Compute initialization error
            init_per_lm_error, init_avg_error = compute_landmark_error(
                python_init_landmarks,
                openface_landmarks
            )
            print(f"\nPython initialization error vs OpenFace:")
            print(f"  Average: {init_avg_error:.2f} pixels")
            print(f"  Max: {init_per_lm_error.max():.2f} pixels (landmark {init_per_lm_error.argmax()})")
            print(f"  Median: {np.median(init_per_lm_error):.2f} pixels")

            # Run Python CLNF
            python_clnf_landmarks, converged, num_iters, debug_history = run_python_clnf(
                frame,
                python_init_landmarks,
                weights_dir,
                scale_idx=2,  # 0.50 scale (OpenFace default)
                regularization=0.5
            )

            # Compute CLNF error
            clnf_per_lm_error, clnf_avg_error = compute_landmark_error(
                python_clnf_landmarks,
                openface_landmarks
            )
            print(f"\nPython CLNF error vs OpenFace:")
            print(f"  Average: {clnf_avg_error:.2f} pixels")
            print(f"  Max: {clnf_per_lm_error.max():.2f} pixels (landmark {clnf_per_lm_error.argmax()})")
            print(f"  Median: {np.median(clnf_per_lm_error):.2f} pixels")

            # Check if CLNF improved or degraded
            improvement = init_avg_error - clnf_avg_error
            if improvement > 0:
                print(f"\n✅ Python CLNF IMPROVED by {improvement:.2f} pixels ({improvement/init_avg_error*100:.1f}%)")
            else:
                degradation = -improvement
                print(f"\n❌ Python CLNF DEGRADED by {degradation:.2f} pixels ({degradation/init_avg_error*100:.1f}%)")

            # Print iteration summaries
            print(f"\n{'='*80}")
            print("ITERATION-BY-ITERATION ANALYSIS")
            print(f"{'='*80}")

            for i in range(len(debug_history)):
                print_iteration_summary(debug_history, i)

            # Visualize comparison
            vis_path = output_dir / f"{test['name']}_frame{test['frame']}_comparison.jpg"
            visualize_comparison(
                frame,
                python_init_landmarks,
                python_clnf_landmarks,
                openface_landmarks,
                vis_path
            )

            # Save diagnostic report
            report_path = output_dir / f"{test['name']}_frame{test['frame']}_diagnostic.json"
            save_diagnostic_report(
                report_path,
                test_info={
                    'name': test['name'],
                    'frame': test['frame'],
                    'video': test['video'],
                },
                errors={
                    'init_avg_error': float(init_avg_error),
                    'init_max_error': float(init_per_lm_error.max()),
                    'clnf_avg_error': float(clnf_avg_error),
                    'clnf_max_error': float(clnf_per_lm_error.max()),
                    'improvement': float(improvement),
                    'converged': bool(converged),
                    'num_iterations': int(num_iters),
                },
                debug_history=debug_history
            )

        except Exception as e:
            print(f"\nERROR processing {test['name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print("\nKey findings:")
    print("1. Check visualization images to see landmark differences")
    print("2. Review JSON diagnostic files for detailed iteration data")
    print("3. Look for divergence in response maps, jacobian, or parameter updates")


if __name__ == '__main__':
    main()
