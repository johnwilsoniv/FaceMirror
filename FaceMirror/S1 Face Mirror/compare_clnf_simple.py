#!/usr/bin/env python3
"""
Simplified CLNF comparison that reuses existing test infrastructure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
import json
from clnf_debug_logger import DebugCLNFDetector, print_iteration_summary


def load_openface_landmarks(csv_path, frame_idx):
    """Load OpenFace landmarks from CSV."""
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


def compute_landmark_error(pred, gt):
    """Compute per-landmark and average L2 distance."""
    per_landmark_error = np.sqrt(np.sum((pred - gt)**2, axis=1))
    avg_error = per_landmark_error.mean()
    return per_landmark_error, avg_error


def main():
    """Simplified comparison using existing diagnose_clnf_failure.py initialization."""

    print("="*80)
    print("SIMPLIFIED CLNF COMPARISON")
    print("="*80)

    # Use existing diagnostic infrastructure
    sys.path.insert(0, str(Path(__file__).parent))
    from pyfaceau_detector import PyFaceAU68LandmarkDetector

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

            # Use existing PyFaceAU detector (RetinaFace + PFLD)
            print("\nInitializing PyFaceAU detector...")
            detector = PyFaceAU68LandmarkDetector(
                model_dir=weights_dir,
                debug_mode=False  # Quiet mode
            )

            # Get Python initialization (PFLD)
            print("\nDetecting face and initial landmarks...")
            faces, _ = detector.face_detector.detect_faces(frame)
            if len(faces) == 0:
                print("ERROR: No face detected")
                continue

            bbox = faces[0][:4]  # Extract just x1,y1,x2,y2
            python_init_landmarks, _ = detector.landmark_detector.detect_landmarks(frame, bbox)

            # Compute initialization error
            init_per_lm_error, init_avg_error = compute_landmark_error(
                python_init_landmarks,
                openface_landmarks
            )
            print(f"\nPython initialization error vs OpenFace:")
            print(f"  Average: {init_avg_error:.2f} pixels")
            print(f"  Max: {init_per_lm_error.max():.2f} pixels (landmark {init_per_lm_error.argmax()})")

            # Run Python CLNF with debug logging
            print("\nRunning Python CLNF with debug logging...")
            clnf = DebugCLNFDetector(
                model_dir=weights_dir / 'clnf',
                max_iterations=10,
                convergence_threshold=0.01
            )

            python_clnf_landmarks, converged, num_iters = clnf.refine_landmarks(
                frame,
                python_init_landmarks,
                scale_idx=2,  # 0.50 scale
                regularization=0.5,
                multi_scale=False
            )

            # Compute CLNF error
            clnf_per_lm_error, clnf_avg_error = compute_landmark_error(
                python_clnf_landmarks,
                openface_landmarks
            )
            print(f"\nPython CLNF error vs OpenFace:")
            print(f"  Average: {clnf_avg_error:.2f} pixels")
            print(f"  Max: {clnf_per_lm_error.max():.2f} pixels (landmark {clnf_per_lm_error.argmax()})")
            print(f"  Converged: {converged}, Iterations: {num_iters}")

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

            debug_history = clnf.get_debug_history()
            for i in range(min(len(debug_history), 5)):  # Print first 5 iterations
                print_iteration_summary(debug_history, i)

            # Save diagnostic report
            report_path = output_dir / f"{test['name']}_frame{test['frame']}_diagnostic.json"
            report = {
                'test_info': {
                    'name': test['name'],
                    'frame': test['frame'],
                },
                'errors': {
                    'init_avg_error': float(init_avg_error),
                    'clnf_avg_error': float(clnf_avg_error),
                    'improvement': float(improvement),
                    'converged': bool(converged),
                    'num_iterations': int(num_iters),
                },
                'iterations': []
            }

            # Save iteration data
            for log in debug_history:
                iteration_data = {
                    'iteration': int(log['iteration']),
                    'avg_movement': float(log['avg_movement']),
                    'converged': bool(log['converged']),
                }
                if 'residual_norm' in log:
                    iteration_data['residual_norm'] = float(log['residual_norm'])
                if 'delta_params_norm' in log:
                    iteration_data['delta_params_norm'] = float(log['delta_params_norm'])
                if 'jacobian_condition' in log:
                    iteration_data['jacobian_condition'] = float(log['jacobian_condition'])
                report['iterations'].append(iteration_data)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\nSaved diagnostic report to: {report_path}")

            # Visualize
            vis = frame.copy()
            for x, y in python_init_landmarks:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)  # RED: Python init
            for x, y in python_clnf_landmarks:
                cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)  # BLUE: Python CLNF
            for x, y in openface_landmarks:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)  # GREEN: OpenFace

            cv2.putText(vis, "Red: Python Init | Blue: Python CLNF | Green: OpenFace",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            vis_path = output_dir / f"{test['name']}_frame{test['frame']}_comparison.jpg"
            cv2.imwrite(str(vis_path), vis)
            print(f"Saved visualization to: {vis_path}")

        except Exception as e:
            print(f"\nERROR processing {test['name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
