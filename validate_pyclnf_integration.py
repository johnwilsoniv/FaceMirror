#!/usr/bin/env python3
"""
Validate PyFaceAU pipeline with pyclnf integration.

Tests:
1. PyMTCNN face detection
2. Initial landmark initialization (PDM mean shape → bbox)
3. CLNF landmark refinement with pyclnf
4. Comparison to C++ OpenFace reference landmarks
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau.pipeline import FullPythonAUPipeline


def load_cpp_reference_landmarks(frame_idx=0):
    """Load C++ OpenFace reference landmarks for comparison."""
    # Load from cpp_reference directory
    cpp_dir = Path("cpp_reference")
    landmark_file = cpp_dir / f"frame_{frame_idx:05d}.csv"

    if not landmark_file.exists():
        print(f"Warning: C++ reference not found: {landmark_file}")
        return None

    # Parse OpenFace CSV (skip header, extract x_0...x_67, y_0...y_67)
    import pandas as pd
    df = pd.read_csv(landmark_file)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[f'x_{i}'].iloc[0]
        landmarks[i, 1] = df[f'y_{i}'].iloc[0]

    return landmarks


def compute_landmark_error(landmarks_python, landmarks_cpp):
    """Compute mean Euclidean error between landmarks."""
    if landmarks_cpp is None:
        return None

    errors = np.sqrt(np.sum((landmarks_python - landmarks_cpp) ** 2, axis=1))
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return {
        'mean': mean_error,
        'max': max_error,
        'errors': errors
    }


def visualize_comparison(frame, landmarks_python, landmarks_cpp, bbox, save_path):
    """Visualize Python vs C++ landmarks side by side."""
    vis = frame.copy()

    # Draw bbox
    x, y, w, h = bbox
    cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 2)

    # Draw Python landmarks (green)
    for x, y in landmarks_python:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Draw C++ landmarks (red) if available
    if landmarks_cpp is not None:
        for x, y in landmarks_cpp:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Add legend
    cv2.putText(vis, "Green: Python (pyclnf)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if landmarks_cpp is not None:
        cv2.putText(vis, "Red: C++ OpenFace", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(save_path, vis)
    print(f"Saved visualization: {save_path}")


def main():
    print("=" * 80)
    print("PyFaceAU Pipeline Validation with PyCLNF Integration")
    print("=" * 80)

    # 1. Initialize pipeline
    print("\n[1/5] Initializing PyFaceAU pipeline...")
    try:
        pipeline = FullPythonAUPipeline(
            model_dir="weights",
            detector="pymtcnn",
            use_cuda=False,
            verbose=True
        )
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load test video
    print("\n[2/5] Loading test video...")
    video_path = "calibration_frames/patient1_trial1_frame_000000.jpg"

    if not Path(video_path).exists():
        print(f"✗ Test frame not found: {video_path}")
        return

    frame = cv2.imread(video_path)
    print(f"✓ Loaded frame: {frame.shape}")

    # 3. Run face detection
    print("\n[3/5] Running face detection (PyMTCNN)...")
    try:
        # Detect face
        detections, _ = pipeline.face_detector.detect_faces(frame)

        if len(detections) == 0:
            print("✗ No faces detected")
            return

        # Get first face bbox
        det = detections[0]
        bbox_xyxy = det[:4].astype(int)
        confidence = det[4]

        print(f"✓ Face detected:")
        print(f"  Bbox (x1,y1,x2,y2): {bbox_xyxy}")
        print(f"  Confidence: {confidence:.3f}")

        # Convert to (x, y, w, h) for pyclnf
        bbox_x = bbox_xyxy[0]
        bbox_y = bbox_xyxy[1]
        bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
        bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
        bbox_xywh = (bbox_x, bbox_y, bbox_w, bbox_h)

        print(f"  Bbox (x,y,w,h): {bbox_xywh}")

    except Exception as e:
        print(f"✗ Face detection failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Run CLNF landmark detection
    print("\n[4/5] Running CLNF landmark detection (pyclnf)...")
    try:
        # Call pyclnf directly
        landmarks_68, info = pipeline.landmark_detector.fit(frame, bbox_xywh)

        print(f"✓ CLNF landmarks detected:")
        print(f"  Landmarks: {landmarks_68.shape}")
        print(f"  Converged: {info['converged']}")
        print(f"  Iterations: {info['iterations']}")
        print(f"  Final update: {info.get('final_update', 'N/A')}")

        # Show sample landmarks
        print(f"\n  Sample landmarks:")
        for i in [0, 30, 36, 48, 60]:  # Jaw, nose, eye, mouth, brow
            print(f"    Landmark {i:2d}: ({landmarks_68[i, 0]:7.2f}, {landmarks_68[i, 1]:7.2f})")

    except Exception as e:
        print(f"✗ CLNF landmark detection failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Compare to C++ OpenFace reference
    print("\n[5/5] Comparing to C++ OpenFace reference...")
    try:
        landmarks_cpp = load_cpp_reference_landmarks(frame_idx=0)

        if landmarks_cpp is not None:
            error_stats = compute_landmark_error(landmarks_68, landmarks_cpp)

            print(f"✓ Comparison complete:")
            print(f"  Mean error: {error_stats['mean']:.3f} pixels")
            print(f"  Max error:  {error_stats['max']:.3f} pixels")

            # Show per-landmark errors for outliers
            outliers = np.where(error_stats['errors'] > 10.0)[0]
            if len(outliers) > 0:
                print(f"\n  Outliers (>10px error):")
                for idx in outliers[:5]:  # Show first 5
                    err = error_stats['errors'][idx]
                    py_pt = landmarks_68[idx]
                    cpp_pt = landmarks_cpp[idx]
                    print(f"    Landmark {idx:2d}: {err:.2f}px "
                          f"(Python: {py_pt}, C++: {cpp_pt})")

            # Visualize
            visualize_comparison(
                frame, landmarks_68, landmarks_cpp, bbox_xywh,
                "pyclnf_integration_validation.jpg"
            )

            # Validation result
            print("\n" + "=" * 80)
            if error_stats['mean'] < 5.0:
                print("✓ VALIDATION PASSED: Mean error < 5px")
            elif error_stats['mean'] < 10.0:
                print("⚠ VALIDATION WARNING: Mean error 5-10px (acceptable)")
            else:
                print("✗ VALIDATION FAILED: Mean error > 10px")
            print("=" * 80)

        else:
            print("⚠ No C++ reference available for comparison")

            # Still visualize Python landmarks
            visualize_comparison(
                frame, landmarks_68, None, bbox_xywh,
                "pyclnf_integration_validation.jpg"
            )

    except Exception as e:
        print(f"⚠ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
