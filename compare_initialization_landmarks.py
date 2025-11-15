"""
Compare initialization landmarks (PDM from bbox) vs final landmarks (after CLNF)
for both C++ OpenFace and Python PyFaceAU.

C++ and Python both initialize from bbox → PDM → initial 68 landmarks → CLNF → final 68 landmarks
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import cv2

# Add pyclnf to path for PDM
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))
from core.pdm import PDM

# Paths
PROJECT_ROOT = Path(__file__).parent
CPP_OUTPUT = PROJECT_ROOT / "validation_output" / "cpp_baseline"
PYTHON_OUTPUT = PROJECT_ROOT / "validation_output" / "python_baseline"
TEST_FRAMES = PROJECT_ROOT / "calibration_frames"
VIS_OUTPUT = PROJECT_ROOT / "validation_output" / "initialization_comparison"
PDM_DIR = PROJECT_ROOT / "pyclnf" / "models" / "exported_pdm"

# Create output directory
VIS_OUTPUT.mkdir(parents=True, exist_ok=True)

# Test frames
FRAMES = [
    'patient1_frame1',
    'patient1_frame2',
    'patient1_frame3',
]


def load_cpp_mtcnn_landmarks(frame_name):
    """Load C++ MTCNN 5-point landmarks and bbox."""
    mtcnn_csv = CPP_OUTPUT / f"{frame_name}_mtcnn_debug.csv"
    if not mtcnn_csv.exists():
        return None, None

    df = pd.read_csv(mtcnn_csv)

    # Get bbox
    bbox_x = df['bbox_x'].iloc[-1]
    bbox_y = df['bbox_y'].iloc[-1]
    bbox_w = df['bbox_w'].iloc[-1]
    bbox_h = df['bbox_h'].iloc[-1]
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

    # Get 5-point landmarks (normalized [0, 1])
    lm_norm = np.array([
        [df['lm1_x'].iloc[-1], df['lm1_y'].iloc[-1]],
        [df['lm2_x'].iloc[-1], df['lm2_y'].iloc[-1]],
        [df['lm3_x'].iloc[-1], df['lm3_y'].iloc[-1]],
        [df['lm4_x'].iloc[-1], df['lm4_y'].iloc[-1]],
        [df['lm5_x'].iloc[-1], df['lm5_y'].iloc[-1]],
    ])

    # Convert normalized to absolute coordinates
    lm_abs = lm_norm.copy()
    lm_abs[:, 0] = bbox_x + lm_norm[:, 0] * bbox_w
    lm_abs[:, 1] = bbox_y + lm_norm[:, 1] * bbox_h

    return lm_abs, bbox


def load_cpp_final_landmarks(frame_name):
    """Load C++ final 68-point landmarks from OpenFace CSV."""
    csv_path = CPP_OUTPUT / f"{frame_name}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Extract x_0...x_67 and y_0...y_67
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    x = df[x_cols].iloc[-1].values
    y = df[y_cols].iloc[-1].values

    landmarks = np.column_stack([x, y])
    return landmarks


def load_python_data(frame_name):
    """Load Python landmarks and MTCNN info from debug JSON."""
    json_path = PYTHON_OUTPUT / f"{frame_name}_result.json"
    if not json_path.exists():
        return None, None, None

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get bbox (xyxy format)
    face_info = data['debug_info']['face_detection']
    bbox_xyxy = face_info['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1])

    # Get final 68-point landmarks
    lm_info = data['debug_info']['landmark_detection']
    landmarks_final = np.array(lm_info['landmarks_68'])

    # Note: PyMTCNN 5-point landmarks not stored in our current pipeline
    # We'll need to extract them from the image using PyMTCNN

    return landmarks_final, bbox, None


def init_landmarks_from_pdm(bbox, pdm):
    """
    Initialize 68-point landmarks from bounding box using PDM.
    This mimics what OpenFace/CLNF does for initialization.
    """
    # Initialize PDM parameters from bbox
    params = pdm.init_params(bbox)

    # Convert params to 2D landmarks
    landmarks_init = pdm.params_to_landmarks_2d(params)

    return landmarks_init


def draw_comparison(image_path, cpp_init, cpp_final, python_init, python_final,
                    cpp_bbox, python_bbox, frame_name, output_path):
    """
    Draw initialization and final landmarks comparison on the original image.

    Color coding:
    - Green = C++ landmarks
    - Blue = Python landmarks
    - Solid circles = Final (after CLNF)
    - Empty circles = Init (before CLNF)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not load image {image_path}")
        return

    vis = img.copy()

    # Draw bboxes
    x, y, w, h = cpp_bbox
    cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)  # Green

    x, y, w, h = python_bbox
    cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)  # Blue

    # Draw C++ initialization landmarks (green, empty circles)
    for i, (x, y) in enumerate(cpp_init):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), 1)  # Green outline

    # Draw Python initialization landmarks (blue, empty circles)
    for i, (x, y) in enumerate(python_init):
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), 1)  # Blue outline

    # Draw C++ final landmarks (green, solid circles) - smaller to overlay
    for i, (x, y) in enumerate(cpp_final):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green filled

    # Draw Python final landmarks (blue, solid circles) - smaller to overlay
    for i, (x, y) in enumerate(python_final):
        cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue filled

    # Add legend
    legend_y = 30
    cv2.putText(vis, "Green: C++ OpenFace | Blue: Python pyclnf", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "Empty circle: Init (PDM) | Solid circle: Final (CLNF)", (10, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    cv2.imwrite(str(output_path), vis)
    print(f"  ✓ Saved visualization: {output_path.name}")


def compute_error(landmarks1, landmarks2):
    """Compute per-landmark Euclidean error."""
    return np.linalg.norm(landmarks1 - landmarks2, axis=1)


def main():
    print("=" * 80)
    print("Initialization vs Final Landmark Comparison")
    print("=" * 80)

    # Load PDM
    pdm = PDM(str(PDM_DIR))
    print(f"Loaded PDM from {PDM_DIR}")
    print(f"  PDM shape parameters: Mean shape, PC, eigenvalues loaded\n")

    results = []

    for frame_name in FRAMES:
        print(f"\nProcessing {frame_name}...")
        print("-" * 80)

        # Load C++ data
        cpp_mtcnn_lm5, cpp_bbox = load_cpp_mtcnn_landmarks(frame_name)
        cpp_final = load_cpp_final_landmarks(frame_name)

        # Load Python data
        python_final, python_bbox, _ = load_python_data(frame_name)

        if cpp_bbox is None or python_bbox is None:
            print(f"  ✗ Missing data for {frame_name}")
            continue

        # Generate init landmarks for both using PDM
        cpp_init = init_landmarks_from_pdm(cpp_bbox, pdm)
        python_init = init_landmarks_from_pdm(python_bbox, pdm)

        # Generate visualization
        image_path = TEST_FRAMES / f"{frame_name}.jpg"
        vis_path = VIS_OUTPUT / f"{frame_name}_init_vs_final.jpg"
        draw_comparison(image_path, cpp_init, cpp_final, python_init, python_final,
                       cpp_bbox, python_bbox, frame_name, vis_path)

        # Compute errors vs C++ final (ground truth)
        cpp_init_error = compute_error(cpp_init, cpp_final)
        cpp_final_error = np.zeros(68)  # C++ final vs itself = 0
        python_init_error = compute_error(python_init, cpp_final)
        python_final_error = compute_error(python_final, cpp_final)

        # Print summary
        print(f"  C++ MTCNN bbox:   ({cpp_bbox[0]:.1f}, {cpp_bbox[1]:.1f}, {cpp_bbox[2]:.1f}, {cpp_bbox[3]:.1f})")
        print(f"  Python MTCNN bbox: ({python_bbox[0]:.1f}, {python_bbox[1]:.1f}, {python_bbox[2]:.1f}, {python_bbox[3]:.1f})")
        print(f"  Bbox difference:   dx={abs(cpp_bbox[0]-python_bbox[0]):.1f}, dy={abs(cpp_bbox[1]-python_bbox[1]):.1f}, dw={abs(cpp_bbox[2]-python_bbox[2]):.1f}, dh={abs(cpp_bbox[3]-python_bbox[3]):.1f}")
        print()
        print(f"  C++ Init → Final:    {cpp_init_error.mean():.3f} ± {cpp_init_error.std():.3f} px (CLNF improvement)")
        print(f"  Python Init error:   {python_init_error.mean():.3f} ± {python_init_error.std():.3f} px (vs C++ final)")
        print(f"  Python Final error:  {python_final_error.mean():.3f} ± {python_final_error.std():.3f} px (vs C++ final)")

        # Check if CLNF improved or degraded
        if python_final_error.mean() < python_init_error.mean():
            improvement = python_init_error.mean() - python_final_error.mean()
            print(f"  Python CLNF:         ✓ Improved by {improvement:.3f} px")
        else:
            degradation = python_final_error.mean() - python_init_error.mean()
            print(f"  Python CLNF:         ✗ Degraded by {degradation:.3f} px")

        results.append({
            'frame': frame_name,
            'cpp_init_error': cpp_init_error.mean(),
            'python_init_error': python_init_error.mean(),
            'python_final_error': python_final_error.mean(),
            'python_clnf_delta': python_final_error.mean() - python_init_error.mean(),
            'bbox_diff_x': abs(cpp_bbox[0] - python_bbox[0]),
            'bbox_diff_y': abs(cpp_bbox[1] - python_bbox[1]),
            'bbox_diff_w': abs(cpp_bbox[2] - python_bbox[2]),
            'bbox_diff_h': abs(cpp_bbox[3] - python_bbox[3]),
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    print(f"\nC++ Init error (PDM from bbox):     {df_results['cpp_init_error'].mean():.3f} ± {df_results['cpp_init_error'].std():.3f} px")
    print(f"Python Init error (PDM from bbox):  {df_results['python_init_error'].mean():.3f} ± {df_results['python_init_error'].std():.3f} px")
    print(f"Python Final error (after CLNF):    {df_results['python_final_error'].mean():.3f} ± {df_results['python_final_error'].std():.3f} px")
    print(f"\nPython CLNF impact:                 {df_results['python_clnf_delta'].mean():.3f} px (negative = improvement)")

    print(f"\nBbox differences (C++ vs Python):")
    print(f"  Position: {df_results['bbox_diff_x'].mean():.1f} px (x), {df_results['bbox_diff_y'].mean():.1f} px (y)")
    print(f"  Size:     {df_results['bbox_diff_w'].mean():.1f} px (w), {df_results['bbox_diff_h'].mean():.1f} px (h)")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if df_results['python_clnf_delta'].mean() < -1.0:
        print("✓ Python CLNF is IMPROVING landmarks (reducing error)")
    elif df_results['python_clnf_delta'].mean() > 1.0:
        print("✗ Python CLNF is DEGRADING landmarks (increasing error)")
    else:
        print("− Python CLNF has minimal impact (< 1px change)")

    if df_results['python_init_error'].mean() > df_results['cpp_init_error'].mean() + 2.0:
        print("✗ Python initialization is WORSE than C++ (bbox or PDM issue)")
    else:
        print("✓ Python initialization is similar to C++ (bbox and PDM working)")

    print()


if __name__ == "__main__":
    main()
