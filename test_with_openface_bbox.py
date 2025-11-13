"""
Test PyCLNF using OpenFace's bbox to isolate bbox vs algorithm issues.

This script:
1. Runs OpenFace C++ (which does its own face detection)
2. Computes bbox from OpenFace's detected landmarks
3. Runs PyCLNF with that exact bbox
4. Compares the results
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import pandas as pd
from pyclnf import CLNF


def run_openface_cpp(image_path: str, output_dir: str):
    """Run OpenFace C++ and extract landmarks + bbox."""
    openface_bin = Path("~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction").expanduser()

    cmd = [
        str(openface_bin),
        "-f", image_path,
        "-out_dir", output_dir,
        "-2Dfp"
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Read landmarks from CSV
    csv_file = Path(output_dir) / (Path(image_path).stem + ".csv")
    df = pd.read_csv(csv_file)

    # Extract landmarks
    landmarks = np.zeros((68, 2))
    for i in range(68):
        x_col = f'x_{i}'
        y_col = f'y_{i}'
        if x_col in df.columns:
            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]

    # Compute bbox from landmarks
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)

    # Add margin (OpenFace typically uses ~10% margin)
    width = x_max - x_min
    height = y_max - y_min
    margin = 0.1

    x_min -= width * margin
    y_min -= height * margin
    width *= (1 + 2 * margin)
    height *= (1 + 2 * margin)

    bbox = (int(x_min), int(y_min), int(width), int(height))

    return landmarks, bbox


def extract_frame(video_path: str, frame_num: int, output_path: str):
    """Extract a frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(output_path, frame)
        return frame
    return None


def draw_comparison(image, landmarks_openface, landmarks_pyclnf, bbox):
    """Draw side-by-side comparison."""
    h, w = image.shape[:2]

    # Create three panels
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)

    # Left: OpenFace only
    left_img = image.copy()
    for i, (x, y) in enumerate(landmarks_openface):
        cv2.circle(left_img, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.circle(left_img, (int(x), int(y)), 5, (255, 255, 255), 1)
    cv2.rectangle(left_img, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    cv2.putText(left_img, "OpenFace C++", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Middle: PyCLNF only
    mid_img = image.copy()
    for i, (x, y) in enumerate(landmarks_pyclnf):
        cv2.circle(mid_img, (int(x), int(y)), 4, (255, 0, 0), -1)
        cv2.circle(mid_img, (int(x), int(y)), 5, (255, 255, 255), 1)
    cv2.rectangle(mid_img, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    cv2.putText(mid_img, "PyCLNF", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Right: Overlay
    overlay_img = image.copy()
    for i, (x, y) in enumerate(landmarks_openface):
        cv2.circle(overlay_img, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.circle(overlay_img, (int(x), int(y)), 5, (255, 255, 255), 1)
    for i, (x, y) in enumerate(landmarks_pyclnf):
        cv2.circle(overlay_img, (int(x), int(y)), 4, (255, 0, 0), -1)
        cv2.circle(overlay_img, (int(x), int(y)), 5, (255, 255, 255), 1)
    cv2.rectangle(overlay_img, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 2)
    cv2.putText(overlay_img, "Overlay (Green=C++, Red=Py)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Compute error metrics
    mean_error = np.linalg.norm(landmarks_openface - landmarks_pyclnf, axis=1).mean()
    median_error = np.median(np.linalg.norm(landmarks_openface - landmarks_pyclnf, axis=1))
    max_error = np.linalg.norm(landmarks_openface - landmarks_pyclnf, axis=1).max()

    cv2.putText(overlay_img, f"Mean: {mean_error:.1f}px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay_img, f"Median: {median_error:.1f}px", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay_img, f"Max: {max_error:.1f}px", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine
    vis[:, :w] = left_img
    vis[:, w:2*w] = mid_img
    vis[:, 2*w:] = overlay_img

    return vis, mean_error, median_error, max_error


def main():
    print("=" * 80)
    print("Testing PyCLNF with OpenFace C++ Bbox")
    print("=" * 80)

    # Initialize PyCLNF
    print("\nInitializing PyCLNF...")
    clnf = CLNF(
        model_dir="pyclnf/models",
        scale=0.25,
        regularization=25.0,
        max_iterations=5,
        sigma=1.5,
        window_sizes=[11, 9, 7]  # Removed ws=5 (no sigma components)
    )
    print("PyCLNF initialized")

    # Test videos and frames
    test_cases = [
        ("Patient Data/Normal Cohort/IMG_0434.MOV", 50, "IMG_0434_frame_50"),
        ("Patient Data/Normal Cohort/IMG_0428.MOV", 50, "IMG_0428_frame_50"),
        ("Patient Data/Normal Cohort/IMG_0433.MOV", 50, "IMG_0433_frame_50"),
    ]

    output_dir = Path("openface_bbox_test_results")
    output_dir.mkdir(exist_ok=True)

    results = []

    for video_path, frame_num, name in test_cases:
        print(f"\n{'='*80}")
        print(f"Processing: {name}")
        print(f"{'='*80}")

        # Extract frame
        frame_path = f"/tmp/{name}.jpg"
        print(f"Extracting frame {frame_num}...")
        frame = extract_frame(video_path, frame_num, frame_path)

        if frame is None:
            print(f"  Failed to extract frame")
            continue

        # Run OpenFace C++ (does its own face detection)
        print(f"Running OpenFace C++ (with internal face detection)...")
        with tempfile.TemporaryDirectory() as tmpdir:
            landmarks_openface, bbox = run_openface_cpp(frame_path, tmpdir)

        print(f"  OpenFace detected bbox: {bbox}")
        print(f"  OpenFace landmarks range: x=[{landmarks_openface[:, 0].min():.1f}, {landmarks_openface[:, 0].max():.1f}], "
              f"y=[{landmarks_openface[:, 1].min():.1f}, {landmarks_openface[:, 1].max():.1f}]")

        # Run PyCLNF with OpenFace's bbox
        print(f"Running PyCLNF with OpenFace's bbox...")
        landmarks_pyclnf, info = clnf.fit(frame, bbox)

        print(f"  PyCLNF converged: {info['converged']}, iterations: {info['iterations']}")
        print(f"  PyCLNF landmarks range: x=[{landmarks_pyclnf[:, 0].min():.1f}, {landmarks_pyclnf[:, 0].max():.1f}], "
              f"y=[{landmarks_pyclnf[:, 1].min():.1f}, {landmarks_pyclnf[:, 1].max():.1f}]")

        # Create comparison visualization
        vis, mean_err, median_err, max_err = draw_comparison(
            frame, landmarks_openface, landmarks_pyclnf, bbox
        )

        output_path = output_dir / f"{name}_comparison.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"  Saved comparison to {output_path}")
        print(f"  Error - Mean: {mean_err:.1f}px, Median: {median_err:.1f}px, Max: {max_err:.1f}px")

        results.append({
            'name': name,
            'bbox': bbox,
            'mean_error': mean_err,
            'median_error': median_err,
            'max_error': max_err,
            'converged': info['converged'],
            'iterations': info['iterations']
        })

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Bbox: {r['bbox']}")
        print(f"  Mean error: {r['mean_error']:.1f}px")
        print(f"  Median error: {r['median_error']:.1f}px")
        print(f"  Max error: {r['max_error']:.1f}px")
        print(f"  Converged: {r['converged']}, Iterations: {r['iterations']}")

    avg_mean = np.mean([r['mean_error'] for r in results])
    avg_median = np.mean([r['median_error'] for r in results])
    print(f"\nOverall Average:")
    print(f"  Mean error: {avg_mean:.1f}px")
    print(f"  Median error: {avg_median:.1f}px")


if __name__ == "__main__":
    main()
