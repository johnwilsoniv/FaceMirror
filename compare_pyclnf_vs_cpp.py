"""
Compare PyCLNF vs OpenFace C++ Implementation

Runs both PyCLNF and OpenFace C++ on the same frames and creates
side-by-side visualizations to validate correctness.
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from pyclnf import CLNF


def run_openface_cpp(image_path: str, output_dir: str) -> np.ndarray:
    """
    Run OpenFace C++ FeatureExtraction on an image.

    Returns:
        landmarks: (68, 2) array of landmark positions, or None if failed
    """
    openface_bin = Path("~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction").expanduser()

    if not openface_bin.exists():
        print(f"Warning: OpenFace binary not found at {openface_bin}")
        return None

    # Run OpenFace
    cmd = [
        str(openface_bin),
        "-f", image_path,
        "-out_dir", output_dir,
        "-2Dfp"  # Output 2D landmarks
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Read landmarks from CSV output
        csv_file = Path(output_dir) / (Path(image_path).stem + ".csv")

        if not csv_file.exists():
            print(f"OpenFace output not found: {csv_file}")
            return None

        # Parse CSV
        import pandas as pd
        df = pd.read_csv(csv_file)

        # Extract landmark columns (x_0, y_0, x_1, y_1, ..., x_67, y_67)
        landmarks = np.zeros((68, 2))
        for i in range(68):
            x_col = f'x_{i}'
            y_col = f'y_{i}'
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = df[x_col].iloc[0]
                landmarks[i, 1] = df[y_col].iloc[0]
            else:
                # Try with leading space
                x_col = f' x_{i}'
                y_col = f' y_{i}'
                if x_col in df.columns and y_col in df.columns:
                    landmarks[i, 0] = df[x_col].iloc[0]
                    landmarks[i, 1] = df[y_col].iloc[0]

        return landmarks

    except Exception as e:
        print(f"OpenFace C++ failed: {e}")
        return None


def detect_face_bbox(image: np.ndarray) -> tuple:
    """
    Simple face detection using OpenCV's Haar Cascade.

    Returns:
        bbox: (x, y, width, height) or None
    """
    # Load face detector - try multiple possible paths
    cascade_paths = [
        '/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]

    face_cascade = None
    for path in cascade_paths:
        if Path(path).exists():
            face_cascade = cv2.CascadeClassifier(path)
            break

    if face_cascade is None or face_cascade.empty():
        # Fallback: use dlib or return center bbox
        print("Warning: Haar cascade not found, using center bbox estimate")
        h, w = image.shape[:2]
        # Estimate face in center 60% of image
        size = min(h, w) * 0.6
        x = (w - size) / 2
        y = (h - size) / 2
        return (int(x), int(y), int(size), int(size))

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Return largest face
    largest = max(faces, key=lambda f: f[2] * f[3])
    return tuple(largest)


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, color: tuple = (0, 255, 0), label: str = "") -> np.ndarray:
    """Draw landmarks on image with label."""
    vis = image.copy()

    # Draw landmarks with larger circles for better visibility
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)  # Larger radius
        cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), 1)  # White outline

    # Draw label
    if label:
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return vis


def compare_landmarks(landmarks1: np.ndarray, landmarks2: np.ndarray) -> dict:
    """
    Compute comparison metrics between two sets of landmarks.

    Returns:
        metrics: Dictionary with comparison statistics
    """
    if landmarks1 is None or landmarks2 is None:
        return None

    # Compute per-landmark distance
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)

    return {
        'mean_distance': distances.mean(),
        'median_distance': np.median(distances),
        'max_distance': distances.max(),
        'std_distance': distances.std(),
        'distances': distances
    }


def create_comparison_visualization(image: np.ndarray,
                                   landmarks_cpp: np.ndarray,
                                   landmarks_py: np.ndarray,
                                   title: str = "") -> np.ndarray:
    """
    Create side-by-side comparison visualization.

    Args:
        image: Original image
        landmarks_cpp: OpenFace C++ landmarks
        landmarks_py: PyCLNF landmarks
        title: Title for the visualization

    Returns:
        comparison: Side-by-side visualization
    """
    # Create visualizations
    vis_cpp = draw_landmarks(image, landmarks_cpp, color=(0, 255, 0), label="OpenFace C++")
    vis_py = draw_landmarks(image, landmarks_py, color=(255, 0, 0), label="PyCLNF")

    # Create overlay
    vis_overlay = image.copy()
    vis_overlay = draw_landmarks(vis_overlay, landmarks_cpp, color=(0, 255, 0), label="")
    vis_overlay = draw_landmarks(vis_overlay, landmarks_py, color=(255, 0, 0), label="")
    cv2.putText(vis_overlay, "Overlay (Green=C++, Red=Py)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Compute metrics
    metrics = compare_landmarks(landmarks_cpp, landmarks_py)

    # Create metrics text
    if metrics:
        metrics_text = [
            f"Mean Error: {metrics['mean_distance']:.2f} px",
            f"Median Error: {metrics['median_distance']:.2f} px",
            f"Max Error: {metrics['max_distance']:.2f} px",
            f"Std Error: {metrics['std_distance']:.2f} px"
        ]

        # Add metrics to overlay
        y_offset = 60
        for text in metrics_text:
            cv2.putText(vis_overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

    # Stack horizontally
    comparison = np.hstack([vis_cpp, vis_py, vis_overlay])

    # Add title
    if title:
        title_height = 50
        title_bar = np.zeros((title_height, comparison.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        comparison = np.vstack([title_bar, comparison])

    return comparison


def process_video(video_path: str,
                 frame_indices: list,
                 clnf: CLNF,
                 output_dir: str) -> list:
    """
    Process specific frames from a video with both implementations.

    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to process
        clnf: PyCLNF CLNF instance
        output_dir: Directory to save results

    Returns:
        results: List of comparison images
    """
    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem

    results = []
    temp_dir = Path(tempfile.mkdtemp())

    try:
        for frame_idx in frame_indices:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Failed to read frame {frame_idx} from {video_name}")
                continue

            print(f"\nProcessing {video_name} frame {frame_idx}...")

            # Save frame to temp file for OpenFace
            temp_image = temp_dir / f"frame_{frame_idx}.jpg"
            cv2.imwrite(str(temp_image), frame)

            # Detect face
            bbox = detect_face_bbox(frame)

            if bbox is None:
                print(f"  No face detected in frame {frame_idx}")
                continue

            print(f"  Face detected: {bbox}")

            # Run OpenFace C++
            print("  Running OpenFace C++...")
            openface_output = temp_dir / "openface_output"
            openface_output.mkdir(exist_ok=True)
            landmarks_cpp = run_openface_cpp(str(temp_image), str(openface_output))

            if landmarks_cpp is None:
                print("  OpenFace C++ failed")
                continue

            print(f"  OpenFace C++ landmarks: {landmarks_cpp.shape}")

            # Run PyCLNF
            print("  Running PyCLNF...")
            try:
                landmarks_py, info = clnf.fit(frame, bbox, return_params=True)
                print(f"  PyCLNF landmarks: {landmarks_py.shape}")
                print(f"  PyCLNF converged: {info['converged']}, iterations: {info['iterations']}")
                print(f"  PyCLNF params: scale={info['params'][0]:.3f}, wx={info['params'][1]:.3f}, wy={info['params'][2]:.3f}, wz={info['params'][3]:.3f}, tx={info['params'][4]:.1f}, ty={info['params'][5]:.1f}")
                print(f"  PyCLNF landmark range: x=[{landmarks_py[:, 0].min():.1f}, {landmarks_py[:, 0].max():.1f}], y=[{landmarks_py[:, 1].min():.1f}, {landmarks_py[:, 1].max():.1f}]")
                print(f"  OpenFace landmark range: x=[{landmarks_cpp[:, 0].min():.1f}, {landmarks_cpp[:, 0].max():.1f}], y=[{landmarks_cpp[:, 1].min():.1f}, {landmarks_cpp[:, 1].max():.1f}]")
            except Exception as e:
                print(f"  PyCLNF failed: {e}")
                continue

            # Create comparison visualization
            title = f"{video_name} - Frame {frame_idx}"
            comparison = create_comparison_visualization(
                frame, landmarks_cpp, landmarks_py, title
            )

            # Save comparison
            output_path = Path(output_dir) / f"{video_name}_frame_{frame_idx}_comparison.jpg"
            cv2.imwrite(str(output_path), comparison)
            print(f"  Saved comparison to {output_path}")

            results.append(comparison)

    finally:
        cap.release()
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def main():
    """Main comparison script."""
    print("=" * 80)
    print("PyCLNF vs OpenFace C++ Comparison")
    print("=" * 80)

    # Initialize PyCLNF
    print("\nInitializing PyCLNF...")
    # Test with increased iterations to allow for better convergence
    # OpenFace uses 5 iterations but has shape-based early stopping
    # Testing with 25 iterations to see if convergence improves
    clnf = CLNF(model_dir="pyclnf/models", scale=0.25, max_iterations=25)
    print("PyCLNF initialized (max_iterations=25)")

    # Test videos (you can modify these paths)
    test_videos = [
        "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0434.MOV",
        "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0428.MOV",
        "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0433.MOV",
    ]

    # Frame indices to test (beginning, middle, end)
    frame_indices = [10, 50, 100]

    # Output directory
    output_dir = Path("pyclnf_comparison_results")
    output_dir.mkdir(exist_ok=True)

    # Process each video
    all_results = []
    for video_path in test_videos:
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            print(f"\nSkipping {video_path_obj.name}: File not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing: {video_path_obj.name}")
        print(f"{'=' * 80}")

        results = process_video(video_path, frame_indices, clnf, str(output_dir))
        all_results.extend(results)

    print(f"\n{'=' * 80}")
    print(f"Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Total comparisons: {len(all_results)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
