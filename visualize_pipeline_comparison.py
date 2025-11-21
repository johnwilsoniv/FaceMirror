#!/usr/bin/env python3
"""
Visualization script for comparing Python and C++ AU pipelines.

Generates side-by-side visualizations showing:
1. Face detection bounding boxes
2. Landmark points (68 facial landmarks)
3. AU predictions as heatmaps
4. Performance metrics comparison
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import subprocess
import pandas as pd
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Optional

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0), size=2):
    """Draw 68 facial landmarks on image."""
    image_copy = image.copy()

    # Define landmark connections for face outline, eyes, nose, mouth
    jaw_points = list(range(0, 17))
    right_eyebrow = list(range(17, 22))
    left_eyebrow = list(range(22, 27))
    nose_bridge = list(range(27, 31))
    nose_bottom = list(range(31, 36))
    right_eye = list(range(36, 42))
    left_eye = list(range(42, 48))
    outer_mouth = list(range(48, 60))
    inner_mouth = list(range(60, 68))

    # Draw connections
    connections = [
        jaw_points,
        right_eyebrow,
        left_eyebrow,
        nose_bridge,
        nose_bottom
    ]

    # Draw closed shapes
    closed_shapes = [
        right_eye,
        left_eye,
        outer_mouth,
        inner_mouth
    ]

    # Draw open connections
    for connection in connections:
        for i in range(len(connection) - 1):
            pt1 = tuple(landmarks[connection[i]].astype(int))
            pt2 = tuple(landmarks[connection[i + 1]].astype(int))
            cv2.line(image_copy, pt1, pt2, color, 1)

    # Draw closed shapes
    for shape in closed_shapes:
        for i in range(len(shape)):
            pt1 = tuple(landmarks[shape[i]].astype(int))
            pt2 = tuple(landmarks[shape[(i + 1) % len(shape)]].astype(int))
            cv2.line(image_copy, pt1, pt2, color, 1)

    # Draw landmark points
    for landmark in landmarks:
        cv2.circle(image_copy, tuple(landmark.astype(int)), size, color, -1)

    return image_copy


def draw_bbox(image: np.ndarray, bbox: List[int], color=(255, 0, 0), thickness=2):
    """Draw bounding box on image."""
    image_copy = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy


def process_with_python_pipeline(video_path: str, output_dir: str = "python_output"):
    """Process video with Python pipeline."""
    print("\n" + "="*60)
    print("PYTHON PIPELINE PROCESSING")
    print("="*60)

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    # Initialize components
    print("Initializing Python components...")
    detector = MTCNN()
    clnf = CLNF(model_dir="pyclnf/models")
    au_pipeline = FullPythonAUPipeline(
        pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        au_models_dir="pyfaceau/weights/AU_predictors",
        triangulation_file="pyfaceau/weights/tris_68_full.txt",
        patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
        verbose=False
    )

    # Process video
    cap = cv2.VideoCapture(video_path)

    results = []
    frame_count = 0
    processing_times = []

    Path(output_dir).mkdir(exist_ok=True)

    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:  # Process every 30th frame for visualization
            print(f"  Frame {frame_count}...")

            start_time = time.perf_counter()

            # Detect face
            faces = detector.detect(frame)

            if faces:
                face = faces[0]  # Use first face
                bbox = face.get('box', face.get('bbox', [0, 0, frame.shape[1], frame.shape[0]]))

                # Ensure bbox is a list of 4 integers
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    bbox = [int(b) for b in bbox]
                else:
                    # Fallback to full frame
                    bbox = [0, 0, frame.shape[1], frame.shape[0]]

                # Get landmarks
                clnf.initialize_from_bbox(frame, bbox)
                success = clnf.fit_image(frame)

                if success:
                    landmarks = clnf.get_landmarks()

                    # Get AUs
                    pipeline_result = au_pipeline.process_frame(frame)
                    aus = pipeline_result.get('aus', {}) if pipeline_result else {}

                    processing_time = time.perf_counter() - start_time
                    processing_times.append(processing_time)

                    # Save visualization
                    vis_frame = frame.copy()
                    # Skip bbox drawing if there are issues
                    try:
                        if bbox and len(bbox) == 4:
                            vis_frame = draw_bbox(vis_frame, bbox, color=(255, 0, 0))
                    except:
                        pass  # Skip bbox drawing if there's an error
                    vis_frame = draw_landmarks(vis_frame, landmarks, color=(0, 255, 0))

                    cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", vis_frame)

                    results.append({
                        'frame': frame_count,
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'aus': aus,
                        'processing_time': processing_time
                    })

        frame_count += 1

        if frame_count >= 150:  # Limit to first 150 frames
            break

    cap.release()

    if processing_times:
        avg_time = np.mean(processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nPython Pipeline Performance:")
        print(f"  Average processing time: {avg_time*1000:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")

    return results


def process_with_openface(video_path: str, output_dir: str = "openface_output"):
    """Process video with OpenFace C++ pipeline."""
    print("\n" + "="*60)
    print("OPENFACE C++ PIPELINE PROCESSING")
    print("="*60)

    # Find OpenFace executable
    openface_path = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    if not Path(openface_path).exists():
        print("Error: OpenFace not found at expected path")
        return None

    Path(output_dir).mkdir(exist_ok=True)

    # Run OpenFace with visualization output
    cmd = [
        openface_path,
        "-f", video_path,
        "-out_dir", output_dir,
        "-aus",  # Action Units
        "-2Dfp",  # 2D landmarks
        "-pose",  # Head pose
        "-gaze",  # Gaze
        "-tracked",  # Output tracked video
        "-verbose"
    ]

    print(f"Running: {' '.join(cmd)}")
    start_time = time.perf_counter()

    result = subprocess.run(cmd, capture_output=True, text=True)

    processing_time = time.perf_counter() - start_time

    # Parse CSV output
    video_name = Path(video_path).stem
    csv_path = Path(output_dir) / f"{video_name}.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Calculate performance
        n_frames = len(df)
        fps = n_frames / processing_time if processing_time > 0 else 0

        print(f"\nOpenFace Performance:")
        print(f"  Total processing time: {processing_time:.2f}s")
        print(f"  Frames processed: {n_frames}")
        print(f"  Average FPS: {fps:.1f}")

        return df
    else:
        print(f"Error: OpenFace output not found at {csv_path}")
        return None


def create_comparison_visualization(python_results: List, openface_df: pd.DataFrame,
                                   video_path: str, output_path: str = "pipeline_comparison.png"):
    """Create comprehensive comparison visualization."""
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*60)

    # Load sample frames from video
    cap = cv2.VideoCapture(video_path)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Sample frames to visualize
    sample_frames = [0, 30, 60, 90]

    for idx, frame_num in enumerate(sample_frames):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Python pipeline visualization
        ax_py = fig.add_subplot(gs[0, idx])
        ax_py.imshow(frame_rgb)
        ax_py.set_title(f"Python - Frame {frame_num}", fontsize=10)
        ax_py.axis('off')

        # Find corresponding Python result
        py_result = next((r for r in python_results if r['frame'] == frame_num), None)
        if py_result:
            # Draw bbox
            bbox = py_result['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                    linewidth=2, edgecolor='r', facecolor='none')
            ax_py.add_patch(rect)

            # Draw landmarks
            landmarks = py_result['landmarks']
            ax_py.plot(landmarks[:, 0], landmarks[:, 1], 'g.', markersize=2)

        # OpenFace visualization
        ax_of = fig.add_subplot(gs[1, idx])
        ax_of.imshow(frame_rgb)
        ax_of.set_title(f"OpenFace - Frame {frame_num}", fontsize=10)
        ax_of.axis('off')

        # Get OpenFace landmarks for this frame
        if openface_df is not None and frame_num < len(openface_df):
            row = openface_df.iloc[frame_num]

            # Extract 2D landmarks (x_0 to x_67, y_0 to y_67)
            landmarks_x = [row[f'x_{i}'] for i in range(68) if f'x_{i}' in row]
            landmarks_y = [row[f'y_{i}'] for i in range(68) if f'y_{i}' in row]

            if landmarks_x and landmarks_y:
                ax_of.plot(landmarks_x, landmarks_y, 'g.', markersize=2)

    # AU comparison heatmap
    ax_au = fig.add_subplot(gs[2, :2])

    if python_results and openface_df is not None:
        # Common AUs
        au_names = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07',
                   'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17']

        # Create comparison matrix
        n_samples = min(5, len(python_results))
        comparison_matrix = np.zeros((len(au_names), n_samples * 2))

        for i, au in enumerate(au_names):
            for j in range(n_samples):
                # Python AU
                py_result = python_results[j]
                au_num = int(au[2:])
                py_value = py_result['aus'].get(f'au{au_num}', 0.0) if py_result['aus'] else 0.0
                comparison_matrix[i, j*2] = py_value

                # OpenFace AU
                of_key = f"{au}_r"
                if of_key in openface_df.columns and j*30 < len(openface_df):
                    of_value = openface_df.iloc[j*30][of_key]
                    comparison_matrix[i, j*2 + 1] = of_value

        im = ax_au.imshow(comparison_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=5)
        ax_au.set_yticks(range(len(au_names)))
        ax_au.set_yticklabels(au_names)
        ax_au.set_xticks(range(n_samples * 2))
        ax_au.set_xticklabels(['Py', 'OF'] * n_samples, rotation=45)
        ax_au.set_title("AU Intensity Comparison", fontsize=12)
        plt.colorbar(im, ax=ax_au)

    # Performance comparison
    ax_perf = fig.add_subplot(gs[2, 2:])

    if python_results:
        py_times = [r['processing_time'] * 1000 for r in python_results]
        py_fps = 1000 / np.mean(py_times) if py_times else 0

        # Estimate OpenFace FPS from total processing
        of_fps = 30  # Placeholder - should be calculated from actual timing

        categories = ['Python Pipeline', 'OpenFace C++']
        fps_values = [py_fps, of_fps]
        colors = ['#3498db', '#e74c3c']

        bars = ax_perf.bar(categories, fps_values, color=colors)
        ax_perf.set_ylabel('FPS')
        ax_perf.set_title('Performance Comparison', fontsize=12)
        ax_perf.set_ylim(0, max(fps_values) * 1.2)

        # Add value labels on bars
        for bar, val in zip(bars, fps_values):
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f} FPS', ha='center', va='bottom')

    cap.release()

    plt.suptitle("Python vs OpenFace Pipeline Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig


def main():
    """Run pipeline comparison with visualization."""
    print("="*80)
    print("VISUAL PIPELINE COMPARISON: Python vs OpenFace C++")
    print("="*80)

    # Video to process
    video_path = "Patient Data/Normal Cohort/Shorty.mov"

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"\nProcessing video: {video_path}")

    # Process with Python pipeline
    try:
        python_results = process_with_python_pipeline(video_path)
    except Exception as e:
        print(f"Error in Python pipeline: {e}")
        python_results = []

    # Process with OpenFace
    try:
        openface_df = process_with_openface(video_path)
    except Exception as e:
        print(f"Error in OpenFace pipeline: {e}")
        openface_df = None

    # Create comparison visualization
    if python_results or openface_df is not None:
        create_comparison_visualization(
            python_results,
            openface_df,
            video_path,
            output_path="pipeline_comparison.png"
        )

    # Create detailed frame comparisons
    if python_results:
        print("\nCreating detailed frame comparisons...")

        cap = cv2.VideoCapture(video_path)

        for i, result in enumerate(python_results[:3]):  # First 3 processed frames
            frame_num = result['frame']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                # Create side-by-side comparison
                h, w = frame.shape[:2]
                comparison = np.zeros((h, w*2, 3), dtype=np.uint8)

                # Python side
                py_frame = frame.copy()
                # Try to draw bbox if available
                try:
                    if 'bbox' in result and result['bbox'] and len(result['bbox']) == 4:
                        py_frame = draw_bbox(py_frame, result['bbox'], color=(255, 0, 0))
                except:
                    pass  # Skip bbox if there's an error

                # Draw landmarks
                if 'landmarks' in result and result['landmarks'] is not None:
                    py_frame = draw_landmarks(py_frame, result['landmarks'], color=(0, 255, 0))
                comparison[:, :w] = py_frame

                # OpenFace side (if available)
                of_frame = frame.copy()
                if openface_df is not None and frame_num < len(openface_df):
                    row = openface_df.iloc[frame_num]

                    # Extract landmarks
                    landmarks = []
                    for j in range(68):
                        if f'x_{j}' in row and f'y_{j}' in row:
                            landmarks.append([row[f'x_{j}'], row[f'y_{j}']])

                    if landmarks:
                        landmarks = np.array(landmarks)
                        of_frame = draw_landmarks(of_frame, landmarks, color=(0, 255, 255))

                comparison[:, w:] = of_frame

                # Add labels
                cv2.putText(comparison, "Python Pipeline", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, "OpenFace C++", (w + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Add performance info
                proc_time = result['processing_time'] * 1000
                cv2.putText(comparison, f"{proc_time:.1f}ms", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imwrite(f"comparison_frame_{frame_num:04d}.jpg", comparison)
                print(f"  Saved comparison_frame_{frame_num:04d}.jpg")

        cap.release()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()