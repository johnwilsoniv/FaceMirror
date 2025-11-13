#!/usr/bin/env python3
"""
Collect BBox Dataset for Adaptive Correction Model Training

Extracts bbox data from 112 patient videos:
- C++ OpenFace MTCNN (ground truth)
- Python RetinaFace (raw detection)

Goal: Build dataset to train optimal correction transform.
"""

import subprocess
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

# Import video rotation detection from S1
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
from video_rotation import get_video_rotation, normalize_rotation


def find_patient_videos(data_dir):
    """Find all patient video files."""
    data_dir = Path(data_dir)
    video_extensions = ['.MOV', '.mov', '.mp4', '.MP4', '.avi', '.AVI']

    videos = []
    for ext in video_extensions:
        videos.extend(data_dir.rglob(f"*{ext}"))

    # Filter out hidden/temp files
    videos = [v for v in videos if not any(part.startswith('.') for part in v.parts)]

    return sorted(videos)


def create_rotated_video_with_ffmpeg(input_path, output_path):
    """
    Pre-process video with FFmpeg's auto-rotate to handle rotation metadata correctly.

    This is the same approach used by S1 Face Mirror and is much more robust than
    trying to handle rotation on a per-frame basis with cv2.VideoCapture.

    Args:
        input_path: Path to original video
        output_path: Path to save rotated video

    Returns:
        str: Path to rotated video file
    """
    import subprocess

    # Check if rotation is needed
    rotation_raw = get_video_rotation(str(input_path))
    rotation = normalize_rotation(rotation_raw)

    if rotation == 0:
        # No rotation needed, just use original
        return str(input_path)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use FFmpeg's default auto-rotation (same approach as S1 Face Mirror)
    # FFmpeg automatically reads rotation metadata and bakes it into the output video
    # No need for explicit transpose filters - FFmpeg handles it via -autorotate (default=1)
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', str(input_path),
        '-c:v', 'libx264',  # H.264 codec
        '-preset', 'ultrafast',  # Fast encoding
        '-crf', '23',  # Good quality
        '-pix_fmt', 'yuv420p',  # Compatibility
        '-an',  # No audio (faster)
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=90)
        return str(output_path)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Warning: FFmpeg rotation failed for {input_path}: {e}")
        # Fall back to original video
        return str(input_path)


def extract_frames_from_video(video_path, num_frames=10, output_dir=None):
    """
    Extract evenly spaced frames from video using FFmpeg preprocessing for rotation.

    This approach matches S1 Face Mirror: pre-process video with FFmpeg's auto-rotate
    filter, then extract frames with cv2.VideoCapture. This is much more robust than
    trying to handle rotation on a per-frame basis.

    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        output_dir: Directory to save rotated video (if needed)

    Returns:
        List of frame dictionaries
    """
    # Detect video rotation from metadata
    rotation_raw = get_video_rotation(str(video_path))
    rotation = normalize_rotation(rotation_raw)

    # Pre-process video with FFmpeg if rotation needed
    if rotation != 0 and output_dir:
        rotated_video_path = Path(output_dir) / f"rotated_{Path(video_path).name}"
        video_to_read = create_rotated_video_with_ffmpeg(video_path, rotated_video_path)
    else:
        video_to_read = str(video_path)

    # Now extract frames from (potentially rotated) video
    cap = cv2.VideoCapture(video_to_read)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    # Evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append({
                'frame_idx': frame_idx,
                'frame': frame,
                'rotation': rotation
            })

    cap.release()
    return frames


def run_cpp_openface_bbox(image_path, output_dir):
    """Run C++ OpenFace and extract bbox from debug output."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-wild",  # Video mode
        "-verbose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract bbox
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_scale_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+)', output)

    data = {'success': False}

    if bbox_match:
        bbox = tuple(float(bbox_match.group(i)) for i in range(1, 5))
        data['bbox'] = bbox
        data['center_x'] = bbox[0] + bbox[2] / 2
        data['center_y'] = bbox[1] + bbox[3] / 2
        data['width'] = bbox[2]
        data['height'] = bbox[3]
        data['aspect_ratio'] = bbox[2] / bbox[3] if bbox[3] > 0 else 0
        data['size'] = np.sqrt(bbox[2] * bbox[3])
        data['success'] = True

    if init_scale_match:
        data['init_scale'] = float(init_scale_match.group(1))

    return data


def run_retinaface_bbox(image):
    """Run RetinaFace and extract raw bbox."""
    from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

    # Use cached detector
    if not hasattr(run_retinaface_bbox, 'detector'):
        retinaface_model = Path("S1 Face Mirror/weights/retinaface_mobilenet025_coreml.onnx")
        run_retinaface_bbox.detector = ONNXRetinaFaceDetector(
            str(retinaface_model),
            use_coreml=False,
            confidence_threshold=0.5,
            nms_threshold=0.4
        )

    detections, _ = run_retinaface_bbox.detector.detect_faces(image, resize=1.0)

    data = {'success': False}

    if len(detections) > 0:
        # Get first (highest confidence) detection
        x1, y1, x2, y2 = detections[0][:4]
        bbox = (x1, y1, x2 - x1, y2 - y1)
        confidence = detections[0][4] if len(detections[0]) > 4 else 1.0

        data['bbox'] = bbox
        data['center_x'] = bbox[0] + bbox[2] / 2
        data['center_y'] = bbox[1] + bbox[3] / 2
        data['width'] = bbox[2]
        data['height'] = bbox[3]
        data['aspect_ratio'] = bbox[2] / bbox[3] if bbox[3] > 0 else 0
        data['size'] = np.sqrt(bbox[2] * bbox[3])
        data['confidence'] = confidence
        data['success'] = True

    return data


def process_video(video_path, patient_id, output_dir):
    """Process single video: extract frames and bbox data."""
    video_name = video_path.stem

    # Create temp directory for this video
    video_output_dir = output_dir / f"patient_{patient_id:03d}_{video_name}"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames (with FFmpeg rotation preprocessing if needed)
    frames = extract_frames_from_video(video_path, num_frames=10, output_dir=video_output_dir)

    if len(frames) == 0:
        return []

    # Create subdirectories for frames and C++ output
    frames_dir = video_output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    cpp_dir = video_output_dir / "cpp_output"
    cpp_dir.mkdir(exist_ok=True)

    results = []

    for frame_data in frames:
        frame_idx = frame_data['frame_idx']
        frame = frame_data['frame']
        rotation = frame_data['rotation']

        # Save frame (already rotated)
        frame_filename = f"frame_{frame_idx:05d}.jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        # Get C++ OpenFace bbox
        cpp_data = run_cpp_openface_bbox(frame_path, cpp_dir)

        # Get RetinaFace bbox
        rf_data = run_retinaface_bbox(frame)

        # Only keep if both succeeded
        if cpp_data['success'] and rf_data['success']:
            result = {
                'patient_id': patient_id,
                'video_name': video_name,
                'video_path': str(video_path),
                'frame_idx': frame_idx,
                'frame_path': str(frame_path),
                'image_width': frame.shape[1],
                'image_height': frame.shape[0],
                'rotation': rotation,

                # C++ MTCNN (ground truth)
                'cpp_bbox_x': cpp_data['bbox'][0],
                'cpp_bbox_y': cpp_data['bbox'][1],
                'cpp_bbox_w': cpp_data['bbox'][2],
                'cpp_bbox_h': cpp_data['bbox'][3],
                'cpp_center_x': cpp_data['center_x'],
                'cpp_center_y': cpp_data['center_y'],
                'cpp_size': cpp_data['size'],
                'cpp_aspect_ratio': cpp_data['aspect_ratio'],
                'cpp_init_scale': cpp_data.get('init_scale', None),

                # RetinaFace (raw)
                'rf_bbox_x': rf_data['bbox'][0],
                'rf_bbox_y': rf_data['bbox'][1],
                'rf_bbox_w': rf_data['bbox'][2],
                'rf_bbox_h': rf_data['bbox'][3],
                'rf_center_x': rf_data['center_x'],
                'rf_center_y': rf_data['center_y'],
                'rf_size': rf_data['size'],
                'rf_aspect_ratio': rf_data['aspect_ratio'],
                'rf_confidence': rf_data['confidence'],

                # Differences (what we want to learn)
                'center_offset_x': cpp_data['center_x'] - rf_data['center_x'],
                'center_offset_y': cpp_data['center_y'] - rf_data['center_y'],
                'center_offset_total': np.sqrt(
                    (cpp_data['center_x'] - rf_data['center_x'])**2 +
                    (cpp_data['center_y'] - rf_data['center_y'])**2
                ),
                'width_diff': cpp_data['width'] - rf_data['width'],
                'height_diff': cpp_data['height'] - rf_data['height'],
                'size_ratio': cpp_data['size'] / rf_data['size'] if rf_data['size'] > 0 else 1.0
            }

            results.append(result)

    return results


def main():
    """Collect bbox dataset from all patient videos."""
    data_dir = Path("Patient Data")
    output_dir = Path("bbox_dataset")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("BBOX DATASET COLLECTION")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Find all videos
    print("Finding patient videos...")
    videos = find_patient_videos(data_dir)
    print(f"Found {len(videos)} patient videos")
    print()

    if len(videos) == 0:
        print("ERROR: No videos found!")
        return

    # Print first few for verification
    print("Sample videos:")
    for v in videos[:5]:
        print(f"  {v}")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")
    print()

    # Confirm
    print(f"Will extract 10 frames from each video = {len(videos) * 10} total frames")
    print("This will take approximately 2-3 hours.")
    print()

    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return

    print("\nStarting data collection...")
    print()

    # Process all videos
    all_results = []
    failed_videos = []

    for patient_id, video_path in enumerate(tqdm(videos, desc="Processing videos")):
        try:
            results = process_video(video_path, patient_id, output_dir)
            all_results.extend(results)

            if len(results) == 0:
                failed_videos.append(str(video_path))
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            failed_videos.append(str(video_path))
            continue

    # Save dataset
    print("\n" + "="*80)
    print("SAVING DATASET")
    print("="*80)

    if len(all_results) > 0:
        df = pd.DataFrame(all_results)

        # Save CSV
        csv_path = output_dir / "bbox_dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        print(f"  {len(df)} successful frames")

        # Save JSON (for easy loading)
        json_path = output_dir / "bbox_dataset.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"✓ Saved JSON: {json_path}")

        # Save summary statistics
        summary = {
            'total_videos': len(videos),
            'successful_videos': len(videos) - len(failed_videos),
            'failed_videos': len(failed_videos),
            'total_frames': len(df),
            'frames_per_video_mean': len(df) / (len(videos) - len(failed_videos)) if len(videos) > len(failed_videos) else 0,

            # Data statistics
            'cpp_size_mean': float(df['cpp_size'].mean()),
            'cpp_size_std': float(df['cpp_size'].std()),
            'cpp_size_min': float(df['cpp_size'].min()),
            'cpp_size_max': float(df['cpp_size'].max()),

            'rf_size_mean': float(df['rf_size'].mean()),
            'rf_size_std': float(df['rf_size'].std()),

            'center_offset_mean': float(df['center_offset_total'].mean()),
            'center_offset_std': float(df['center_offset_total'].std()),
            'center_offset_max': float(df['center_offset_total'].max()),

            'width_diff_mean': float(df['width_diff'].mean()),
            'height_diff_mean': float(df['height_diff'].mean()),
        }

        summary_path = output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary: {summary_path}")

        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(f"Total videos processed: {len(videos)}")
        print(f"Successful: {len(videos) - len(failed_videos)}")
        print(f"Failed: {len(failed_videos)}")
        print(f"Total frames collected: {len(df)}")
        print()

        print(f"Face size range (C++ MTCNN):")
        print(f"  Mean: {summary['cpp_size_mean']:.1f}px")
        print(f"  Std:  {summary['cpp_size_std']:.1f}px")
        print(f"  Min:  {summary['cpp_size_min']:.1f}px")
        print(f"  Max:  {summary['cpp_size_max']:.1f}px")
        print()

        print(f"RetinaFace vs C++ MTCNN offsets:")
        print(f"  Center offset mean: {summary['center_offset_mean']:.1f}px")
        print(f"  Center offset std:  {summary['center_offset_std']:.1f}px")
        print(f"  Center offset max:  {summary['center_offset_max']:.1f}px")
        print(f"  Width diff mean:    {summary['width_diff_mean']:.1f}px")
        print(f"  Height diff mean:   {summary['height_diff_mean']:.1f}px")
        print()

        if len(failed_videos) > 0:
            print("Failed videos:")
            for v in failed_videos[:10]:
                print(f"  {v}")
            if len(failed_videos) > 10:
                print(f"  ... and {len(failed_videos) - 10} more")

        print("\n" + "="*80)
        print("✅ DATASET COLLECTION COMPLETE")
        print("="*80)
        print(f"Dataset ready for model training!")
        print(f"Next step: Run model training script on {csv_path}")

    else:
        print("ERROR: No successful frames collected!")


if __name__ == "__main__":
    main()
