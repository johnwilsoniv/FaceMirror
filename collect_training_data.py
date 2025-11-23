#!/usr/bin/env python3
"""
Training Data Collection for Neural Network AU Pipeline

Processes all videos through the current Python pipeline and saves:
1. AU MLP training data: HOG features + geometric features → AU predictions
2. Landmark CNN training data: face crops → CLNF landmarks

Usage:
    python collect_training_data.py --data-dir "Patient Data" --output-dir "training_data"

Output structure:
    training_data/
    ├── au_features/
    │   ├── video_001.h5  (hog_features, geom_features, au_predictions)
    │   └── ...
    ├── landmarks/
    │   ├── video_001.h5  (face_crops, landmarks, bboxes)
    │   └── ...
    └── metadata.json
"""

import cv2
import numpy as np
import h5py
import json
import time
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

# Import video rotation logic
from importlib.util import spec_from_file_location, module_from_spec
rotation_spec = spec_from_file_location("video_rotation", "S1 Face Mirror/video_rotation.py")
video_rotation = module_from_spec(rotation_spec)
rotation_spec.loader.exec_module(video_rotation)


def find_videos(data_dir: str) -> list:
    """Find all video files in directory."""
    data_path = Path(data_dir)
    extensions = ['.mov', '.MOV', '.mp4', '.MP4', '.avi', '.AVI']

    videos = []
    for ext in extensions:
        videos.extend(data_path.rglob(f'*{ext}'))

    return sorted(videos)


def get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video."""
    return video_rotation.get_video_rotation(video_path)


def extract_face_crop(frame: np.ndarray, bbox: tuple, landmarks: np.ndarray,
                      output_size: int = 256) -> tuple:
    """
    Extract face crop for landmark CNN training.

    Returns:
        face_crop: RGB face crop (output_size, output_size, 3)
        adjusted_landmarks: Landmarks adjusted to crop coordinates (68, 2)
    """
    x, y, w, h = bbox

    # Expand bbox by 20% for context
    expand = 0.2
    cx, cy = x + w/2, y + h/2
    size = max(w, h) * (1 + expand)

    # Calculate crop region
    x1 = int(cx - size/2)
    y1 = int(cy - size/2)
    x2 = int(cx + size/2)
    y2 = int(cy + size/2)

    # Handle boundary cases
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - frame.shape[1])
    pad_bottom = max(0, y2 - frame.shape[0])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    # Extract and pad
    crop = frame[y1:y2, x1:x2]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=0)

    # Resize to output size
    crop_resized = cv2.resize(crop, (output_size, output_size))

    # Adjust landmarks to crop coordinates
    scale = output_size / size
    adjusted_landmarks = landmarks.copy()
    adjusted_landmarks[:, 0] = (landmarks[:, 0] - (cx - size/2)) * scale
    adjusted_landmarks[:, 1] = (landmarks[:, 1] - (cy - size/2)) * scale

    return crop_resized, adjusted_landmarks


def process_video(video_path: Path, pipeline, output_dir: Path,
                  video_idx: int, total_videos: int) -> dict:
    """
    Process a single video and save training data.
    Uses incremental HDF5 writing to avoid memory issues with long videos.

    Returns:
        stats: Dictionary with processing statistics
    """
    video_name = video_path.stem
    print(f"\n[{video_idx+1}/{total_videos}] Processing: {video_name}")

    # Check for rotation
    rotation = get_video_rotation(str(video_path))
    if rotation in [90, 180, 270]:
        print(f"  Video needs rotation: {rotation}°")
        # Create temp rotated video
        temp_path = output_dir / f"temp_{video_name}.mov"
        video_rotation.auto_rotate_video(str(video_path), str(temp_path))
        video_path = temp_path
        cleanup_temp = True
    else:
        cleanup_temp = False

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open video")
        return {'frames': 0, 'success': 0, 'failed': 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    # Create output files with resizable datasets for incremental writing
    au_file = output_dir / 'au_features' / f'{video_name}.h5'
    lm_file = output_dir / 'landmarks' / f'{video_name}.h5'
    au_file.parent.mkdir(parents=True, exist_ok=True)
    lm_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize HDF5 files with resizable datasets
    au_h5 = h5py.File(au_file, 'w')
    lm_h5 = h5py.File(lm_file, 'w')

    # Create resizable datasets (unknown final size)
    au_h5.create_dataset('hog_features', shape=(0, 4464), maxshape=(None, 4464),
                         dtype='float32', chunks=(100, 4464))
    au_h5.create_dataset('geom_features', shape=(0, 238), maxshape=(None, 238),
                         dtype='float32', chunks=(100, 238))
    au_h5.create_dataset('au_predictions', shape=(0, 17), maxshape=(None, 17),
                         dtype='float32', chunks=(100, 17))
    au_h5.create_dataset('frame_indices', shape=(0,), maxshape=(None,),
                         dtype='int32', chunks=(100,))

    lm_h5.create_dataset('face_crops', shape=(0, 256, 256, 3), maxshape=(None, 256, 256, 3),
                         dtype='uint8', chunks=(10, 256, 256, 3))
    lm_h5.create_dataset('landmarks', shape=(0, 68, 2), maxshape=(None, 68, 2),
                         dtype='float32', chunks=(100, 68, 2))
    lm_h5.create_dataset('bboxes', shape=(0, 4), maxshape=(None, 4),
                         dtype='float32', chunks=(100, 4))
    lm_h5.create_dataset('frame_indices', shape=(0,), maxshape=(None,),
                         dtype='int32', chunks=(100,))

    # Batch buffers for efficient writing
    BATCH_SIZE = 100
    au_batch = {'hog': [], 'geom': [], 'aus': [], 'idx': []}
    lm_batch = {'crops': [], 'lm': [], 'bbox': [], 'idx': []}

    def flush_au_batch():
        """Write AU batch to HDF5 and clear buffers."""
        if not au_batch['hog']:
            return
        n = len(au_batch['hog'])
        old_size = au_h5['hog_features'].shape[0]
        new_size = old_size + n

        au_h5['hog_features'].resize(new_size, axis=0)
        au_h5['geom_features'].resize(new_size, axis=0)
        au_h5['au_predictions'].resize(new_size, axis=0)
        au_h5['frame_indices'].resize(new_size, axis=0)

        au_h5['hog_features'][old_size:new_size] = np.array(au_batch['hog'])
        au_h5['geom_features'][old_size:new_size] = np.array(au_batch['geom'])
        au_h5['au_predictions'][old_size:new_size] = np.array(au_batch['aus'])
        au_h5['frame_indices'][old_size:new_size] = np.array(au_batch['idx'])

        au_batch['hog'].clear()
        au_batch['geom'].clear()
        au_batch['aus'].clear()
        au_batch['idx'].clear()

    def flush_lm_batch():
        """Write landmark batch to HDF5 and clear buffers."""
        if not lm_batch['crops']:
            return
        n = len(lm_batch['crops'])
        old_size = lm_h5['face_crops'].shape[0]
        new_size = old_size + n

        lm_h5['face_crops'].resize(new_size, axis=0)
        lm_h5['landmarks'].resize(new_size, axis=0)
        lm_h5['bboxes'].resize(new_size, axis=0)
        lm_h5['frame_indices'].resize(new_size, axis=0)

        lm_h5['face_crops'][old_size:new_size] = np.array(lm_batch['crops'])
        lm_h5['landmarks'][old_size:new_size] = np.array(lm_batch['lm'])
        lm_h5['bboxes'][old_size:new_size] = np.array(lm_batch['bbox'])
        lm_h5['frame_indices'][old_size:new_size] = np.array(lm_batch['idx'])

        lm_batch['crops'].clear()
        lm_batch['lm'].clear()
        lm_batch['bbox'].clear()
        lm_batch['idx'].clear()

    # Process frames
    frame_idx = 0
    success_count = 0
    failed_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Process through pipeline with debug mode to get features
            result = pipeline._process_frame(frame, frame_idx=frame_idx,
                                            timestamp=frame_idx/fps if fps > 0 else 0,
                                            return_debug=True)

            if result is None or not result.get('success', False):
                failed_count += 1
                frame_idx += 1
                continue

            # Extract features from stored_features (pipeline stores them internally)
            # The last entry in stored_features is from this frame
            if hasattr(pipeline, 'stored_features') and pipeline.stored_features:
                last_stored = pipeline.stored_features[-1]
                if last_stored[0] == frame_idx:
                    _, hog_features, geom_features = last_stored

                    au_batch['hog'].append(hog_features.copy())
                    au_batch['geom'].append(geom_features.copy())

                    # Get AU predictions from result
                    aus = np.array([result.get(f'AU{au:02d}_r', 0.0)
                                   for au in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]])
                    au_batch['aus'].append(aus)
                    au_batch['idx'].append(frame_idx)

                    # Flush if batch is full
                    if len(au_batch['hog']) >= BATCH_SIZE:
                        flush_au_batch()

            # Extract landmark training data from debug_info
            debug_info = result.get('debug_info', {})
            lm_info = debug_info.get('landmark_detection', {})
            det_info = debug_info.get('face_detection', {})

            if 'landmarks_68' in lm_info and 'bbox' in det_info:
                landmarks = lm_info['landmarks_68']
                bbox = det_info['bbox']

                # Extract face crop
                face_crop, adjusted_landmarks = extract_face_crop(
                    frame, bbox, landmarks, output_size=256
                )

                lm_batch['crops'].append(face_crop)
                lm_batch['lm'].append(adjusted_landmarks)
                lm_batch['bbox'].append(bbox)
                lm_batch['idx'].append(frame_idx)

                # Flush if batch is full
                if len(lm_batch['crops']) >= BATCH_SIZE:
                    flush_lm_batch()

            success_count += 1

        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print(f"  Frame {frame_idx} failed: {e}")

        frame_idx += 1

        # Progress update
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Progress: {frame_idx}/{total_frames} ({fps_actual:.1f} fps)")

    cap.release()

    # Flush remaining batches
    flush_au_batch()
    flush_lm_batch()

    # Cleanup temp file
    if cleanup_temp and Path(video_path).exists():
        Path(video_path).unlink()

    # Get final counts and close HDF5 files
    au_count = au_h5['hog_features'].shape[0]
    lm_count = lm_h5['face_crops'].shape[0]

    au_h5.close()
    lm_h5.close()

    # Remove empty files
    if au_count == 0:
        au_file.unlink()
    else:
        print(f"  Saved AU data: {au_count} samples")

    if lm_count == 0:
        lm_file.unlink()
    else:
        print(f"  Saved landmark data: {lm_count} samples")

    elapsed = time.time() - start_time
    print(f"  Complete: {success_count} success, {failed_count} failed, {elapsed:.1f}s")

    return {
        'frames': frame_idx,
        'success': success_count,
        'failed': failed_count,
        'time': elapsed
    }


def main():
    parser = argparse.ArgumentParser(description="Collect training data for neural network AU pipeline")
    parser.add_argument("--data-dir", type=str, default="Patient Data",
                       help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, default="training_data",
                       help="Output directory for training data")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process")
    args = parser.parse_args()

    print("=" * 80)
    print("NEURAL NETWORK TRAINING DATA COLLECTION")
    print("=" * 80)

    # Find videos
    videos = find_videos(args.data_dir)
    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"\nFound {len(videos)} videos in {args.data_dir}")

    if len(videos) == 0:
        print("ERROR: No videos found")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    from pyfaceau.pipeline import FullPythonAUPipeline

    pipeline = FullPythonAUPipeline(
        pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
        au_models_dir="pyfaceau/weights/AU_predictors",
        triangulation_file="pyfaceau/weights/tris_68_full.txt",
        verbose=False,
        debug_mode=False
    )

    # Process all videos
    all_stats = []
    total_start = time.time()

    for idx, video_path in enumerate(videos):
        stats = process_video(video_path, pipeline, output_dir, idx, len(videos))
        stats['video'] = video_path.name
        all_stats.append(stats)

    total_elapsed = time.time() - total_start

    # Save metadata
    metadata = {
        'num_videos': len(videos),
        'total_frames': sum(s['frames'] for s in all_stats),
        'total_success': sum(s['success'] for s in all_stats),
        'total_failed': sum(s['failed'] for s in all_stats),
        'total_time': total_elapsed,
        'videos': [str(v) for v in videos],
        'stats': all_stats
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"\nVideos processed: {len(videos)}")
    print(f"Total frames: {metadata['total_frames']}")
    print(f"Successful: {metadata['total_success']}")
    print(f"Failed: {metadata['total_failed']}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Average FPS: {metadata['total_frames']/total_elapsed:.1f}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
