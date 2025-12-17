#!/usr/bin/env python3
"""
Generate Training Data for Neural Network Pipeline

Processes videos through the full Python pipeline and extracts:
- Aligned face crops (112x112)
- 68 landmarks (x, y)
- Pose parameters (scale, rx, ry, rz, tx, ty)
- Local PDM parameters (34 values)
- AU intensities (17 values)

Output: Per-video HDF5 files that can be merged for training.
"""
import argparse
import numpy as np
import cv2
import h5py
import os
import sys
import time
from pathlib import Path

def get_frame_count(video_path: str) -> int:
    """Get frame count for a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    except:
        return 0


# Video list will be generated dynamically
def get_video_list(data_dir="S Data", sort_by_frames=True):
    """Get all video files from S Data directory, optionally sorted by frame count (longest first)."""
    videos = []
    data_path = Path(data_dir)

    # Normal Cohort (handle both .MOV and .mov)
    normal_dir = data_path / "Normal Cohort"
    if normal_dir.exists():
        mov_files = list(normal_dir.glob("*.MOV")) + list(normal_dir.glob("*.mov"))
        for f in sorted(set(mov_files), key=lambda x: x.name.lower()):
            videos.append(("Normal Cohort", f.name))

    # Paralysis Cohort (handle both .MOV and .mov)
    paralysis_dir = data_path / "Paralysis Cohort"
    if paralysis_dir.exists():
        mov_files = list(paralysis_dir.glob("*.MOV")) + list(paralysis_dir.glob("*.mov"))
        for f in sorted(set(mov_files), key=lambda x: x.name.lower()):
            videos.append(("Paralysis Cohort", f.name))

    # Sort by frame count (longest first) so longest jobs start first
    if sort_by_frames and videos:
        videos_with_frames = []
        for cohort, video_name in videos:
            video_path = str(data_path / cohort / video_name)
            frame_count = get_frame_count(video_path)
            videos_with_frames.append((cohort, video_name, frame_count))

        # Sort by frame count descending
        videos_with_frames.sort(key=lambda x: x[2], reverse=True)
        videos = [(cohort, name) for cohort, name, _ in videos_with_frames]

    return videos


def get_video_rotation(video_path: str) -> int:
    """Get video rotation from metadata."""
    import subprocess
    import json
    try:
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{video_path}"'
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        metadata = json.loads(output)
        for stream in metadata.get('streams', []):
            rotation = stream.get('tags', {}).get('rotate')
            if rotation is None:
                rotation = stream.get('rotation')
            if rotation:
                return int(rotation)
    except:
        pass
    return 0


def apply_frame_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation correction to a frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Generate training data from video')
    parser.add_argument('--video-index', type=int, required=True, help='Video index')
    parser.add_argument('--video-dir', default='S Data', help='Video directory')
    parser.add_argument('--output-dir', default='training_data', help='Output directory')
    parser.add_argument('--stagger-delay', type=int, default=0, help='Stagger delay in seconds')
    args = parser.parse_args()

    # Stagger initialization
    if args.stagger_delay > 0:
        delay = (args.video_index % 20) * args.stagger_delay  # Stagger in groups of 20
        print(f"Staggering initialization by {delay} seconds...")
        sys.stdout.flush()
        time.sleep(delay)

    print("=" * 70)
    print("TRAINING DATA GENERATION")
    print("=" * 70)
    sys.stdout.flush()

    # Get video list
    videos = get_video_list(args.video_dir)
    if args.video_index >= len(videos):
        print(f"ERROR: Video index {args.video_index} out of range (max {len(videos)-1})")
        sys.exit(1)

    cohort, video_name = videos[args.video_index]
    video_path = os.path.join(args.video_dir, cohort, video_name)

    print(f"\nVideo {args.video_index}/{len(videos)}: {video_name}")
    print(f"Cohort: {cohort}")
    print(f"Path: {video_path}")
    sys.stdout.flush()

    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    sys.stdout.flush()

    from pyfaceau import FullPythonAUPipeline

    base_path = os.environ.get('PYFACEAU_BASE', os.path.expanduser('~/pyfaceau'))
    weights_path = f'{base_path}/pyfaceau/weights'

    pipeline = FullPythonAUPipeline(
        pdm_file=f'{weights_path}/In-the-wild_aligned_PDM_68.txt',
        au_models_dir=f'{weights_path}/AU_predictors',
        triangulation_file=f'{weights_path}/tris_68_full.txt',
        patch_expert_file=f'{weights_path}/patch_experts/cen_patches_0.25_of.dat',
        track_faces=True,
        verbose=False,
        debug_mode=True  # Enable to get landmarks and params
    )

    print("Pipeline initialized!")
    sys.stdout.flush()

    # Get video rotation
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"Detected video rotation: {rotation}°")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {fps:.1f} FPS, {total_frames} frames")
    sys.stdout.flush()

    # Storage for training data
    aligned_faces = []
    landmarks = []
    pose_params = []  # [scale, rx, ry, rz, tx, ty]
    local_params = []  # 34 PDM params
    au_intensities = []  # 17 AUs
    frame_indices = []
    confidences = []
    warp_matrices = []  # (2, 3) affine transform from original frame to aligned face

    # AU names for reference
    AU_NAMES = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

    # Process frames
    frame_idx = 0
    successful_frames = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation correction
            if rotation != 0:
                frame = apply_frame_rotation(frame, rotation)

            timestamp = frame_idx / fps

            # Process frame with debug mode
            result = pipeline._process_frame(frame, frame_idx, timestamp, return_debug=True)

            if result.get('success') and 'debug_info' in result:
                debug_info = result['debug_info']

                # Extract landmarks
                lm = debug_info.get('landmark_detection', {}).get('landmarks_68')
                if lm is None:
                    frame_idx += 1
                    continue

                # Extract aligned face
                aligned = debug_info.get('alignment', {}).get('aligned_face_shape')
                # We need to actually get the aligned face from the aligner
                # For now, we'll store the frame crop based on landmarks

                # Get pose from debug info
                pose_info = debug_info.get('pose_estimation', {})
                scale = pose_info.get('scale', 1.0)
                rotation_xyz = pose_info.get('rotation', [0, 0, 0])
                translation = pose_info.get('translation', [0, 0])

                # Get local params (if available in the pipeline's landmark detector)
                # These come from the CLNF optimization
                local_p = np.zeros(34, dtype=np.float32)
                if hasattr(pipeline, 'landmark_detector') and hasattr(pipeline.landmark_detector, '_last_params'):
                    params = pipeline.landmark_detector._last_params
                    if params is not None and len(params) > 6:
                        local_p = params[6:].astype(np.float32)

                # Extract AU values
                au_vals = np.array([result.get(au, 0.0) for au in AU_NAMES], dtype=np.float32)

                # Get aligned face using the aligner (with mask like OpenFace uses)
                # Also get the warp_matrix for transforming landmarks to aligned face space
                try:
                    aligned_face, warp_matrix = pipeline.face_aligner.align_face_with_matrix(
                        image=frame,
                        landmarks_68=lm,
                        pose_tx=translation[0],
                        pose_ty=translation[1],
                        p_rz=rotation_xyz[2],
                        apply_mask=True,  # Apply triangulation mask like OpenFace
                        triangulation=pipeline.triangulation
                    )
                    # Convert BGR to RGB for storage
                    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                except:
                    frame_idx += 1
                    continue

                # Store data
                aligned_faces.append(aligned_face)
                landmarks.append(lm.astype(np.float32))
                pose_params.append(np.array([scale] + rotation_xyz + translation, dtype=np.float32))
                local_params.append(local_p)
                au_intensities.append(au_vals)
                frame_indices.append(frame_idx)
                confidences.append(1.0)  # Could use detection confidence
                warp_matrices.append(warp_matrix.astype(np.float32))

                successful_frames += 1

            # Progress update
            if (frame_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (frame_idx + 1) / elapsed
                print(f"  Progress: {frame_idx + 1}/{total_frames} ({rate:.1f} FPS), {successful_frames} samples", flush=True)

            frame_idx += 1

    finally:
        cap.release()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s")
    print(f"Successful samples: {successful_frames}")
    sys.stdout.flush()

    if successful_frames == 0:
        print("ERROR: No successful frames extracted!")
        sys.exit(1)

    # Apply post-processing to AU intensities (like pipeline.finalize_predictions)
    print("\nApplying AU post-processing (smoothing, median correction)...")
    sys.stdout.flush()

    import pandas as pd

    # Create DataFrame with raw AU values
    au_df_data = {'frame': frame_indices, 'success': [True] * len(frame_indices)}
    for i, au_name in enumerate(AU_NAMES):
        au_df_data[au_name] = [au_intensities[j][i] for j in range(len(au_intensities))]

    raw_au_df = pd.DataFrame(au_df_data)

    # Apply finalize_predictions (two-pass, smoothing, cutoff)
    processed_au_df = pipeline.finalize_predictions(raw_au_df)

    # Extract post-processed AU values back to array
    au_intensities_processed = []
    for idx in range(len(processed_au_df)):
        au_vals = np.array([processed_au_df.iloc[idx][au] for au in AU_NAMES], dtype=np.float32)
        au_intensities_processed.append(au_vals)

    print(f"  Post-processing complete")
    sys.stdout.flush()

    # Save to HDF5
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'training_video_{args.video_index:03d}.h5')

    print(f"\nSaving to {output_file}...")
    sys.stdout.flush()

    with h5py.File(output_file, 'w') as f:
        # Store arrays
        f.create_dataset('aligned_faces', data=np.array(aligned_faces, dtype=np.uint8),
                        compression='gzip', compression_opts=4)
        f.create_dataset('landmarks', data=np.array(landmarks, dtype=np.float32))
        f.create_dataset('pose_params', data=np.array(pose_params, dtype=np.float32))
        f.create_dataset('local_params', data=np.array(local_params, dtype=np.float32))
        f.create_dataset('au_intensities', data=np.array(au_intensities_processed, dtype=np.float32))
        f.create_dataset('frame_indices', data=np.array(frame_indices, dtype=np.int32))
        f.create_dataset('confidences', data=np.array(confidences, dtype=np.float32))
        f.create_dataset('warp_matrices', data=np.array(warp_matrices, dtype=np.float32))

        # Metadata
        f.attrs['video_name'] = video_name
        f.attrs['cohort'] = cohort
        f.attrs['video_index'] = args.video_index
        f.attrs['total_frames'] = frame_idx
        f.attrs['successful_frames'] = successful_frames
        f.attrs['fps'] = fps
        f.attrs['au_names'] = AU_NAMES

    print(f"Saved {successful_frames} samples to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 70)
    print(f"COMPLETED: Video {args.video_index} ({video_name})")
    print("=" * 70)


if __name__ == '__main__':
    main()
