#!/usr/bin/env python3
"""
Generate training data using C++ OpenFace for landmarks (ground truth) - BigRed200 version.
This avoids "baking in" pyCLNF landmark errors into the neural network.

Usage:
    python generate_cpp_training_data_bigred.py --video-index 0 --video-dir "S Data" --output-dir cpp_training_data
"""
import cv2
import numpy as np
import subprocess
import tempfile
import argparse
import h5py
from pathlib import Path
import pandas as pd
import sys
import json
import os

# Add paths for BigRed200
sys.path.insert(0, '/N/u/jw411/BigRed200/pyfaceau/pyfaceau')
sys.path.insert(0, '/N/u/jw411/BigRed200/pyfaceau/pymtcnn')
sys.path.insert(0, '/N/u/jw411/BigRed200/pyfaceau/pyclnf')

from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.features.triangulation import TriangulationParser


def get_video_list(data_dir="S Data", sort_by_frames=True):
    """Get all video files from S Data directory, sorted by frame count (longest first)."""
    videos = []
    data_path = Path(data_dir)

    # Normal Cohort
    normal_dir = data_path / "Normal Cohort"
    if normal_dir.exists():
        mov_files = list(normal_dir.glob("*.MOV")) + list(normal_dir.glob("*.mov"))
        for f in sorted(set(mov_files), key=lambda x: x.name.lower()):
            videos.append(("Normal Cohort", f.name))

    # Paralysis Cohort
    paralysis_dir = data_path / "Paralysis Cohort"
    if paralysis_dir.exists():
        mov_files = list(paralysis_dir.glob("*.MOV")) + list(paralysis_dir.glob("*.mov"))
        for f in sorted(set(mov_files), key=lambda x: x.name.lower()):
            videos.append(("Paralysis Cohort", f.name))

    # Sort by frame count if requested
    if sort_by_frames and videos:
        videos_with_frames = []
        for cohort, video_name in videos:
            video_path = str(data_path / cohort / video_name)
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except:
                frame_count = 0
            videos_with_frames.append((cohort, video_name, frame_count))

        videos_with_frames.sort(key=lambda x: x[2], reverse=True)
        videos = [(cohort, name) for cohort, name, _ in videos_with_frames]

    return videos


def get_video_rotation(video_path: str) -> int:
    """Get video rotation from metadata."""
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


def run_openface(video_path: str, openface_bin: str) -> pd.DataFrame:
    """Run C++ OpenFace on a video and return the results as a DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            openface_bin,
            "-f", video_path,
            "-out_dir", tmpdir,
            "-2Dfp",   # 2D landmarks
            "-pose",   # Head pose
            "-aus",    # Action units
            "-wild",   # Better for in-the-wild
        ]

        print(f"Running: {' '.join(cmd[:3])}...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"OpenFace error: {result.stderr[:500]}", flush=True)
            return None

        # Find output CSV
        csv_files = list(Path(tmpdir).glob("*.csv"))
        if not csv_files:
            print("No CSV output found", flush=True)
            return None

        df = pd.read_csv(csv_files[0])
        # Strip whitespace from column names (C++ OpenFace adds spaces)
        df.columns = df.columns.str.strip()
        return df


def extract_landmarks(row: pd.Series) -> np.ndarray:
    """Extract 68 landmarks from a DataFrame row."""
    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']
    return landmarks


def extract_pose(row: pd.Series) -> np.ndarray:
    """Extract pose parameters from a DataFrame row."""
    return np.array([
        row['pose_Tx'],
        row['pose_Ty'],
        row['pose_Tz'],
        row['pose_Rx'],
        row['pose_Ry'],
        row['pose_Rz'],
    ])


def extract_aus(row: pd.Series) -> np.ndarray:
    """Extract 17 AU intensities from a DataFrame row."""
    au_names = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    return np.array([row[au] for au in au_names])


def compute_pose_translation(landmarks: np.ndarray, rigid_indices: list) -> tuple:
    """Compute pose_tx and pose_ty from landmark centroid."""
    rigid_landmarks = landmarks[rigid_indices]
    centroid = np.mean(rigid_landmarks, axis=0)
    return centroid[0], centroid[1]


def process_video(video_path: str, output_dir: Path, openface_bin: str, video_index: int):
    """Process a single video and generate training data using C++ OpenFace landmarks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {video_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Step 1: Run C++ OpenFace
    print("\n[1/3] Running C++ OpenFace...", flush=True)
    df = run_openface(video_path, openface_bin)
    if df is None:
        return None

    print(f"      Total frames: {len(df)}", flush=True)
    success_mask = df['success'] == 1
    print(f"      Successful: {success_mask.sum()} ({100*success_mask.mean():.1f}%)", flush=True)

    # Step 2: Open video and process frames
    print("\n[2/3] Processing frames...", flush=True)
    cap = cv2.VideoCapture(video_path)

    # Get video rotation
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"      Detected video rotation: {rotation} degrees", flush=True)

    # Initialize aligner
    pdm_file = '/N/u/jw411/BigRed200/pyfaceau/S1 Face Mirror/weights/In-the-wild_aligned_PDM_68.txt'
    triangulation_file = '/N/u/jw411/BigRed200/pyfaceau/pyfaceau/alignment/tris_68_full.txt'

    aligner = OpenFace22FaceAligner(pdm_file)
    triangulation = TriangulationParser(triangulation_file)

    rigid_indices = aligner.RIGID_INDICES

    # Storage
    aligned_faces = []
    landmarks_list = []
    landmarks_aligned_list = []
    poses_list = []
    aus_list = []
    warp_matrices = []
    frame_indices = []
    confidences = []

    # Process each successful frame
    df_success = df[success_mask].reset_index(drop=True)

    for idx, row in df_success.iterrows():
        frame_num = int(row['frame']) - 1  # OpenFace is 1-indexed

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        if rotation != 0:
            frame = apply_frame_rotation(frame, rotation)

        landmarks = extract_landmarks(row)
        pose = extract_pose(row)
        aus = extract_aus(row)
        confidence = row['confidence']

        pose_tx, pose_ty = compute_pose_translation(landmarks, rigid_indices)
        pose_rz = pose[5]

        try:
            aligned_face, warp_matrix = aligner.align_face_with_matrix(
                image=frame,
                landmarks_68=landmarks,
                pose_tx=pose_tx,
                pose_ty=pose_ty,
                p_rz=pose_rz,
                apply_mask=True,
                triangulation=triangulation
            )

            landmarks_homo = np.hstack([landmarks, np.ones((68, 1))])
            landmarks_aligned = landmarks_homo @ warp_matrix.T

            aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

            aligned_faces.append(aligned_face_rgb)
            landmarks_list.append(landmarks.astype(np.float32))
            landmarks_aligned_list.append(landmarks_aligned.astype(np.float32))
            poses_list.append(pose.astype(np.float32))
            aus_list.append(aus.astype(np.float32))
            warp_matrices.append(warp_matrix.astype(np.float32))
            frame_indices.append(frame_num)
            confidences.append(confidence)

        except Exception as e:
            print(f"      Frame {frame_num}: alignment failed - {e}", flush=True)
            continue

        if (idx + 1) % 200 == 0:
            print(f"      Processed {idx + 1}/{len(df_success)} frames", flush=True)

    cap.release()

    print(f"      Successfully processed: {len(aligned_faces)} frames", flush=True)

    if len(aligned_faces) == 0:
        print("      No frames processed!", flush=True)
        return None

    # Step 3: Save to HDF5
    print("\n[3/3] Saving to HDF5...", flush=True)
    h5_path = output_dir / f'cpp_training_video_{video_index:03d}.h5'

    AU_NAMES = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('aligned_faces', data=np.array(aligned_faces, dtype=np.uint8),
                        compression='gzip', compression_opts=4)
        f.create_dataset('landmarks', data=np.array(landmarks_list, dtype=np.float32))
        f.create_dataset('landmarks_aligned', data=np.array(landmarks_aligned_list, dtype=np.float32))
        f.create_dataset('pose_params', data=np.array(poses_list, dtype=np.float32))
        f.create_dataset('au_intensities', data=np.array(aus_list, dtype=np.float32))
        f.create_dataset('warp_matrices', data=np.array(warp_matrices, dtype=np.float32))
        f.create_dataset('frame_indices', data=np.array(frame_indices, dtype=np.int32))
        f.create_dataset('confidences', data=np.array(confidences, dtype=np.float32))

        f.attrs['video_path'] = str(video_path)
        f.attrs['video_name'] = video_name
        f.attrs['video_index'] = video_index
        f.attrs['source'] = 'C++ OpenFace'
        f.attrs['num_samples'] = len(aligned_faces)
        f.attrs['au_names'] = AU_NAMES

    file_size = os.path.getsize(h5_path) / 1024 / 1024
    print(f"      Saved: {h5_path} ({file_size:.1f} MB)", flush=True)

    return {
        'num_frames': len(aligned_faces),
        'h5_path': str(h5_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate C++ training data (BigRed200)')
    parser.add_argument('--video-index', type=int, required=True, help='Video index')
    parser.add_argument('--video-dir', default='S Data', help='Video directory')
    parser.add_argument('--output-dir', default='cpp_training_data', help='Output directory')
    parser.add_argument('--openface', type=str, required=True, help='Path to OpenFace binary')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    # Get video list
    videos = get_video_list(args.video_dir)
    print(f"Found {len(videos)} videos", flush=True)

    if args.video_index >= len(videos):
        print(f"ERROR: Video index {args.video_index} out of range (max {len(videos)-1})", flush=True)
        sys.exit(1)

    cohort, video_name = videos[args.video_index]
    video_path = os.path.join(args.video_dir, cohort, video_name)

    print(f"Video {args.video_index}/{len(videos)}: {video_name}", flush=True)
    print(f"Cohort: {cohort}", flush=True)

    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}", flush=True)
        sys.exit(1)

    result = process_video(
        video_path,
        Path(args.output_dir),
        args.openface,
        args.video_index
    )

    if result:
        print(f"\n{'='*60}", flush=True)
        print("SUCCESS", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Processed {result['num_frames']} frames", flush=True)
        print(f"Output: {result['h5_path']}", flush=True)


if __name__ == '__main__':
    main()
