#!/usr/bin/env python3
"""
S1 Face Mirror - Optimized Batch Processor for BigRed200

FULL PARITY with local S1_FaceMirror pipeline:
1. Video rotation (auto-detect and apply using ffmpeg) -> saves _source video
2. Face detection with bbox calibration (MTCNN -> CLNF mapping)
3. Landmark detection with temporal smoothing (5-frame weighted average)
4. Anatomical midline mirroring with gradient blending (6-pixel)
5. AU extraction on both mirrored versions

Usage: python s1_batch_process.py <video_path> --output-dir <output_dir>
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
import subprocess
import threading
from queue import Empty
import gc
import json
import shutil

# Add paths for local development (handles both local and BigRed200)
parent = Path(__file__).parent.parent
for name in ["S1_FaceMirror", "S1 Face Mirror", "pyclnf", "pymtcnn", "pyfaceau", "pyfhog"]:
    path = parent / name
    if path.exists():
        sys.path.insert(0, str(path))

# Configuration
BATCH_SIZE = 100
DETECTION_INTERVAL = 30  # Re-detect face every N frames
HISTORY_SIZE = 5  # Temporal smoothing window


# =============================================================================
# VIDEO ROTATION (matches video_rotation.py)
# =============================================================================

def get_video_rotation(input_path):
    """Get video rotation from metadata using ffprobe."""
    commands = [
        f'ffprobe -v quiet -print_format json -show_streams "{input_path}"',
        f'ffprobe -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 "{input_path}"',
    ]

    for command in commands:
        try:
            output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()

            if 'json' in command:
                try:
                    metadata = json.loads(output)
                    for stream in metadata.get('streams', []):
                        rotation = stream.get('tags', {}).get('rotate')
                        if rotation is None:
                            rotation = stream.get('rotation')
                        if rotation is not None:
                            return int(rotation)

                        # Check displaymatrix in side data
                        if 'side_data_list' in stream:
                            for side_data in stream['side_data_list']:
                                if 'rotation' in str(side_data).lower():
                                    if '-90' in str(side_data):
                                        return -90
                                    elif '90' in str(side_data):
                                        return 90
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                try:
                    return int(output)
                except ValueError:
                    continue
        except subprocess.CalledProcessError:
            continue

    # Check if portrait orientation (mobile video)
    try:
        dim_cmd = f'ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height -of json "{input_path}"'
        dim_output = subprocess.check_output(dim_cmd, shell=True, universal_newlines=True).strip()
        dim_data = json.loads(dim_output)

        if dim_data.get('streams'):
            width = int(dim_data['streams'][0].get('width', 0))
            height = int(dim_data['streams'][0].get('height', 0))

            if height > width * 1.2:
                print(f"  Portrait orientation detected ({width}x{height}), assuming -90 rotation")
                return -90
    except Exception:
        pass

    return 0


def normalize_rotation(rotation):
    """Normalize rotation to 0, 90, 180, 270."""
    if rotation < 0:
        rotation = 360 + rotation
    return round(rotation / 90) * 90 % 360


def rotate_video_with_ffmpeg(input_path, output_path):
    """Rotate video using ffmpeg's auto-rotation feature."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    rotation = get_video_rotation(str(input_path))
    normalized = normalize_rotation(rotation)

    print(f"  Detected rotation: {rotation}°, Normalized: {normalized}°")

    if normalized in [90, 180, 270]:
        print(f"  Rotating video with ffmpeg...")

        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-map', '0',
            '-map_metadata', '0',
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"  Rotation complete: {output_path.name}")
                return output_path
            else:
                print(f"  FFmpeg rotation failed: {result.stderr[:300]}")
                shutil.copy2(input_path, output_path)
                return output_path
        except subprocess.TimeoutExpired:
            print("  FFmpeg rotation timeout!")
            shutil.copy2(input_path, output_path)
            return output_path
    else:
        print("  No rotation needed, copying original...")
        shutil.copy2(input_path, output_path)
        return output_path


# =============================================================================
# FACE DETECTION & LANDMARKS (matches pyfaceau_detector.py)
# =============================================================================

class LandmarkDetector:
    """
    Face detection and 68-point landmark tracking with FULL PARITY to local pipeline.

    Features:
    - MTCNN face detection with bbox calibration coefficients
    - CLNF landmark detection with video convergence profile
    - 5-frame weighted temporal smoothing for landmarks
    - 5-frame simple average for midline points
    """

    def __init__(self):
        from pymtcnn import MTCNN
        from pyclnf import CLNF

        # Initialize MTCNN for face detection
        self.face_detector = MTCNN(backend='auto')
        backend_info = self.face_detector.get_backend_info()
        print(f"  MTCNN: {backend_info.get('backend', 'unknown')}")

        # Initialize CLNF with video-optimized settings (matches local)
        self.clnf = CLNF(
            convergence_profile='video',
            use_gpu=True,
            use_validator=False,
            use_eye_refinement=True
        )
        gpu_status = "GPU" if self.clnf.use_gpu else "CPU"
        print(f"  CLNF: Loaded ({gpu_status} accelerated)")

        # Bbox calibration coefficients (MTCNN -> CLNF mapping)
        # These map PyMTCNN output to CLNF-expected input
        self.bbox_coeffs = (-0.0075, 0.2459, 1.0323, 0.7751)

        # Tracking state
        self.cached_bbox = None
        self.last_landmarks = None
        self._frame_idx = 0

        # Thread lock for detector access
        self._lock = threading.Lock()

        # Temporal smoothing history (5-frame, matches local)
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.history_size = HISTORY_SIZE

        # Warmup models
        self._warmup()

    def _warmup(self):
        """Warm up models with dummy inference."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            _ = self.face_detector.detect(dummy_frame)
            dummy_bbox = (100, 100, 200, 200)
            _ = self.clnf.fit(dummy_frame, dummy_bbox)
            # Reset temporal state after warmup
            self.clnf.reset_temporal_state()
        except Exception:
            pass

    def reset_tracking(self):
        """Reset tracking state for new video."""
        self.cached_bbox = None
        self.last_landmarks = None
        self._frame_idx = 0
        self.landmarks_history.clear()
        self.glabella_history.clear()
        self.chin_history.clear()
        if hasattr(self.clnf, 'reset_temporal_state'):
            self.clnf.reset_temporal_state()

    def get_landmarks(self, frame, force_detect=False):
        """
        Get 68-point facial landmarks with temporal smoothing.

        Returns:
            landmarks: (68, 2) array of smoothed landmarks, or None
        """
        with self._lock:
            h, w = frame.shape[:2]

            # Increment frame index at START (matches local pipeline)
            self._frame_idx += 1

            # Face detection (first frame or periodic refresh)
            # Matches local: detect on first frame, then every 30 frames
            should_detect = (
                self.cached_bbox is None or
                force_detect or
                (self._frame_idx % DETECTION_INTERVAL == 0)
            )

            if should_detect and self.face_detector is not None:
                try:
                    boxes, _ = self.face_detector.detect(frame)

                    if boxes is not None and len(boxes) > 0:
                        # MTCNN returns [x, y, w, h, conf]
                        x, y, bw, bh = boxes[0][:4]

                        # Apply calibration coefficients (CRITICAL for CLNF accuracy)
                        # Keep as floats to match local pipeline precision
                        cx, cy, cw, ch = self.bbox_coeffs
                        self.cached_bbox = (
                            x + bw * cx,
                            y + bh * cy,
                            bw * cw,
                            bh * ch
                        )
                except Exception:
                    pass

            # CLNF landmark detection
            if self.cached_bbox is None:
                return None

            try:
                landmarks, info = self.clnf.fit(frame, self.cached_bbox, return_params=True)

                if landmarks is None or len(landmarks) != 68:
                    return self.last_landmarks.copy() if self.last_landmarks is not None else None

                landmarks = landmarks.astype(np.float32)

                # Temporal smoothing (5-frame weighted average)
                self.landmarks_history.append(landmarks.copy())
                if len(self.landmarks_history) > self.history_size:
                    self.landmarks_history.pop(0)

                # Weighted average (more weight to recent frames)
                weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
                weights = weights / np.sum(weights)

                smoothed = np.zeros_like(landmarks, dtype=np.float32)
                for pts, weight in zip(self.landmarks_history, weights):
                    smoothed += pts * weight

                self.last_landmarks = smoothed.copy()

                return smoothed

            except Exception:
                return self.last_landmarks.copy() if self.last_landmarks is not None else None

    def get_facial_midline(self, landmarks):
        """
        Calculate anatomical midline with temporal smoothing.

        Returns:
            glabella: Smoothed midpoint between medial eyebrows
            chin: Smoothed chin center point
        """
        if landmarks is None or len(landmarks) != 68:
            return None, None

        landmarks = landmarks.astype(np.float32)

        # Get medial eyebrow points (indices 21, 22)
        left_medial_brow = landmarks[21]
        right_medial_brow = landmarks[22]

        # Calculate glabella and chin
        glabella = (left_medial_brow + right_medial_brow) / 2
        chin = landmarks[8]

        # Add to history for temporal smoothing
        self.glabella_history.append(glabella)
        self.chin_history.append(chin)

        if len(self.glabella_history) > self.history_size:
            self.glabella_history.pop(0)
        if len(self.chin_history) > self.history_size:
            self.chin_history.pop(0)

        # Calculate smooth midline points (simple average)
        smooth_glabella = np.mean(self.glabella_history, axis=0)
        smooth_chin = np.mean(self.chin_history, axis=0)

        return smooth_glabella.astype(np.float32), smooth_chin.astype(np.float32)

    def clear_cache(self):
        """Clear GPU cache."""
        if hasattr(self.clnf, 'clear_gpu_cache'):
            self.clnf.clear_gpu_cache()


# =============================================================================
# FACE MIRRORING (matches face_mirror.py)
# =============================================================================

class FaceMirror:
    """
    Handles creation of mirrored face images with FULL PARITY to local pipeline.

    Features:
    - Anatomical midline-based reflection
    - 6-pixel gradient blending along midline
    - Cached coordinate grids for efficiency
    """

    def __init__(self, detector):
        self.detector = detector
        self._cached_coords = None
        self._cached_size = None

    def _get_coords(self, height, width):
        """Get or create cached coordinate grid."""
        if self._cached_size != (height, width):
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            self._cached_coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
            self._cached_size = (height, width)
        return self._cached_coords

    def create_mirrored_faces(self, frame, landmarks):
        """
        Create mirrored faces by reflecting along the anatomical midline.

        Returns:
            anatomical_right_face: Right side of face mirrored (patient's right preserved)
            anatomical_left_face: Left side of face mirrored (patient's left preserved)
        """
        height, width = frame.shape[:2]

        # Get smoothed midline points
        glabella, chin = self.detector.get_facial_midline(landmarks)

        if glabella is None or chin is None:
            return frame.copy(), frame.copy()

        # Calculate midline direction vector
        direction = chin - glabella
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        if norm < 1e-6:
            return frame.copy(), frame.copy()
        direction = direction / norm

        # Get cached coordinate grid
        coords = self._get_coords(height, width)

        # Calculate perpendicular vector (pointing to anatomical right)
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)

        # Calculate signed distances from midline
        diff_x = coords[..., 0] - glabella[0]
        diff_y = coords[..., 1] - glabella[1]
        distances = diff_x * perpendicular[0] + diff_y * perpendicular[1]

        # Calculate reflection map for cv2.remap
        map_x = coords[..., 0] - 2 * distances * perpendicular[0]
        map_y = coords[..., 1] - 2 * distances * perpendicular[1]

        # Clip to image bounds
        np.clip(map_x, 0, width - 1, out=map_x)
        np.clip(map_y, 0, height - 1, out=map_y)

        # Create reflected frame
        reflected_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Create side masks
        right_mask = distances >= 0  # Points on anatomical right
        left_mask = ~right_mask  # Points on anatomical left

        # Initialize output frames
        anatomical_right_face = frame.copy()
        anatomical_left_face = frame.copy()

        # Convert masks to uint8 for OpenCV
        left_mask_u8 = left_mask.astype(np.uint8) * 255
        right_mask_u8 = right_mask.astype(np.uint8) * 255

        # Apply reflections using copyTo
        cv2.copyTo(reflected_frame, left_mask_u8, anatomical_right_face)
        cv2.copyTo(reflected_frame, right_mask_u8, anatomical_left_face)

        # Apply gradient blending along the midline (6 pixels)
        blend_width = 6

        if abs(direction[1]) > 1e-6:
            y_range = np.arange(height)
            t = (y_range - glabella[1]) / direction[1]
            x_midline = (glabella[0] + t * direction[0]).astype(np.int32)
            np.clip(x_midline, blend_width // 2, width - blend_width // 2 - 1, out=x_midline)

            blend_weights = np.linspace(0, 1, blend_width, dtype=np.float32)
            offsets = np.arange(blend_width) - blend_width // 2

            for offset, weight in zip(offsets, blend_weights):
                x_coords = np.clip(x_midline + offset, 0, width - 1)
                blend_w = weight if offset >= 0 else (1 - weight)
                inv_blend = 1.0 - blend_w

                anatomical_right_face[y_range, x_coords] = (
                    anatomical_right_face[y_range, x_coords] * inv_blend +
                    frame[y_range, x_coords] * blend_w
                ).astype(np.uint8)
                anatomical_left_face[y_range, x_coords] = (
                    anatomical_left_face[y_range, x_coords] * inv_blend +
                    frame[y_range, x_coords] * blend_w
                ).astype(np.uint8)

        return anatomical_right_face, anatomical_left_face


# =============================================================================
# VIDEO WRITERS
# =============================================================================

class FFmpegWriter:
    """Hardware-accelerated video writer using FFmpeg + NVENC."""

    def __init__(self, output_path, fps, width, height, use_nvenc=True):
        self.output_path = str(output_path)
        self.fps = fps
        self.width = int(width)
        self.height = int(height)
        self.process = None
        self.use_nvenc = use_nvenc
        self.failed = False
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        if self.use_nvenc:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-rc', 'vbr',
                '-cq', '18',  # Match local: visually lossless
                '-pix_fmt', 'yuv420p',
                self.output_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',  # Match local: visually lossless
                '-pix_fmt', 'yuv420p',
                self.output_path
            ]

        try:
            # Use DEVNULL for stdout/stderr to prevent deadlock on long videos
            # When using PIPE, the buffer fills up and ffmpeg blocks waiting to write,
            # while Python blocks waiting to write to stdin = deadlock
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"    FFmpeg started: {Path(self.output_path).name}")
        except Exception as e:
            print(f"    FFmpeg failed to start: {e}")
            self.failed = True
            self.process = None

    def write(self, frame):
        if self.failed or self.process is None:
            return
        if self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                self.failed = True
                print(f"    FFmpeg pipe broken for: {Path(self.output_path).name}")

    def release(self):
        if self.process:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except:
                    pass
            self.process.wait()
            if self.process.returncode != 0:
                print(f"    FFmpeg exit code {self.process.returncode} for: {Path(self.output_path).name}")
            self.process = None


class CV2Writer:
    """Fallback video writer using OpenCV."""

    def __init__(self, output_path, fps, width, height):
        self.output_path = str(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (int(width), int(height)))
        self.failed = not self.writer.isOpened()
        if self.failed:
            print(f"    CV2Writer failed to open: {output_path}")

    def write(self, frame):
        if not self.failed:
            self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def check_nvenc_available():
    """Check if NVENC is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


def process_video_mirroring(video_path, output_dir, output_stem, use_ffmpeg=True, use_nvenc=True):
    """
    Process video to create left and right mirrored versions.
    Uses full parity pipeline with temporal smoothing and gradient blending.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    right_path = output_dir / f"{output_stem}_right_mirrored.mp4"
    left_path = output_dir / f"{output_stem}_left_mirrored.mp4"

    # Open input video (already rotated)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Input: {w}x{h}, {total_frames} frames, {fps:.1f} fps")

    # Initialize detector and mirror (FULL PARITY)
    print("  Initializing face detection pipeline...")
    detector = LandmarkDetector()
    mirror = FaceMirror(detector)

    # Create video writers
    if use_ffmpeg:
        print(f"  Writer: FFmpeg ({'NVENC' if use_nvenc else 'libx264'})")
        right_writer = FFmpegWriter(right_path, fps, w, h, use_nvenc=use_nvenc)
        left_writer = FFmpegWriter(left_path, fps, w, h, use_nvenc=use_nvenc)
    else:
        print("  Writer: OpenCV (mp4v)")
        right_writer = CV2Writer(str(right_path), fps, w, h)
        left_writer = CV2Writer(str(left_path), fps, w, h)

    # Process frames
    processed_count = 0

    try:
        with tqdm(total=total_frames, desc="Mirroring") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get landmarks with temporal smoothing
                force_detect = (frame_idx == 0)
                landmarks = detector.get_landmarks(frame, force_detect=force_detect)

                # Create mirrored frames with gradient blending
                if landmarks is not None:
                    right_frame, left_frame = mirror.create_mirrored_faces(frame, landmarks)
                else:
                    right_frame = frame
                    left_frame = frame

                # Write frames
                right_writer.write(right_frame)
                left_writer.write(left_frame)

                processed_count += 1
                frame_idx += 1
                pbar.update(1)

                # Periodic GC
                if frame_idx % 500 == 0:
                    gc.collect()

    except Exception as e:
        print(f"  ERROR during mirroring: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cap.release()
        right_writer.release()
        left_writer.release()
        detector.clear_cache()

    print(f"  Mirrored {processed_count} frames")

    return right_path, left_path


def process_au_extraction_sequential(right_path, left_path, csv_dir, weights_dir):
    """Run AU extraction on both mirrored videos sequentially."""
    from pyfaceau.processor import OpenFaceProcessor

    stem = Path(right_path).stem.replace('_right_mirrored', '')
    results = {}

    processor = OpenFaceProcessor(
        weights_dir=str(weights_dir),
        use_clnf_refinement=True,
        verbose=False
    )

    # Process right mirrored
    right_csv = csv_dir / f"{stem}_right_mirrored.csv"
    print(f"    Processing: {Path(right_path).name}...")
    try:
        right_frames = processor.process_video(str(right_path), str(right_csv))
        results[f"{stem}_right_mirrored"] = right_frames
        print(f"    {Path(right_path).stem}: {right_frames} frames")
    except Exception as e:
        print(f"    {Path(right_path).stem}: ERROR - {e}")
        results[f"{stem}_right_mirrored"] = 0

    # Process left mirrored
    left_csv = csv_dir / f"{stem}_left_mirrored.csv"
    print(f"    Processing: {Path(left_path).name}...")
    try:
        left_frames = processor.process_video(str(left_path), str(left_csv))
        results[f"{stem}_left_mirrored"] = left_frames
        print(f"    {Path(left_path).stem}: {left_frames} frames")
    except Exception as e:
        print(f"    {Path(left_path).stem}: ERROR - {e}")
        results[f"{stem}_left_mirrored"] = 0

    processor.clear_cache()
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='S1 Face Mirror - Full Parity Batch Processor')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--weights-dir', type=str, default=None, help='Path to AU weights')
    parser.add_argument('--no-nvenc', action='store_true', help='Disable NVENC (use libx264)')
    parser.add_argument('--use-cv2', action='store_true', help='Use OpenCV writer instead of FFmpeg')
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)

    # Default weights directory
    if args.weights_dir:
        weights_dir = Path(args.weights_dir)
    else:
        weights_dir = Path(__file__).parent.parent / "S1_FaceMirror" / "weights"

    # Check NVENC availability
    use_ffmpeg = not args.use_cv2
    use_nvenc = not args.no_nvenc and check_nvenc_available()

    print("=" * 60)
    print("S1 FACE MIRROR - FULL PARITY BATCH PROCESSOR")
    print("=" * 60)
    print(f"Input: {video_path.name}")
    print(f"Output: {output_dir}")
    print(f"Video writer: {'FFmpeg' if use_ffmpeg else 'OpenCV'}")
    if use_ffmpeg:
        print(f"NVENC available: {check_nvenc_available()}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Temporal smoothing: {HISTORY_SIZE} frames")
    print()

    start_time = time.time()

    # Create output directories matching local convention:
    # - Combined Data: CSVs and _source videos
    # - Face Mirror 1.0 Output: mirrored videos
    combined_data_dir = output_dir / "Combined Data"
    mirror_dir = output_dir / "Face Mirror 1.0 Output"
    combined_data_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Rotate video (save as _source)
    print("[1/3] Rotating video...")
    rotation_start = time.time()
    source_path = combined_data_dir / f"{video_path.stem}_source{video_path.suffix}"
    source_path = rotate_video_with_ffmpeg(video_path, source_path)
    rotation_elapsed = time.time() - rotation_start
    print(f"  Source: {source_path.name}")
    print(f"  Rotation time: {rotation_elapsed:.1f}s")

    # Step 2: Create mirrored videos from rotated source (FULL PARITY)
    print("\n[2/3] Creating mirrored videos (full parity pipeline)...")
    mirror_start = time.time()
    right_path, left_path = process_video_mirroring(
        source_path, mirror_dir, video_path.stem,
        use_ffmpeg=use_ffmpeg, use_nvenc=use_nvenc
    )
    mirror_elapsed = time.time() - mirror_start
    print(f"  Right: {right_path.name}")
    print(f"  Left: {left_path.name}")
    print(f"  Mirroring time: {mirror_elapsed:.1f}s")

    # Step 3: Extract AUs sequentially
    print("\n[3/3] Extracting Action Units...")
    au_start = time.time()
    results = process_au_extraction_sequential(right_path, left_path, combined_data_dir, weights_dir)
    au_elapsed = time.time() - au_start
    print(f"  AU extraction time: {au_elapsed:.1f}s")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"  Rotation: {rotation_elapsed:.1f}s")
    print(f"  Mirroring: {mirror_elapsed:.1f}s")
    print(f"  AU extraction: {au_elapsed:.1f}s")
    print(f"Outputs:")
    print(f"  Source video: {source_path}")
    print(f"  Mirrored videos: {mirror_dir}")
    print(f"  AU CSVs: {combined_data_dir}")


if __name__ == '__main__':
    main()
