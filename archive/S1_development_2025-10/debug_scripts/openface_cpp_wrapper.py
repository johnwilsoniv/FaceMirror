#!/usr/bin/env python3
"""
OpenFace C++ Wrapper
Wraps the OpenFace 2.2 C++ FeatureExtraction binary for Components 1-4:
  - Component 1: Video Input
  - Component 2: Face Detection
  - Component 3: 68-point Landmark Detection
  - Component 4: 3D Pose Estimation (CalcParams)

This provides guaranteed-accurate outputs that match the C++ baseline.
"""

import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os


class OpenFaceCppWrapper:
    """
    Wrapper for OpenFace 2.2 C++ FeatureExtraction binary.

    Handles Components 1-4 of the pipeline, producing CSV output
    that can be fed into the Python AU extraction pipeline.
    """

    def __init__(self, binary_path, model_dir=None):
        """
        Initialize the C++ wrapper.

        Args:
            binary_path: Path to FeatureExtraction binary
            model_dir: Path to OpenFace models directory (optional)
        """
        self.binary_path = Path(binary_path)

        if not self.binary_path.exists():
            raise FileNotFoundError(f"FeatureExtraction binary not found: {binary_path}")

        self.model_dir = Path(model_dir) if model_dir else None

    def process_video(self, video_path, output_dir=None, frames=None):
        """
        Process a video through OpenFace C++ pipeline.

        Args:
            video_path: Path to input video
            output_dir: Output directory (uses temp if None)
            frames: Number of frames to process (None = all)

        Returns:
            DataFrame with OpenFace outputs (landmarks, pose, shape params, AUs)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directory
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="openface_")
            output_dir = Path(temp_dir)
            cleanup = True
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False

        try:
            # Build command
            cmd = [
                str(self.binary_path),
                "-f", str(video_path),
                "-out_dir", str(output_dir)
            ]

            # Add model directory if specified
            if self.model_dir:
                cmd.extend(["-mloc", str(self.model_dir)])

            # Run FeatureExtraction
            print(f"Running OpenFace C++ FeatureExtraction...")
            print(f"  Video: {video_path.name}")
            print(f"  Output: {output_dir}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FeatureExtraction failed:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            # Find output CSV
            csv_files = list(output_dir.glob("*.csv"))

            if not csv_files:
                raise RuntimeError(f"No CSV output found in {output_dir}")

            csv_path = csv_files[0]

            # Load CSV
            df = pd.read_csv(csv_path)

            # Limit frames if requested
            if frames is not None:
                df = df.head(frames)

            print(f"  Processed {len(df)} frames")

            return df

        finally:
            # Cleanup temp directory if created
            if cleanup and output_dir.exists():
                shutil.rmtree(output_dir)

    def extract_landmarks(self, df, frame_idx):
        """
        Extract 68 2D landmarks from CSV row.

        Args:
            df: DataFrame from process_video()
            frame_idx: Frame index

        Returns:
            landmarks: (68, 2) array of (x, y) coordinates
        """
        row = df.iloc[frame_idx]

        landmarks = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            landmarks[i, 0] = row[f'x_{i}']
            landmarks[i, 1] = row[f'y_{i}']

        return landmarks

    def extract_pose_params(self, df, frame_idx):
        """
        Extract pose parameters from CSV row.

        Args:
            df: DataFrame from process_video()
            frame_idx: Frame index

        Returns:
            pose_params: (6,) array [scale, rx, ry, rz, tx, ty]
        """
        row = df.iloc[frame_idx]

        pose_params = np.array([
            row['p_scale'],
            row['p_rx'],
            row['p_ry'],
            row['p_rz'],
            row['p_tx'],
            row['p_ty']
        ], dtype=np.float32)

        return pose_params

    def extract_shape_params(self, df, frame_idx):
        """
        Extract shape parameters from CSV row.

        Args:
            df: DataFrame from process_video()
            frame_idx: Frame index

        Returns:
            shape_params: (34,) array of PCA shape parameters
        """
        row = df.iloc[frame_idx]

        shape_params = np.array([row[f'p_{i}'] for i in range(34)], dtype=np.float32)

        return shape_params

    def extract_aus(self, df, frame_idx):
        """
        Extract AU intensities from CSV row (for validation).

        Args:
            df: DataFrame from process_video()
            frame_idx: Frame index

        Returns:
            aus: dict of {AU_name: intensity}
        """
        row = df.iloc[frame_idx]

        aus = {}
        au_names = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                   'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                   'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

        for au in au_names:
            col_name = f' {au}'  # OpenFace CSVs have leading space
            if col_name in row:
                aus[au] = row[col_name]

        return aus


def test_wrapper():
    """Test the OpenFace C++ wrapper"""
    print("="*80)
    print("OpenFace C++ Wrapper Test")
    print("="*80)

    # Paths
    OF_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

    # Initialize wrapper
    print("\n1. Initializing wrapper...")
    wrapper = OpenFaceCppWrapper(OF_BINARY)
    print("  ✓ Wrapper initialized")

    # Process first 10 frames
    print("\n2. Processing video (first 10 frames)...")
    df = wrapper.process_video(VIDEO_PATH, frames=10)
    print(f"  ✓ Processed {len(df)} frames")

    # Extract data from frame 0
    print("\n3. Extracting frame 0 data...")
    landmarks = wrapper.extract_landmarks(df, 0)
    pose = wrapper.extract_pose_params(df, 0)
    shape = wrapper.extract_shape_params(df, 0)
    aus = wrapper.extract_aus(df, 0)

    print(f"  Landmarks: {landmarks.shape}")
    print(f"  Pose: {pose}")
    print(f"  Shape: {shape[:5]}... (first 5)")
    print(f"  AUs: {list(aus.keys())[:5]}... (first 5)")

    print("\n" + "="*80)
    print("✓ Wrapper test passed!")
    print("="*80)


if __name__ == "__main__":
    test_wrapper()
