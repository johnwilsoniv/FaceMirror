#!/usr/bin/env python3
"""
Extract initial landmarks from OpenFace C++ to compare with Python CLNF.

Since OpenFace doesn't expose pre-CLNF landmarks directly, we'll use a workaround:
1. Run OpenFace on a single frame
2. Extract the final landmarks from CSV
3. Use those as our "initialization" reference
4. Then run Python CLNF from MTCNN/FAN initialization

Alternative approach:
- Extract first frame landmarks from OpenFace
- Use temporal tracking: frame N uses frame N-1 as init
- Compare Python's per-frame reinitialization vs temporal tracking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
import subprocess
import tempfile
import shutil


def extract_openface_landmarks(video_path, frame_idx, openface_binary):
    """
    Extract OpenFace landmarks for a single frame.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        openface_binary: Path to OpenFace FeatureExtraction binary

    Returns:
        landmarks: 68-point landmarks (68, 2)
        metadata: Additional OpenFace metadata (confidence, pose, etc.)
    """
    # Create temporary directory for OpenFace output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract single frame to image
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")

        # Save frame as image
        frame_img = temp_path / "frame.jpg"
        cv2.imwrite(str(frame_img), frame)

        # Run OpenFace on single frame
        print(f"\nRunning OpenFace on frame {frame_idx}...")
        result = subprocess.run(
            [
                str(openface_binary),
                "-f", str(frame_img),
                "-out_dir", str(temp_path),
                "-2Dfp",  # Output 2D landmarks
                "-pose",  # Output head pose
                "-aus",   # Output AUs
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("OpenFace stderr:", result.stderr)
            raise RuntimeError(f"OpenFace failed with code {result.returncode}")

        # Read CSV output
        csv_files = list(temp_path.glob("*.csv"))
        if not csv_files:
            raise RuntimeError("OpenFace did not produce CSV output")

        df = pd.read_csv(csv_files[0])

        # Extract landmarks
        row = df.iloc[0]
        landmarks = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            landmarks[i, 0] = row[f'x_{i}']
            landmarks[i, 1] = row[f'y_{i}']

        # Extract metadata
        metadata = {
            'confidence': row['confidence'],
            'success': row['success'],
            'pose_Tx': row['pose_Tx'],
            'pose_Ty': row['pose_Ty'],
            'pose_Tz': row['pose_Tz'],
            'pose_Rx': row['pose_Rx'],
            'pose_Ry': row['pose_Ry'],
            'pose_Rz': row['pose_Rz'],
        }

        print(f"OpenFace confidence: {metadata['confidence']:.3f}")
        print(f"OpenFace success: {metadata['success']}")

        return landmarks, metadata, frame


def save_initialization_data(output_path, frame, landmarks, metadata, source_info):
    """
    Save initialization data for later comparison.

    Args:
        output_path: Path to save NPZ file
        frame: Image frame (H, W, 3)
        landmarks: 68-point landmarks (68, 2)
        metadata: OpenFace metadata dict
        source_info: Dict with video_path, frame_idx
    """
    np.savez(
        output_path,
        frame=frame,
        landmarks=landmarks,
        **metadata,
        **source_info
    )
    print(f"\nSaved initialization data to: {output_path}")


def main():
    """Extract OpenFace initialization data for test cases."""

    # Configuration
    openface_binary = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")
    test_cases = [
        {
            'video': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV',
            'name': 'IMG_9330',
            'frame': 100,
        },
        {
            'video': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV',
            'name': 'IMG_8401',
            'frame': 100,
        },
    ]

    output_dir = Path("/tmp/clnf_diagnostic_data")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("EXTRACTING OPENFACE INITIALIZATION DATA")
    print("="*80)

    for test in test_cases:
        print(f"\n### Processing {test['name']} frame {test['frame']} ###")

        try:
            landmarks, metadata, frame = extract_openface_landmarks(
                test['video'],
                test['frame'],
                openface_binary
            )

            # Save initialization data
            output_path = output_dir / f"{test['name']}_frame{test['frame']}_openface_init.npz"
            save_initialization_data(
                output_path,
                frame,
                landmarks,
                metadata,
                {
                    'video_path': test['video'],
                    'frame_idx': test['frame'],
                    'video_name': test['name'],
                }
            )

        except Exception as e:
            print(f"ERROR processing {test['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nData saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run compare_clnf_cpp_vs_python.py to compare C++ vs Python CLNF")
    print("2. Use clnf_debug_logger.py to see detailed iteration-by-iteration comparison")


if __name__ == '__main__':
    main()
