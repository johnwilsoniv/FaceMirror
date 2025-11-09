#!/usr/bin/env python3
"""
Test C++ OpenFace on the same videos for comparison.
"""

import subprocess
import sys
from pathlib import Path
import shutil

# C++ OpenFace binary location
cpp_binary = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")

if not cpp_binary.exists():
    print(f"ERROR: C++ OpenFace binary not found at {cpp_binary}")
    sys.exit(1)

# Test videos
test_videos = [
    ("IMG_8401_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0942_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

output_dir = Path(__file__).parent / "test_output" / "cpp_openface"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TESTING C++ OPENFACE ON SAME VIDEOS")
print("="*80)
print()

for name, video_path in test_videos:
    print(f"\nTesting {name}...")
    print("-"*80)

    # Create subdirectory for this video
    video_output = output_dir / name
    if video_output.exists():
        shutil.rmtree(video_output)
    video_output.mkdir(parents=True, exist_ok=True)

    # Run C++ OpenFace
    cmd = [
        str(cpp_binary),
        "-f", video_path,
        "-out_dir", str(video_output),
        "-simalign",  # Output aligned face images
        "-simscale", "0.7",  # Scale for visualization
        "-wild",  # For in-the-wild videos
    ]

    print(f"Running: {' '.join(cmd[:4])} ...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✓ Success")

            # Check output files
            csv_files = list(video_output.glob("*.csv"))
            img_files = list(video_output.glob("*.bmp"))

            print(f"  Output: {len(csv_files)} CSV files, {len(img_files)} aligned images")

            # Check for landmarks CSV
            landmarks_csv = None
            for f in csv_files:
                if f.name.endswith(".csv"):
                    landmarks_csv = f
                    break

            if landmarks_csv:
                # Read first few lines to check landmark count
                with open(landmarks_csv, 'r') as f:
                    header = f.readline()
                    landmark_cols = [col for col in header.split(',') if col.strip().startswith(('x_', 'y_', 'X_', 'Y_'))]
                    print(f"  Detected {len(landmark_cols)//2} landmarks (68-point system)")
        else:
            print(f"❌ Failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after 60 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

print()
print("="*80)
print("C++ OpenFace test complete")
print(f"Output directory: {output_dir}")
print("="*80)
