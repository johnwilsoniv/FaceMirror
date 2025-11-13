#!/usr/bin/env python3
"""
FAST baseline test: Reduced to 5 iterations to get results quickly.
Once we confirm accuracy, we can optimize performance for full 20 iterations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

import cv2
import numpy as np
import subprocess
import csv
from pyclnf import CLNF


def run_cpp_openface_video_mode(image_path, output_dir):
    """Run C++ OpenFace in video mode (-wild flag)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_binary = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(cpp_binary),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-2Dfp",
        "-wild"  # Video mode / in-the-wild mode
    ]

    print("  Running C++ OpenFace...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise FileNotFoundError("No CSV from C++ OpenFace")

    csv_path = csv_files[0]

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_row = next(reader)

    # Extract landmarks
    x_indices = [i for i, h in enumerate(header) if h.strip().startswith('x_') and h.strip()[2:].isdigit()]
    y_indices = [i for i, h in enumerate(header) if h.strip().startswith('y_') and h.strip()[2:].isdigit()]

    x_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))
    y_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))

    landmarks = []
    for x_idx, y_idx in zip(x_indices, y_indices):
        x = float(data_row[x_idx])
        y = float(data_row[y_idx])
        landmarks.append([x, y])

    return np.array(landmarks)


def run_python_pyclnf_fast(image, bbox):
    """Run Python pyCLNF with REDUCED iterations (5) for fast baseline."""
    print("  Running Python pyCLNF (5 iterations - FAST MODE)...")

    import time
    start_time = time.time()

    # USE ONLY 5 ITERATIONS for fast testing
    clnf = CLNF(max_iterations=5, convergence_threshold=0.01, detector=None)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    total_time = time.time() - start_time
    print(f"     ✓ Completed in {total_time:.2f}s ({info['iterations']} iterations)")

    return landmarks, info


def calculate_error(landmarks, cpp_landmarks):
    """Calculate per-landmark errors."""
    distances = np.sqrt(np.sum((landmarks - cpp_landmarks)**2, axis=1))

    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'std': np.std(distances),
        'distances': distances
    }


def extract_frames_from_videos(video_dir, num_people=5, frames_per_person=2):
    """Extract test frames from patient videos."""
    video_dir = Path(video_dir)
    video_files = sorted(list(video_dir.glob("*.MOV")) + list(video_dir.glob("*.mp4")))[:num_people]

    frames = []
    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, frames_per_person, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Save frame
                frame_name = f"{video_path.stem}_frame_{frame_idx:04d}.jpg"
                frame_path = Path("test_output/baseline_fast/frames") / frame_name
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_path), frame)
                frames.append({
                    'path': frame_path,
                    'video': video_path.name,
                    'frame_idx': frame_idx
                })

        cap.release()

    return frames


def main():
    print("\n📊 FAST BASELINE TEST (5 iterations)")
    print("=" * 70)
    print("C++ OpenFace (video mode, 20 iters) vs Python pyCLNF (5 iters)")
    print("Testing: 2 frames per person × 5 people = 10 frames")
    print("=" * 70)
    print("\n⚡ NOTE: Using 5 iterations instead of 20 for SPEED")
    print("         This will show if accuracy gap is from convergence params")
    print("         or from something else entirely.")
    print("=" * 70)

    # Extract test frames
    print("\n📁 Extracting test frames...")
    frames = extract_frames_from_videos(
        "Patient Data/Normal Cohort",
        num_people=5,
        frames_per_person=2
    )
    print(f"   ✅ Extracted {len(frames)} frames")

    # Process each frame
    all_errors = []
    all_iterations = []

    for i, frame_info in enumerate(frames):
        print(f"\n{'=' * 70}")
        print(f"Frame {i+1}/{len(frames)}: {frame_info['video']} (frame {frame_info['frame_idx']})")
        print(f"{'=' * 70}")

        image_path = frame_info['path']
        image = cv2.imread(str(image_path))

        # Run C++ OpenFace in video mode
        cpp_output_dir = Path(f"test_output/baseline_fast/cpp_output/frame_{i:03d}")
        try:
            cpp_landmarks = run_cpp_openface_video_mode(image_path, cpp_output_dir)
            print(f"     ✅ C++ landmarks: {len(cpp_landmarks)} points")
        except Exception as e:
            print(f"     ❌ C++ failed: {e}")
            continue

        # Estimate bbox from C++ landmarks
        x_min, y_min = cpp_landmarks.min(axis=0)
        x_max, y_max = cpp_landmarks.max(axis=0)
        margin = 0.1
        width = x_max - x_min
        height = y_max - y_min
        x_min -= margin * width
        y_min -= margin * height
        width *= (1 + 2*margin)
        height *= (1 + 2*margin)
        bbox = (x_min, y_min, width, height)

        # Run Python pyCLNF (5 iterations only)
        try:
            py_landmarks, py_info = run_python_pyclnf_fast(image, bbox)
            print(f"     ✅ Python landmarks: {len(py_landmarks)} points")
        except Exception as e:
            print(f"     ❌ Python failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Calculate accuracy
        error = calculate_error(py_landmarks, cpp_landmarks)
        all_errors.append(error['mean'])
        all_iterations.append(py_info['iterations'])

        print(f"\n  Results:")
        print(f"     Mean error: {error['mean']:.3f}px")
        print(f"     Iterations: {py_info['iterations']}")
        print(f"     Converged:  {py_info['converged']}")

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (ALL FRAMES)")
    print("=" * 70)

    if len(all_errors) == 0:
        print("❌ No frames processed successfully")
        return

    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    std_error = np.std(all_errors)
    min_error = np.min(all_errors)
    max_error = np.max(all_errors)

    print(f"\nAccuracy Statistics (n={len(all_errors)} frames, 5 iterations):")
    print(f"  Mean error:     {mean_error:.3f}px")
    print(f"  Median error:   {median_error:.3f}px")
    print(f"  Std deviation:  {std_error:.3f}px")
    print(f"  Min error:      {min_error:.3f}px")
    print(f"  Max error:      {max_error:.3f}px")

    print(f"\nIterations:")
    print(f"  Mean:           {np.mean(all_iterations):.1f}")
    print(f"  Range:          {np.min(all_iterations)}-{np.max(all_iterations)}")

    # Per-frame breakdown
    print(f"\nPer-Frame Results:")
    for i, (err, iters) in enumerate(zip(all_errors, all_iterations)):
        status = "✅" if err < 5.0 else "⚠️" if err < 10.0 else "❌"
        print(f"  {status} Frame {i+1}: {err:6.3f}px ({iters} iterations)")

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    print(f"\n⚡ FAST MODE (5 iterations):")
    print(f"   Mean error: {mean_error:.3f}px")

    print(f"\n📊 Comparison to previous results:")
    print(f"   vs C++ OpenFace (20 iters):    Reference benchmark")
    print(f"   vs PyMTCNN (16.4px):           {((16.4 - mean_error)/16.4)*100:+.1f}% better")
    print(f"   vs Previous pyCLNF (8.23px):   {((8.23 - mean_error)/8.23)*100:+.1f}% change")

    print(f"\n🎯 NEXT STEPS:")
    if mean_error < 5.0:
        print(f"   ✅ EXCELLENT accuracy even with 5 iterations!")
        print(f"   ➡️  Now we need to optimize performance to run 20 iterations faster")
    elif mean_error < 10.0:
        print(f"   ✅ GOOD accuracy with 5 iterations")
        print(f"   ➡️  With 20 iterations, should achieve <5px")
        print(f"   ➡️  Need to optimize response map computation speed")
    else:
        print(f"   ⚠️  Higher error than expected - may indicate other issues")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
