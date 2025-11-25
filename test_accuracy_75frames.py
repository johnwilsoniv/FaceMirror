#!/usr/bin/env python3
"""
Test CLNF accuracy on 75 frames from video.
Compare with OpenFace C++ if available.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import time
from pathlib import Path

from pyclnf.clnf import CLNF

def main():
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0942.MOV'
    num_frames = 75

    print("="*60)
    print("CLNF Accuracy Test - 75 Frames")
    print("="*60)
    print(f"Video: {video_path}")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")

    # Initialize CLNF
    print("\nInitializing CLNF...")
    t0 = time.perf_counter()
    clnf = CLNF('pyclnf/pyclnf/models', regularization=40)
    init_time = time.perf_counter() - t0
    print(f"  Initialization time: {init_time*1000:.1f}ms")

    # Process frames
    print(f"\nProcessing {num_frames} frames...")

    results = []
    times = []

    # Sample frames evenly across video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"  Frame {i+1}/{num_frames}: Failed to read")
            continue

        # Detect and fit
        t0 = time.perf_counter()
        result = clnf.detect_and_fit(frame)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if result is not None and result[0] is not None:
            landmarks_2d, info = result
            results.append({
                'frame_idx': frame_idx,
                'landmarks_2d': landmarks_2d,
                'info': info,
                'time': elapsed
            })

            # Print progress every 10 frames
            if (i + 1) % 10 == 0:
                fps_val = 1.0 / elapsed if elapsed > 0 else 0
                print(f"  Frame {i+1}/{num_frames}: {elapsed*1000:.1f}ms ({fps_val:.1f} FPS)")
        else:
            results.append(None)
            if (i + 1) % 10 == 0:
                print(f"  Frame {i+1}/{num_frames}: No face detected")

    cap.release()

    # Statistics
    successful = sum(1 for r in results if r is not None)
    times = np.array(times)

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Successful detections: {successful}/{num_frames}")
    print(f"Average time: {np.mean(times)*1000:.1f}ms")
    print(f"Average FPS: {1.0/np.mean(times):.1f}")
    print(f"Min time: {np.min(times)*1000:.1f}ms")
    print(f"Max time: {np.max(times)*1000:.1f}ms")

    # Analyze landmark stability (for consecutive frames)
    if successful >= 2:
        print("\n" + "-"*60)
        print("Landmark Stability Analysis")
        print("-"*60)

        # Find consecutive successful detections
        prev_landmarks = None
        movements = []

        for r in results:
            if r is not None:
                curr_landmarks = r['landmarks_2d']
                if prev_landmarks is not None:
                    # Compute movement between frames
                    diff = curr_landmarks - prev_landmarks
                    movement = np.sqrt(np.sum(diff**2, axis=1))
                    movements.append(movement)
                prev_landmarks = curr_landmarks

        if movements:
            movements = np.array(movements)
            avg_movement = np.mean(movements)
            max_movement = np.max(movements)

            # Per-landmark statistics
            landmark_avg = np.mean(movements, axis=0)
            worst_landmarks = np.argsort(landmark_avg)[-5:]

            print(f"Average inter-frame movement: {avg_movement:.2f} pixels")
            print(f"Maximum movement: {max_movement:.2f} pixels")
            print(f"\nWorst 5 landmarks (highest movement):")
            for idx in worst_landmarks[::-1]:
                print(f"  Landmark {idx}: {landmark_avg[idx]:.2f} px avg movement")

    # Save sample visualization
    print("\n" + "-"*60)
    print("Saving visualization...")
    print("-"*60)

    # Get a sample frame with landmarks
    cap = cv2.VideoCapture(video_path)
    sample_result = None
    sample_frame = None

    for r in results:
        if r is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r['frame_idx'])
            ret, frame = cap.read()
            if ret:
                sample_frame = frame
                sample_result = r
                break

    cap.release()

    if sample_frame is not None and sample_result is not None:
        # Draw landmarks
        vis_frame = sample_frame.copy()
        landmarks = sample_result['landmarks_2d']

        for i, (x, y) in enumerate(landmarks):
            # Color code by region
            if i < 17:  # Jaw
                color = (0, 255, 0)
            elif i < 27:  # Eyebrows
                color = (255, 0, 0)
            elif i < 36:  # Nose
                color = (0, 255, 255)
            elif i < 48:  # Eyes
                color = (255, 255, 0)
            else:  # Mouth
                color = (0, 0, 255)

            cv2.circle(vis_frame, (int(x), int(y)), 2, color, -1)

        # Save
        output_path = 'accuracy_test_sample.png'
        cv2.imwrite(output_path, vis_frame)
        print(f"Saved sample visualization to: {output_path}")

    # Check for OpenFace comparison
    openface_csv = Path(video_path).stem + '.csv'
    if Path(openface_csv).exists():
        print(f"\nFound OpenFace CSV: {openface_csv}")
        print("Comparison analysis available")
    else:
        print(f"\nNo OpenFace CSV found for comparison")
        print("Run OpenFace FeatureExtraction on the video to enable accuracy comparison")

    return results


if __name__ == '__main__':
    main()
