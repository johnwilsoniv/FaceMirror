#!/usr/bin/env python3
"""
Test current S1 Face Mirror implementation on IMG_0942.MOV
Benchmark performance and verify landmark quality.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add S1 Face Mirror and pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector
from face_splitter import StableFaceSplitter
import config_paths

print("="*80)
print("S1 FACE MIRROR - CURRENT IMPLEMENTATION TEST")
print("="*80)
print()

# Test video
video_path = Path(__file__).parent / "Patient Data" / "Normal Cohort" / "IMG_0942.MOV"
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)

print(f"Input: {video_path}")
print(f"Output: {output_dir}")
print()

# Get video info
cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video Info:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps:.2f}")
print(f"  Total frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.2f} seconds")
print()

# Initialize detector
print("Initializing PyFaceAU detector...")
start_init = time.time()
detector = PyFaceAU68LandmarkDetector(
    debug_mode=True,
    use_clnf_refinement=True,
    skip_redetection=True  # Use tracking mode for speed
)
init_time = time.time() - start_init
print(f"✓ Initialization took {init_time:.2f} seconds")
print()

# Test on first 100 frames for quick benchmark
test_frames = min(100, total_frames)
print(f"Processing first {test_frames} frames for benchmark...")
print()

cap = cv2.VideoCapture(str(video_path))
frame_times = []
landmark_errors = []
successful_frames = 0

# Read and process frames
for frame_idx in range(test_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # Time landmark detection
    start_time = time.time()
    landmarks, _ = detector.get_face_mesh(frame)
    frame_time = time.time() - start_time
    frame_times.append(frame_time)

    if landmarks is not None:
        successful_frames += 1

        # Check landmark quality
        bbox_width = width * 0.8  # Approximate face bbox
        bbox_height = height * 0.8

        # Calculate spread
        x_std = np.std(landmarks[:, 0])
        y_std = np.std(landmarks[:, 1])
        expected_x_std = bbox_width * 0.25
        expected_y_std = bbox_height * 0.25

        quality = "good" if (x_std > expected_x_std * 0.8 and y_std > expected_y_std * 0.8) else "poor"

        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx:4d}: {frame_time*1000:6.2f}ms | Landmarks: {len(landmarks)} | Quality: {quality}")
    else:
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx:4d}: {frame_time*1000:6.2f}ms | FAILED")

cap.release()

# Calculate statistics
frame_times = np.array(frame_times)
mean_time = np.mean(frame_times) * 1000  # ms
median_time = np.median(frame_times) * 1000
std_time = np.std(frame_times) * 1000
min_time = np.min(frame_times) * 1000
max_time = np.max(frame_times) * 1000

fps_achieved = 1.0 / np.mean(frame_times)

print()
print("="*80)
print("PERFORMANCE RESULTS")
print("="*80)
print()
print(f"Frames processed: {test_frames}")
print(f"Successful detections: {successful_frames}/{test_frames} ({100*successful_frames/test_frames:.1f}%)")
print()
print("Per-frame timing:")
print(f"  Mean:   {mean_time:6.2f} ms")
print(f"  Median: {median_time:6.2f} ms")
print(f"  Std:    {std_time:6.2f} ms")
print(f"  Min:    {min_time:6.2f} ms")
print(f"  Max:    {max_time:6.2f} ms")
print()
print(f"Processing FPS: {fps_achieved:.1f} FPS")
print()

# Expected performance from commit 57bd6da
expected_fps = 22.5  # From commit message
print("Expected performance (from commit 57bd6da):")
print(f"  Target: {expected_fps:.1f} FPS (mirroring)")
print(f"  Actual: {fps_achieved:.1f} FPS (detection only)")
print()

if fps_achieved >= 20:
    print("✅ PASS: Achieving 20+ FPS target")
elif fps_achieved >= 15:
    print("⚠️  WARNING: Below target but acceptable (15-20 FPS)")
else:
    print("❌ FAIL: Below 15 FPS - performance degradation")

print()
print("="*80)
print("COMPONENT BREAKDOWN")
print("="*80)
print()
print("Current implementation uses:")
print("  • PFLD detector (4.37% NME)")
print("  • TargetedCLNFRefiner (SVR-based, 1-2ms)")
print("  • RetinaFace (frame 0 only, then cached)")
print("  • 5-frame temporal smoothing")
print()
print("Note: This is detection-only. Full mirroring pipeline adds:")
print("  • Face mirroring computation")
print("  • Video I/O overhead")
print("  • Expected impact: ~5-10 FPS reduction")
print()
print(f"Estimated full pipeline FPS: {fps_achieved * 0.6:.1f} - {fps_achieved * 0.8:.1f} FPS")
print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
