#!/usr/bin/env python3
"""
Test current Python implementation on rotated _source videos.
These are pre-rotated files from the processed data folder.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

print("="*80)
print("ROTATED SOURCE FILES TEST (Python Implementation)")
print("="*80)
print()

# Test videos from processed folder
base_dir = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data")
test_videos = [
    {
        'name': 'IMG_8401_source (Paralysis - Surgical Markings)',
        'path': base_dir / "IMG_8401_source.MOV",
        'category': 'CHALLENGING',
        'notes': 'Pre-rotated, surgical markings'
    },
    {
        'name': 'IMG_9330_source (Paralysis - Severe)',
        'path': base_dir / "IMG_9330_source.MOV",
        'category': 'CHALLENGING',
        'notes': 'Pre-rotated, severe paralysis'
    },
    {
        'name': 'IMG_0434_source (Normal Cohort)',
        'path': base_dir / "IMG_0434_source.MOV",
        'category': 'BASELINE',
        'notes': 'Pre-rotated, normal patient'
    },
    {
        'name': 'IMG_0942_source (Normal Cohort)',
        'path': base_dir / "IMG_0942_source.MOV",
        'category': 'BASELINE',
        'notes': 'Pre-rotated, normal patient'
    }
]

output_dir = Path(__file__).parent / "test_output" / "source_videos_python"
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize detector
print("Initializing PyFaceAU detector...")
detector = PyFaceAU68LandmarkDetector(
    debug_mode=False,
    use_clnf_refinement=True,
    skip_redetection=False
)
print("✓ Detector ready")
print()

results = []

for video_info in test_videos:
    print("="*80)
    print(f"Testing: {video_info['name']}")
    print("="*80)

    video_path = video_info['path']

    if not video_path.exists():
        print(f"⚠️  Video not found: {video_path}")
        results.append({**video_info, 'status': 'NOT_FOUND'})
        print()
        continue

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}, {total_frames} frames @ {fps:.1f} FPS")

    # Test on multiple frames to get better average
    test_frames = [10, 30, 50, 70, 90] if total_frames > 100 else [10, 30, 50]
    frame_results = []

    for frame_idx in test_frames:
        if frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect landmarks
        start_time = time.time()
        landmarks, confidence = detector.get_face_mesh(frame)
        detection_time = (time.time() - start_time) * 1000

        # Get bbox
        bbox = None
        if hasattr(detector, 'cached_bbox') and detector.cached_bbox is not None:
            bbox = detector.cached_bbox

        if landmarks is not None and len(landmarks) == 68:
            # Quality metrics
            x_range = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
            y_range = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])

            frame_results.append({
                'frame': frame_idx,
                'success': True,
                'time': detection_time,
                'spread': (x_range, y_range),
                'bbox': bbox
            })

            # Save visualization for frame 50
            if frame_idx == 50:
                vis = frame.copy()

                # Draw bbox
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    cv2.putText(vis, "Bbox", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Draw landmarks
                for i, (x, y) in enumerate(landmarks):
                    if i < 17:  # Jaw
                        color = (0, 255, 0)
                    elif i < 27:  # Eyebrows
                        color = (255, 0, 0)
                    elif i < 36:  # Nose
                        color = (0, 255, 255)
                    elif i < 48:  # Eyes
                        color = (255, 0, 255)
                    else:  # Mouth
                        color = (0, 128, 255)

                    cv2.circle(vis, (int(x), int(y)), 4, color, -1)

                    # Label key points
                    if i in [0, 8, 16, 27, 30, 33, 36, 39, 42, 45, 48, 54, 57, 62, 66]:
                        cv2.putText(vis, str(i), (int(x)+5, int(y)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Midline
                glabella, chin = detector.get_facial_midline(landmarks)
                cv2.line(vis, (int(glabella[0]), int(glabella[1])),
                        (int(chin[0]), int(chin[1])), (0, 0, 255), 3)

                # Info overlay
                info_text = [
                    f"{video_info['name']}",
                    f"Python: PFLD + SVR CLNF",
                    f"Frame: {frame_idx}",
                    f"Time: {detection_time:.1f}ms",
                    f"Spread: {x_range:.0f}x{y_range:.0f}px"
                ]
                y_offset = 40
                for text in info_text:
                    cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (255, 255, 255), 3)
                    cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 0, 0), 1)
                    y_offset += 35

                output_path = output_dir / f"{video_path.stem}_python.jpg"
                cv2.imwrite(str(output_path), vis)

        else:
            frame_results.append({
                'frame': frame_idx,
                'success': False,
                'time': detection_time
            })

    cap.release()

    # Summarize results for this video
    successful = sum(1 for r in frame_results if r['success'])
    if successful > 0:
        avg_time = np.mean([r['time'] for r in frame_results if r['success']])
        spreads = [r['spread'] for r in frame_results if r['success']]
        avg_spread = np.mean(spreads, axis=0)

        print(f"✓ Detection successful on {successful}/{len(frame_results)} frames")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Average spread: {avg_spread[0]:.0f}×{avg_spread[1]:.0f} pixels")
        print(f"  Saved: {output_dir / (video_path.stem + '_python.jpg')}")

        results.append({
            **video_info,
            'status': 'SUCCESS',
            'frames_tested': len(frame_results),
            'frames_successful': successful,
            'avg_time_ms': avg_time,
            'avg_spread': avg_spread
        })
    else:
        print(f"❌ All frames failed detection")
        results.append({
            **video_info,
            'status': 'FAILED',
            'frames_tested': len(frame_results)
        })

    print()

# Summary
print("="*80)
print("SUMMARY - PYTHON IMPLEMENTATION ON ROTATED SOURCES")
print("="*80)
print()

for r in results:
    if r['status'] == 'SUCCESS':
        print(f"✓ {r['name']}")
        print(f"  Success: {r['frames_successful']}/{r['frames_tested']} frames")
        print(f"  Speed: {r['avg_time_ms']:.2f} ms/frame")
        print(f"  Spread: {r['avg_spread'][0]:.0f}×{r['avg_spread'][1]:.0f} px")
    else:
        print(f"✗ {r['name']}: {r['status']}")
    print()

# Create comparison grid
print("Creating comparison grid...")
grid_files = []
for r in results:
    video_name = r['path'].stem
    img_path = output_dir / f"{video_name}_python.jpg"
    if img_path.exists():
        grid_files.append(img_path)

if len(grid_files) == 4:
    imgs = [cv2.imread(str(f)) for f in grid_files]

    # Resize to same height
    target_h = min(img.shape[0] for img in imgs) // 2
    imgs_resized = []
    for img in imgs:
        aspect = img.shape[1] / img.shape[0]
        target_w = int(target_h * aspect)
        imgs_resized.append(cv2.resize(img, (target_w, target_h)))

    # Pad widths to match
    max_w = max(img.shape[1] for img in imgs_resized)
    imgs_padded = []
    for img in imgs_resized:
        if img.shape[1] < max_w:
            pad_w = max_w - img.shape[1]
            img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode='constant')
        imgs_padded.append(img)

    # Create grid
    top_row = np.hstack([imgs_padded[0], imgs_padded[1]])
    bottom_row = np.hstack([imgs_padded[2], imgs_padded[3]])
    grid = np.vstack([top_row, bottom_row])

    # Add title
    grid_with_title = np.zeros((grid.shape[0] + 80, grid.shape[1], 3), dtype=np.uint8)
    grid_with_title[80:, :] = grid

    title = "Python Implementation: PFLD + SVR CLNF (Rotated Source Files)"
    cv2.putText(grid_with_title, title, (30, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(grid_with_title, title, (30, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

    grid_path = output_dir / "python_comparison_grid.jpg"
    cv2.imwrite(str(grid_path), grid_with_title)
    print(f"✓ Comparison grid saved: {grid_path}")

print()
print("="*80)
print("NOTE: C++ OpenFace comparison not possible due to dependency issues")
print("  - Missing: libopenvino.2500.dylib, boost version mismatch")
print("  - This is exactly why pure Python implementation is preferred!")
print("="*80)
print()
print("TEST COMPLETE")
print("="*80)
