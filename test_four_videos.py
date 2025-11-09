#!/usr/bin/env python3
"""
Test current implementation on 4 videos:
- 2 from Paralysis Cohort (challenging cases identified on Nov 2)
- 2 from Normal Cohort (baseline comparison)
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
print("FOUR-VIDEO COMPARISON TEST")
print("="*80)
print()

# Test videos
base_dir = Path(__file__).parent / "Patient Data"
test_videos = [
    {
        'name': 'IMG_8401 (Paralysis - Surgical Markings)',
        'path': base_dir / "Paralysis Cohort" / "IMG_8401.MOV",
        'category': 'CHALLENGING',
        'notes': 'Patient with surgical markings - identified as failing on Nov 2'
    },
    {
        'name': 'IMG_9330 (Paralysis - Severe)',
        'path': base_dir / "Paralysis Cohort" / "IMG_9330.MOV",
        'category': 'CHALLENGING',
        'notes': 'Severe paralysis - identified as failing on Nov 2'
    },
    {
        'name': 'IMG_0434 (Normal Cohort)',
        'path': base_dir / "Normal Cohort" / "IMG_0434.MOV",
        'category': 'BASELINE',
        'notes': 'Normal patient - should work well'
    },
    {
        'name': 'IMG_0942 (Normal Cohort)',
        'path': base_dir / "Normal Cohort" / "IMG_0942.MOV",
        'category': 'BASELINE',
        'notes': 'Normal patient - already tested (107.8 FPS)'
    }
]

output_dir = Path(__file__).parent / "test_output" / "four_video_comparison"
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize detector once
print("Initializing PyFaceAU detector...")
detector = PyFaceAU68LandmarkDetector(
    debug_mode=False,
    use_clnf_refinement=True,
    skip_redetection=False  # Use full detection for accuracy test
)
print("✓ Detector ready")
print()

results = []

for video_info in test_videos:
    print("="*80)
    print(f"Testing: {video_info['name']}")
    print(f"Category: {video_info['category']}")
    print(f"Notes: {video_info['notes']}")
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

    # Test on frame 50 (skip initial frames)
    test_frame = min(50, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Failed to read frame {test_frame}")
        results.append({**video_info, 'status': 'READ_FAILED'})
        print()
        continue

    # Detect landmarks and measure time
    start_time = time.time()
    landmarks, confidence = detector.get_face_mesh(frame)
    detection_time = (time.time() - start_time) * 1000  # ms

    # Get bbox from last detection
    bbox = None
    if hasattr(detector, 'cached_bbox') and detector.cached_bbox is not None:
        bbox = detector.cached_bbox

    if landmarks is not None and len(landmarks) == 68:
        # Calculate landmark quality metrics
        x_std = np.std(landmarks[:, 0])
        y_std = np.std(landmarks[:, 1])
        x_range = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        y_range = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])

        # Calculate spread relative to bbox
        if bbox is not None:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            x_coverage = x_range / bbox_width if bbox_width > 0 else 0
            y_coverage = y_range / bbox_height if bbox_height > 0 else 0
        else:
            x_coverage = 0
            y_coverage = 0

        # Check for clustering (poor quality indicator)
        expected_x_std = width * 0.05  # Expect landmarks spread across ~10% of frame width
        expected_y_std = height * 0.05
        quality_score = min(x_std / expected_x_std, y_std / expected_y_std)

        if quality_score > 0.8:
            quality = "EXCELLENT"
        elif quality_score > 0.5:
            quality = "GOOD"
        elif quality_score > 0.3:
            quality = "FAIR"
        else:
            quality = "POOR"

        print(f"✓ Detection successful:")
        print(f"  Time: {detection_time:.2f} ms")
        print(f"  Landmarks: {len(landmarks)}")
        print(f"  Quality: {quality} (score: {quality_score:.2f})")
        print(f"  Spread: {x_range:.0f}×{y_range:.0f} pixels")
        if bbox is not None:
            print(f"  Bbox coverage: {x_coverage*100:.1f}% × {y_coverage*100:.1f}%")

        # Create visualization
        vis = frame.copy()

        # Draw bbox if available
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan bbox
            cv2.putText(vis, "Bbox", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw landmarks with color coding
        for i, (x, y) in enumerate(landmarks):
            # Color by region
            if i < 17:  # Jaw
                color = (0, 255, 0)  # Green
            elif i < 27:  # Eyebrows
                color = (255, 0, 0)  # Blue
            elif i < 36:  # Nose
                color = (0, 255, 255)  # Yellow
            elif i < 48:  # Eyes
                color = (255, 0, 255)  # Magenta
            else:  # Mouth
                color = (0, 128, 255)  # Orange

            cv2.circle(vis, (int(x), int(y)), 3, color, -1)
            # Draw landmark number for key points
            if i % 5 == 0:
                cv2.putText(vis, str(i), (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Get midline
        glabella, chin = detector.get_facial_midline(landmarks)
        cv2.line(vis, (int(glabella[0]), int(glabella[1])),
                (int(chin[0]), int(chin[1])), (0, 0, 255), 2)

        # Add text info
        info_text = [
            f"{video_info['name']}",
            f"Quality: {quality}",
            f"Time: {detection_time:.1f}ms",
            f"Landmarks: {len(landmarks)}"
        ]
        y_offset = 30
        for text in info_text:
            cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 30

        # Save
        output_path = output_dir / f"{video_path.stem}_landmarks.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"  Saved: {output_path}")

        results.append({
            **video_info,
            'status': 'SUCCESS',
            'quality': quality,
            'quality_score': quality_score,
            'detection_time_ms': detection_time,
            'landmark_count': len(landmarks),
            'spread': (x_range, y_range),
            'coverage': (x_coverage, y_coverage) if bbox is not None else None
        })

    else:
        print(f"❌ Detection failed")
        print(f"  Time: {detection_time:.2f} ms")
        print(f"  Landmarks: {landmarks.shape if landmarks is not None else 'None'}")

        # Save frame with failure marker
        vis = frame.copy()
        cv2.putText(vis, "DETECTION FAILED", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis, f"{video_info['name']}", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_path = output_dir / f"{video_path.stem}_FAILED.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"  Saved failure frame: {output_path}")

        results.append({
            **video_info,
            'status': 'DETECTION_FAILED',
            'detection_time_ms': detection_time
        })

    print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()

challenging = [r for r in results if r.get('category') == 'CHALLENGING']
baseline = [r for r in results if r.get('category') == 'BASELINE']

print("CHALLENGING CASES (Paralysis Cohort):")
for r in challenging:
    status_icon = "✓" if r['status'] == 'SUCCESS' else "✗"
    quality = r.get('quality', 'N/A')
    print(f"  {status_icon} {r['name']}: {r['status']} - Quality: {quality}")

print()
print("BASELINE CASES (Normal Cohort):")
for r in baseline:
    status_icon = "✓" if r['status'] == 'SUCCESS' else "✗"
    quality = r.get('quality', 'N/A')
    print(f"  {status_icon} {r['name']}: {r['status']} - Quality: {quality}")

print()
print("OVERALL RESULTS:")
successful = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f"  Success rate: {successful}/{len(results)} ({100*successful/len(results):.0f}%)")

if successful > 0:
    avg_time = np.mean([r['detection_time_ms'] for r in results if r['status'] == 'SUCCESS'])
    print(f"  Average detection time: {avg_time:.2f} ms")

    quality_scores = [r['quality_score'] for r in results if r['status'] == 'SUCCESS']
    avg_quality = np.mean(quality_scores)
    print(f"  Average quality score: {avg_quality:.2f}")

print()
print(f"Visualizations saved to: {output_dir}")
print()

# Create 2x2 comparison grid
print("Creating comparison grid...")
grid_files = []
for r in results:
    video_name = r['path'].stem
    success_path = output_dir / f"{video_name}_landmarks.jpg"
    fail_path = output_dir / f"{video_name}_FAILED.jpg"

    if success_path.exists():
        grid_files.append(success_path)
    elif fail_path.exists():
        grid_files.append(fail_path)

if len(grid_files) == 4:
    # Load images
    imgs = [cv2.imread(str(f)) for f in grid_files]

    # Resize all to same size (use smallest dimensions)
    target_h = min(img.shape[0] for img in imgs) // 2
    target_w = min(img.shape[1] for img in imgs) // 2

    imgs_resized = [cv2.resize(img, (target_w, target_h)) for img in imgs]

    # Create 2x2 grid
    top_row = np.hstack([imgs_resized[0], imgs_resized[1]])
    bottom_row = np.hstack([imgs_resized[2], imgs_resized[3]])
    grid = np.vstack([top_row, bottom_row])

    # Add title
    grid_with_title = np.zeros((grid.shape[0] + 60, grid.shape[1], 3), dtype=np.uint8)
    grid_with_title[60:, :] = grid

    # Title text
    title = "Four-Video Comparison: Current Implementation (PFLD + SVR CLNF)"
    cv2.putText(grid_with_title, title, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    grid_path = output_dir / "comparison_grid_2x2.jpg"
    cv2.imwrite(str(grid_path), grid_with_title)
    print(f"✓ Comparison grid saved: {grid_path}")
else:
    print(f"⚠️  Only {len(grid_files)} images available, need 4 for grid")

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
