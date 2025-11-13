#!/usr/bin/env python3
"""
Comprehensive MTCNN bbox correction analysis across multiple frames and videos.

This script:
1. Analyzes 3 frames from 3 different videos (9 total samples)
2. Runs pyMTCNN detection on each frame
3. Tests convergence with both original and corrected bboxes
4. Applies OpenFace C++ correction formula empirically
5. Compares convergence performance before/after correction
6. Creates visualizations showing bboxes

The goal is to determine if OpenFace's hardcoded MTCNN correction formula
improves convergence when applied to pyMTCNN detections.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNNDetector
from pyclnf import CLNF

# Configuration
VIDEO_CONFIGS = [
    ('Patient Data/Normal Cohort/IMG_0433.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0434.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0435.MOV', [50, 150, 250]),
]

OUTPUT_DIR = Path('/tmp/mtcnn_analysis')
VIS_DIR = OUTPUT_DIR / 'visualizations'

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

# OpenFace C++ correction coefficients (from FaceDetectorMTCNN.cpp:919-924)
OPENFACE_DX_COEFF = -0.0075  # Shift left 0.75%
OPENFACE_DY_COEFF = 0.2459   # Shift DOWN 24.59%
OPENFACE_W_SCALE = 1.0323    # Expand width by 3.23%
OPENFACE_H_SCALE = 0.7751    # Shrink height to 77.51%


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def run_pymtcnn(frame):
    """Run OpenFace MTCNN detector on frame."""
    detector = OpenFaceMTCNNDetector(
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        scale_factor=0.709
    )

    bboxes, _ = detector.detect_faces(frame)

    if len(bboxes) == 0:
        return None

    # Return first detection
    det = bboxes[0]
    x1, y1, x2, y2 = det[:4]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def apply_openface_correction(bbox):
    """Apply OpenFace C++ MTCNN bbox correction formula."""
    x, y, w, h = bbox

    x_new = x + w * OPENFACE_DX_COEFF
    y_new = y + h * OPENFACE_DY_COEFF
    w_new = w * OPENFACE_W_SCALE
    h_new = h * OPENFACE_H_SCALE

    return (int(x_new), int(y_new), int(w_new), int(h_new))


def test_convergence(gray, bbox, max_iterations=20):
    """Test CLNF convergence with given bbox."""
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=max_iterations)
    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    return {
        'converged': info['converged'],
        'iterations': info['iterations'],
        'final_update': info['final_update'],
        'ratio_to_target': info['final_update'] / 0.005
    }


def draw_bbox_comparison(frame, original_bbox, corrected_bbox, output_path, frame_info, convergence_info):
    """Draw both bboxes on frame for visualization."""
    vis = frame.copy()

    # Draw original pyMTCNN bbox in red
    if original_bbox:
        x, y, w, h = original_bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
        label = f'Original: {convergence_info["original_ratio"]:.1f}x'
        cv2.putText(vis, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw corrected bbox in green
    if corrected_bbox:
        x, y, w, h = corrected_bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f'Corrected: {convergence_info["corrected_ratio"]:.1f}x'
        cv2.putText(vis, label, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add frame info
    info_text = f"{frame_info['video_name']} - Frame {frame_info['frame_num']}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add improvement info
    improvement_pct = convergence_info['improvement_pct']
    color = (0, 255, 0) if improvement_pct > 0 else (0, 0, 255)
    improvement_text = f"Improvement: {improvement_pct:+.1f}%"
    cv2.putText(vis, improvement_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(str(output_path), vis)
    return vis


def main():
    print("="*80)
    print("COMPREHENSIVE MTCNN BBOX CORRECTION ANALYSIS")
    print("="*80)
    print()
    print(f"Videos: {len(VIDEO_CONFIGS)}")
    print(f"Frames per video: {len(VIDEO_CONFIGS[0][1])}")
    print(f"Total samples: {len(VIDEO_CONFIGS) * len(VIDEO_CONFIGS[0][1])}")
    print()
    print("This script tests OpenFace's MTCNN bbox correction formula:")
    print(f"  x += width * {OPENFACE_DX_COEFF}")
    print(f"  y += height * {OPENFACE_DY_COEFF}")
    print(f"  width *= {OPENFACE_W_SCALE}")
    print(f"  height *= {OPENFACE_H_SCALE}")
    print()

    # Collect all results
    all_results = []

    # Process each video and frame
    for video_idx, (video_path, frame_nums) in enumerate(VIDEO_CONFIGS, 1):
        video_name = Path(video_path).stem
        print(f"\n{'='*80}")
        print(f"VIDEO {video_idx}/{len(VIDEO_CONFIGS)}: {video_name}")
        print(f"{'='*80}\n")

        for frame_idx, frame_num in enumerate(frame_nums, 1):
            print(f"  Frame {frame_idx}/{len(frame_nums)} (frame #{frame_num}):")

            # Extract frame
            try:
                frame = extract_frame(video_path, frame_num)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"    ✗ Failed to extract frame: {e}")
                continue

            # Run pyMTCNN detection
            print(f"    Running pyMTCNN detection...")
            original_bbox = run_pymtcnn(frame)

            if original_bbox is None:
                print(f"    ✗ pyMTCNN failed to detect face")
                continue

            print(f"    ✓ pyMTCNN bbox: {original_bbox}")
            x, y, w, h = original_bbox
            print(f"      (x={x}, y={y}, w={w}, h={h}, aspect={w/h:.3f})")

            # Apply OpenFace correction
            corrected_bbox = apply_openface_correction(original_bbox)
            print(f"    ✓ Corrected bbox: {corrected_bbox}")
            x, y, w, h = corrected_bbox
            print(f"      (x={x}, y={y}, w={w}, h={h}, aspect={w/h:.3f})")

            # Test convergence with original bbox
            print(f"    Testing convergence with ORIGINAL bbox...")
            original_conv = test_convergence(gray, original_bbox)
            print(f"      Converged: {original_conv['converged']}")
            print(f"      Final update: {original_conv['final_update']:.6f}")
            print(f"      Ratio to target: {original_conv['ratio_to_target']:.1f}x")

            # Test convergence with corrected bbox
            print(f"    Testing convergence with CORRECTED bbox...")
            corrected_conv = test_convergence(gray, corrected_bbox)
            print(f"      Converged: {corrected_conv['converged']}")
            print(f"      Final update: {corrected_conv['final_update']:.6f}")
            print(f"      Ratio to target: {corrected_conv['ratio_to_target']:.1f}x")

            # Compute improvement
            improvement_pct = ((original_conv['final_update'] - corrected_conv['final_update']) /
                              original_conv['final_update'] * 100)
            print(f"    Improvement: {improvement_pct:+.1f}%")

            # Create visualization
            vis_path = VIS_DIR / f'{video_name}_frame_{frame_num}_comparison.jpg'
            frame_info = {'video_name': video_name, 'frame_num': frame_num}
            convergence_info = {
                'original_ratio': original_conv['ratio_to_target'],
                'corrected_ratio': corrected_conv['ratio_to_target'],
                'improvement_pct': improvement_pct
            }
            draw_bbox_comparison(frame, original_bbox, corrected_bbox, vis_path, frame_info, convergence_info)
            print(f"    ✓ Visualization saved: {vis_path.name}")
            print()

            # Store results
            all_results.append({
                'video': video_name,
                'frame': frame_num,
                'original_bbox': original_bbox,
                'corrected_bbox': corrected_bbox,
                'original_final_update': original_conv['final_update'],
                'original_ratio': original_conv['ratio_to_target'],
                'corrected_final_update': corrected_conv['final_update'],
                'corrected_ratio': corrected_conv['ratio_to_target'],
                'improvement_pct': improvement_pct
            })

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    if len(all_results) == 0:
        print("No valid detections found!")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)

    print(f"Successfully analyzed {len(df)} frames")
    print()

    # Compute statistics
    print("="*80)
    print("CONVERGENCE PERFORMANCE")
    print("="*80)
    print()

    print(f"{'Frame':<20} {'Original':<15} {'Corrected':<15} {'Improvement':<15}")
    print("-"*70)
    for _, row in df.iterrows():
        frame_name = f"{row['video']} f{row['frame']}"
        orig_ratio = row['original_ratio']
        corr_ratio = row['corrected_ratio']
        improvement = row['improvement_pct']
        print(f"{frame_name:<20} {orig_ratio:>14.1f}x {corr_ratio:>14.1f}x {improvement:>13.1f}%")

    print()
    print("SUMMARY STATISTICS:")
    print(f"  Mean original ratio: {df['original_ratio'].mean():.1f}x target")
    print(f"  Mean corrected ratio: {df['corrected_ratio'].mean():.1f}x target")
    print(f"  Mean improvement: {df['improvement_pct'].mean():+.1f}%")
    print(f"  Median improvement: {df['improvement_pct'].median():+.1f}%")
    print(f"  Std improvement: {df['improvement_pct'].std():.1f}%")
    print()

    # Count improvements vs regressions
    num_improved = (df['improvement_pct'] > 0).sum()
    num_regressed = (df['improvement_pct'] < 0).sum()
    num_unchanged = (df['improvement_pct'] == 0).sum()

    print(f"  Improved: {num_improved}/{len(df)} ({num_improved/len(df)*100:.0f}%)")
    print(f"  Regressed: {num_regressed}/{len(df)} ({num_regressed/len(df)*100:.0f}%)")
    print(f"  Unchanged: {num_unchanged}/{len(df)} ({num_unchanged/len(df)*100:.0f}%)")
    print()

    # Find best and worst cases
    best_idx = df['improvement_pct'].idxmax()
    worst_idx = df['improvement_pct'].idxmin()

    print("BEST CASE:")
    best = df.loc[best_idx]
    print(f"  {best['video']} frame {best['frame']}")
    print(f"  Original: {best['original_ratio']:.1f}x → Corrected: {best['corrected_ratio']:.1f}x")
    print(f"  Improvement: {best['improvement_pct']:+.1f}%")
    print()

    print("WORST CASE:")
    worst = df.loc[worst_idx]
    print(f"  {worst['video']} frame {worst['frame']}")
    print(f"  Original: {worst['original_ratio']:.1f}x → Corrected: {worst['corrected_ratio']:.1f}x")
    print(f"  Degradation: {worst['improvement_pct']:+.1f}%")
    print()

    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    mean_improvement = df['improvement_pct'].mean()

    if mean_improvement > 10:
        print(f"✓ OpenFace MTCNN correction significantly improves convergence (+{mean_improvement:.1f}%)")
        print(f"  Recommend implementing this correction in production pipeline")
    elif mean_improvement > 5:
        print(f"~ OpenFace MTCNN correction provides moderate improvement (+{mean_improvement:.1f}%)")
        print(f"  Worth considering for production use")
    elif mean_improvement > 0:
        print(f"~ OpenFace MTCNN correction provides minor improvement (+{mean_improvement:.1f}%)")
        print(f"  May not be worth the added complexity")
    else:
        print(f"✗ OpenFace MTCNN correction degrades performance ({mean_improvement:.1f}%)")
        print(f"  Do NOT implement this correction")

    print()

    # Check consistency
    if df['improvement_pct'].std() > 20:
        print("⚠️  WARNING: High variance in results across frames!")
        print(f"   Standard deviation: {df['improvement_pct'].std():.1f}%")
        print(f"   The correction may help some frames but hurt others")
        print(f"   Consider investigating frame-specific factors")
    else:
        print(f"✓ Consistent results across frames (std={df['improvement_pct'].std():.1f}%)")

    print()
    print(f"Visualizations saved to: {VIS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
