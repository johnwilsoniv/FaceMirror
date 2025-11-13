#!/usr/bin/env python3
"""
Test OpenFace MTCNN bbox correction applied to pyMTCNN output.

This script:
1. Runs pyMTCNN detector on test frames
2. Applies OpenFace's hardcoded MTCNN correction formula
3. Tests convergence with both original and corrected bboxes
4. Determines if OpenFace's correction helps pyMTCNN output

The correction formula (from FaceDetectorMTCNN.cpp:919-924):
  x = x + width * (-0.0075)     [shift left 0.75%]
  y = y + height * 0.2459        [shift DOWN 24.59%]
  width = width * 1.0323         [expand 3.23%]
  height = height * 0.7751       [shrink to 77.51%]
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
from pyclnf import CLNF

# Configuration
VIDEO_CONFIGS = [
    ('Patient Data/Normal Cohort/IMG_0433.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0434.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0435.MOV', [50, 150, 250]),
]

# OpenFace C++ MTCNN correction coefficients (from FaceDetectorMTCNN.cpp:919-924)
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
    """Run pyMTCNN detector on frame WITHOUT OpenFace's built-in correction.

    We monkey-patch the correction method to return raw uncorrected bboxes,
    so we can test the correction separately.
    """
    detector = OpenFaceMTCNN(
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7]
    )

    # Monkey-patch to disable built-in correction (return raw bboxes)
    import numpy as np
    detector._apply_openface_correction = lambda bboxes: bboxes.copy()

    bboxes, _ = detector.detect(frame)

    if len(bboxes) == 0:
        return None

    # Return first detection (convert from x1,y1,x2,y2 to x,y,w,h)
    det = bboxes[0]
    x1, y1, x2, y2 = det[:4]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def apply_openface_mtcnn_correction(bbox):
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


def main():
    print("="*80)
    print("OPENFACE MTCNN CORRECTION APPLIED TO pyMTCNN OUTPUT")
    print("="*80)
    print()
    print(f"Videos: {len(VIDEO_CONFIGS)}")
    print(f"Frames per video: {len(VIDEO_CONFIGS[0][1])}")
    print(f"Total samples: {len(VIDEO_CONFIGS) * len(VIDEO_CONFIGS[0][1])}")
    print()
    print("OpenFace MTCNN correction formula:")
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
            corrected_bbox = apply_openface_mtcnn_correction(original_bbox)
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
        print(f"✓ OpenFace MTCNN correction significantly improves pyMTCNN convergence (+{mean_improvement:.1f}%)")
        print(f"  Recommend implementing this correction in production pipeline")
        print(f"  Despite pyMTCNN and C++ MTCNN being different, the correction transfers well")
    elif mean_improvement > 5:
        print(f"~ OpenFace MTCNN correction provides moderate improvement (+{mean_improvement:.1f}%)")
        print(f"  Worth considering for production use")
    elif mean_improvement > 0:
        print(f"~ OpenFace MTCNN correction provides minor improvement (+{mean_improvement:.1f}%)")
        print(f"  May not be worth the added complexity")
        print(f"  OpenFace-style PDM initialization may be sufficient")
    else:
        print(f"✗ OpenFace MTCNN correction degrades pyMTCNN performance ({mean_improvement:.1f}%)")
        print(f"  Do NOT implement this correction for pyMTCNN")
        print(f"  pyMTCNN and C++ MTCNN are too different")

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
    print("NEXT STEPS:")
    print()
    if mean_improvement > 5:
        print("1. Implement OpenFace MTCNN correction in detector pipeline")
        print("2. Combine with OpenFace-style PDM initialization (already implemented)")
        print("3. Test on full videos to verify production performance")
    else:
        print("1. OpenFace-style PDM initialization alone is sufficient")
        print("2. No need for bbox correction - implementations are too different")
        print("3. Focus on other optimization areas (iterations, convergence thresholds)")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
