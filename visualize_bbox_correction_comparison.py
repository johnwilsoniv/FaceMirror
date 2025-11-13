#!/usr/bin/env python3
"""
Visualize bbox correction comparison: Original vs Corrected vs C++ bbox.

Shows side-by-side comparison for all 9 test frames to understand what
characteristics lead to improvement vs regression in convergence.
"""

import cv2
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
from pyclnf import CLNF

# Configuration
VIDEO_CONFIGS = [
    ('Patient Data/Normal Cohort/IMG_0433.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0434.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0435.MOV', [50, 150, 250]),
]

# OpenFace MTCNN correction coefficients
OPENFACE_DX_COEFF = -0.0075
OPENFACE_DY_COEFF = 0.2459
OPENFACE_W_SCALE = 1.0323
OPENFACE_H_SCALE = 0.7751


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def get_raw_pymtcnn_bbox(detector, frame):
    """Get raw pyMTCNN bbox by inverting the built-in correction."""
    bboxes, _ = detector.detect(frame)

    if len(bboxes) == 0:
        return None

    # The detect() method returns corrected bboxes
    # Invert the correction to get raw bbox
    det = bboxes[0]
    x_corr, y_corr, x2, y2 = det[:4]
    w_corr = x2 - x_corr
    h_corr = y2 - y_corr

    # Invert OpenFace correction
    w_raw = w_corr / 1.0323
    h_raw = h_corr / 0.7751
    x_raw = x_corr - w_raw * (-0.0075)
    y_raw = y_corr - h_raw * 0.2459

    return (int(x_raw), int(y_raw), int(w_raw), int(h_raw))


def apply_openface_mtcnn_correction(bbox):
    """Apply OpenFace C++ MTCNN bbox correction formula."""
    x, y, w, h = bbox

    x_new = x + w * OPENFACE_DX_COEFF
    y_new = y + h * OPENFACE_DY_COEFF
    w_new = w * OPENFACE_W_SCALE
    h_new = h * OPENFACE_H_SCALE

    return (int(x_new), int(y_new), int(w_new), int(h_new))


def get_cpp_bbox_from_csv(csv_path):
    """Estimate OpenFace C++ effective bbox from landmarks in CSV."""
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Extract landmark positions
    x_cols = [c for c in df.columns if c.startswith('x_')]
    y_cols = [c for c in df.columns if c.startswith('y_')]

    if len(x_cols) == 0:
        return None

    x_coords = [df[c].values[0] for c in x_cols]
    y_coords = [df[c].values[0] for c in y_cols]

    # Estimate bbox from landmarks with 10% margin
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    w = max_x - min_x
    h = max_y - min_y
    margin_w = w * 0.1
    margin_h = h * 0.1

    x = int(min_x - margin_w)
    y = int(min_y - margin_h)
    w = int(w + 2 * margin_w)
    h = int(h + 2 * margin_h)

    return (x, y, w, h)


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


def draw_bbox_on_frame(frame, bbox, color, label, thickness=3):
    """Draw bbox on frame with label."""
    x, y, w, h = bbox
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)

    # Draw label with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Position label at top-left of bbox
    text_x = x
    text_y = max(y - 10, text_h + 10)

    # Draw background rectangle for text
    cv2.rectangle(frame_copy,
                  (text_x - 5, text_y - text_h - 5),
                  (text_x + text_w + 5, text_y + baseline + 5),
                  (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame_copy, label, (text_x, text_y), font, font_scale, color, font_thickness)

    return frame_copy


def main():
    print("="*80)
    print("VISUALIZING BBOX CORRECTION: Original vs Corrected vs C++ bbox")
    print("="*80)
    print()

    # Initialize detector once
    print("Initializing pyMTCNN detector...")
    detector = OpenFaceMTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.7])
    print()

    # Create figure: 3x3 grid (one per frame)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Bbox Correction Comparison: Original (Red) vs Corrected (Green) vs C++ (Blue)',
                 fontsize=16, y=0.995)

    all_results = []
    output_dir = Path('/tmp/mtcnn_correction_derivation')

    # Process each video and frame
    for video_idx, (video_path, frame_nums) in enumerate(VIDEO_CONFIGS):
        video_name = Path(video_path).stem

        for frame_idx, frame_num in enumerate(frame_nums):
            ax = axes[video_idx, frame_idx]

            print(f"Processing {video_name} frame {frame_num}...")

            try:
                # Extract frame
                frame = extract_frame(video_path, frame_num)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = frame.shape[:2]

                # Get raw pyMTCNN bbox
                raw_bbox = get_raw_pymtcnn_bbox(detector, frame)

                if raw_bbox is None:
                    ax.text(0.5, 0.5, f"{video_name}\nFrame {frame_num}\nNo detection",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue

                # Apply correction
                corrected_bbox = apply_openface_mtcnn_correction(raw_bbox)

                # Get C++ bbox from previous test results
                cpp_csv = output_dir / f'{video_name}_frame_{frame_num}' / 'cpp_output' / 'temp.csv'
                cpp_bbox = get_cpp_bbox_from_csv(cpp_csv)

                # Test convergence
                raw_conv = test_convergence(gray, raw_bbox)
                corr_conv = test_convergence(gray, corrected_bbox)

                improvement = ((raw_conv['final_update'] - corr_conv['final_update']) /
                              raw_conv['final_update'] * 100)

                # Store results
                all_results.append({
                    'video': video_name,
                    'frame': frame_num,
                    'improvement_pct': improvement,
                    'raw_final_update': raw_conv['final_update'],
                    'corr_final_update': corr_conv['final_update']
                })

                # Draw all bboxes on frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw raw bbox (RED)
                x, y, w_box, h_box = raw_bbox
                rect_raw = patches.Rectangle((x, y), w_box, h_box,
                                            linewidth=3, edgecolor='red',
                                            facecolor='none', label='Raw')
                ax.add_patch(rect_raw)

                # Draw corrected bbox (GREEN)
                x, y, w_box, h_box = corrected_bbox
                rect_corr = patches.Rectangle((x, y), w_box, h_box,
                                             linewidth=3, edgecolor='lime',
                                             facecolor='none', label='Corrected')
                ax.add_patch(rect_corr)

                # Draw C++ bbox (BLUE) if available
                if cpp_bbox is not None:
                    x, y, w_box, h_box = cpp_bbox
                    rect_cpp = patches.Rectangle((x, y), w_box, h_box,
                                                linewidth=3, edgecolor='blue',
                                                facecolor='none', label='C++',
                                                linestyle='--')
                    ax.add_patch(rect_cpp)

                # Show frame
                ax.imshow(frame_rgb)
                ax.axis('off')

                # Create title with convergence info
                color = 'green' if improvement > 5 else ('orange' if improvement > 0 else 'red')
                title = f"{video_name} f{frame_num}\n"
                title += f"Raw: {raw_conv['ratio_to_target']:.1f}x → Corr: {corr_conv['ratio_to_target']:.1f}x\n"
                title += f"Improvement: {improvement:+.1f}%"
                ax.set_title(title, fontsize=10, color=color, weight='bold')

            except Exception as e:
                print(f"  Error: {e}")
                ax.text(0.5, 0.5, f"{video_name}\nFrame {frame_num}\nError: {str(e)[:50]}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

    # Add legend
    handles = [
        patches.Patch(color='red', label='Raw pyMTCNN'),
        patches.Patch(color='lime', label='Corrected pyMTCNN'),
        patches.Patch(color='blue', label='C++ bbox (target)')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, 0.99), fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_file = 'bbox_correction_visual_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print()
    print(f"Saved visualization to: {output_file}")
    print()

    # Print summary statistics
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)

        print("="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print()

        improved = df[df['improvement_pct'] > 0]
        regressed = df[df['improvement_pct'] < 0]

        print(f"Frames improved: {len(improved)}/{len(df)} ({len(improved)/len(df)*100:.0f}%)")
        print(f"Frames regressed: {len(regressed)}/{len(df)} ({len(regressed)/len(df)*100:.0f}%)")
        print()
        print(f"Mean improvement: {df['improvement_pct'].mean():+.1f}%")
        print(f"Median improvement: {df['improvement_pct'].median():+.1f}%")
        print(f"Std: {df['improvement_pct'].std():.1f}%")
        print()

        # Show best and worst cases
        print("BEST CASES (most improvement):")
        best = df.nlargest(3, 'improvement_pct')
        for _, row in best.iterrows():
            print(f"  {row['video']} f{row['frame']}: {row['improvement_pct']:+.1f}%")
        print()

        print("WORST CASES (most regression):")
        worst = df.nsmallest(3, 'improvement_pct')
        for _, row in worst.iterrows():
            print(f"  {row['video']} f{row['frame']}: {row['improvement_pct']:+.1f}%")
        print()

    print("="*80)
    print(f"Open {output_file} to visually compare bboxes!")
    print("="*80)


if __name__ == "__main__":
    main()
