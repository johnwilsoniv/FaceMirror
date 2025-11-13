#!/usr/bin/env python3
"""
Visualize RAW MTCNN outputs: pyMTCNN vs C++ MTCNN (both uncorrected).

This reverses OpenFace's static correction from C++ bboxes to compare
the raw outputs of the two MTCNN implementations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

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


def invert_openface_correction(corrected_bbox):
    """Reverse OpenFace's static MTCNN correction to get raw C++ MTCNN output.

    OpenFace applies:
        x_new = x + w * -0.0075
        y_new = y + h * 0.2459
        w_new = w * 1.0323
        h_new = h * 0.7751

    We invert this to recover the original raw MTCNN output.
    """
    x_corr, y_corr, w_corr, h_corr = corrected_bbox

    # Invert scale first
    w_raw = w_corr / OPENFACE_W_SCALE
    h_raw = h_corr / OPENFACE_H_SCALE

    # Invert translation
    x_raw = x_corr - w_raw * OPENFACE_DX_COEFF
    y_raw = y_corr - h_raw * OPENFACE_DY_COEFF

    return (int(x_raw), int(y_raw), int(w_raw), int(h_raw))


# Pre-computed C++ corrected bboxes from derive_pymtcnn_to_cpp_correction.py
# We'll invert these to get raw C++ MTCNN output
CPP_CORRECTED_BBOXES = {
    ('IMG_0433', 50): (287, 688, 424, 408),
    ('IMG_0433', 150): (307, 726, 409, 394),
    ('IMG_0433', 250): (301, 740, 402, 383),
    ('IMG_0434', 50): (295, 767, 408, 414),
    ('IMG_0434', 150): (288, 780, 399, 401),
    ('IMG_0434', 250): (293, 794, 385, 372),
    ('IMG_0435', 50): (329, 669, 388, 393),
    ('IMG_0435', 150): (322, 651, 396, 397),
    ('IMG_0435', 250): (323, 659, 385, 372),
}

# Pre-computed raw pyMTCNN bboxes
RAW_PYMTCNN_BBOXES = {
    ('IMG_0433', 50): (262, 581, 431, 555),
    ('IMG_0433', 150): (274, 587, 445, 564),
    ('IMG_0433', 250): (241, 645, 422, 536),
    ('IMG_0434', 50): (268, 687, 398, 518),
    ('IMG_0434', 150): (321, 664, 393, 485),
    ('IMG_0434', 250): (328, 644, 361, 450),
    ('IMG_0435', 50): (328, 539, 409, 544),
    ('IMG_0435', 150): (305, 566, 383, 472),
    ('IMG_0435', 250): (329, 507, 427, 532),
}


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def compute_bbox_diff(bbox1, bbox2):
    """Compute differences between two bboxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    dx = x2 - x1
    dy = y2 - y1
    dw = w2 - w1
    dh = h2 - h1

    # Also compute IoU
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        iou = 0.0
    else:
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        iou = intersection / union

    return {
        'dx': dx,
        'dy': dy,
        'dw': dw,
        'dh': dh,
        'iou': iou
    }


def main():
    print("="*80)
    print("VISUALIZING RAW MTCNN OUTPUTS: pyMTCNN vs C++ MTCNN (both uncorrected)")
    print("="*80)
    print()
    print("OpenFace applies a STATIC correction to C++ MTCNN output:")
    print(f"  x_new = x + width * {OPENFACE_DX_COEFF}")
    print(f"  y_new = y + height * {OPENFACE_DY_COEFF}")
    print(f"  width_new = width * {OPENFACE_W_SCALE}")
    print(f"  height_new = height * {OPENFACE_H_SCALE}")
    print()
    print("We'll INVERT this correction from C++ bboxes to compare raw outputs.")
    print()

    # Create figure: 3x3 grid (one per frame)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Raw MTCNN Comparison: pyMTCNN (Red) vs C++ MTCNN (Blue, uncorrected)',
                 fontsize=16, y=0.995)

    all_diffs = []

    # Process each video and frame
    for video_idx, (video_path, frame_nums) in enumerate(VIDEO_CONFIGS):
        video_name = Path(video_path).stem

        for frame_idx, frame_num in enumerate(frame_nums):
            ax = axes[video_idx, frame_idx]

            print(f"Processing {video_name} frame {frame_num}...")

            try:
                # Get bboxes
                key = (video_name, frame_num)

                if key not in RAW_PYMTCNN_BBOXES or key not in CPP_CORRECTED_BBOXES:
                    ax.text(0.5, 0.5, f"{video_name}\nFrame {frame_num}\nNo data",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue

                raw_pymtcnn = RAW_PYMTCNN_BBOXES[key]
                cpp_corrected = CPP_CORRECTED_BBOXES[key]

                # INVERT the OpenFace correction to get raw C++ MTCNN
                raw_cpp_mtcnn = invert_openface_correction(cpp_corrected)

                # Compute differences
                diff = compute_bbox_diff(raw_pymtcnn, raw_cpp_mtcnn)
                all_diffs.append({
                    'video': video_name,
                    'frame': frame_num,
                    **diff
                })

                # Extract frame
                frame = extract_frame(video_path, frame_num)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw raw pyMTCNN bbox (RED)
                x, y, w_box, h_box = raw_pymtcnn
                rect_py = patches.Rectangle((x, y), w_box, h_box,
                                           linewidth=3, edgecolor='red',
                                           facecolor='none', label='pyMTCNN')
                ax.add_patch(rect_py)

                # Draw raw C++ MTCNN bbox (BLUE)
                x, y, w_box, h_box = raw_cpp_mtcnn
                rect_cpp = patches.Rectangle((x, y), w_box, h_box,
                                             linewidth=3, edgecolor='blue',
                                             facecolor='none', label='C++ MTCNN',
                                             linestyle='--')
                ax.add_patch(rect_cpp)

                # Show frame
                ax.imshow(frame_rgb)
                ax.axis('off')

                # Create title with bbox difference info
                title = f"{video_name} f{frame_num}\n"
                title += f"Δx={diff['dx']:+d} Δy={diff['dy']:+d} "
                title += f"Δw={diff['dw']:+d} Δh={diff['dh']:+d}\n"
                title += f"IoU={diff['iou']:.3f}"

                # Color based on IoU
                color = 'green' if diff['iou'] > 0.85 else ('orange' if diff['iou'] > 0.70 else 'red')
                ax.set_title(title, fontsize=10, color=color, weight='bold')

            except Exception as e:
                print(f"  Error: {e}")
                ax.text(0.5, 0.5, f"{video_name}\nFrame {frame_num}\nError: {str(e)[:50]}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

    # Add legend
    handles = [
        patches.Patch(color='red', label='Raw pyMTCNN'),
        patches.Patch(color='blue', label='Raw C++ MTCNN (inverted from corrected)')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 0.99), fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_file = 'raw_mtcnn_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print()
    print(f"Saved visualization to: {output_file}")
    print()

    # Print summary statistics
    if len(all_diffs) > 0:
        print("="*80)
        print("SUMMARY: RAW MTCNN IMPLEMENTATION DIFFERENCES")
        print("="*80)
        print()

        dx_vals = [d['dx'] for d in all_diffs]
        dy_vals = [d['dy'] for d in all_diffs]
        dw_vals = [d['dw'] for d in all_diffs]
        dh_vals = [d['dh'] for d in all_diffs]
        iou_vals = [d['iou'] for d in all_diffs]

        print("Differences between raw pyMTCNN and raw C++ MTCNN:")
        print()
        print(f"  X offset:    mean={np.mean(dx_vals):+6.1f}px  std={np.std(dx_vals):5.1f}px  range=[{min(dx_vals):+d}, {max(dx_vals):+d}]")
        print(f"  Y offset:    mean={np.mean(dy_vals):+6.1f}px  std={np.std(dy_vals):5.1f}px  range=[{min(dy_vals):+d}, {max(dy_vals):+d}]")
        print(f"  Width diff:  mean={np.mean(dw_vals):+6.1f}px  std={np.std(dw_vals):5.1f}px  range=[{min(dw_vals):+d}, {max(dw_vals):+d}]")
        print(f"  Height diff: mean={np.mean(dh_vals):+6.1f}px  std={np.std(dh_vals):5.1f}px  range=[{min(dh_vals):+d}, {max(dh_vals):+d}]")
        print()
        print(f"  IoU overlap: mean={np.mean(iou_vals):.3f}  std={np.std(iou_vals):.3f}  range=[{min(iou_vals):.3f}, {max(iou_vals):.3f}]")
        print()

        # Show per-frame breakdown
        print("PER-FRAME COMPARISON:")
        print()
        print(f"{'Frame':<20} {'Δx':<8} {'Δy':<8} {'Δw':<8} {'Δh':<8} {'IoU':<8}")
        print("-"*70)
        for d in all_diffs:
            frame_name = f"{d['video']} f{d['frame']}"
            print(f"{frame_name:<20} {d['dx']:>7} {d['dy']:>7} {d['dw']:>7} {d['dh']:>7} {d['iou']:>7.3f}")

        print()
        print("="*80)
        print("INTERPRETATION:")
        print("="*80)
        print()

        mean_iou = np.mean(iou_vals)
        if mean_iou > 0.90:
            print(f"✓ VERY SIMILAR: Mean IoU={mean_iou:.3f} - implementations are nearly identical")
        elif mean_iou > 0.75:
            print(f"~ SIMILAR: Mean IoU={mean_iou:.3f} - implementations produce similar bboxes")
            print("  Differences may explain why OpenFace's correction doesn't transfer perfectly")
        else:
            print(f"✗ DIFFERENT: Mean IoU={mean_iou:.3f} - implementations produce significantly different bboxes!")
            print("  pyMTCNN and C++ MTCNN are fundamentally different detectors")

        print()

        # Check if there's a systematic bias
        mean_dy = np.mean(dy_vals)
        if abs(mean_dy) > 20:
            print(f"⚠ SYSTEMATIC Y BIAS: pyMTCNN is {mean_dy:+.0f}px {'higher' if mean_dy < 0 else 'lower'} than C++ MTCNN")
            print("  This could explain eyebrow coverage issues!")

    print()
    print("="*80)
    print(f"Open {output_file} to see the raw MTCNN comparison!")
    print("="*80)


if __name__ == "__main__":
    main()
