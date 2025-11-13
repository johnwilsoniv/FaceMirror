#!/usr/bin/env python3
"""
Visualize bbox correction comparison using pre-computed results.

Shows side-by-side comparison for all 9 test frames to understand what
characteristics lead to improvement vs regression in convergence.
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


def apply_openface_correction(bbox):
    """Apply OpenFace C++ MTCNN bbox correction formula."""
    x, y, w, h = bbox
    x_new = int(x + w * OPENFACE_DX_COEFF)
    y_new = int(y + h * OPENFACE_DY_COEFF)
    w_new = int(w * OPENFACE_W_SCALE)
    h_new = int(h * OPENFACE_H_SCALE)
    return (x_new, y_new, w_new, h_new)


# Pre-computed results from derive_pymtcnn_to_cpp_correction.py
RESULTS = {
    ('IMG_0433', 50): {
        'raw_bbox': (262, 581, 431, 555),
        'corrected_bbox': apply_openface_correction((262, 581, 431, 555)),
        'cpp_bbox': (287, 688, 424, 408),
        'raw_final_update': 4.751236,
        'corrected_final_update': 2.388319,
        'improvement_pct': 49.7
    },
    ('IMG_0433', 150): {
        'raw_bbox': (274, 587, 445, 564),
        'corrected_bbox': apply_openface_correction((274, 587, 445, 564)),
        'cpp_bbox': (307, 726, 409, 394),
        'raw_final_update': 3.054164,
        'corrected_final_update': 3.637293,
        'improvement_pct': -19.1
    },
    ('IMG_0433', 250): {
        'raw_bbox': (241, 645, 422, 536),
        'corrected_bbox': apply_openface_correction((241, 645, 422, 536)),
        'cpp_bbox': (301, 740, 402, 383),
        'raw_final_update': 2.728321,
        'corrected_final_update': 3.644355,
        'improvement_pct': -33.6
    },
    ('IMG_0434', 50): {
        'raw_bbox': (268, 687, 398, 518),
        'corrected_bbox': apply_openface_correction((268, 687, 398, 518)),
        'cpp_bbox': (295, 767, 408, 414),
        'raw_final_update': 2.607083,
        'corrected_final_update': 2.302473,
        'improvement_pct': 11.7
    },
    ('IMG_0434', 150): {
        'raw_bbox': (321, 664, 393, 485),
        'corrected_bbox': apply_openface_correction((321, 664, 393, 485)),
        'cpp_bbox': (288, 780, 399, 401),
        'raw_final_update': 2.256071,
        'corrected_final_update': 2.798522,
        'improvement_pct': -24.0
    },
    ('IMG_0434', 250): {
        'raw_bbox': (328, 644, 361, 450),
        'corrected_bbox': apply_openface_correction((328, 644, 361, 450)),
        'cpp_bbox': (293, 794, 385, 372),
        'raw_final_update': 4.105178,
        'corrected_final_update': 2.368879,
        'improvement_pct': 42.3
    },
    ('IMG_0435', 50): {
        'raw_bbox': (328, 539, 409, 544),
        'corrected_bbox': apply_openface_correction((328, 539, 409, 544)),
        'cpp_bbox': (329, 669, 388, 393),
        'raw_final_update': 2.388213,
        'corrected_final_update': 2.639067,
        'improvement_pct': -10.5
    },
    ('IMG_0435', 150): {
        'raw_bbox': (305, 566, 383, 472),
        'corrected_bbox': apply_openface_correction((305, 566, 383, 472)),
        'cpp_bbox': (322, 651, 396, 397),
        'raw_final_update': 2.264731,
        'corrected_final_update': 1.832060,
        'improvement_pct': 19.1
    },
    ('IMG_0435', 250): {
        'raw_bbox': (329, 507, 427, 532),
        'corrected_bbox': apply_openface_correction((329, 507, 427, 532)),
        'cpp_bbox': (323, 659, 385, 372),
        'raw_final_update': 1.809889,
        'corrected_final_update': 1.864548,
        'improvement_pct': -3.0
    },
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


def main():
    print("="*80)
    print("VISUALIZING BBOX CORRECTION: Original vs Corrected vs C++ bbox")
    print("="*80)
    print()
    print("Using pre-computed results from derive_pymtcnn_to_cpp_correction.py")
    print()

    # Create figure: 3x3 grid (one per frame)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Bbox Correction Comparison: Original (Red) vs Corrected (Green) vs C++ (Blue)',
                 fontsize=16, y=0.995)

    # Process each video and frame
    for video_idx, (video_path, frame_nums) in enumerate(VIDEO_CONFIGS):
        video_name = Path(video_path).stem

        for frame_idx, frame_num in enumerate(frame_nums):
            ax = axes[video_idx, frame_idx]

            print(f"Processing {video_name} frame {frame_num}...")

            try:
                # Get pre-computed results
                key = (video_name, frame_num)
                if key not in RESULTS:
                    ax.text(0.5, 0.5, f"{video_name}\nFrame {frame_num}\nNo data",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue

                result = RESULTS[key]
                raw_bbox = result['raw_bbox']
                corrected_bbox = result['corrected_bbox']
                cpp_bbox = result['cpp_bbox']
                improvement = result['improvement_pct']
                raw_ratio = result['raw_final_update'] / 0.005
                corr_ratio = result['corrected_final_update'] / 0.005

                # Extract frame
                frame = extract_frame(video_path, frame_num)
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

                # Draw C++ bbox (BLUE) - dashed
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
                title += f"Raw: {raw_ratio:.1f}x → Corr: {corr_ratio:.1f}x\n"
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
    improvements = [r['improvement_pct'] for r in RESULTS.values()]

    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()

    improved = [imp for imp in improvements if imp > 0]
    regressed = [imp for imp in improvements if imp < 0]

    print(f"Frames improved: {len(improved)}/{len(improvements)} ({len(improved)/len(improvements)*100:.0f}%)")
    print(f"Frames regressed: {len(regressed)}/{len(improvements)} ({len(regressed)/len(improvements)*100:.0f}%)")
    print()
    print(f"Mean improvement: {np.mean(improvements):+.1f}%")
    print(f"Median improvement: {np.median(improvements):+.1f}%")
    print(f"Std: {np.std(improvements):.1f}%")
    print()

    # Show best and worst cases
    print("BEST CASES (most improvement):")
    sorted_results = sorted(RESULTS.items(), key=lambda x: x[1]['improvement_pct'], reverse=True)
    for (video, frame), data in sorted_results[:3]:
        print(f"  {video} f{frame}: {data['improvement_pct']:+.1f}%")
    print()

    print("WORST CASES (most regression):")
    for (video, frame), data in sorted_results[-3:]:
        print(f"  {video} f{frame}: {data['improvement_pct']:+.1f}%")
    print()

    print("="*80)
    print(f"Open {output_file} to visually compare bboxes!")
    print("="*80)


if __name__ == "__main__":
    main()
